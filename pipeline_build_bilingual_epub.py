#!/usr/bin/env python3
"""
End-to-end:
- read two EPUBs (or JSONs) -> chapter dicts
- align chapters (DTW over mean paragraph embeddings)
- for each aligned chapter, align paragraphs with vecalign.align_in_memory
- write a dual-column EPUB + a JSON with all aligned pairs

Deps: pip install ebooklib beautifulsoup4 lxml onnxruntime
(If you prefer ST fallback: pip install sentence-transformers)
"""

from __future__ import annotations
import argparse, json, re, sys, html
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# --- use your modules ---
from onnx_encoder import OnnxSentenceEncoder              # your ONNX wrapper  :contentReference[oaicite:2]{index=2}
from vecalign import align_in_memory                      # your in-memory aligner  :contentReference[oaicite:3]{index=3}

# --- third-party for EPUB IO & HTML parsing ---
from ebooklib import epub
import ebooklib
from bs4 import BeautifulSoup
import numpy as np


# ========== text utilities ==========
NBSP = "\u00A0"
WORD_RX = re.compile(r"\w", re.UNICODE)

def norm(s: str) -> str:
    s = s.replace(NBSP, " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def keep_texty(s: str) -> bool:
    return bool(s and WORD_RX.search(s))


# ========== EPUB -> {section -> [paragraphs]} ==========
BAN_WORDS = [
    "title","toc","content","copyright","jacket","dedication",
    "acknowledg","appendix","ad-card","advert","back","front","about",
    "cover","einde","imprint"
]

def is_banned_key(name: str) -> bool:
    n = name.lower()
    return any(b in n for b in BAN_WORDS)

_num_re = re.compile(r"(\d{1,4})")
def natnum(name: str) -> int:
    m = _num_re.search(name)
    return int(m.group(1)) if m else 10**9

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def html_to_paras(xhtml: bytes | str) -> Tuple[str, List[str]]:
    """
    Return (title, paragraphs). We accept either HTML or XHTML.
    We try XML mode first, then fall back to lxml's HTML mode.
    """
    text = xhtml.decode("utf-8", errors="ignore") if isinstance(xhtml, (bytes, bytearray)) else xhtml

    # Try strict XML if the doc looks like XHTML
    if "<?xml" in text or "xmlns" in text:
        soup = BeautifulSoup(text, features="xml")
    else:
        soup = BeautifulSoup(text, "lxml")

    title = ""
    for tag in soup.select("h1,h2,h3,title"):
        title = norm(tag.get_text(" ", strip=True))
        if title:
            break

    # collect paragraphs & list-items
    nodes = soup.find_all(["p", "li"])
    paras = [norm(n.get_text(" ", strip=True)) for n in nodes if n.get_text(strip=True)]
    paras = [p for p in paras if keep_texty(p)]
    return title, paras



def epub_to_chapter_dict(epub_path: Path) -> Dict[str, List[str]]:
    """Key = item filename; Value = list of paragraphs."""
    book = epub.read_epub(str(epub_path))
    raw: Dict[str, List[str]] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):  # type: ignore
        name = item.get_name()
        if is_banned_key(name): 
            continue
        title, paras = html_to_paras(item.get_content())
        key = name
        raw[key] = paras
    # stable natural order
    ordered = {k: raw[k] for k in sorted(raw.keys(), key=lambda k: (natnum(k), k))}
    return ordered


# ========== Chapter-level DTW (many:many allowed, monotonic) ==========
def encode_means(encoder: OnnxSentenceEncoder, sections: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    names = list(sections.keys())
    means = []
    for k in names:
        paras = sections[k]
        if not paras:
            means.append(np.zeros((384,), dtype=np.float32))  # fallback dim; overwritten below
            continue
        embs = encoder.encode(paras, batch_size=128, normalize_embeddings=True)
        means.append(embs.mean(axis=0))
    dim = means[0].shape[0] if means else 384
    means = [m if m.size else np.zeros((dim,), dtype=np.float32) for m in means]
    return names, np.stack(means, axis=0).astype(np.float32)

def dtw_align_sections_np(
    en_names: List[str], en_vecs: np.ndarray,
    tr_names: List[str], tr_vecs: np.ndarray,
    gap: float = -0.25
) -> List[Dict]:
    """Needleman–Wunsch on cosine sims (vectors are unit length)."""
    E, N = en_vecs.shape[0], tr_vecs.shape[0]
    # Cosine similarity matrix via dot-product (already normalized)
    S = en_vecs @ tr_vecs.T

    dp = np.full((E+1, N+1), -1e9, dtype=np.float32)
    bt = np.zeros((E+1, N+1), dtype=np.int8)  # 1=diag 2=up 3=left
    dp[0,0] = 0.0
    for i in range(E+1):
        for j in range(N+1):
            if i < E and j < N:
                v = dp[i,j] + S[i,j]
                if v > dp[i+1,j+1]:
                    dp[i+1,j+1] = v; bt[i+1,j+1] = 1
            if i < E:
                v = dp[i,j] + gap
                if v > dp[i+1,j]:
                    dp[i+1,j] = v; bt[i+1,j] = 2
            if j < N:
                v = dp[i,j] + gap
                if v > dp[i,j+1]:
                    dp[i,j+1] = v; bt[i,j+1] = 3

    # backtrack
    i, j = E, N
    pairs: List[Tuple[int,int]] = []
    while i>0 or j>0:
        mv = bt[i,j]
        if mv == 1: i,j = i-1,j-1; pairs.append((i,j))
        elif mv == 2: i = i-1;     pairs.append((i,-1))
        else:          j = j-1;     pairs.append((-1,j))
    pairs.reverse()

    # group j's per i (collect consecutive tr chapters for each en chapter)
    agg: Dict[int, List[int]] = {}
    sims: Dict[int, List[float]] = {}
    for ii, jj in pairs:
        if ii >= 0 and jj >= 0:
            agg.setdefault(ii, []).append(jj)
            sims.setdefault(ii, []).append(float(S[ii, jj]))
        elif ii >= 0 and jj < 0:
            agg.setdefault(ii, [])  # gap on TR side

    out = []
    for idx, name in enumerate(en_names):
        js = sorted(set(agg.get(idx, [])))
        out.append({
            "en_section": name,
            "tr_sections": [tr_names[j] for j in js],
            "similarity": None if idx not in sims else round(float(np.mean(sims.get(idx, []))), 3)
        })
    return out


# ========== Chapter paragraph alignment (vecalign) ==========
def align_paragraphs_vecalign(
    src_paras: List[str], tgt_paras: List[str], encoder: OnnxSentenceEncoder,
    alignment_max_size: int = 10
) -> List[Tuple[str,str,float]]:
    if not src_paras or not tgt_paras:
        return []
    aligns, scores = align_in_memory(
        [norm(x) for x in src_paras],
        [norm(y) for y in tgt_paras],
        model=encoder,
        alignment_max_size=alignment_max_size,
        one_to_many=None
    )
    out = []
    for (xs, ys), s in zip(aligns, scores):
        left  = " ".join(src_paras[i] for i in xs).strip()
        right = " ".join(tgt_paras[j] for j in ys).strip()
        if left or right:
            out.append((left, right, float(s)))
    return out


# ========== EPUB builder ==========
def chapter_html(title: str, pairs: List[Tuple[str,str,float]]) -> str:
    rows = []
    for i, (a,b,score) in enumerate(pairs, 1):
        rows.append(
            f'<div class="pair">'
            f'  <div class="cell left"><div class="n">{i}</div><p>{html.escape(a)}</p></div>'
            f'  <div class="cell right"><div class="n">{i}</div><p>{html.escape(b)}</p></div>'
            f'</div>'
        )
    body = "\n".join(rows) if rows else "<p><em>No alignment for this chapter.</em></p>"
    # full XHTML (we'll strip body below before giving to ebooklib)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <title>{html.escape(title)}</title>
  <meta charset="utf-8"/>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,Noto Sans,sans-serif; line-height:1.5; margin:1.25rem; }}
    h1 {{ font-size:1.3rem; margin:0 0 1rem 0; }}
    .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:.75rem 0; align-items:start; }}
    .cell {{ background:#fafafa; border:1px solid #eee; border-radius:12px; padding:.75rem; }}
    .n {{ float:right; font-size:.75rem; color:#999; }}
    p {{ margin:0; text-align:justify; }}
    @media (max-width:680px) {{ .pair {{ grid-template-columns:1fr; }} .n {{ float:none; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>"""


def _extract_body(html_str: str) -> str:
    """Return the inner HTML of <body>…</body>; fallback to the whole string."""
    m = re.search(r"<body[^>]*>(.*?)</body>", html_str, flags=re.I | re.S)
    return m.group(1).strip() if m else html_str.strip()

def write_epub(book_title: str, author: str, chapters: List[Tuple[str,str]], out_path: Path):
    from ebooklib import epub
    book = epub.EpubBook()
    book.set_identifier("bilingual-" + re.sub(r"[^a-z0-9]+", "-", book_title.lower()))
    book.set_title(book_title)
    if author:
        book.add_author(author)

    spine = ["nav"]
    toc = []
    items = []

    # Build only chapters that actually have body content
    for idx, (title, html_str) in enumerate(chapters, 1):
        body_html = _extract_body(html_str)
        if not body_html.strip():
            # guarantee non-empty body to keep lxml happy
            body_html = f"<h1>{html.escape(title)}</h1><p></p>"

        file_name = f"chap_{idx:03d}.xhtml"
        c = epub.EpubHtml(title=title, file_name=file_name, lang="en")

        # IMPORTANT: set *body-only* content as bytes
        c.content = body_html.encode("utf-8")

        book.add_item(c)
        items.append(c)
        toc.append(epub.Link(file_name, title, f"chap{idx:03d}"))
        spine.append(c)

    # If nothing to add, create a tiny placeholder so nav writer won't crash
    if not items:
        placeholder = epub.EpubHtml(title="Empty", file_name="chap_000.xhtml", lang="en")
        placeholder.content = b"<h1>Empty</h1><p>No aligned content.</p>"
        book.add_item(placeholder)
        items.append(placeholder)
        toc.append(epub.Link("chap_000.xhtml", "Empty", "chap000"))
        spine.append(placeholder)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Set the spine! (ebooklib relies on this)
    book.spine = spine

    epub.write_epub(str(out_path), book)
    print(f"[epub] Wrote {out_path}")


# ========== Main pipeline ==========
def main():
    ap = argparse.ArgumentParser(description="EPUB → aligned bilingual EPUB (chapters & paragraphs).")
    ap.add_argument("--en_epub", type=Path, help="English EPUB path")
    ap.add_argument("--tr_epub", type=Path, help="Translation EPUB path")
    ap.add_argument("--en_json", type=Path, help="Optional: pre-extracted JSON {section:[paras]}")
    ap.add_argument("--tr_json", type=Path, help="Optional: pre-extracted JSON {section:[paras]}")
    ap.add_argument("--map_json", type=Path, help="Optional: precomputed chapter alignment JSON")
    ap.add_argument("--title", required=True, help="EPUB title")
    ap.add_argument("--author", default="", help="EPUB author string")
    ap.add_argument("--output", default="bilingual.epub", help="Output EPUB path")
    ap.add_argument("--model_dir", required=True, help="Folder containing your ONNX model + tokenizer.json")
    ap.add_argument("--gap", type=float, default=-0.25, help="Gap penalty for chapter DTW")
    ap.add_argument("--max_para_span", type=int, default=10, help="Max N+M span for paragraph alignments (vecalign)")
    args = ap.parse_args()

    # 1) Load chapters (from EPUB or JSON)
    if args.en_json and args.tr_json:
        en_chap = json.loads(Path(args.en_json).read_text(encoding="utf-8"))
        tr_chap = json.loads(Path(args.tr_json).read_text(encoding="utf-8"))
    else:
        if not (args.en_epub and args.tr_epub):
            raise SystemExit("Provide --en_epub/--tr_epub or --en_json/--tr_json.")
        print("[extract] reading EPUBs → chapters…")
        en_chap = epub_to_chapter_dict(args.en_epub)
        tr_chap = epub_to_chapter_dict(args.tr_epub)

    # 2) Chapter alignment (DTW) — or load mapping if you already have it
    enc = OnnxSentenceEncoder(
        args.model_dir,
        providers=("CUDAExecutionProvider","CPUExecutionProvider"),
        max_seq_length=256,
        prefer_int8=False
    )
    if args.map_json and Path(args.map_json).exists():
        print("[chapters] using existing mapping JSON")
        mapping = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
        en_order = mapping.get("en_order", list(mapping["alignment"].keys()))
        align_list = [
            {"en_section": k, "tr_sections": mapping["alignment"][k].get("tr_sections") or mapping["alignment"][k].get("nl_sections", [])}
            for k in en_order
        ]
    else:
        print("[chapters] computing alignment (DTW)…")
        en_names, en_means = encode_means(enc, en_chap)
        tr_names, tr_means = encode_means(enc, tr_chap)
        matches = dtw_align_sections_np(en_names, en_means, tr_names, tr_means, gap=args.gap)
        align_list = matches

    # 3) Paragraph alignment per chapter (vecalign)
    chapters_html: List[Tuple[str,str]] = []
    bilingual_dump: Dict[str, Dict] = {}

    for m in align_list:
        en_key = m["en_section"]
        tr_keys = m.get("tr_sections", [])
        en_paras = en_chap.get(en_key, [])
        tr_paras = []
        for tk in tr_keys: tr_paras.extend(tr_chap.get(tk, []))

        pairs = align_paragraphs_vecalign(en_paras, tr_paras, enc, alignment_max_size=args.max_para_span)
        title = en_key  # you can prettify this if you want
        chapters_html.append((title, chapter_html(title, pairs)))
        bilingual_dump[en_key] = {
            "original": en_paras,
            "translation": tr_paras,
            "aligned": [{"original": a, "translation": b, "score": s} for a,b,s in pairs],
            "tr_sections": tr_keys,
        }

    # 4) Write EPUB + JSON
    print(f"[chapters] writing {len(chapters_html)} chapters")
    for i,(t,h) in enumerate(chapters_html[:3],1):
        print(f"  {i:02d}. {t}  (len={len(h)})")
    out_epub = Path(args.output)
    write_epub(args.title, args.author, chapters_html, out_epub)
    aux_json = out_epub.with_suffix(".bilingual.json")
    aux_json.write_text(json.dumps(bilingual_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] wrote {aux_json}")

if __name__ == "__main__":
    main()
