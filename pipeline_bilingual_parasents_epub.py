#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a bilingual EPUB:
- EPUB/JSON -> chapters {section: [paragraphs]}
- DTW chapter alignment (robust to splits/insertions)
- Paragraph alignment per aligned chapter (vecalign + ONNX encoder)
- Sentence splitting & alignment inside each aligned paragraph
- Render EPUB where each chapter shows numbered paragraph blocks,
  and inside each block sentences are stacked with '==' separators.

Deps:
  pip install ebooklib beautifulsoup4 lxml onnxruntime
Optional (better sentence splits):
  pip install blingfire

Local modules used (already in your project):
  - onnx_encoder.OnnxSentenceEncoder
  - vecalign.align_in_memory
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# suppress BS4 XML/HTML mode warning noise
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# --- your local modules ---
try:
    from onnx_encoder import OnnxSentenceEncoder
except Exception as e:
    raise SystemExit(f"Couldn't import OnnxSentenceEncoder. Make sure onnx_encoder.py is on PYTHONPATH. Error: {e}")

try:
    from vecalign import align_in_memory
except Exception as e:
    raise SystemExit(f"Couldn't import vecalign.align_in_memory. Make sure vecalign.py is on PYTHONPATH. Error: {e}")


# ======================== Text utilities ========================

NBSP = "\u00A0"
WORD_RX = re.compile(r"\w", re.UNICODE)

def norm(s: str) -> str:
    s = s.replace(NBSP, " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def keep_texty(s: str) -> bool:
    return bool(s and WORD_RX.search(s))


# ================= EPUB -> {section -> [paragraphs]} =================

BAN_WORDS = [
    "title", "toc", "content", "copyright", "jacket", "dedication",
    "acknowledg", "appendix", "ad-card", "advert", "back", "front",
    "about", "cover", "imprint"
]

def is_banned_key(name: str) -> bool:
    n = name.lower()
    return any(b in n for b in BAN_WORDS)

_num_re = re.compile(r"(\d{1,4})")
def natnum(name: str) -> int:
    m = _num_re.search(name)
    return int(m.group(1)) if m else 10**9

def html_to_paras(xhtml: bytes | str) -> Tuple[str, List[str]]:
    """
    Return (title, paragraphs). Accept XHTML or HTML.
    Prefer XML mode when it looks like XHTML; fallback to HTML mode otherwise.
    """
    text = xhtml.decode("utf-8", errors="ignore") if isinstance(xhtml, (bytes, bytearray)) else xhtml
    soup = BeautifulSoup(text, features="xml") if ("<?xml" in text or "xmlns" in text) else BeautifulSoup(text, "lxml")

    title = ""
    for tag in soup.select("h1,h2,h3,title"):
        title = norm(tag.get_text(" ", strip=True))
        if title:
            break

    nodes = soup.find_all(["p", "li", "blockquote"])
    paras = [norm(n.get_text(" ", strip=True)) for n in nodes if n.get_text(strip=True)]
    paras = [p for p in paras if keep_texty(p)]
    return title, paras

def epub_to_chapter_dict(epub_path: Path) -> Dict[str, List[str]]:
    """
    Key = item's filename (stable), Value = list of paragraphs extracted in order.
    Non-narrative/ads/etc. filtered by is_banned_key + natural sort.
    """
    book = epub.read_epub(str(epub_path))
    raw: Dict[str, List[str]] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = item.get_name()
        if is_banned_key(name):
            continue
        _, paras = html_to_paras(item.get_content())
        raw[name] = paras

    ordered = {k: raw[k] for k in sorted(raw.keys(), key=lambda k: (natnum(k), k))}
    return ordered


# ================= Chapter-level DTW alignment =================

def encode_means(encoder: OnnxSentenceEncoder, sections: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    """
    For each section (list of paragraphs), encode paragraphs and take mean embedding.
    Use normalize_embeddings=True to make dot-product == cosine sim.
    """
    names = list(sections.keys())
    means = []
    for k in names:
        paras = sections[k]
        if not paras:
            means.append(None)
            continue
        embs = encoder.encode(paras, batch_size=128, normalize_embeddings=True)  # (P, D)
        means.append(embs.mean(axis=0))

    # replace Nones with zeros of proper dim
    dim = means[0].shape[0] if means and means[0] is not None else 384
    fixed = [m if m is not None else np.zeros((dim,), dtype=np.float32) for m in means]
    return names, np.stack(fixed, axis=0).astype(np.float32)

def dtw_align_sections(
    en_names: List[str], en_vecs: np.ndarray,
    tr_names: List[str], tr_vecs: np.ndarray,
    gap: float = -0.25
) -> List[Dict]:
    """
    Needleman–Wunsch style DP on cosine similarity (vectors should be unit length; we rely on normalized embeddings).
    Allowed moves:
      diag: align EN[i] with TR[j]   (score += S[i,j])
      up  : EN[i] with GAP           (score += gap)
      left: TR[j] with GAP           (score += gap)
    Returns list of dicts: {en_section, tr_sections[], similarity (mean over aligned pairs)}.
    """
    E, N = en_vecs.shape[0], tr_vecs.shape[0]
    S = en_vecs @ tr_vecs.T  # cosine if normalized

    dp = np.full((E + 1, N + 1), -1e9, dtype=np.float32)
    bt = np.zeros((E + 1, N + 1), dtype=np.int8)  # 1=diag, 2=up, 3=left
    dp[0, 0] = 0.0

    for i in range(E + 1):
        for j in range(N + 1):
            if i < E and j < N:
                v = dp[i, j] + S[i, j]
                if v > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = v
                    bt[i + 1, j + 1] = 1
            if i < E:
                v = dp[i, j] + gap
                if v > dp[i + 1, j]:
                    dp[i + 1, j] = v
                    bt[i + 1, j] = 2
            if j < N:
                v = dp[i, j] + gap
                if v > dp[i, j + 1]:
                    dp[i, j + 1] = v
                    bt[i, j + 1] = 3

    # backtrack
    i, j = E, N
    pairs: List[Tuple[int, int]] = []
    while i > 0 or j > 0:
        mv = bt[i, j]
        if mv == 1:
            i -= 1; j -= 1; pairs.append((i, j))
        elif mv == 2:
            i -= 1; pairs.append((i, -1))
        else:
            j -= 1; pairs.append((-1, j))
    pairs.reverse()

    # collect TR indices per EN index
    agg: Dict[int, List[int]] = {}
    sims: Dict[int, List[float]] = {}
    for ii, jj in pairs:
        if ii >= 0 and jj >= 0:
            agg.setdefault(ii, []).append(jj)
            sims.setdefault(ii, []).append(float(S[ii, jj]))
        elif ii >= 0 and jj < 0:
            agg.setdefault(ii, [])  # gap on translation side

    out = []
    for idx, name in enumerate(en_names):
        js = sorted(set(agg.get(idx, [])))
        out.append({
            "en_section": name,
            "tr_sections": [tr_names[j] for j in js],
            "similarity": None if idx not in sims else round(float(np.mean(sims.get(idx, []))), 3)
        })
    return out


# ================= Paragraph alignment (vecalign) =================

def align_paragraphs_vecalign(
    src_paras: List[str], tgt_paras: List[str], encoder: OnnxSentenceEncoder, alignment_max_size: int = 10
) -> List[Tuple[str, str, float]]:
    """
    Align paragraphs lists, return list of (joined_src_para_block, joined_tgt_para_block, score).
    """
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
        left = " ".join(src_paras[i] for i in xs).strip()
        right = " ".join(tgt_paras[j] for j in ys).strip()
        if left or right:
            out.append((left, right, float(s)))
    return out


# ================= Sentence splitting & alignment =================

def split_sentences(text: str) -> List[str]:
    """
    Use blingfire if available (fast, robust), else regex heuristic:
    split on [.!?…] followed by space + capital/quote/open paren.
    """
    try:
        import blingfire as bf
        return [norm(s) for s in bf.text_to_sentences(text).split("\n") if keep_texty(s)]
    except Exception:
        rx = re.compile(r"(?<=[\.\?\!…])\s+(?=[“\"'\(]?[A-ZÀ-Ý])")
        parts = rx.split(text)
        return [norm(s) for s in parts if keep_texty(s)]

def align_sentence_pairs(src_para: str, tgt_para: str, encoder: OnnxSentenceEncoder, alignment_max_size: int = 5) -> List[Tuple[str, str, float]]:
    """
    Align sentence lists inside a paragraph pair; return [(src_sent_block, tgt_sent_block, score), ...]
    """
    src_s = split_sentences(src_para)
    tgt_s = split_sentences(tgt_para)
    if not src_s or not tgt_s:
        return []
    aligns, scores = align_in_memory(
        src_s, tgt_s, model=encoder, alignment_max_size=alignment_max_size, one_to_many=None
    )
    out = []
    for (xs, ys), s in zip(aligns, scores):
        left = " ".join(src_s[i] for i in xs).strip()
        right = " ".join(tgt_s[j] for j in ys).strip()
        if left or right:
            out.append((left, right, float(s)))
    return out


# ================= EPUB rendering (stacked sentences) =================

def chapter_html_stacked(
    title: str,
    para_pairs: List[Tuple[str, str, float]],
    sent_pairs_per_para: List[List[Tuple[str, str, float]]]
) -> str:
    """
    Render chapter where each paragraph block is numbered,
    and inside: stacked sentence pairs separated by '=='.
    """
    blocks = []
    for idx, (para_pair, sent_pairs) in enumerate(zip(para_pairs, sent_pairs_per_para), start=1):
        # Fallback: if no sentence alignment, show full paragraphs as single pair
        if not sent_pairs:
            sent_pairs = [(para_pair[0], para_pair[1], para_pair[2])]

        lines = []
        for si, (se, st, _sscore) in enumerate(sent_pairs):
            lines.append(f'<div class="line orig">{html.escape(se)}</div>')
            lines.append(f'<div class="line trgt">{html.escape(st)}</div>')
            if si != len(sent_pairs) - 1:
                lines.append('<div class="sep">==</div>')

        blocks.append(
            f"""
            <div class="para">
              <div class="phead">{idx}</div>
              <div class="content">
                {''.join(lines)}
              </div>
            </div>
            """
        )

    body = "\n".join(blocks) if blocks else "<p><em>No aligned content.</em></p>"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <title>{html.escape(title)}</title>
  <meta charset="utf-8"/>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,"Noto Sans",sans-serif; line-height:1.55; margin:1.25rem; }}
    h1 {{ font-size:1.25rem; margin:0 0 1rem 0; }}
    .para {{ border:1px solid #eee; border-radius:12px; padding:.75rem; margin:1rem 0; background:#fafafa; }}
    .phead {{ font-weight:600; margin-bottom:.5rem; }}
    .content {{ background:#fff; border:1px solid #eee; border-radius:10px; padding:.75rem; }}
    .line {{ margin:.25rem 0; }}
    .orig {{ }}
    .trgt {{ color:#222; }}
    .sep {{ color:#666; margin:.35rem 0; font-family:monospace; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>"""

def _extract_body(html_str: str) -> str:
    """Return inner <body>…</body> or the whole string if not found."""
    m = re.search(r"<body[^>]*>(.*?)</body>", html_str, flags=re.I | re.S)
    return m.group(1).strip() if m else html_str.strip()

def write_epub(book_title: str, author: str, chapters: List[Tuple[str, str]], out_path: Path):
    """
    chapters: list of (chapter_title, full_xhtml_string) – we extract body-only
    to keep ebooklib's nav builder happy; ensure non-empty content.
    """
    book = epub.EpubBook()
    book.set_identifier("bilingual-" + re.sub(r"[^a-z0-9]+", "-", book_title.lower()))
    book.set_title(book_title)
    if author:
        book.add_author(author)

    spine = ["nav"]
    toc = []
    items = []

    for idx, (title, html_str) in enumerate(chapters, 1):
        body_html = _extract_body(html_str)
        if not body_html.strip():
            body_html = f"<h1>{html.escape(title)}</h1><p></p>"

        file_name = f"chap_{idx:03d}.xhtml"
        c = epub.EpubHtml(title=title, file_name=file_name, lang="en")
        # IMPORTANT: body-only bytes
        c.content = body_html.encode("utf-8")

        book.add_item(c)
        items.append(c)
        toc.append(epub.Link(file_name, title, f"chap{idx:03d}"))
        spine.append(c)

    if not items:
        # placeholder to avoid empty book crash
        ph = epub.EpubHtml(title="Empty", file_name="chap_000.xhtml", lang="en")
        ph.content = b"<h1>Empty</h1><p>No aligned content.</p>"
        book.add_item(ph)
        items.append(ph)
        toc.append(epub.Link("chap_000.xhtml", "Empty", "chap000"))
        spine.append(ph)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = spine

    epub.write_epub(str(out_path), book)
    print(f"[epub] Wrote {out_path}")


# ============================ Main pipeline ============================

def main():
    ap = argparse.ArgumentParser(description="EPUB -> bilingual EPUB with paragraph + sentence alignment (stacked).")
    # Inputs
    ap.add_argument("--en_epub", type=Path, help="English EPUB path")
    ap.add_argument("--tr_epub", type=Path, help="Translation EPUB path")
    ap.add_argument("--en_json", type=Path, help="Optional pre-extracted JSON {section:[paragraphs]} for EN")
    ap.add_argument("--tr_json", type=Path, help="Optional pre-extracted JSON {section:[paragraphs]} for TR")
    ap.add_argument("--map_json", type=Path, help="Optional chapter mapping JSON from a previous run")
    # Output
    ap.add_argument("--title", required=True, help="EPUB title")
    ap.add_argument("--author", default="", help="EPUB author label")
    ap.add_argument("--output", default="bilingual.epub", help="Output EPUB path")
    # Model / alignment params
    ap.add_argument("--model_dir", required=True, help="Folder with ONNX model + tokenizer.json")
    ap.add_argument("--gap", type=float, default=-0.25, help="Gap penalty for chapter DTW (more negative => fewer gaps)")
    ap.add_argument("--max_para_span", type=int, default=10, help="Vecalign window for paragraph alignment")
    ap.add_argument("--sent_window", type=int, default=5, help="Vecalign window for sentence alignment")
    args = ap.parse_args()

    # 1) Load chapter dicts
    if args.en_json and args.tr_json:
        en_chap = json.loads(Path(args.en_json).read_text(encoding="utf-8"))
        tr_chap = json.loads(Path(args.tr_json).read_text(encoding="utf-8"))
    else:
        if not (args.en_epub and args.tr_epub):
            raise SystemExit("Provide --en_epub/--tr_epub or --en_json/--tr_json.")
        print("[extract] reading EPUBs …")
        en_chap = epub_to_chapter_dict(args.en_epub)
        tr_chap = epub_to_chapter_dict(args.tr_epub)

    # 2) Encoder
    enc = OnnxSentenceEncoder(
        args.model_dir,
        providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
        max_seq_length=256,
        prefer_int8=False
    )

    # 3) Chapter alignment (use precomputed mapping if provided)
    if args.map_json and Path(args.map_json).exists():
        mapping = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
        en_order = mapping.get("en_order", list(mapping["alignment"].keys()))
        align_list = [
            {"en_section": k, "tr_sections": mapping["alignment"][k].get("tr_sections") or mapping["alignment"][k].get("nl_sections", [])}
            for k in en_order
        ]
        print(f"[chapters] using existing mapping for {len(align_list)} chapters")
    else:
        print("[chapters] aligning (DTW) …")
        en_names, en_means = encode_means(enc, en_chap)
        tr_names, tr_means = encode_means(enc, tr_chap)
        align_list = dtw_align_sections(en_names, en_means, tr_names, tr_means, gap=args.gap)
        print(f"[chapters] produced {len(align_list)} aligned entries")

    # 4) For each aligned chapter: gather paragraphs, align paragraphs, then sentences
    chapters_html: List[Tuple[str, str]] = []
    for m in align_list:
        en_key = m["en_section"]
        tr_keys = m.get("tr_sections", [])
        en_paras = en_chap.get(en_key, [])
        tr_paras: List[str] = []
        for tk in tr_keys:
            tr_paras.extend(tr_chap.get(tk, []))

        para_pairs = align_paragraphs_vecalign(en_paras, tr_paras, enc, alignment_max_size=args.max_para_span)

        sent_pairs_per_para: List[List[Tuple[str, str, float]]] = []
        for (lp, rp, _score) in para_pairs:
            sent_pairs = align_sentence_pairs(lp, rp, enc, alignment_max_size=args.sent_window)
            sent_pairs_per_para.append(sent_pairs)

        title = en_key  # you can prettify if you want
        chapters_html.append((title, chapter_html_stacked(title, para_pairs, sent_pairs_per_para)))

    # 5) Write EPUB
    out_epub = Path(args.output)
    write_epub(args.title, args.author, chapters_html, out_epub)

if __name__ == "__main__":
    main()
