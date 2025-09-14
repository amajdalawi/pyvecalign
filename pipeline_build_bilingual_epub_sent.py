#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, sys, html, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from ebooklib import epub
import numpy as np

# your local modules
from onnx_encoder import OnnxSentenceEncoder
from vecalign import align_in_memory

# ---------------- text utils ----------------
NBSP = "\u00A0"
WORD_RX = re.compile(r"\w", re.UNICODE)

def norm(s: str) -> str:
    s = s.replace(NBSP, " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s

def keep_texty(s: str) -> bool:
    return bool(s and WORD_RX.search(s))

# ----------- EPUB -> {section: [paragraphs]} -----------
BAN_WORDS = [
    "title","toc","content","copyright","jacket","dedication",
    "acknowledg","appendix","ad-card","advert","back","front","about",
    "cover","einde","imprint"
]
def is_banned_key(name: str) -> bool:
    n = name.lower(); return any(b in n for b in BAN_WORDS)

_num_re = re.compile(r"(\d{1,4})")
def natnum(name: str) -> int:
    m = _num_re.search(name); return int(m.group(1)) if m else 10**9

def html_to_paras(xhtml: bytes | str) -> Tuple[str, List[str]]:
    text = xhtml.decode("utf-8", errors="ignore") if isinstance(xhtml,(bytes,bytearray)) else xhtml
    soup = BeautifulSoup(text, features="xml") if ("<?xml" in text or "xmlns" in text) else BeautifulSoup(text, "lxml")

    title = ""
    for tag in soup.select("h1,h2,h3,title"):
        title = norm(tag.get_text(" ", strip=True)); 
        if title: break

    nodes = soup.find_all(["p","li","blockquote"])
    paras = [norm(n.get_text(" ", strip=True)) for n in nodes if n.get_text(strip=True)]
    paras = [p for p in paras if keep_texty(p)]
    return title, paras

def epub_to_chapter_dict(epub_path: Path) -> Dict[str, List[str]]:
    book = epub.read_epub(str(epub_path))
    raw: Dict[str,List[str]] = {}
    for item in book.get_items_of_type(epub.ITEM_DOCUMENT):
        name = item.get_name()
        if is_banned_key(name): 
            continue
        _, paras = html_to_paras(item.get_content())
        raw[name] = paras
    ordered = {k: raw[k] for k in sorted(raw.keys(), key=lambda k: (natnum(k), k))}
    return ordered

# --------------- chapter-level DTW ---------------
def encode_means(encoder: OnnxSentenceEncoder, sections: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    names = list(sections.keys()); means=[]
    for k in names:
        paras = sections[k]
        if not paras:
            means.append(np.zeros((384,), dtype=np.float32)); continue
        embs = encoder.encode(paras, batch_size=128, normalize_embeddings=True)
        means.append(embs.mean(axis=0))
    dim = means[0].shape[0] if means else 384
    means = [m if m.size else np.zeros((dim,), dtype=np.float32) for m in means]
    return names, np.stack(means, axis=0).astype(np.float32)

def dtw_align_sections_np(en_names: List[str], en_vecs: np.ndarray,
                          tr_names: List[str], tr_vecs: np.ndarray,
                          gap: float=-0.25) -> List[Dict]:
    E,N = en_vecs.shape[0], tr_vecs.shape[0]
    S = en_vecs @ tr_vecs.T  # cosine if normalized

    dp = np.full((E+1,N+1), -1e9, dtype=np.float32)
    bt = np.zeros((E+1,N+1), dtype=np.int8)  # 1=diag,2=up,3=left
    dp[0,0]=0.0
    for i in range(E+1):
        for j in range(N+1):
            if i<E and j<N:
                v=dp[i,j]+S[i,j]
                if v>dp[i+1,j+1]: dp[i+1,j+1]=v; bt[i+1,j+1]=1
            if i<E:
                v=dp[i,j]+gap
                if v>dp[i+1,j]: dp[i+1,j]=v; bt[i+1,j]=2
            if j<N:
                v=dp[i,j]+gap
                if v>dp[i,j+1]: dp[i,j+1]=v; bt[i,j+1]=3

    i,j=E,N; pairs=[]
    while i>0 or j>0:
        mv=bt[i,j]
        if mv==1: i-=1; j-=1; pairs.append((i,j))
        elif mv==2: i-=1; pairs.append((i,-1))
        else: j-=1; pairs.append((-1,j))
    pairs.reverse()

    agg: Dict[int,List[int]] = {}; sims: Dict[int,List[float]] = {}
    for ii,jj in pairs:
        if ii>=0 and jj>=0:
            agg.setdefault(ii,[]).append(jj)
            sims.setdefault(ii,[]).append(float(S[ii,jj]))
        elif ii>=0 and jj<0:
            agg.setdefault(ii,[])
    out=[]
    for idx,name in enumerate(en_names):
        js = sorted(set(agg.get(idx,[])))
        out.append({"en_section":name, "tr_sections":[tr_names[j] for j in js],
                    "similarity": None if idx not in sims else round(float(np.mean(sims.get(idx,[]))),3)})
    return out

# --------------- paragraph alignment (vecalign) ---------------
def align_paragraphs_vecalign(src_paras: List[str], tgt_paras: List[str],
                              encoder: OnnxSentenceEncoder, alignment_max_size:int=10
) -> List[Tuple[str,str,float]]:
    if not src_paras or not tgt_paras: return []
    aligns, scores = align_in_memory(
        [norm(x) for x in src_paras], [norm(y) for y in tgt_paras],
        model=encoder, alignment_max_size=alignment_max_size, one_to_many=None
    )
    out=[]
    for (xs,ys),s in zip(aligns,scores):
        left=" ".join(src_paras[i] for i in xs).strip()
        right=" ".join(tgt_paras[j] for j in ys).strip()
        if left or right: out.append((left,right,float(s)))
    return out

# --------------- sentence splitting & alignment ---------------
def split_sentences(text: str) -> List[str]:
    """Try blingfire if available; fallback to a regex split."""
    try:
        import blingfire as bf
        # bf.text_to_sentences puts one sentence per line
        return [norm(s) for s in bf.text_to_sentences(text).split("\n") if keep_texty(s)]
    except Exception:
        # simple heuristic: split on .,?!… followed by space+capital/quote
        rx = re.compile(r"(?<=[\.\?\!…])\s+(?=[“\"'\(]?[A-ZÀ-Ý])")
        parts = rx.split(text)
        return [norm(s) for s in parts if keep_texty(s)]

def align_sentence_pairs(src_para: str, tgt_para: str, encoder: OnnxSentenceEncoder,
                         alignment_max_size:int=5) -> List[Tuple[str,str,float]]:
    src_s = split_sentences(src_para)
    tgt_s = split_sentences(tgt_para)
    if not src_s or not tgt_s: return []
    aligns, scores = align_in_memory(
        src_s, tgt_s, model=encoder, alignment_max_size=alignment_max_size, one_to_many=None
    )
    out=[]
    for (xs,ys),s in zip(aligns,scores):
        left=" ".join(src_s[i] for i in xs).strip()
        right=" ".join(tgt_s[j] for j in ys).strip()
        if left or right: out.append((left,right,float(s)))
    return out

# ---------------- EPUB builder ----------------
def _extract_body(html_str: str) -> str:
    m = re.search(r"<body[^>]*>(.*?)</body>", html_str, flags=re.I|re.S)
    return m.group(1).strip() if m else html_str.strip()

def chapter_html(title: str,
                 pairs: List[Tuple[str,str,float]],
                 sent_pairs: List[List[Tuple[str,str,float]]] | None = None) -> str:
    rows=[]
    for i,(a,b,score) in enumerate(pairs,1):
        # sentence rows (optional)
        sent_html=""
        if sent_pairs is not None and i-1 < len(sent_pairs) and sent_pairs[i-1]:
            srows=[]
            for k,(sa,sb,ss) in enumerate(sent_pairs[i-1],1):
                srows.append(
                    f'<div class="srow"><div class="sleft"><span class="sn">{k}</span> {html.escape(sa)}</div>'
                    f'<div class="sright"><span class="sn">{k}</span> {html.escape(sb)}</div></div>'
                )
            sent_html = '<div class="sbox">' + "\n".join(srows) + "</div>"

        rows.append(
            f'<div class="pair">'
            f'  <div class="cell left"><div class="n">{i}</div><p>{html.escape(a)}</p></div>'
            f'  <div class="cell right"><div class="n">{i}</div><p>{html.escape(b)}</p></div>'
            f'  {sent_html}'
            f'</div>'
        )
    body = "\n".join(rows) if rows else "<p><em>No alignment for this chapter.</em></p>"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <title>{html.escape(title)}</title>
  <meta charset="utf-8"/>
  <style>
    body {{ font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,Noto Sans,sans-serif; line-height:1.5; margin:1.25rem; }}
    h1 {{ font-size:1.3rem; margin:0 0 1rem 0; }}
    .pair {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin:1rem 0; align-items:start; }}
    .cell {{ background:#fafafa; border:1px solid #eee; border-radius:12px; padding:.75rem; }}
    .n {{ float:right; font-size:.75rem; color:#999; }}
    p {{ margin:0; text-align:justify; }}
    .sbox {{ grid-column:1 / span 2; border-left:3px solid #ddd; padding:.5rem .75rem; margin-top:.25rem; background:#fff; }}
    .srow {{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; margin:.25rem 0; }}
    .sleft,.sright {{ font-size:.94em; }}
    .sn {{ color:#888; font-size:.8em; margin-right:.35rem; }}
    @media (max-width:680px) {{ .pair {{ grid-template-columns:1fr; }} .srow {{ grid-template-columns:1fr; }} .n {{ float:none; }} }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>"""

def write_epub(book_title: str, author: str, chapters: List[Tuple[str,str]], out_path: Path):
    book = epub.EpubBook()
    book.set_identifier("bilingual-" + re.sub(r"[^a-z0-9]+","-", book_title.lower()))
    book.set_title(book_title)
    if author: book.add_author(author)

    spine=["nav"]; toc=[]; items=[]
    for idx,(title,html_str) in enumerate(chapters,1):
        body_html = _extract_body(html_str)
        if not body_html.strip():
            body_html = f"<h1>{html.escape(title)}</h1><p></p>"
        fn = f"chap_{idx:03d}.xhtml"
        c = epub.EpubHtml(title=title, file_name=fn, lang="en")
        c.content = body_html.encode("utf-8")
        book.add_item(c); items.append(c)
        toc.append(epub.Link(fn, title, f"chap{idx:03d}"))
        spine.append(c)

    if not items:
        ph = epub.EpubHtml(title="Empty", file_name="chap_000.xhtml", lang="en")
        ph.content = b"<h1>Empty</h1><p>No aligned content.</p>"
        book.add_item(ph); items.append(ph); toc.append(epub.Link("chap_000.xhtml","Empty","chap000")); spine.append(ph)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx()); book.add_item(epub.EpubNav())
    book.spine = spine
    epub.write_epub(str(out_path), book)
    print(f"[epub] Wrote {out_path}")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="EPUB -> bilingual EPUB with paragraph and sentence alignment.")
    ap.add_argument("--en_epub", type=Path, help="English EPUB path")
    ap.add_argument("--tr_epub", type=Path, help="Translation EPUB path")
    ap.add_argument("--en_json", type=Path, help="Optional {section:[paras]} for EN")
    ap.add_argument("--tr_json", type=Path, help="Optional {section:[paras]} for TR")
    ap.add_argument("--map_json", type=Path, help="Optional chapter mapping JSON")
    ap.add_argument("--title", required=True); ap.add_argument("--author", default="")
    ap.add_argument("--output", default="bilingual.epub")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--gap", type=float, default=-0.25)
    ap.add_argument("--max_para_span", type=int, default=10)
    ap.add_argument("--sentence_level", action="store_true", help="Also align sentences inside each paragraph pair")
    args = ap.parse_args()

    # load chapters
    if args.en_json and args.tr_json:
        en_chap = json.loads(Path(args.en_json).read_text(encoding="utf-8"))
        tr_chap = json.loads(Path(args.tr_json).read_text(encoding="utf-8"))
    else:
        if not (args.en_epub and args.tr_epub):
            raise SystemExit("Provide --en_epub/--tr_epub or --en_json/--tr_json.")
        print("[extract] reading EPUBs …")
        en_chap = epub_to_chapter_dict(args.en_epub)
        tr_chap = epub_to_chapter_dict(args.tr_epub)

    # encoder
    enc = OnnxSentenceEncoder(
        args.model_dir,
        providers=("CUDAExecutionProvider","CPUExecutionProvider"),
        max_seq_length=256, prefer_int8=False
    )

    # chapter mapping
    if args.map_json and Path(args.map_json).exists():
        mapping = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
        en_order = mapping.get("en_order", list(mapping["alignment"].keys()))
        align_list = [{"en_section":k, "tr_sections": mapping["alignment"][k].get("tr_sections") or mapping["alignment"][k].get("nl_sections", [])} for k in en_order]
    else:
        print("[chapters] aligning (DTW) …")
        en_names, en_means = encode_means(enc, en_chap)
        tr_names, tr_means = encode_means(enc, tr_chap)
        align_list = dtw_align_sections_np(en_names, en_means, tr_names, tr_means, gap=args.gap)

    chapters_html: List[Tuple[str,str]] = []
    for m in align_list:
        en_key = m["en_section"]; tr_keys = m.get("tr_sections", [])
        en_paras = en_chap.get(en_key, [])
        tr_paras = []; [tr_paras.extend(tr_chap.get(tk, [])) for tk in tr_keys]

        para_pairs = align_paragraphs_vecalign(en_paras, tr_paras, enc, alignment_max_size=args.max_para_span)

        sent_details: List[List[Tuple[str,str,float]]] = []
        if args.sentence_level:
            for (lp, rp, _score) in para_pairs:
                sent_details.append(align_sentence_pairs(lp, rp, enc, alignment_max_size=5))
        else:
            sent_details = None

        title = en_key
        chapters_html.append((title, chapter_html(title, para_pairs, sent_details)))

    write_epub(args.title, args.author, chapters_html, Path(args.output))

if __name__ == "__main__":
    main()
