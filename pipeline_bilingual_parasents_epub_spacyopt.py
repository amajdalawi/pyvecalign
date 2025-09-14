#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EPUB -> bilingual EPUB with paragraph alignment + sentence alignment.
Sentence splitting can use:
  - FAST (default): blingfire if available, else regex (protects quoted spans)
  - spaCy (toggle with --use_spacy_split), with quote-repair for «»/“”

Requires local modules:
  - onnx_encoder.OnnxSentenceEncoder
  - vecalign.align_in_memory

Deps:
  pip install ebooklib beautifulsoup4 lxml onnxruntime
Optional:
  pip install blingfire
  pip install spacy && python -m spacy download en_core_web_sm fr_core_news_sm nl_core_news_sm
"""

from __future__ import annotations
import argparse, html, json, re, sys, warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# ---- local modules ----
try:
    from onnx_encoder import OnnxSentenceEncoder
except Exception as e:
    raise SystemExit(f"Import error: OnnxSentenceEncoder (onnx_encoder.py). {e}")
try:
    from vecalign import align_in_memory
except Exception as e:
    raise SystemExit(f"Import error: align_in_memory (vecalign.py). {e}")

# ========================= Text utils =========================
NBSP = "\u00A0"
WORD_RX = re.compile(r"\w", re.UNICODE)

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace(NBSP, " ").strip())

def keep_texty(s: str) -> bool:
    return bool(s and WORD_RX.search(s))

# ================= EPUB -> {section: [paragraphs]} =================
BAN_WORDS = [
    "title","toc","content","copyright","jacket","dedication",
    "acknowledg","appendix","ad-card","advert","back","front","about",
    "cover","imprint"
]
def is_banned_key(name: str) -> bool:
    n = (name or "").lower()
    return any(b in n for b in BAN_WORDS)

_num_re = re.compile(r"(\d{1,4})")
def natnum(name: str) -> int:
    m = _num_re.search(name or "")
    return int(m.group(1)) if m else 10**9

def html_to_paras(xhtml: bytes | str) -> Tuple[str, List[str]]:
    text = xhtml.decode("utf-8", errors="ignore") if isinstance(xhtml,(bytes,bytearray)) else (xhtml or "")
    soup = BeautifulSoup(text, features="xml") if ("<?xml" in text or "xmlns" in text) else BeautifulSoup(text, "lxml")
    title = ""
    for t in soup.select("h1,h2,h3,title"):
        title = norm(t.get_text(" ", strip=True))
        if title: break
    nodes = soup.find_all(["p","li","blockquote"])
    paras = [norm(n.get_text(" ", strip=True)) for n in nodes if n.get_text(strip=True)]
    paras = [p for p in paras if keep_texty(p)]
    return title, paras

def epub_to_chapter_dict(epub_path: Path) -> Dict[str, List[str]]:
    book = epub.read_epub(str(epub_path))
    raw: Dict[str, List[str]] = {}
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        name = item.get_name()
        if is_banned_key(name): 
            continue
        _, paras = html_to_paras(item.get_content())
        raw[name] = paras
    return {k: raw[k] for k in sorted(raw.keys(), key=lambda k: (natnum(k), k))}

# ================= Chapter DTW alignment =================
def encode_means(encoder: OnnxSentenceEncoder, sections: Dict[str, List[str]]) -> Tuple[List[str], np.ndarray]:
    names = list(sections.keys())
    means = []
    for k in names:
        paras = sections[k]
        if not paras:
            means.append(None); continue
        embs = encoder.encode(paras, normalize_embeddings=True)
        means.append(embs.mean(axis=0))
    dim = means[0].shape[0] if means and means[0] is not None else 384
    fixed = [m if m is not None else np.zeros((dim,), dtype=np.float32) for m in means]
    return names, np.stack(fixed, axis=0).astype(np.float32)

def dtw_align_sections(en_names: List[str], en_vecs: np.ndarray,
                       tr_names: List[str], tr_vecs: np.ndarray,
                       gap: float=-0.25) -> List[Dict]:
    E, N = en_vecs.shape[0], tr_vecs.shape[0]
    S = en_vecs @ tr_vecs.T
    dp = np.full((E+1,N+1), -1e9, dtype=np.float32)
    bt = np.zeros((E+1,N+1), dtype=np.int8)  # 1=diag 2=up 3=left
    dp[0,0] = 0.0
    for i in range(E+1):
        for j in range(N+1):
            if i<E and j<N:
                v = dp[i,j] + S[i,j]
                if v > dp[i+1,j+1]: dp[i+1,j+1]=v; bt[i+1,j+1]=1
            if i<E:
                v = dp[i,j] + gap
                if v > dp[i+1,j]: dp[i+1,j]=v; bt[i+1,j]=2
            if j<N:
                v = dp[i,j] + gap
                if v > dp[i,j+1]: dp[i,j+1]=v; bt[i,j+1]=3
    # backtrack
    i,j=E,N; pairs=[]
    while i>0 or j>0:
        mv = bt[i,j]
        if mv==1: i-=1; j-=1; pairs.append((i,j))
        elif mv==2: i-=1; pairs.append((i,-1))
        else: j-=1; pairs.append((-1,j))
    pairs.reverse()
    agg, sims = {}, {}
    for ii,jj in pairs:
        if ii>=0 and jj>=0:
            agg.setdefault(ii, []).append(jj)
            sims.setdefault(ii, []).append(float(S[ii,jj]))
        elif ii>=0 and jj<0:
            agg.setdefault(ii, [])
    out=[]
    for idx,name in enumerate(en_names):
        js = sorted(set(agg.get(idx, [])))
        out.append({"en_section": name,
                    "tr_sections": [tr_names[j] for j in js],
                    "similarity": None if idx not in sims else float(np.mean(sims.get(idx,[])))})
    return out

# ================= Paragraph alignment =================
def align_paragraphs_vecalign(src_paras: List[str], tgt_paras: List[str],
                              encoder: OnnxSentenceEncoder, alignment_max_size:int=10
) -> List[Tuple[str,str,float]]:
    if not src_paras or not tgt_paras: return []
    aligns, scores = align_in_memory(
        [norm(x) for x in src_paras], [norm(y) for y in tgt_paras],
        model=encoder, alignment_max_size=alignment_max_size, one_to_many=None
    )
    out=[]
    for (xs,ys), s in zip(aligns, scores):
        left  = " ".join(src_paras[i] for i in xs).strip()
        right = " ".join(tgt_paras[j] for j in ys).strip()
        if left or right: out.append((left, right, float(s)))
    return out

# ================= Sentence splitters =================
# FAST: blingfire -> regex (protect quoted spans)
QUOTE_SPAN_FR = re.compile(r"«[^»]*»", flags=re.DOTALL)
QUOTE_SPAN_EN = re.compile(r"“[^”]*”", flags=re.DOTALL)
def split_sentences_fast(text: str) -> List[str]:
    t = norm(text)
    # try blingfire
    try:
        import blingfire as bf
        return [norm(s) for s in bf.text_to_sentences(t).split("\n") if keep_texty(s)]
    except Exception:
        pass
    # mask quoted spans so we don't split inside
    placeholders = {}
    def _mask(pattern, s, prefix):
        idx=0
        def rep(m):
            nonlocal idx
            key=f"⟦{prefix}{idx}⟧"; idx+=1; placeholders[key]=m.group(0); return key
        return pattern.sub(rep, s)
    masked = _mask(QUOTE_SPAN_FR, t, "QF")
    masked = _mask(QUOTE_SPAN_EN, masked, "QE")
    # split on [.?!…] + space + optional quote/paren + capital
    rx = re.compile(r"(?<=[\.\?\!…])\s+(?=[“\"'\(»]?[A-ZÀ-Ý])")
    parts = rx.split(masked)
    # restore
    out=[]
    for p in parts:
        for k,v in placeholders.items():
            if k in p: p = p.replace(k, v)
        p = norm(p)
        if keep_texty(p): out.append(p)
    return out

# spaCy + quote-repair (cache models, optional)
_SPACY_CACHE = {}
_QONLY = re.compile(r"^[«»“”\"'‹›„‟]+$")
def _load_spacy(lang: str):
    try:
        import spacy, spacy.cli
    except Exception as e:
        raise RuntimeError(f"spaCy not installed: {e}")
    name = {"en":"en_core_web_sm","fr":"fr_core_news_sm","nl":"nl_core_news_sm"}.get(lang.lower(), "xx_sent_ud_sm")
    if name in _SPACY_CACHE: return _SPACY_CACHE[name]
    try:
        nlp = spacy.load(name, exclude=["ner","tagger","lemmatizer","morphologizer"])
    except OSError:
        spacy.cli.download(name)
        nlp = spacy.load(name, exclude=["ner","tagger","lemmatizer","morphologizer"])
    if "senter" not in nlp.pipe_names and "parser" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    _SPACY_CACHE[name] = nlp
    return nlp

def split_sentences_spacy_fixed(text: str, lang: str) -> List[str]:
    try:
        nlp = _load_spacy(lang)
    except Exception as e:
        print(f"[warn] spaCy unavailable ({e}); falling back to fast splitter.", file=sys.stderr)
        return split_sentences_fast(text)
    sents = [norm(s.text) for s in nlp(norm(text)).sents if s.text.strip()]
    out=[]; carry=""
    for t in sents:
        if _QONLY.fullmatch(t):
            if t in {"«","“"}: carry = t
            elif out: out[-1] = (out[-1] + " " + t).strip()
            else: carry = (carry + " " + t).strip()
            continue
        if (t.startswith("»") or t.startswith("”")) and out:
            out[-1] = (out[-1] + " " + t).strip(); continue
        if carry:
            t = (carry + " " + t).strip(); carry=""
        if t.endswith("«") or t.endswith("“"):
            carry = t[-1]; t = t[:-1].rstrip()
        out.append(t)
    if carry and out: out[-1] = (out[-1] + " " + carry).strip()
    return out

def get_splitter(use_spacy: bool, lang: str):
    if use_spacy:
        return lambda txt: split_sentences_spacy_fixed(txt, lang)
    return split_sentences_fast

# ================= Sentence alignment =================
def align_sentence_pairs(src_para: str, tgt_para: str,
                         encoder: OnnxSentenceEncoder,
                         split_src, split_tgt,
                         alignment_max_size:int=5) -> List[Tuple[str,str,float]]:
    src_s = split_src(src_para)
    tgt_s = split_tgt(tgt_para)
    if not src_s or not tgt_s: return []
    aligns, scores = align_in_memory(
        src_s, tgt_s, model=encoder, alignment_max_size=alignment_max_size, one_to_many=None
    )
    out=[]
    for (xs,ys), s in zip(aligns, scores):
        left  = " ".join(src_s[i] for i in xs).strip()
        right = " ".join(tgt_s[j] for j in ys).strip()
        if left or right: out.append((left, right, float(s)))
    return out

# ================= EPUB rendering (stacked sentences) =================
def chapter_html_stacked(title: str,
                         para_pairs: List[Tuple[str,str,float]],
                         sent_pairs_per_para: List[List[Tuple[str,str,float]]]) -> str:
    blocks=[]
    for idx, (para_pair, sent_pairs) in enumerate(zip(para_pairs, sent_pairs_per_para), start=1):
        if not sent_pairs:  # fallback
            sent_pairs = [(para_pair[0], para_pair[1], para_pair[2])]
        lines=[]
        for si,(se,st,_ss) in enumerate(sent_pairs):
            lines.append(f'<div class="line orig">{html.escape(se)}</div>')
            lines.append(f'<div class="line trgt">{html.escape(st)}</div>')
            if si != len(sent_pairs)-1:
                lines.append('<div class="sep">==</div>')
        blocks.append(f"""
        <div class="para">
          <div class="phead">{idx}</div>
          <div class="content">
            {''.join(lines)}
          </div>
        </div>""")
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
    .sep {{ color:#666; margin:.35rem 0; font-family:monospace; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  {body}
</body>
</html>"""

def _extract_body(html_str: str) -> str:
    m = re.search(r"<body[^>]*>(.*?)</body>", html_str, flags=re.I|re.S)
    return (m.group(1) if m else html_str).strip()

def write_epub(book_title: str, author: str, chapters: List[Tuple[str,str]], out_path: Path):
    book = epub.EpubBook()
    book.set_identifier("bilingual-" + re.sub(r"[^a-z0-9]+","-", book_title.lower()))
    book.set_title(book_title)
    if author: book.add_author(author)
    spine=["nav"]; toc=[]; items=[]
    for idx,(title, html_str) in enumerate(chapters,1):
        body_html = _extract_body(html_str) or f"<h1>{html.escape(title)}</h1><p></p>"
        fn = f"chap_{idx:03d}.xhtml"
        c = epub.EpubHtml(title=title, file_name=fn, lang="en")
        c.content = body_html.encode("utf-8")   # body-only bytes
        book.add_item(c); items.append(c)
        toc.append(epub.Link(fn, title, f"chap{idx:03d}")); spine.append(c)
    if not items:
        ph = epub.EpubHtml(title="Empty", file_name="chap_000.xhtml", lang="en")
        ph.content = b"<h1>Empty</h1><p>No aligned content.</p>"
        book.add_item(ph); items.append(ph); toc.append(epub.Link("chap_000.xhtml","Empty","chap000")); spine.append(ph)
    book.toc = tuple(toc); book.add_item(epub.EpubNcx()); book.add_item(epub.EpubNav()); book.spine = spine
    epub.write_epub(str(out_path), book)
    print(f"[epub] Wrote {out_path}")

# ============================ Main ============================
def main():
    ap = argparse.ArgumentParser(description="Build bilingual EPUB with paragraph+sentence alignment; spaCy splitting optional.")
    # Inputs
    ap.add_argument("--en_epub", type=Path, help="English EPUB path")
    ap.add_argument("--tr_epub", type=Path, help="Translation EPUB path")
    ap.add_argument("--en_json", type=Path, help="Optional: pre-extracted JSON {section:[paragraphs]} for EN")
    ap.add_argument("--tr_json", type=Path, help="Optional: pre-extracted JSON {section:[paragraphs]} for TR")
    ap.add_argument("--map_json", type=Path, help="Optional: chapter mapping JSON")
    # Output
    ap.add_argument("--title", required=True); ap.add_argument("--author", default="")
    ap.add_argument("--output", default="bilingual.epub")
    # Model / params
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--gap", type=float, default=-0.25)
    ap.add_argument("--max_para_span", type=int, default=10)
    ap.add_argument("--sent_window", type=int, default=5)
    # NEW: sentence splitting control
    ap.add_argument("--use_spacy_split", action="store_true", help="Use spaCy-based sentence splitting (slower, more accurate).")
    ap.add_argument("--src_lang", default="en", help="Language code for source (spaCy mode): en|fr|nl|…")
    ap.add_argument("--tr_lang",  default="fr", help="Language code for target (spaCy mode): en|fr|nl|…")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--prefer_int8", action="store_true")  # INT8 also runs on CUDA in ORT
    args = ap.parse_args()

    # Load chapters
    if args.en_json and args.tr_json:
        en_chap = json.loads(Path(args.en_json).read_text(encoding="utf-8"))
        tr_chap = json.loads(Path(args.tr_json).read_text(encoding="utf-8"))
    else:
        if not (args.en_epub and args.tr_epub):
            raise SystemExit("Provide --en_epub/--tr_epub or --en_json/--tr_json.")
        print("[extract] reading EPUBs …")
        en_chap = epub_to_chapter_dict(args.en_epub)
        tr_chap = epub_to_chapter_dict(args.tr_epub)

    # Encoder

    # create encoder (GPU only)
    enc = OnnxSentenceEncoder(
        r".\models\miniLM-L12",   # folder containing model.onnx (and optionally /onnx/model_qint8*.onnx)
        gpu_id=0,
        max_seq_length=256,
        default_batch_size=64,
        prefer_int8=False,        # usually prefer FP model on CUDA
        enable_cuda_graph=False   # leave off on consumer GPUs
)    # Chapter mapping
    if args.map_json and Path(args.map_json).exists():
        mapping = json.loads(Path(args.map_json).read_text(encoding="utf-8"))
        en_order = mapping.get("en_order", list(mapping["alignment"].keys()))
        align_list = [{"en_section": k,
                       "tr_sections": mapping["alignment"][k].get("tr_sections") or mapping["alignment"][k].get("nl_sections", [])}
                      for k in en_order]
        print(f"[chapters] using mapping for {len(align_list)} chapters")
    else:
        print("[chapters] aligning (DTW) …")
        en_names, en_means = encode_means(enc, en_chap)
        tr_names, tr_means = encode_means(enc, tr_chap)
        align_list = dtw_align_sections(en_names, en_means, tr_names, tr_means, gap=args.gap)
        print(f"[chapters] produced {len(align_list)} entries")

    # Choose splitters
    split_src = get_splitter(args.use_spacy_split, args.src_lang)
    split_tr  = get_splitter(args.use_spacy_split, args.tr_lang)

    # Per-chapter: paragraph align, then sentence align with chosen splitter
    chapters_html: List[Tuple[str,str]] = []
    for m in align_list:
        en_key = m["en_section"]; tr_keys = m.get("tr_sections", [])
        en_paras = en_chap.get(en_key, [])
        tr_paras = []; [tr_paras.extend(tr_chap.get(tk, [])) for tk in tr_keys]

        para_pairs = align_paragraphs_vecalign(en_paras, tr_paras, enc, alignment_max_size=args.max_para_span)

        sent_pairs_per_para: List[List[Tuple[str,str,float]]] = []
        for (lp, rp, _score) in para_pairs:
            sent_pairs = align_sentence_pairs(lp, rp, enc, split_src, split_tr, alignment_max_size=args.sent_window)
            sent_pairs_per_para.append(sent_pairs)

        title = en_key
        chapters_html.append((title, chapter_html_stacked(title, para_pairs, sent_pairs_per_para)))

    # Write EPUB
    write_epub(args.title, args.author, chapters_html, Path(args.output))

if __name__ == "__main__":
    main()
