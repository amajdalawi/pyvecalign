#!/usr/bin/env python3
import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


# ----------------------------- I/O & Cleaning ------------------------------

def load_sections(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cleaned = {}
    for sec, sents in data.items():
        if not isinstance(sents, list):
            continue
        ss = [s.strip() for s in sents if isinstance(s, str) and s.strip()]
        if ss:
            cleaned[sec] = ss
    return cleaned


# Heuristics: keep only story chapters; drop front/back matter.
BAN_WORDS = [
    "title", "toc", "content", "copyright", "jacket", "dedication",
    "dankwoord", "acknowledgement", "acknowledgments", "acknowledgements",
    "appendix", "ad-card", "back", "front", "part", "einde", "about", "cover"
]

CH_EN_RE = re.compile(r"(?:^|/)(?:chapter|chap|ch|bk)?\s*0*(\d+)\.(?:x?html|xml|txt)$", re.I)
CH_NL_RE = re.compile(r"(?:^|/)(?:text/)?(?:chapter|chap|ch|hoofdstuk)?\s*0*(\d+)\.(?:x?html|html|xml|txt)$", re.I)

def is_banned(key: str) -> bool:
    k = key.lower()
    return any(b in k for b in BAN_WORDS)

def natural_num(key: str, patterns: Iterable[re.Pattern]) -> int:
    for pat in patterns:
        m = pat.search(key)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    # fallback: pull any trailing number
    m = re.search(r"(\d+)", key)
    return int(m.group(1)) if m else 10**9

def filter_and_sort(d: Dict[str, List[str]], is_english: bool) -> Dict[str, List[str]]:
    pat_list = [CH_EN_RE] if is_english else [CH_NL_RE]
    kept = {k: v for k, v in d.items() if not is_banned(k)}
    ordered_keys = sorted(kept.keys(), key=lambda k: (natural_num(k, pat_list), k))
    return {k: kept[k] for k in ordered_keys}


# ----------------------------- Embeddings ----------------------------------

def encode_sections(
    model: SentenceTransformer,
    sections: Dict[str, List[str]],
    batch_size: int = 64
) -> Tuple[List[str], List[torch.Tensor]]:
    names = list(sections.keys())
    embs: List[torch.Tensor] = []
    for name in names:
        sents = sections[name]
        if not sents:
            embs.append(None)  # placeholder
            continue
        arr = model.encode(sents, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        if isinstance(arr, np.ndarray):
            arr = torch.from_numpy(arr)
        pooled = arr.mean(dim=0)
        embs.append(pooled)
    # replace Nones (shouldn’t happen after cleaning) with zeros of correct size
    dim = embs[0].shape[-1] if embs and embs[0] is not None else 512
    embs = [e if e is not None else torch.zeros(dim) for e in embs]
    return names, embs


# ----------------------------- DTW Alignment -------------------------------

def dtw_align_sections(
    en_names: List[str], en_embs: List[torch.Tensor],
    nl_names: List[str], nl_embs: List[torch.Tensor],
    gap_penalty: float = -0.25
) -> List[Dict]:
    """
    Needleman–Wunsch style DP on section-level cosine similarities.
    Allowed moves:
      diag -> align EN[i] with NL[j]    (score = cos_sim)
      up   -> EN[i] aligned to GAP      (score = gap_penalty)
      left -> NL[j] aligned to GAP      (score = gap_penalty)
    Returns list of {en_section, nl_sections[], similarity(mean over aligned pairs)}.
    """
    E, N = len(en_names), len(nl_names)

    # similarity matrix S[i, j]
    S = torch.zeros((E, N))
    for i in range(E):
        for j in range(N):
            S[i, j] = util.cos_sim(en_embs[i], nl_embs[j])

    # DP
    dp = torch.full((E + 1, N + 1), -1e9)
    bt = np.zeros((E + 1, N + 1), dtype=np.int8)  # 1=diag, 2=up, 3=left
    dp[0, 0] = 0.0

    for i in range(E + 1):
        for j in range(N + 1):
            if i < E and j < N:
                val = dp[i, j] + S[i, j]
                if val > dp[i + 1, j + 1]:
                    dp[i + 1, j + 1] = val
                    bt[i + 1, j + 1] = 1
            if i < E:
                val = dp[i, j] + gap_penalty
                if val > dp[i + 1, j]:
                    dp[i + 1, j] = val
                    bt[i + 1, j] = 2
            if j < N:
                val = dp[i, j] + gap_penalty
                if val > dp[i, j + 1]:
                    dp[i, j + 1] = val
                    bt[i, j + 1] = 3

    # Backtrack to a path of pairs (i or None, j or None)
    i, j = E, N
    path: List[Tuple[int, int]] = []  # (i or -1, j or -1)
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 1:       # diag
            i, j = i - 1, j - 1
            path.append((i, j))
        elif move == 2:     # up (gap in NL)
            i = i - 1
            path.append((i, -1))
        else:               # left (gap in EN)
            j = j - 1
            path.append((-1, j))
    path.reverse()

    # Aggregate: for each EN section, collect all NL indices aligned alongside it
    agg: Dict[int, List[int]] = {}
    pair_sims: Dict[int, List[float]] = {}
    current_en = None

    for ii, jj in path:
        if ii >= 0 and jj >= 0:  # real pair
            current_en = ii
            agg.setdefault(ii, []).append(jj)
            pair_sims.setdefault(ii, []).append(float(S[ii, jj]))
        elif ii >= 0 and jj < 0:  # EN aligned to gap -> ensure EN key exists with empty span
            current_en = ii
            agg.setdefault(ii, [])
        else:
            # NL aligned to gap; if we have a current_en, treat as continued span
            pass

    # Build results in EN order
    results = []
    for idx, en in enumerate(en_names):
        nls = sorted(set(agg.get(idx, [])))
        sim = None
        sims = pair_sims.get(idx, [])
        if sims:
            sim = round(float(np.mean(sims)), 3)
        results.append({
            "en_section": en,
            "nl_sections": [nl_names[j] for j in nls],
            "similarity": sim
        })
    return results


# ----------------------------- CLI & Runner --------------------------------

def main():
    ap = argparse.ArgumentParser(description="Align sections (EN ↔ NL) via DTW.")
    ap.add_argument("--en", required=True, type=Path, help="English JSON (section -> [sentences])")
    ap.add_argument("--nl", required=True, type=Path, help="Dutch JSON (section -> [sentences])")
    ap.add_argument("--model", default="distiluse-base-multilingual-cased-v2",
                    help="SentenceTransformer model (multilingual)")
    ap.add_argument("--gap", type=float, default=-0.25, help="Gap penalty for DTW (more negative → fewer gaps)")
    ap.add_argument("--output", type=Path, default=Path("section_alignment.json"),
                    help="Output mapping JSON")
    args = ap.parse_args()

    print("Loading sections…")
    en_raw = load_sections(args.en)
    nl_raw = load_sections(args.nl)

    # Filter to story chapters & natural sort
    en_secs = filter_and_sort(en_raw, is_english=True)
    nl_secs = filter_and_sort(nl_raw, is_english=False)

    if not en_secs or not nl_secs:
        raise SystemExit("After filtering, one of the inputs is empty. Check your JSONs / patterns.")

    print(f"Loaded EN sections: {len(en_secs)} | NL sections: {len(nl_secs)}")
    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Encoding English sections…")
    en_names, en_embs = encode_sections(model, en_secs)
    print("Encoding Dutch sections…")
    nl_names, nl_embs = encode_sections(model, nl_secs)

    print("Aligning (DTW)…")
    matches = dtw_align_sections(en_names, en_embs, nl_names, nl_embs, gap_penalty=args.gap)

    # Write JSON
    mapping = {
        m["en_section"]: {
            "nl_sections": m["nl_sections"],
            "similarity": m["similarity"]
        } for m in matches
    }

    print(f"Writing: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "gap_penalty": args.gap,
            "english_file": str(args.en),
            "dutch_file": str(args.nl),
            "alignment": mapping,
            "en_order": en_names,
            "nl_order": nl_names
        }, f, ensure_ascii=False, indent=2)

    # Pretty table
    print("\nEN section  ->  NL sections  (mean similarity)")
    print("-" * 64)
    for m in matches:
        nl_join = " + ".join(m["nl_sections"]) if m["nl_sections"] else "(none)"
        sim = "NA" if m["similarity"] is None else f"{m['similarity']:.3f}"
        print(f"{m['en_section']:<22} -> {nl_join}  ({sim})")


if __name__ == "__main__":
    main()
