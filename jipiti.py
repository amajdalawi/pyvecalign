#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


def load_sections(path: Path) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Ensure values are lists of non-empty strings
    cleaned = {}
    for sec, sents in data.items():
        if not isinstance(sents, list):
            continue
        sents2 = [s.strip() for s in sents if isinstance(s, str) and s.strip()]
        if sents2:
            cleaned[sec] = sents2
    return cleaned


def mean_pool(emb_list: List[torch.Tensor]) -> torch.Tensor:
    # stack then mean; guard empty
    if not emb_list:
        # tiny vector to avoid crashes; will yield low cosine sim
        return torch.zeros(512)  # model-dependent default; overwritten later
    return torch.stack(emb_list, dim=0).mean(dim=0)


def encode_sections(
    model: SentenceTransformer,
    sections: Dict[str, List[str]],
    batch_size: int = 64,
) -> Tuple[List[str], List[torch.Tensor], List[List[torch.Tensor]]]:
    """
    Returns:
      names: list of section names (in original order of dict iteration)
      sec_embs: one embedding per section (mean of sentence embeddings)
      sent_embs_per_sec: list aligned with names, each is a list of sentence embeddings
    """
    names = list(sections.keys())
    sent_embs_per_sec = []
    sec_embs = []

    # Infer embedding size after first encode
    inferred_dim = None

    for name in names:
        sents = sections[name]
        if len(sents) == 0:
            sent_embs_per_sec.append([])
            sec_embs.append(None)
            continue

        sent_embs = model.encode(
            sents, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False
        )
        if isinstance(sent_embs, np.ndarray):
            sent_embs = torch.from_numpy(sent_embs)
        # Split into list of tensors per sentence to make pooling easy
        sent_list = [sent_embs[i] for i in range(sent_embs.shape[0])]
        pooled = sent_embs.mean(dim=0)

        if inferred_dim is None:
            inferred_dim = pooled.shape[-1]

        sent_embs_per_sec.append(sent_list)
        sec_embs.append(pooled)

    # Replace Nones with zeros of proper size
    if inferred_dim is None:
        inferred_dim = 512  # fallback
    for i, e in enumerate(sec_embs):
        if e is None:
            sec_embs[i] = torch.zeros(inferred_dim)

    return names, sec_embs, sent_embs_per_sec


def combine_mean(embs: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(embs, dim=0).mean(dim=0) if embs else embs


def align_greedy_windows(
    en_names: List[str],
    en_embs: List[torch.Tensor],
    nl_names: List[str],
    nl_embs: List[torch.Tensor],
    max_window: int = 3,
    min_sim: float = 0.40,
) -> List[Dict]:
    """
    Left-to-right greedy alignment.
    For each English section i, we choose a Dutch span [j..j+w-1] (w<=max_window)
    that maximizes cosine similarity between mean embeddings.

    We advance j past the chosen window to keep alignment monotonic and non-overlapping.

    Returns a list of dicts with:
      {
        "en_section": str,
        "nl_sections": [str, ...],
        "similarity": float
      }
    """
    results = []
    j = 0
    N = len(nl_names)

    for i, en_name in enumerate(en_names):
        best = (-1.0, None, None)  # (sim, start_j, window)
        for start in range(j, N):
            # try windows of size 1..max_window
            for w in range(1, max_window + 1):
                end = start + w
                if end > N:
                    break
                nl_span_emb = combine_mean(nl_embs[start:end])
                sim = util.cos_sim(en_embs[i], nl_span_emb).item()
                if sim > best[0]:
                    best = (sim, start, w)

            # Optional small early break heuristic: if we’re far to the right and
            # sim is dropping, you could break. Kept simple here.

        sim, start, w = best
        if start is None:
            # No candidate found (shouldn't happen) — assign empty
            results.append(
                {"en_section": en_name, "nl_sections": [], "similarity": float("nan")}
            )
            continue

        chosen = nl_names[start : start + w]
        if sim < min_sim:
            # Low confidence; still record, but mark
            results.append(
                {
                    "en_section": en_name,
                    "nl_sections": chosen,
                    "similarity": sim,
                    "warning": f"similarity {sim:.3f} below min_sim {min_sim}",
                }
            )
        else:
            results.append(
                {"en_section": en_name, "nl_sections": chosen, "similarity": sim}
            )

        # Move the Dutch pointer forward to keep monotonic alignment
        j = start + w

        # If we’ve exhausted Dutch sections, the remaining English sections get empty matches
        if j >= N and i + 1 < len(en_names):
            for k in range(i + 1, len(en_names)):
                results.append(
                    {
                        "en_section": en_names[k],
                        "nl_sections": [],
                        "similarity": float("nan"),
                        "warning": "no Dutch sections left",
                    }
                )
            break

    return results


def main():
    parser = argparse.ArgumentParser(description="Align English and Dutch sections.")
    parser.add_argument("--en", required=True, type=Path, help="Path to English JSON")
    parser.add_argument("--nl", required=True, type=Path, help="Path to Dutch JSON")
    parser.add_argument(
        "--model",
        default="distiluse-base-multilingual-cased-v2",
        help="SentenceTransformer model",
    )
    parser.add_argument(
        "--max_window",
        type=int,
        default=3,
        help="Max number of consecutive Dutch sections to merge per English section",
    )
    parser.add_argument(
        "--min_sim",
        type=float,
        default=0.40,
        help="Minimum cosine similarity to consider a confident match",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("section_alignment.json"),
        help="Where to write alignment JSON",
    )
    args = parser.parse_args()

    print("Loading sections…")
    en_sections = load_sections(args.en)
    nl_sections = load_sections(args.nl)

    if not en_sections or not nl_sections:
        raise SystemExit("One of the inputs is empty after cleaning. Check your JSONs.")

    print(f"Loaded EN sections: {len(en_sections)} | NL sections: {len(nl_sections)}")

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Encoding English sections…")
    en_names, en_sec_embs, _ = encode_sections(model, en_sections)
    print("Encoding Dutch sections…")
    nl_names, nl_sec_embs, _ = encode_sections(model, nl_sections)

    print("Aligning (greedy windows)…")
    matches = align_greedy_windows(
        en_names, en_sec_embs, nl_names, nl_sec_embs, args.max_window, args.min_sim
    )

    # Build mapping dict: EN section -> {nl_sections, similarity}
    mapping = {
        m["en_section"]: {
            "nl_sections": m["nl_sections"],
            "similarity": round(m["similarity"], 4) if isinstance(m["similarity"], float) and not math.isnan(m["similarity"]) else None,
            **({"warning": m["warning"]} if "warning" in m else {}),
        }
        for m in matches
    }

    print(f"Writing: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "max_window": args.max_window,
                "min_sim": args.min_sim,
                "english_file": str(args.en),
                "dutch_file": str(args.nl),
                "alignment": mapping,
                "en_order": en_names,
                "nl_order": nl_names,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Also print a concise table to stdout
    print("\nEN section  ->  NL sections  (similarity)")
    print("-" * 60)
    for m in matches:
        nl_join = " + ".join(m["nl_sections"]) if m["nl_sections"] else "(none)"
        sim = f"{m['similarity']:.3f}" if isinstance(m["similarity"], float) and not math.isnan(m["similarity"]) else "NA"
        warn = f"  [{m['warning']}]" if "warning" in m else ""
        print(f"{m['en_section']:<20} -> {nl_join}  ({sim}){warn}")


if __name__ == "__main__":
    main()
