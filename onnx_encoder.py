# onnx_encoder.py
import os
from pathlib import Path
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

def _find_tokenizer_dir(start: Path) -> Path:
    """Walk up until we see tokenizer files."""
    p = start if start.is_dir() else start.parent
    for cand in [p, *p.parents]:
        if (cand / "tokenizer.json").exists() or (cand / "vocab.txt").exists() or (cand / "sentencepiece.bpe.model").exists():
            return cand
    raise FileNotFoundError(f"Could not find tokenizer files starting from {start}")

def _pick_onnx_path(model_path: Path, prefer_int8: bool) -> Path:
    """Return a concrete .onnx file path. Search recursively if given a dir."""
    if model_path.suffix.lower() == ".onnx":
        return model_path

    # search order: ./onnx/*.onnx → ./*.onnx → any nested *.onnx
    roots = []
    if (model_path / "onnx").exists():
        roots.append(model_path / "onnx")
    roots.append(model_path)

    candidates = []
    for root in roots:
        for p in root.rglob("*.onnx"):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No .onnx file found under {model_path}")

    # scoring: prefer int8 for CPU; otherwise prefer non-quantized "model.onnx"
    def score(p: Path):
        name = p.name.lower()
        is_int8 = "qint8" in name or "int8" in name
        is_model = name == "model.onnx"
        # higher score wins
        return (
            3 if (prefer_int8 and is_int8) else 0
        ) + (
            2 if (not prefer_int8 and is_model) else 0
        ) + (
            1 if "avx512" in name or "vnni" in name else 0
        )

    candidates.sort(key=score, reverse=True)
    return candidates[0]

class OnnxSentenceEncoder:
    """
    Minimal SentenceTransformer-like wrapper:
      - encode(list[str], batch_size=..., normalize_embeddings=True) -> np.float32 [N, dim]
      - mean-pooling with attention mask (SBERT-style), then optional L2 norm
    """
    def __init__(self, model_path, providers=("CPUExecutionProvider",), max_seq_length=256, prefer_int8=True):
        model_path = Path(model_path)
        self.onnx_path = _pick_onnx_path(model_path, prefer_int8=prefer_int8)
        tok_dir = _find_tokenizer_dir(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=True)
        self.session = ort.InferenceSession(str(self.onnx_path), providers=list(providers))
        self.max_seq_length = max_seq_length

    def encode(self, texts, batch_size=256, normalize_embeddings=True, **_):
        out = []
        # Cache required input names once
        input_names = {i.name for i in self.session.get_inputs()}

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="np",
                return_token_type_ids=True,       # ask tokenizer for it
            )

            # Build the feed dynamically based on what the ONNX graph wants
            feed = {}
            if "input_ids" in input_names:
                feed["input_ids"] = enc["input_ids"].astype(np.int64, copy=False)
            if "attention_mask" in input_names:
                feed["attention_mask"] = enc["attention_mask"].astype(np.int64, copy=False)
            if "token_type_ids" in input_names:
                tti = enc.get("token_type_ids")
                if tti is None:
                    # some tokenizers (e.g., RoBERTa) don't use segment IDs
                    tti = np.zeros_like(enc["input_ids"], dtype=np.int64)
                else:
                    tti = tti.astype(np.int64, copy=False)
                feed["token_type_ids"] = tti

            # Run ONNX: assume first output is last_hidden_state: (B, S, H)
            last_hidden, = self.session.run(None, feed)

            # mean pooling with mask
            mask = enc["attention_mask"][..., None].astype(np.float32)
            summed = (last_hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            embs = summed / counts

            if normalize_embeddings:
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

            out.append(embs.astype(np.float32))
        return np.vstack(out)