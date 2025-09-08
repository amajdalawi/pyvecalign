# onnx_encoder.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ---------- helpers ----------

def _find_tokenizer_dir(start: Path) -> Path:
    """
    Walk up from `start` until we find tokenizer files.
    Accepts either a model folder (with tokenizer.json at top)
    or a subfolder like .../onnx/.
    """
    p = start if start.is_dir() else start.parent
    markers = {"tokenizer.json", "vocab.txt", "sentencepiece.bpe.model", "spiece.model"}
    for cand in [p, *p.parents]:
        if any((cand / m).exists() for m in markers):
            return cand
    raise FileNotFoundError(f"Could not find tokenizer files above {start}")

def _pick_onnx_path(model_path: Path, prefer_int8: bool) -> Path:
    """
    If `model_path` is a file, return it. If it's a dir, search recursively.
    Preference:
      CPU (prefer_int8=True): model_qint8_* (AVX512/VNNI, AVX2) > model_qint8.onnx > model.onnx > first *.onnx
      GPU (prefer_int8=False): model.onnx > first *.onnx
    """
    if model_path.suffix.lower() == ".onnx":
        return model_path

    roots = []
    if (model_path / "onnx").exists():
        roots.append(model_path / "onnx")
    roots.append(model_path)

    cands = [p for root in roots for p in root.rglob("*.onnx")]
    if not cands:
        raise FileNotFoundError(f"No .onnx file found under {model_path}")

    def score(p: Path) -> int:
        name = p.name.lower()
        is_int8 = ("qint8" in name) or ("int8" in name)
        is_model = (name == "model.onnx")
        return (
            (3 if (prefer_int8 and is_int8) else 0) +
            (2 if (not prefer_int8 and is_model) else 0) +
            (1 if ("avx512" in name or "vnni" in name) else 0)
        )

    cands.sort(key=score, reverse=True)
    return cands[0]

# ---------- encoder ----------

class OnnxSentenceEncoder:
    """
    Minimal SentenceTransformer-like wrapper using ONNX Runtime.
    - encode(list[str], batch_size=..., normalize_embeddings=True) -> np.float32 [N, dim]
    - SBERT-style mean pooling with attention mask, then L2 normalize.
    """

    def __init__(
        self,
        model_path,
        providers=("CPUExecutionProvider",),
        max_seq_length: int = 256,
        prefer_int8: bool = True,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = 1,
        optimized_model_cache: bool = False,
    ):
        model_path = Path(model_path)

        self.onnx_path = _pick_onnx_path(model_path, prefer_int8=prefer_int8)
        tok_dir = _find_tokenizer_dir(model_path)

        # Load tokenizer.json directly (no transformers import)
        tok_json = tok_dir / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {tok_dir}")
        self.tok = Tokenizer.from_file(str(tok_json))

        # Set truncation / padding once
        pad_id = self.tok.token_to_id("[PAD]") or 0
        self.tok.enable_truncation(max_length=max_seq_length)
        self.tok.enable_padding(length=max_seq_length, pad_id=pad_id, pad_token="[PAD]")

        # Build ORT session with sensible opts
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is None:
            intra_op_threads = os.cpu_count() or 4
        so.intra_op_num_threads = intra_op_threads
        if inter_op_threads is not None:
            so.inter_op_num_threads = inter_op_threads
        so.enable_mem_pattern = True
        if optimized_model_cache:
            so.optimized_model_filepath = str(self.onnx_path.with_suffix(".opt.onnx"))

        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=so, providers=list(providers))
        self.max_seq_length = max_seq_length

        # Cache required input names once
        self._input_names = {i.name for i in self.session.get_inputs()}

    def _batch_encode_ids(self, texts):
        encs = self.tok.encode_batch(texts)
        ids  = np.stack([np.array(e.ids, dtype=np.int64) for e in encs])
        mask = np.stack([np.array(e.attention_mask, dtype=np.int64) for e in encs])
        if encs and encs[0].type_ids:
            tti = np.stack([np.array(e.type_ids, dtype=np.int64) for e in encs])
        else:
            tti = np.zeros_like(ids, dtype=np.int64)
        return ids, mask, tti

    def encode(self, texts, batch_size=256, normalize_embeddings=True, **_):
        outs = []
        need = self._input_names

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            ids, mask, tti = self._batch_encode_ids(batch)

            feed = {}
            if "input_ids" in need:       feed["input_ids"] = ids
            if "attention_mask" in need:  feed["attention_mask"] = mask
            if "token_type_ids" in need:  feed["token_type_ids"] = tti

            (last_hidden,) = self.session.run(None, feed)  # (B, S, H)

            # Mean pooling with mask
            m = mask[..., None].astype(np.float32)
            embs = (last_hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

            if normalize_embeddings:
                embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

            outs.append(embs.astype(np.float32))

        return np.vstack(outs)
