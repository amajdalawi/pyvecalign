# onnx_encoder.py
from __future__ import annotations
from pathlib import Path
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

# ---------- helpers ----------
def _find_tokenizer_dir(model_dir: Path) -> Path:
    """
    Look for a directory that contains tokenizer.json: either the root,
    ./onnx/, or any immediate child.
    """
    candidates = [
        model_dir,
        model_dir / "onnx",
        *[p for p in model_dir.iterdir() if p.is_dir()]
    ]
    for p in candidates:
        if (p / "tokenizer.json").exists():
            return p
    raise FileNotFoundError(f"tokenizer.json not found under {model_dir}")


def _pick_onnx_path(model_dir: Path, prefer_int8: bool) -> Path:
    """
    Choose an ONNX file:
      - prefer INT8 quantized (model_qint8*.onnx) if prefer_int8=True AND file exists
      - else use model.onnx
      - search ./, ./onnx/
    """
    search_dirs = [model_dir, model_dir / "onnx"]
    q8: List[Path] = []
    f32: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        q8.extend(d.glob("model_qint8*.onnx"))
        f32.extend(d.glob("model.onnx"))
    if prefer_int8 and q8:
        return sorted(q8)[0]
    if f32:
        return sorted(f32)[0]
    if q8:
        return sorted(q8)[0]
    raise FileNotFoundError(f"No ONNX model found in {search_dirs}. "
                            f"Expected model.onnx or model_qint8*.onnx")


# ---------- encoder ----------

class OnnxSentenceEncoder:
    """
    Minimal SentenceTransformer-like GPU encoder (CUDA only).
      - encode(list[str], batch_size=..., normalize_embeddings=True) -> float32 [N, dim]
      - mean pooling with attention mask, then L2 normalize (SBERT style)
      - adaptive batching: if CUDA throws (e.g., illegal access / OOM), halve the batch and retry on GPU
    """

    def __init__(
        self,
        model_path: str | os.PathLike,
        *,
        gpu_id: int = 0,
        max_seq_length: int = 256,
        default_batch_size: int = 64,
        prefer_int8: bool = False,
        enable_cuda_graph: bool = False,
    ):
        model_path = Path(model_path)
        # ---- Files ----
        self.onnx_path = _pick_onnx_path(model_path, prefer_int8=prefer_int8)
        tok_dir = _find_tokenizer_dir(model_path)

        # ---- Tokenizer ----
        tok_json = tok_dir / "tokenizer.json"
        self.tok = Tokenizer.from_file(str(tok_json))
        self.max_seq_length = int(max_seq_length)
        self.default_batch_size = int(default_batch_size)

        # ensure truncation/padding are set (Tokenizers handles special tokens via post-processor)
        self.pad_id = self.tok.token_to_id("[PAD]") or 0
        self.tok.enable_truncation(max_length=self.max_seq_length)
        self.tok.enable_padding(length=self.max_seq_length, pad_id=self.pad_id, pad_token="[PAD]")

        # ---- ONNX Runtime (CUDA only) ----
        # sanity: ensure CUDA EP is available
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider not available. "
                               "Install 'onnxruntime-gpu' and verify your CUDA/cuDNN setup.")

        so = ort.SessionOptions()
        # More conservative settings for consumer GPUs / Windows laptops:
        so.enable_mem_pattern = False
        so.log_severity_level = 2  # INFO
        # If you still see CUDA error 700, you can also try:
        # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        cuda_opts = {
            "device_id": int(gpu_id),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": 1,
            "cudnn_conv_algo_search": "DEFAULT",
            "enable_cuda_graph": 1 if enable_cuda_graph else 0,
        }
        providers = [("CUDAExecutionProvider", cuda_opts)]
        self.session = ort.InferenceSession(str(self.onnx_path), so, providers=providers)

        # Infer hidden size from model IO to allocate arrays correctly
        out0 = self.session.get_outputs()[0]
        # last_hidden_state: (batch, seq, hidden)
        self.hidden_size = int(out0.shape[-1]) if isinstance(out0.shape[-1], int) else 384

    # --------------------------- public API ---------------------------

    def encode(
        self,
        texts: List[str],
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of strings to L2-normalized sentence embeddings (float32).
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list/tuple of strings")
        if len(texts) == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        if batch_size is None:
            batch_size = self.default_batch_size
        batch_size = int(max(1, batch_size))

        # Tokenize once
        input_ids, attention_mask = self._tokenize(texts, self.max_seq_length)  # int64 [N,S], [N,S]

        outputs: List[np.ndarray] = []
        i = 0
        cur_bs = batch_size

        while i < len(texts):
            j = min(i + cur_bs, len(texts))
            feed = {
                "input_ids": input_ids[i:j],
                "attention_mask": attention_mask[i:j],
            }
            try:
                (last_hidden,) = self.session.run(None, feed)  # (B,S,H)
            except Exception as e:
                # CUDA illegal access / OOM / driver hiccup → shrink the batch and retry on GPU
                if cur_bs == 1:
                    raise  # cannot shrink further; propagate the error
                cur_bs = max(1, cur_bs // 2)
                continue  # retry same i with smaller batch

            # SBERT mean pooling with attention mask
            mask = attention_mask[i:j].astype(np.float32)[:, :, None]  # (B,S,1)
            summed = (last_hidden * mask).sum(axis=1)                   # (B,H)
            denom = np.clip(mask.sum(axis=1), 1e-9, None)               # (B,1)
            emb = (summed / denom).astype(np.float32)                   # (B,H)

            outputs.append(emb)
            i = j  # advance

        embs = np.vstack(outputs)
        if normalize_embeddings and embs.size:
            embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        return embs
    
        # ------------------------- helpers -------------------------

    def _tokenize(self, texts: List[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
        # tokenizers handles padding/truncation configured in __init__
        encs = self.tok.encode_batch([str(t) if t is not None else "" for t in texts])
        # because we set fixed padding, all sequences should be the same length
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        # attention_mask may or may not be present depending on the tokenizer.json
        if hasattr(encs[0], "attention_mask") and encs[0].attention_mask is not None:
            attn = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        else:
            attn = (ids != self.pad_id).astype(np.int64)
        # safety clamp to max_len (should already be exact)
        if ids.shape[1] != max_len:
            ids = ids[:, :max_len]
            attn = attn[:, :max_len]
        return ids, attn