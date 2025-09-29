# onnx_encoder.py — GPU-only encoder with dynamic input mapping (handles token_type_ids/position_ids)
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional
import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def _find_tokenizer_dir(model_dir: Path) -> Path:
    candidates = [model_dir, model_dir / "onnx", *[p for p in model_dir.iterdir() if p.is_dir()]]
    for p in candidates:
        if (p / "tokenizer.json").exists():
            return p
    raise FileNotFoundError(f"tokenizer.json not found under {model_dir}")


def _pick_onnx_path(model_dir: Path, prefer_int8: bool) -> Path:
    search_dirs = [model_dir, model_dir / "onnx"]
    q8, f32 = [], []
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
    raise FileNotFoundError(f"No ONNX model found in {search_dirs} (need model.onnx or model_qint8*.onnx)")


class OnnxSentenceEncoder:
    """
    Sentence embedding encoder (CUDA only) with SBERT-style mean pooling.
    - encode(List[str], batch_size=..., normalize_embeddings=True) -> float32 [N, H]
    - Adaptive GPU batching (halves batch on CUDA errors).
    - Dynamically feeds required inputs: input_ids, attention_mask, token_type_ids, position_ids.
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

        # --- files
        self.onnx_path = _pick_onnx_path(model_path, prefer_int8=prefer_int8)
        tok_dir = _find_tokenizer_dir(model_path)

        # --- tokenizer
        tok_json = tok_dir / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {tok_dir}")
        self.tok = Tokenizer.from_file(str(tok_json))
        self.max_seq_length = int(max_seq_length)
        self.default_batch_size = int(default_batch_size)

        self.pad_id = self.tok.token_to_id("[PAD]") or 0
        self.tok.enable_truncation(max_length=self.max_seq_length)
        self.tok.enable_padding(length=self.max_seq_length, pad_id=self.pad_id, pad_token="[PAD]")

        # --- ORT (CUDA only)
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider not available. Install 'onnxruntime-gpu' and valid CUDA/cuDNN.")
        so = ort.SessionOptions()
        so.enable_mem_pattern = False
        so.log_severity_level = 2
        # If you still hit CUDA error 700, you may also try:
        # so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC

        cuda_opts = {
            "device_id": int(gpu_id),
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": 1,
            "cudnn_conv_algo_search": "DEFAULT",
            "enable_cuda_graph": 1 if enable_cuda_graph else 0,
        }
        self.session = ort.InferenceSession(str(self.onnx_path), so, providers=[("CUDAExecutionProvider", cuda_opts)])

        # --- output dim
        out0 = self.session.get_outputs()[0]
        self.hidden_size = int(out0.shape[-1]) if isinstance(out0.shape[-1], int) else 384

        # --- discover required input names
        self._map_inputs()

    # --------------------------- public API ---------------------------

    def encode(
        self,
        texts,
        *,
        batch_size: int | None = None,
        normalize_embeddings: bool = True,
        # Accept ST-like kwargs (we ignore them; ORT controls the device)
        device=None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        convert_to_tensor: bool = False,
        **kwargs,
    ):
        """
        Encode a list of strings to sentence embeddings (float32) using CUDA.
        Extra kwargs are accepted for API compatibility with SentenceTransformers.
        """
        if not isinstance(texts, (list, tuple)):
            raise TypeError("texts must be a list/tuple of strings")
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.hidden_size), dtype=np.float32)

        if batch_size is None:
            batch_size = self.default_batch_size
        cur_bs = max(1, int(batch_size))

        # Tokenize once
        ids_all, mask_all = self._tokenize(texts, self.max_seq_length)

        # Optional arrays if your _make_feed needs them (depends on your class):
        type_all = np.zeros_like(ids_all, dtype=np.int64) if getattr(self, "need_token_type", False) else None
        pos_all = None
        if getattr(self, "need_position", False):
            pos_row = np.arange(ids_all.shape[1], dtype=np.int64)[None, :]
            pos_all = np.repeat(pos_row, repeats=n, axis=0)

        outputs = []
        i = 0
        while i < n:
            j = min(i + cur_bs, n)

            # Build feed according to discovered input names
            if hasattr(self, "_make_feed"):
                feed = self._make_feed(ids_all[i:j], mask_all[i:j],
                                    None if type_all is None else type_all[i:j],
                                    None if pos_all is None else pos_all[i:j])
            else:
                feed = {
                    "input_ids": ids_all[i:j].astype(np.int64, copy=False),
                    "attention_mask": mask_all[i:j].astype(np.int64, copy=False),
                }

            try:
                (last_hidden,) = self.session.run(None, feed)  # (B,S,H)
            except Exception:
                # Adaptive batching on CUDA errors
                if cur_bs == 1:
                    raise
                cur_bs = max(1, cur_bs // 2)
                continue  # retry same i with smaller batch

            # SBERT mean pooling with attention mask
            mask = mask_all[i:j].astype(np.float32)[:, :, None]   # (B,S,1)
            summed = (last_hidden * mask).sum(axis=1)              # (B,H)
            denom = np.clip(mask.sum(axis=1), 1e-9, None)          # (B,1)
            emb = (summed / denom).astype(np.float32)
            outputs.append(emb)
            i = j

        embs = np.vstack(outputs)
        if normalize_embeddings and embs.size:
            embs /= (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)

        # Respect convert_to_tensor when requested (optional)
        if convert_to_tensor:
            try:
                import torch
                return torch.from_numpy(embs)
            except Exception:
                # fall back to numpy if torch isn't available
                pass

        return embs  # numpy
    # -------------------------- internals ----------------------------

    def _map_inputs(self) -> None:
        """
        Inspect the model inputs and figure out the exact names to feed.
        Handles variants like: attention_mask|input_mask|mask, token_type_ids|segment_ids, position_ids.
        """
        infos = self.session.get_inputs()
        names = [i.name for i in infos]
        lowers = [n.lower() for n in names]

        def pick(candidates: List[str]) -> Optional[str]:
            for cand in candidates:
                cand_l = cand.lower()
                # exact or contains (to survive weird exporter names)
                for n, nl in zip(names, lowers):
                    if nl == cand_l or cand_l in nl:
                        return n
            return None

        self.name_ids = pick(["input_ids", "input", "ids"])
        if not self.name_ids:
            raise RuntimeError(f"Could not find input_ids in model inputs: {names}")

        self.name_mask = pick(["attention_mask", "input_mask", "att_mask", "mask"])
        self.name_type = pick(["token_type_ids", "segment_ids", "token_type"])
        self.name_pos  = pick(["position_ids", "position_id", "pos_ids", "position_ids_0"])

        self.need_mask = self.name_mask is not None
        self.need_token_type = self.name_type is not None
        self.need_position = self.name_pos is not None

    def _make_feed(
        self,
        ids: np.ndarray,
        mask: np.ndarray,
        type_ids: Optional[np.ndarray],
        pos_ids: Optional[np.ndarray],
    ) -> dict:
        feed = {self.name_ids: ids.astype(np.int64, copy=False)}
        if self.need_mask:
            feed[self.name_mask] = mask.astype(np.int64, copy=False)
        if self.need_token_type:
            if type_ids is None:
                type_ids = np.zeros_like(ids, dtype=np.int64)
            feed[self.name_type] = type_ids
        if self.need_position:
            if pos_ids is None:
                pos_row = np.arange(ids.shape[1], dtype=np.int64)[None, :]
                pos_ids = np.repeat(pos_row, repeats=ids.shape[0], axis=0)
            feed[self.name_pos] = pos_ids
        return feed

    def _tokenize(self, texts: List[str], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
        encs = self.tok.encode_batch([str(t) if t is not None else "" for t in texts])
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        # attention_mask may be absent in some tokenizer.json; derive from PAD if needed
        if hasattr(encs[0], "attention_mask") and encs[0].attention_mask is not None:
            attn = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        else:
            attn = (ids != self.pad_id).astype(np.int64)
        # safety clamp
        if ids.shape[1] != max_len:
            ids = ids[:, :max_len]
            attn = attn[:, :max_len]
        return ids, attn
