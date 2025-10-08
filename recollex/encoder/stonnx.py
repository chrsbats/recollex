# splade_encoder_onnx.py
from __future__ import annotations
from pathlib import Path
from typing import List, Sequence, Tuple, Optional, Union
import numpy as np
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None
from transformers import AutoTokenizer

class SparseEncoderONNX:
    """
    Minimal ST-compatible sparse encoder on ONNX Runtime.
    Exposes .encode(...) returning CSR rows as (indices, values).
    """

    def __init__(
        self,
        model_dir: str,
        providers: Optional[list[Union[str, tuple[str, dict]]]] = None,
        tokenizer_id: Optional[str] = None,
        max_active_dims: Optional[int] = 350,   # ST parity: cap non-zero terms
    ):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Pick an ORT: install 'recollex[cpu]' for CPU, or one of 'recollex[gpu]', 'recollex[rocm]', 'recollex[directml]', 'recollex[openvino]', 'recollex[silicon]' for accelerators.")
        self.model_dir = str(model_dir)
        self.onnx_path = (Path(model_dir) / "model.onnx") if not str(model_dir).endswith(".onnx") else Path(model_dir)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {self.onnx_path}")

        provs = providers or ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(self.onnx_path), providers=provs)

        tok_id = tokenizer_id or model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tok_id, use_fast=True)
        self.vocab_size = int(getattr(self.tokenizer, "vocab_size", 0))

        self.max_active_dims = max_active_dims

        self._input_names = {i.name for i in self.sess.get_inputs()}
        self._output_name = self.sess.get_outputs()[0].name

    def to(self, device: str):  
        #TODO: no-op for now
        return self

    def eval(self):  # no-op for compatibility
        return self

    @property
    def dims(self) -> int:
        return int(self.vocab_size)

    def encode(self, texts: Sequence[str] | str, device: Optional[str] = None, batch_size: int = 32):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="np",
                return_token_type_ids=True,
            )
            ids = toks["input_ids"].astype(np.int64)

            # Build attention mask; match torch reference: use attention_mask only
            am_np = toks.get("attention_mask")
            if am_np is None:
                am_np = np.ones_like(ids, dtype=np.int64)

            am3 = am_np.astype(np.float32)[:, :, None]    # (B, T, 1)

            ort_inputs = {}
            if "input_ids" in self._input_names:
                ort_inputs["input_ids"] = ids
            if "attention_mask" in self._input_names:
                ort_inputs["attention_mask"] = am_np.astype(np.int64)
            if "token_type_ids" in self._input_names:
                tt = toks.get("token_type_ids")
                if tt is None:
                    tt = np.zeros_like(ids, dtype=np.int64)
                ort_inputs["token_type_ids"] = tt.astype(np.int64)
            if "position_ids" in self._input_names:
                seq_len = int(ids.shape[1])
                pos = np.arange(seq_len, dtype=np.int64)[None, :]
                pos = np.repeat(pos, ids.shape[0], axis=0)
                ort_inputs["position_ids"] = pos
            # Run model → logits over vocab (B, T, V) or (B, V) depending on export
            outputs = self.sess.run(None, ort_inputs)
            logits = outputs[0]

            # If (B, T, V): pool per token
            if logits.ndim == 3:
                # Torch reference: vector = max(log1p(ReLU(logits)) * attention_mask[:, :, None], dim=1)
                pos = np.maximum(logits, 0.0)            # (B, T, V)
                x = np.log1p(pos) * am3                  # (B, T, V)
                logits = x.max(axis=1)               # (B, V)
            elif logits.ndim == 2:
                logits = np.maximum(logits, 0.0)         # (B, V)
            else:
                raise RuntimeError(f"Unexpected output shape {logits.shape}")

            # Convert each row to sparse indices/values
            for vrow in logits:
                mad = self.max_active_dims
                if mad is not None:
                    # cap to top-k among strictly positive weights
                    pos_mask = vrow > 0.0
                    pos_idx = np.nonzero(pos_mask)[0]
                    if pos_idx.size == 0:
                        idx = np.empty((0,), dtype=np.int32)
                        vals = np.empty((0,), dtype=np.float32)
                    elif pos_idx.size <= mad:
                        idx = pos_idx.astype(np.int32)
                        vals = vrow[idx].astype(np.float32)
                    else:
                        pv = vrow[pos_idx]
                        k = int(mad)
                        keep_local = np.argpartition(pv, -k)[-k:]
                        idx = pos_idx[keep_local].astype(np.int32)
                        vals = pv[keep_local].astype(np.float32)
                    order = np.argsort(idx)
                    idx, vals = idx[order], vals[order]
                else:
                    mask = vrow > 0.0
                    idx = np.nonzero(mask)[0].astype(np.int32)
                    vals = vrow[idx].astype(np.float32)
                    order = np.argsort(idx)
                    idx, vals = idx[order], vals[order]

                rows.append((idx.tolist(), vals.tolist()))
        # Return list-of-rows; caller can wrap to CSR if needed
        return rows
