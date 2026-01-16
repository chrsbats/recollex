"""
Thin wrapper around sentence_transformers.SparseEncoder (SPLADE) with ONNX backend.
"""
from __future__ import annotations

from typing import List, Tuple, Sequence, Dict, Optional, Any
from pathlib import Path
import threading
import os
from importlib import metadata as importlib_metadata
try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None
from huggingface_hub import snapshot_download

try:
    from sentence_transformers import SparseEncoder as STSparseEncoder  # type: ignore
except Exception:  # pragma: no cover
    STSparseEncoder = None  # type: ignore

_ENC_CACHE: Dict[Tuple[str, str, str, Tuple[Tuple[str, Any], ...]], Any] = {}
_CACHE_LOCK = threading.Lock()

def _get_env_precision() -> Optional[str]:
    val = os.getenv("RECOLLEX_ONNX_PRECISION")
    if val:
        val = val.strip().lower()
        if val in {"int8", "fp16", "fp32"}:
            return val
    return None

def _installed_ort_flavor() -> Optional[str]:
    names = [
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "onnxruntime-rocm",
        "onnxruntime-silicon",
        "onnxruntime",
    ]
    for name in names:
        try:
            importlib_metadata.version(name)
            return name
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None

def _has_accel_provider() -> bool:
    if ort is None:
        return False
    try:
        providers = set(ort.get_available_providers())
    except Exception:
        return False
    accel = {
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "ROCMExecutionProvider",
        "CoreMLExecutionProvider",
    }
    return bool(providers.intersection(accel))

def _default_providers() -> list[str]:
    if ort is None:
        return ["CPUExecutionProvider"]
    try:
        provs = set(ort.get_available_providers() or [])
    except Exception:
        return ["CPUExecutionProvider"]
    ordered = []
    for p in ("CUDAExecutionProvider","ROCMExecutionProvider","DmlExecutionProvider","CoreMLExecutionProvider","CPUExecutionProvider"):
        if p in provs:
            ordered.append(p)
    return ordered or ["CPUExecutionProvider"]

def _choose_onnx_precision() -> str:
    # 1) Env override
    envp = _get_env_precision()
    if envp:
        return envp
    # 2) Wheel detection
    flavor = _installed_ort_flavor()
    if flavor and flavor != "onnxruntime":
        return "fp16"
    # 3) Providers
    if _has_accel_provider():
        return "fp16"
    # 4) Fallback
    return "int8"

class SpladeEncoder:
    def __init__(
        self,
        model: str = "seerware/Splade_PP_en_v2",
        backend: str = "onnx",
        device: str = "cpu",
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Ensure local cache directory
        models_dir = Path("./models").resolve()
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(model)
        if model_path.exists():
            # If a file (e.g. /.../model.onnx) is supplied, use its parent as the model folder.
            local_path = model_path.parent.resolve() if model_path.is_file() else model_path.resolve()
        else:
            local_path = models_dir / model.replace("/", "__")

        # Prefetch depending on backend; for ONNX, select precision unless provided
        mk: Optional[Dict[str, Any]] = model_kwargs
        if backend == "onnx":
            if mk is None:
                precision = _choose_onnx_precision()
                mk = {"subfolder": f"onnx/{precision}", "file_name": "model.onnx"}
            required = local_path / mk.get("subfolder", "onnx/int8") / mk.get("file_name", "model.onnx")
            if not required.exists():
                prefetch(model, str(models_dir), backend=backend, model_kwargs=mk)  # pragma: no cover
        else:
            if not local_path.exists():
                prefetch(model, str(models_dir), backend=backend, model_kwargs=model_kwargs)  # pragma: no cover

        # Cache key should include model_kwargs to avoid collisions across variants
        mk_items: Tuple[Tuple[str, Any], ...] = tuple(sorted((mk or {}).items())) if backend == "onnx" else tuple(sorted((model_kwargs or {}).items()))
        key = (str(local_path), str(backend), str(device), mk_items)
        from recollex.encoder.stonnx import SparseEncoderONNX
        with _CACHE_LOCK:
            se = _ENC_CACHE.get(key)
            if se is None:
                if backend == "onnx":
                    if ort is None:
                        raise RuntimeError("onnxruntime is not installed. Pick an ORT: install 'recollex' for CPU, or one of 'recollex-gpu[cuda]', 'recollex-gpu[rocm]', 'recollex-gpu[directml]', 'recollex-gpu[silicon]' for accelerators.")
                    # Use our ONNXRuntime-based encoder by default
                    subfolder = (mk or {}).get("subfolder", "onnx/int8")
                    se = SparseEncoderONNX(
                        model_dir=str(local_path / subfolder),
                        tokenizer_id=str(local_path),
                        providers=_default_providers(),
                        max_active_dims=350,
                    )
                else:
                    if STSparseEncoder is None:  # pragma: no cover
                        raise RuntimeError("sentence_transformers is required for backend != 'onnx'")
                    se = STSparseEncoder(
                        str(local_path),
                        backend=backend,
                        device=device,
                        model_kwargs=model_kwargs,
                    )
                _ENC_CACHE[key] = se
        self._se = se
        # Determine dims robustly
        self._device = device
        dims = None
        tok = getattr(se, "tokenizer", None)
        if tok is not None:
            dims = getattr(tok, "vocab_size", None)
        if dims is None:
            try:
                mat = self._se.encode([""], device=self._device)
            except TypeError:  # pragma: no cover
                mat = self._se.encode([""])
            try:
                # SciPy CSR
                dims = int(mat.shape[1])  # type: ignore[attr-defined]
            except Exception:
                try:
                    # Torch sparse
                    dims = int(mat.size(1))  # type: ignore[attr-defined]
                except Exception:
                    dims = 0
        self._dims = int(dims)
        try:
            # sentence_transformers models typically support .to(device)
            self._se.to(self._device)
        except Exception:  # pragma: no cover
            # Some versions route device only via encode(..., device=...)
            pass

    @property
    def dims(self) -> int:
        return int(self._dims)

    @property
    def sparsify_topk(self) -> Optional[int]:
        se = self._se
        for name in ("max_active_dims", "top_k", "topk", "max_tokens", "logits_to_sparse_topk"):
            val = getattr(se, name, None)
            if isinstance(val, int):
                return int(val)
        return None

    @property
    def sparsify_threshold(self) -> Optional[float]:
        se = self._se
        for name in ("threshold", "logits_threshold", "min_weight", "min_term_weight", "thr"):
            val = getattr(se, name, None)
            if isinstance(val, (int, float)):
                return float(val)
        return None

    def _row_to_indices_values(self, row) -> Tuple[List[int], List[float]]:
        # ONNX wrapper returns rows as a (indices, values) pair
        if isinstance(row, (tuple, list)) and len(row) == 2:
            idx, vals = row
            return list(idx), list(vals)
        # torch COO sparse
        if hasattr(row, "coalesce"):
            row = row.coalesce()
            indices = row.indices().tolist()[0]
            values = row.values().tolist()
            return indices, values
        # scipy CSR
        if hasattr(row, "indices") and hasattr(row, "data"):
            indices = row.indices.tolist() if hasattr(row.indices, "tolist") else list(row.indices)
            data = row.data
            values = data.tolist() if hasattr(data, "tolist") else list(data)
            return indices, values
        # Fallback dense
        try:
            import numpy as np  # pragma: no cover
            if hasattr(row, "A1"):
                arr = row.A1
            else:
                arr = np.asarray(row).ravel()
            nz = np.nonzero(arr)[0].tolist()
            vals = arr[nz].astype(float).tolist()
            return nz, vals
        except Exception:  # pragma: no cover
            return [], []

    def encode(self, text: str) -> Tuple[List[int], List[float]]:
        # Use available encoding method; returns 1xD sparse row
        try:
            mat = self._se.encode([text], device=self._device)
        except TypeError:  # pragma: no cover
            mat = self._se.encode([text])

        # SciPy CSR vs torch sparse
        if hasattr(mat, "getrow"):
            row = mat.getrow(0)
        else:
            row = mat[0]
        return self._row_to_indices_values(row)

    def encode_many(self, texts: Sequence[str]) -> List[Tuple[List[int], List[float]]]:
        try:
            mats = self._se.encode(texts, device=self._device)
        except TypeError:  # pragma: no cover
            mats = self._se.encode(texts)
        out: List[Tuple[List[int], List[float]]] = []
        if hasattr(mats, "getrow"):  # SciPy CSR
            for i in range(mats.shape[0]):
                row = mats.getrow(i)
                out.append(self._row_to_indices_values(row))
        else:
            try:
                mats = mats.coalesce()
            except Exception:
                pass
            for row in mats:
                out.append(self._row_to_indices_values(row))
        return out


def prefetch(
    model: str = "seerware/Splade_PP_en_v2",
    models_dir: str = "./models",
    *,
    backend: str = "onnx",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Download the model into models_dir if missing. Returns the local path.
    For backend=='onnx', only fetch the chosen precision subtree and tokenizer/config.
    """
    base = Path(models_dir).resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = base / model.replace("/", "__")
    preexisting = target.exists()
    target.mkdir(parents=True, exist_ok=True)

    cache_dir = base / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = None
    local_dir_use_symlinks = True
    required_path: Optional[Path] = None

    if backend == "onnx":
        if model_kwargs is None:
            precision = _choose_onnx_precision()
            kw = {"subfolder": f"onnx/{precision}", "file_name": "model.onnx"}
        else:
            kw = model_kwargs
        subfolder = kw.get("subfolder", "onnx/int8")
        file_name = kw.get("file_name", "model.onnx")
        required_path = target / subfolder / file_name
        allow_patterns = [
            f"{subfolder}/*",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "merges.txt",
            "config.json",
        ]
        local_dir_use_symlinks = False
        if required_path.exists():
            return str(target)
        # If model dir already existed before this call, skip download (unit-test friendly)
        if preexisting:
            return str(target)

    if required_path is None and target.exists():
        return str(target)

    try:
        try:
            snapshot_download(
                repo_id=model,
                local_dir=str(target),
                local_dir_use_symlinks=local_dir_use_symlinks,
                cache_dir=str(cache_dir),
                allow_patterns=allow_patterns,
            )
        except TypeError:
            snapshot_download(
                repo_id=model,
                local_dir=str(target),
                local_dir_use_symlinks=local_dir_use_symlinks,
            )
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to download model '{model}' into '{target}': {e}") from e
    return str(target)


def prefetch_main(argv=None):
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="seerware/Splade_PP_en_v2")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--backend", default="onnx")
    p.add_argument("--quant", choices=["int8", "fp16", "fp32"])
    args = p.parse_args(argv)
    mk = {"subfolder": f"onnx/{args.quant}", "file_name": "model.onnx"} if (args.backend == "onnx" and args.quant) else None
    path = prefetch(model=args.model, models_dir=args.models_dir, backend=args.backend, model_kwargs=mk)
    print(path)
    return 0


def clean_main(argv=None):
    import argparse
    import shutil
    import sys
    from pathlib import Path

    def _is_subpath(child: Path, parent: Path) -> bool:
        try:
            return child.resolve().is_relative_to(parent.resolve())
        except AttributeError:
            try:
                child.resolve().relative_to(parent.resolve())
                return True
            except Exception:
                return False

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="seerware/Splade_PP_en_v2")
    p.add_argument("--models-dir", default="./models")
    p.add_argument("--backend", default="onnx")
    p.add_argument("--quant", choices=["int8", "fp16", "fp32"])
    p.add_argument("--dry", action="store_true", help="Dry run: print paths that would be removed")
    args = p.parse_args(argv)

    base = Path(args.models_dir).resolve()
    # Refuse to operate on filesystem root
    if base == Path(base.anchor):
        print(f"Refusing to operate on root models_dir: {base}", file=sys.stderr)
        return 2

    target = base / args.model.replace("/", "__")
    paths_to_remove = []

    if args.quant and args.backend == "onnx":
        to_remove = target / "onnx" / args.quant
        if not _is_subpath(to_remove, base):
            print("Refusing: quant path not within models_dir", file=sys.stderr)
            return 2
        paths_to_remove.append(to_remove)
    else:
        if not _is_subpath(target, base):
            print("Refusing: target not within models_dir", file=sys.stderr)
            return 2
        paths_to_remove.append(target)
        cache = base / ".hf_cache"
        if _is_subpath(cache, base):
            paths_to_remove.append(cache)

    if args.dry:
        for pth in paths_to_remove:
            print(f"DRY: would remove {pth}")
        return 0

    for pth in paths_to_remove:
        shutil.rmtree(pth, ignore_errors=True)
        print(str(pth))
    return 0
