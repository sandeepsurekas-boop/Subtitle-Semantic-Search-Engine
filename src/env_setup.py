"""Runtime settings to reduce macOS bus errors (PyTorch / Chroma / OpenMP)."""
from __future__ import annotations

import os
import sys


def configure_safe_runtime() -> None:
    """Call before importing torch, chromadb, or sentence_transformers."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    # Avoid MPS on Apple Silicon (can cause bus errors with some torch builds)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if sys.platform == "darwin":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def configure_torch_threads() -> None:
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
