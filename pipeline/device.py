"""Device policy: GPU only. There is no CPU fallback anywhere in this repo's training/eval paths.

Every trainer and evaluator resolves its device through `cuda()` (or `resolve()` when a CLI flag is allowed to
name a device). If CUDA is unavailable the call raises instead of silently running 50x slower on the CPU — a silent
fallback is how a "GPU run" quietly becomes an overnight CPU run.
"""
from __future__ import annotations

from typing import Optional

import torch


def cuda(index: int = 0) -> torch.device:
    """The GPU, or a hard error. Use this instead of `"cuda" if available else "cpu"`."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required (GPU-only policy): torch.cuda.is_available() is False. "
            f"torch={torch.__version__}. Install the CUDA wheel (uv add torch --index pytorch-cu132) "
            "and check the NVIDIA driver; CPU execution is disabled on purpose."
        )
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"cuda:{index} requested but only {torch.cuda.device_count()} device(s) visible")
    return torch.device("cuda", index)


def resolve(spec: Optional[str]) -> torch.device:
    """`None`/'auto' -> `cuda()`; 'cuda[:N]' -> that GPU; anything else -> error (CPU is not an option)."""
    s = (spec or "auto").strip().lower()
    if s in ("auto", "gpu", "cuda"):
        return cuda()
    if s.startswith("cuda"):
        _, _, idx = s.partition(":")
        return cuda(int(idx) if idx else 0)
    raise RuntimeError(f"device {spec!r} refused: this project runs GPU-only (use 'cuda', 'cuda:0', or omit it)")
