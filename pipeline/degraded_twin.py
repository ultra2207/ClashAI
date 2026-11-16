"""Degraded-twin S1 dataset: the clean corpus re-observed through the measured live-noise model.

`tools/merge_aug.py` is the partner half of the `v6aug` initialisation named in `rl_royale.yaml:6`; it needs a
*paired* twin — same rows, same targets, same split, only the observations degraded. This module produces that twin
by wrapping `dataset.from_engine` (the single seam where an observation becomes a BoardState) with
`e1_view.live_view`, so every row is built exactly as the clean build builds it and only `tok`/`sc`/`past` differ.

Noise is seeded per (tag, side, tick) so the twin is reproducible and independent of file order.

Usage:
    uv run python -m pipeline.degraded_twin icebow --corpus <corpus dir> --out icebow/data/pipeline/s1_dataset_degraded.npz
    python tools/merge_aug.py --clean s1_dataset.npz --degraded s1_dataset_degraded.npz --out s1_dataset_v6aug.npz
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Optional

import numpy as np


def build(deck_name: str, corpus: Path, out: Path, *, seed: int = 0, wait_stride: int = 40,
          play_window: int = 20, val_pct: int = 15, limit: int = 0, log=None) -> dict:
    from pipeline import dataset, e1_view

    deck_ref = dataset.load_deck(deck_name)
    original = dataset.from_engine

    def degraded(obs, my_side, deck, **kw):
        bs = original(obs, my_side, deck, **kw)
        tag = int(zlib.crc32(f"{obs.get('tick')}|{my_side}|{seed}".encode()) & 0xFFFFFFFF)
        return e1_view.live_view(bs, np.random.default_rng(tag), deck)

    dataset.from_engine = degraded  # the one seam: build_replay calls this per frame
    try:
        return dataset.build(deck_name, corpus=corpus, out=out, wait_stride=wait_stride,
                             play_window=play_window, val_pct=val_pct, limit=limit, log=log)
    finally:
        dataset.from_engine = original


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck", nargs="?", default="icebow")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    stats = build(a.deck, a.corpus, a.out, seed=a.seed, limit=a.limit)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
