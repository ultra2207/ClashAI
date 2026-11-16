# Experiment budget — 5 minute hard cap

# Out of scope

- **KataCR** (`github.com/vegetableleaf/KataCR`) — deleted 2026-11-15 by the owner's instruction. Do not reference it,
  analyse it, cite it as prior art, or reintroduce its approach. This project is the engine/state branch: observations
  are symbolic entity tokens built from RoyaleSim state (`obs_contract.from_engine` → `to_tokens`), never pixels, and
  no vision pipeline is on the roadmap.

Every experiment, benchmark, sweep or training probe run in this repo is capped at **5 minutes wall time**.

- Set an explicit timeout on the command itself (`timeout: 300` or lower). Never launch an unbounded run.
- If a run is still going at 5 minutes, **kill it and rescope** — do not raise the timeout and retry. The run was
  mis-scoped, not unlucky.
- Before running anything: slice it. One config, one seed, one update, `max_updates=1`, `--limit N`, `screen_every=999`,
  `proagree_every=999`. A sweep of N configs is N experiments, each individually capped — and if the total is going to
  exceed ~5 minutes, reduce the matrix rather than running it serially.
- Prefer bypassing the harness over driving it: if a full run only answers a question that a direct call to the
  function under test answers, call the function.
- Projected vs actual wall time gets reported in the answer whenever an experiment informs a decision.

# Artifacts — where things go

`scratchpad/` is throwaway. Nothing durable, nothing a doc points at, nothing needed to resume.

| Artifact | Location |
|---|---|
| Model weights (S1, generalist, fine-tunes) | `icebow/data/pipeline/*.pt` |
| RL checkpoints (`_uNNNN`, `_latest`, `_snap_uNNNN`) | `icebow/data/bench/rl_royale/<run>/` (`CKPT_ROOT`) |
| Datasets (`.npz` + meta `.json`) | `icebow/data/pipeline/` |
| Driven corpora (`corpus_v3/...`) | `scratchpad/gauntlet/ext/` (regenerable from the crawl) |
| Run bookkeeping (config.yaml, train.log, entries cache) | `scratchpad/gauntlet/L68/rl/<run>/` (`RUN_ROOT`) — reproducible, but the checkpoint is the state |
| Scratch experiments, probes, one-off probes | `scratchpad/` |

Rules: a checkpoint or dataset referenced by a README command, a config key (`init`, `pool`, `proagree_data*`), or a
resume path MUST live under `icebow/data/`, never in `scratchpad/`. Copy it there when a probe produces something
worth keeping, and delete the scratch copy once nothing is reading it.
