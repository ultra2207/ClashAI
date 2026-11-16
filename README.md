# ClashAI

A research project that trains a Clash Royale agent for the X-Bow control deck ("icebow") in a fast
simulator. It starts from imitation learning on professional replays and improves with self-play
reinforcement learning. Not affiliated with or endorsed by Supercell.

Results are in [RESULTS.md](RESULTS.md).

## What is here

| Path | What it is |
| :--- | :--------- |
| `pipeline/` | Observation contract, imitation dataset and model, PPO trainer (`rl_royale.py`), evaluation (`e1_eval.py`), scoring and gates. `human_play.py` is the playable window against a checkpoint. |
| `icebow/`, `hogeq/` | Deck configuration, card data and the small `clashrl` helper package. |
| `tools/` | Public replay download and dataset helpers. |

## Simulator

The code in this repository trains and evaluates on the public [RoyaleSim](https://pypi.org/project/royalesim/) package.
Some models in RESULTS.md were trained with other engines, which are not part of this repository.

In this code the observation holds only what a player can see. The opponent's hand, next card and elixir are never
fed to a model; opponent elixir is estimated from public events (`pipeline/opp_elixir_count.py`).

## Running

These steps have not been tested on a clean install.

```
uv sync                          # .venv with the dependencies in pyproject.toml / uv.lock

# reinforcement learning on the simulator: --smoke runs a tiny check, key=value overrides pipeline/rl_royale.yaml
uv run python -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run smoke --smoke

# paired evaluation of a checkpoint
uv run python -m pipeline.e1_eval --help
```

`ROYALE_RUNTIME` names a *pinned* engine build under `research/ext/`. Unset, a checkout without one runs the
RoyaleSim/RoyaleGym wheels in `.venv` — the same thing `ROYALE_RUNTIME=unpinned` selects — and stamps every run with
runtime id `unpinned-linux`.

Checkpoints and datasets are not in this repository, so training needs an initial checkpoint and a replay pool of your own
(see the `init` and `pool` keys in `pipeline/rl_royale.yaml`). `tools/` downloads a public replay set.

### Resuming an interrupted run

Both trainers survive a cut; neither restarts from zero.

| Trainer | Flag | What is restored |
|---|---|---|
| generalist pretraining (`pipeline.train_gen`) | `--resume <ckpt>` | model, optimizer, LR schedule, epoch counter, per-epoch history, row-order RNG. Rewrites `gen_<tag>_latest.pt` **every epoch** (best val still goes to `gen_<tag>.pt`), so a cut costs at most one epoch. Refuses a checkpoint whose `card_vocab`/`d_c`/`d`/`layers`/`grid` differ; if `--epochs` changed it says so and restarts the LR schedule over the remaining epochs. Already-finished runs exit 0 with `ALREADY_COMPLETE`. |
| RL (`pipeline.rl_royale`) | `--resume` (same `--run` name) | update counter, beta, optimizer + moments, RNG, visit counts, guards, league pool/snapshots, from `<run>_latest.pt` (written every update). `max_updates` may be raised: it logs the config diff and continues. |

```
# pretraining: continue the 6-epoch generalist run after a cut
uv run python -m pipeline.train_gen --data icebow/data/pipeline/gen_dataset_v1.npz \
  --out-dir icebow/data/pipeline --tag gen_v1 --seed 0 --epochs 6 --grid lattice --deck-weighting sqrt \
  --resume icebow/data/pipeline/gen_gen_v1_s0_latest.pt

# RL: continue the run named gen_rl_v1 from its last update
uv run python -m pipeline.rl_royale --config pipeline/rl_royale.yaml --run gen_rl_v1 --resume \
  init=... pool=... league=true max_updates=100
```

## Playing against a checkpoint

`pipeline/human_play.py` opens a window with a real-time RoyaleSim match: you play one side, the checkpoint plays the
other. The bot decides on the same code path as `e1_eval` (true board, `p_gate > tau`, `TAU_LIVE = 0.27`), so what you
see is the evaluated policy, not a separate "demo" rule.

Both checkpoint kinds work: an S1 specialist and a generalist / RL checkpoint (`gen` flag) — the bot seat dispatches
on it. That includes a run that is still training, so you can play each new update:

```
uv run python -m pipeline.human_play --ckpt icebow/data/bench/rl_royale/gen_rl_v1/gen_rl_v1_latest.pt   # live RL run
uv run python -m pipeline.human_play --ckpt icebow/data/pipeline/gen_v1_s0.pt                          # pretrained generalist
```

```
export ROYALE_RUNTIME=unpinned   # PowerShell: $env:ROYALE_RUNTIME='unpinned'
uv run python -m pipeline.human_play --ckpt icebow/data/pipeline/s1_icebow_v6aug_s1.pt   # the S1 icebow specialist
uv run python -m pipeline.human_play --ckpt <run>/gen_s0.pt                             # a GENERALIST checkpoint
```

`--ckpt` is either kind, chosen by the file itself (`e1_eval.load_policy` dispatches on the checkpoint's `gen` key):

* **S1** (`s1_icebow_v6aug_s1.pt`, the default when present): the icebow slot observation + one `model_forward` per
  decision; it plays the human's deck unless `--bot-deck` names another.
* **Generalist** (`{"gen": True}`, e.g. `gen_s0.pt`): seated on `e1_eval.SelfPlaySide`, the same seat `rl_royale`
  plays generalist policies with, so its input row (v3 sim features / opponent context / cue / target / memory rows
  when the checkpoint declares them) is built by `e1_eval`. A generalist is deck-agnostic, so with no `--bot-deck` it
  plays a deck drawn from the league census under `pipeline/rl_royale.yaml`'s own rule (`league_decks` weighted by
  `league_deck_alpha`/`league_deck_floor`, icebow at `league_icebow_share`), restricted to decks the checkpoint's
  `card_vocab` knows. A timing-head generalist (slot hazard, no gate) is refused with a message.

Other flags: `--bot none` (idle opponent, no checkpoint), `--bot-deck <yaml>` (name the bot's deck explicitly),
`--side 0|1`, `--seed`, `--tau`, `--tick-rate` (20 = real time), `--device`, `--deck` (the human's).

| Control | Action |
| :--- | :--- |
| `1` `2` `3` `4` | select hand slot (the sidebar outlines the selection) |
| left click | deploy the selected card at the cursor; the crosshair preview is green for a legal point and red with the engine's refusal code otherwise |
| `space` | pause / resume |
| `r` | new match (new deal, same checkpoint) -- also on the result banner |
| `n` | card names under the bodies on/off (default on) |
| `esc` | quit |
| right click | clear the cursor preview |

The board has no sprites: a unit is a coloured circle, and the name under it is what identifies it.

The sidebar shows crowns, tower HP, elixir, your hand and costs, your next card, the bot's decision/play counts and
gate range, and a `DEPLOYS` timeline of every accepted deploy by either side. The match ends on the engine's result
(an early three-crown finish, or the final whistle at 3:00 and then overtime) as `WIN` / `LOSS` / `DRAW` with crowns,
and `r` starts the next one.

Headless self-check (scripted human, assertions on progress and on the policy deploying; exit 2 = failed):

```
uv run python -m pipeline.human_play --headless 200 --save-frame frame.png
uv run python -m pipeline.human_play --headless 200 --ckpt <run>/gen_s0.pt --save-frame frame.png
```

