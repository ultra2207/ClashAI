# Where the Clash Royale agent goes next — a decision report

**Scope.** What learning/search method to invest in next for the `ClashAI` S1 policy, judged against four bodies of
evidence: this repo's own RL + search stack, AlphaZero, the AlphaGo→MuZero lineage, and AlphaStar. Written after
two days of measurement on the local tree; every local number below was produced on this machine (RTX 4080 Laptop,
torch 2.15.0+cu132, `royalesim 0.1.17` / `royalegym 0.1.18`).

**Evidence standard.** Three kinds of claim appear, and they are labelled:
`[M]` measured here, this session · `[C]` cited from the literature (source given) · `[I]` inference from the two.

---

## 0. Executive summary

1. **The bottleneck is not RL compute, it is supervision.** A 6-minute match is 7,200 ticks and **0.105 s of
   simulator time** `[M]`. Policy inference and reward sparsity dominate, not the engine.
2. **This repo's search never became a teacher.** `pipeline/search_s0.py` scores 13 candidate forks per decision and
   picks one; it never writes `(s, π_search, z)` into training. That is AlphaZero without its improvement operator —
   the single highest-leverage missing piece, and the cheapest to add.
3. **The branch-and-prune idea is right in structure and wrong in its pruning criterion.** The repo already measured
   what happens when you prune on elixir: the scorer "chose WAIT on 78% of decisions and did not win more"
   (`pipeline/branch_score.py:3`). Prune on win probability, never on elixir.
4. **Plain PUCT will not work at this action-space size.** 6,624 legal actions per decision `[M]` against AlphaZero's
   35. With 800 sims, PUCT leaves ~88% of root actions unvisited, and visit-count targets are only a policy
   improvement when every plausible root action is visited (Danihelka et al., ICLR 2022 `[C]`). Use
   Gumbel-Top-k + Sequential Halving, which learns from n=2 `[C]`.
5. **A tree is affordable only if you stop paying for restores.** `load_state` costs **2,709 µs = 38 advances**
   `[M]`. A 2-ply 13×13 tree is ~458 ms of restore alone, i.e. 5.7× a whole 12 s-horizon search budget `[M+I]`.
   Depth must be bought with prefix sharing or batched lockstep, not with more forks.
6. **Placement (val 0.233 tile top-1) is the head that caps strength, and it is the head search can least improve
   and PPO cannot improve at all.** Only more human data fixes it; the crawl is proven and cheap.

---

## 1. The local system as measured today

| Quantity | Value | How |
|---|---|---|
| Simulator throughput, both sides interacting | **68,327 ticks/s** | `RoyaleSelfPlayEnv.advance_to`, 10-tick decisions |
| One 6-minute match | 7,200 ticks = **0.105 s** | derived |
| `RustEngine.save_state()` | 25,774 bytes | measured |
| `RustEngine.load_state()` | **2,709 µs** | 200 restores |
| `step([], 10)` | **71 µs** | 200 calls |
| Restore vs advance | **38×** | derived |
| Action space per decision | 8 slots × 2,304 cells = **18,432** | `arena` grid |
| Legal fraction at tick 600, 10 elixir | **6,624 (35.9%)** | `check_deploy` over the full grid |
| Legality scan cost | 1 µs/call → **15 ms** for the full grid | measured |
| S1 model | 1,264,808 params, d=128, 4 layers, 144 patch tokens | `final_icebow_v6aug_s1.json` |
| S1 val (12 epochs, 347 s) | card 0.640 · cell-tile 0.233 · within-1-tile 0.377 · joint 0.174 · gate 0.848 · wait 0.487 · value 0.548 | `[M]` |
| Corpus | 766 replays ≈ **48 player-hours**, 27,083 play decisions, 93,538 rows | `[M]` |
| Decision cadence | 0.5 s (10 ticks); 360 decision slots per match, ~180 with elixir | `rl_royale.yaml` |

Two defects found and fixed this session, both worth carrying forward as a warning about the data path:
the card head was being trained against a hand mask that excluded its own label on 51% of PLAY rows (fixed —
100.00% now), and the `deploying` / `deploying_known` token features are constant 0 across all 710,939 tokens while
the degraded twin forges HP as a constant 1.0 tagged `hp_known=1`.

---

## 2. Track 1 — this repo's own RL and search

### 2.1 The PPO

PPO-clip on a behaviour-cloned init, with a KL leash to that init. `match_loo` advantage: $A_i = R_i-\overline{R}_{j\ne i}$
over one entry's $G=4$ rollouts, $|A|\le 2$, no std normalisation (`rl_royale.yaml:61,64`). Rollout unit is
`E=16 × G=4 = 64 matches/update`. `ppo_epochs 2`, minibatch 256 decisions, `lr 1e-5`, `beta0 0.3` adaptive around
`kl_target 0.10`. Guards are unusually thorough (pro-agreement stops at −3 pp cell/card, tripwire, screen stop, plays/min
bounds, exploit guards on the held-out 174-match screen).

**The arithmetic that matters:** 64 binary rewards per update, and the repo's own screening noise is ~8 pp per seed
(`rl_royale.yaml:127`). A 500-update run therefore resolves only large effects. This is the structural reason the repo's
history is a long list of ±1–7 pp results that did not replicate.

### 2.2 `search_s0` — flat, one-ply, best-of-13

- Fork = `save_state()` once, then a pooled env + `load_state()` per candidate, plus `deepcopy` of both sides' python
  state (`search_s0.py:406-432`).
- Candidates = WAIT + top-4 affordable cards × 3 cells = **13 forks**, each scored by **one** 12 s (240-tick) rollout
  (`search_s0.py:455-467, 1059-1060`). No second ply, no reuse, no backup.
- Lockstep batching: all forks advance in the same rounds so the opponent's decisions are one batched forward
  (`search_s0.py:600-618`).
- Scorer: `+enemy tower damage − own tower damage − spent_elixir×0.061 + Δboard + Δcrowns` (`search_s0.py:53-56`).
- **Result: WAIT on 78% of decisions, no win gain** (`branch_score.py:3`). The successor scorer dropped the elixir term
  entirely.
- **No recorded run artifacts exist** — no `summary.json`, no `matches.jsonl`, no ledger. Treat "search helps/did not
  help by X pp" as *unmeasured in this repo*, with the one-line verdict above as the only surviving evidence.

Cost floor per searched decision `[M+I]`: 13 × 2.709 ms restores = **35.2 ms**, plus 13 × 240 ticks = 3,120 ticks =
**22.1 ms**, so **≥57 ms**, realistically 100–250 ms with the two-side deepcopy and batched forwards.

### 2.3 What is missing between "repo" and "working self-play"

| Prerequisite | State |
|---|---|
| `icebow/data/ghost_pool/pool_env_v1.jsonl` + split | **absent** (`e1_pool.py:132`) — trainer dies at startup |
| generalist init `gen_v1_s0/gen_s0.pt` | **absent**, and `league: true` **refuses an S1 init** (`rl_royale.py:4572`) |
| the league | **off by default** (`rl_royale.yaml:30`) |
| branch workers (`branch_gate`) | off; require league + sampling opponent + non-null temperature |
| S1 init checkpoint | **present** now: `icebow/data/pipeline/s1_icebow_v6aug_s1.pt` |

---

## 3. Track 2 — AlphaZero

**Algorithm.** MCTS with PUCT over $\{N,W,Q,P\}$; leaf evaluated once by $(p,v)=f_\theta$ (no rollouts); backup is a mean
over the subtree; target $\pi=N(\cdot)^{1/\tau}$, `z` = final result from the mover's view; loss $=(z-v)^2-\pi^\top\log p$;
19×256 residual net; Dirichlet root noise $\varepsilon=0.25$.

**Numbers `[C]`.** Chess: 44M self-play games, 800 sims/move, 9 h on 5,000 TPUv1 + 64 TPUv2, 700k steps × 4,096 = 2.87×10⁹
positions, 63k positions/s (vs Stockfish 58M/s — **875× fewer**). Match: **+155 −6 =839/1000 (57.5%)** at 3 h+15 s
against Stockfish 8 with 32 GB hash and syzygy; the disputed preprint match ran Stockfish at 1 GB hash, no tablebases.

**Search is an improvement operator with a measured multiplier.** Raw AlphaGo Zero net (argmax p, no search) rates
**3,055**; with search, **5,185** — **+2,130 Elo**. Lc0's 1,717-Elo net blunders on 8.5% of tablebase positions and 0.62%
with 400 nodes (**13.8× fewer**); its 3,062-Elo net goes 3,075 → 36 (**85×**).

**The compute floor is about n vs |A|, not absolute compute.** Works at 64 sims on 9×9 Hex (3 h on one RTX 2080 Ti);
AlphaZero/MuZero **fail to learn at ≤16 sims on 9×9 Go**; Gumbel-MuZero learns from **n=2**. KataGo bought a 50×
compute reduction with domain improvements and beat ELF OpenGo's final model in 19 days on <30 GPUs.

**Chess is a friendly case for a specific reason:** each decision is individually resolvable by an *exact* search over
~35 actions, so the ~80-ply gap to reward is spanned by search, not by credit assignment. That is precisely the
property a 180-decision Clash Royale episode lacks.

---

## 4. Track 3 — AlphaGo → AlphaGo Zero → MuZero

| | AlphaGo 2016 | AlphaGo Zero 2017 | AlphaZero 2018 | MuZero 2020 |
|---|---|---|---|---|
| Needs | 30M human positions, hand features, rollouts | rules only | rules only | nothing but observations |
| Search | APV-MCTS 5 s/move | 1,600 sims | 800 sims | 800 (board) / 50 (Atari) |
| Volume | 30M self-play positions | 4.9M games | 44M games | 1M minibatches, 20B Atari frames |
| Compute | 48 CPU + 8 GPU | 64 workers + 19 servers | 5,000 TPUv1 + 64 TPUv2 | 1,000 TPU self-play + 16 train |
| Result | 99.8% vs programs | 100–0 vs Lee | 25W/25D/0L vs Stockfish | matches AlphaZero without rules |

**Credit-assignment mechanisms, in order of usefulness here `[C]`:**
1. search-value bootstrap (every node's target comes from search, not raw MC);
2. n-step returns with discounting (Atari: n=10, γ=0.997);
3. recurrent latent unroll (K=5);
4. **Reanalyze** — 80% of policy targets refreshed by re-running MCTS on old trajectories with current weights;
5. EfficientZero's consistency loss + value prefix.

**Sample-efficiency datapoints.** MuZero Reanalyze reaches 731% median HNS on Atari with **200M frames**; EfficientZero
reaches 194.3% mean with **100k frames (2 h) at 1-GPU scale**, vs DQN needing 200M frames. MuZero's own framing: AlphaZero's
21M games would take a human 11,500 years.

**Where search does not help `[C]`:** hard exploration (MuZero scores **0.0** on Montezuma's Revenge — search improves
policies given reachable states, it does not visit them); large/factored actions (Sampled MuZero, K≈5); real-time control
(AlphaStar reached Grandmaster with **no search at execution**); imperfect information (DeepStack belief search, DeepNash
model-free).

**Two warnings that apply directly here:**
- **Value overfitting from correlated targets:** AlphaGo's first value net memorised whole games (0.37 vs 0.19 MSE) until
  trained on **one position per game**. Our 27,083 decisions come from 766 games — the same pathology is guaranteed
  unless sampling is de-correlated.
- **Horizon bias:** this repo measured that horizon Q-values exaggerate differences **~1.4×** vs terminal, and that
  halving the horizon (600→300 ticks) nearly doubled sample SNR (`cf_credit.py:111-130`). Short-horizon counterfactuals
  beat long rollouts — the opposite of `search_s0`'s 12 s forks.

---

## 5. Track 4 — AlphaStar

**The transferable trick is decomposition, not scale.** ~10²⁶ joint actions → a chain of small categorical heads
(function → delay → queue → selected units → target unit → target location), each a few hundred to 65k logits. Our
space is 10⁴–10⁵, so factorisation here buys sample efficiency and masking, not feasibility.

**Ablations `[C]` — the most decision-relevant table in the literature for us:**

| Configuration | Elo |
|---|---|
| RL from scratch, no human data | **149** |
| Supervised only | 936 |
| SL init → RL, no KL | 1,020 |
| + KL to the SL policy | **1,400** (+380) |
| + z-statistics pseudo-rewards | **1,540** (+140) |
| Self-play only | 1,519 Elo but **46%** min win-rate vs its own past |
| PFSP + SP | 1,540 Elo, **71%** min win-rate |
| League exploiters added | Relative Population Performance **6% → 35% → 62%** |
| Value without opponent info | 22% win-rate (82% with) |

Cost: 44 days, 32 TPUv3/agent, 12 actor–learner replicas, ~50,000 agent steps/s per learner, ~900 players.
**At 1/30 compute the faithful recipe collapses** to two strategies (TStarBot-X), and mini-AlphaStar at 4 GPUs / 3 days
scores **0.00 vs a level-7 bot** — while their own shaped, macro-action variant reaches 0.93. At low compute, **reward
shaping and action-space reduction beat faithful cloning.**

Also: **action-rate limits act as a regulariser** — Elo peaks at the imposed cap and falls both at 50% and 200% of it.

---

## 6. Synthesis — what all four tracks agree on

| # | Finding | Evidence |
|---|---|---|
| F1 | Dense per-decision targets beat sparse terminal credit | AZ +2,130 Elo from search targets; AlphaStar +380 from keeping the SL policy as a KL anchor |
| F2 | Human data is not optional | AlphaStar 149 → 1,540 Elo; our placement head is the weakest at 0.233 |
| F3 | Search compensates for a weak prior, gracefully, at ~10²–10³ nodes | Lc0 13.8×–85× blunder reduction |
| F4 | Few simulations + many actions breaks the improvement guarantee | fails ≤16 sims on 9×9 Go; Gumbel-Top-k works at n=2 |
| F5 | Plain self-play forgets; PFSP + one exploiter fixes most of it | 46% → 70–71% minimum win-rate |
| F6 | Short horizons + fresh targets beat long rollouts | Reanalyze; our own 300-tick SNR doubling; 1.4× horizon bias |
| F7 | Model error is amplified by depth | MuZero's Atari plateau at ~100 sims; our sim is *assumed* exact |
| F8 | Low compute ⇒ shaping and coarser actions, not architectural fidelity | mini-AlphaStar 0.00 vs shaped 0.93 |

---

## 7. The "branched multiverse" idea, evaluated

The proposal: from the root state, branch over (exact timing × exact placement × card × order), prune as you go, and
select the line that finishes the defence with the largest elixir advantage.

**Verdict: the structure is exactly MCTS with a perfect simulator — the AlphaZero regime — and this repo already has the
fork machinery. Three things in the sketch would sink it, and each has a fix with a number attached.**

### 7.1 Enumeration is impossible; the branch factor is measured, not estimated

18,432 (slot, cell) pairs, **6,624 legal** at a representative tick `[M]`. Depth 2 = 4.4×10⁷, depth 3 = 2.9×10¹¹.
At 71 µs per 10-tick advance, a million nodes is 71 s single-core — so the search must be *prior-guided*, and the guide
is the policy the repo already has.

### 7.2 Prune on win probability, never on elixir

This is not a matter of taste: the repo shipped the elixir-charged scorer and it "chose WAIT on 78% of decisions and
did not win more" (`branch_score.py:3`). Elixir advantage is a *conservative* quantity; a defence that spends 10 elixir
killing a Golem push and leaves a tower at 5% is better than one that spends 4 and loses the tower. Any prune term must
be $\hat V(s)$ (win prob) or tower-HP-weighted potential, evaluated by the critic.

### 7.3 The economics of branching: restores are the enemy, not ticks

| Component | Cost `[M]` |
|---|---|
| advance 10 ticks | 71 µs |
| restore a fork | 2,709 µs = **38 advances** |
| current `search_s0` per decision (13 forks, 240-tick horizon) | ≥57 ms |
| 2-ply 13×13 tree in restores alone | **458 ms** |
| a 0.5 s decision budget at 71 µs/node | ~7,000 nodes **if you never restore** |

So the reachable design is: **one restore per decision, then expand by advancing**, sharing prefixes, with the tree
grown in lockstep batches. That makes a 10³-node search ≈ 71 ms of engine time, which fits a 0.5 s decision with room
for ~200 batched network evals.

### 7.4 Replace 12 s rollouts with a value bootstrap

`search_s0` scores each candidate with a single 240-tick rollout and, by default, our own side **idles** after the
candidate (`--rollout-self idle`) — so follow-ups never happen and the value is truncated at 12 s. AlphaZero's answer
is the opposite and is 8× cheaper: simulate ~16–32 ticks, then bootstrap with $\hat V$. At 1.26M params a batched forward
is ~1 ms, so 200 leaf evaluations ≈ 0.2 s batched, and the engine cost is ~2,300 ticks ≈ 33 ms. **Total ≈ 0.25 s per
decision for a 200-node search** — feasible at the 0.5 s cadence, per match ~90 s of search.

### 7.5 The published fix for "few sims, many actions"

Gumbel-Top-k sampling + Sequential Halving with completed-Q targets: a *provable* policy improvement at n=2–16, where
plain visit-count PUCT stalls. MiniZero measured 9.3× (n=16) and 16.2× (n=2) cheaper per training step than AlphaZero
at n=200 on Othello. This is not an optimisation; at 6,624 legal actions it is the difference between working and not.

### 7.6 What the sketch is missing that the literature says matters most

The sketch is an *acting* algorithm. Every result in track 2–4 says the payoff comes from turning search into
*targets*: $\pi \leftarrow$ visits, $z \leftarrow$ outcome, then regress the net onto them (policy improvement as a
KL-regularised operator). Search that only picks moves buys you the Lc0 multiplier at test time; search that writes
training rows raises the floor permanently.

---

## 8. The plan

Four phases, each with a kill criterion, ordered by information-per-GPU-hour. Budgets assume one 4080 laptop and the
current sim throughput.

### Phase 0 — Repair the measurement (≈2 h wall, 0 GPU-h, no new science)

Do these first because every later result is uninterpretable without them.

1. **Unblock the RL path**: either build the ghost pool (`pool_env_v1.jsonl` + split) or make the league accept the S1
   init rather than hard-refusing it (`rl_royale.py:4572`); produce a generalist `gen_s0.pt` or explicitly run
   S1-only.
2. **Fix the two dead channels**: `deploying`/`deploying_known` are constant 0 across all tokens (corpus frames carry no
   deployment state — either record it or drop the features so they stop consuming capacity), and the degraded twin's
   HP constant must not be labelled `hp_known=1`.
3. **Re-run `search_s0` with the v2 scorer** over ≥60 paired seeds and arms `plain, force_play, random, never, search`.
   `force_play` vs `search` cleanly separates "play more" from "choose better" — the exact question the 78%-WAIT result
   left open. The cost is well under one PPO run.
   **Kill criterion:** if `search` ≤ `plain` + 2 pp on both crown and tower-HP deltas, the flat 1-ply design is dead and
   Phase 2 must change shape, not parameters.

### Phase 1 — Search as a teacher, no tree (≈1 GPU-day)

For every decision state in the corpus (27,083 plays + a sample of wait rows), run the Phase-0 fork set and store
`(s, π_search, z)` where $z$ is the outcome of the *best* line found (or the frozen critic's value). Train the existing
heads on $\pi$ (cross-entropy on card and cell, KL to the gate) on top of the S1 init, keeping the imitation term.
This is AlphaZero's improvement operator at depth 1, using machinery that already exists.
**Gates:** pro-agreement must not drop >1 pp; ghost-screen must improve ≥3 pp with a CI that excludes 0; `plays/min`
must stay in [0.6×, 1.6×].
**Why first:** it is the only phase that can move placement at all without new data, and it reuses `search_s0` verbatim.

### Phase 2 — Gumbel-Top-k MCTS with a value bootstrap (≈2–3 GPU-days)

Replace flat best-of-13 with a real tree:
- **Action abstraction:** 8 card slots × **96 tiles** (3×3-cell blocks — the model already predicts half-tile cells, so
  aggregation is a sum over the logits) × 3 timing bins, not 2,304 cells.
- **Expansion:** one `save_state` per decision; grow the tree by advancing shared prefixes; never `load_state` in the
  inner loop.
- **Evaluation:** 16–32 tick simulations + $\hat V$ at the leaf, ~200 nodes/decision (measured budget ≈0.25 s).
- **Targets:** Gumbel-Top-k + Sequential Halving with completed-Q; store $(\pi,z)$ for Phase 3.
**Kill criterion:** if 200-node search does not beat 1-ply search by ≥2 pp on the paired sim screen, the value head is
the bottleneck and Phase 3 must start with critic training (`critic_warmup_updates`, `vf_trunk_grad`) before any tree.

### Phase 3 — League, PFSP, KL anchor (≈5–10 GPU-days)

Only after Phase 1–2 have a policy worth anchoring. Config already in the repo, needs turning on:
`league: true`, `league_mix` biased by PFSP $f_\text{hard}(x)=(1-x)^p$, one main exploiter (not three), KL to S1 at
`beta0 0.3`, and the z-statistics pseudo-rewards (deck-cycle edit distance, elixir-spend curve, tower-damage timing —
the AlphaStar analogue of +140 Elo). Keep `league_decks` at 183 decks; it is the cheapest opponent diversity available.
**Kill criterion:** any pro-agreement drop >3 pp, or a ghost-screen gain that reverses on the *pro-agreement* metric — that
signature means the policy is learning the simulator, not the game.

### Phase 4 — Data (CPU-only, wall-clock-bound)

The placement head is 0.233 tile top-1 on 48 player-hours. The crawler already scanned 252,238 sides and kept 766 exact-deck
games; ingest needs no code change. Target 5,000–10,000 games, filter to top-ladder winners (AlphaStar's +9 pp came from
16,000 high-MMR winning games), and keep one position per game for the value head (AlphaGo's memorisation pathology).
**Expected:** every 10× in unique replays should move placement materially; nothing else in this plan does.

### Sequencing and cost

| Phase | GPU-hours | Wall | Answers |
|---|---|---|---|
| 0 | 0 | ~2 h | Does search help at all, at depth 1? |
| 1 | ~8–16 | 1 day | Can search-labelled targets improve placement without new data? |
| 2 | ~24–48 | 2–3 days | Does depth beat breadth at 200 nodes? |
| 3 | ~120 | 5–10 days | Does RL add robustness beyond the imitation prior? |
| 4 | 0 (CPU) | 1–2 days | How much of the ceiling is data? |

Phases 0, 1 and 4 are independent and can run concurrently; Phase 2 depends on 1; Phase 3 depends on 2.

---

## 9. Risks and falsifiers

| Risk | Why it bites here | Falsifier |
|---|---|---|
| Simulator fidelity | search optimises the model; `royalesim` exactness is assumed, not measured, and only ~48h of the real game was ever witnessed by this repo | 100 hand-checked sim-vs-client situations; if >2% of elixir/targeting outcomes differ, every search result is capped and the client-side eval must be the arbiter |
| Value-head quality | AlphaZero's leaf is the net; ours is a 7-class crown-diff head at 0.548 accuracy | if Phase 2's 200-node search loses to 1-ply, this is why |
| Correlated targets | 27,083 decisions from 766 games | one-position-per-game sampling for the value head; watch for 0.37-vs-0.19-style memorisation |
| Elixir-first pruning | already measured to produce 78% WAIT | never ship a scorer with an elixir term |
| League drift / simulator exploits | naive SP = 46% worst-case win-rate | PFSP + one exploiter + pro-agreement as the exploit tripwire |
| Measurement noise | 8 pp SE per seed on the screen | 3 seeds × 58 entries = 174 matches; never report a single-seed delta |

---

## 10. What to build, in one paragraph

Keep the S1 net. Add the missing half of `search_s0`: make it write `(s, π, z)` rows, and re-run it with the v2
scorer, a non-idle self-side, and a short horizon plus value bootstrap. Then, if and only if that moves the screen,
grow it into a Gumbel-Top-k tree over 8 cards × 96 tiles × 3 timing bins with ~200 nodes per decision and one restore
per decision. Turn the league on with PFSP and one exploiter, anchored by a KL to S1. In parallel, crawl 10× more
replays, because placement is the ceiling and only data raises it. Do not rewrite the search in CUDA: the engine is
71 µs per 10 ticks and the network is the bottleneck; do not enumerate the tree; do not prune on elixir.

---

## Appendix — reference numbers

**Local `[M]`**: 68,327 ticks/s · 0.105 s/match · save 25,774 B · load 2,709 µs · 10 ticks 71 µs · 18,432 actions,
6,624 legal · legality scan 15 ms · 13 forks ≥57 ms/decision · 2-ply 13×13 = 458 ms restores · S1 1.26M params,
card 0.640 / cell 0.233 / 1-tile 0.377 / joint 0.174 / gate 0.848 · corpus 766 games, 27,083 plays, 93,538 rows.

**Literature `[C]`**: AlphaZero 44M games / 9 h / 5,000+64 TPUs / 800 sims / 2.87×10⁹ positions / 57.5% vs SF8 /
+2,130 Elo search-over-raw-net / Lc0 13.8×–85× blunder reduction / fails ≤16 sims, Gumbel works at n=2 / Hex 9×9 in
3 h on one 2080 Ti / KataGo 50× at <30 GPUs / AlphaGo 30M positions / AGZ 4.9M games / MuZero 800 & 50 sims, Atari
plateau ~100 sims, Montezuma 0.0 / EfficientZero 100k frames at 1-GPU scale / AlphaStar 10²⁶→10³ logits, ablations
149/936/1020/1400/1540, SP 46% vs PFSP 70–71%, exploiters 6→35→62% RPP, 44 days / 32 TPUv3 / 50k steps/s, mini-AlphaStar
0.00 vs level-7 at 4 GPUs / TStarBot-X collapsed to two strategies at 1/30 compute.

**Sources.** Silver et al., *Nature* 529:484 (2016); *Nature* 550:354 (2017); *Science* 362:1140 (2018,
arXiv:1712.01815); Schrittwieser et al., *Nature* 588:604 (2020, arXiv:1911.08265); Danihelka et al., ICLR 2022;
Hubert et al., ICML 2021; Ye et al., NeurIPS 2021; Wu, arXiv:1902.10565 (KataGo); Jones, "Scaling Scaling Laws with
Board Games" (2021); MiniZero (Wu et al., 2023); Haque, Wei & Müller, ICGA 2021; Vinyals et al., *Nature* 575:350
(2019); Liu et al., JAIR 75:213 (2022, mini-AlphaStar); Han et al., arXiv:2011.13729 (TStarBot-X); Grill et al.,
ICML 2020; Hafner et al., *Nature* 2025 (DreamerV3).

---

## 11. Evidence from the reference agent (added after §7–8 were written)

The owner has watched gameplay of the model built by this repo's author (`vegetableleaf`). Reported, first-hand:

- It reached **rank 1 in the world** in the hands of a player, so its weights are demonstrably superhuman, not a
  promising prototype.
- It acts with **zero delay** and **exact placement**, and it beat a top-1 pro who was playing a hard counter to it.
- The edge is **not placement alone**: timing, decision-making and macro play are all reported as superhuman, above
  any human, including moves no human has performed (e.g. a 0-hit Hog Rider defence with Ice Golem + Skeletons).

### 11.1 What this revises

`§7` argued the branch-and-prune idea was worth pursuing as an inference-time search. The reference agent is direct
evidence **against inference-time search being the path to superhuman play**:

- zero-latency, no dithering, consistent across micro and macro is the signature of a **feed-forward policy at tick
  granularity**, not of a lookahead engine. This repo's own search arm chose WAIT on 78% of decisions
  (`branch_score.py:3`) — search makes an agent more conservative here, not sharper.
- a single strong policy explains superhumanity on *all* axes at once; a bag of tricks does not.

So the ceiling is **not** the architecture family (S1's gate + card + cell heads, 1.26M params). That family reaches
world #1. The gap is in what it is trained on and at what resolution.

### 11.2 Where this pipeline is measurably too coarse

| Axis the reference is superhuman on | This pipeline's resolution | Factor |
|---|---|---|
| Timing (~10 ms reported) | decision cadence `decision_ticks=10` = **0.5 s** (`royale_env.py:195`); WAIT rows sampled every `wait_stride=40` ticks = **2 s** (`dataset.py:161,244`) | **50× too coarse to act, 200× too sparse to learn** |
| Macro play | 766 games; value head = 7-class crown diff at **0.548** accuracy | an outcome head this weak cannot drive elixir economy or tower trades |
| Placement | 2,304 half-tile cells; whether the engine even accepts sub-tile placement is **unresolved** (my probe diffed a stale entity and is void) | open question, not a design input |

The timing row is the actionable one: a policy trained on 2-second samples has **no gradient for "wait 0.2 s more"**,
which is the regime a 10 ms-precision agent operates in. You cannot learn that resolution from data that does not
contain it.

### 11.3 Revised priority (supersedes the §8 table)

| Phase | Change | Why |
|---|---|---|
| 0 · repair measurement | unchanged | still the gate on everything |
| **0b · cadence** (new, ≈2 days) | rebuild the corpus at 1–2 tick decision granularity (or 5 at worst), `y_wait_dt` in ticks; re-run S1 | the one axis known to be 50× wrong, and it is pure dataset work |
| 1 · search as teacher | **keep, but only at the finer cadence**; otherwise it inherits the coarseness | dense per-decision targets remain the best explanation for AlphaZero/AlphaStar gains |
| 2 · inference-time tree search | **demoted** to an optional robustness tool | the reference is superhuman without it; search here is 78%-WAIT |
| 3 · league/PPO | unchanged in shape, but its scoreboard is now "how far is this from a known-superhuman agent", not "does it beat the ghost pool" | |
| 4 · data | **promoted to the critical path** | the architecture already demonstrably reaches #1; scale is the reason ours does not |

### 11.4 The calibration opportunity

If the leaked weights can be obtained, a mirror match against them is the strongest evaluation this project can have:
it converts every abstract metric (card_top1, tile top1, gate accuracy, screen win-rate) into a single distance to a
known-superhuman reference. Worth doing before any further training spend.
