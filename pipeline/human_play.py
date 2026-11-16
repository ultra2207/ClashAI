"""A Pygame window in which a human plays one real-time RoyaleSim match against a trained policy.

    uv run python -m pipeline.human_play                 # window, human = side 0, bot = the S1 checkpoint
    uv run python -m pipeline.human_play --bot none      # window, opponent idle (no checkpoint needed)
    uv run python -m pipeline.human_play --ckpt PATH      # window against ANOTHER checkpoint: an S1 .pt, or a
                                                         # GENERALIST gen_s0.pt -- the seat follows the checkpoint
    uv run python -m pipeline.human_play --headless 400 --save-frame frame.png

  TWO CHECKPOINT KINDS, SELECTED BY THE FILE (``e1_eval.load_policy`` dispatches on its ``gen`` key):

  * S1 (``policy_io.load_model``): ``PolicyBot``, the icebow-specialist path -- the S1 slot observation, one
    ``e1_eval.model_forward`` + ``live_decide`` per decision. Its deck is the human's unless ``--bot-deck`` names one.
  * GENERALIST (``GenPolicy``, i.e. a ``{"gen": True}`` checkpoint such as ``gen_s0.pt``): ``GenBot``, which seats
    ``e1_eval.SelfPlaySide`` -- the same seat ``rl_royale`` plays generalist policies with, so the gen input row
    (``GenPolicy.row`` plus v3 sim features / opponent context / cue / target / memory rows the checkpoint declares)
    is built by ``e1_eval`` and never re-implemented here. A generalist is deck-agnostic: with no ``--bot-deck`` it
    plays a deck drawn from the LEAGUE census (``rl_royale.yaml``'s ``league_decks`` / ``league_deck_alpha`` /
    ``league_deck_floor`` / ``league_icebow_share``), keeping only decks its card_vocab knows. A timing-head
    generalist (slot hazard, no gate) is refused: the live loop is a gate decide.

CONTROLS
  1 2 3 4    select hand slot (left to right in the sidebar; the selected slot is outlined)
  click      deploy the selected card at the cursor: left button, anywhere on the board. The crosshair preview
             shows the engine's verdict BEFORE the click (green = OK, red = the refusal's name)
  space      pause / resume        r    a NEW match (new deal, same checkpoint; also works on the end banner)
  n          card names under the bodies on/off (default on)
  esc        quit                  right button: clear the cursor preview

  The sidebar carries everything needed to judge the policy by eye: both crown counts and tower HP, your elixir
  pips and X/10, your hand with costs (dim = unaffordable), your next card, the bot's decision/play counters and
  its gate-probability range, and DEPLOYS: one line per ACCEPTED deploy of either side, in engine time (the
  opponent's cards appear there the moment they land, and their bodies are drawn on the board in red). A
  refused click never changes the board; the engine's own status is shown as the transient red message.
  The match ends on the engine's own result -- an early three-crown finish, or the final whistle (3:00, then
  overtime if the crowns are level) -- shown as WIN / LOSS / DRAW with both crown counts; ``r`` starts the next one.

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT
  ``royale_env.RoyaleSelfPlayEnv`` runs the battle: ANY two 8-card decks, no ghost, a seeded deal, the 90-tick
  warm-up already advanced. This module owns only presentation, wall-clock pacing and the human's input; it
  never re-implements the engine and never re-implements inference. The bot seat decides on ``e1_eval``'s own
  path, unchanged:

      policy_io.load_model / e1_eval.load_policy   -> the checkpoint
      obs_contract.from_engine(compact_raw(raw))   -> the board in the bot's own frame      [S1]
      obs_contract.to_tokens + dataset._past       -> (tok, mask, sc, past)                 [S1]
      e1_eval.model_forward                        -> (enc, heads, p_gate, hand mask)       [S1]
      e1_eval.allowed_slots                        -> affordable hand slots                 [S1]
      e1_eval.live_decide                          -> slot + cell, or WAIT                  [S1]
      policy_io.cell_to_engine                     -> engine coordinates                    [S1]

      e1_eval.SelfPlaySide.prepare/gen_row/decide_row -> the generalist's observation + decide [gen]
      GenPolicy.forward_batch                      -> (enc, heads, p_gate, hand mask)        [gen]

  Whichever kind the checkpoint is, only the ACTION is this module's: ``LiveMatch.deploy``, one placement path for
  both seats (``check_deploy`` first, the engine's own refusal codes, the same deploy timeline).

  The observation is the TRUE board (``e1_eval --obs clean``): ``e1_view.live_view``'s deployment noise exists
  to simulate a human's imperfect reads, and here the imperfect human is actually sitting at the keyboard.
  The decision rule is the eval rule (play iff p_gate > tau, ``e1_eval.TAU_LIVE``); anti-stall is left off
  because ``e1_eval`` leaves it off by default (``STALL_ELIXIR_LIVE`` is None).

COORDINATE FRAMES -- all conversion happens in ``ArenaView`` and nowhere else
  * window pixels: the board occupies ``tiles_x * TILE_PX`` by ``tiles_y * TILE_PX`` at the window's top left.
  * tiles: ``royalegym``'s 18 x 32 arena.
  * engine units: ``DeployCommand.x/y`` and ``EntityState.x/y`` are the engine's own subtile frame, i.e.
    ``Arena.subtile`` = 18000 units per tile (arena 324000 x 576000). ``royale_env.raw()`` divides by
    ``royale_env.SCALE`` for the 1000-per-tile observation frame the policy consumes -- that frame stops here.
  Placement is resolved by the ENGINE, not by this client: an accepted play may be snapped to a tile centre or
  moved off an own body, so the drawn body is where it actually landed, never where the click was. Refusals are
  the engine's own ``check_deploy`` status codes, asked BEFORE the engine is touched, so a click can never be a
  silent no-op: the refused status is shown on the board and in the sidebar.

DISPLAY
  The board is drawn in the engine's frame with one DISPLAY-ONLY flip about the river, so the human always
  plays from the bottom of the window whichever side ``--side`` names (the engine puts team 0 at low y). Every
  pixel mapping applies the same flip, so a click still lands where the cursor is; the bot's coordinates are
  engine coordinates and never pass through the flip.

  Every body is a coloured circle with its CARD NAME written underneath, taken from the engine catalogue's own
  ``card_id`` row (``label_of`` spaces the CamelCase: ``HogRider`` -> ``Hog Rider``). This branch has no sprite
  set -- a blob is the whole unit -- so the name is what tells a Musketeer from an Ice Wizard. Swarms stack, and
  a label whose box would sit on one already placed this frame is dropped rather than overdrawn; ``n`` hides
  them all. Crown towers are drawn from their footprint and keep their shapes, so they carry no label.

PACING
  The match runs on the wall clock at ``--tick-rate`` ticks per second (20 = RoyaleSim's real time). Each frame
  converts elapsed time into a WHOLE target tick (never a float accumulator) and advances the engine to it. A
  stall (window drag, breakpoint, a slow first forward) can therefore never fast-forward the match: the excess
  beyond ``MAX_CATCHUP_TICKS`` is dropped by re-anchoring the clock, so the sim resumes from where it was rather
  than replaying the stall. Pausing re-anchors the same way.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:                                     # `python pipeline/human_play.py` too
    sys.path.insert(0, str(REPO))

TICK_S = 0.05                      # RoyaleSim runs 20 ticks/s (BattleState.tick_ms)
DECIDE_TICKS = 10                  # the bot decides at most once per 0.5 s of engine time
MAX_CATCHUP_TICKS = 10             # a stall may advance the match by at most 0.5 s at once
DOUBLE_ELIXIR_TICK = 2400          # 2:00 -- elixir generation doubles
FEED_ROWS = 7                      # sidebar lines of the deploy timeline (human + bot), newest last
TILE_PX = 24                       # board scale; 18 * 24 = 432 px wide, 32 * 24 = 768 px tall
SIDEBAR_PX = 264
FORM_SUFFIX = {1: "@evolution", 2: "@hero"}      # royale_env's form -> engine deck-name suffix
DEFAULT_TAU = 0.27                 # e1_eval.TAU_LIVE: the eval decision rule
# First existing wins. The first is the checkpoint `pipeline/rl_royale.yaml:6` initialises from, so the client
# plays the same weights the RL stage will; the others are the same run under another seed or output root.
DEFAULT_CKPTS = (Path("icebow/data/pipeline/s1_icebow_v6aug_s1.pt"),   # the path pipeline/rl_royale.yaml:6 names
                 Path("icebow/data/pipeline/s1_icebow_v6aug_s0.pt"),   # same run, other seed
                 Path("data/pipeline/s1_icebow_v6aug_s0.pt"))          # a --data build that wrote into data/
DEFAULT_LEAGUE_DECKS = Path("scratchpad/gauntlet/L68/selfplay/loadable_decks.json")   # rl_royale.yaml: league_decks

# --- palette (the headless frame check asserts these exact colours are on screen) -------------------------
GRASS_LO = (52, 120, 66)           # team 0's half
GRASS_HI = (46, 108, 60)           # team 1's half
WATER = (36, 84, 150)
BRIDGE = (150, 116, 74)
GRID = (255, 255, 255)
TEAM0 = (70, 130, 200)             # BLUE
TEAM1 = (200, 70, 70)              # RED
SIDEBAR_BG = (24, 26, 32)
PANEL_BG = (38, 41, 50)
INK = (238, 238, 238)
INK_DIM = (150, 155, 165)
OK_GREEN = (110, 210, 130)
BAD_RED = (235, 90, 90)
ELIXIR = (196, 92, 214)
HP_GREEN = (120, 220, 120)
HP_RED = (230, 120, 120)
TEAM_COLOURS = {0: TEAM0, 1: TEAM1}
TEAM_NAMES = {0: "BLUE", 1: "RED"}


# ------------------------------------------------------------------------------------------------------
# deck -> engine names
# ------------------------------------------------------------------------------------------------------
def engine_deck(deck, cards) -> list[str]:
    """The deck's 8 cards in the engine's spelling, evolution/hero forms as ``Name@evolution``/``Name@hero``.

    ``cards`` is the engine catalogue (``RustEngine.cards()``): the deck yaml names detector classes
    (``x_bow``, ``the_log``), the engine names CamelCase cards (``Xbow``, ``Log``); ``vocab`` is the one
    mapping between them. A card the engine does not load is a hard error, never a silent substitution.
    """
    from pipeline import vocab
    by_key: dict[str, str] = {}
    for c in cards:
        key = vocab.engine_key(c.name)
        if key:
            by_key.setdefault(key, c.name)
    out: list[str] = []
    for cls in deck.cards:
        name = by_key.get(vocab.base_key(cls))
        if name is None:
            raise SystemExit(f"human_play: deck {deck.name!r} card {cls!r} is not in the engine's catalogue")
        form = 1 if cls.endswith(("_evo", "_evolution")) else 2 if cls.endswith("_hero") else 0
        out.append(name + FORM_SUFFIX.get(form, ""))
    if len(set(out)) != 8:
        raise SystemExit(f"human_play: deck {deck.name!r} does not map to 8 distinct engine cards: {out}")
    return out


# ------------------------------------------------------------------------------------------------------
# geometry
# ------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ArenaView:
    """The board's three frames: tiles, engine units and window pixels.

    ``water`` and ``bridges`` are in TILES, half-open, taken from the arena object so a river or bridge lane
    move in the engine data does not need a code change. ``tile`` is ``Arena.subtile``: engine units per tile.
    ``flip_y`` is a DISPLAY-ONLY mirror about the river: the engine puts team 0 at low y, so side 0's human
    would otherwise play from the top of the window. Both directions of the pixel mapping apply it, so a click
    still lands where the cursor is.
    """
    tiles_x: int
    tiles_y: int
    tile: int
    water: tuple[float, float]
    bridges: tuple[tuple[float, float], ...]
    flip_y: bool = False
    origin: tuple[int, int] = (0, 0)

    @property
    def width_px(self) -> int:
        return self.tiles_x * TILE_PX

    @property
    def height_px(self) -> int:
        return self.tiles_y * TILE_PX

    def px(self, x: float, y: float) -> tuple[int, int]:
        """Engine units (EntityState frame) -> window pixels, clamped to the board."""
        ox, oy = self.origin
        ty = self.tiles_y - y / self.tile if self.flip_y else y / self.tile
        return (ox + round(x / self.tile * TILE_PX), oy + round(ty * TILE_PX))

    def radius_px(self, radius: float) -> int:
        return max(2, round(radius / self.tile * TILE_PX))

    def px_to_engine(self, px: float, py: float) -> tuple[int, int]:
        """Window pixels -> the DeployCommand frame. Rounding at the last step keeps the click's own point,
        which is what the engine's placer expects (it resolves the tile itself)."""
        ox, oy = self.origin
        ty = (py - oy) / TILE_PX
        if self.flip_y:
            ty = self.tiles_y - ty
        return (round((px - ox) / TILE_PX * self.tile), round(ty * self.tile))

    @property
    def screen_water(self) -> tuple[float, float]:
        """``water`` in SCREEN tile rows (a flip about the river moves a band that is not symmetric)."""
        lo, hi = self.water
        return (self.tiles_y - hi, self.tiles_y - lo) if self.flip_y else (lo, hi)

    def obs_to_engine(self, x: float, y: float) -> tuple[int, int]:
        """The OBSERVATION frame -> engine units. ``policy_io.cell_to_engine`` works in
        ``obs_contract.ENGINE_X`` (1000 per tile, what the policy sees); ``DeployCommand`` and
        ``EntityState`` are ``Arena.subtile`` units per tile (18000, ``royale_env.SCALE``)."""
        k = self.tile / 1000.0
        return round(x * k), round(y * k)


def load_arena(core, flip_y: bool = False) -> ArenaView:
    """The engine's own arena geometry; ``royalegym.render.Arena`` from ``arena.json`` if the core refuses;
    the 18 x 32 layout (river at tiles 15-17, bridges at x 2.5-4.5 and 13.5-15.5) only if both refuse."""
    try:
        a = core.arena()
        half = int(a.half)                                   # half-cells per tile
        return ArenaView(tiles_x=int(a.tiles_x), tiles_y=int(a.tiles_y), tile=int(a.subtile),
                         water=(a.water_half_rows[0] / half, (a.water_half_rows[1] + 1) / half),
                         bridges=tuple((lo / half, (hi + 1) / half) for lo, hi in a.bridges_half_cols),
                         flip_y=flip_y)
    except Exception as exc:                                 # noqa: BLE001 -- a core without geometry, or a broken one
        print(f"human_play: core.arena() unavailable ({exc.__class__.__name__}: {exc}); trying royalegym.render",
              file=sys.stderr)
    try:
        from royalegym.render import Arena, default_calibration
        a = Arena.load(default_calibration())
        half = int(a.half)
        return ArenaView(tiles_x=int(a.tiles_x), tiles_y=int(a.tiles_y), tile=int(a.subtile),
                         water=(a.water_half_rows[0] / half, (a.water_half_rows[1] + 1) / half),
                         bridges=tuple((lo / half, (hi + 1) / half) for lo, hi in a.bridges_half_cols),
                         flip_y=flip_y)
    except Exception as exc:                                 # noqa: BLE001 -- no arena.json, or an unreadable one
        print(f"human_play: royalegym.render.Arena refused ({exc.__class__.__name__}: {exc}); "
              "using the static 18x32 layout", file=sys.stderr)
    return ArenaView(tiles_x=18, tiles_y=32, tile=18000, water=(15.0, 17.0),
                     bridges=((2.5, 4.5), (13.5, 15.5)), flip_y=flip_y)


# ------------------------------------------------------------------------------------------------------
# the bot seat
# ------------------------------------------------------------------------------------------------------
class PolicyBot:
    """The trained policy on one side of the match, deciding on ``e1_eval``'s code path.

    ``decide`` is the whole per-decision flow quoted in the module docstring: one forward per call, called at
    most once per ``DECIDE_TICKS`` engine ticks by ``LiveMatch.bot_turn``. The only match state this class keeps
    is ``plays`` -- the ``_past`` history of accepted own plays, ``(tick, deck slot, board x, board y)`` -- which
    is an input of the next observation; the rest is counters for the sidebar and the headless report.
    """

    def __init__(self, ckpt: Path, deck, deck_names: Sequence[str], deck_ids: Sequence[int], core, side: int,
                 device, tau: float, loaded=None):
        from pipeline import vocab
        from pipeline.e1_eval import GenPolicy, load_policy
        if loaded is None:
            try:
                loaded = load_policy(ckpt, str(device))
            except Exception as exc:                         # a half-written checkpoint, or a foreign file
                raise SystemExit(f"human_play: could not load {ckpt}: {exc.__class__.__name__}: {exc}")
        if isinstance(loaded[0], GenPolicy):
            raise SystemExit("human_play: a generalist checkpoint loads as a GenPolicy and takes the GenBot seat; "
                             "main dispatches on the checkpoint's 'gen' flag, so reaching this line means a caller "
                             "bypassed that dispatch")
        self.model, self.info = loaded
        self.grid = str(self.info.get("grid") or "floor")     # model_v3.cell_xy's convention
        self.tau = float(tau)
        self.device = device
        self.side = int(side)
        self.mirror = self.side == 1
        self.deck = deck
        self.engine_deck = list(deck_names)
        slot_of = {deck.slot_of(vocab.engine_key(n)): i for i, n in enumerate(self.engine_deck)}
        if sorted(slot_of) != list(range(8)):
            raise SystemExit(f"human_play: engine deck {self.engine_deck} does not cover the deck's 8 slots")
        self.deck_index_of_slot = slot_of
        # per obs-contract deck slot costs, in the engine's own deck order (``env.deck_ids``)
        cost = {c.card_id: float(c.elixir) for c in core.cards()}
        by_index = [cost[cid] for cid in deck_ids]
        self.costs = [by_index[slot_of[s]] for s in range(8)]
        self.plays: list[tuple[int, int, float, float]] = []
        self.unmapped: set[str] = set()
        self.next_tick: Optional[int] = None
        self.n_decisions = 0
        self.n_plays = 0
        self.p_min, self.p_max = 1.0, 0.0
        self.waits: Counter = Counter()

    def due(self, tick: int) -> bool:
        return self.next_tick is None or tick >= self.next_tick

    def decide(self, tick: int, raw: dict) -> Optional[dict]:
        """One decision at ``tick``. -> ``{"slot", "cell", "x", "y", "p", "why"}`` for a play, or None.

        ``slot`` is the obs-contract deck slot (0-7) the model chose; ``x``/``y`` are the ``cell_to_engine``
        point in the OBSERVATION frame (1000 per tile) and keep that frame -- ``LiveMatch`` is what converts
        them to the engine's units. ``p``/``why`` are the gate and ``live_decide``'s reason, for the sidebar.
        """
        from pipeline import policy_io as ep
        from pipeline.dataset import _past
        from pipeline.e1_eval import MAX_U, allowed_slots, live_decide, model_forward
        from pipeline.obs_contract import from_engine, to_tokens

        self.next_tick = int(tick) + DECIDE_TICKS
        board = from_engine(ep.compact_raw(raw), self.side, self.deck, engine_deck=self.engine_deck,
                            unmapped=self.unmapped, feature_version=1)
        tok, mask, sc = to_tokens(board, MAX_U)
        enc, heads, p, hand = model_forward(self.model, tok, mask, sc, _past(self.plays, tick),
                                            str(self.device))
        allowed = allowed_slots(hand, self.costs, float(int(board.my_elixir)), afford_mask=True)
        decision = live_decide(self.model, enc, heads, p, allowed, tau=self.tau, stalled=False,
                               device=str(self.device))
        self.n_decisions += 1
        self.p_min, self.p_max = min(self.p_min, p), max(self.p_max, p)
        if not decision["play"]:
            self.waits[decision["why"]] += 1
            return None
        X, Y = ep.cell_to_engine(int(decision["cell"]), self.mirror, self.grid)
        return {"slot": int(decision["slot"]), "cell": int(decision["cell"]), "x": int(X), "y": int(Y),
                "p": float(p), "why": decision["why"]}

    def record(self, tick: int, play: dict) -> None:
        """Keep an ACCEPTED play for the next observation's ``past`` tensor (board frame, as ``_past`` wants)."""
        from pipeline import policy_io as ep
        x, y = ep.cell_center(int(play["cell"]), self.grid)
        self.plays.append((int(tick), int(play["slot"]), float(x), float(y)))
        self.n_plays += 1

    def reset(self) -> None:
        """Forget the finished match (``LiveMatch.restart``): no ``past`` row and no counter may cross a deal."""
        self.plays.clear()
        self.unmapped.clear()
        self.next_tick = None
        self.n_decisions = self.n_plays = 0
        self.p_min, self.p_max = 1.0, 0.0
        self.waits.clear()

    def summary(self) -> str:
        rng = f"{self.p_min:.2f}-{self.p_max:.2f}" if self.n_decisions else "n/a"
        waits = ",".join(f"{k}={v}" for k, v in sorted(self.waits.items())) or "none"
        return (f"bot: {self.n_decisions} decisions, {self.n_plays} accepted deploys, p in {rng}, "
                f"waits {waits}")


class GenBot:
    """A GENERALIST (``{"gen": True}``) checkpoint on the bot seat.

    Observation and decide are ``e1_eval.SelfPlaySide``'s -- the seat ``rl_royale`` plays generalist policies with --
    so the input row (``GenPolicy.row`` plus the v3 sim features / opponent context / cue / target / memory rows the
    checkpoint declares) is built by ``e1_eval``, never re-implemented here. This class only schedules the seat's
    decisions at ``DECIDE_TICKS`` and hands the chosen slot/cell to ``LiveMatch.deploy`` (this module's one placement
    path); an accepted play is then written into the seat's ``done_plays``/``last_play_tick`` exactly as
    ``Match._land`` writes it, so the next ``prepare`` sees it in ``past``.
    """

    def __init__(self, policy, info: dict, env, deck_names: Sequence[str], side: int, device, tau: float):
        from pipeline.e1_eval import SelfPlaySide, enable_sim_features, is_timing
        if is_timing(policy):
            raise SystemExit("human_play: this generalist checkpoint has the P2 slot TIMING head (a hazard, no gate); "
                             "the live loop decides with the gate rule, so this client cannot seat it")
        self.policy, self.info = policy, info
        self.env = env
        self.side = int(side)
        self.device = str(device)
        self.tau = float(tau)
        self.grid = str(info.get("grid") or "floor")
        self.deck_names = list(deck_names)
        # the eval condition: true board (obs clean), no look-ahead, no anti-stall, the gate rule at --
        self.cfg = {"decide_every": DECIDE_TICKS, "obs": "clean", "noise": (), "policy": "live",
                    "tau": self.tau, "afford_mask": True, "stall_elixir": None, "stall_seconds": 0.0,
                    "grid": self.grid, "device": self.device,
                    "feature_version": int(enable_sim_features(env, policy))}
        self._make_seat()
        self.next_tick: Optional[int] = None
        self.n_decisions = 0
        self.n_plays = 0
        self.p_min, self.p_max = 1.0, 0.0
        self.waits: Counter = Counter()

    def _make_seat(self) -> None:
        """Build ``e1_eval.SelfPlaySide`` for this side and validate the checkpoint against the deck it must read."""
        from pipeline.e1_eval import SelfPlaySide, enable_sim_features
        self.cfg["feature_version"] = int(enable_sim_features(self.env, self.policy))
        self.seat = SelfPlaySide(self.env, self.deck_names, self.side, "human_play", 0, self.cfg, self.policy)
        self.deck_index_of_slot = self.seat.deck_index_of_slot
        self.costs = self.seat.costs
        self.unmapped = self.seat.unmapped
        try:                                                 # gen_row's per-slot card ids: fail at startup, not mid-match
            self.policy.slot_ident(self.seat.engine_deck, self.deck_index_of_slot)
        except KeyError as exc:                              # "card 'X' (x) not in the generalist's card_vocab"
            raise SystemExit(f"human_play: {exc.args[0] if exc.args else exc}; the bot's deck must be one the "
                             "checkpoint was trained on -- pass --bot-deck with a deck of known cards")

    def due(self, tick: int) -> bool:
        return self.next_tick is None or tick >= self.next_tick

    def decide(self, tick: int, raw: Optional[dict] = None) -> Optional[dict]:
        """One decision at ``tick``: ``SelfPlaySide.prepare`` -> ``gen_row`` -> the policy's batched forward ->
        ``decide_row`` (``e1_eval``'s decide rule). -> the play in ``PolicyBot.decide``'s shape, or None."""
        from pipeline import policy_io as ep
        self.next_tick = int(tick) + DECIDE_TICKS
        self.seat.state = self.env.raw() if raw is None else raw
        self.seat.prepare()
        enc, heads, p, hand = self.policy.forward_batch([self.seat.gen_row(self.policy)], self.device)
        decision = self.seat.decide_row(self.policy, enc, heads, p[0], hand[0])
        self.n_decisions += 1
        p_gate = float(p[0])
        self.p_min, self.p_max = min(self.p_min, p_gate), max(self.p_max, p_gate)
        if not decision["play"]:
            self.waits[decision["why"]] += 1
            return None
        X, Y = ep.cell_to_engine(int(decision["cell"]), self.seat.mirror, self.grid)
        return {"slot": int(decision["slot"]), "cell": int(decision["cell"]), "x": int(X), "y": int(Y),
                "p": p_gate, "why": decision["why"]}

    def record(self, tick: int, play: dict) -> None:
        """An ACCEPTED play -> the seat's ``past`` bookkeeping (``Match._land``'s ``done_plays`` / clock lines)."""
        from pipeline import policy_io as ep
        x, y = ep.cell_center(int(play["cell"]), self.grid)
        self.seat.done_plays.append((int(tick), int(play["slot"]), float(x), float(y)))
        self.seat.last_play_tick = int(tick)
        self.n_plays += 1

    def reset(self) -> None:
        """A re-dealt match (``LiveMatch.restart``): rebuild the seat, so no observer, memory, ``past`` or counter
        from the finished match survives into the new one."""
        self._make_seat()
        self.next_tick = None
        self.n_decisions = self.n_plays = 0
        self.p_min, self.p_max = 1.0, 0.0
        self.waits.clear()

    def summary(self) -> str:
        rng = f"{self.p_min:.2f}-{self.p_max:.2f}" if self.n_decisions else "n/a"
        waits = ",".join(f"{k}={v}" for k, v in sorted(self.waits.items())) or "none"
        return (f"bot (generalist): {self.n_decisions} decisions, {self.n_plays} accepted deploys, p in {rng}, "
                f"waits {waits}")


# ------------------------------------------------------------------------------------------------------
# the match host
# ------------------------------------------------------------------------------------------------------
class LiveMatch:
    """One RoyaleSim battle plus the human's side. Both seats deploy through ``deploy`` -- one placement path,
    so the engine's own legality rule (``check_deploy``) and refusal codes apply to the human and the bot alike.

    ``restart(seed)`` re-deals the SAME env in place (a new seeded shuffle, warm-up included) and clears every
    counter, the deploy timeline and the bot's ``past`` history, so the ``r`` key is the only restart mechanism and
    there is no second, half-reset match path.
    """

    def __init__(self, env, side: int, bot, view: ArenaView,
                 decks: Optional[Sequence[Sequence[str]]] = None):
        self.env = env
        self.side = int(side)
        self.bot = bot
        self.view = view
        self.core = env.core
        # (side 0's engine deck, side 1's) -- a generalist bot normally plays its own league deck, not the human's
        self.decks = None if decks is None else tuple(list(d) for d in decks)
        self.seed: Optional[int] = None          # the deal restart() last used; the sidebar shows it
        self.names = {c.card_id: c.name for c in self.core.cards()}
        self.attempts: Counter = Counter()      # "human" / "bot" -> attempted deploys
        self.accepted: Counter = Counter()
        self.refusals: dict[str, Counter] = {"human": Counter(), "bot": Counter()}   # status name -> count
        self.feed: list[tuple[int, str, str]] = []         # (tick, who, card) -- newest last, FEED_ROWS kept
        self._state_cache: tuple[int, object] = (-1, None)

    # -- state -----------------------------------------------------------------------------------------
    @property
    def tick(self) -> int:
        return int(self.env.tick)

    def state(self):
        """``BattleState``, decoded at most once per tick (the UI asks for it several times a frame)."""
        tick = self.tick
        if self._state_cache[0] != tick:
            self._state_cache = (tick, self.core.state())
        return self._state_cache[1]

    @property
    def over(self) -> bool:
        return bool(self.env.terminated) or bool(self.state().game_over)

    def hand(self) -> list[int]:
        return self.hand_of(self.side)

    # -- engine ----------------------------------------------------------------------------------------
    def advance(self, ticks: int) -> None:
        if ticks > 0 and not self.over:
            self.env.advance_to(self.tick + int(ticks))

    def deploy(self, side: int, hand_slot: int, x: int, y: int, tag: str) -> tuple[bool, str]:
        """Ask ``check_deploy`` FIRST, then mutate. -> (accepted, status name); nothing mutates on refusal.

        The engine is the only placement authority: an empty hand slot, an unaffordable card and a point
        outside the arena or the card's territory all come back as the engine's own ``DeployStatus``.
        """
        from royalegym.protocol import DeployCommand, DeployStatus
        self.attempts[tag] += 1
        hand = self.state().players[side].hand
        card = int(hand[int(hand_slot)]) if 0 <= int(hand_slot) < len(hand) else -1
        cmd = DeployCommand(side, int(hand_slot), int(x), int(y))
        status = int(self.core.check_deploy(cmd))
        if status == int(DeployStatus.OK):
            (result,) = self.core.step([cmd], 0)
            status = int(result.status)
        name = DeployStatus(status).name
        if status != int(DeployStatus.OK):
            self.refusals[tag][name] += 1
            return False, name
        self._note(tag, card)                                # the timeline is what actually reached the board
        self.accepted[tag] += 1
        self._state_cache = (-1, None)                       # the board changed at this tick
        return True, "OK"

    def _note(self, tag: str, card: int) -> None:
        """One line of the sidebar's deploy timeline (who, which card); refusals stay in ``refusals``."""
        self.feed.append((self.tick, tag, self.names.get(card, "--")))
        del self.feed[:-FEED_ROWS]

    def restart(self, seed: int) -> bool:
        """A NEW deal on the same env and seats -- the way back from a finished match. False without deck names.

        ``RoyaleSelfPlayEnv.reset`` re-runs the whole setup (seeded deal, warm-up, ``terminated`` back to False), so
        every counter, the sidebar feed and the bot's ``past`` history are cleared here: nothing survives a restart.
        """
        if not self.decks:
            return False
        self.env.reset(list(self.decks[0]), list(self.decks[1]), int(seed))
        self.seed = int(seed)
        self.attempts.clear()
        self.accepted.clear()
        for c in self.refusals.values():
            c.clear()
        self.feed.clear()
        self._state_cache = (-1, None)
        if self.bot is not None:
            self.bot.reset()
        return True

    def deploy_human(self, hand_slot: int, x: int, y: int) -> tuple[bool, str]:
        return self.deploy(self.side, hand_slot, x, y, "human")

    def bot_turn(self) -> Optional[dict]:
        """One bot decision if it is due. -> the play it made (None = it waited, or is not due/armed)."""
        if self.bot is None or self.over or not self.bot.due(self.tick):
            return None
        play = self.bot.decide(self.tick, self.env.raw())
        if play is None:
            return None
        hand = self.hand_of(1 - self.side)
        card = self.env.deck_ids[1 - self.side][self.bot.deck_index_of_slot[play["slot"]]]
        if card not in hand:                                 # a hand slot is what the engine takes
            self.attempts["bot"] += 1
            self.refusals["bot"]["CARD_NOT_IN_HAND"] += 1
            return None
        x, y = self.view.obs_to_engine(play["x"], play["y"])
        ok, _ = self.deploy(1 - self.side, hand.index(card), x, y, "bot")
        if ok:
            self.bot.record(self.tick, play)
        return play

    def hand_of(self, side: int) -> list[int]:
        return list(self.state().players[side].hand)

    # -- outcome ---------------------------------------------------------------------------------------
    def outcome(self) -> tuple[str, tuple[int, int]]:
        """(verdict for the human's side, (human crowns, bot crowns))."""
        from royalegym.protocol import Winner
        st = self.state()
        crowns = (int(st.players[self.side].crowns), int(st.players[1 - self.side].crowns))
        if not (st.game_over or self.env.terminated):
            return "in progress", crowns
        winner = int(st.winner)
        # Winner.DRAW (2, the engine's own timed-out verdict) and Winner.NONE (-1) are not a side: crowns decide
        if winner not in (int(Winner.BLUE), int(Winner.RED)):
            verdict = "draw" if crowns[0] == crowns[1] else "win" if crowns[0] > crowns[1] else "loss"
        else:
            verdict = "win" if winner == self.side else "loss"
        return verdict, crowns


# ------------------------------------------------------------------------------------------------------
# the window
# ------------------------------------------------------------------------------------------------------
_MAX_LABEL = 15                    # characters of a card name that fit under a body at the 13 px label font


def label_of(name: Optional[str]) -> str:
    """A body's card name as drawn under it: catalogue spelling -> spaced words, cut at a word boundary.

    ``MegaKnight`` -> ``Mega Knight``, ``RoyalRecruits_Chess`` -> ``Royal Recruits..``. Empty for a card the
    catalogue does not name (crown towers carry ``EMPTY_CARD``): they are drawn as towers and need no label.
    """
    if not name:
        return ""
    words: list[str] = []
    for chunk in str(name).replace("-", " ").replace("_", " ").split():
        letters: list[str] = []
        for i, ch in enumerate(chunk):
            if i and ch.isupper() and (chunk[i - 1].islower() or
                                       (chunk[i - 1].isupper() and i + 1 < len(chunk) and chunk[i + 1].islower())):
                letters.append(" ")                      # CamelCase word boundary, not an acronym's
            letters.append(ch)
        words.append("".join(letters))
    text = " ".join(words)
    if len(text) <= _MAX_LABEL:
        return text
    cut = text[:_MAX_LABEL - 2].rstrip()
    if " " in cut:
        cut = cut[:cut.rfind(" ")]
    return cut + ".."


def _label_ink(colour: tuple[int, int, int]) -> tuple[int, int, int]:
    """The team colour lifted toward white: a label has to read on dark grass AND on its own body."""
    return tuple(min(255, c + 80) for c in colour)


@dataclass
class UIState:
    selected: int = 0
    hover: Optional[tuple[int, int]] = None                  # engine units under the cursor
    paused: bool = False
    labels: bool = True                                      # card names under the bodies ('n' toggles)
    message: str = ""
    message_until: float = field(default=0.0)

    def say(self, text: str, seconds: float = 2.5) -> None:
        self.message, self.message_until = text, time.perf_counter() + seconds

    def current_message(self) -> str:
        return self.message if time.perf_counter() < self.message_until else ""


class Renderer:
    """Draws ``BattleState`` + the HUD onto the window surface. The board (grass, river, bridges, grid) is
    painted once; towers, bodies, spells, the deploy preview and the HUD are drawn per frame.

    Bodies are coloured blobs by engine design (no sprites in this branch), so each one carries its CARD NAME
    underneath, taken from the engine catalogue's ``card_id`` row. That is the only thing that makes a board of
    circles readable: without it a Musketeer and an Ice Wizard are the same dot in a different shade.
    """

    def __init__(self, view: ArenaView, title: str):
        import pygame
        self.pygame = pygame
        pygame.display.init()
        pygame.font.init()
        self.view = view
        self.screen = pygame.display.set_mode((view.width_px + SIDEBAR_PX, view.height_px))
        pygame.display.set_caption(title)
        self.font = pygame.font.Font(None, 20)
        self.font_big = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 16)
        self.font_tiny = pygame.font.Font(None, 13)          # board labels: one card name fits ~3 tiles
        self._text_cache: dict[tuple[str, tuple, int], object] = {}
        self._labels: list = []                              # name boxes already placed this frame
        self.overlay = pygame.Surface((view.width_px + SIDEBAR_PX, view.height_px), pygame.SRCALPHA)
        self.board = self._paint_board()

    # -- static board ----------------------------------------------------------------------------------
    def _paint_board(self):
        pygame = self.pygame
        v = self.view
        surface = pygame.Surface((v.width_px, v.height_px))
        top, bottom = (GRASS_HI, GRASS_LO) if v.flip_y else (GRASS_LO, GRASS_HI)   # team 0's half is LOW y
        mid = round(v.tiles_y / 2 * TILE_PX)
        surface.fill(top, pygame.Rect(0, 0, v.width_px, mid))
        surface.fill(bottom, pygame.Rect(0, mid, v.width_px, v.height_px - mid))
        water = pygame.Surface((v.width_px, v.height_px), pygame.SRCALPHA)
        y0, y1 = int(v.screen_water[0] * TILE_PX), int(v.screen_water[1] * TILE_PX)
        pygame.draw.rect(water, (*WATER, 255), pygame.Rect(0, y0, v.width_px, y1 - y0))
        pygame.draw.line(water, (80, 130, 200), (0, (y0 + y1) // 2), (v.width_px, (y0 + y1) // 2), 2)
        for lo, hi in v.bridges:
            pygame.draw.rect(water, (*BRIDGE, 255),
                             pygame.Rect(int(lo * TILE_PX), y0, int((hi - lo) * TILE_PX) + 1, y1 - y0))
        surface.blit(water, (0, 0))
        grid = pygame.Surface((v.width_px, v.height_px), pygame.SRCALPHA)
        for tx in range(1, v.tiles_x):
            pygame.draw.line(grid, (*GRID, 18), (tx * TILE_PX, 0), (tx * TILE_PX, v.height_px))
        for ty in range(1, v.tiles_y):
            pygame.draw.line(grid, (*GRID, 18), (0, ty * TILE_PX), (v.width_px, ty * TILE_PX))
        surface.blit(grid, (0, 0))
        pygame.draw.rect(surface, (16, 18, 22), pygame.Rect(0, 0, v.width_px, v.height_px), 3)
        return surface

    # -- helpers ---------------------------------------------------------------------------------------
    def _render(self, text: str, colour, font=None):
        """One cached ``Font.render`` per (text, colour, font): the board labels repeat every frame."""
        f = font or self.font
        key = (text, colour, id(f))
        surf = self._text_cache.get(key)
        if surf is None:
            surf = self._text_cache[key] = f.render(text, True, colour)
        return surf

    def _text(self, text: str, pos, colour=INK, font=None, anchor="topleft") -> None:
        surf = self._render(text, colour, font)
        self.screen.blit(surf, surf.get_rect(**{anchor: pos}))

    def _name(self, card: int, centre: tuple[int, int], radius: int, colour, names) -> None:
        """The card's name under its body -- what turns a coloured blob into "Musketeer".

        A swarm (three Skeletons in one tile) would stack three identical labels on top of each other, so a name
        whose box would sit on one already placed this frame is dropped: the body is still drawn, and the sidebar's
        DEPLOYS line names the play. Nothing here is engine state: ``names`` is the catalogue's own ``card_id`` row.
        """
        text = label_of(names.get(card))
        if not text:
            return
        pygame = self.pygame
        rect = self._render(text, colour, self.font_tiny).get_rect(midtop=(centre[0], centre[1] + radius + 2))
        rect.clamp_ip(pygame.Rect(0, 0, self.view.width_px, self.view.height_px))
        if rect.collidelist(self._labels) != -1:
            return
        self._labels.append(rect)
        self.screen.blit(self._render(text, (10, 12, 16), self.font_tiny), rect.move(1, 1))
        self.screen.blit(self._render(text, colour, self.font_tiny), rect)

    def _bar(self, rect, frac: float, colour, back=(20, 22, 28)) -> None:
        pygame = self.pygame
        pygame.draw.rect(self.screen, back, rect)
        if frac > 0:
            pygame.draw.rect(self.screen, colour, pygame.Rect(rect[0], rect[1],
                                                              max(1, int(rect[2] * min(1.0, frac))), rect[3]))

    # -- per frame -------------------------------------------------------------------------------------
    def draw(self, match: LiveMatch, ui: UIState) -> None:
        pygame = self.pygame
        state = match.state()
        self.screen.blit(self.board, (0, 0))
        self.overlay.fill((0, 0, 0, 0))
        self._labels.clear()
        self._draw_spells(state, match.names, ui.labels)
        self._draw_entities(state, match.names, ui.labels)
        self._draw_preview(match, ui)
        self.screen.blit(self.overlay, (0, 0))
        self._draw_sidebar(match, ui, state)
        if match.over:
            self._draw_banner(match)
        pygame.display.flip()

    def _draw_spells(self, state, names, labels: bool) -> None:
        pygame = self.pygame
        for spell in state.spells:
            colour = TEAM_COLOURS.get(int(spell.team), INK)
            centre = self.view.px(spell.x, spell.y)
            pygame.draw.circle(self.overlay, (*colour, 70), centre, 18)
            pygame.draw.circle(self.overlay, (*colour, 200), centre, 18, 2)
            aim = self.view.px(spell.aim_x, spell.aim_y)
            pygame.draw.line(self.overlay, (*colour, 140), centre, aim, 1)
            if labels:
                self._name(int(spell.card_id), centre, 18, _label_ink(colour), names)

    def _draw_entities(self, state, names, labels: bool) -> None:
        from royalegym.protocol import EntityKind
        pygame = self.pygame
        for e in state.entities:
            colour = TEAM_COLOURS.get(int(e.team), INK)
            if e.kind in (int(EntityKind.KING_TOWER), int(EntityKind.PRINCESS_TOWER)):
                self._draw_tower(e, colour)
            else:
                centre = self.view.px(e.x, e.y)
                radius = min(2 * TILE_PX, self.view.radius_px(e.radius))
                if e.deploy_ticks > 0:                       # still deploying: hollow, dimmed
                    pygame.draw.circle(self.screen, tuple(c // 2 for c in colour), centre, radius)
                    pygame.draw.circle(self.screen, INK, centre, radius + 2, 1)
                else:
                    pygame.draw.circle(self.screen, colour, centre, radius)
                    pygame.draw.circle(self.screen, tuple(c // 3 for c in colour), centre, radius, 1)
                if e.flying:
                    pygame.draw.circle(self.screen, INK, centre, radius + 3, 1)
                self._hp_bar(centre[0] - radius, centre[1] - radius - 6, 2 * radius, e.hp, e.max_hp)
                if labels:
                    self._name(int(e.card_id), centre, radius, _label_ink(colour), names)

    def _draw_tower(self, e, colour: tuple[int, int, int]) -> None:
        """Crown towers from their ENGINE footprint, so a bigger king block is drawn bigger."""
        from royalegym.protocol import EntityKind
        pygame = self.pygame
        xa, ya = self.view.px(e.footprint[0], e.footprint[1])
        xb, yb = self.view.px(e.footprint[2], e.footprint[3])
        rect = pygame.Rect(min(xa, xb), min(ya, yb), max(6, abs(xb - xa)), max(6, abs(yb - ya)))
        pygame.draw.rect(self.screen, colour, rect, border_radius=4)
        pygame.draw.rect(self.screen, tuple(c // 3 for c in colour), rect, 2, border_radius=4)
        if e.kind == int(EntityKind.KING_TOWER):             # the king's raised centre
            pygame.draw.rect(self.screen, tuple(c // 2 for c in colour), rect.inflate(-8, -8), border_radius=3)
        self._hp_bar(rect.x, rect.y - 7, rect.width, e.hp, e.max_hp)

    def _hp_bar(self, x: int, y: int, width: int, hp: int, max_hp: int) -> None:
        pygame = self.pygame
        frac = 0.0 if max_hp <= 0 else max(0.0, min(1.0, hp / max_hp))
        width = max(4, width)
        pygame.draw.rect(self.screen, (20, 20, 24), pygame.Rect(x, y, width, 4))
        self._bar((x, y, width, 4), frac, HP_GREEN if frac > 0.35 else HP_RED)

    def _draw_preview(self, match: LiveMatch, ui: UIState) -> None:
        """Where the selected card would go, and the ENGINE's verdict on that exact command."""
        pygame = self.pygame
        if ui.hover is None or match.over:
            return
        from royalegym.protocol import DeployCommand, DeployStatus
        x, y = ui.hover
        if not (0 <= x < self.view.tiles_x * self.view.tile and 0 <= y < self.view.tiles_y * self.view.tile):
            return                                           # the cursor is over the sidebar, not the board
        hand = match.hand()
        if not 0 <= ui.selected < len(hand) or hand[ui.selected] < 0:
            return
        status = int(match.core.check_deploy(DeployCommand(match.side, ui.selected, x, y)))
        colour = OK_GREEN if status == int(DeployStatus.OK) else BAD_RED
        centre = self.view.px(x, y)
        pygame.draw.circle(self.screen, colour, centre, 10, 2)
        pygame.draw.line(self.screen, colour, (centre[0] - 6, centre[1]), (centre[0] + 6, centre[1]), 1)
        pygame.draw.line(self.screen, colour, (centre[0], centre[1] - 6), (centre[0], centre[1] + 6), 1)
        if status != int(DeployStatus.OK):
            self._text(DeployStatus(status).name, (centre[0] + 14, centre[1] - 8), BAD_RED, self.font_small)

    def _draw_sidebar(self, match: LiveMatch, ui: UIState, state) -> None:
        pygame = self.pygame
        x0 = self.view.width_px
        panel = pygame.Rect(x0, 0, SIDEBAR_PX, self.view.height_px)
        pygame.draw.rect(self.screen, SIDEBAR_BG, panel)
        pygame.draw.line(self.screen, (60, 64, 74), (x0, 0), (x0, self.view.height_px), 2)
        left, width = x0 + 12, SIDEBAR_PX - 24
        y = 10
        you, them = state.players[match.side], state.players[1 - match.side]
        self._text(f"YOU = {TEAM_NAMES[match.side]}   BOT = {TEAM_NAMES[1 - match.side]}", (left, y), INK,
                   self.font_small)
        y += 20
        clock = state.tick * TICK_S
        phase = "OVERTIME" if state.overtime else ("DOUBLE ELIXIR" if state.tick >= DOUBLE_ELIXIR_TICK
                                                   else "REGULAR")
        self._text(f"{int(clock) // 60}:{int(clock) % 60:02d}", (left, y), INK, self.font_big)
        self._text(phase, (left + width, y + 8), INK_DIM, self.font_small, "topright")
        y += 32
        self._text(f"crowns {you.crowns} - {them.crowns}    hp {sum(you.tower_hp)} - {sum(them.tower_hp)}",
                   (left, y), INK_DIM, self.font_small)
        y += 22
        elixir = you.elixir_milli / 1000.0
        self._text(f"elixir {elixir:.1f}", (left, y), ELIXIR, self.font_small)
        y += 16
        tenths = round(min(10.0, elixir) * 10)               # 10 pips, fractional from elixir_milli
        for i in range(10):
            filled = max(0, min(10, tenths - i * 10))
            rect = pygame.Rect(left + i * (width // 10), y, width // 10 - 2, 14)
            pygame.draw.rect(self.screen, (44, 34, 52), rect)
            if filled:
                pygame.draw.rect(self.screen, ELIXIR, pygame.Rect(rect.x, rect.y, rect.w * filled // 10, rect.h))
        y += 26
        self._text("HAND   (1-4 selects, click deploys)", (left, y), INK_DIM, self.font_small)
        y += 18
        for slot in range(4):
            card = you.hand[slot] if slot < len(you.hand) else -1
            cost = you.hand_costs[slot] if slot < len(you.hand_costs) else 0
            rect = pygame.Rect(left, y, width, 34)
            selected = slot == ui.selected
            pygame.draw.rect(self.screen, PANEL_BG if not selected else (70, 62, 96), rect, border_radius=3)
            if selected:
                pygame.draw.rect(self.screen, ELIXIR, rect, 2, border_radius=3)
            label = f"{slot + 1}  {match.names.get(card, '--') if card >= 0 else '--'}"
            self._text(label, (rect.x + 8, rect.y + 9), INK, self.font)
            affordable = card >= 0 and cost <= elixir
            self._text(f"{cost}", (rect.right - 10, rect.y + 9), ELIXIR if affordable else INK_DIM,
                       self.font, "topright")
            y += 38
        nxt = you.next_card
        self._text(f"next: {match.names.get(nxt, '--') if nxt >= 0 else '--'}", (left, y), INK_DIM,
                   self.font_small)
        y += 24
        if match.bot is not None:
            self._text(f"bot: {match.bot.n_decisions} decisions / {match.bot.n_plays} plays", (left, y),
                       INK_DIM, self.font_small)
            y += 18
            self._text(f"bot p {match.bot.p_min:.2f}-{match.bot.p_max:.2f}  tau {match.bot.tau:.2f}",
                       (left, y), INK_DIM, self.font_small)
        else:
            self._text("bot: none (opponent idle)", (left, y), INK_DIM, self.font_small)
        y += 24
        seed = "" if match.seed is None else f"  seed {match.seed}"
        self._text(f"DEPLOYS{seed}", (left, y), INK_DIM, self.font_small)
        refused = sum(match.refusals["human"].values()) + sum(match.refusals["bot"].values())
        if refused:
            self._text(f"{refused} refused", (left + width, y), BAD_RED, self.font_small, "topright")
        y += 16
        for feed_tick, feed_tag, feed_card in match.feed[-FEED_ROWS:]:
            secs = int(feed_tick * TICK_S)
            line = f"{'you' if feed_tag == 'human' else 'bot'} {secs // 60}:{secs % 60:02d} {feed_card}"
            self._text(line, (left, y), INK, self.font_small)
            y += 15
        y += 10
        if ui.paused:
            self._text("PAUSED (space)", (left, y), OK_GREEN, self.font_big)
            y += 26
        msg = ui.current_message()
        if msg:
            self._text(msg, (left, y), BAD_RED, self.font_small)
            y += 20
        footer = ["1-4 select a hand card", "click: deploy at the cursor", "space: pause   r: restart",
                  "n: names on/off   esc: quit"]
        for i, line in enumerate(reversed(footer)):
            self._text(line, (left, self.view.height_px - 14 - 16 * i), INK_DIM, self.font_small)

    def _draw_banner(self, match: LiveMatch) -> None:
        """``BattleState.winner`` when the engine ended it, crowns when it timed out."""
        pygame = self.pygame
        verdict, crowns = match.outcome()
        if verdict == "in progress":
            return
        height = 104
        rect = pygame.Rect(0, (self.view.height_px - height) // 2, self.view.width_px, height)
        banner = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        banner.fill((0, 0, 0, 190))
        self.screen.blit(banner, rect.topleft)
        colour = OK_GREEN if verdict == "win" else BAD_RED if verdict == "loss" else INK
        clock = int(match.state().tick * TICK_S)
        self._text(f"{verdict.upper()}  {crowns[0]}-{crowns[1]}", (self.view.width_px // 2, rect.y + 24),
                   colour, self.font_big, "center")
        self._text(f"{clock // 60}:{clock % 60:02d}", (self.view.width_px // 2, rect.y + 56), INK_DIM,
                   self.font_small, "center")
        self._text("r: new match   esc: quit", (self.view.width_px // 2, rect.y + 76), INK,
                   self.font_small, "center")


# ------------------------------------------------------------------------------------------------------
# loops
# ------------------------------------------------------------------------------------------------------
def _refusal_text(status: str) -> str:
    return status.replace("_", " ").lower()


def run_window(match: LiveMatch, renderer: Renderer, ui: UIState, args, save_frame: Optional[Path]) -> int:
    """The interactive loop. Wall-clock driven: whole ticks from elapsed time, never a float accumulator."""
    pygame = renderer.pygame
    clock = pygame.time.Clock()
    base_time, base_tick = time.perf_counter(), match.tick
    restarts = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    ui.paused = not ui.paused
                    base_time, base_tick = time.perf_counter(), match.tick
                    ui.say("paused" if ui.paused else "resumed", 1.0)
                elif event.key == pygame.K_r:
                    restarts += 1
                    seed = int(args.seed) + restarts         # a new deal per restart, so "again" is a new match
                    if match.restart(seed):
                        ui.paused = False
                        ui.say(f"new match, seed {seed}", 2.5)
                        base_time, base_tick = time.perf_counter(), match.tick
                    else:
                        ui.say("restart unavailable: no deck names", 2.5)
                elif event.key == pygame.K_n:
                    ui.labels = not ui.labels
                    ui.say("card names on" if ui.labels else "card names off", 1.0)
                elif pygame.K_1 <= event.key <= pygame.K_4:
                    ui.selected = event.key - pygame.K_1
            elif event.type == pygame.MOUSEMOTION:
                ui.hover = renderer.view.px_to_engine(*event.pos)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                ui.hover = renderer.view.px_to_engine(*event.pos)
                if event.pos[0] >= renderer.view.width_px:
                    ui.say("click inside the arena to deploy", 1.5)
                elif ui.paused:
                    ui.say("paused: resume with space", 1.5)
                else:
                    ok, status = match.deploy_human(ui.selected, *ui.hover)
                    ui.say(f"slot {ui.selected + 1} deployed" if ok else _refusal_text(status), 2.0)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
                ui.hover = None
        if ui.paused or match.over:
            base_time, base_tick = time.perf_counter(), match.tick
        else:
            now = time.perf_counter()
            want = base_tick + int((now - base_time) * args.tick_rate)
            if want - match.tick > MAX_CATCHUP_TICKS:        # a stall: drop the debt, never fast-forward
                base_time, base_tick = now, match.tick + MAX_CATCHUP_TICKS
                want = base_tick
            match.advance(want - match.tick)
            match.bot_turn()                                 # at most one decision per DECIDE_TICKS
        renderer.draw(match, ui)
        clock.tick(args.fps)
    if save_frame is not None:
        renderer.draw(match, ui)
        pygame.image.save(renderer.screen, str(save_frame))
        print(f"human_play: saved {save_frame}")
    return 0


def run_headless(match: LiveMatch, renderer: Renderer, ui: UIState, args, save_frame: Optional[Path]) -> int:
    """``--headless N``: N ticks of the same engine path with a scripted human, no wall clock, no window events.

    Every 20 ticks the script plays the CHEAPEST card in hand into its own half (elixir permitting -- when it cannot
    afford any card the engine's own NOT_ENOUGH_ELIXIR is what a player would see), and every fourth attempt is the
    classic misclick into the opponent's half. The bot seat runs on its normal 10-tick cadence. Ticks run on a whole
    5-tick step while keeping ``N`` an exact budget. This mode is the scripted proof: it ASSERTS the match advanced
    and, with ``--bot policy``, that the policy actually deployed; either failing exits 2.
    """
    pygame = renderer.pygame
    start = match.tick
    stop = start + int(args.headless)
    legal_y = 11.0 if match.side == 0 else 21.0
    # three points in the human's own half, then the classic misclick: the opponent's half (the engine refuses
    # it for a troop or a building and allows it for a spell, which is exactly what a player would see)
    spots = ((3.5, legal_y), (9.0, legal_y + 1.0), (14.5, legal_y), (9.0, 31.0 - legal_y))
    k = 0
    while match.tick < stop and not match.over:
        if (match.tick - start) % 20 == 0:
            hand = match.hand()
            costs = match.state().players[match.side].hand_costs
            slot = min(range(len(hand)), key=lambda s: costs[s] if hand[s] >= 0 else 99)   # cheapest in hand
            tx, ty = spots[k % len(spots)]
            ok, status = match.deploy_human(slot, int(tx * renderer.view.tile), int(ty * renderer.view.tile))
            ui.say(f"slot {slot + 1} deployed" if ok else _refusal_text(status), 1.0)
            k += 1
        match.bot_turn()
        match.advance(min(5, stop - match.tick))
    renderer.draw(match, ui)
    if save_frame is not None:
        pygame.image.save(renderer.screen, str(save_frame))
        print(f"human_play: saved {save_frame}")
    verdict, crowns = match.outcome()
    refused = match.attempts["human"] - match.accepted["human"]
    print(f"human_play: ticks {start}->{match.tick}  verdict {verdict}  crowns {crowns[0]}-{crowns[1]}")
    print(f"human_play: deploys attempted {match.attempts['human']} accepted {match.accepted['human']} "
          f"refused {refused}"
          + (f" ({dict(sorted(match.refusals['human'].items()))})" if match.refusals["human"] else ""))
    if match.bot is not None:
        print(f"human_play: {match.bot.summary()} (attempts {match.attempts['bot']}, "
              f"accepted {match.accepted['bot']}, refused {dict(sorted(match.refusals['bot'].items()))})")
        if match.bot.unmapped:
            print(f"human_play: unmapped entity names dropped from the bot's view: "
                  f"{sorted(match.bot.unmapped)}")
    else:
        print("human_play: bot none (opponent idle)")
    for feed_tick, feed_tag, feed_card in match.feed:
        secs = int(feed_tick * TICK_S)
        print(f"human_play: deploy {feed_tag:5s} {secs // 60}:{secs % 60:02d} {feed_card}")
    advanced = match.tick - start
    failed = []
    if advanced <= 0:
        failed.append("the match did not advance")
    if match.bot is not None and match.accepted["bot"] <= 0:
        failed.append("the policy made no accepted deploy")
    if failed:
        print("human_play: FAILED -- " + "; ".join(failed))
        return 2
    print(f"human_play: OK -- {advanced} ticks, {match.accepted['human']} human + "
          f"{match.accepted['bot']} bot accepted deploys")
    return 0


# ------------------------------------------------------------------------------------------------------
# entry point
# ------------------------------------------------------------------------------------------------------
def resolve_ckpt(explicit: Optional[str]) -> Path:
    """``--ckpt`` wins; otherwise the first existing default. Missing -> one line, no traceback."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"human_play: checkpoint {p} does not exist")
        return p
    for candidate in DEFAULT_CKPTS:
        if candidate.is_file():
            return candidate
    looked = ", ".join(str(p) for p in DEFAULT_CKPTS)
    raise SystemExit(f"human_play: no checkpoint found (looked for: {looked}); "
                     "pass --ckpt PATH or run with --bot none")


def league_bot_deck(seed: int, policy) -> tuple[list[str], str, str]:
    """The deck a GENERALIST opponent plays, by the LEAGUE rule ``pipeline/rl_royale.yaml`` already defines.

    A generalist is deck-agnostic (it was trained on every deck), so -- unlike the S1 specialist, which is
    icebow-only -- its deck comes from the census draw ``rl_royale`` uses for league self-play: a ``loadable_decks.json``
    deck weighted by ``deck_weights(sides ** league_deck_alpha, floor)``, or icebow with probability
    ``league_icebow_share``. Decks holding a card the checkpoint's ``card_vocab`` does not know are dropped (the
    gen row's slot ids would be undefined), and so is icebow when it holds one. -> (engine names, deck name, why).
    """
    import numpy as np
    from pipeline.dataset_gen import card_key
    from pipeline.rl_royale import ICEBOW_DECK, deck_weights, league_decks, sample_deck
    cfg: dict = {}
    league_yaml = REPO / "pipeline" / "rl_royale.yaml"
    if league_yaml.is_file():
        import yaml
        cfg = yaml.safe_load(league_yaml.read_text(encoding="utf-8")) or {}
    path = REPO / str(cfg.get("league_decks") or DEFAULT_LEAGUE_DECKS)
    if not path.is_file():
        raise SystemExit(f"human_play: no league deck census at {path} (rl_royale.yaml: league_decks); pass "
                         "--bot-deck with a deck of known cards")
    known = lambda names: all(card_key(str(n).split("@")[0]) in policy.gid for n in names)   # noqa: E731
    census = [d for d in league_decks(path) if known(d["engine"])]
    if not census:
        raise SystemExit(f"human_play: none of {path.name}'s decks is inside the checkpoint's card_vocab "
                         f"({len(policy.gid)} cards); pass --bot-deck with a deck of known cards")
    p = deck_weights([d["sides"] for d in census], float(cfg.get("league_deck_alpha", 0.5)),
                     float(cfg.get("league_deck_floor", 0.5)))
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x1EA6]))
    icebow_share = float(cfg.get("league_icebow_share", 0.2)) if known(ICEBOW_DECK["engine"]) else 0.0
    drawn = sample_deck(rng, census, p, icebow_share)
    return (list(drawn["engine"]), str(drawn["name"]),
            f"league census {path.name}: {len(census)} decks fit the checkpoint vocab, icebow share {icebow_share:g}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="uv run python -m pipeline.human_play", description=__doc__.splitlines()[0])
    ap.add_argument("--deck", default="icebow", help="the HUMAN's deck: deck yaml name or path")
    ap.add_argument("--bot-deck", default=None,
                    help="the BOT's deck (deck yaml name or path). Default: the human's deck for an S1 checkpoint; "
                         "for a generalist, a deck drawn from the league census (rl_royale.yaml)")
    ap.add_argument("--seed", type=int, default=0, help="deal seed (env.reset; also seeds the league deck draw)")
    ap.add_argument("--side", type=int, choices=(0, 1), default=0, help="the human's side (0 = BLUE, bottom)")
    ap.add_argument("--bot", choices=("policy", "none"), default="policy",
                    help="the other side: the trained policy, or idle")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (S1 or a generalist's gen_s0.pt)")
    ap.add_argument("--tick-rate", type=int, default=20, help="engine ticks per wall-clock second (20 = real time)")
    ap.add_argument("--fps", type=int, default=60, help="window frames per second")
    ap.add_argument("--headless", type=int, default=0, metavar="N",
                    help="run N ticks with scripted actions and no window")
    ap.add_argument("--save-frame", default=None, metavar="PATH", help="write a PNG of the final frame")
    ap.add_argument("--device", default=None, help="cuda device for the bot (default: auto -> cuda:0)")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU,
                    help="gate threshold of the eval decision rule (e1_eval.TAU_LIVE)")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.tick_rate <= 0 or args.fps <= 0:
        raise SystemExit("human_play: --tick-rate and --fps must be positive")
    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # before any display init
    ckpt = resolve_ckpt(args.ckpt) if args.bot == "policy" else None

    from pipeline.obs_contract import load_deck
    from pipeline.royale_env import RoyaleSelfPlayEnv

    deck = load_deck(args.deck)                              # the human's deck
    env = RoyaleSelfPlayEnv(forms_mode="deck", tail_cap=7200)
    human_names = engine_deck(deck, env.core.cards())

    loaded, device = None, None
    if ckpt is not None:                                     # --bot none never touches torch or CUDA
        from pipeline.device import resolve as resolve_device
        from pipeline.e1_eval import GenPolicy, load_policy
        device = resolve_device(args.device)
        try:
            loaded = load_policy(ckpt, str(device))
        except SystemExit:
            raise
        except Exception as exc:                             # a half-written checkpoint, or a foreign file
            raise SystemExit(f"human_play: could not load {ckpt}: {exc.__class__.__name__}: {exc}")
    gen = loaded is not None and isinstance(loaded[0], GenPolicy)

    bot_deck, bot_label = None, "none"
    if loaded is None:
        bot_names = human_names
    elif gen:
        policy = loaded[0]
        if args.bot_deck:                                    # an explicit bot deck wins over the league draw
            bot_deck = load_deck(args.bot_deck)
            bot_names = engine_deck(bot_deck, env.core.cards())
            bot_label = f"{ckpt} on {bot_deck.name}"
        else:
            bot_names, drawn, why = league_bot_deck(args.seed, policy)
            bot_label = f"{ckpt} (generalist) on league deck {drawn} -- {why}"
    else:
        bot_deck = load_deck(args.bot_deck) if args.bot_deck else deck
        bot_names = engine_deck(bot_deck, env.core.cards()) if args.bot_deck else human_names
        bot_label = f"{ckpt} on {bot_deck.name}"

    decks = (human_names, bot_names) if args.side == 0 else (bot_names, human_names)
    env.reset(list(decks[0]), list(decks[1]), int(args.seed))
    view = load_arena(env.core, flip_y=args.side == 0)

    bot = None
    if loaded is not None:
        bot_side = 1 - args.side
        if gen:
            bot = GenBot(loaded[0], loaded[1], env, bot_names, bot_side, device, args.tau)
        else:
            bot = PolicyBot(ckpt, bot_deck, bot_names, env.deck_ids[bot_side], env.core, bot_side, device,
                            args.tau, loaded=loaded)
    match = LiveMatch(env, args.side, bot, view, decks)
    match.seed = int(args.seed)
    ui = UIState()
    save_frame = Path(args.save_frame) if args.save_frame else None
    renderer = Renderer(view, f"RoyaleSim - {deck.name} ({'you: ' + TEAM_NAMES[args.side]})")
    print(f"human_play: you = {deck.name} / {TEAM_NAMES[args.side]}, seed {args.seed}, "
          f"bot = {bot_label}, ticks/turn {DECIDE_TICKS}, rate {args.tick_rate}/s")
    print(f"human_play: decks side0={','.join(decks[0])}  side1={','.join(decks[1])}")
    if args.headless:
        return run_headless(match, renderer, ui, args, save_frame)
    return run_window(match, renderer, ui, args, save_frame)


if __name__ == "__main__":
    raise SystemExit(main())
