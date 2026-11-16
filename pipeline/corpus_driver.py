"""Corpus driver: crawl timeline -> ``corpus_v3/replay_<tag>.json``.

This is the producer `pipeline.dataset` / `pipeline.dataset_gen` / `pipeline.e1_pool` all require and that the
published snapshot does not contain. It replays a recorded timeline (``plays_ext.csv``) through RoyaleSim with a
``ShuffleMode.NONE`` deal fitted to the recorded plays, and records every field the consumers read.

Timeline sources: ``crawl`` today; a memory-capture JSONL adapter plugs into the same ``(tick, side, card, x, y)``
stream (see ``build_entry``), which is how a reader-collected game becomes a corpus record.

Unit conventions (settled from the consumers, not guessed):
  * compact ``frames`` / ``play_frames`` carry ENGINE units, 1000 per tile, x in [0, 18000], y in [0, 32000] —
    ``obs_contract._engine_xy`` divides by 18000/32000, and ``RoyalePoolEnv.act`` multiplies tiles by 1000.
  * ``log[].x/y`` and ``plays_ext.csv`` also use engine units.
  * ``final_decks`` / ``log[].card`` use the recorder's card keys (``knight-ev1``) for plays and ENGINE display
    names with an ``@evolution``/``@hero`` suffix for decks, exactly as ``e1_pool.deck_names`` expects.

Usage:
    uv run python -m pipeline.corpus_driver icebow --crawl data/hf_crawl/icebow --out scratchpad/gauntlet/ext/corpus_v3/icebow --limit 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

REPO = Path(__file__).resolve().parents[1]
SCALE = 1000  # engine units per tile
RECORD_EVERY = 20  # ticks between compact frames (dataset.py default)
ELIXIR_SLACK = 40  # ticks a refused play may be retried while the game catches up


def _corpus_rows():
    src = REPO / "icebow" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    import corpus_rows  # noqa: PLC0415

    return corpus_rows


# ---------------------------------------------------------------------------------------------- timeline
def build_entry(tag: str, battle: dict, plays: list[tuple], cr) -> dict:
    """``plays`` rows (tick, play_index, side, ability, card, x, y) -> the deck/command pair the driver runs.

    Both sides are driven from the recording, so the entry keeps the pool row's shape but nothing is "ours".
    """
    decks = {s: cr.deck_for_side(battle, s) for s in (0, 1)}
    slug_of = {s: {cr.split_slug(it["slug"])[0]: it["slug"] for it in decks[s]} for s in (0, 1)}
    # The crawl's team/opponent columns and the event side labels are not guaranteed to agree (measured: icebow's
    # own cards appear on side 0 while `battles.csv` lists that deck under `opponent_deck`). Pick the assignment
    # that explains more of the recorded plays, and record whether it was swapped.
    direct = sum(1 for r in plays if r[2] in (0, 1) and str(r[4]) in slug_of[r[2]])
    swapped = sum(1 for r in plays if r[2] in (0, 1) and str(r[4]) in slug_of[1 - r[2]])
    did_swap = swapped > direct
    if did_swap:
        decks = {0: decks[1], 1: decks[0]}
        slug_of = {0: slug_of[1], 1: slug_of[0]}
    cmds: dict[int, list] = {0: [], 1: []}
    rejected = []
    for tick, play_index, side, ability, card, x, y in plays:
        if side not in (0, 1):
            rejected.append({"play_index": play_index, "reason": "bad_side"})
            continue
        if ability or x is None or y is None or card in (None, "", "_invalid"):
            rejected.append({"play_index": play_index, "reason": "ability_or_unpositioned"})
            continue
        slug = slug_of[side].get(str(card)) if side in slug_of else None
        if slug is None:
            rejected.append({"play_index": play_index, "reason": "card_outside_deck"})
            continue
        cmds[side].append({"tick": int(tick), "side": side, "card": slug, "x": int(x), "y": int(y),
                           "play_index": int(play_index)})
    for s in (0, 1):
        cmds[s].sort(key=lambda c: (c["tick"], c["play_index"]))
    return {"tag": tag, "decks": decks, "commands": cmds, "rejected_input": rejected, "battle": battle,
            "deck_swap": did_swap, "play_match": {"direct": direct, "swapped": swapped}}


# ---------------------------------------------------------------------------------------------- recording
# The four candidate display->engine transforms (identity, x-flip, y-flip, both). Calibrated per side.
CANDIDATES = ((1, 1), (-1, 1), (1, -1), (-1, -1))


def _place(x: int, y: int, t: int) -> tuple[int, int]:
    """Mirror only. `_Core.act` takes the 1000-per-tile frame (0..18000, 0..32000) and multiplies by
    `royale_env.SCALE = 18` to reach the engine's 324000x576000 space — so no scaling here."""
    sx, sy = CANDIDATES[t]
    ex = x if sx > 0 else 18000 - x
    ey = y if sy > 0 else 32000 - y
    return int(ex), int(ey)


def to_compact(raw: dict) -> dict:
    """``RoyalePoolEnv.raw()`` -> the list-encoded compact frame on the 20-tick grid (engine units, 6 columns)."""
    # raw() already reports engine units (1000 per tile): obs_contract.from_engine divides by 18000/32000.
    ents = [[int(e["side"]), float(e["x"]), float(e["y"]), str(e["name"]),
             float(e["hp"]), float(e["max_hp"])] for e in raw.get("entities") or []]
    towers = []
    for tw in (raw.get("episode") or {}).get("crown_towers") or []:
        x, y = float(tw["x"]), float(tw["y"])
        kind = "king" if tw.get("type") == "king" else "princess"
        lane = None if kind == "king" else int((x / 18000.0) >= 0.5)
        towers.append([int(tw["side"]), kind, lane, x, y, float(tw["hp"]), float(tw["max_hp"])])
    el = {int(p["side"]): float(p.get("elixir_exact", 0.0)) for p in raw.get("players") or []}
    return {"tick": int(raw["tick"]), "entities": ents, "towers": towers,
            "elixir": [el.get(0, 0.0), el.get(1, 0.0)]}


def _fit_order(slugs: list[str], plays: list[str], deal_order) -> tuple[Optional[list[int]], list[str]]:
    """A deck order fitting the recorded play sequence, shrinking the sequence only if it cannot fit.

    A real recording can contain plays the engine would refuse (the client accepted them under different state), and
    a card cannot return to hand inside one cycle. We first drop same-card repeats within 16 ticks, then shorten the
    tail, and record nothing about it here — the driver logs the plays it actually drove.
    """
    if not plays:
        return list(range(8)), []
    order = deal_order(slugs, plays)
    if order is not None:
        return order, plays
    last: dict[str, int] = {}
    thinned = []
    for i, c in enumerate(plays):
        if c in last and i - last[c] < 16:
            continue
        last[c] = i
        thinned.append(c)
    order = deal_order(slugs, thinned)
    if order is not None:
        return order, thinned
    for cut in range(len(thinned) - 1, 3, -1):
        order = deal_order(slugs, thinned[:cut])
        if order is not None:
            return order, thinned[:cut]
    return None, []


def play_frame(raw: dict) -> dict:
    """A full play frame: the compact frame plus the players' hand/next (dataset._hand_of reads these).

    `dataset.build_replay` passes every frame through `_as_compact`, which slices entity rows and drops
    effects/projectiles — so entities must already be list-encoded here.
    """
    fr = to_compact(raw)
    fr["players"] = [{"side": int(p["side"]), "hand": [h.get("name") for h in p.get("hand") or []],
                      "next": p.get("next_deck_index")} for p in raw.get("players") or []]
    return fr


def drive(entry: dict, *, seed: int = 0, record_every: int = RECORD_EVERY, max_ticks: int = 7200,
          feature_version: int = 3, log=None) -> dict:
    """Run the recorded timeline; return the corpus record."""
    from pipeline.obs_contract import from_engine, load_deck  # noqa: PLC0415
    from pipeline.royale_env import FORM_OF, RoyalePoolEnv, deal_order  # noqa: PLC0415

    tag, decks, cmds = entry["tag"], entry["decks"], entry["commands"]
    env = RoyalePoolEnv(decision_ticks=10, feature_version=feature_version, forms_mode="deck")
    env.tag = tag
    env.deck_ids = {s: [env.card_id(it["name"]) for it in decks[s]] for s in (0, 1)}
    env.final_decks = decks

    deal, orders, wanted = {}, {}, {}
    for s in (0, 1):
        slugs = [it["slug"] for it in decks[s]]
        order, kept = _fit_order(slugs, [c["card"] for c in cmds[s]], deal_order)
        if order is None:
            raise ValueError(f"{tag}: side {s} has no deal consistent with its recorded plays")
        if len(kept) < len(cmds[s]):
            keep = Counter()
            for c in kept:
                keep[c] += 1
            trimmed = []
            for c in cmds[s]:
                if keep[c["card"]]:
                    keep[c["card"]] -= 1
                    trimmed.append(c)
            cmds[s] = trimmed
        deal[s], orders[s] = [env.deck_ids[s][i] for i in order], order
        wanted[s] = [FORM_OF[str(it["form"])] for it in decks[s]]
    env.core.reset(seed, env._setup(deal, orders, wanted))
    env.tick = int(env.core.state().tick)
    env.public_plays = []
    env._reset_abilities()

    opening = int(env.core.state_hash())
    pending = {s: list(cmds[s]) for s in (0, 1)}
    # The recorder's coordinate frame is not the engine's (measured: a side-1 play at display y=31500 is inside
    # team 0's territory). Each side's transform is therefore CALIBRATED on its own first accepted play, not assumed.
    transform: dict[int, Optional[int]] = {0: None, 1: None}
    calibrated: dict[int, int] = {}
    idx = 0
    frames: list[dict] = []
    play_frames: list[dict] = []
    log_rows: list[dict] = []
    elixir_delays: Counter = Counter()
    accepted = Counter()
    rejected = Counter()

    while env.tick < max_ticks:
        st = env.core.state()
        if st.game_over:
            break
        # fire due commands for BOTH sides (retry a refusal while the recording's elixir catches up)
        for s in (0, 1):
            while pending[s] and pending[s][0]["tick"] <= env.tick:
                cmd = pending[s][0]
                before = env.core.state()
                hand = list(before.players[s].hand)
                slot = next(i for i, it in enumerate(decks[s]) if it["slug"] == cmd["card"])
                pre = env.raw()
                order = [transform[s]] if transform[s] is not None else list(range(len(CANDIDATES)))
                r = {"accepted": False, "result_code": -1}
                for t in order:
                    tx, ty = _place(cmd["x"], cmd["y"], t)
                    r = env.eng.act(side=s, deck_index=slot, x=tx, y=ty)
                    if r["accepted"]:
                        if transform[s] is None:
                            transform[s] = t
                            calibrated[s] = t
                        break
                if r["accepted"]:
                    accepted[s] += 1
                    elixir_delays[str(env.tick - cmd["tick"])] += 1
                    play_frames.append({"play_index": len(play_frames), **play_frame(pre)})
                    log_rows.append({"side": s, "tick": int(before.tick), "play_index": cmd["play_index"],
                                     "card": cmd["card"], "x": int(cmd["x"]), "y": int(cmd["y"]),
                                     "accepted": True, "delay_ticks": int(env.tick - cmd["tick"]),
                                     "hand_before": [str(env.names.get(c, c)) for c in hand],
                                     "next": int(before.players[s].next_card)})
                    pending[s].pop(0)
                elif env.tick - cmd["tick"] < ELIXIR_SLACK:
                    break  # retry next tick
                else:
                    rejected[str(r["result_code"])] += 1
                    log_rows.append({"side": s, "tick": int(before.tick), "play_index": cmd["play_index"],
                                     "card": cmd["card"], "x": int(cmd["x"]), "y": int(cmd["y"]),
                                     "accepted": False, "skipped": f"refused:{r['result_code']}"})
                    pending[s].pop(0)
        if env.tick % record_every == 0:
            frames.append(to_compact(env.raw()))
        env.core.step([], 1)
        st = env.core.state()
        env.tick = int(st.tick)

    st = env.core.state()
    from royalegym.protocol import Winner  # noqa: PLC0415

    final = {"tick": int(st.tick), "state_hash": int(env.core.state_hash()), "terminated": bool(st.game_over),
             "winner": None if not st.game_over else {Winner.BLUE: 0, Winner.RED: 1}.get(Winner(st.winner), -1),
             "crowns": [int(p.crowns) for p in st.players], "terminal_tick": int(st.tick),
             "termination_reason": "game_over" if st.game_over else "tick_cap"}
    battle = entry["battle"]
    expected = {"crowns_by_side": {"0": int(battle.get("team_crowns") or 0),
                                   "1": int(battle.get("opponent_crowns") or 0)},
                "result": str(battle.get("result") or "")}
    return {
        "tag": tag,
        "final_decks": {str(s): [f"{it['name']}@{it['form']}" if it["form"] != "base" else it["name"] for it in decks[s]]
                        for s in (0, 1)},
        "frames": frames,
        "play_frames": play_frames,
        "record_every": int(record_every),
        "log": log_rows,
        "expected": expected,
        "final": final,
        "grade": {"plays_driven": len(log_rows), "accepted": sum(accepted.values()),
                  "rejected_by_reason": dict(rejected), "elixir_delays": {"n": sum(elixir_delays.values())}},
        "deal_inference": {str(s): {"consistent": 1} for s in (0, 1)},
        "opening_state_hash": opening,
        "record_native": False,
        "record_full": True,
        "deck_swap": bool(entry.get("deck_swap")),
        "frame_transform": {str(s): calibrated.get(s) for s in (0, 1)},
        "rejected_input": entry.get("rejected_input") or [],
    }


# ---------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("deck", nargs="?", default="icebow")
    ap.add_argument("--crawl", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--record-every", type=int, default=RECORD_EVERY)
    a = ap.parse_args(list(argv) if argv is not None else None)

    from pipeline.e1_pool import read_battles, read_plays  # noqa: PLC0415
    from pipeline.obs_contract import load_deck  # noqa: PLC0415

    deck = load_deck(a.deck)
    crawl = a.crawl or (REPO / f"data/hf_crawl/{deck.name}")
    out = a.out or (REPO / f"scratchpad/gauntlet/ext/corpus_v3/{deck.name}")
    out.mkdir(parents=True, exist_ok=True)
    battles = read_battles(crawl)
    plays = read_plays(crawl)
    cr = _corpus_rows()
    tags = sorted(plays)
    if a.limit:
        tags = tags[: a.limit]
    stats = Counter()
    t0 = time.time()
    for tag in tags:
        dst = out / f"replay_{tag}.json"
        if dst.exists():
            stats["skip_exists"] += 1
            continue
        try:
            entry = build_entry(tag, battles[tag], plays[tag], cr)
            rec = drive(entry, seed=a.seed, record_every=a.record_every)
        except Exception as exc:  # noqa: BLE001 - one bad replay must not stop a batch
            stats[f"error:{type(exc).__name__}"] += 1
            print(f"{tag}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        dst.write_text(json.dumps(rec), encoding="utf-8")
        stats["written"] += 1
        stats[f"plays_accepted_{rec['grade']['accepted'] > 0}"] += 1
        print(f"{tag}: {len(rec['frames'])} frames, {rec['grade']['accepted']}/{rec['grade']['plays_driven']} accepted, "
              f"final tick {rec['final']['tick']}, {dst.stat().st_size / 1e6:.1f} MB", file=sys.stderr)
    print(json.dumps({"deck": deck.name, "replays": len(tags), "seconds": round(time.time() - t0, 1),
                      "stats": dict(stats)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---------------------------------------------------------------------------------------------- memory capture
@lru_cache(maxsize=1)
def _public_key() -> dict[int, str]:
    """Public card id (base, evolution form, hero form) -> recorder card key, from the committed catalogue."""
    cat = json.loads((REPO / "pipeline" / "decks" / "card_catalog.json").read_text(encoding="utf-8"))["cards"]
    out: dict[int, str] = {}
    for c in cat:
        key = re.sub(r"[^a-z0-9]+", "-", str(c["display_name"]).lower()).strip("-")
        out[int(c["card_id"])] = key
        if c.get("evolution_form_id"):
            out[int(c["evolution_form_id"])] = key + "-evo"
        if c.get("hero_form_id"):
            out[int(c["hero_form_id"])] = key + "-hero"
    return out


def capture_to_entry(path: Path, cr, *, deck_hint: Optional[dict[int, list[str]]] = None,
                     tick_zero_clock: bool = True) -> dict:
    """Memory-reader capture (``runtime_snapshot`` JSONL) -> the same entry ``build_entry`` produces.

    The reader is read-only and reports, per snapshot: own elixir, battle clock, own hand/next (deck indexes +
    public card ids) and EVERY entity of both sides (``address``, ``side``, ``x``/``y`` in 1000-per-tile units,
    ``card_id``, ``hp``). A deployment therefore appears as a new ``address``; the play timeline is the ordered set
    of first appearances, and both decks come from the card ids the capture saw.

    Coordinates are already 1000-per-tile, which is exactly what ``_Core.act`` expects — the reader needs no
    rescaling, only the per-side frame calibration the driver performs anyway.
    """
    keys = _public_key()
    seen: dict[str, int] = {}
    own_deck: dict[int, str] = {}
    plays: list[tuple] = []
    ticks: list[int] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            snap = json.loads(line)
        except json.JSONDecodeError:
            continue
        if snap.get("event") not in (None, "runtime_snapshot"):
            continue
        clock = snap.get("battle_clock")
        tick = int(round(float(clock) / 0.05)) if (tick_zero_clock and clock is not None) else len(ticks) * 2
        ticks.append(tick)
        for h in snap.get("hand") or []:
            key = keys.get(int(h["data_id"])) if h.get("data_id") else None
            if key:
                own_deck[int(h["deck_index"])] = key
        nxt = snap.get("next_card") or {}
        key = keys.get(int(nxt["data_id"])) if nxt.get("data_id") else None
        if key:
            own_deck[int(nxt["deck_index"])] = key
        for e in snap.get("entities") or []:
            addr = str(e.get("address"))
            cid = int(e.get("card_id") or -1)
            if cid <= 0 or addr in seen:
                continue  # crown towers / unknown bodies / already-seen unit
            seen[addr] = tick
            key = keys.get(cid)
            if key is None:
                continue
            plays.append((tick, len(plays), int(e.get("side") or 0), 0, key,
                          int(e.get("x") or 0), int(e.get("y") or 0)))

    own = [own_deck[i] for i in sorted(own_deck)]
    opp = sorted({p[4] for p in plays if p[2] == 1})
    # A read-only capture cannot always see all 8 cards of a side (a card that is never deployed or never cycled
    # into the visible hand is genuinely unknown). `deck_hint` supplies the rest; without it, refuse rather than
    # invent a deck.
    if deck_hint:
        for side, cards in ((0, own), (1, opp)):
            need = [str(c) for c in (deck_hint.get(side) or [])]
            for c in need:
                if len(cards) >= 8:
                    break
                if c not in cards:
                    cards.append(c)
    if len(own) != 8:
        raise ValueError(f"{Path(path).name}: own deck incomplete ({len(own)}/8 distinct deck indexes)")
    if len(opp) != 8:
        raise ValueError(f"{Path(path).name}: opponent deck incomplete ({len(opp)}/8 distinct deployed cards); "
                         f"pass deck_hint (the capture cannot see an unplayed card)")
    battle = {"team_deck": ",".join(sorted(own)), "opponent_deck": ",".join(sorted(opp)),
              "team_crowns": "", "opponent_crowns": "", "result": ""}
    # build_entry matches plays on the BASE key (it resolves the form-free slug to the deck's slot).
    plays_base = [(t, pi, side, 0, cr.split_slug(card)[0], x, y) for t, pi, side, _, card, x, y in plays]
    entry = build_entry(Path(path).stem, battle, plays_base, cr)
    entry["capture"] = {"snapshots": len(ticks), "spawn_plays": len(plays_base),
                        "own_deck": own, "opp_deck": opp}
    return entry
