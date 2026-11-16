"""Pick a deck-stratified replay selection for the GENERALIST corpus from the HF IL_Replay parquet.

Writes a ``--tags-file`` for ``tools/hf_to_crawl.py --tags-file`` (multi-deck mode, both sides, no deck filter),
plus a report of what the 1.8 GB can actually supply per deck.

A replay is a candidate when, from the parquet alone (no engine):
  * it is 1v1 -- one player with an 8-card deck per side (``hf_to_crawl.one_v_one``'s rule);
  * every card on BOTH sides resolves through ``corpus_rows.item_for_slug`` (else ``deck_for_side`` KeyErrors);
  * (opt-in, ``--use-blockers``) no card is one of the census's ``loadable_decks.json`` ``top_blockers``. OFF by
    default because that list does NOT describe ``corpus_driver``: measured 2026-11-15, 60/60 replays whose decks
    carry the top blockers in base AND evolution/hero form drove cleanly (40 base-form + 20 evo-form, 0 failures),
    and the census list alone would have discarded 75% of the dataset (189,549 of 252,238 replays);
  * every ``play_card`` row in ``actions/`` is positioned (``native_x``/``native_y`` non-null) and there are >= 8
    of them, mirroring ``hf_to_crawl.positioned_plays`` -- the replay driver cannot place an unpositioned play;
  * ``battle_type`` is in ``--modes`` (default ``pathOfLegend``, the league's Ranked setting).

Selection is then stratified by the team deck (8 base slugs): decks are visited round-robin in weight order
(``sides ** alpha`` over the candidate volume, the league's own ``league_deck_alpha``) and each contributes at most
``--per-deck-cap`` replays, so one common deck cannot fill the budget.

usage (from the repo root):
  uv run python tools\\hf_gen_select.py --budget 40000 --out scratchpad/gauntlet/gen_v1/tags.json
  ... --parts-limit 2 --report-only            # slice first (repo profiling rule), prints the report only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (REPO, REPO / "icebow" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from pipeline import vocab  # noqa: E402

DEFAULT_HF = REPO / "data" / "hf"
DEFAULT_DECKS = REPO / "scratchpad" / "gauntlet" / "L68" / "selfplay" / "loadable_decks.json"
DEFAULT_MODES = "pathOfLegend"


def hyphen(key: str) -> str:
    """Pipeline/vocab key (``x_bow``) -> RoyaleAPI base slug (``x-bow``)."""
    return vocab.base_key(str(key)).replace("_", "-")


def _corpus_rows():
    import corpus_rows  # noqa: PLC0415  (icebow/src is on sys.path)

    return corpus_rows


def blocker_slugs(path: Path) -> tuple[set[str], Counter]:
    """Census blocker engine names -> base slugs, with the census count kept for the report."""
    deck_doc = json.loads(path.read_text(encoding="utf-8"))
    out: Counter = Counter()
    for name, n in deck_doc.get("top_blockers") or []:
        key = vocab.engine_key(name)
        if key:
            out[hyphen(key)] += int(n)
    return set(out), out


def side_decks(payload: str, cr, stats: Counter) -> dict[str, list[str]] | None:
    """``payload_json`` -> {'team': [8 card_keys], 'opponent': [...]} for a drivable 1v1 replay, else None."""
    try:
        battle = json.loads(payload)["battle"]
    except (json.JSONDecodeError, KeyError, TypeError):
        stats["skip_bad_payload"] += 1
        return None
    decks: dict[str, list[str]] = {}
    for side in ("team", "opponent"):
        try:
            players = battle[side]["players"]
        except (KeyError, TypeError):
            stats["skip_no_side"] += 1
            return None
        if len(players) != 1:
            stats["skip_not_1v1"] += 1
            return None
        keys = [str(c.get("card_key") or "") for c in (players[0].get("deck") or [])]
        if len(keys) != 8 or len(set(keys)) != 8 or not all(keys):
            stats["skip_deck_size"] += 1
            return None
        decks[side] = keys
    return decks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--hf", type=Path, default=DEFAULT_HF, help="dataset root holding replays/ and actions/")
    ap.add_argument("--decks", type=Path, default=DEFAULT_DECKS, help="loadable_decks.json (16 Deck-census)")
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad" / "gauntlet" / "gen_v1" / "tags.json")
    ap.add_argument("--budget", type=int, default=40_000, help="max replays to select (0 = every candidate)")
    ap.add_argument("--per-deck-cap", type=int, default=0, help="max replays per team deck (0 = budget/400)")
    ap.add_argument("--alpha", type=float, default=0.5, help="deck weight = candidate_volume ** alpha (default 0.5)")
    ap.add_argument("--modes", default=DEFAULT_MODES, help="comma list of battle_type values to keep ('' = all)")
    ap.add_argument("--min-plays", type=int, default=8, help="drop replays with fewer positioned play_card rows")
    ap.add_argument("--use-blockers", action="store_true",
                    help="also drop replays carrying a census top_blocker card (OFF: the census list does not "
                         "describe corpus_driver -- measured 60/60 blocker decks drove cleanly)")
    ap.add_argument("--parts-limit", type=int, default=0, help="only the first N parquet parts (profiling slice)")
    ap.add_argument("--report-only", action="store_true", help="print the report; write no tags file")
    ap.add_argument("--seed", type=int, default=0, help="selection tie-break seed (deterministic order)")
    a = ap.parse_args(argv)

    t0 = time.time()
    cr = _corpus_rows()
    blockers, blocker_counts = blocker_slugs(a.decks)
    if not a.use_blockers:
        blockers = set()
    deck_doc = json.loads(a.decks.read_text(encoding="utf-8"))
    loadable = [tuple(sorted(d["slugs"])) for d in deck_doc.get("decks") or []]
    keep_modes = {m.strip() for m in str(a.modes).split(",") if m.strip()}

    replay_parts = sorted((a.hf / "replays").glob("part-*.parquet"))
    action_parts = sorted((a.hf / "actions").glob("part-*.parquet"))
    if a.parts_limit:
        replay_parts, action_parts = replay_parts[: a.parts_limit], action_parts[: a.parts_limit]
    if not replay_parts or not action_parts:
        raise SystemExit(f"no parquet parts under {a.hf} -- run tools/hf_download.py first")

    # ---- pass B first: `actions/` is small and tells us which replays the driver can actually replay.
    stats: Counter = Counter()
    modes: Counter = Counter()
    usable: dict[str, tuple[int, str]] = {}          # tag -> (positioned play_card rows, kind of reject)
    for part in action_parts:
        df = pl.scan_parquet(part).select(["replay_tag", "kind", "native_x", "native_y"]).collect()
        plays = df.filter(pl.col("kind") == "play_card")
        bad = plays.filter(pl.col("native_x").is_null() | pl.col("native_y").is_null()).group_by("replay_tag").len()
        bad_tags = set(bad["replay_tag"].to_list())
        for tag, n in plays.group_by("replay_tag").len().iter_rows():
            usable[tag] = (int(n), "unpositioned" if tag in bad_tags else "")
        stats["action_parts"] += 1
        print(f"actions {part.name}: {len(usable)} replays seen", flush=True)

    # ---- pass A: `replays/` decks + mode filter, kept only where the actions say the replay is drivable.
    by_deck: dict[tuple[str, ...], list[str]] = defaultdict(list)     # team deck -> candidate tags
    opp_decks: Counter = Counter()
    for part in replay_parts:
        df = pl.read_parquet(part, columns=["replay_tag", "payload_json", "battle_type"])
        for tag, payload, mode in zip(df["replay_tag"], df["payload_json"], df["battle_type"]):
            stats["replays_scanned"] += 1
            if keep_modes and str(mode) not in keep_modes:
                stats["skip_mode"] += 1
                continue
            hit = usable.get(tag)
            if hit is None:
                stats["skip_no_actions"] += 1
                continue
            n_plays, why = hit
            if why:
                stats[f"skip_{why}"] += 1
                continue
            if n_plays < a.min_plays:
                stats["skip_few_plays"] += 1
                continue
            decks = side_decks(payload, cr, stats)
            if decks is None:
                continue
            slugs: dict[str, tuple[str, ...]] = {}
            ok = True
            for side, keys in decks.items():
                bases = []
                for k in keys:
                    try:
                        base, _form = cr.split_slug(k)
                        cr.item_for_slug(k)                       # KeyError -> the engine has no such card
                    except (KeyError, ValueError):
                        stats["skip_unknown_card"] += 1
                        ok = False
                        break
                    if base in blockers:
                        stats["skip_blocker"] += 1
                        ok = False
                        break
                    bases.append(base)
                if not ok:
                    break
                slugs[side] = tuple(sorted(bases))
            if not ok:
                continue
            key = slugs["team"]
            by_deck[key].append(tag)
            opp_decks[slugs["opponent"]] += 1
            modes[str(mode)] += 1
            stats["candidates"] += 1
        print(f"replays {part.name}: candidates {stats['candidates']}", flush=True)

    # ---- stratified selection: round-robin over decks in weight order, capped per deck.
    budget = a.budget if a.budget > 0 else stats["candidates"]
    cap = a.per_deck_cap or max(1, budget // 400)
    order = sorted(by_deck, key=lambda k: (-(len(by_deck[k]) ** a.alpha), k))
    for k in order:
        by_deck[k].sort()
    picked: list[str] = []
    idx = {k: 0 for k in order}
    while len(picked) < budget:
        progressed = False
        for k in order:
            if len(picked) >= budget:
                break
            i = idx[k]
            tags = by_deck[k]
            if i >= len(tags) or i >= cap:
                continue
            picked.append(tags[i])
            idx[k] = i + 1
            progressed = True
        if not progressed:
            break

    picked_set = set(picked)
    covered = {k for k in by_deck if any(t in picked_set for t in by_deck[k][: idx[k]])}
    loadable_hit = sum(1 for k in covered if k in set(loadable))
    rep = {
        "hf": str(a.hf), "parts": [len(replay_parts), len(action_parts)], "seconds": round(time.time() - t0, 1),
        "stats": dict(stats), "modes": dict(modes.most_common()),
        "decks_seen": len(by_deck), "decks_picked": len(covered), "distinct_opponent_decks": len(opp_decks),
        "budget": budget, "per_deck_cap": cap, "alpha": a.alpha, "picked": len(picked),
        "loadable_decks_in_census": len(loadable), "loadable_decks_covered": loadable_hit,
        "top_decks": [[list(k)[:3], len(v)] for k, v in sorted(by_deck.items(), key=lambda kv: -len(kv[1]))[:10]],
        "blockers": dict(blocker_counts.most_common(6)),
    }
    print(json.dumps(rep, indent=1), flush=True)

    if not a.report_only and picked:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        payload = [{"tag": t} for t in picked]
        a.out.write_text(json.dumps(payload), encoding="utf-8")
        (a.out.parent / "select_report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"wrote {len(payload)} tags to {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
