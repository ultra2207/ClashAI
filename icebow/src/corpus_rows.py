"""crawl battle row -> the 8-card deck of one side, in the pipeline's canonical item form.

This is the module `pipeline.e1_pool` expects from `icebow/src` (`_rd()`, `build_row`), and the only producer of
the `canon` deck lists the pool build needs. It was absent from the snapshot, so it is reconstructed here.

What it must return, per item: ``{"slug", "name", "card_id", "form", "cost"}`` where

* ``name`` is the ENGINE display name (``Knight``, ``Xbow``, ``AngryBarbarians``) — the same spelling as
  ``corpus_v3.final_decks`` and ``royale_env.RoyalePoolEnv.names``. Forms are expressed as the ``@evolution`` /
  ``@hero`` suffix that ``e1_pool.deck_names`` expects.
* ``slug`` is the recorder's card key verbatim (``knight``, ``knight-ev1``) because crawl plays carry that key and
  ``obs_contract.deal_order`` matches plays to deck slots by it.
* ``form`` is one of ``base`` / ``evolution`` / ``hero`` (``e1_pool.final_forms`` vocabulary).

Column convention (matches ``hf_to_crawl``: ``attr_s`` team->red->side 0, opponent->blue->side 1):
side 0 reads ``team_deck``, side 1 reads ``opponent_deck``.

The engine card table is read from RoyaleSim's embedded cards JSON, so name / elixir / id always agree with the
engine that will replay the match. Nothing here touches the runtime's pinned bytes beyond importing ``royalesim``.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Optional

_SLUG_FORM = re.compile(r"-(ev|evo|hero)\d*$")


_VARIANTS = ("Super", "MergeMaiden", "SkeletonWarriors_Spooky", "RoyalRecruits_Chess", "TriWizards", "PrinceBuff")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@lru_cache(maxsize=1)
def _cards() -> tuple[dict[str, dict], dict[str, dict]]:
    """(by normalized display name, by vocab key) -> {"name": engine display name, "cost": elixir}.

    The recorder's key is the API/human card name (``elite-barbarians``, ``hog-rider``, ``the-log``), so the display
    name is the authoritative index. RoyaleSim also carries internal variants that alias onto the same card
    (``SuperKnight`` -> knight, ``GlobalLightning`` -> lightning, ``HogRider`` vs ``SuperHogRiderTerry``); they are
    excluded so a real card can never be resolved to a variant.
    """
    import royalesim

    from pipeline import vocab

    table = json.loads(royalesim.EMBEDDED_CARDS_JSON)
    by_display: dict[str, dict] = {}
    by_key: dict[str, dict] = {}
    # Two passes: real cards claim their keys first, then internal variants (MergeMaiden -> spirit_empress, ...)
    # fill whatever the catalogue does not expose under its own name.
    for pass_no in (0, 1):
        for c in table["cards"]:
            name = str(c["name"])
            variant = name.startswith(_VARIANTS) or "_" in name
            if variant != bool(pass_no):
                continue
            item = {"name": name, "cost": int(c.get("elixir") or 0)}
            for label in (c.get("display_name"), name):
                if label:
                    by_display.setdefault(_norm(label), item)
            key = vocab.engine_key(name)
            if key:
                by_key.setdefault(key, item)
    return by_display, by_key



@lru_cache(maxsize=1)
def _ids() -> dict[str, int]:
    """engine display name -> engine card id (what ``RoyalePoolEnv.card_id`` resolves against)."""
    from royalegym.rust_engine import RustEngine

    return {c.name: int(c.card_id) for c in RustEngine().cards()}


def split_slug(slug: str) -> tuple[str, str]:
    """``knight-ev1`` -> ('knight', 'evolution'); ``knight`` -> ('knight', 'base')."""
    m = _SLUG_FORM.search(slug)
    if not m:
        return slug, "base"
    # The recorder writes -ev1 / -hero1; the pipeline's vocabulary (and `dataset.crawl_slot`) speaks
    # `_evo` / `_hero` after hyphen->underscore, so forms are canonicalised to -evo / -hero here.
    return slug[: m.start()], "evolution" if m.group(1).startswith("ev") else "hero"


def item_for_slug(slug: str) -> dict:
    """One recorder card key -> a pipeline deck item. Raises KeyError when the engine has no such card."""
    base, form = split_slug(slug)
    by_display, by_key = _cards()
    info = by_display.get(_norm(base)) or by_key.get(base.replace("-", "_"))
    if info is None:
        raise KeyError(slug)
    name = info["name"]
    canon = base + ("-evo" if form == "evolution" else "-hero" if form == "hero" else "")
    return {"slug": canon, "name": name, "card_id": _ids().get(name, 0), "form": form, "cost": info["cost"]}


def deck_for_side(battle: dict, side: int) -> list[dict]:
    """The 8-card deck of ``side`` from a ``battles.csv`` row (or any dict with the same columns).

    Raises KeyError (missing column, unknown card) or SystemExit-free ValueError (wrong size) — ``build_row``
    already catches ``(SystemExit, KeyError)`` and counts the refusal as ``deck_unresolvable``.
    """
    col = "team_deck" if int(side) == 0 else "opponent_deck"
    raw = (battle.get(col) or "").strip()
    if not raw:
        raise KeyError(col)
    slugs = [s.strip() for s in raw.split(",") if s.strip()]
    if len(slugs) != 8 or len(set(slugs)) != 8:
        raise ValueError(f"{col}: expected 8 distinct cards, got {slugs}")
    return [item_for_slug(s) for s in slugs]


def deck_key(slugs: list[str]) -> str:
    """Stable one-line identity for a deck (base keys, form-free), for grouping/debug output."""
    return "|".join(sorted(split_slug(s)[0] for s in slugs))


def next_deck_index(deck: list[dict], name: Optional[str]) -> Optional[int]:
    """Deck index of a card named by the engine, ignoring its form suffix (``Xbow``, ``Knight@evolution``)."""
    if not name:
        return None
    base = str(name).split("@", 1)[0]
    for i, it in enumerate(deck):
        if it["name"] == base:
            return i
    return None
