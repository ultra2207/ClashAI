# tools/

Repo-level command-line tools for the pro-replay (HuggingFace) data path. Run every command from the repo root
with the deck venv's Python (e.g. `icebow\.venv\Scripts\python.exe`). No paths are hard-coded: each tool finds the
repo from its own location. Default outputs go under the repo-root `data/` folder, which `.gitignore` excludes.

| Tool | What it does | Usage |
|---|---|---|
| `hf_download.py` | Downloads + sha256-checks the parquet parts listed in `hf_manifest.json`; resumable, stdlib only. The manifest is `replays/` (52 parts, 0.83 GB, 252,238 replays) + `actions/` (52 parts, 1.06 GB, 17.8M events). | `python tools\hf_download.py [--prefix actions/] [--dry-run] [--dest data\hf]` |
| `hf_manifest.json` | File list (path, bytes, sha256) of the `VanguardX101/IL_Replay` dataset, read by `hf_download.py`. | -- |
| `hf_to_crawl.py` | Converts the parquet parts into a crawl2-shaped folder (battles.csv, plays_ext.csv, tags.json, dedupe_report.json) for one deck; dedupes against the deck's own crawl if present. | `python tools\hf_to_crawl.py icebow [--hf data\hf\replays] [--out data\hf_crawl\icebow]` |
| `hf_gen_select.py` | Picks a deck-stratified replay selection for the GENERALIST corpus and writes a `--tags-file` for `hf_to_crawl.py --tags-file` (multi-deck, both sides). Drops only replays the driver cannot replay (not 1v1, unpositioned plays, unknown card). Measured: 207,034 of 252,238 replays are drivable; the `loadable_decks.json` blocker list does NOT describe `corpus_driver` (60/60 blocker decks drove cleanly) so it is opt-in. | `python tools\hf_gen_select.py --budget 60000 --out scratchpad\gauntlet\gen_v1\tags.json` |
| `merge_aug.py` | Merges a clean S1 dataset with its degraded twin's train rows (val stays clean). | `python tools\merge_aug.py --clean <npz> --degraded <npz> --out <npz>` |

Offline tests (no network, no engine): `icebow\.venv\Scripts\python.exe -m unittest tools.tests.test_tools`
