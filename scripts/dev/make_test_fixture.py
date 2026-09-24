#!/usr/bin/env python3
"""Build tests/fixtures/dataset_subset from the released dataset.

Keeps only what the report and `from-released` need for the seven paired
competitions: slim action rows, the paired trajectory-index rows, and the
state/action rows of two trajectories. The per-competition reference computed
from this subset equals the one computed from the full dataset.

    python scripts/dev/make_test_fixture.py [--dataset-dir DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from traceml_toolkit import assets, config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "fixtures" / "dataset_subset"
ACTION_COLS = ["key_id", "comp", "group", "is_agent", "is_best_branch", "v_old", "v_new",
               "coarse_actions", "intents"]
# One human and one Codex trajectory on commonlit, for from-released tests.
KEEP_TRAJECTORIES = 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir")
    a = ap.parse_args()
    action = pd.read_parquet(assets.resolve_file("data/paired/action.parquet", a.dataset_dir))
    state = pd.read_parquet(assets.resolve_file("data/paired/state.parquet", a.dataset_dir))
    idx = pd.read_parquet(assets.resolve_file(assets.TRAJECTORY_INDEX, a.dataset_dir))

    idx = idx[idx["comp"].isin(config.PAIRED_COMPETITIONS)].reset_index(drop=True)
    slim = action[ACTION_COLS].reset_index(drop=True)

    cl = idx[idx["comp"] == "commonlitreadabilityprize"]
    human = cl[~cl["is_agent"].fillna(False).astype(bool)].sort_values("n_versions").iloc[len(cl) // 4]["key_id"]
    agent = cl[cl["group"] == "codex"].sort_values("n_versions").iloc[0]["key_id"]
    keep = {str(human), str(agent)}
    st_keep = state[state["key_id"].astype(str).isin(keep)].reset_index(drop=True)
    ac_keep = action[action["key_id"].astype(str).isin(keep)].reset_index(drop=True)

    (OUT / "data" / "paired").mkdir(parents=True, exist_ok=True)
    (OUT / "extras").mkdir(parents=True, exist_ok=True)
    # action.parquet: slim rows for all paired transitions plus full rows of the kept trajectories
    ac_full_cols = ac_keep.reindex(columns=action.columns)
    rest = slim[~slim["key_id"].astype(str).isin(keep)].reindex(columns=action.columns)
    pd.concat([rest, ac_full_cols], ignore_index=True).to_parquet(OUT / "data/paired/action.parquet", index=False)
    st_keep.to_parquet(OUT / "data/paired/state.parquet", index=False)
    idx.to_parquet(OUT / "extras/trajectory_index.parquet", index=False)
    (OUT / "KEY_IDS.txt").write_text(f"human {human}\ncodex {agent}\n")
    for p in sorted(OUT.rglob("*.parquet")):
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1024:.0f} KB")
    print(f"human={human} codex={agent}")


if __name__ == "__main__":
    main()
