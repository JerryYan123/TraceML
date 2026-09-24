"""Reference cohorts from the released paired split.

Ported from the paper's ``figure_code/loader.py`` with one correction.
Humans are bucketed per competition into top10 / top10-40 / top40-70 /
top70-100 by the percentile rank of their **best** leaderboard score. The
original ranked every human by ``max_score``, which on lower-is-better metrics
(RMSE, KL divergence) is the human's *worst* submission; the toolkit now uses
``min_score`` for those competitions. Agents map to ``codex``, ``mlev_best``
(the best branch of an MLEvolve search) and ``mlev_normal``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from traceml_toolkit import assets, config

_ACTION_JSON_COLS = ["coarse_actions", "fine_actions", "intents"]


def _parse_json(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return x


def human_best_score(idx: pd.DataFrame) -> pd.Series:
    """Each trajectory's best score: min_score if its competition is lower-is-better."""
    comps = config.competitions()
    lower = idx["comp"].map(lambda c: comps.get(c, {}).get("score_direction") == "lower").astype(bool)
    return idx["min_score"].where(lower, idx["max_score"])


def human_buckets(idx: pd.DataFrame) -> dict[str, str]:
    """key_id -> human cohort, by within-competition percentile of the best score (cuts .9/.6/.3)."""
    comps = config.competitions()
    is_human = ~idx["is_agent"].fillna(False).astype(bool)
    h = idx.loc[is_human].assign(best=human_best_score(idx.loc[is_human])).dropna(subset=["best"])
    out: dict[str, str] = {}
    for comp, g in h.groupby("comp"):
        higher_better = comps.get(comp, {}).get("score_direction", "higher") == "higher"
        pct = g["best"].rank(pct=True, ascending=higher_better)
        for kid, p in zip(g["key_id"].astype(str), pct):
            out[kid] = ("top10" if p >= 0.90 else
                        "top10-40" if p >= 0.60 else
                        "top40-70" if p >= 0.30 else
                        "top70-100")
    return out


def assign_group7(df: pd.DataFrame, kid2bucket: dict[str, str]) -> pd.Series:
    is_agent = df["is_agent"].fillna(False).astype(bool)
    group = df["group"].astype(object)
    best = (df["is_best_branch"].fillna(False).astype(bool) if "is_best_branch" in df
            else pd.Series(False, index=df.index))
    g = pd.Series(pd.NA, index=df.index, dtype=object)
    g[~is_agent] = df.loc[~is_agent, "key_id"].astype(str).map(kid2bucket)
    g[is_agent & (group == "codex")] = "codex"
    g[is_agent & (group == "mlevolve") & best] = "mlev_best"
    g[is_agent & (group == "mlevolve") & ~best] = "mlev_normal"
    return g


@lru_cache(maxsize=4)
def _load_reference(action_path: str, index_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["key_id", "comp", "group", "is_agent", "is_best_branch", "coarse_actions", "intents"]
    action = pd.read_parquet(action_path, columns=cols)
    idx = pd.read_parquet(index_path)
    # Humans without author-tier metadata are dropped, as in the paper.
    nan_keys = set(idx.loc[idx["group"].isna(), "key_id"].astype(str))
    if nan_keys:
        idx = idx[~idx["key_id"].astype(str).isin(nan_keys)].reset_index(drop=True)
        action = action[~action["key_id"].astype(str).isin(nan_keys)].reset_index(drop=True)
    for c in ("coarse_actions", "intents"):
        action[c] = action[c].map(_parse_json)
    kid2b = human_buckets(idx)
    idx["group7"] = assign_group7(idx, kid2b)
    action["group7"] = assign_group7(action, kid2b)
    action["primary_intent"] = action["intents"].map(
        lambda xs: xs[0].get("intent", "unknown") if isinstance(xs, list) and xs and isinstance(xs[0], dict)
        else "unknown")
    for ca in config.COARSE_ACTIONS:
        action[f"has_{ca}"] = action["coarse_actions"].map(
            lambda xs, c=ca: int(c in xs) if isinstance(xs, list) else 0)
    return action, idx


def load_reference(dataset_dir: str | Path | None = None, download: bool = True
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(paired-split transitions with group7/primary_intent/has_<action>, trajectory index with group7)."""
    action_path = assets.resolve_file("data/paired/action.parquet", dataset_dir, download)
    index_path = assets.resolve_file(assets.TRAJECTORY_INDEX, dataset_dir, download)
    action, idx = _load_reference(str(action_path), str(index_path))
    return action.copy(), idx.copy()


def cohort_profiles(action: pd.DataFrame, comp: str | None) -> tuple[dict, bool]:
    """Per-cohort action distribution and primary-intent shares.

    Uses the competition's own transitions when it is one of the paired
    competitions, else all seven pooled. Transitions without labels (about 2%
    of the paired split, e.g. one 221-version commonlit trajectory) are
    skipped, as unparsed rows of the analyzed run are.
    """
    labeled = action["coarse_actions"].map(lambda xs: isinstance(xs, list) and len(xs) > 0)
    df = action[labeled & action["group7"].isin(config.COHORTS)]
    in_comp = bool(comp) and comp in set(df["comp"])
    ref = df[df["comp"] == comp] if in_comp else df
    out = {}
    for g, gdf in ref.groupby("group7"):
        has = np.array([gdf[f"has_{c}"].mean() for c in config.COARSE_ACTIONS], float)
        dist = has / has.sum() if has.sum() else has
        intents = {i: round(float((gdf["primary_intent"] == i).mean()), 4) for i in config.INTENTS}
        out[g] = {"dist": dist, "intents": intents, "n_transitions": int(len(gdf)),
                  "n_trajectories": int(gdf["key_id"].nunique())}
    return out, in_comp


def human_scores(idx: pd.DataFrame, comp: str) -> pd.Series:
    """Best score of every human trajectory in a competition."""
    h = idx[(idx["comp"] == comp) & (~idx["is_agent"].fillna(False).astype(bool))]
    return human_best_score(h).dropna()
