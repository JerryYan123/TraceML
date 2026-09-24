"""Paths, environment variables and constants shared across the toolkit."""
from __future__ import annotations

import json
import os
from functools import cache
from importlib.resources import files
from pathlib import Path

# ---------------------------------------------------------------------------
# Label vocabularies (must match manifests/schemas/*.json)
# ---------------------------------------------------------------------------
COARSE_ACTIONS = ["data", "features", "augmentation", "model", "training",
                  "ensemble", "validation", "inference", "infra", "housekeeping"]
INTENTS = ["exploration", "optimization", "pivoting", "debugging",
           "restructuring", "verification"]
COARSE_TAGS = ["data_io", "feature_eng", "model_def", "training_cfg",
               "validation_cv", "ensemble_blend", "inference_submit", "infra_util"]
MAGNITUDES = ["micro", "minor", "major", "overhaul"]
SCORE_EFFECTS = ["improving", "plateau", "regressing", "unknown"]

# Reference cohorts of the paired split: four human leaderboard tiers
# (per-competition percentile of each human's best score) plus the two agent
# scaffolds studied in the paper.
COHORTS = ["top10", "top10-40", "top40-70", "top70-100", "codex", "mlev_normal", "mlev_best"]
HUMAN_COHORTS = COHORTS[:4]

# The seven competitions worked by both humans and agents in the paper.
PAIRED_COMPETITIONS = [
    "amex-default-prediction",
    "commonlitreadabilityprize",
    "equity-post-hct-survival-predictions",
    "google-quest-challenge",
    "hms-harmful-brain-activity-classification",
    "learning-agency-lab-automated-essay-scoring-2",
    "ranzcr-clip-catheter-line-classification",
]

# ---------------------------------------------------------------------------
# HuggingFace dataset and local cache
# ---------------------------------------------------------------------------
DEFAULT_HF_REPO = "jerryyan/TraceML"
LABELER_DIRS = {
    "state": "models/qwen3-1.7b-state/final",
    "action": "models/qwen3-1.7b-action/final",
}
LABELER_MAX_POSITIONS = 40960  # max_position_embeddings of the released checkpoints


def hf_repo() -> str:
    return os.environ.get("TRACEML_HF_REPO", DEFAULT_HF_REPO)


def hf_revision() -> str | None:
    return os.environ.get("TRACEML_HF_REVISION") or None


def traceml_home() -> Path:
    """Cache root for downloaded reference data and labelers."""
    return Path(os.environ.get("TRACEML_HOME", Path.home() / ".cache" / "traceml")).expanduser()


def default_dataset_dir() -> Path:
    """Where the toolkit downloads its slice of the dataset (release-repo layout)."""
    return traceml_home() / "dataset"


def user_dataset_dir(override: str | Path | None = None) -> Path | None:
    """A dataset clone the user already has (``--dataset-dir`` or ``TRACEML_DATASET_DIR``)."""
    p = override or os.environ.get("TRACEML_DATASET_DIR")
    return Path(p).expanduser() if p else None


def user_models_dir(override: str | Path | None = None) -> Path | None:
    """A directory holding ``qwen3-1.7b-{state,action}/final`` (``--models-dir`` or ``TRACEML_MODELS_DIR``)."""
    p = override or os.environ.get("TRACEML_MODELS_DIR")
    return Path(p).expanduser() if p else None


def mlebench_bin(override: str | None = None) -> str:
    return override or os.environ.get("TRACEML_MLEBENCH", "mlebench")


def mlebench_cache(override: str | Path | None = None) -> Path | None:
    p = override or os.environ.get("TRACEML_MLEBENCH_CACHE")
    return Path(p).expanduser() if p else None


# ---------------------------------------------------------------------------
# Bundled resources
# ---------------------------------------------------------------------------
def resource_text(relpath: str) -> str:
    return files("traceml_toolkit.resources").joinpath(relpath).read_text(encoding="utf-8")


@cache
def competitions() -> dict:
    """The competition manifest (launch, deadline, metric, score direction)."""
    return json.loads(resource_text("manifests/competitions.json"))


@cache
def schema(name: str) -> dict:
    """One of schema_state, schema_action, fine_tag_state, fine_tag_action."""
    return json.loads(resource_text(f"schemas/{name}.json"))


def score_direction(slug: str | None) -> str | None:
    """'higher' or 'lower' for a known competition slug, else None."""
    if not slug:
        return None
    entry = competitions().get(slug)
    return entry.get("score_direction") if entry else None
