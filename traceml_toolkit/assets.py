"""Locate or download the parts of the TraceML dataset the toolkit needs.

Two kinds of assets live in the HuggingFace dataset repo:

* reference tables used by the report (the paired split and the trajectory
  index, about 11 MB), and
* the two distilled Qwen3-1.7B labelers (about 3.4 GB each).

Lookup order for every asset: a dataset clone the user points at
(``--dataset-dir`` / ``TRACEML_DATASET_DIR``), then the toolkit cache
(``$TRACEML_HOME/dataset``, default ``~/.cache/traceml/dataset``). Missing
assets are downloaded into the cache only, never into a user's clone. Git-LFS
pointer files (a clone made without ``git lfs pull``) count as missing.
"""
from __future__ import annotations

from pathlib import Path

from traceml_toolkit import config

TRAJECTORY_INDEX = "extras/trajectory_index.parquet"
SPLITS = ("paired", "humans_only", "experiment_run")
# What the behavior report needs.
REFERENCE_FILES = ("data/paired/action.parquet", TRAJECTORY_INDEX)
# What `traceml download-models --data-only` fetches.
DATA_FILES = ("data/paired/state.parquet", "data/paired/action.parquet", TRAJECTORY_INDEX)


class MissingAsset(FileNotFoundError):
    """A dataset file or labeler that is neither present locally nor downloadable."""


def split_file(split: str, table: str) -> str:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    if table not in ("state", "action"):
        raise ValueError(f"unknown table {table!r}; expected 'state' or 'action'")
    return f"data/{split}/{table}.parquet"


def is_lfs_pointer(path: Path) -> bool:
    """True if `path` is a Git-LFS pointer stub rather than the real file."""
    try:
        if path.stat().st_size > 1024:
            return False
        with path.open("rb") as fh:
            return fh.read(64).startswith(b"version https://git-lfs")
    except OSError:
        return False


def _usable(path: Path) -> bool:
    return path.is_file() and not is_lfs_pointer(path)


def _roots(dataset_dir: str | Path | None) -> list[Path]:
    roots = []
    user = config.user_dataset_dir(dataset_dir)
    if user is not None:
        roots.append(user)
    roots.append(config.default_dataset_dir())
    return roots


def download(patterns: list[str], ignore: list[str] | None = None) -> Path:
    """Download matching files from the HF dataset repo into the toolkit cache."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:  # pragma: no cover - huggingface_hub is a core dependency
        raise MissingAsset("huggingface_hub is required to download TraceML assets") from e
    local = config.default_dataset_dir()
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=config.hf_repo(),
        repo_type="dataset",
        revision=config.hf_revision(),
        allow_patterns=patterns,
        ignore_patterns=ignore,
        local_dir=str(local),
    )
    return local


def find_file(relpath: str, dataset_dir: str | Path | None = None) -> Path | None:
    for root in _roots(dataset_dir):
        p = root / relpath
        if _usable(p):
            return p
    return None


def resolve_file(relpath: str, dataset_dir: str | Path | None = None,
                 download_missing: bool = True) -> Path:
    """Path to a dataset file, downloading it into the cache if needed."""
    p = find_file(relpath, dataset_dir)
    if p is not None:
        return p
    if not download_missing:
        raise MissingAsset(
            f"{relpath} not found under {', '.join(str(r) for r in _roots(dataset_dir))}. "
            "Run `traceml download-models --data-only` or pass --dataset-dir.")
    download([relpath])
    p = config.default_dataset_dir() / relpath
    if not _usable(p):
        raise MissingAsset(f"download of {relpath} from {config.hf_repo()} did not produce a usable file")
    return p


def _labeler_ok(d: Path) -> bool:
    return ((d / "config.json").is_file() and _usable(d / "model.safetensors")
            and _usable(d / "tokenizer.json"))


def labeler_candidates(task: str, models_dir: str | Path | None = None,
                       dataset_dir: str | Path | None = None) -> list[Path]:
    rel = Path(config.LABELER_DIRS[task])
    out = []
    user_models = config.user_models_dir(models_dir)
    if user_models is not None:
        out.append(user_models / rel.relative_to("models"))
    out.extend(root / rel for root in _roots(dataset_dir))
    return out


def find_labeler(task: str, models_dir: str | Path | None = None,
                 dataset_dir: str | Path | None = None) -> Path | None:
    for c in labeler_candidates(task, models_dir, dataset_dir):
        if _labeler_ok(c):
            return c
    return None


def resolve_labeler(task: str, models_dir: str | Path | None = None,
                    dataset_dir: str | Path | None = None,
                    download_missing: bool = True) -> Path:
    """Directory of the released `task` labeler ('state' or 'action')."""
    if task not in config.LABELER_DIRS:
        raise ValueError(f"unknown labeler task {task!r}")
    found = find_labeler(task, models_dir, dataset_dir)
    if found is not None:
        return found
    if not download_missing:
        raise MissingAsset(
            f"the {task} labeler was not found (looked in: "
            f"{', '.join(str(c) for c in labeler_candidates(task, models_dir, dataset_dir))}). "
            "Run `traceml download-models` or pass --models-dir.")
    rel = config.LABELER_DIRS[task]
    print(f"[traceml] downloading the {task} labeler (~3.4 GB) from {config.hf_repo()} ...", flush=True)
    download([f"{rel}/*"], ignore=["*training_args.bin"])
    d = config.default_dataset_dir() / rel
    if not _labeler_ok(d):
        raise MissingAsset(f"download of {rel} from {config.hf_repo()} is incomplete")
    return d


def local_revision(relpath: str) -> str | None:
    """Commit hash of a file downloaded into the cache (from huggingface_hub's metadata)."""
    meta = config.default_dataset_dir() / ".cache" / "huggingface" / "download" / f"{relpath}.metadata"
    try:
        first = meta.read_text().splitlines()[0].strip()
        return first or None
    except (OSError, IndexError):
        return None
