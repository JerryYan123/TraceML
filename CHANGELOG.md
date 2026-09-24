# Changelog

## 0.1.0 (2026-09)

First packaged release. It is a port of the scripts released as `code/05_toolkit/` in the
[TraceML dataset](https://huggingface.co/datasets/jerryyan/TraceML) (the "paper-time toolkit"
used for Appendix B.5 of the paper), made installable and runnable outside the authors' machines.

### New

- `pip install`-able package with a `traceml` command: `analyze` (all stages in one command),
  `extract`, `label`, `schema`, `report`, `record`, `download-models`, `from-released`, `info`.
- Reference data and labelers are downloaded on demand from HuggingFace into `~/.cache/traceml`
  (or taken from an existing dataset clone with `--dataset-dir` / `--models-dir`).
- A `transformers` labeling backend for machines without vLLM (slow, for small runs).
- `report.json` next to `report.md`, and `labels/label_log.json` with per-task labeling statistics.
- The report runs without labels (extraction, score percentile and progress only).

### Fixed

These change numbers relative to the paper-time toolkit.

- **Human percentile and cohorts on lower-is-better competitions.** Humans were ranked by
  `max_score`, which for RMSE- or KL-type metrics is a human's *worst* submission. The toolkit
  now ranks by each human's best score. On CommonLit the Claude Code demo run (RMSE 0.733) moves
  from "beats 52.4% of the human cohort" to 20.4%, and the Gemini CLI demo from 88.3% to 72.8%.
  Cohort membership (top10 ... top70-100) changes for 734 of 4,392 bucketed human trajectories,
  all on lower-is-better competitions, which shifts the JSD table for commonlit and hms runs.
- **False "run truncated" banner.** Sessions ended by the recorder's own time cap (exit code 124)
  were counted as failed sessions. The Gemini CLI demo run was wrongly flagged (1 of 3 sessions).
- **Silently dropped action rows.** The action labeler ran with an 8,192-token window, so long
  diffs were skipped without output, and the retry (6,000 output tokens in the same window)
  could never generate. The window is now 16,384 tokens, prompts that still do not fit are
  re-rendered with shorter excerpts, and retries size their window to the prompt. The Claude Code
  demo run goes from 11 to 12 of 12 transitions labeled.
- **Orphaned vLLM engines.** Each labeling call now runs in its own process group that is
  terminated afterwards; previously vLLM's engine process could survive and hold GPU memory,
  making the next call on the same GPU fail.
- **Grading tags with negative or exponent scores** (`v3-true--0.5`, `v4-true-1e-05`) were ignored.
- The released labeler paths, `<DATASET_ROOT>` placeholders and the unreleased cohort loader are
  replaced by package resources and downloads.
- Cohort reference profiles skip transitions without labels (2.2% of the paired split).
- The memory profile no longer prints a constant taken from an unreleased analysis file
  (`cohort_codex_skip_parent@0.9`); it reports agent sessions from the recorder log instead.

### Changed

- Action inputs now include real `+added -removed` line counts in the prompt (the paper-time
  toolkit always showed `+0 -0`). The labeler prompts are otherwise byte-identical to the
  released ones. Relabeling the Claude Code demo with this version changes 3 of the 11
  transitions labeled by both versions from exploration to optimization, so its exploration
  share goes from 36% to 8%. Its state labels are unchanged (coarse-tag Jaccard 1.0).
- The recorder writes `run_meta.json` when absent, keeps an existing `.gitignore`, works on macOS
  (`md5`, `gtimeout`) and fails fast when `task.md`, `cont.md` or the grader are missing.
- Output goes to `./traceml_out/<run name>/` and never into the run directory.
