# TraceML toolkit

[![ci](https://github.com/JerryYan123/TraceML/actions/workflows/ci.yml/badge.svg)](https://github.com/JerryYan123/TraceML/actions/workflows/ci.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2608.26086-b31b1b.svg)](https://arxiv.org/abs/2608.26086)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-jerryyan%2FTraceML-yellow)](https://huggingface.co/datasets/jerryyan/TraceML)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Read any ML-engineering agent run against human practice. The toolkit turns a finished agent
run into a [TraceML](https://jerryyan123.github.io/TraceML/) trajectory, labels every code
version and every edit with the released labelers, and reports how the run's development
process compares with 4,465 human Kaggle trajectories and with the agents studied in the paper.

**[Project page](https://jerryyan123.github.io/TraceML/) · [Paper](https://arxiv.org/abs/2608.26086) · [Dataset](https://huggingface.co/datasets/jerryyan/TraceML)**

```bash
pip install "traceml-toolkit[label] @ git+https://github.com/JerryYan123/TraceML"   # Linux + CUDA GPU
traceml analyze path/to/agent_run
```

```
traceml_out/<run name>/
  extracted/trajectory.json     every distinct code state, its score and grader calls
  extracted/versions/           main file (vNNN.py) and whole codebase (vNNN_all.txt) per version
  labels/                       state and action labels (JSONL) and label_log.json
  data/{state,action}.parquet   the run in the dataset's schema, ready to concatenate with it
  report.md, report.json, figs/ behavior report against the human cohorts
```

## What the report contains

For a one-hour Claude Code run on CommonLit Readability ([full example](examples/claudecode_haiku45_1h_commonlit/report.md)):

- **Outcome in human terms.** 14 grader calls collapse into 13 distinct code states. The best
  RMSE, 0.733, is better than 20.4% of the 103 human trajectories on this competition.
- **Behavioral fingerprint.** The share of each coarse action (data, features, model, training,
  ensemble, validation, inference, ...) against top-10% humans and the paper's Codex runs, the
  Jensen-Shannon divergence to every cohort, and the mix of intents (exploration, optimization,
  pivoting, debugging, ...).
- **Score progress** against the middle half of the human cohort of the same competition.
- **Memory profile**: grader-call redundancy and agent sessions, and a warning when a run spent
  its budget in a crash loop (quota or authentication failures).

<p align="center">
  <img src="docs/figures/demo-actions.png" width="49%" alt="Coarse-action shares of the example run against top-10% humans and Codex">
  <img src="docs/figures/demo-progress.png" width="49%" alt="Best RMSE so far of the example run against the human cohort">
</p>

## Install

```bash
# Full pipeline (Linux, one CUDA GPU, Python 3.10-3.12). vLLM reserves half of the
# card's memory by default; change it with --gpu-mem-util.
pip install "traceml-toolkit[label] @ git+https://github.com/JerryYan123/TraceML"

# Extraction and reports only, or labeling with plain transformers (CPU / Apple silicon; slow)
pip install "git+https://github.com/JerryYan123/TraceML"
pip install "traceml-toolkit[transformers] @ git+https://github.com/JerryYan123/TraceML"
```

The labelers (two Qwen3-1.7B checkpoints, about 3.4 GB each) and the reference tables (about
11 MB) download from the HuggingFace dataset on first use into `~/.cache/traceml`. Fetch them
ahead of time with `traceml download-models`, or point at an existing clone of the dataset with
`--dataset-dir` / `--models-dir`. `traceml info` shows what was found.

## Try it without a GPU

The repository ships two labeled example runs. Reports need no GPU:

```bash
git clone https://github.com/JerryYan123/TraceML && cd TraceML
pip install -e .
cp -r examples/claudecode_haiku45_1h_commonlit /tmp/example
traceml report /tmp/example
```

Any trajectory of the dataset, human or agent, can be reported on the same way:

```bash
traceml from-released 18221654 --out /tmp/human_run   # a key_id from the paired split
traceml report /tmp/human_run
```

## Recording a run

`traceml analyze` accepts three kinds of run directories and detects which one it is given:

| Layout | How it is produced | Scores |
|---|---|---|
| git sidecar with `v<N>-true-<score>` tags | `traceml record` (below), or the Codex grading shim used in the paper | read from the tags |
| AIDE / MLEvolve search journal (`logs/journal.json`) | the scaffold itself | the agent's own CV metric |
| git repository that tracks `submission.csv` | any agent in a git workspace | re-graded with `mlebench grade-sample` |

`traceml record` runs any command-line agent under a recorder. It commits the workspace whenever
code changes, grades every new `submission.csv` with MLE-bench, tags the commit with the score,
and restarts the agent until the time budget is spent. It needs
[MLE-bench](https://github.com/openai/mle-bench) with the competition's data prepared, and GNU
`timeout` (`brew install coreutils` on macOS).

```bash
mkdir -p runs/cc_commonlit && cp my_task_prompt.md runs/cc_commonlit/task.md
export TRACEML_MLEBENCH_CACHE=/path/to/mle-bench/data

traceml record runs/cc_commonlit --slug commonlitreadabilityprize --minutes 60 --harness claude-code \
    -- claude -p {PROMPT} --model claude-haiku-4-5 --dangerously-skip-permissions

traceml analyze runs/cc_commonlit
```

The token `{PROMPT}` is replaced by the text of `task.md` for the first session and of `cont.md`
(written for you if missing) for later ones. Commands like the one above let the agent act
without confirmation, so run them in a container or VM. The demo runs used
`gemini --yolo -m gemini-2.5-flash -p {PROMPT}` for Gemini CLI.

## How it works

**Versions.** A version is a distinct code state: the set of `.py` and `.ipynb` files in a commit,
excluding submissions, binaries, prompts and logs. Consecutive grader calls on unchanged code
merge into one version that keeps the best of their scores. This mirrors a Kaggle "save version",
the unit of the human side of the dataset.

**Labels.** Each version gets state labels (8 coarse pipeline stages, 136 fine tags). Each
transition gets action labels (10 coarse, 85 fine), up to two intents, a magnitude, and a score
effect computed from the measured scores. Labels come from the two released Qwen3-1.7B labelers
with the prompts they were trained on. By default the action diff covers the whole codebase, so
edits outside the main file count; `--diff-scope main` diffs the main file only, as the dataset's
agent labels did, and agrees more closely with them (see Validation). It misses work that an
agent puts in new files.

**Backends.** vLLM (default on a CUDA machine) runs greedy decoding with prefix caching disabled:
with caching on, batches of near-identical prompts, which is what one run produces, degenerate
into repetition loops. Rows that come back unparsed are retried in small batches with a larger
output budget, and any that still fail are counted in `labels/label_log.json`, never dropped
silently. `--backend transformers` works without vLLM but is much slower.

**Reference cohorts.** Humans on the seven paired competitions are split per competition into
top 10%, 10-40%, 40-70% and bottom 30% by each human's best leaderboard score. Codex and MLEvolve
come from the paper's runs. Runs on other competitions are compared with all seven pooled; the
human percentile is only reported for the paired competitions.

## Validation

Measured on one RTX A6000 (48 GB) with the vLLM backend:

| Check | Result |
|---|---|
| Claude Code demo, end to end | 14 grader calls → 13 versions; 13/13 states and 12/12 transitions labeled; 2 min 19 s |
| Gemini CLI demo, end to end | 3 grader calls → 2 versions; all rows labeled; no false truncation flag; 1 min 50 s |
| Paper regression run (Codex, skill rep1, commonlit) | 2,505 grader calls → 16 versions, the released count; state coarse tags identical on 16/16 versions (fine-tag Jaccard 0.64); action coarse Jaccard 0.10 with whole-codebase diffs and 0.33 with `--diff-scope main` |
| `transformers` backend vs vLLM | state coarse tags identical on 13/13 versions, fine-tag Jaccard 0.98; 192 s against 67 s on the same GPU |

The regression run was processed both by the dataset pipeline and by this toolkit. Extraction
reproduces the released version count and the state labels. Action and intent labels agree less.
The dataset's agent labels were built from main-file diffs plus AST-derived change atoms that
are not part of the public release, with vLLM prefix caching on. With `--diff-scope main` the
toolkit matches what the paper-time scripts measured on this run exactly (action Jaccard 0.33,
primary intent 27%); with whole-codebase diffs it gives 0.10 where they gave 0.12.

CI runs the CPU test suite (extraction on synthetic git sidecars, prompt parity with the released
labeler prompts, retry and refit logic, cohort bucketing, report regeneration for both examples,
the recorder end to end with a fake agent and grader) on Python 3.10 and 3.12.

## Command reference

| Command | Purpose |
|---|---|
| `traceml analyze RUN [--gpu N] [--skip-label]` | all stages; `--skip-label` needs no GPU |
| `traceml extract RUN` | code states and scores only |
| `traceml label OUT [--backend vllm\|transformers] [--task state\|action]` | label an extracted run |
| `traceml schema OUT` | write `data/{state,action}.parquet` |
| `traceml report OUT [--comp SLUG]` | write `report.md`, `report.json`, `figs/` |
| `traceml record RUN --slug S --minutes M --harness H -- CMD {PROMPT}` | record an agent |
| `traceml download-models [--data-only]` | fetch labelers and reference data |
| `traceml from-released KEY_ID [--split paired]` | materialize a dataset trajectory for `report` |
| `traceml info` | resolved paths, assets and backends |

| Environment variable | Default | Meaning |
|---|---|---|
| `TRACEML_HOME` | `~/.cache/traceml` | download cache |
| `TRACEML_DATASET_DIR`, `TRACEML_MODELS_DIR` | unset | use an existing dataset clone or labeler directory |
| `TRACEML_HF_REPO`, `TRACEML_HF_REVISION` | `jerryyan/TraceML`, `main` | dataset to download from |
| `TRACEML_VLLM_PYTHON` | current Python | run the vLLM worker from another environment |
| `TRACEML_MLEBENCH`, `TRACEML_MLEBENCH_CACHE` | `mlebench`, unset | grader for `record` and the re-grade path |

## Changes from the paper-time scripts

This package supersedes `code/05_toolkit/` in the dataset. It fixes the human percentile on
lower-is-better metrics (the paper's Appendix B.5 reports 52% for the Claude Code demo, which
becomes 20%), a false crash-loop flag, silently dropped action rows, and orphaned vLLM engines.
See [CHANGELOG.md](CHANGELOG.md).

## Citation

```bibtex
@article{yan2026traceml,
  title   = {TraceML: An Empirical Analysis of Human-Agent Planning in Machine Learning Development},
  author  = {Yan, Jiarui and Sun, Weiwei and Li, Sijie and Li, Wenhan and Yang, Yiming},
  journal = {arXiv preprint arXiv:2608.26086},
  year    = {2026}
}
```

## License

Code: Apache-2.0. The bundled schemas, manifest, prompts, examples and test fixtures come from the
TraceML dataset (CC BY 4.0); see [NOTICE](NOTICE).
