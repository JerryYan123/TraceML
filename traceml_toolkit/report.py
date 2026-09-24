"""Behavior report: a labeled run -> ``report.md``, ``report.json`` and ``figs/``.

Sections: run summary, behavioral fingerprint against the released cohorts
(coarse-action distribution, Jensen-Shannon divergence to each cohort,
primary-intent shares), best-so-far score against the human cohort of the same
competition, and a memory profile (grader-call redundancy, agent sessions).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from traceml_toolkit import __version__, assets, cohorts, config
from traceml_toolkit.labeling.inputs import read_jsonl

# Chart tokens (light surface). Three categorical slots validated for all-pairs
# use (dataviz reference palette); identity is also carried by marker shape.
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
SERIES = {"this run": ("#2a78d6", "o"), "top10": ("#eb6834", "D"), "codex": ("#1baf7a", "s")}
COHORT_LABELS = {
    "top10": "top-10% humans", "top10-40": "humans, 10-40%", "top40-70": "humans, 40-70%",
    "top70-100": "humans, bottom 30%", "codex": "Codex (paper)", "mlev_normal": "MLEvolve",
    "mlev_best": "MLEvolve, best branch",
}
FATAL_RE = re.compile(r"usage limit|rate_limit_exceeded|insufficient_quota|authentication|invalid_api_key")
TIMEOUT_RC = 124  # exit code of `timeout` when the recorder caps a session


def jsd(p, q) -> float:
    """Base-2 Jensen-Shannon divergence between two distributions."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def load_transitions(out_dir: Path) -> list[dict]:
    """Labeled action rows (rows the labeler failed on are excluded)."""
    p = Path(out_dir) / "labels" / "action_output.jsonl"
    if not p.exists():
        return []
    return [r for r in read_jsonl(p) if r.get("coarse_actions")]


def action_dist(transitions: list[dict]) -> np.ndarray:
    """Share of each coarse action among transitions, renormalized to sum to 1."""
    has = np.zeros(len(config.COARSE_ACTIONS))
    for t in transitions:
        cs = set(t.get("coarse_actions") or [])
        for i, c in enumerate(config.COARSE_ACTIONS):
            has[i] += c in cs
    if transitions:
        has /= len(transitions)
    return has / has.sum() if has.sum() else has


def intent_shares(transitions: list[dict]) -> tuple[dict, int]:
    counts = {i: 0 for i in config.INTENTS}
    n = 0
    for t in transitions:
        ints = t.get("intents") or []
        prim = ints[0].get("intent") if ints and isinstance(ints[0], dict) else None
        if prim in counts:
            counts[prim] += 1
            n += 1
    return {k: round(v / n, 4) if n else 0.0 for k, v in counts.items()}, n


def scored_versions(traj: dict) -> list[dict]:
    return [v for v in traj["versions"] if v.get("linked_score") is not None]


def best_score(traj: dict, direction: str) -> float | None:
    vals = [v["linked_score"] for v in scored_versions(traj)]
    if not vals:
        return None
    return min(vals) if direction == "lower" else max(vals)


def human_percentile(human_best, score: float | None, direction: str) -> float | None:
    """Percent of human trajectories whose best score is worse than `score`."""
    if score is None or len(human_best) == 0:
        return None
    hs = np.asarray(human_best, float)
    better = (hs > score).mean() if direction == "lower" else (hs < score).mean()
    return round(float(better) * 100, 1)


def run_health(run_dir: str | Path | None) -> dict | None:
    """Session-loop health from the recorder's ``native_session.log``, when present.

    A run whose sessions mostly exit with an error (quota, auth) spends its
    budget respawning rather than working. Sessions stopped by the recorder's
    own time cap (exit code 124) are counted as capped, not failed.
    """
    if not run_dir:
        return None
    log = Path(run_dir) / "native_session.log"
    if not log.is_file():
        return None
    starts = failed = capped = fatal = 0
    with open(log, errors="replace") as fh:
        for line in fh:
            if line.startswith("[respawn"):
                if " remain=" in line:
                    starts += 1
                elif " exited rc=" in line:
                    m = re.search(r"exited rc=(-?\d+)", line)
                    if not m:
                        continue
                    rc = int(m.group(1))
                    if rc == TIMEOUT_RC:
                        capped += 1
                    elif rc != 0:
                        failed += 1
            elif line.startswith('{"type":"error"') and FATAL_RE.search(line):
                fatal += 1
    if not starts:
        return None
    share = failed / starts
    return {"sessions": starts, "failed": failed, "capped_by_recorder": capped,
            "failed_share": round(share, 4), "fatal_provider_errors": fatal,
            "healthy": share <= 0.10 and fatal == 0}


def memory_profile(traj: dict, health: dict | None) -> dict:
    out = {}
    nt, nv = traj.get("n_tags_total"), traj.get("version_count")
    if nt:
        out["grader_calls"] = nt
        out["distinct_code_states"] = nv
        out["redundancy"] = round(1 - nv / nt, 4)
    if health:
        out["agent_sessions"] = health["sessions"]
        if nv:
            out["code_states_per_session"] = round(nv / health["sessions"], 2)
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": MUTED, "ytick.color": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False,
    })
    return plt


def fig_actions(run_dist, ref: dict, out_png: Path) -> None:
    """Dot plot of coarse-action shares: this run vs top-10% humans vs Codex."""
    plt = _plt()
    series = [("this run", run_dist)] + [(g, ref[g]["dist"]) for g in ("top10", "codex") if g in ref]
    n = len(config.COARSE_ACTIONS)
    y = np.arange(n)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for k, (name, dist) in enumerate(series):
        color, marker = SERIES[name]
        label = "this run" if name == "this run" else COHORT_LABELS[name]
        ax.scatter(np.asarray(dist) * 100, y, s=70 if k == 0 else 48, marker=marker, color=color,
                   edgecolors=SURFACE, linewidths=1.5, zorder=3 + (len(series) - k), label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(config.COARSE_ACTIONS)
    ax.set_xlabel("share of coarse-action labels (%)")
    ax.grid(axis="x")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    from matplotlib.ticker import MaxNLocator
    xmax = max(float(np.max(d)) for _, d in series) * 100
    ax.set_xlim(-0.03 * xmax, 1.06 * xmax)  # room for markers sitting at zero
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series), fontsize=9,
              handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def fig_progress(traj: dict, direction: str, human_best, metric: str, out_png: Path) -> bool:
    """Best-so-far score over the run, against the human cohort's best scores."""
    scored = sorted(((v.get("ctime") or 0), v["linked_score"]) for v in scored_versions(traj))
    if len(scored) < 2:
        return False
    plt = _plt()
    t0, t1 = scored[0][0], scored[-1][0]
    xs = [100 * (t - t0) / max(t1 - t0, 1) for t, _ in scored]
    best, cur = [], None
    for _, s in scored:
        cur = s if cur is None else (min(cur, s) if direction == "lower" else max(cur, s))
        best.append(cur)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    if len(human_best):
        # Middle half of the human cohort (each human at their best score) and its median.
        q_lo, q_med, q_hi = np.percentile(np.asarray(human_best, float), [25, 50, 75])
        ax.axhspan(q_lo, q_hi, color=BAND, zorder=0, lw=0, label="middle 50% of humans")
        ax.axhline(q_med, color=MUTED, lw=1, zorder=1)
        ax.text(101, q_med, "median\nhuman", va="center", fontsize=8, color=INK_2)
    color, _ = SERIES["this run"]
    ax.step(xs, best, where="post", color=color, lw=2, zorder=3)
    ax.plot(xs, best, "o", ms=5.5, color=color, mec=SURFACE, mew=1.5, zorder=4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("elapsed share of the run (%)")
    ax.set_ylabel(f"best {metric} so far\n({direction} is better)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def build_report(out_dir: str | Path, comp: str | None = None, dataset_dir: str | Path | None = None,
                 run_dir: str | Path | None = None, download: bool = True) -> dict:
    out_dir = Path(out_dir)
    traj = json.loads((out_dir / "extracted" / "trajectory.json").read_text())
    comp = comp or traj.get("competition")
    manifest = config.competitions().get(comp or "", {})
    direction = traj.get("score_direction") or manifest.get("score_direction") or "higher"
    metric = manifest.get("metric", "score")
    run_name = traj.get("run_name") or Path(traj.get("run_dir", str(out_dir))).name

    action_ref, idx = cohorts.load_reference(dataset_dir, download=download)
    ref, in_paired = cohorts.cohort_profiles(action_ref, comp)
    human_best = cohorts.human_scores(idx, comp) if (comp and in_paired) else []

    trans = load_transitions(out_dir)
    run_dist = action_dist(trans)
    ish, n_int = intent_shares(trans)
    best = best_score(traj, direction)
    pct = human_percentile(human_best, best, direction) if in_paired else None
    # The recorder log lives in the run directory; a copy next to the report also works.
    health = run_health(run_dir or traj.get("run_dir")) or run_health(out_dir)
    figs = out_dir / "figs"
    figs.mkdir(exist_ok=True)

    has_labels = bool(trans)
    if has_labels:
        fig_actions(run_dist, ref, figs / "fig1_actions.png")
        jsd_bits = {g: round(jsd(run_dist, r["dist"]), 4) for g, r in ref.items() if r["dist"].sum()}
    else:
        jsd_bits = {}
    nearest = sorted(jsd_bits, key=jsd_bits.get)
    prog_ok = fig_progress(traj, direction, human_best, metric, figs / "fig2_progress.png") if in_paired \
        else False

    ref_rev = assets.local_revision("data/paired/action.parquet")
    rep = {
        "run_name": run_name,
        "harness": traj.get("harness"),
        "competition": comp,
        "metric": metric,
        "score_direction": direction,
        "reference": "paired" if in_paired else "pooled",
        "version_count": traj.get("version_count"),
        "grader_calls": traj.get("n_tags_total"),
        "scored_versions": len(scored_versions(traj)),
        "best_score": best,
        "human_percentile": pct,
        "n_human_trajectories": len(human_best),
        "n_labeled_transitions": len(trans),
        "action_distribution": ({"this_run": dict(zip(config.COARSE_ACTIONS, np.round(run_dist, 4).tolist()))}
                                | {g: dict(zip(config.COARSE_ACTIONS, np.round(r["dist"], 4).tolist()))
                                   for g, r in ref.items()}) if has_labels else {},
        "jsd_bits": {g: jsd_bits[g] for g in nearest},
        "nearest_cohorts": nearest[:3],
        "intent_shares": ({"this_run": ish, "n_this_run": n_int}
                          | {g: r["intents"] for g, r in ref.items()}) if has_labels else {},
        "cohort_sizes": {g: {"transitions": r["n_transitions"], "trajectories": r["n_trajectories"]}
                         for g, r in ref.items()},
        "memory": memory_profile(traj, health),
        "health": health,
        "figures": {"actions": "figs/fig1_actions.png" if has_labels else None,
                    "progress": "figs/fig2_progress.png" if prog_ok else None},
        "provenance": {
            "toolkit_version": __version__,
            "reference_dataset": config.hf_repo(),
            "reference_revision": ref_rev,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    (out_dir / "report.md").write_text(render_markdown(rep))
    return rep


def _fmt(x) -> str:
    return "-" if x is None else f"{x:.3f}" if isinstance(x, float) else str(x)


def render_markdown(rep: dict) -> str:
    L = [f"# Trajectory behavior report: {rep['run_name']}", ""]
    h = rep.get("health")
    if h and not h["healthy"]:
        live = h["sessions"] - h["failed"]
        L += [f"> **Run truncated: the effective budget is shorter than the wall clock.** "
              f"{h['failed']} of {h['sessions']} agent sessions exited with an error "
              f"({h['fatal_provider_errors']} provider-side failures such as quota or auth), "
              f"leaving about {live} sessions that did work. Per-transition shares below remain "
              f"valid; session counts and comparisons that assume a full-length run do not.", ""]
    ref_note = ("paired reference: this competition's own human and agent trajectories"
                if rep["reference"] == "paired"
                else "pooled reference: competition not in the paired split, all seven pooled")
    L.append(f"- harness: **{rep['harness']}** · competition: **{rep['competition']}** "
             f"({rep['metric']}, {rep['score_direction']} is better; {ref_note})")
    calls = f" from **{rep['grader_calls']}** grader calls" if rep.get("grader_calls") else ""
    L.append(f"- versions: **{rep['version_count']}** distinct code states{calls} · "
             f"scored: {rep['scored_versions']}")
    line = f"- best score: **{rep['best_score']}**"
    if rep.get("human_percentile") is not None:
        line += (f" → better than **{rep['human_percentile']}%** of the "
                 f"{rep['n_human_trajectories']} human trajectories on this competition "
                 f"(each human counted at their best score)")
    L.append(line)

    if rep["action_distribution"]:
        dist = rep["action_distribution"]
        cols = [c for c in ("this_run", "top10", "codex") if c in dist]
        L += ["", "## Behavioral fingerprint", "", "![actions](figs/fig1_actions.png)", "",
              "Share of coarse-action labels (%):", "",
              "| action | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for a in config.COARSE_ACTIONS:
            L.append(f"| {a} | " + " | ".join(f"{100 * dist[c][a]:.1f}" for c in cols) + " |")
        L += ["", "Jensen-Shannon divergence (bits) between this run's action distribution and "
              "each reference cohort, nearest first:", "", "| cohort | JSD |", "|---|---|"]
        for g, v in rep["jsd_bits"].items():
            L.append(f"| {g} | {v:.4f} |")
        ish = rep["intent_shares"]
        L += ["", f"Primary-intent shares over {ish['n_this_run']} labeled transitions:", "",
              "| intent | this run | top10 | codex |", "|---|---|---|---|"]
        for i in config.INTENTS:
            L.append(f"| {i} | {ish['this_run'][i]:.3f} | {_fmt(ish.get('top10', {}).get(i))} "
                     f"| {_fmt(ish.get('codex', {}).get(i))} |")
    else:
        L += ["", "_No action labels yet: run `traceml label` to add the behavioral fingerprint._"]

    if rep["figures"].get("progress"):
        L += ["", "## Score progress", "", "![progress](figs/fig2_progress.png)", "",
              "The line is this run's best score so far. The shaded band is the middle half of the "
              "human trajectories on this competition, each at its best score."]
    mem = rep.get("memory") or {}
    if mem:
        labels = {"grader_calls": "grader calls", "distinct_code_states": "distinct code states",
                  "redundancy": "redundancy (share of grader calls on an unchanged code state)",
                  "agent_sessions": "agent sessions (from native_session.log)",
                  "code_states_per_session": "code states per session"}
        L += ["", "## Memory profile", ""]
        L += [f"- {labels.get(k, k)}: **{v}**" for k, v in mem.items()]
    p = rep["provenance"]
    rev = f"@{p['reference_revision'][:7]}" if p.get("reference_revision") else ""
    L += ["", "---",
          f"*Generated {p['generated_at'][:16].replace('T', ' ')} by traceml-toolkit {p['toolkit_version']}. "
          f"Labels: released Qwen3-1.7B TraceML labelers. Reference: TraceML paired split "
          f"({p['reference_dataset']}{rev}).*", ""]
    return "\n".join(L)
