"""Action/intent (per-transition) prompt, verbatim from the released ``llm_action_prompt.py``.

The labeler was fine-tuned on exactly these prompts, so the text must not
change. Only schema loading (bundled JSON instead of repo-relative paths) and
an optional ``max_lines`` for refitting over-long diffs are new. The OpenAI
client of the original script (used only to build teacher labels) is gone.
"""
from __future__ import annotations

import json
from functools import cache

from traceml_toolkit import config

MAX_DIFF_LINES = 300

VALID_COARSE = {"data", "features", "augmentation", "model", "training",
                "ensemble", "validation", "inference", "infra", "housekeeping", "other"}
VALID_INTENT = {"exploration", "optimization", "pivoting",
                "debugging", "restructuring", "verification", "other"}


def truncate_diff(diff: str, max_lines: int = MAX_DIFF_LINES) -> str:
    lines = diff.splitlines()
    if len(lines) <= max_lines:
        return diff
    kept = lines[:max_lines]
    kept.append(f"\n... [{len(lines) - max_lines} more diff lines truncated] ...")
    return "\n".join(kept)


@cache
def build_system_prompt() -> str:
    """Build the long system prompt (identical for every transition)."""
    schema = config.schema("schema_action")
    fine = config.schema("fine_tag_action")

    coarse_defs = "\n".join(
        f"  - {tag}: {desc}"
        for tag, desc in schema["coarse_actions"].items()
        if tag != "_doc"
    )

    intent_defs = "\n".join(
        f"  - {tag}: {desc}"
        for tag, desc in schema["intent_categories"].items()
        if tag != "_doc"
    )

    fine_sections = []
    for parent, tags_dict in fine["tags"].items():
        doc = tags_dict.get("_doc", "")
        tag_list = "\n".join(
            f"      {t}: {desc}" for t, desc in tags_dict.items() if t != "_doc"
        )
        fine_sections.append(f"    {parent} ({doc}):\n{tag_list}")
    fine_list = "\n".join(fine_sections)

    return f"""You annotate a single version-to-version transition in a Kaggle ML competition kernel.

You receive:
1. State BEFORE (Phase 1 annotation: coarse tags, key fine tags, summary)
2. State AFTER (same format)
3. Code diff (unified diff of the actual code change)
4. Atoms (structured list of detected changes from AST analysis)
5. Metadata (competition, position in trajectory, scores)

Your task: classify WHAT changed (Action) and WHY (Intent).

═══ ACTION: WHAT changed (multi-label) ═══

COARSE ACTIONS — select ALL that apply:
{coarse_defs}

FINE ACTIONS — for each coarse action selected, assign specific fine tags.
Each fine tag MUST include confidence: "high" (explicit in diff), "mid" (clear from context), "low" (ambiguous).
If no fine tag fits, use other_<parent> and you MUST provide BOTH "proposed_tag" (a short snake_case name) and "description" (one sentence). Never leave proposed_tag empty.

{fine_list}

═══ INTENT: WHY the change was made (1-2 labels, primary first) ═══

List 1-2 intents with confidence. The FIRST intent is the PRIMARY one.
Most transitions need only 1 intent. Add a second only when the transition genuinely serves two distinct purposes.

{intent_defs}

Each intent gets confidence: "high" (clear from diff), "mid" (reasonable inference), "low" (genuinely ambiguous).

INTENT DISAMBIGUATION RULES (follow strictly):

1. exploration vs optimization — the MOST IMPORTANT distinction:
   - EXPLORATION = trying something QUALITATIVELY DIFFERENT that might not work:
     * Switching to a completely different model FAMILY (e.g. LGBM → neural net, ResNet → Swin)
     * Adding an entirely new type of feature that didn't exist before
     * Trying a fundamentally different approach or pipeline structure
     * Building the first working pipeline from scratch
     * Key signal: the author doesn't know if this will help; it's a bet
   - OPTIMIZATION = REFINING something that already works:
     * Tuning hyperparameters (lr, epochs, batch size, regularization)
     * Swapping checkpoints or pretrained weights within the SAME model family
     * Adjusting augmentation parameters, thresholds, or blend weights
     * Adding/removing a feature from an existing feature set
     * Ensemble weight tuning, late-stage submission polishing
     * Key signal: the existing approach stays; only knobs are turned
   - RULE OF THUMB: if the model architecture family stays the same → optimization.
     If it changes to something structurally different → exploration.

2. exploration vs optimization — additional signals:
   - If the change replaces the model backbone with a DIFFERENT family
     (e.g. RandomForest → LightGBM, CNN → Transformer, Conv1D → GRU),
     that is exploration even if hyperparameters are also tuned.
   - If the change swaps a variant WITHIN the same family
     (e.g. EfficientNet-B0 → B4, DeBERTa-base → large, same architecture but different size/checkpoint),
     that is optimization.
   - Large-scale changes (near-complete rewrites, adding a whole new training script,
     major pipeline restructuring) that also involve switching the core model or approach
     strongly suggest exploration, not optimization.

3. debugging vs optimization:
   - DEBUGGING = the code was BROKEN or produced errors/wrong output.
     Look for evidence in the diff: commenting out broken code, fixing variable names,
     fixing import errors, correcting tensor shapes, adding missing data loading steps
     that caused NameError/KeyError, disabling a feature that caused crashes.
   - OPTIMIZATION = the code WORKED correctly, this change aims to IMPROVE its score.
   - When in doubt: if the diff shows something being "fixed" or "corrected"
     (especially reverting a recent change, or the summary mentions "fix"),
     lean toward debugging. If the diff shows parameters being "tuned" or
     "adjusted", lean toward optimization.

4. verification:
   - Minimal or zero code change, purpose is to re-run and check the score
   - Typical for NOOP or near-NOOP transitions

Key principle: Action and Intent are ORTHOGONAL.
- Same action (e.g., "model") can have different intents
- Same intent (e.g., "optimization") can involve different actions

═══ REASONING STEPS ═══

Think step by step:
1. Read the state summaries — what was there before and after?
2. Read the diff — what concrete code changes happened?
3. Read the atoms — what structured changes were detected?
4. Identify ALL coarse action categories that apply
5. For each coarse action, pick the most specific fine tags
6. Determine the dominant strategic purpose (intent)
7. Assess magnitude and score effect
8. Write a brief diff_summary and goal_nl

═══ OUTPUT FORMAT ═══

Output ONLY valid JSON:
{{
  "coarse_actions": ["action1", "action2"],
  "fine_actions": [
    {{"action": "tag_name", "parent": "coarse_parent", "confidence": "high"}},
    {{"action": "other_training", "parent": "training", "confidence": "mid", "proposed_tag": "warmup_steps", "description": "Changed number of warmup steps"}}
  ],
  "intents": [
    {{"intent": "optimization", "confidence": "high"}},
    {{"intent": "debugging", "confidence": "low"}}
  ],
  "goal_nl": "Fine-tuning learning rate to improve convergence",
  "diff_summary": "Changed lr from 0.01 to 0.003, added cosine scheduler",
  "magnitude": "minor",
  "score_effect": "improving",
  "other_action_note": null,
  "other_intent_note": null
}}

Rules:
- coarse_actions: multi-label list, pick ALL that apply
- fine_actions: list of fine tags with parent and confidence
- intents: 1-2 intents, PRIMARY FIRST. Most transitions need only 1. Each with confidence (high/mid/low).
- magnitude: "micro" (<5 lines, cosmetic), "minor" (one component), "major" (significant multi-component), "overhaul" (near-complete rewrite)
- score_effect: "improving" / "plateau" / "regressing" / "unknown" (based on score_old vs score_new)
- If coarse_actions includes "other": set other_action_note = {{"proposed_tag": "...", "description": "..."}}
- If any intent is "other": set other_intent_note = {{"proposed_tag": "...", "description": "..."}}
- For NOOP (identical code): coarse_actions=["housekeeping"], fine_actions=[{{"action":"noop","parent":"housekeeping","confidence":"high"}}], intents=[{{"intent":"verification","confidence":"high"}}]
"""


def format_state(state: dict, label: str) -> str:
    coarse = ", ".join(state.get("coarse_tags", []))
    fine_parts = []
    for ft in state.get("fine_tags", [])[:15]:
        conf = ft.get("confidence", "")
        fine_parts.append(f"{ft['tag']}({conf})")
    fine_str = ", ".join(fine_parts) if fine_parts else "(none)"
    summary = state.get("summary", "")
    return f"[{label}] Coarse: {coarse}\n  Fine: {fine_str}\n  Summary: {summary}"


def format_atoms(atoms: list) -> str:
    if not atoms:
        return "(no atoms detected)"
    parts = []
    for a in atoms[:20]:
        if isinstance(a, dict):
            atype = a.get("type", a.get("atom_type", "?"))
            detail = a.get("detail", a.get("description", ""))
            parts.append(f"  - {atype}: {detail}" if detail else f"  - {atype}")
        elif isinstance(a, str):
            parts.append(f"  - {a}")
    if len(atoms) > 20:
        parts.append(f"  ... [{len(atoms) - 20} more atoms]")
    return "\n".join(parts)


def format_score(score) -> str:
    if score is None:
        return "none"
    if isinstance(score, dict):
        pub = score.get("public", "?")
        return f"{pub}"
    return str(score)


def build_user_prompt(rec: dict, max_lines: int | None = None) -> str:
    """Build the per-transition user prompt."""
    state_old = format_state(rec.get("state_old", {}), "BEFORE")
    state_new = format_state(rec.get("state_new", {}), "AFTER")

    diff = rec.get("code_diff", "")
    diff = truncate_diff(diff) if max_lines is None else truncate_diff(diff, max_lines)
    atoms = format_atoms(rec.get("atoms", []))

    score_old = format_score(rec.get("score_old"))
    score_new = format_score(rec.get("score_new"))
    delta = rec.get("score_delta")
    delta_str = f"{delta:+.6f}" if delta is not None else "N/A"

    meta = (
        f"Competition: {rec['comp']}\n"
        f"Group: {rec['group']} | Kind: {rec['kind']}\n"
        f"Transition: v{rec['v_old']} → v{rec['v_new']} "
        f"(position {rec['position_in_trajectory']:.1%}, {rec['total_versions']} total versions)\n"
        f"Score: {score_old} → {score_new} (delta: {delta_str})\n"
        f"Churn: {rec.get('churn_class', '?')} | Lines: +{rec.get('n_added', 0)} -{rec.get('n_removed', 0)}"
    )

    return f"""{meta}

{state_old}

{state_new}

ATOMS:
{atoms}

DIFF:
```diff
{diff}
```"""


def parse_response(content_str: str) -> dict:
    """Parse and validate the labeler's JSON output."""
    try:
        parsed = json.loads(content_str)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        return {
            "coarse_actions": [], "fine_actions": [], "intent": "PARSE_ERROR",
            "goal_nl": "", "diff_summary": "", "magnitude": "unknown",
            "score_effect": "unknown", "other_action_note": None, "other_intent_note": None,
        }

    coarse = [c for c in parsed.get("coarse_actions", []) if c in VALID_COARSE]

    fine = []
    for ft in parsed.get("fine_actions", []):
        if isinstance(ft, dict) and "action" in ft and "parent" in ft:
            conf = ft.get("confidence", "mid")
            if conf not in ("high", "mid", "low"):
                conf = "mid"
            entry = {"action": str(ft["action"]), "parent": str(ft["parent"]), "confidence": conf}
            if str(ft["action"]).startswith("other_"):
                entry["proposed_tag"] = str(ft.get("proposed_tag", ""))
                entry["description"] = str(ft.get("description", ""))
            fine.append(entry)

    # Handle both old "intent" (single) and new "intents" (list) formats
    raw_intents = parsed.get("intents", [])
    if not raw_intents and "intent" in parsed:
        raw_intents = [{"intent": parsed["intent"], "confidence": "mid"}]
    intents = []
    for it in raw_intents[:2]:
        if isinstance(it, dict) and "intent" in it:
            iname = it["intent"]
            if iname not in VALID_INTENT:
                iname = "other"
            conf = it.get("confidence", "mid")
            if conf not in ("high", "mid", "low"):
                conf = "mid"
            intents.append({"intent": iname, "confidence": conf})
        elif isinstance(it, str):
            intents.append({"intent": it if it in VALID_INTENT else "other", "confidence": "mid"})
    if not intents:
        intents = [{"intent": "other", "confidence": "low"}]

    return {
        "coarse_actions": coarse,
        "fine_actions": fine,
        "intents": intents,
        "goal_nl": str(parsed.get("goal_nl", ""))[:300],
        "diff_summary": str(parsed.get("diff_summary", ""))[:300],
        "magnitude": parsed.get("magnitude", "unknown"),
        "score_effect": parsed.get("score_effect", "unknown"),
        "other_action_note": parsed.get("other_action_note"),
        "other_intent_note": parsed.get("other_intent_note"),
    }
