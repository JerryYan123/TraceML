"""State (per-version) prompt, verbatim from the released ``llm_state_prompt.py``.

The labeler was fine-tuned on exactly these prompts, so the text must not
change. Only schema loading (bundled JSON instead of repo-relative paths) and
an optional ``max_lines`` for refitting over-long files are new.
"""
from __future__ import annotations

from functools import cache

from traceml_toolkit import config

MAX_CODE_LINES = 1200


def truncate_code(code: str, max_lines: int = MAX_CODE_LINES) -> str:
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return code
    header_lines = []
    body_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_structural = (
            stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("class ")
            or stripped.startswith("def ")
            or stripped.startswith("# ---")
            or stripped.startswith("if __name__")
        )
        if is_structural or i < 50:
            header_lines.append(line)
        else:
            body_lines.append(line)
    budget = max_lines - len(header_lines) - 5
    if budget > 0 and body_lines:
        step = max(1, len(body_lines) // budget)
        sampled_body = body_lines[::step][:budget]
    else:
        sampled_body = body_lines[:200]
    return "\n".join(header_lines + ["\n# ... [truncated] ...\n"] + sampled_body)


@cache
def build_system_prompt() -> str:
    schema = config.schema("schema_state")
    fine_v3 = config.schema("fine_tag_state")

    coarse_defs = "\n".join(
        f"  - {tag}: {desc}" for tag, desc in schema["coarse_tags"].items()
    )

    fine_sections = []
    for parent, tags_dict in fine_v3["tags"].items():
        doc = tags_dict.get("_doc", "")
        tag_list = "\n".join(
            f"      {t}: {desc}" for t, desc in tags_dict.items() if t != "_doc"
        )
        fine_sections.append(f"  {parent} ({doc}):\n{tag_list}")
    fine_list = "\n".join(fine_sections)

    return f"""You annotate what functional components are PRESENT in a Kaggle ML competition kernel version.

Your task: read the full source code and identify what this version CONTAINS — what models, what features,
what training setup, what validation strategy. Focus on WHAT IS THERE, not what changed from a prior version.

COARSE TAGS (multi-label — select ALL that apply):
{coarse_defs}

FINE TAGS — for each coarse tag, assign fine tags from the list below.
Each fine tag MUST include a confidence level:
  "high" = certain (explicit import, class instantiation, clear API call)
  "mid"  = likely (clear usage pattern but indirect)
  "low"  = uncertain (heuristic guess, ambiguous code)

Use tags from the list. If an important component has NO good match, use other_<parent>
with proposed_tag + description. Only use other when nothing fits.

FINE TAGS:
{fine_list}

Also produce:
- summary: 1 sentence — what this code version is (e.g., "LightGBM with GroupBy features and 5-fold stratified CV")
- keywords: 3-5 key technical terms

Output ONLY valid JSON:
{{
  "coarse_tags": ["tag1", "tag2", ...],
  "fine_tags": [
    {{"tag": "existing_tag", "parent": "...", "confidence": "high"}},
    {{"tag": "other_model_def", "parent": "model_def", "confidence": "mid", "proposed_tag": "name", "description": "..."}},
    ...
  ],
  "summary": "...",
  "keywords": ["...", "..."]
}}"""


def build_user_prompt(rec: dict, max_lines: int | None = None) -> str:
    code = rec.get("code_text", "")
    if max_lines is None:
        code = truncate_code(code)
    else:
        # Refit path: the structural header alone can exceed a small budget,
        # so hard-cap the sampled result as well.
        code = "\n".join(truncate_code(code, max_lines).splitlines()[:max_lines])
    meta = (f"Competition: {rec['comp']}\n"
            f"Group: {rec['group']}\n"
            f"Version: {rec['version_number']}\n"
            f"Lines: {rec.get('code_lines', '?')}")
    return f"{meta}\n\n```python\n{code}\n```"
