"""Turn raw labeler generations into label rows (same shapes as the released pipeline)."""
from __future__ import annotations

import json

from traceml_toolkit.labeling.prompts import action as action_prompt

TRACK = "qwen3_1.7b_distill"


def parse_state_output(text: str) -> dict:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = {"coarse_tags": [], "fine_tags": [], "summary": "PARSE_ERROR", "keywords": []}
    valid_fine = []
    for ft in parsed.get("fine_tags", []) or []:
        if isinstance(ft, dict) and "tag" in ft and "parent" in ft:
            conf = ft.get("confidence", "mid")
            if conf not in ("high", "mid", "low"):
                conf = "mid"
            entry = {"tag": str(ft["tag"]), "parent": str(ft["parent"]), "confidence": conf}
            if str(ft["tag"]).startswith("other_"):
                entry["proposed_tag"] = str(ft.get("proposed_tag", ""))
                entry["description"] = str(ft.get("description", ""))
            valid_fine.append(entry)
    keywords = parsed.get("keywords", [])
    return {
        "coarse_tags": [t for t in (parsed.get("coarse_tags", []) or []) if isinstance(t, str)],
        "fine_tags": valid_fine,
        "summary": str(parsed.get("summary", ""))[:300],
        "keywords": list(keywords)[:7] if isinstance(keywords, (list, tuple)) else [],
    }


def state_row(rec: dict, text: str, model_name: str) -> dict:
    text = (text or "").strip()
    return {
        "key_id": rec.get("key_id"),
        "comp": rec.get("comp"),
        "group": rec.get("group"),
        "version_number": rec.get("version_number"),
        "track": TRACK,
        "model": model_name,
        **parse_state_output(text),
        "raw_text_len": len(text),
    }


def action_row(rec: dict, text: str, model_name: str) -> dict:
    text = (text or "").strip()
    return {
        "key_id": rec.get("key_id"),
        "v_old": rec.get("v_old"),
        "v_new": rec.get("v_new"),
        "comp": rec.get("comp"),
        "group": rec.get("group"),
        "kind": rec.get("kind"),
        "model": model_name,
        **action_prompt.parse_response(text),
        "raw_text_len": len(text),
    }


def make_row(task: str, rec: dict, text: str, model_name: str) -> dict:
    return state_row(rec, text, model_name) if task == "state" else action_row(rec, text, model_name)


def is_parsed(task: str, row: dict) -> bool:
    return bool(row.get("coarse_tags" if task == "state" else "coarse_actions"))


def row_id(task: str, row: dict):
    return row.get("version_number") if task == "state" else (row.get("v_old"), row.get("v_new"))
