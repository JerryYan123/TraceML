#!/usr/bin/env python3
"""Fail if an href/src in the website points to a local file that does not exist.

    python scripts/dev/check_docs_links.py docs
"""
from __future__ import annotations

import re
import sys
from html.parser import HTMLParser
from pathlib import Path


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs: list[str] = []
        self.ids: set[str] = set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])


def main(root: str) -> int:
    root_p = Path(root)
    bad = []
    for page in sorted(root_p.rglob("*.html")):
        p = Links()
        p.feed(page.read_text(encoding="utf-8"))
        for ref in p.refs:
            if re.match(r"^(https?:|mailto:|data:)", ref):
                continue
            if ref.startswith("#"):
                if ref[1:] and ref[1:] not in p.ids:
                    bad.append(f"{page}: missing anchor {ref}")
                continue
            target = (page.parent / ref.split("#")[0]).resolve()
            if not target.exists():
                bad.append(f"{page}: missing file {ref}")
    for b in bad:
        print(b)
    print(f"checked {root_p}: {'OK' if not bad else f'{len(bad)} problems'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "docs"))
