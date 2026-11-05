#!/usr/bin/env python3
"""Render one K100AI run-config template without leaving unresolved host placeholders."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

PLACEHOLDER = re.compile(r"\$\{([A-Z][A-Z0-9_]*)\}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("template", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    text = args.template.read_text(encoding="utf-8")
    missing = sorted({name for name in PLACEHOLDER.findall(text) if not os.environ.get(name)})
    if missing:
        print("missing required environment variable(s): " + ", ".join(missing), file=sys.stderr)
        return 2
    rendered = PLACEHOLDER.sub(lambda m: os.environ[m.group(1)], text)
    config = json.loads(rendered)
    if any(isinstance(value, str) and "${" in value for value in config.get("args", [])):
        print("the rendered config still has an unresolved argument placeholder", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(config, out, ensure_ascii=False, indent=2)
        out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
