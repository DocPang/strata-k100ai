#!/usr/bin/env python3
"""Run a fixed, deterministic API quality set and save the full responses."""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.request


CASES = [
    {
        "name": "france_capital_zh",
        "prompt": "用一句简洁的中文回答：法国的首都是哪里？",
        "test": lambda text: "巴黎" in text,
    },
    {
        "name": "arithmetic",
        "prompt": "只给出计算结果：17 × 23 = ?",
        "test": lambda text: re.search(r"(?<!\d)391(?!\d)", text) is not None,
    },
    {
        "name": "python_add",
        "prompt": "用 Python 写一个 add(a, b) 函数并返回两数之和，只给代码。",
        "test": lambda text: "def add" in text and "return" in text and "a+b" in text.replace(" ", ""),
    },
    {
        "name": "unit_conversion_zh",
        "prompt": "用一句简洁的中文回答：1 千米等于多少米？",
        "test": lambda text: "1000" in text and "米" in text,
    },
]


def request(host: str, port: int, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
        "max_tokens": max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18195)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    rows = []
    failed = []
    for case in CASES:
        response = request(args.host, args.port, case["prompt"], args.max_tokens)
        choice = response["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        passed = bool(case["test"](content))
        row = {
            "name": case["name"],
            "prompt": case["prompt"],
            "passed": passed,
            "finish_reason": choice.get("finish_reason"),
            "content": content,
            "reasoning": reasoning,
            "usage": response.get("usage"),
            "response": response,
        }
        rows.append(row)
        if not passed:
            failed.append(case["name"])
        print(f"{case['name']}: {'ok' if passed else 'FAIL'} ({choice.get('finish_reason')})")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"schema": 1, "cases": rows, "failed": failed}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
