#!/usr/bin/env python3
"""Parse DS4_PROF output into per-prompt and per-token module costs.

The engine reports profiler stage events summed across all participating
devices.  For each prompt this script therefore records both:

* staged device microseconds per prompt token (the raw summed value / tokens);
* four-card wall-equivalent microseconds per prompt token (the raw summed
  value / 4 / tokens).

The wall-equivalent column is useful for comparing module cost with the
engine's prompt wall time.  The raw column avoids hiding device duplication.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re


PROMPT_RE = re.compile(
    r"prompt (?P<tokens>\d+) tokens = (?P<reused>\d+) reused \+ (?P<read>\d+) "
    r"read in (?P<prompt_ms>\d+) ms \((?P<prompt_tps>[\d.]+) tok/s\), "
    r"(?P<generated>\d+) generated in (?P<decode_ms>\d+) ms "
    r"\((?P<decode_tps>[\d.]+) tok/s\)"
)
STAGE_RE = re.compile(r"ds4 prof\s+(?P<name>[^ ].*?)\s+(?P<ms>[\d.]+) ms\s+(?P<pct>[\d.]+)%")
EXPERT_RE = re.compile(
    r"ds4 prof\s+expert kernels type (?P<type>\S+)\s+(?P<phase>gate\+up|down)\s+"
    r"(?P<ms>[\d.]+) ms\s+calls\s+(?P<calls>\d+)\s+experts\s+(?P<experts>\d+)\s+"
    r"\((?P<us_each>[\d.]+) us each, (?P<per_call>[\d.]+) a call\)\s+"
    r"rows/expert\s+(?P<rows_expert>[\d.]+)\s+~\s+(?P<tflops>[\d.]+) TFLOPS"
)


def prompt_record(tokens: int, prompt_ms: float, stages: list[dict], experts: list[dict]) -> dict:
    return {
        "prompt_tokens": tokens,
        "prompt_ms": prompt_ms,
        "wall_us_per_token": prompt_ms * 1000.0 / tokens,
        "stages": stages,
        "experts": experts,
    }


def parse(text: str) -> list[dict]:
    prompts: list[dict] = []
    pending_stages: list[dict] = []
    pending_experts: list[dict] = []
    tokens = 0
    for line in text.splitlines():
        match = PROMPT_RE.search(line)
        if match:
            tokens = int(match.group("tokens"))
            current = prompt_record(
                tokens,
                float(match.group("prompt_ms")),
                pending_stages,
                pending_experts,
            )
            current["meta"] = {
                "reused": int(match.group("reused")),
                "generated": int(match.group("generated")),
                "prompt_tps": float(match.group("prompt_tps")),
                "decode_ms": float(match.group("decode_ms")),
                "decode_tps": float(match.group("decode_tps")),
            }
            for event in pending_stages:
                event["summed_us_per_token"] = event["ms"] * 1000.0 / tokens
                event["four_card_wall_equiv_us_per_token"] = event["ms"] * 1000.0 / 4.0 / tokens
            for event in pending_experts:
                event["summed_us_per_token"] = event["ms"] * 1000.0 / tokens
                event["four_card_wall_equiv_us_per_token"] = event["ms"] * 1000.0 / 4.0 / tokens
            prompts.append(current)
            pending_stages = []
            pending_experts = []
            continue

        match = EXPERT_RE.search(line)
        if match:
            ms = float(match.group("ms"))
            pending_experts.append(
                {
                    "type": match.group("type"),
                    "phase": match.group("phase"),
                    "ms": ms,
                    "calls": int(match.group("calls")),
                    "experts": int(match.group("experts")),
                    "us_each": float(match.group("us_each")),
                    "experts_per_call": float(match.group("per_call")),
                    "rows_per_expert": float(match.group("rows_expert")),
                    "achieved_tflops": float(match.group("tflops")),
                }
            )
            continue

        match = STAGE_RE.search(line)
        if match:
            ms = float(match.group("ms"))
            pending_stages.append(
                {
                    "name": match.group("name"),
                    "ms": ms,
                    "share_pct": float(match.group("pct")),
                }
            )
    return prompts


def print_summary(prompts: list[dict]) -> None:
    for prompt in prompts:
        meta = prompt.get("meta", {})
        print(
            f"\n{prompt['prompt_tokens']} tokens, wall "
            f"{prompt['prompt_ms'] / 1000.0:.3f} s, "
            f"{prompt['wall_us_per_token']:.1f} us/token, "
            f"decode {meta.get('decode_tps', 0):.2f} tok/s"
        )
        print(f"{'模块':34} {'总ms':>9} {'占比':>7} {'四卡和us/tok':>14} {'四卡均值us/tok':>16}")
        for stage in prompt["stages"]:
            print(
                f"{stage['name']:34} {stage['ms']:9.1f} {stage['share_pct']:6.1f}% "
                f"{stage['summed_us_per_token']:14.1f} "
                f"{stage['four_card_wall_equiv_us_per_token']:16.1f}"
            )
        if prompt["experts"]:
            print(f"{'专家类型':10} {'阶段':8} {'总ms':>9} {'四卡均值us/tok':>16} {'us/专家':>10} {'TFLOPS':>8}")
            for expert in prompt["experts"]:
                print(
                    f"{expert['type']:10} {expert['phase']:8} {expert['ms']:9.1f} "
                    f"{expert['four_card_wall_equiv_us_per_token']:16.1f} "
                    f"{expert['us_each']:10.1f} {expert['achieved_tflops']:8.1f}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--json", type=pathlib.Path)
    args = parser.parse_args()

    prompts = parse(args.log.read_text(encoding="utf-8", errors="replace"))
    if not prompts:
        raise SystemExit("DS4_PROF prompt records not found")
    print_summary(prompts)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"schema": 1, "prompts": prompts}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
