#!/usr/bin/env python3
"""Measure cold, streamed API requests against a running Strata server."""

from __future__ import annotations

import argparse
import http.client
import json
import pathlib
import statistics
import time
import urllib.parse


def get_json(host: str, port: int, path: str) -> dict:
    conn = http.client.HTTPConnection(host, port, timeout=60)
    conn.request("GET", path)
    response = conn.getresponse()
    payload = response.read()
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"GET {path}: HTTP {response.status}: {payload[:300]!r}")
    return json.loads(payload)


def stream_chat(host: str, port: int, prompt: dict, max_tokens: int) -> dict:
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {
                "role": "system",
                "content": "You are a benchmark target. Be concise. Do not echo the input.",
            },
            {"role": "user", "content": prompt["text"]},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 1234,
    }
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=600)
    started = time.perf_counter()
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=encoded,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"chat HTTP {response.status}: {response.read()[:1000]!r}")

    first_token_s = None
    usage = None
    finish_reason = None
    chunks = 0
    while True:
        line = response.readline()
        if not line:
            break
        text = line.decode("utf-8", "replace").strip()
        if not text.startswith("data:"):
            continue
        data = text[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        chunks += 1
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if first_token_s is None and any(
                delta.get(key) for key in ("content", "reasoning", "reasoning_content")
            ):
                first_token_s = time.perf_counter() - started
    total_s = time.perf_counter() - started
    conn.close()

    status = get_json(host, port, "/v1/status")
    timings = status.get("last_timings") or {}
    usage = usage or {}
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")
              if usage else None)
    return {
        "name": prompt["name"],
        "kind": prompt["kind"],
        "target_tokens": prompt["target_tokens"],
        "raw_prompt_tokens": prompt["raw_token_count"],
        "prompt_sha256": prompt["sha256"],
        "external_ttft_s": first_token_s,
        "total_s": total_s,
        "stream_chunks": chunks,
        "finish_reason": finish_reason,
        "usage": usage,
        "cached_tokens": cached,
        "timings": timings,
        "status": {
            key: status.get(key)
            for key in ("cache_max_tokens", "model", "engine", "last_timings")
        },
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["name"], []).append(row)
    result = []
    for name, values in groups.items():
        prompt_ms = [float(v["timings"].get("prompt_ms") or 0) for v in values]
        prompt_n = [int(v["timings"].get("prompt_n") or 0) for v in values]
        decode_s = [float(v["timings"].get("predicted_per_second") or 0) for v in values]
        ttft = [float(v["external_ttft_s"] or 0) for v in values]
        total = [float(v["total_s"]) for v in values]
        result.append({
            "name": name,
            "repeats": len(values),
            "prompt_tokens_median": statistics.median(prompt_n),
            "cached_tokens_max": max(int(v["cached_tokens"] or 0) for v in values),
            "prompt_ms_median": statistics.median(prompt_ms),
            "prompt_tokens_s_median": statistics.median(
                [n * 1000 / ms for n, ms in zip(prompt_n, prompt_ms) if ms > 0]
            ),
            "decode_tokens_s_median": statistics.median(decode_s),
            "external_ttft_s_median": statistics.median(ttft),
            "external_ttft_s_p95": percentile(ttft, 0.95),
            "total_s_median": statistics.median(total),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18195)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("--allow-cache", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompts = payload["prompts"]
    wanted = set(args.name)
    if wanted:
        prompts = [prompt for prompt in prompts if prompt["name"] in wanted]
    if not prompts:
        raise SystemExit("no prompts selected")

    health = get_json(args.host, args.port, "/health")
    rows = []
    for prompt in prompts:
        for iteration in range(args.repeat):
            row = stream_chat(args.host, args.port, prompt, args.max_tokens)
            row["iteration"] = iteration
            row["health"] = health
            if not args.allow_cache and row["cached_tokens"] != 0:
                raise RuntimeError(
                    f"{prompt['name']} iteration {iteration} reused "
                    f"{row['cached_tokens']} prompt tokens"
                )
            rows.append(row)
            timing = row["timings"]
            print(
                f"{prompt['name']} run={iteration} prompt={timing.get('prompt_n')} "
                f"prompt_ms={timing.get('prompt_ms')} "
                f"decode_tok_s={timing.get('predicted_per_second')} "
                f"ttft_s={row['external_ttft_s']}",
                flush=True,
            )
    result = {"schema": 1, "rows": rows, "summary": summarize(rows)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
