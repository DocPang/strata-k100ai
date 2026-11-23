#!/usr/bin/env python3
"""Reproduce fixed Qwen IQ3_S requests against one server or two candidates."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

CASES = {
    32768: (1535, 17, "785670272829e99e26a26353f8477178a7996b3caa9fcb3e2fc102d883792ac0"),
    261632: (11854, 9, "71bbb229fed8ead3fe7d109bf106ace3517fbaa95dca870b0c6b20009aac5dc7"),
}
CODES = ("ALPHA-7391", "BRAVO-2846", "CHARLIE-6158")


def prompt(length):
    records, padding, expected = CASES[length]
    needles = dict(zip((0, records // 2, records - 1), CODES))
    lines = ["Benchmark trial 0. Read all records."]
    for index in range(records):
        line = f"Record {index}: warehouse cedar has blue crates and green labels; inventory is checked daily."
        if index in needles:
            line += f" Secret code: {needles[index]}."
        lines.append(line)
    lines += [
        "Padding:" + " x" * padding,
        "Return the three secret codes in order, then write a detailed inventory auditing guide of at "
        "least 1500 words. Do not stop after listing the codes.",
    ]
    text = "\n".join(lines)
    if hashlib.sha256(text.encode()).hexdigest() != expected:
        raise RuntimeError(f"Fixture hash mismatch for {length} tokens")
    return text


def request(base, model, text, output_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
        "seed": 42,
        "max_tokens": output_tokens,
        "cache_prompt": False,
        "presence_penalty": 0,
        "top_p": 1,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = Request(base + "/v1/chat/completions", json.dumps(payload).encode(),
                  {"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    content, usage, timings = [], {}, {}
    with urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(event["error"])
            usage = event.get("usage") or usage
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                if first is None and (delta.get("content") or delta.get("reasoning_content")):
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
    return {"text": "".join(content), "usage": usage, "timings": timings,
            "ttft_s": first, "wall_s": time.monotonic() - started}


def check(row, length):
    usage = row["usage"]
    timings = row["timings"]
    if (usage.get("prompt_tokens"), usage.get("completion_tokens")) != (length, 256):
        raise RuntimeError(f"Unexpected token counts for {length}: {usage}")
    if timings.get("cache_n") != 0 or not all(timings.get(key, 0) > 0 for key in
                                              ("prompt_per_second", "predicted_per_second")):
        raise RuntimeError(f"Unexpected cache or timing data for {length}: {timings}")
    positions = [row["text"].find(code) for code in CODES]
    if min(positions) < 0 or positions != sorted(positions):
        raise RuntimeError(f"Retrieval failed at {length} tokens: {positions}")
    row["retrieval"] = True
    row["output_sha256"] = hashlib.sha256(row.pop("text").encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18096")
    parser.add_argument("--candidate-url")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--lengths", type=int, choices=CASES, nargs="+", default=list(CASES))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    servers = {"baseline": args.base_url.rstrip("/")}
    if args.candidate_url:
        servers["candidate"] = args.candidate_url.rstrip("/")
    args.out.mkdir(parents=True, exist_ok=False)
    for base in servers.values():
        with urlopen(base + "/health", timeout=10) as response:
            health = json.load(response)
        if health.get("status") != "ok" or health.get("max_context") != 262144:
            raise RuntimeError(f"{base}: expected a healthy 262,144-token service: {health}")
        request(base, args.model, "Explain how a refrigerator works.", 48)
    for index, length in enumerate(args.lengths):
        names = list(servers)
        if index % 2:
            names.reverse()
        for name in names:
            row = request(servers[name], args.model, prompt(length), 256)
            check(row, length)
            (args.out / f"{name}-{length}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps({"server": name, "input": length, **row}), flush=True)
        if len(servers) == 2:
            results = [json.loads((args.out / f"{name}-{length}.json").read_text()) for name in servers]
            print(json.dumps({"input": length, "output_equal": results[0]["output_sha256"] ==
                              results[1]["output_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
