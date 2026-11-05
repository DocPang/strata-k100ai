#!/usr/bin/env bash
set -euo pipefail

base=${BASE_URL:-http://127.0.0.1:8095}
model=${MODEL:-qwen3.8-flash-next}
prompt=${PROMPT:-用一句话回答：法国的首都是哪里？}
max_tokens=${MAX_TOKENS:-16}
out=${OUT:-"api-validation-$(date +%Y%m%d-%H%M%S)"}
expect_draft_metrics=${EXPECT_DRAFT_METRICS:-0}

mkdir -p "$out"
curl --fail --silent --show-error "$base/health" | tee "$out/health.json"
printf '\n'
curl --fail --silent --show-error "$base/v1/models" | tee "$out/models.json"
printf '\n'

python3 - "$base" "$model" "$prompt" "$max_tokens" "$out/chat.json" <<'PY'
import json
import sys
import time
import urllib.request

base, model, prompt, max_tokens, out = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "stream": False,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
start = time.monotonic()
with urllib.request.urlopen(req, timeout=1200) as response:
    body = response.read()
elapsed = time.monotonic() - start
open(out, "wb").write(body)
parsed = json.loads(body)
choice = parsed["choices"][0]
print(json.dumps({
    "elapsed_s": round(elapsed, 6),
    "finish_reason": choice.get("finish_reason"),
    "content": choice["message"].get("content", ""),
    "usage": parsed.get("usage"),
}, ensure_ascii=False, indent=2))
PY

python3 - "$base" "$model" "$prompt" "$max_tokens" "$out/chat-stream.jsonl" <<'PY'
import json
import sys
import time
import urllib.request

base, model, prompt, max_tokens, out = sys.argv[1:]
payload = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "temperature": 0,
    "max_tokens": int(max_tokens),
    "stream": True,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
start = time.monotonic()
first_token = None
last_token = None
chunks = 0
usage = None
finish_reason = None
with urllib.request.urlopen(req, timeout=1200) as response, open(out, "wb") as f:
    for raw in response:
        f.write(raw)
        if not raw.startswith(b"data: "):
            continue
        data = raw[6:].strip()
        if data == b"[DONE]":
            break
        try:
            event = json.loads(data)
        except ValueError:
            continue
        usage = event.get("usage") or usage
        for choice in event.get("choices") or []:
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.monotonic()
                chunks += 1
                first_token = now if first_token is None else first_token
                last_token = now
end = time.monotonic()
decode_s = (last_token - first_token) if first_token is not None and last_token is not None else None
print(json.dumps({
    "external_ttft_s": round(first_token - start, 6) if first_token is not None else None,
    "decode_s_after_first_token": round(decode_s, 6) if decode_s is not None else None,
    "token_events": chunks,
    "finish_reason": finish_reason,
    "usage": usage,
    "total_s": round(end - start, 6),
}, ensure_ascii=False, indent=2))
PY

curl --fail --silent --show-error "$base/metrics" | tee "$out/metrics.json"
printf '\n'
curl --fail --silent --show-error "$base/metrics/prometheus" | tee "$out/metrics.prom"
printf '\n'

if [[ "$expect_draft_metrics" == 1 ]]; then
  python3 - "$out/metrics.prom" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
values = {}
for line in text.splitlines():
    if line.startswith("#"):
        continue
    match = re.match(r"^(strata:drafts_(?:offered|accepted)_total)\s+([0-9.]+)$", line)
    if match:
        values[match.group(1)] = float(match.group(2))
for name in ("strata:drafts_offered_total", "strata:drafts_accepted_total"):
    if values.get(name, 0) <= 0:
        raise SystemExit(f"{name} must be nonzero after a successful MTP request")
print(f"draft metrics: offered={values['strata:drafts_offered_total']:g} "
      f"accepted={values['strata:drafts_accepted_total']:g}")
PY
fi
