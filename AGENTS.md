# AGENTS.md

This repository is the `strata-k100ai` product branch for **Hygon K100AI / gfx928** on
**Linux x86_64 with DTK 2604**. Product version is **v0.3.0**.

Read [docs/k100ai/README.md](docs/k100ai/README.md) first. It is the index for the
current installation, runtime, model, validation, provenance and reproduction
documents. Historical v0.2.3 material is under
[docs/k100ai/archive/v0.2.3](docs/k100ai/archive/v0.2.3/README.md).

## Supported models

- **DeepSeek-V4.1-Flash**: `bin/strata-dsv41`, with EXL3 TP1, EXL3 TP4 and
  GGUF TP2 production configurations.
- **Qwen3.8-Flash-Next**: `bin/strata-0.1.47` is the default fast engine and
  `bin/strata-0.1.55` is the exact comparison engine.

The release contains engines and runtime code. Model weights, tokenizers, MTP
runtime files and DeepSeek expert profiles are external assets.

## Project layout

- `serving and API`: `serve/`, `strata_control/`
- `engine`: `src/`, `include/`, `cmake/`, `third_party/`
- `K100AI assets and tests`: `data/`, `tests/`, `scripts/k100ai/`
- `model configurations`: `configs/k100ai/`
- `K100AI documentation`: `docs/k100ai/`
- `desktop and packaging`: `apps/`, `packaging/`

Keep current product documents at stable, non-versioned paths. Put release notes
under `docs/k100ai/release-notes/` and retired instructions under
`docs/k100ai/archive/<version>/`.

## Documentation standards

- Follow the official Strata page structure: logo and title, short product
  statement, inline navigation, then Models, Install, Hardware, Tools and Help.
- Keep the README concise. Put implementation details, raw timings and long
  evidence chains in `docs/k100ai/`.
- Use Chinese for the primary README and K100AI guides. Use plain language and
  short sections.
- Link to canonical files instead of duplicating versioned instructions.
- Use `tok/s` for input and output speed. Preserve seconds only when needed to
  explain the original TTFT evidence, and state that it is a conversion basis.
- Do not describe estimated end-to-end throughput as engine prefill timing.
- Put Decode results in the speculative-decoding table only when the
  speculative path was enabled. Keep native controls separate.
- Every performance claim must name the model, topology, timing source and
  sample scope. Do not present a median as a fixed guarantee.

## Setup and validation

Start with [docs/k100ai/INSTALL.md](docs/k100ai/INSTALL.md) and
[docs/k100ai/RUNTIME.md](docs/k100ai/RUNTIME.md). The public configuration index
is [configs/k100ai/README.md](configs/k100ai/README.md).

Before starting a GPU service:

```sh
bash scripts/k100ai/preflight.sh
```

Quick checks that do not need a GPU:

```sh
python3 -m py_compile scripts/k100ai/*.py
bash -n scripts/k100ai/*.sh
python3 -m json.tool configs/k100ai/exl3-tp1-prod.json >/dev/null
python3 -m json.tool configs/k100ai/exl3-tp4-prod.json >/dev/null
python3 -m json.tool configs/k100ai/gguf-tp2-prod.json >/dev/null
git diff --check
```

Use [docs/k100ai/VALIDATION.md](docs/k100ai/VALIDATION.md) for Qwen evidence and
[docs/k100ai/DEEPSEEK41.md](docs/k100ai/DEEPSEEK41.md) for DeepSeek production
results. Do not mix v0.2.3 measurements with v0.3.0 configuration claims without
marking the historical scope.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md). Start issue and pull request titles with
an area: `[Engine]`, `[GUI]`, `[TUI]`, `[Benchmark]`, `[Docs]` or `[Other]`.

- One problem per pull request.
- Lead with the problem and result, then the approach, validation and limits.
- Mark unverified source findings as static analysis.
- Do not submit logs or measurements you did not produce.
- Build every backend touched by a change.
- Keep new features opt-in and preserve the existing default unless a measured
  improvement justifies changing it.
