#!/usr/bin/env python3
"""Build the self-contained K100AI runtime layout and a metadata-free tar.gz archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path

PROFILE_FILES = (
    "data/expert-profile.bin",
    "data/dsv41/profile-2.00.bin",
    "data/dsv41/profile-2.00.bin.sha256",
)
SCRIPT_FILES = (
    "serve-k100ai.sh",
    "preflight.sh",
    "render-config.py",
    "setup-runtime.sh",
    "validate-api.sh",
)
DOC_FILES = ("README.md", "VALIDATION.md", "PROVENANCE.md")
SOURCE_SKIP_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "run",
}
SOURCE_SKIP_PREFIXES = (".", "build-", "dist-", "release-build-", "src-prefill-", "src-qwen-")
SOURCE_SKIP_SUFFIXES = (".log", ".logits", ".tar.gz", ".tar.zst")
MODEL_WEIGHT_SUFFIXES = {
    ".gguf",
    ".onnx",
    ".pt",
    ".pth",
    ".safetensors",
}
RESEARCH_SOURCE_SKIP = {
    "configs/k100ai/dsv41-server.json",
    "configs/k100ai/qwen-server.json",
    "scripts/k100ai/build-in-dtk.sh",
    "scripts/k100ai/build-qwen-remote.sh",
    "scripts/k100ai/pack-dsv41-2.00.sh",
    "scripts/k100ai/run-qwen-ab.sh",
    "scripts/k100ai/run-qwen-parity.sh",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalized_add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    """Add one path with stable ownership, timestamps, and archive ordering."""
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if info.isfile():
        with path.open("rb") as source:
            tar.addfile(info, source)
    else:
        tar.addfile(info)


def source_ignore(src: Path):
    """Ignore build products while retaining an extracted, directly readable source tree."""
    ignored_dot_dirs = {".git", ".venv"}

    def ignore(directory: str, names: list[str]) -> set[str]:
        out = set()
        for name in names:
            path = Path(directory) / name
            if path == src:
                continue
            relative = path.relative_to(src).as_posix()
            if name in SOURCE_SKIP_NAMES or name in ignored_dot_dirs:
                out.add(name)
            elif relative in RESEARCH_SOURCE_SKIP:
                # Exclude diagnostic build launchers whose defaults point at
                # host-specific research build directories.
                out.add(name)
            elif directory == str(src) and name.startswith(SOURCE_SKIP_PREFIXES):
                out.add(name)
            elif relative.startswith("data/") and (
                name in {"experts.bin", "dense.bin", "dense.txt"}
                or (name.startswith("draft_vocab") and name.endswith(".bin"))
            ):
                # The Qwen MTP runtime is external to both the runtime layout
                # and the nested source tree. Never let a tracked vocabulary
                # or draft layer turn into an implicit release payload.
                out.add(name)
            elif name.endswith(SOURCE_SKIP_SUFFIXES):
                out.add(name)
            elif path.suffix.lower() in MODEL_WEIGHT_SUFFIXES:
                # The nested source tree must not smuggle model weights or
                # model-derived control vectors into a release archive.
                out.add(name)
            elif name == "__pycache__":
                out.add(name)
        return out

    return ignore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--qwen-bin", type=Path, required=True)
    parser.add_argument("--qwen-current-bin", type=Path)
    parser.add_argument("--qwen-fast-source-commit", required=True)
    parser.add_argument("--dsv41-bin", type=Path, required=True)
    parser.add_argument("--ds4-test-bin", type=Path, required=True)
    parser.add_argument("--ggml-dir", type=Path)
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-source-commit", required=True)
    parser.add_argument("--package-source-commit", required=True)
    parser.add_argument("--engine-version", default="0.1.55")
    args = parser.parse_args()

    src = args.source.resolve()
    root = args.output.resolve()
    if root.exists():
        raise SystemExit(f"output already exists: {root}")
    required_binaries = [args.qwen_bin, args.dsv41_bin, args.ds4_test_bin]
    if args.qwen_current_bin is not None:
        required_binaries.append(args.qwen_current_bin)
    for binary in required_binaries:
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise SystemExit(f"missing executable: {binary}")

    root.mkdir(parents=True)
    (root / "bin").mkdir()
    shutil.copy2(args.qwen_bin, root / "bin/strata")
    if args.qwen_current_bin is not None:
        shutil.copy2(args.qwen_current_bin, root / "bin/strata-0.1.55")
    shutil.copy2(args.dsv41_bin, root / "bin/strata-dsv41")
    shutil.copy2(args.ds4_test_bin, root / "bin/ds4_gguf_test")
    for binary in (root / "bin").iterdir():
        binary.chmod(0o755)

    cache_ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    if args.ggml_dir is not None:
        gguf_py = args.ggml_dir / "gguf-py"
        if not (gguf_py / "gguf").is_dir():
            raise SystemExit(f"missing gguf-py: {gguf_py}")
        llama_root = root / "third_party/llama.cpp"
        llama_root.mkdir(parents=True)
        shutil.copytree(gguf_py, llama_root / "gguf-py", symlinks=True, ignore=cache_ignore)
        shutil.copy2(args.ggml_dir / "LICENSE", llama_root / "LICENSE")

    shutil.copytree(src / "serve", root / "serve", symlinks=True, ignore=cache_ignore)
    shutil.copytree(src / "strata_control", root / "strata_control", symlinks=True, ignore=cache_ignore)
    (root / "tools").mkdir()
    # serve/server.py adds tools/ to sys.path and imports workspace modules such as strata_tokenizer and gguf_reader
    # directly. Keep every top-level helper module; nested research-only directories are not runtime dependencies.
    for source in sorted((src / "tools").glob("*.py")):
        shutil.copy2(source, root / "tools" / source.name)

    (root / "configs/k100ai").mkdir(parents=True)
    for name in ("qwen-server.template.json", "dsv41-server.template.json"):
        shutil.copy2(src / "configs/k100ai" / name, root / "configs/k100ai" / name)

    for relative in PROFILE_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / relative, target)

    (root / "scripts/k100ai").mkdir(parents=True)
    for name in SCRIPT_FILES:
        target = root / "scripts/k100ai" / name
        shutil.copy2(src / "scripts/k100ai" / name, target)
        target.chmod(0o755)

    (root / "docs/k100ai").mkdir(parents=True)
    for name in DOC_FILES:
        source = src / "docs/k100ai" / name
        if source.is_file():
            shutil.copy2(source, root / "docs/k100ai" / name)
    shutil.copytree(src, root / "source", symlinks=True, ignore=source_ignore(src))
    shutil.copy2(src / "requirements.txt", root / "requirements.txt")
    shutil.copy2(src / "LICENSE", root / "LICENSE")

    build = {
        "engine_version": args.engine_version,
        "engine_source_commit": args.engine_source_commit,
        "package_source_commit": args.package_source_commit,
        "container_image": args.base_image,
        "platform": "linux-x86_64-gfx928-dtk2604",
        "models_included": False,
        "qwen_engines": {
            "default_fast": {
                "binary": "bin/strata",
                "engine_version": "0.1.47",
                "engine_source_commit": args.qwen_fast_source_commit,
                "numeric_profile": "drift-accepted-fast",
            },
            "current": {
                "binary": "bin/strata-0.1.55",
                "engine_version": "0.1.55",
                "engine_source_commit": args.engine_source_commit,
                "numeric_profile": "current-exact-expert",
            },
        },
        "dsv41_engine": {
            "binary": "bin/strata-dsv41",
            "engine_version": args.engine_version,
            "engine_source_commit": args.engine_source_commit,
        },
    }
    (root / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")

    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            rows.append(f"{digest(path)}  {path.relative_to(root).as_posix()}\n")
    (root / "SHA256SUMS").write_text("".join(rows), encoding="utf-8")

    archive = root.parent / f"{root.name}.tar.gz"
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    with archive.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in (root, *sorted(root.rglob("*"), key=lambda item: item.as_posix())):
                    normalized_add(tar, path, path.relative_to(root.parent).as_posix())

    print(f"package={root}")
    print(f"archive={archive}")
    print(f"archive_sha256={digest(archive)}")
    print(f"files={len(rows) + 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
