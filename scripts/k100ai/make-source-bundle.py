#!/usr/bin/env python3
"""Create the source inputs needed to rebuild the K100AI release engines."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

SKIP_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "dist",
    "run",
}
SKIP_PREFIXES = ("build-", "dist-", "release-build-", "src-prefill-", "src-qwen-")
SKIP_SUFFIXES = (".gguf", ".onnx", ".pt", ".pth", ".safetensors", ".log", ".tar.gz", ".tar.zst")
RUNTIME_DATA_NAMES = {
    "data/experts.bin",
    "data/dense.bin",
    "data/dense.txt",
}
RUNTIME_DATA_PREFIXES = ("data/draft_vocab",)


def ignore_tree(src: Path):
    def ignore(directory: str, names: list[str]) -> set[str]:
        out: set[str] = set()
        for name in names:
            path = Path(directory) / name
            rel = path.relative_to(src)
            if name in SKIP_NAMES or name == "__pycache__":
                out.add(name)
            elif len(rel.parts) == 1 and name.startswith(SKIP_PREFIXES):
                out.add(name)
            elif (
                len(rel.parts) == 1
                and name.startswith(".")
                and name not in {".github", ".gitignore", ".gitattributes"}
            ):
                out.add(name)
            elif rel.as_posix() in RUNTIME_DATA_NAMES:
                out.add(name)
            elif rel.as_posix().startswith(RUNTIME_DATA_PREFIXES) and name.endswith(".bin"):
                out.add(name)
            elif path.suffix.lower() in SKIP_SUFFIXES:
                out.add(name)
        return out

    return ignore


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: Path) -> str:
    """Hash paths, sizes, and contents without depending on timestamps or owners."""
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        h.update(str(path.stat().st_size).encode("ascii") + b"\0")
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def normalized_add(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    info = tar.gettarinfo(str(path), arcname=arcname)
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    if info.isfile():
        with path.open("rb") as f:
            tar.addfile(info, f)
    else:
        tar.addfile(info)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-src", type=Path, required=True)
    parser.add_argument("--fast-src", type=Path, required=True)
    parser.add_argument("--llama-src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-commit", required=True)
    parser.add_argument("--fast-commit", required=True)
    parser.add_argument("--engine-commit", required=True)
    parser.add_argument("--container-image", required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    staging = args.output.parent / f".{args.output.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    for name, source in (("main", args.main_src), ("qwen-fast-0.1.47", args.fast_src), ("llama.cpp", args.llama_src)):
        if not source.is_dir():
            raise SystemExit(f"missing source tree: {source}")
        shutil.copytree(source, staging / name, symlinks=True, ignore=ignore_tree(source))

    build_inputs = {
        "main_source_commit": args.main_commit,
        "qwen_fast_source_commit": args.fast_commit,
        "engine_source_commit": args.engine_commit,
        "container_image": args.container_image,
        "source_trees": {},
    }
    for name in ("main", "qwen-fast-0.1.47", "llama.cpp"):
        files = sorted(p for p in (staging / name).rglob("*") if p.is_file())
        build_inputs["source_trees"][name] = {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "sha256": tree_digest(staging / name),
        }
    (staging / "BUILD_INPUTS.json").write_text(
        json.dumps(build_inputs, indent=2) + "\n", encoding="utf-8"
    )

    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for path in sorted(staging.rglob("*"), key=lambda p: p.as_posix()):
                    arcname = f"{args.output.name.removesuffix('.tar.gz')}/{path.relative_to(staging).as_posix()}"
                    normalized_add(tar, path, arcname)
    shutil.rmtree(staging)
    print(f"source_bundle={args.output}")
    print(f"source_bundle_sha256={digest(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
