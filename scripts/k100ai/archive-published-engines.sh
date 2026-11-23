#!/usr/bin/env bash
set -euo pipefail

# Package a committed K100AI release tree with its versioned engine binaries.
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <git-ref> <output-directory>" >&2
  exit 2
fi

repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
revision=$(git -C "$repo" rev-parse --verify "$1^{commit}")
version=$(git -C "$repo" show "$revision:BUILD_INFO.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["version"])')
[[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
name="strata-k100ai-v$version"
mkdir -p "$2"
out=$(cd -- "$2" && pwd -P)
test ! -e "$out/$name.tar.gz"
test ! -e "$out/$name.tar.gz.sha256"

stage=$(mktemp -d "$out/.strata-package.XXXXXX")
trap 'python3 -c "import shutil,sys; shutil.rmtree(sys.argv[1])" "$stage"' EXIT
git -C "$repo" archive --format=tar --prefix="$name/" "$revision" | tar -xf - -C "$stage"
root="$stage/$name"
(cd "$root" && sha256sum -c --quiet SHA256SUMS)
bash "$root/scripts/k100ai/fetch-llama-gguf-py.sh" "$root"
python3 - "$root" "$stage/$name.tar.gz" <<'PY'
import gzip
import hashlib
from pathlib import Path
import sys
import tarfile

root, archive = map(Path, sys.argv[1:])
llama = root / "third_party/llama.cpp"
manifest = root / "SHA256SUMS"
with manifest.open("a", encoding="utf-8") as stream:
    for path in sorted(p for p in llama.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        stream.write(f"{digest}  {path.relative_to(root).as_posix()}\n")

def normalize(info):
    info.uid = info.gid = info.mtime = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if info.isdir() or info.mode & 0o111 else 0o644
    return info

with archive.open("xb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
    with tarfile.open(fileobj=compressed, mode="w") as tar:
        tar.add(root, arcname=root.name, filter=normalize)
PY
(cd "$root" && sha256sum -c --quiet SHA256SUMS)
mv "$stage/$name.tar.gz" "$out/"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
