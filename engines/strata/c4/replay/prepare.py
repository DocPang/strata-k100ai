#!/usr/bin/env python3
"""Verify pristine input files, apply the frozen production delta, and verify outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strata-source', type=Path, required=True)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    recipe = Path(__file__).resolve().parent
    manifest = json.loads((recipe / 'manifest.json').read_text())
    output = args.out.resolve()
    if output.exists():
        parser.error('output must not exist; use a fresh directory')
    inputs = [('strata', args.strata_source.resolve()), ('llama', args.llama_source.resolve())]
    for kind, source in inputs:
        if not source.is_dir() or output == source or source in output.parents:
            parser.error(f'{kind}: invalid input/output relationship')
        for name, expected in manifest[kind + '_base_files'].items():
            path = source / name
            if not path.is_file() or path.is_symlink() or sha(path) != expected:
                raise SystemExit(f'{kind}: pristine base mismatch: {name}')
    for name, expected in manifest['patches'].items():
        if sha(recipe / 'patches' / name) != expected:
            raise SystemExit(f'patch hash mismatch: {name}')
    output.mkdir(parents=True)
    results = {}
    for kind, source in inputs:
        destination = output / kind
        for name in manifest[kind + '_base_files']:
            path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, path)
        subprocess.run(['git', 'init', '-q'], cwd=destination, check=True)
        patch = recipe / 'patches' / ('strata-c4.patch' if kind == 'strata' else 'llama-gfx928.patch')
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=destination, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=destination, check=True)
        expected_files = dict(manifest[kind + '_base_files'])
        expected_files.update(manifest[kind + '_output_files'])
        for name, expected in expected_files.items():
            if sha(destination / name) != expected:
                raise SystemExit(f'{kind}: post-apply hash mismatch: {name}')
        results[kind] = {'verified_files': len(expected_files), 'patched_files': len(manifest[kind + '_output_files'])}
    record = {'status': 'source_replay_verified', 'target': manifest['target'], 'manifest_sha256': sha(recipe / 'manifest.json'),
              'results': results, 'clean_build_verified': False, 'runtime_equivalence_verified': False}
    (output / 'SOURCE_REPLAY_VERIFIED.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
