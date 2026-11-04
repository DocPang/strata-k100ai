#!/usr/bin/env python3
"""Compare two raw little-endian float32 first-token logit dumps."""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def load_raw(path: Path) -> tuple[tuple[float, ...], tuple[int, ...]]:
    raw = path.read_bytes()
    if len(raw) % 4:
        raise SystemExit(f"{path}: size {len(raw)} is not a multiple of 4")
    count = len(raw) // 4
    return struct.unpack(f"<{count}f", raw), struct.unpack(f"<{count}I", raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()

    left, left_bits = load_raw(args.left)
    right, right_bits = load_raw(args.right)
    if len(left) != len(right):
        raise SystemExit(f"length mismatch: {len(left)} != {len(right)}")

    finite_pairs = [
        (a, b) for a, b in zip(left, right)
        if math.isfinite(a) and math.isfinite(b)
    ]
    squared_error = sum((b - a) ** 2 for a, b in finite_pairs)
    squared_left = sum(a * a for a in left if math.isfinite(a))
    max_abs = max((abs(b - a) for a, b in finite_pairs), default=float("nan"))
    rel_l2 = math.sqrt(squared_error / squared_left) if squared_left else math.sqrt(squared_error)

    print(f"values {len(left)}")
    print(f"argmax {max(range(len(left)), key=left.__getitem__)} {max(range(len(right)), key=right.__getitem__)}")
    print(f"value_ndiff {sum(a != b for a, b in zip(left, right))}")
    print(f"bit_ndiff {sum(a != b for a, b in zip(left_bits, right_bits))}")
    print(f"nonfinite_pairs {len(left) - len(finite_pairs)}")
    print(f"max_abs {max_abs:.9g}")
    print(f"rel_l2 {rel_l2:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
