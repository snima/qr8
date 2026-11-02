#!/usr/bin/env python3
"""Compute rank and condition metadata from prepared dense real matrices."""

from __future__ import annotations

import argparse
import json
import pathlib
import struct

import numpy as np


def load(path: pathlib.Path) -> np.ndarray:
    with path.open("rb") as stream:
        if stream.read(8) != b"QR8MAT01":
            raise ValueError(f"invalid QR8 matrix: {path}")
        rows, columns = struct.unpack("<QQ", stream.read(16))
        values = np.fromfile(stream, dtype="<f8", count=rows * columns)
    return values.reshape((rows, columns), order="F")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=pathlib.Path, default=pathlib.Path("data/datasets.lock.json"))
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    for item in lock["matrices"]:
        path = args.lock.parent / f"{item['id']}.qr8"
        matrix = load(path)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        tolerance = singular_values[0] * max(matrix.shape) * np.finfo(np.float64).eps
        item["measured_rank_fp64"] = int(np.count_nonzero(singular_values > tolerance))
        item["sigma_max_fp64"] = float(singular_values[0])
        item["sigma_min_fp64"] = float(singular_values[-1])
        item["condition_2_fp64"] = float(singular_values[0] / singular_values[-1]) if singular_values[-1] else "inf"
        print(item["id"], item["measured_rank_fp64"], item["condition_2_fp64"])
    args.lock.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
