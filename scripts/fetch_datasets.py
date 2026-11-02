#!/usr/bin/env python3
"""Fetch SuiteSparse matrices and convert them to QR8 dense column-major files.

The script never changes matrix values. A manifest-requested transpose is recorded
in metadata. Archive and output SHA-256 hashes make a completed acquisition
snapshot reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import struct
import tarfile
import urllib.request

import numpy as np

MAGIC = b"QR8MAT01"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_matrix_market(path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    with path.open("rt", encoding="ascii") as stream:
        banner = stream.readline().strip().lower().split()
        if len(banner) != 5 or banner[:2] != ["%%matrixmarket", "matrix"]:
            raise ValueError(f"invalid Matrix Market banner in {path}")
        _, _, representation, field, symmetry = banner
        if representation != "coordinate":
            raise ValueError(f"only coordinate Matrix Market files are supported: {path}")
        if field == "complex":
            raise ValueError(f"complex matrix is outside the real benchmark contract: {path}")
        comments = []
        line = stream.readline()
        while line.startswith("%"):
            comments.append(line.rstrip("\n"))
            line = stream.readline()
        rows, cols, entries = (int(value) for value in line.split())
        matrix = np.zeros((rows, cols), dtype=np.float64, order="F")
        for _ in range(entries):
            fields = stream.readline().split()
            row, col = int(fields[0]) - 1, int(fields[1]) - 1
            value = 1.0 if field == "pattern" else float(fields[2])
            matrix[row, col] += value
            if symmetry in {"symmetric", "hermitian"} and row != col:
                matrix[col, row] += value
            elif symmetry == "skew-symmetric" and row != col:
                matrix[col, row] -= value
        return matrix, comments


def write_qr8(path: pathlib.Path, matrix: np.ndarray) -> None:
    matrix = np.asfortranarray(matrix, dtype=np.float64)
    with path.open("wb") as stream:
        stream.write(MAGIC)
        stream.write(struct.pack("<QQ", matrix.shape[0], matrix.shape[1]))
        stream.write(matrix.tobytes(order="F"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, default=pathlib.Path("config/datasets.json"))
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    lock_path = args.output / "datasets.lock.json"
    previous = {}
    if lock_path.exists():
        previous_lock = json.loads(lock_path.read_text(encoding="utf-8"))
        previous = {item["id"]: item for item in previous_lock.get("matrices", [])}
    lock = {"schema_version": 1, "source_manifest": str(args.manifest), "matrices": []}

    for item in manifest["matrices"]:
        key = f"{item['group']}_{item['name']}"
        archive = args.output / f"{key}.tar.gz"
        url = manifest["download_pattern"].format(**item)
        if args.force or not archive.exists():
            print(f"download {url}")
            urllib.request.urlretrieve(url, archive)
        extract_dir = args.output / f".{key}.extract"
        extract_dir.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:gz") as bundle:
            expected_name = f"{item['name']}.mtx"
            members = [member for member in bundle.getmembers()
                       if member.isfile() and pathlib.PurePosixPath(member.name).name == expected_name]
            if len(members) != 1:
                raise RuntimeError(f"expected one .mtx file in {archive}, found {len(members)}")
            member = members[0]
            member_path = pathlib.Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise RuntimeError(f"unsafe archive path: {member.name}")
            bundle.extract(member, extract_dir)
        matrix, matrix_market_comments = read_matrix_market(extract_dir / member.name)
        if item["transpose"]:
            matrix = np.asfortranarray(matrix.T)
        if matrix.shape[0] < matrix.shape[1]:
            raise RuntimeError(f"{key} remains wide after manifest transform: {matrix.shape}")
        output = args.output / f"{key}.qr8"
        write_qr8(output, matrix)
        metadata = {
            "id": key,
            "source_url": url,
            "archive_sha256": sha256(archive),
            "output_sha256": sha256(output),
            "rows": matrix.shape[0],
            "columns": matrix.shape[1],
            "transpose": item["transpose"],
            "class": item["class"],
            "license": manifest["license"],
            "license_url": manifest["license_url"],
            "matrix_market_comments": matrix_market_comments
        }
        prior = previous.get(key, {})
        if prior.get("output_sha256") == metadata["output_sha256"]:
            for field in ("measured_rank_fp64", "sigma_max_fp64", "sigma_min_fp64", "condition_2_fp64"):
                if field in prior:
                    metadata[field] = prior[field]
        (args.output / f"{key}.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        lock["matrices"].append(metadata)
        print(f"prepared {output}: {matrix.shape[0]} x {matrix.shape[1]}")

    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
