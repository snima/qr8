#!/usr/bin/env python3
"""Structural and numerical invariants for raw benchmark records."""

from __future__ import annotations

import argparse
import csv
import math


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file")
    args = parser.parse_args()
    with open(args.csv_file, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise AssertionError("no result rows")
    for row in rows:
        success = row["factorization_success"] == "1"
        passes = int(row["passes"])
        refinements = int(row["refinements"])
        algorithm = row["algorithm"]
        if success and algorithm == "FP64-CholeskyQR2":
            assert (passes, refinements) == (2, 1)
        if success and algorithm == "FP64-shiftedCholeskyQR":
            assert (passes, refinements) == (1, 0)
            assert float(row["shift"]) > 0.0
        if success and algorithm == "FP64-shiftedCholeskyQR3":
            assert (passes, refinements) == (3, 2)
            assert float(row["shift"]) > 0.0
        if row["stop_reason"] == "tolerance_met":
            assert row["target_met"] == "1"
        if row["target_met"] == "1":
            assert math.isfinite(float(row["final_orth_fro"]))
        if success and row["output"] == "QR":
            assert math.isfinite(float(row["residual_fro"]))
            assert float(row["residual_fro"]) < 1.0e-10
        assert float(row["total_ms"]) >= 0.0
    print(f"validated {len(rows)} rows from {args.csv_file}")


if __name__ == "__main__":
    main()
