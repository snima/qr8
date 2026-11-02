#!/usr/bin/env python3
"""Print compact numerical summaries for QR8 raw CSV files.

This script is intentionally small-output: it reports only success/target rates,
representative accuracy extrema, and median timings, so large CSVs do not need to
be inspected manually.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics
from collections import defaultdict


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.csv_files:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (path.name, row["matrix_id"], row["algorithm"], row["output"])
                groups[key].append(row)

    header = ("file", "matrix", "algorithm", "out", "ok", "target", "stop", "ref",
              "deg", "init_orth", "final_orth", "residual", "time_ms")
    print("\t".join(header))
    for (filename, matrix, algorithm, output), rows in sorted(groups.items()):
        successful = [row for row in rows if row["factorization_success"] == "1"]
        targets = [row for row in rows if row["target_met"] == "1"]
        stops = sorted({row["stop_reason"] for row in rows})
        refinements = [int(row["refinements"]) for row in rows]
        degrees = [int(row["chebyshev_degree"]) for row in rows if row.get("chebyshev_degree", "0") != "0"]
        initial = [float(row["initial_orth_fro"]) for row in rows if finite(row["initial_orth_fro"])]
        final = [float(row["final_orth_fro"]) for row in rows if finite(row["final_orth_fro"])]
        residual = [float(row["residual_fro"]) for row in rows if finite(row["residual_fro"])]
        times = [float(row["total_ms"]) for row in successful]
        fields = (
            filename,
            matrix,
            algorithm,
            output,
            f"{len(successful)}/{len(rows)}",
            f"{len(targets)}/{len(rows)}",
            "|".join(stops),
            f"{max(refinements) if refinements else 0}",
            f"{max(degrees) if degrees else 0}",
            f"{max(initial):.3e}" if initial else "nan",
            f"{max(final):.3e}" if final else "nan",
            f"{max(residual):.3e}" if residual else "nan",
            f"{statistics.median(times):.3f}" if times else "nan",
        )
        print("\t".join(fields))


if __name__ == "__main__":
    main()
