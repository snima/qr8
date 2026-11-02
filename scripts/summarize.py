#!/usr/bin/env python3
"""Recompute publication summaries from immutable raw-trial CSV records."""

from __future__ import annotations

import argparse
import csv
import math
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv")
    parser.add_argument("summary_csv")
    args = parser.parse_args()
    with open(args.raw_csv, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    keys = ("matrix_id", "source_kind", "requested_condition", "seed", "m", "n", "algorithm", "output", "equilibrated", "timing_mode")
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    fields = list(keys) + ["successful_trials", "trials", "mean_ms", "stddev_ms", "median_ms", "min_ms"]
    with open(args.summary_csv, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, group in sorted(groups.items()):
            successful = [row for row in group if row["factorization_success"] == "1"]
            times = [float(row["total_ms"]) for row in successful]
            record = dict(zip(keys, key))
            record.update({
                "successful_trials": len(successful),
                "trials": len(group),
                "mean_ms": statistics.fmean(times) if times else math.nan,
                "stddev_ms": statistics.stdev(times) if len(times) > 1 else 0.0 if times else math.nan,
                "median_ms": statistics.median(times) if times else math.nan,
                "min_ms": min(times) if times else math.nan,
            })
            writer.writerow(record)


if __name__ == "__main__":
    main()
