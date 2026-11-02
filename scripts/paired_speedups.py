#!/usr/bin/env python3
"""Compute paired INT8-over-adaptive-FP64 speedups from raw trials only."""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import statistics

BASELINE = "FP64-CholeskyQR-Adaptive-Control"
INT8_METHODS = ("INT8L1-CholeskyQR-Adaptive", "INT8L2-CholeskyQR-Adaptive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_csv", type=pathlib.Path)
    parser.add_argument("output_csv", type=pathlib.Path)
    args = parser.parse_args()
    with args.raw_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    key_fields = ("matrix_id", "source_kind", "requested_condition", "seed", "m", "n",
                  "output", "equilibrated", "timing_mode")
    groups: dict[tuple[str, ...], dict[str, dict[str, dict[str, str]]]] = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        groups.setdefault(key, {}).setdefault(row["algorithm"], {})[row["trial"]] = row
    fields = list(key_fields) + ["method", "paired_trials", "mean_speedup", "stddev_speedup",
                                 "ci95_half_width", "minimum_speedup", "maximum_speedup"]
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, methods in sorted(groups.items()):
            baseline = methods.get(BASELINE, {})
            for method in INT8_METHODS:
                candidate = methods.get(method, {})
                speedups = []
                for trial in sorted(set(baseline) & set(candidate), key=int):
                    reference, test = baseline[trial], candidate[trial]
                    if reference["factorization_success"] != "1" or test["factorization_success"] != "1":
                        continue
                    if reference["target_met"] != "1" or test["target_met"] != "1":
                        continue
                    speedups.append(float(reference["total_ms"]) / float(test["total_ms"]))
                record = dict(zip(key_fields, key))
                record["method"] = method
                record["paired_trials"] = len(speedups)
                if speedups:
                    mean = statistics.fmean(speedups)
                    deviation = statistics.stdev(speedups) if len(speedups) > 1 else 0.0
                    record.update(mean_speedup=mean, stddev_speedup=deviation,
                                  ci95_half_width=1.96 * deviation / math.sqrt(len(speedups)),
                                  minimum_speedup=min(speedups), maximum_speedup=max(speedups))
                else:
                    record.update(mean_speedup=math.nan, stddev_speedup=math.nan,
                                  ci95_half_width=math.nan, minimum_speedup=math.nan,
                                  maximum_speedup=math.nan)
                writer.writerow(record)


if __name__ == "__main__":
    main()
