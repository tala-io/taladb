#!/usr/bin/env python3
"""Compare identical vector workloads on two checkouts on the same runner."""
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import tomllib


def measure(repo, template, count, repeats):
    package = tomllib.loads((repo / "packages/core/Cargo.toml").read_text())["package"]["name"]
    crate = package.replace("-", "_")
    # The baseline runs the candidate's harness, not a potentially stale suite.
    harness = repo / "packages/core/examples/vector_ci_profile.rs"
    if harness.exists():
        raise RuntimeError(f"temporary harness already exists: {harness}")
    harness.write_text(template.replace("taladb::", f"{crate}::"))
    samples = []
    try:
        for _ in range(repeats):
            result = subprocess.run([
                "cargo", "run", "--locked", "--release", "-p", package,
                "--example", "vector_ci_profile", "--", str(count), "0.6", "--json",
            ], cwd=repo, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            samples.append(json.loads(result.stdout))
    finally:
        harness.unlink(missing_ok=True)
    report = samples[0].copy()
    for key in ("insert_ms", "build_ms", "exact_mean_ms"):
        report[key] = statistics.median(s[key] for s in samples)
    report["ann"] = []
    for index, row in enumerate(samples[0]["ann"]):
        report["ann"].append({key: row[key] if key == "ef_search" else statistics.median(s["ann"][index][key] for s in samples)
                              for key in row})
    report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report["samples"] = samples
    return report


def compare(before, after):
    failures = []
    if after["build_ms"] > max(before["build_ms"] * 1.3, before["build_ms"] + 500):
        failures.append("graph build regressed by more than 30% and 500 ms")
    if after["exact_mean_ms"] > max(before["exact_mean_ms"] * 1.3, before["exact_mean_ms"] + 0.25):
        failures.append("exact search regressed by more than 30% and 0.25 ms")
    for old, new in zip(before["ann"], after["ann"], strict=True):
        assert old["ef_search"] == new["ef_search"]
        if new["recall_at_10"] + 0.03 < old["recall_at_10"]:
            failures.append(f"recall at ef={new['ef_search']} fell by more than 3 percentage points")
        if new["p50_ms"] > max(old["p50_ms"] * 1.5, old["p50_ms"] + 0.25):
            failures.append(f"ANN median latency at ef={new['ef_search']} regressed by more than 50% and 0.25 ms")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("vector-benchmark.json"))
    args = parser.parse_args()
    if args.count < 10 or args.repeats < 1:
        parser.error("count must be >= 10 and repeats must be positive")
    template = (args.candidate / "packages/core/examples/hnsw_profile.rs").read_text()
    before = measure(args.baseline.resolve(), template, args.count, args.repeats)
    after = measure(args.candidate.resolve(), template, args.count, args.repeats)
    failures = compare(before, after)
    args.output.write_text(json.dumps({"schema": 1, "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "baseline": before, "candidate": after, "failures": failures}, indent=2) + "\n")
    lines = [f"Vector benchmark: {args.count:,} vectors, 384 dimensions, median of {args.repeats} runs",
             "", "| Metric | Baseline | Candidate |", "|---|---:|---:|"]
    for key in ("build_ms", "exact_mean_ms"):
        lines.append(f"| {key} | {before[key]:.3f} | {after[key]:.3f} |")
    for old, new in zip(before["ann"], after["ann"], strict=True):
        lines.append(f"| ANN ef={new['ef_search']} p50/p95 ms | {old['p50_ms']:.3f}/{old['p95_ms']:.3f} | {new['p50_ms']:.3f}/{new['p95_ms']:.3f} |")
        lines.append(f"| Recall@10 ef={new['ef_search']} | {old['recall_at_10']:.1%} | {new['recall_at_10']:.1%} |")
    lines.extend(["", *(f"FAIL: {message}" for message in failures)])
    summary = "\n".join(lines) + "\n"
    print(summary)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as file:
            file.write(summary)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
