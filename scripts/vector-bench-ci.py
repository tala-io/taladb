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


def measure(repo, template, count, repeats, quantization, queries, batch_size=None, target_dir=None, workload="hnsw"):
    package = tomllib.loads((repo / "packages/core/Cargo.toml").read_text())["package"]["name"]
    crate = package.replace("-", "_")
    # The baseline runs the candidate's harness, not a potentially stale suite.
    harness = repo / "packages/core/examples/vector_ci_profile.rs"
    if harness.exists():
        raise RuntimeError(f"temporary harness already exists: {harness}")
    harness.write_text(template.replace("taladb::", f"{crate}::"))
    samples = []
    env = os.environ.copy()
    if target_dir is not None:
        env["CARGO_TARGET_DIR"] = str(target_dir)
    try:
        for _ in range(repeats):
            command = [
                "cargo", "run", "--locked", "--release", "-p", package,
                "--example", "vector_ci_profile", "--", str(count),
            ]
            if workload == "hnsw":
                command.extend(["0.6", "--json", "--quantization", quantization])
            command.extend(["--queries", str(queries)])
            if batch_size is not None:
                command.extend(["--batch-size", str(batch_size)])
            result = subprocess.run(command, cwd=repo, env=env, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(result.stderr)
            samples.append(json.loads(result.stdout))
    finally:
        harness.unlink(missing_ok=True)
    report = samples[0].copy()
    if workload == "filtered":
        for sample in samples[1:]:
            if filter_signature(sample) != filter_signature(samples[0]):
                raise RuntimeError("filtered workload results changed between repeated runs")
        report["cases"] = [dict(row,
            p50_ms=statistics.median(s["cases"][i]["p50_ms"] for s in samples),
            p95_ms=statistics.median(s["cases"][i]["p95_ms"] for s in samples))
            for i, row in enumerate(samples[0]["cases"])]
    else:
        aggregate_hnsw(report, samples)
    report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    report["samples"] = samples
    return report


def aggregate_hnsw(report, samples):
    for key in ("insert_ms", "build_ms", "exact_mean_ms"):
        report[key] = statistics.median(s[key] for s in samples)
    if report.get("step_p95_ms") is not None:
        report["step_p95_ms"] = statistics.median(s["step_p95_ms"] for s in samples)
    report["ann"] = []
    for index, row in enumerate(samples[0]["ann"]):
        report["ann"].append({key: row[key] if key == "ef_search" else statistics.median(s["ann"][index][key] for s in samples)
                              for key in row})


def filter_signature(report):
    return (tuple(report[k] for k in ("count", "dimensions", "queries", "body_bytes")),
            [(r["mode"], r["warm"], r["filter"], r["fingerprint"], r["distances"])
             for r in report["cases"]])


def compare_filters(before, after):
    if filter_signature(before) != filter_signature(after):
        return ["filtered workload inputs, result fingerprints or distance counts changed"]
    failures = []
    for old, new in zip(before["cases"], after["cases"], strict=True):
        if new["p50_ms"] > max(old["p50_ms"] * 1.5, old["p50_ms"] + 0.25):
            failures.append(f"{new['mode']} {new['filter']} warm={new['warm']} median latency regressed by more than 50% and 0.25 ms")
    return failures


def compare(before, after):
    if "cases" in before or "cases" in after:
        return compare_filters(before, after)
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
    parser.add_argument("--quantization", choices=("none", "scalar", "binary"), default="none")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--workload", choices=("hnsw", "filtered"), default="hnsw")
    parser.add_argument("--batch-size", type=int, help="use resumable builds with 1..1024 vectors per step")
    parser.add_argument("--output", type=Path, default=Path("vector-benchmark.json"))
    args = parser.parse_args()
    if args.count < 10 or args.repeats < 1 or args.queries < 1:
        parser.error("count must be >= 10; repeats and queries must be positive")
    if args.batch_size is not None and not 1 <= args.batch_size <= 1024:
        parser.error("batch size must be between 1 and 1024")
    if args.workload == "filtered" and (args.batch_size is not None or args.quantization != "none"):
        parser.error("filtered workloads use a fixed binary graph; batch size and quantization apply only to hnsw")
    example = "vector_filter_profile.rs" if args.workload == "filtered" else "hnsw_profile.rs"
    template = (args.candidate / "packages/core/examples" / example).read_text()
    # Separate artifacts: a shared target directory can reuse a library from the
    # other checkout when its source timestamps predate that checkout's build.
    target_root = Path(os.environ.get("CARGO_TARGET_DIR", args.candidate.resolve() / "target")).resolve()
    before = measure(args.baseline.resolve(), template, args.count, args.repeats, args.quantization, args.queries, args.batch_size, target_root / "vector-baseline", args.workload)
    after = measure(args.candidate.resolve(), template, args.count, args.repeats, args.quantization, args.queries, args.batch_size, target_root / "vector-candidate", args.workload)
    failures = compare(before, after)
    args.output.write_text(json.dumps({"schema": 1, "workload": args.workload, "platform": platform.platform(),
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "baseline": before, "candidate": after, "failures": failures}, indent=2) + "\n")
    quantization = "binary" if args.workload == "filtered" else args.quantization
    lines = [f"Vector benchmark: {args.count:,} vectors, {before['dimensions']} dimensions, {args.workload} workload, {quantization} quantization, "
             f"{args.queries} queries, {'sync' if args.batch_size is None else f'batch size {args.batch_size}'}, "
             f"median of {args.repeats} runs",
             "", "| Metric | Baseline | Candidate |", "|---|---:|---:|"]
    if args.workload == "filtered":
        for old, new in zip(before["cases"], after["cases"], strict=True):
            label = f"{new['mode']} {new['filter']} {'warm' if new['warm'] else 'cold'} p50/p95 ms"
            lines.append(f"| {label} | {old['p50_ms']:.3f}/{old['p95_ms']:.3f} | {new['p50_ms']:.3f}/{new['p95_ms']:.3f} |")
    else:
        for key in ("build_ms", "exact_mean_ms"):
            lines.append(f"| {key} | {before[key]:.3f} | {after[key]:.3f} |")
        if args.batch_size is not None:
            lines.append(f"| Build step p95 ms | {before['step_p95_ms']:.3f} | {after['step_p95_ms']:.3f} |")
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
