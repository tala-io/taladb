---
title: Vector benchmarks
description: Measure vector search through a browser worker and OPFS on desktops and phones
---

# Vector benchmarks

Run the browser workload on the device where your app will search. It measures
the production worker, WASM and OPFS path, including messages, JSON, exact
rescoring and returned documents. The default profile uses 2,000 synthetic
clustered vectors, 128 dimensions, binary quantization, and an 8 MiB shared
cache budget. Queries are deterministic and independent of collection size.

## Browser worker and OPFS

Build the browser binding, then run the vector suite in a local Chrome install:

```sh
pnpm --filter @taladb/web build
node scripts/bench-web.mjs --vectors --json
```

Set `CHROME_BIN` to an existing Chrome binary if it is not installed in a
standard location. For a more constrained profile:

```sh
node scripts/bench-web.mjs --vectors --count 5000 --dims 384 \
  --queries 30 --quantization scalar --cache-bytes 1048576 --json
```

To exercise adaptive sizing and pressure/recovery on the same browser or phone:

```sh
node scripts/bench-web.mjs --vectors --cache-bytes auto \
  --memory-hint-bytes 2147483648 --pressure --json
```

This simulates a 2 GiB host hint and runs normal → moderate → critical → normal
pressure after the timing suite. `pressureCycle` records latency, recall,
fallbacks and cache accounting for dense grouped queries, and checks that
retention is evicted and the baseline restored. These are injected signals,
not an OS low-memory test. Omit `--memory-hint-bytes` to use the browser's hint
when available, or its platform fallback. Fixed-budget profiles also support
`--pressure`. Use `--serve` to run this profile manually on a phone.

Results include build time and batch-step p50/p95 for 32-vector steps. For
unfiltered queries, a 95% dense filter, dense grouped pagination, one group per
document, a 10% tenant filter, a 1% bucket filter, a skewed AND intersection, a
1% scalar window, and an array comparison combined with the bucket filter,
the suite records exact mode and ANN at `efSearch` 64, 100 and 200:

- Worker/database reopen time and the first query after reopening.
- Warm query p50/p95, distance computations, and recall against exact results
  from the same query and filter.
- Queued bursts of four concurrent UI requests, including queue latency p50/p95
  and memory-fallback counts. Set `--concurrency 1..16` to change the burst size.
  The worker serializes core operations; these bursts do not run Rust searches
  in parallel.
- Retained bytes and peak reserved search allowance when the binding supports
  `cacheStats`; these are conservative estimates, not measured peak heap/RSS.
- Stable result fingerprints using document ordinals and rescored f32 scores.
- Origin memory before and after the suite, where the browser supports
  `measureUserAgentSpecificMemory`. Unsupported or timed-out measurements are
  `null`. These measurements include JavaScript and workers; they are not the
  graph cache size or a process peak-memory measurement.

The AND case combines the 10% tenant branch with the 1% bucket branch. The
scalar window constrains indexed ordinals from both sides. The array case
requires separate elements to satisfy contradictory scalar bounds; it catches
incorrect range narrowing while measuring a multi-index filter. Exact ground
truth uses the same filter and group/pagination options, and result fingerprints
can be compared across checkouts. Grouped cases check quotas and unique results.
The dense case exceeds the capped ID-set path at the default collection size;
the one-group-per-document case exercises bounded group retention.

The first query starts with a fresh worker and decoded-node cache. OS and
storage caches can remain warm. OPFS is required: an IndexedDB fallback fails
the run instead of being reported as OPFS performance. Each run creates and
removes its own temporary database.

## Run on a phone or another browser

Use manual mode to print the benchmark URL without launching Chrome:

```sh
node scripts/bench-web.mjs --vectors --serve --port 3000 --json
```

For Android connected over USB, forward that localhost port:

```sh
adb reverse tcp:3000 tcp:3000
```

Open the printed URL with `127.0.0.1:3000` on the phone. Keep the tab visible
and avoid other intensive work during the run. The page displays progress and
offers a JSON download; the host also saves the report when the run finishes.
Remove the forwarding afterward with `adb reverse --remove tcp:3000`.

For iOS Safari, serve the benchmark assets and built WASM package from an HTTPS
origin available to the phone. The page at `scripts/bench-web/vector.html`
accepts `count`, `dims`, `queries`, `quantization`, `concurrency`, and `cache-bytes` URL
parameters. Preserve the repository-relative asset paths. COOP `same-origin`
and COEP `require-corp` headers enable origin-memory measurements in browsers
that support them. Vector queries still run when that memory API is absent.

Mobile-browser results measure WASM on that device. Measure a native Kotlin,
Swift or React Native app separately to include its binding and storage path.
Desktop headless Chrome and native ARM64 CI results do not substitute for
physical-phone measurements.

## Core comparison and CI

The Rust harness defaults to an 8 MiB budget and reports retained cache bytes:

```sh
cargo run --release -p taladb --example hnsw_profile -- 10000 0.6 \
  --json --dims 128 --m 8 --ef-construction 64 --quantization binary \
  --queries 30 --cache-bytes 8388608
```

For real concurrent core searches, run the native memory workload. It starts
1, 2 and 4 threads together, records latency, recall, exact-fallback counts and
peak reserved allowance, and checks that all loans are released:

```sh
cargo run --release -p taladb --example vector_memory_profile -- \
  --cache-bytes 1048576 --count 2000 --dims 128 --queries 20
```

Repeat with `--cache-bytes 8388608`. The CI native jobs run both budgets and
publish `vector-memory-*.json`. These core runs exercise native concurrency;
they do not substitute for Kotlin/Swift device benchmarks. Large-filter ANN
bitmaps count toward the graph reservation. Capped filter ID sets, page windows,
group keys, record decoding, build scratch and storage-engine memory are outside
the reported search allowance. Budget limits can switch ANN to exact,
so compare latency together with fallback counts and recall.

CI compares baseline and candidate builds on the same runner. Chromium runs
the production worker/OPFS suite with both 1 MiB and 8 MiB budgets when the
baseline supports configurable budgets; older baselines use their 8 MiB WASM
default. PR comparisons use medians from three runs and retain individual
samples; larger scheduled/release workloads run once. All reports include
first-query latency, warm percentiles, recall and optional memory
measurements. Regression gates check build time, warm median latency and
recall. First-query latency and memory are reported without hard gates because
browser startup, garbage collection and API availability vary across runners.

Synthetic vectors help isolate implementation changes. Validate index tuning
and recall with representative embeddings, filters and dimensions from your
app before selecting device defaults.
