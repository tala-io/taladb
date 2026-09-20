# Changelog

## 0.11.8 — 2026-09-20

- Roughly halved cosine vector scoring by splitting the fused dot-product-and-norm pass into two loops. The two `[f32; LANES]` accumulators shared one loop body, which exceeded what LLVM's SLP vectoriser would hold in registers, so it emitted the whole reduction scalar — on aarch64 and wasm it produced almost no vector instructions at all. Splitting the passes costs one extra walk over data already in L1 and restores vectorisation on every target. `vector/score_one/Cosine` improved 53–62% across 128–1536 dimensions and end-to-end `vector/find_nearest` improved 45% at 1,000 vectors and 50% at 10,000, measured on x86_64; aarch64 and wasm were affected more severely and should gain at least as much. Dot and Euclidean scoring were already single-accumulator and are unchanged.
- Retained decoded HNSW nodes across queries instead of rebuilding the cache per query, cutting ANN latency by ~4.6x at 10,000 vectors (`efSearch` 100: 8.31 ms to 1.80 ms; `efSearch` 400, the 99.5%-recall setting: 25.00 ms to 5.92 ms). Recall is unchanged at every setting — this removes repeated storage reads and postcard decodes, which measured at ~97% of ANN query time against under 3% for scoring itself. The cache is shared by every collection handle from one database, pinned to the vector revision its graph was built against, and evicted explicitly on index create, drop and rebuild. The per-graph budget is 64 MiB on native and stays 8 MiB on WASM, where holding tens of megabytes is not an option; the old 8 MiB bound was sized for a cache that lived for a single query and made a 10,000-vector graph fill, flush and refill.
- No behaviour change to scoring: both totals are bit-identical to the fused loop, including on the length-mismatch path, and are pinned by tests against the previous implementation.
- Built the browser module with WebAssembly SIMD (`-C target-feature=+simd128`). The lane-split scoring loops had no vector instructions to compile into on the wasm baseline, so the browser was running them scalar and getting none of the benefit the native targets get for free. The feature is set in `.cargo/config.toml`'s `rustflags` array rather than a `RUSTFLAGS` environment variable, because the environment variable replaces that array instead of merging with it and would silently drop the `getrandom_backend` cfg the wasm build depends on. `wasm-opt` is given `--enable-simd` to accept the result.

  **This raises the browser floor.** A WASM module that uses SIMD does not instantiate on an engine without it, so the minimum is now Safari 16.4, Chrome 91 and Firefox 89 — SIMD support, not OPFS, now sets the floor for every storage backend. The support tables in the web guide and the `@taladb/web` README previously disagreed with each other (16.4+ versus 15.2+) and have both been corrected.

## 0.11.7 — 2026-09-15

- Removed `android.permission.INTERNET` from the React Native binding's manifest. The manifest merger granted it to every consuming application, so an app that never opens a socket still shipped — and had to justify on its store listing — a network permission it did not use. The sync feature that once needed it was removed in 0.11.0; the binding opens no socket and the FFI crate pulls in no network dependency. Apps that need the permission can declare it themselves.

## 0.11.6 — 2026-09-14

- Fixed `libtaladb_jsi.so` being linked with 4 KB page alignment, which made any app bundling the React Native binding fail Google Play's 16 KB page size requirement for Android 15 and later, and prevented it loading at all on a 16 KB device. The C++ glue now links with `-Wl,-z,max-page-size=16384` explicitly; the prebuilt Rust libraries were already aligned.
- Stopped the Android binding pinning `ndkVersion` to r27 unconditionally. It now honours `ext.ndkVersion` from the consuming app's root `build.gradle` — which the React Native template sets — and falls back to r27.1 only when the app declares none. Pinning it forced every consumer onto an NDK whose default page alignment is 4 KB, regardless of what the rest of their application was built with.

- Added a CI check that links the Android glue against NDK r27 and asserts the result is 16 KB aligned. The existing job compiled both translation units but never linked them, so the defect above passed every check it had.

No API, storage-format or behaviour change: this is a link-time fix. Android consumers should rebuild to pick it up; a clean build of the `:taladb_react-native` module is needed because the object is cached per CMake configuration.

## 0.11.5 — 2026-09-13

- Raised HNSW recall by building layer 0 with `M_max0 = 2M` links, matching the algorithm and the pruning limit the same function already applied; recall@10 rose from 91.5% to 97.5% at efSearch 100 over 2,000 vectors, and from 93.5% to 99.5% at efSearch 400 over 10,000.
- Cut HNSW graph construction time by roughly 12%: 12.0s to 10.8s at 2,000 vectors and 155s to 136s at 10,000, both at 384 dimensions.
- Removed a heap clone of the stored vector on every distance computation by borrowing from the graph's decoder instead of returning an owned copy.
- Cached each node's squared norm so cosine scoring no longer recomputes it per comparison, and replaced SipHash with a multiplicative hash for internal graph node ids.
- Shared one node cache across an insert's two readers, across its back-linking pass, and across ANN retry attempts, removing repeated storage reads and decodes of nodes already in memory.
- Carried the full candidate set between layers during insertion instead of only the nearest, per Malkov/Yashunin algorithm 1.
- Added `cargo run -p taladb-core --example hnsw_profile` for measuring build time, query latency and recall against exact ground truth, with a cluster-spread argument for separating index quality from data difficulty.
- Documented the remaining graph-traversal cost and recall-at-scale gaps in the roadmap, and corrected a stale entry that described HNSW as available only in the Node binding.

No storage-format change: existing graphs open and search without a rebuild. Rebuilding an index built by an earlier version picks up the layer-0 improvement.

## 0.11.4 - 2026-09-12

- Added full-text, vector, hybrid, and aggregation queries to the React Native TypeScript surface.
- Replaced the memory-only `instant-distance` layer with portable, transactional HNSW graphs that persist and update on browser, Node.js, and React Native.
- Added configurable graph construction and query-time `efSearch`, exact rescoring, scalar/binary quantization, explicit exact/ANN modes, score thresholds, pagination, and grouping.
- Added persistent index status plus resumable, cancellable batch rebuilds with progress reporting; `db.rebuildVectorIndexes()` remains available for maintenance and legacy migration.
- Added filtered ANN as an explicit opt-in while keeping filtered queries exact by default, plus recall measurement against exact ground truth.
- Fixed the `build:cbindgen` script and config, which regenerated an unusable FFI header.
- Documented persistent HNSW lifecycle, mobile-safe rebuild scheduling, hybrid search, and package choice in the React Native guide.
- Added a CI job that compiles the React Native C++ glue and checks the generated FFI header.

## 0.11.3 — 2026-09-06

- Added authoritative single-owner browser storage and multi-tab RPC.
- Propagated IndexedDB and OPFS failures and bounded browser snapshots.
- Made schema, index, full-text, and vector reads snapshot-consistent.
- Bounded flat vector search memory and added exact fallback for stale HNSW indexes.
- Hardened mutation invariants, vector validation, and React Native async jobs.
- Expanded cross-runtime reliability tests and CI coverage.

## 0.11.2 — 2026-08-28

- Raised the Rust baseline to 1.90 and upgraded to redb 4.2 storage format v3.
- Added legacy storage migration and JSON-depth validation.
- Hardened vector limits, snapshots, passphrase handling, regex filters, and the C FFI.
- Improved array-index maintenance, OPFS validation, and React Native job limits.
- Added MSRV, fuzz, Wasm, and Miri checks.

## 0.11.1 — 2026-08-16

- Fixed Node adapter resolution in browser builds.
- Fixed shared database handling under React StrictMode.
- Fixed concurrent and duplicate hydration.
- Added document-ID validation and shared-handle reset helpers.

## 0.11.0 — 2026-08-15

- Removed sync, replication, conflict-resolution, and sync-backend APIs.
- Added change webhooks, queryable arrays, caller-supplied IDs, and `isPrimary()`.
- Renamed React `useMutation` to `useWrite` and config `sync` to `webhook`.
- Fixed browser multi-tab writes, live queries, owner handoff, and vector dimensions.
- Reduced unnecessary index rewrites.

## 0.10.2 — 2026-08-02

- Accelerated vector scoring, range scans, indexed queries, counts, and browser startup.
- Fixed LWW timestamp abuse, invalid regex handling, vector dimensions, and snapshots.
- Fixed large-number ranges, pagination overflow, React Native errors, and OPFS fallback.
- Improved storage error context, value APIs, benchmarks, lints, and crate documentation.

## 0.10.1 — 2026-08-01

- Added a decoded-vector cache for faster repeated flat search.

## 0.10.0 — 2026-07-25

- Added BM25 full-text search and ranked hybrid search.
- Added Node full-text APIs and richer schema downgrade controls.
- Upgraded full-text index storage.
- Fixed browser fallback values and migration write-back safety.

## 0.9.4 — 2026-07-13

- Added batched ID-based replication writes and deletes.
- Added live aggregation, derived IDs, cursor sync, coverage, and REST replication helpers.
- Added React and Next.js replication APIs.
- Fixed replication echo, configured collection resolution, and API documentation.

## 0.9.3 — 2026-07-12

- Added index-backed pagination and bounded top-K sorting.
- Added projection exclusion and deterministic sort ordering.
- Improved sort performance and union-field filter types.
- Fixed React hooks bypassing collection configuration.

## 0.9.2 — 2026-07-12

- Added import validation, quarantine, document versions, renames, and migrations.
- Added configurable durability, explicit flush, and native migration accessors.
- Applied migrations and validation consistently to reads and subscriptions.
- Fixed quarantine persistence and migration write-back behavior.

## 0.9.1 — 2026-07-12

- Added scoped React replication hooks.
- Added background prefetch for local replicas.

## 0.9.0 — 2026-07-11

- Added browser encryption, compound indexes, browser sync, and React Native sync plumbing.
- Added first-party Next.js sync support, React integration, examples, and end-to-end tests.
- Doubled flat vector-search throughput and bounded two-sided range scans.
- Fixed browser locking, encrypted cross-tab handling, cursors, Node opening, and bundling.

## 0.8.4 — 2026-07-11

- Added bidirectional Node sync and an HTTP sync adapter.
- Added aggregation across all runtimes.
- Added the MongoDB sync adapter.

## 0.8.3 — 2026-07-09

- Enabled HNSW in published Node binaries.
- Added Node and browser benchmark suites.
- Published benchmark results and tuning guidance.

## 0.8.2 — 2026-06-12

- Secured Studio binding, React Native filter parsing, encryption AAD, and ULID entropy.
- Fixed LWW, tombstone, CRDT, numeric-index, index-cache, and mutation races.
- Made audit writes atomic and rekey operations resumable.
- Added core live queries, atomic ID replacement, async Node writes, and primary-key plans.
- Aligned operators, errors, and index metadata across bindings.

## 0.8.0 — 2026-05-14

- Added field-level CRDT sync with deterministic clocks.
- Added grow-only set fields, incremental export, tombstones, and multi-replica merge.

## 0.7.10 — 2026-04-19

- Fixed React Native platform detection and Metro/Hermes bundling.
- Aligned the TypeScript adapter with the native JSI API.

## 0.7.9 — 2026-04-19

- Fixed Android native-library loading and Kotlin compilation.

## 0.7.8 — 2026-04-19

- Fixed React Native C++ header compilation.

## 0.7.7 — 2026-04-19

- Fixed the Android C++ runtime configuration.

## 0.7.6 — 2026-04-19

- Added the missing Android Gradle, codegen, and manifest configuration.

## 0.7.5 — 2026-04-19

- Moved runtime adapters to optional peer dependencies.
- Added a dedicated React Native export for Metro.

## 0.7.4 — 2026-04-18

- Added zero-copy `Float32Array` vector queries on React Native and Node.
- Added background vector and full-scan queries.
- Added React Native vector-index management.

## 0.7.3 — 2026-04-17

- Added database compaction across runtimes.
- Added Cloudflare Durable Objects and Bun support.
- Added the local Studio UI.
- Added Zod and Valibot collection validation.

## 0.7.2

- Added database rekeying and per-field encryption.
- Added append-only mutation audit logs and readers.

## 0.7.1

- Added query timeouts and tracing spans.
- Added snapshot and filter fuzz targets.

## 0.7.0

- Made collection creation fallible with eager name validation.
- Added encrypted-format migration and a snapshot size guard.
- Added structured tracing and bounded webhook workers.

## 0.6.1 — 2026-04-13

- Added automatic change timestamps, tombstones, and indexed change export.
- Added bidirectional Wasm changesets and collection listing.
- Added secondary-tab write propagation and debounced IndexedDB snapshots.
- Added tombstone compaction and sync type exports.

## 0.6.0 — 2026-04-12

- Added HTTP push sync.
- Fixed React Native Android and iOS native packaging.
- Fixed browser bundle resolution and pinned the Android NDK.

## 0.5.0 — 2026-04-12

- Added first-party React live-query hooks.
- Expanded package and release infrastructure.

## 0.4.0 — 2026-04-11

- Added full-text and HNSW vector indexes.
- Added index introspection and query-planner strategies.
- Fixed full-text parsing, index idempotency, and multi-tab OPFS locking.

## 0.2.1 — 2026-04-05

- Fixed ID-based sync, corrupted key handling, and encryption errors.
- Added index metadata caching and faster OR and full-text plans.
- Aligned adapter error reporting and watch backpressure.

## 0.2.0 — 2026-04-05

- Added flat vector indexes with cosine, dot-product, and Euclidean search.
- Added filtered nearest-neighbor search across browser, Node, and React Native.

## 0.1.2 — 2026-03-30

- Fixed release publishing being blocked by crates.io.

## 0.1.1 — 2026-03-30

- Fixed adapter bundling and a Rust lint.

## 0.1.0 — 2026-03-30

- Released the Rust core, CLI, and TypeScript packages.
- Added document queries, updates, indexes, full-text search, and live subscriptions.
- Added browser, Node, React Native, encryption, migration, and snapshot support.
