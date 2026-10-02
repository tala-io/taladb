//! HNSW build and search profile.
//!
//! Criterion is the wrong shape for this: a single 10k-vector graph build takes
//! tens of seconds, so a sampling harness that wants many iterations cannot run
//! it at all. This measures one build and a fixed query sweep, prints the
//! numbers, and exits — which is what an optimisation on this path needs to be
//! judged against.
//!
//!     cargo run --release -p taladb --example hnsw_profile [count]

use std::time::Instant;
use taladb::{Database, GraphOptions, Value, VectorMetric, VectorQueryOptions, VectorSearchMode};

const DIMS: usize = 384;
const CLUSTERS: usize = 512;
const TOP_K: usize = 10;
const QUERIES: usize = 20;

/// Deterministic, so two runs of this example are comparable to each other.
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 40) as f32 / 16_777_216.0
    }
    /// Box–Muller. Gaussian components give a well-behaved direction.
    fn gaussian(&mut self) -> f32 {
        let u = self.next_f32().max(1e-9);
        (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * self.next_f32()).cos()
    }
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
        .max(f32::MIN_POSITIVE);
    for x in &mut v {
        *x /= norm;
    }
    v
}

/// Clustered rather than uniform: in 384 dimensions uniform points are all
/// nearly equidistant, so a graph has no structure to exploit and the measured
/// recall says more about the generator than about the index.
fn make_point(rng: &mut Rng, centroid: &[f32], spread: f32) -> Vec<f32> {
    normalize(
        centroid
            .iter()
            .map(|c| c + rng.gaussian() * spread)
            .collect(),
    )
}

fn to_value(v: &[f32]) -> Value {
    Value::Array(v.iter().map(|f| Value::Float(f64::from(*f))).collect())
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let option = |name: &str, default: u32| -> u32 {
        args.iter().position(|a| a == name).map_or(default, |i| {
            args.get(i + 1)
                .expect("missing parameter value")
                .parse()
                .expect("invalid parameter value")
        })
    };
    let m = option("--m", 16);
    let ef_construction = option("--ef-construction", 200);
    let json_output = std::env::args().any(|a| a == "--json");
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(10_000);
    assert!(count >= TOP_K, "count must be at least {TOP_K}");
    // Cluster spread. Large values wash the clusters out and the set approaches
    // uniform-on-sphere, which is the pathological case for any proximity graph
    // in high dimensions — worth being able to vary before blaming the index.
    let spread: f32 = std::env::args()
        .nth(2)
        .and_then(|a| a.parse().ok())
        .unwrap_or(0.6);
    if !json_output {
        println!("count {count}, spread {spread}, dims {DIMS}, clusters {CLUSTERS}");
    }

    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let centroids: Vec<Vec<f32>> = (0..CLUSTERS)
        .map(|_| normalize((0..DIMS).map(|_| rng.gaussian()).collect()))
        .collect();

    let points: Vec<Vec<f32>> = (0..count)
        .map(|i| make_point(&mut rng, &centroids[i % CLUSTERS], spread))
        .collect();
    let mut probe_rng = Rng(0x2AB7_7CA8_8102_4991);
    let rng = if json_output {
        &mut probe_rng
    } else {
        &mut rng
    };
    let probes: Vec<Vec<f32>> = (0..QUERIES)
        .map(|q| make_point(rng, &centroids[(q * 7) % CLUSTERS], spread))
        .collect();

    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();

    let t = Instant::now();
    col.insert_many(
        points
            .iter()
            .map(|v| vec![("embedding".to_string(), to_value(v))])
            .collect(),
    )
    .unwrap();
    let insert_ms = t.elapsed().as_secs_f64() * 1000.0;
    if !json_output {
        println!("insert {count}      {:>8.2?}", t.elapsed());
    }

    let t = Instant::now();
    col.create_vector_index_with_options(
        "embedding",
        DIMS,
        Some(VectorMetric::Cosine),
        Some(GraphOptions {
            m,
            ef_construction,
            ..Default::default()
        }),
    )
    .unwrap();
    let build = t.elapsed();
    if !json_output {
        println!(
            "hnsw build         {build:>8.2?}   ({:.2} ms/vector)",
            build.as_secs_f64() * 1000.0 / count as f64
        );
    }

    // `Document` carries its ULID as a struct field, not as an `_id` entry in
    // `fields` — that mapping happens in the bindings.
    let ids = |r: taladb::VectorQueryResult| -> Vec<String> {
        r.hits
            .into_iter()
            .map(|h| h.document.id.to_string())
            .collect()
    };

    // Exact is ground truth for the recall numbers below.
    let exact_opts = VectorQueryOptions {
        mode: VectorSearchMode::Exact,
        ..Default::default()
    };
    let mut exact_times = Vec::new();
    let truth: Vec<Vec<String>> = probes
        .iter()
        .map(|q| {
            let start = Instant::now();
            let result = col
                .search_vectors("embedding", q, TOP_K, None, &exact_opts)
                .unwrap();
            exact_times.push(start.elapsed().as_secs_f64() * 1000.0);
            ids(result)
        })
        .collect();
    let exact_ms = exact_times.iter().sum::<f64>() / QUERIES as f64;
    if !json_output {
        println!("\nexact              {exact_ms:>8.3} ms/query");
    }
    let mut measurements = Vec::new();
    for ef in [50usize, 100, 200, 400] {
        let opts = VectorQueryOptions {
            mode: VectorSearchMode::Ann,
            ef_search: Some(ef),
            ..Default::default()
        };
        if json_output {
            for q in &probes {
                col.search_vectors("embedding", q, TOP_K, None, &opts)
                    .unwrap();
            }
        }
        let t = Instant::now();
        let mut query_times = Vec::new();
        let mut recall = 0.0;
        let mut distances = 0usize;
        for (i, q) in probes.iter().enumerate() {
            let start = Instant::now();
            let result = col
                .search_vectors("embedding", q, TOP_K, None, &opts)
                .unwrap();
            query_times.push(start.elapsed().as_secs_f64() * 1000.0);
            assert_eq!(result.execution.path, "hnsw");
            distances += result.execution.distance_computations;
            let found = ids(result)
                .iter()
                .filter(|id| truth[i].contains(id))
                .count();
            recall += found as f64 / TOP_K as f64;
        }
        let elapsed_ms = t.elapsed().as_secs_f64() * 1000.0 / QUERIES as f64;
        if !json_output {
            println!(
                "ann ef={ef:<4}        {:>8.3} ms/query   recall@{TOP_K} {:>5.1}%   {:>6} distances",
                t.elapsed().as_secs_f64() * 1000.0 / QUERIES as f64,
                recall / QUERIES as f64 * 100.0,
                distances / QUERIES
            );
        }
        query_times.sort_by(f64::total_cmp);
        measurements.push(serde_json::json!({ "ef_search": ef, "mean_ms": elapsed_ms,
            "p50_ms": query_times[QUERIES / 2], "p95_ms": query_times[QUERIES * 95 / 100],
            "recall_at_10": recall / QUERIES as f64, "distances": distances / QUERIES }));
    }
    if json_output {
        let peak_rss_bytes = std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|status| {
                status
                    .lines()
                    .find_map(|line| {
                        line.strip_prefix("VmHWM:")
                            .and_then(|rest| rest.split_whitespace().next())
                            .and_then(|n| n.parse::<u64>().ok())
                    })
                    .map(|kb| kb * 1024)
            });
        println!(
            "{}",
            serde_json::json!({ "schema": 1, "count": count, "dimensions": DIMS,
            "spread": spread, "m": m, "ef_construction": ef_construction, "queries": QUERIES,
            "insert_ms": insert_ms, "build_ms": build.as_secs_f64() * 1000.0,
            "exact_mean_ms": exact_ms, "ann": measurements, "peak_rss_bytes": peak_rss_bytes })
        );
    }
}
