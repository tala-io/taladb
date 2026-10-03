//! Exact and ANN filter workloads, including cold and warm vector caches.
//! cargo run --release -p taladb --example vector_filter_profile -- 10000
use std::time::Instant;
use taladb::{
    Database, Filter, GraphOptions, Quantization, Value, VectorQueryOptions, VectorSearchMode,
};

fn main() {
    let count: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let dimensions = 128;
    let args: Vec<_> = std::env::args().collect();
    let queries = args
        .iter()
        .position(|arg| arg == "--queries")
        .map_or(20, |i| args[i + 1].parse::<usize>().unwrap());
    assert!(count >= 10 && queries > 0);
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    let mut rng = 1234567u64;
    let mut vector = || {
        (0..dimensions)
            .map(|_| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                (rng >> 40) as f32 / 8_388_608.0 - 1.0
            })
            .collect::<Vec<_>>()
    };
    col.insert_many(
        (0..count)
            .map(|i| {
                vec![
                    (
                        "v".into(),
                        Value::Array(
                            vector()
                                .into_iter()
                                .map(|v| Value::Float(f64::from(v)))
                                .collect(),
                        ),
                    ),
                    ("ordinal".into(), Value::Int(i as i64)),
                    ("bucket".into(), Value::Int((i % 1000) as i64)),
                    ("tenant".into(), Value::Str((i % 10).to_string())),
                    ("active".into(), Value::Bool(i % 20 != 0)),
                    (
                        "rank".into(),
                        if i % 2 == 0 {
                            Value::Int(i as i64)
                        } else {
                            Value::Float(i as f64)
                        },
                    ),
                    ("body".into(), Value::Bytes(vec![42; 8192])),
                ]
            })
            .collect(),
    )
    .unwrap();
    for field in ["bucket", "tenant", "active", "rank"] {
        col.create_index(field).unwrap();
    }
    col.create_vector_index_with_options(
        "v",
        dimensions,
        None,
        Some(GraphOptions {
            m: 8,
            ef_construction: 64,
            quantization: Quantization::Binary,
        }),
    )
    .unwrap();
    let probes: Vec<_> = (0..queries).map(|_| vector()).collect();
    let filters = [
        ("sparse", Filter::Eq("bucket".into(), Value::Int(0))),
        (
            "moderate",
            Filter::Eq("tenant".into(), Value::Str("0".into())),
        ),
        ("dense", Filter::Eq("active".into(), Value::Bool(true))),
        (
            "range",
            Filter::Gte("rank".into(), Value::Float(count as f64 / 4.0)),
        ),
    ];
    let mut reports = Vec::new();
    for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
        for warm in [false, true] {
            db.set_vector_cache_budget(if warm { 64 * 1024 * 1024 } else { 0 });
            if warm {
                col.search_vectors(
                    "v",
                    &probes[0],
                    10,
                    None,
                    &VectorQueryOptions {
                        mode: VectorSearchMode::Exact,
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            for (name, filter) in &filters {
                let options = VectorQueryOptions {
                    mode,
                    ..Default::default()
                };
                // Exclude cold graph decoding from the warm-query measurements.
                col.search_vectors("v", &probes[0], 10, Some(filter.clone()), &options)
                    .unwrap();
                let mut times = Vec::new();
                let mut fingerprint = 0u64;
                let mut distances = 0usize;
                for query in &probes {
                    let start = Instant::now();
                    let result = col
                        .search_vectors("v", query, 10, Some(filter.clone()), &options)
                        .unwrap();
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                    distances += result.execution.distance_computations;
                    for hit in result.hits {
                        let Some(Value::Int(ordinal)) = hit.document.get("ordinal") else {
                            unreachable!()
                        };
                        fingerprint = fingerprint.wrapping_mul(0x100_0000_01b3)
                            ^ (*ordinal as u64)
                            ^ u64::from(hit.score.to_bits());
                    }
                }
                times.sort_by(f64::total_cmp);
                reports.push(
                    serde_json::json!({ "mode": mode, "warm": warm, "filter": name,
                    "p50_ms": times[queries / 2], "p95_ms": times[queries * 95 / 100],
                    "fingerprint": fingerprint, "distances": distances }),
                );
            }
        }
    }
    println!(
        "{}",
        serde_json::json!({ "schema": 1, "count": count, "dimensions": dimensions,
        "queries": queries, "body_bytes": 8192, "cases": reports })
    );
}
