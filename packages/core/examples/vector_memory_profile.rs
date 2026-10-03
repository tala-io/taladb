//! Concurrent core searches with the same 1 MiB / 8 MiB budgets used on phones.
//! cargo run --release -p taladb --example vector_memory_profile -- --cache-bytes 1048576
//! Peak reservation is conservative accounting, not peak RSS or an allocator trace.
use std::sync::{Arc, Barrier};
use std::time::Instant;
use taladb::{
    Database, Filter, GraphOptions, Quantization, Value, VectorQueryOptions, VectorSearchMode,
};

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let option = |name: &str, default: usize| {
        args.iter()
            .position(|arg| arg == name)
            .map_or(default, |i| {
                args[i + 1].parse::<usize>().expect("invalid integer")
            })
    };
    let budget = option("--cache-bytes", 8 * 1024 * 1024);
    let count = option("--count", 2000);
    let dimensions = option("--dims", 128);
    let queries = option("--queries", 20);
    assert!(count >= 100 && dimensions > 0 && queries > 0);
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("vectors").unwrap();
    let point = |i: usize| {
        (0..dimensions)
            .map(|j| {
                (((i % 32) * 17 + j * 13) as f32).sin() + ((i * 37 + j * 7) as f32).cos() * 0.2
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
                            point(i)
                                .into_iter()
                                .map(|v| Value::Float(f64::from(v)))
                                .collect(),
                        ),
                    ),
                    ("tenant".into(), Value::Int((i % 10) as i64)),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_index("tenant").unwrap();
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
    let probes = Arc::new(
        (0..queries)
            .map(|i| point(count + i * 7))
            .collect::<Vec<_>>(),
    );
    let filter = Filter::Eq("tenant".into(), Value::Int(0));
    let exact_options = VectorQueryOptions {
        mode: VectorSearchMode::Exact,
        ..Default::default()
    };
    let truth = Arc::new(
        probes
            .iter()
            .map(|q| {
                col.search_vectors("v", q, 10, Some(filter.clone()), &exact_options)
                    .unwrap()
                    .hits
                    .into_iter()
                    .map(|h| h.document.id)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
    );
    let mut cases = Vec::new();
    for concurrency in [1, 2, 4] {
        db.set_vector_cache_budget(0);
        db.set_vector_cache_budget(budget); // resets peak accounting before this case
        let gate = Arc::new(Barrier::new(concurrency));
        let started = Instant::now();
        let workers: Vec<_> = (0..concurrency)
            .map(|_| {
                let col = db.collection("vectors").unwrap();
                let gate = gate.clone();
                let probes = probes.clone();
                let truth = truth.clone();
                let filter = filter.clone();
                std::thread::spawn(move || {
                    let options = VectorQueryOptions {
                        mode: VectorSearchMode::Ann,
                        ef_search: Some(100),
                        ..Default::default()
                    };
                    let mut times = Vec::new();
                    let mut fallbacks = 0;
                    let mut recall = 0.0;
                    for (i, query) in probes.iter().enumerate() {
                        gate.wait(); // force overlapping starts without serializing work
                        let start = Instant::now();
                        let result = col
                            .search_vectors("v", query, 10, Some(filter.clone()), &options)
                            .unwrap();
                        times.push(start.elapsed().as_secs_f64() * 1000.0);
                        let ids: Vec<_> = result.hits.iter().map(|h| h.document.id).collect();
                        let found = ids.iter().filter(|id| truth[i].contains(id)).count();
                        recall += found as f64 / truth[i].len() as f64;
                        if result.execution.reason == "memoryBudget" {
                            fallbacks += 1;
                            assert_eq!(ids, truth[i], "fallback must return the exact ranking");
                        } else {
                            assert_eq!(result.execution.path, "hnsw");
                        }
                    }
                    (times, fallbacks, recall)
                })
            })
            .collect();
        let mut times = Vec::new();
        let mut fallbacks = 0;
        let mut recall = 0.0;
        for worker in workers {
            let (t, f, r) = worker.join().unwrap();
            times.extend(t);
            fallbacks += f;
            recall += r;
        }
        times.sort_by(f64::total_cmp);
        let stats = db.vector_cache_stats();
        assert_eq!(stats.active_bytes, 0, "all memory loans must be released");
        assert!(
            stats.peak_bytes <= stats.memory_budget_bytes,
            "accounted search allowance exceeded"
        );
        assert!(stats.retained_bytes <= budget);
        cases.push(
            serde_json::json!({ "concurrency": concurrency, "requests": times.len(),
            "p50_ms": times[times.len()/2], "p95_ms": times[times.len()*95/100],
            "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0, "memory_fallbacks": fallbacks,
            "recall_at_10": recall / times.len() as f64, "cache": stats }),
        );
    }
    println!(
        "{}",
        serde_json::json!({ "schema": 1, "workload": "concurrent-vector-memory", "count": count,
        "dimensions": dimensions, "queries": queries, "cache_bytes": budget, "cases": cases })
    );
}
