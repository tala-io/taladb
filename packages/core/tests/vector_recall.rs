//! Recall regressions where losing query magnitudes changes the candidate set.
use taladb::{Database, GraphOptions, Quantization, Value, VectorQueryOptions, VectorSearchMode};

fn vector(values: &[f32]) -> Value {
    Value::Array(values.iter().map(|&x| Value::Float(f64::from(x))).collect())
}

#[test]
fn binary_traversal_keeps_query_magnitudes_before_exact_rescoring() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    // Both queries below have the same signs, but their nearest documents differ.
    // With sign-only queries the binary traversal picks the second document for
    // both and discards the first before exact rescoring can correct the ranking.
    let first = col
        .insert(vec![("v".into(), vector(&[10.0, -1.0, -1.0]))])
        .unwrap();
    let second = col
        .insert(vec![("v".into(), vector(&[-1.0, 10.0, 10.0]))])
        .unwrap();
    col.create_vector_index_with_options(
        "v",
        3,
        None,
        Some(GraphOptions {
            m: 2,
            ef_construction: 4,
            quantization: Quantization::Binary,
        }),
    )
    .unwrap();
    let ann = VectorQueryOptions {
        mode: VectorSearchMode::Ann,
        ef_search: Some(1),
        oversampling: Some(1),
        ..Default::default()
    };
    let exact = VectorQueryOptions {
        mode: VectorSearchMode::Exact,
        ..Default::default()
    };
    let cold = Database::restore_from_snapshot(&db.export_snapshot().unwrap()).unwrap();
    cold.set_vector_cache_budget(0);
    let cold_col = cold.collection("docs").unwrap();
    for collection in [&col, &cold_col] {
        for (query, expected) in [
            ([10.0, 1.0, 1.0], first),
            ([1.0, 10.0, 10.0], second),
            ([80.0, 8.0, 8.0], first),
            ([0.0, 0.0, 0.0], first),
        ] {
            let truth = collection
                .search_vectors("v", &query, 1, None, &exact)
                .unwrap();
            assert_eq!(truth.hits[0].document.id, expected);
            let result = collection
                .search_vectors("v", &query, 1, None, &ann)
                .unwrap();
            assert_eq!(result.execution.ef_search, Some(1));
            assert_eq!(result.hits[0].document.id, expected);
            assert_eq!(result.hits[0].score, truth.hits[0].score);
        }
    }
}
