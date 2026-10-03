use taladb::{
    Database, Filter, GraphOptions, Quantization, Value, VectorQueryOptions, VectorSearchMode,
};
fn seed(db: &Database, count: usize) {
    let col = db.collection("docs").unwrap();
    let mut state = 1234567u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (f64::from(state as u32) / f64::from(u32::MAX) * 2.0 - 1.0) as f32
    };
    col.insert_many(
        (0..count)
            .map(|i| {
                vec![
                    (
                        "v".into(),
                        Value::Array((0..32).map(|_| Value::Float(f64::from(next()))).collect()),
                    ),
                    ("tag".into(), Value::Int(i as i64)),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_vector_index_with_options(
        "v",
        32,
        None,
        Some(GraphOptions {
            m: 8,
            ef_construction: 64,
            ..Default::default()
        }),
    )
    .unwrap();
}
#[test]
fn empty_filtered_ann_does_not_traverse_and_small_eligible_sets_do_not_retry() {
    let db = Database::open_in_memory().unwrap();
    seed(&db, 1000);
    let col = db.collection("docs").unwrap();
    let options = VectorQueryOptions {
        mode: VectorSearchMode::Ann,
        ..Default::default()
    };
    let empty = col
        .search_vectors(
            "v",
            &[0.5; 32],
            10,
            Some(Filter::Eq("tag".into(), Value::Int(-1))),
            &options,
        )
        .unwrap();
    assert!(empty.hits.is_empty());
    assert_eq!(empty.execution.distance_computations, 0);
    let one = col
        .search_vectors(
            "v",
            &[0.5; 32],
            10,
            Some(Filter::Eq("tag".into(), Value::Int(17))),
            &options,
        )
        .unwrap();
    assert_eq!(one.hits.len(), 1);
    assert_eq!(one.execution.ef_search, Some(100));
}
#[test]
fn quantization_rebuilds_and_staged_slot_reuse_match_a_cold_database() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = Database::open(file.path()).unwrap();
    seed(&db, 1500);
    let col = db.collection("docs").unwrap();
    let query = [0.5; 32];
    let options = VectorQueryOptions {
        mode: VectorSearchMode::Ann,
        ef_search: Some(10),
        oversampling: Some(1),
        ..Default::default()
    };
    col.search_vectors(
        "v",
        &query,
        10,
        None,
        &VectorQueryOptions {
            mode: VectorSearchMode::Ann,
            ef_search: Some(2000),
            ..Default::default()
        },
    )
    .unwrap();
    for quantization in [
        Quantization::Binary,
        Quantization::Scalar,
        Quantization::None,
    ] {
        let build = col
            .begin_vector_build(
                "v",
                Some(GraphOptions {
                    m: 8,
                    ef_construction: 64,
                    quantization,
                }),
            )
            .unwrap();
        while col.step_vector_build("v", &build.id, 128).unwrap().state == "building" {}
        let warm = col.search_vectors("v", &query, 10, None, &options).unwrap();
        let cold = Database::restore_from_snapshot(&db.export_snapshot().unwrap())
            .unwrap()
            .collection("docs")
            .unwrap()
            .search_vectors("v", &query, 10, None, &options)
            .unwrap();
        assert_eq!(
            warm.hits.iter().map(|h| h.document.id).collect::<Vec<_>>(),
            cold.hits.iter().map(|h| h.document.id).collect::<Vec<_>>()
        );
        assert_eq!(
            warm.execution.distance_computations,
            cold.execution.distance_computations
        );
    }
    // Synchronous rebuilds must get a new identity too, at the same vector revision.
    col.rebuild_vector_index(
        "v",
        Some(GraphOptions {
            m: 8,
            ef_construction: 64,
            quantization: Quantization::Binary,
        }),
    )
    .unwrap();
    let warm = col.search_vectors("v", &query, 10, None, &options).unwrap();
    let cold = Database::restore_from_snapshot(&db.export_snapshot().unwrap())
        .unwrap()
        .collection("docs")
        .unwrap()
        .search_vectors("v", &query, 10, None, &options)
        .unwrap();
    assert_eq!(
        warm.execution.distance_computations,
        cold.execution.distance_computations
    );
    assert_eq!(
        warm.hits.iter().map(|h| h.document.id).collect::<Vec<_>>(),
        cold.hits.iter().map(|h| h.document.id).collect::<Vec<_>>()
    );
}
#[test]
fn exact_and_graph_entries_share_one_budget_and_zero_disables_retention() {
    let db = Database::open_in_memory().unwrap();
    seed(&db, 100);
    let col = db.collection("docs").unwrap();
    let q = [0.5; 32];
    let exact = VectorQueryOptions {
        mode: VectorSearchMode::Exact,
        ..Default::default()
    };
    let truth = col.search_vectors("v", &q, 10, None, &exact).unwrap();
    let budget = db.vector_cache_stats().retained_bytes;
    assert!(budget > 0);
    db.set_vector_cache_budget(budget);
    col.search_vectors(
        "v",
        &q,
        10,
        None,
        &VectorQueryOptions {
            mode: VectorSearchMode::Ann,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(db.vector_cache_stats().retained_bytes <= budget);
    db.set_vector_cache_budget(0);
    let result = col.search_vectors("v", &q, 10, None, &exact).unwrap();
    assert_eq!(
        truth.hits.iter().map(|h| h.document.id).collect::<Vec<_>>(),
        result
            .hits
            .iter()
            .map(|h| h.document.id)
            .collect::<Vec<_>>()
    );
    col.search_vectors(
        "v",
        &q,
        10,
        None,
        &VectorQueryOptions {
            mode: VectorSearchMode::Ann,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(db.vector_cache_stats().retained_bytes, 0);
}
