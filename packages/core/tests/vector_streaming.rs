use std::collections::{HashMap, HashSet};
use taladb::{
    Database, Filter, GraphOptions, Value, VectorMetric, VectorQueryOptions, VectorSearchMode,
};

fn seed(count: usize) -> Database {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    col.insert_many(
        (0..count)
            .map(|i| {
                let mut fields = vec![
                    ("ordinal".into(), Value::Int(i as i64)),
                    ("active".into(), Value::Bool(i % 20 != 19)),
                    ("constant".into(), Value::Bool(true)),
                    (
                        "tags".into(),
                        Value::Array(vec![Value::Int(-10), Value::Int(10000), Value::Int(-10)]),
                    ),
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(i as f64), Value::Float(1.0)]),
                    ),
                    ("body".into(), Value::Bytes(vec![42; 4096])),
                ];
                if i != 1100 {
                    let group = if i < 1100 {
                        Value::Str("dominant".into())
                    } else if i == 1101 {
                        Value::Null
                    } else if i == 1102 {
                        Value::Float(-0.0)
                    } else if i == 1103 {
                        Value::Float(0.0)
                    } else if i == 1104 {
                        Value::Array(vec![Value::Int(1)])
                    } else if i == 1105 {
                        Value::Object(vec![("x".into(), Value::Int(1))])
                    } else {
                        Value::Int(i as i64)
                    };
                    fields.push(("meta".into(), Value::Object(vec![("group".into(), group)])));
                }
                fields
            })
            .collect(),
    )
    .unwrap();
    for f in ["active", "tags", "ordinal"] {
        col.create_index(f).unwrap();
    }
    col.create_compound_index(&["active", "constant"]).unwrap();
    col.create_vector_index_with_options(
        "v",
        2,
        Some(VectorMetric::Euclidean),
        Some(GraphOptions {
            m: 4,
            ef_construction: 16,
            ..Default::default()
        }),
    )
    .unwrap();
    db
}
fn opts(mode: VectorSearchMode) -> VectorQueryOptions {
    VectorQueryOptions {
        mode,
        ef_search: Some(4096),
        ..Default::default()
    }
}

#[test]
fn dense_index_array_union_and_residual_filters_stream_unique_snapshot_matches() {
    let db = seed(1500);
    db.set_vector_cache_budget(1024 * 1024);
    let col = db.collection("docs").unwrap();
    let truth = col
        .search_vectors("v", &[0.0, 1.0], 2000, None, &opts(VectorSearchMode::Exact))
        .unwrap()
        .hits;
    let active = Filter::Eq("active".into(), Value::Bool(true));
    let filters = [
        active.clone(),
        Filter::Eq("tags".into(), Value::Int(-10)),
        Filter::And(vec![
            active.clone(),
            Filter::Eq("constant".into(), Value::Bool(true)),
        ]),
        Filter::In("active".into(), vec![Value::Bool(true), Value::Bool(true)]),
        Filter::Gte("ordinal".into(), Value::Int(0)),
        Filter::And(vec![
            Filter::Gte("ordinal".into(), Value::Int(0)),
            Filter::Lte("ordinal".into(), Value::Int(1499)),
        ]),
        Filter::And(vec![
            Filter::Gte("tags".into(), Value::Int(5000)),
            Filter::Lte("tags".into(), Value::Int(0)),
        ]),
        Filter::Or(vec![active.clone(), active.clone()]),
        Filter::And(vec![active, Filter::Regex("body".into(), "nomatch".into())]),
        Filter::Not(Box::new(Filter::Eq("active".into(), Value::Bool(false)))),
        Filter::Exists("body".into(), true),
    ];
    for filter in filters {
        let expected: Vec<_> = truth
            .iter()
            .filter(|h| filter.matches(&h.document).unwrap())
            .take(20)
            .map(|h| (h.document.id, h.score.to_bits()))
            .collect();
        for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
            let result = col
                .search_vectors("v", &[0.0, 1.0], 20, Some(filter.clone()), &opts(mode))
                .unwrap();
            assert_eq!(
                result
                    .hits
                    .iter()
                    .map(|h| (h.document.id, h.score.to_bits()))
                    .collect::<Vec<_>>(),
                expected,
                "{filter:?} {mode:?}"
            );
            assert_eq!(db.vector_cache_stats().active_bytes, 0);
            assert!(
                db.vector_cache_stats().peak_bytes <= db.vector_cache_stats().memory_budget_bytes
            );
        }
    }
    let result = col
        .search_vectors(
            "v",
            &[0.0, 1.0],
            20,
            Some(Filter::In(
                "active".into(),
                vec![Value::Bool(true), Value::Bool(true)],
            )),
            &opts(VectorSearchMode::Exact),
        )
        .unwrap();
    assert_eq!(
        result.execution.distance_computations, 1425,
        "duplicate index ranges must not score vectors twice"
    );
}

#[test]
fn streamed_groups_match_full_ranking_with_missing_null_structured_keys_and_pages() {
    let db = seed(1500);
    db.set_vector_cache_budget(1024 * 1024);
    let col = db.collection("docs").unwrap();
    let truth = col
        .search_vectors("v", &[0.0, 1.0], 2000, None, &opts(VectorSearchMode::Exact))
        .unwrap()
        .hits;
    for filter in [None, Some(Filter::Eq("active".into(), Value::Bool(true)))] {
        for size in [1, 2, 4] {
            for offset in [0, 3, 20, 5000] {
                for threshold in [None, Some(0.5)] {
                    let mut groups = HashMap::<Vec<u8>, usize>::new();
                    let expected: Vec<_> = truth
                        .iter()
                        .filter(|h| {
                            filter
                                .as_ref()
                                .is_none_or(|f| f.matches(&h.document).unwrap())
                        })
                        .filter(|h| threshold.is_none_or(|t| h.score >= t))
                        .filter(|h| {
                            let key = postcard::to_allocvec(
                                h.document.get("meta.group").unwrap_or(&Value::Null),
                            )
                            .unwrap();
                            let n = groups.entry(key).or_default();
                            *n += 1;
                            *n <= size
                        })
                        .map(|h| (h.document.id, h.score.to_bits()))
                        .collect();
                    for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
                        let options = VectorQueryOptions {
                            group_by: Some("meta.group".into()),
                            group_size: Some(size),
                            offset,
                            score_threshold: threshold,
                            ..opts(mode)
                        };
                        let result = col
                            .search_vectors("v", &[0.0, 1.0], 10, filter.clone(), &options)
                            .unwrap();
                        assert_eq!(
                            result
                                .hits
                                .iter()
                                .map(|h| (h.document.id, h.score.to_bits()))
                                .collect::<Vec<_>>(),
                            expected
                                .iter()
                                .copied()
                                .skip(offset)
                                .take(10)
                                .collect::<Vec<_>>(),
                            "size={size} offset={offset} threshold={threshold:?} mode={mode:?}"
                        );
                        assert_eq!(
                            result.next_offset,
                            (expected.len() > offset.saturating_add(10))
                                .then_some(offset.saturating_add(10))
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn dense_bitmap_memory_fallback_and_projection_errors_release_the_allowance() {
    let db = seed(4096);
    let col = db.collection("docs").unwrap();
    let filter = Some(Filter::Eq("active".into(), Value::Bool(true)));
    let exact_options = VectorQueryOptions {
        offset: 3,
        group_by: Some("meta.group".into()),
        group_size: Some(2),
        ..opts(VectorSearchMode::Exact)
    };
    let expected = col
        .search_vectors("v", &[0.0, 1.0], 10, filter.clone(), &exact_options)
        .unwrap();
    db.set_vector_cache_budget(0);
    let result = col
        .search_vectors(
            "v",
            &[0.0, 1.0],
            10,
            filter,
            &VectorQueryOptions {
                mode: VectorSearchMode::Ann,
                ..exact_options
            },
        )
        .unwrap();
    assert_eq!(result.execution.reason, "memoryBudget");
    assert_eq!(
        result
            .hits
            .iter()
            .map(|h| (h.document.id, h.score.to_bits()))
            .collect::<Vec<_>>(),
        expected
            .hits
            .iter()
            .map(|h| (h.document.id, h.score.to_bits()))
            .collect::<Vec<_>>()
    );
    assert_eq!(result.next_offset, expected.next_offset);
    assert_eq!(db.vector_cache_stats().active_bytes, 0);
    assert!(db.vector_cache_stats().peak_bytes <= 65536);
    // The capped first pass stops before this late corrupt record. The second
    // streaming pass fails after the bitmap loan is acquired.
    db.set_vector_cache_budget(1024 * 1024);
    let doc = col
        .find_one(Filter::Eq("ordinal".into(), Value::Int(4095)))
        .unwrap()
        .unwrap();
    let mut txn = db.backend().begin_write().unwrap();
    txn.put("docs::docs", &doc.id.to_bytes(), &[255]).unwrap();
    txn.commit().unwrap();
    let result = col.search_vectors(
        "v",
        &[0.0, 1.0],
        10,
        Some(Filter::Exists("body".into(), true)),
        &opts(VectorSearchMode::Ann),
    );
    assert!(result.is_err());
    assert_eq!(db.vector_cache_stats().active_bytes, 0);
}

#[test]
fn nested_group_and_filter_fields_survive_projection() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    col.insert_many(
        (0..1100)
            .map(|i| {
                vec![
                    (
                        "literal".into(),
                        Value::Object(vec![
                            ("group".into(), Value::Int(i64::from(i % 2))),
                            ("keep".into(), Value::Bool(true)),
                        ]),
                    ),
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(f64::from(i)), Value::Float(1.0)]),
                    ),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_vector_index("v", 2, Some(VectorMetric::Euclidean), None)
        .unwrap();
    let options = VectorQueryOptions {
        group_by: Some("literal.group".into()),
        ..opts(VectorSearchMode::Exact)
    };
    let result = col
        .search_vectors(
            "v",
            &[0.0, 1.0],
            10,
            Some(Filter::Eq("literal.keep".into(), Value::Bool(true))),
            &options,
        )
        .unwrap();
    let groups: HashSet<_> = result
        .hits
        .iter()
        .map(|h| h.document.get("literal.group").unwrap().as_int().unwrap())
        .collect();
    assert_eq!(groups, HashSet::from([0, 1]));
}

#[cfg(feature = "encryption")]
#[test]
fn projected_group_keys_are_decrypted_before_applying_quotas() {
    let db = Database::open_in_memory().unwrap();
    let col = db
        .collection("docs")
        .unwrap()
        .with_field_encryption(vec!["parent".into()], zeroize::Zeroizing::new([42; 32]));
    col.insert_many(
        (0..30)
            .map(|i| {
                vec![
                    ("parent".into(), Value::Str((i % 2).to_string())),
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(f64::from(i)), Value::Float(1.0)]),
                    ),
                    ("body".into(), Value::Bytes(vec![42; 4096])),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_vector_index_with_options(
        "v",
        2,
        Some(VectorMetric::Euclidean),
        Some(GraphOptions::default()),
    )
    .unwrap();
    for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
        let result = col
            .search_vectors(
                "v",
                &[0.0, 1.0],
                10,
                None,
                &VectorQueryOptions {
                    group_by: Some("parent".into()),
                    ..opts(mode)
                },
            )
            .unwrap();
        assert_eq!(result.hits.len(), 2);
        assert_eq!(
            result.hits[0].document.get("parent"),
            Some(&Value::Str("0".into()))
        );
        assert_eq!(
            result.hits[1].document.get("parent"),
            Some(&Value::Str("1".into()))
        );
        assert_eq!(result.next_offset, None);
    }
}
