use std::collections::HashSet;
use std::ops::Bound;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
use taladb::engine::{KvPairs, ReadTxn, ScanFn, WriteTxn};
use taladb::{
    Database, Filter, GraphOptions, RedbBackend, StorageBackend, TalaDbError, Update, Value,
    VectorQueryOptions, VectorSearchMode,
};

#[derive(Default)]
struct Counts {
    documents: AtomicUsize,
    index_keys: AtomicUsize,
    index_points: AtomicUsize,
    vector_points: AtomicUsize,
    vector_scans: AtomicUsize,
    on_index_scan: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}
impl Counts {
    fn reset(&self) {
        self.documents.store(0, Ordering::SeqCst);
        self.index_keys.store(0, Ordering::SeqCst);
        self.index_points.store(0, Ordering::SeqCst);
        self.vector_points.store(0, Ordering::SeqCst);
        self.vector_scans.store(0, Ordering::SeqCst);
    }
    fn reads(&self, table: &str, n: usize) {
        if table.starts_with("idx::") || table.starts_with("cidx::") {
            self.index_points.fetch_add(n, Ordering::SeqCst);
        }
        if table.starts_with("docs::") {
            self.documents.fetch_add(n, Ordering::SeqCst);
        }
        if table.starts_with("vec::") {
            self.vector_points.fetch_add(n, Ordering::SeqCst);
        }
    }
}
struct Backend {
    inner: RedbBackend,
    counts: Arc<Counts>,
}
impl StorageBackend for Backend {
    fn begin_read(&self) -> Result<Box<dyn ReadTxn + '_>, TalaDbError> {
        Ok(Box::new(Reader {
            inner: self.inner.begin_read()?,
            counts: self.counts.clone(),
        }))
    }
    fn begin_write(&self) -> Result<Box<dyn WriteTxn + '_>, TalaDbError> {
        self.inner.begin_write()
    }
}
struct Reader<'a> {
    inner: Box<dyn ReadTxn + 'a>,
    counts: Arc<Counts>,
}
impl ReadTxn for Reader<'_> {
    fn get(&self, t: &str, k: &[u8]) -> Result<Option<Vec<u8>>, TalaDbError> {
        self.counts.reads(t, 1);
        self.inner.get(t, k)
    }
    fn get_many(&self, t: &str, keys: &[&[u8]]) -> Result<Vec<Option<Vec<u8>>>, TalaDbError> {
        self.counts.reads(t, keys.len());
        self.inner.get_many(t, keys)
    }
    fn range(&self, t: &str, s: Bound<&[u8]>, e: Bound<&[u8]>) -> Result<KvPairs, TalaDbError> {
        self.inner.range(t, s, e)
    }
    fn scan_all(&self, t: &str) -> Result<KvPairs, TalaDbError> {
        self.inner.scan_all(t)
    }
    fn list_tables(&self) -> Result<Vec<String>, TalaDbError> {
        self.inner.list_tables()
    }
    fn count_entries(&self, t: &str) -> Result<u64, TalaDbError> {
        self.inner.count_entries(t)
    }
    fn scan(
        &self,
        t: &str,
        s: Bound<&[u8]>,
        e: Bound<&[u8]>,
        f: ScanFn<'_>,
    ) -> Result<(), TalaDbError> {
        if t.starts_with("idx::") {
            let action = self.counts.on_index_scan.lock().unwrap().take();
            if let Some(action) = action {
                action();
            }
        }
        if t.starts_with("vec::") {
            self.counts.vector_scans.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.scan(t, s, e, &mut |k, v| {
            if t.starts_with("idx::") || t.starts_with("cidx::") {
                self.counts.index_keys.fetch_add(1, Ordering::SeqCst);
            }
            if t.starts_with("docs::") {
                self.counts.documents.fetch_add(1, Ordering::SeqCst);
            }
            f(k, v)
        })
    }
}

fn database() -> (Database, Arc<Counts>) {
    let counts = Arc::new(Counts::default());
    let db = Database::open_with_backend(Box::new(Backend {
        inner: RedbBackend::open_in_memory().unwrap(),
        counts: counts.clone(),
    }))
    .unwrap();
    (db, counts)
}
fn seed(db: &Database) -> taladb::Collection {
    let col = db.collection("docs").unwrap();
    col.insert_many(
        (0..80)
            .map(|i| {
                vec![
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(f64::from(i + 1)), Value::Float(1.0)]),
                    ),
                    ("x".into(), Value::Int(i.into())),
                    ("group".into(), Value::Str((i % 2).to_string())),
                    ("residual".into(), Value::Bool(i % 3 == 0)),
                    ("body".into(), Value::Bytes(vec![42; 8192])),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_index("x").unwrap();
    col.create_index("group").unwrap();
    col.create_vector_index_with_options(
        "v",
        2,
        None,
        Some(GraphOptions {
            m: 4,
            ef_construction: 16,
            ..Default::default()
        }),
    )
    .unwrap();
    col
}
fn options(mode: VectorSearchMode) -> VectorQueryOptions {
    VectorQueryOptions {
        mode,
        ef_search: Some(100),
        ..Default::default()
    }
}

#[test]
fn covered_filters_only_load_returned_bodies_and_choose_vector_read_strategy() {
    let (db, c) = database();
    let col = seed(&db);
    let sparse = Filter::Lt("x".into(), Value::Int(4));
    let dense = Filter::Gte("x".into(), Value::Int(10));
    for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
        c.reset();
        let hits = col
            .search_vectors("v", &[1.0, 1.0], 2, Some(sparse.clone()), &options(mode))
            .unwrap()
            .hits;
        assert_eq!(hits.len(), 2);
        assert!(
            c.documents.load(Ordering::SeqCst) <= 3,
            "only output documents may be read"
        );
        if mode == VectorSearchMode::Exact {
            assert_eq!(c.vector_points.load(Ordering::SeqCst), 4);
            assert_eq!(c.vector_scans.load(Ordering::SeqCst), 0);
        }
    }
    db.set_vector_cache_budget(0);
    c.reset();
    col.search_vectors(
        "v",
        &[1.0, 1.0],
        2,
        Some(dense.clone()),
        &options(VectorSearchMode::Exact),
    )
    .unwrap();
    assert_eq!(c.vector_points.load(Ordering::SeqCst), 0);
    assert_eq!(c.vector_scans.load(Ordering::SeqCst), 1);
    assert_eq!(c.documents.load(Ordering::SeqCst), 3);
    assert_eq!(db.vector_cache_stats().retained_bytes, 0);
    db.set_vector_cache_budget(1024 * 1024);
    col.search_vectors("v", &[1.0, 1.0], 2, None, &options(VectorSearchMode::Exact))
        .unwrap();
    c.reset();
    col.search_vectors(
        "v",
        &[1.0, 1.0],
        2,
        Some(dense),
        &options(VectorSearchMode::Exact),
    )
    .unwrap();
    assert_eq!(c.vector_points.load(Ordering::SeqCst), 0);
    assert_eq!(c.vector_scans.load(Ordering::SeqCst), 0);
    assert_eq!(c.documents.load(Ordering::SeqCst), 3);
    c.reset();
    let partial = Filter::And(vec![
        Filter::Eq("residual".into(), Value::Bool(true)),
        sparse,
    ]);
    let result = col
        .search_vectors(
            "v",
            &[1.0, 1.0],
            10,
            Some(partial),
            &options(VectorSearchMode::Exact),
        )
        .unwrap();
    assert_eq!(result.hits.len(), 2);
    assert_eq!(
        c.documents.load(Ordering::SeqCst),
        6,
        "four residual checks plus two output rows"
    );
    let invalid = Filter::And(vec![
        Filter::Eq("x".into(), Value::Int(-1)),
        Filter::Regex("body".into(), "[".into()),
    ]);
    assert!(
        col.search_vectors(
            "v",
            &[1.0, 1.0],
            2,
            Some(invalid),
            &options(VectorSearchMode::Exact)
        )
        .is_err()
    );
}

#[test]
fn indexed_predicates_match_document_evaluation_for_arrays_numbers_and_residuals() {
    let (db, c) = database();
    let col = db.collection("docs").unwrap();
    let values = [
        Value::Int(0),
        Value::Float(-0.0),
        Value::Float(0.0),
        Value::Int(9_007_199_254_740_993),
        Value::Float(9_007_199_254_740_992.0),
        Value::Int(i64::MAX),
        Value::Int(i64::MIN),
        Value::Float(f64::INFINITY),
        Value::Float(f64::NEG_INFINITY),
        Value::Float(f64::NAN),
        Value::Array(vec![Value::Int(-10), Value::Int(20), Value::Int(20)]),
        Value::Array(vec![Value::Float(-0.0), Value::Int(0)]),
        Value::Null,
        Value::Bool(false),
        Value::Str("a\0b".into()),
        Value::Bytes(vec![0, 255, 0]),
        Value::Object(vec![("a".into(), Value::Int(1))]),
    ];
    for (i, x) in values.iter().enumerate() {
        col.insert(vec![
            ("x".into(), x.clone()),
            ("group".into(), Value::Str("g".into())),
            (
                "meta".into(),
                Value::Object(vec![("name".into(), Value::Str("a\0b".into()))]),
            ),
            ("residual".into(), Value::Bool(i % 2 == 0)),
            (
                "v".into(),
                Value::Array(vec![Value::Float(i as f64 + 1.0), Value::Float(1.0)]),
            ),
        ])
        .unwrap();
    }
    // A missing x and a missing vector are different from an explicit null.
    col.insert(vec![(
        "v".into(),
        Value::Array(vec![Value::Float(1.0), Value::Float(1.0)]),
    )])
    .unwrap();
    col.insert(vec![("x".into(), Value::Int(0))]).unwrap();
    col.insert(vec![
        ("x".into(), Value::Int(0)),
        ("group".into(), Value::Str("other".into())),
        (
            "v".into(),
            Value::Array(vec![Value::Float(99.0), Value::Float(1.0)]),
        ),
    ])
    .unwrap();
    for field in ["x", "meta.name"] {
        col.create_index(field).unwrap();
    }
    col.create_compound_index(&["group", "x"]).unwrap();
    col.create_vector_index_with_options(
        "v",
        2,
        None,
        Some(GraphOptions {
            m: 4,
            ef_construction: 16,
            ..Default::default()
        }),
    )
    .unwrap();
    let truth = col
        .search_vectors(
            "v",
            &[1.0, 1.0],
            100,
            None,
            &options(VectorSearchMode::Exact),
        )
        .unwrap()
        .hits;
    let mut filters = Vec::new();
    for x in &values {
        filters.extend([
            Filter::Eq("x".into(), x.clone()),
            Filter::Gt("x".into(), x.clone()),
            Filter::Gte("x".into(), x.clone()),
            Filter::Lt("x".into(), x.clone()),
            Filter::Lte("x".into(), x.clone()),
            Filter::In(
                "x".into(),
                vec![x.clone(), Value::Float(0.0), Value::Float(f64::NAN)],
            ),
            Filter::And(vec![
                Filter::Eq("x".into(), x.clone()),
                Filter::Eq("group".into(), Value::Str("g".into())),
            ]),
        ]);
    }
    let eq = Filter::Eq("x".into(), Value::Int(0));
    let residual = Filter::Eq("residual".into(), Value::Bool(true));
    filters.extend([
        Filter::And(vec![
            Filter::Eq("group".into(), Value::Str("g".into())),
            Filter::Gte("x".into(), Value::Int(0)),
        ]),
        Filter::Or(vec![
            Filter::And(vec![eq.clone(), residual.clone()]),
            Filter::And(vec![
                Filter::Gt("x".into(), Value::Int(0)),
                residual.clone(),
            ]),
        ]),
        Filter::And(vec![
            Filter::Gte("x".into(), Value::Int(10)),
            Filter::Lte("x".into(), Value::Int(-5)),
        ]),
        Filter::And(vec![eq.clone(), Filter::Eq("x".into(), Value::Float(-0.0))]),
        Filter::And(vec![
            Filter::Eq("group".into(), Value::Str("g".into())),
            eq.clone(),
        ]),
        Filter::And(vec![
            Filter::Eq("group".into(), Value::Str("g".into())),
            Filter::Eq("x".into(), Value::Float(0.0)),
        ]),
        Filter::And(vec![residual.clone(), eq.clone()]),
        Filter::Or(vec![residual, eq.clone()]),
        Filter::And(vec![
            Filter::Eq("group".into(), Value::Str("g".into())),
            eq.clone(),
            Filter::Exists("residual".into(), true),
        ]),
        Filter::Or(vec![
            eq.clone(),
            Filter::Eq("meta.name".into(), Value::Str("a\0b".into())),
        ]),
        Filter::In("x".into(), vec![]),
        Filter::Not(Box::new(eq)),
        Filter::Exists("x".into(), false),
        Filter::Eq(
            "_id".into(),
            Value::Str(truth[0].document.id.to_string().to_lowercase()),
        ),
    ]);
    for filter in filters {
        let expected: Vec<_> = truth
            .iter()
            .filter(|r| filter.matches(&r.document).unwrap())
            .map(|r| (r.document.id, r.score.to_bits()))
            .collect();
        for mode in [VectorSearchMode::Exact, VectorSearchMode::Ann] {
            let result = col
                .search_vectors("v", &[1.0, 1.0], 100, Some(filter.clone()), &options(mode))
                .unwrap();
            let got: Vec<_> = result
                .hits
                .iter()
                .map(|r| (r.document.id, r.score.to_bits()))
                .collect();
            assert_eq!(got, expected, "filter {filter:?}, mode {mode:?}");
        }
        let expected_docs: HashSet<_> = col
            .find(Filter::All)
            .unwrap()
            .into_iter()
            .filter(|d| filter.matches(d).unwrap())
            .map(|d| d.id)
            .collect();
        let got_docs: HashSet<_> = col
            .find(filter.clone())
            .unwrap()
            .into_iter()
            .map(|d| d.id)
            .collect();
        assert_eq!(got_docs, expected_docs, "document query {filter:?}");
    }
    c.reset();
    let filter = Filter::And(vec![
        Filter::Eq("group".into(), Value::Str("g".into())),
        Filter::Eq("x".into(), Value::Float(0.0)),
    ]);
    col.search_vectors(
        "v",
        &[1.0, 1.0],
        1,
        Some(filter),
        &options(VectorSearchMode::Exact),
    )
    .unwrap();
    assert_eq!(
        c.documents.load(Ordering::SeqCst),
        2,
        "compound-only fields must be covered without residual reads"
    );
}

#[test]
fn covered_keys_vectors_and_output_share_the_read_snapshot() {
    let (db, c) = database();
    let col = seed(&db);
    let first = col
        .find_one(Filter::Eq("x".into(), Value::Int(0)))
        .unwrap()
        .unwrap();
    let writer = db.collection("docs").unwrap();
    *c.on_index_scan.lock().unwrap() = Some(Box::new(move || {
        writer
            .update_one(
                Filter::Eq("_id".into(), Value::Str(first.id.to_string())),
                Update::Set(vec![("group".into(), Value::Str("changed".into()))]),
            )
            .unwrap();
    }));
    let filter = Filter::And(vec![
        Filter::Eq("group".into(), Value::Str("0".into())),
        Filter::Eq("x".into(), Value::Int(0)),
    ]);
    let old = col
        .search_vectors(
            "v",
            &[1.0, 1.0],
            1,
            Some(filter.clone()),
            &options(VectorSearchMode::Exact),
        )
        .unwrap();
    assert_eq!(old.hits.len(), 1);
    assert_eq!(
        old.hits[0].document.get("group"),
        Some(&Value::Str("0".into()))
    );
    assert!(
        col.search_vectors(
            "v",
            &[1.0, 1.0],
            1,
            Some(filter),
            &options(VectorSearchMode::Exact)
        )
        .unwrap()
        .hits
        .is_empty()
    );
}

#[test]
fn selective_intersections_probe_broad_equality_without_loading_bodies() {
    let (db, counts) = database();
    let col = db.collection("docs").unwrap();
    col.insert_many(
        (0..512)
            .map(|i| {
                vec![
                    (
                        "wide".into(),
                        Value::Array(vec![Value::Str("all".into()), Value::Str("all".into())]),
                    ),
                    ("x".into(), Value::Int(i)),
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(1.0), Value::Float(i as f64 + 1.0)]),
                    ),
                    ("body".into(), Value::Bytes(vec![42; 8192])),
                ]
            })
            .collect(),
    )
    .unwrap();
    for f in ["wide", "x"] {
        col.create_index(f).unwrap();
    }
    col.create_vector_index("v", 2, None, None).unwrap();
    for wide in [
        Filter::Eq("wide".into(), Value::Str("all".into())),
        Filter::In(
            "wide".into(),
            vec![Value::Str("all".into()), Value::Str("absent".into())],
        ),
    ] {
        for reverse in [false, true] {
            let mut filters = vec![wide.clone(), Filter::Eq("x".into(), Value::Int(511))];
            if reverse {
                filters.reverse();
            }
            counts.reset();
            let result = col
                .search_vectors(
                    "v",
                    &[1.0, 1.0],
                    10,
                    Some(Filter::And(filters)),
                    &options(VectorSearchMode::Exact),
                )
                .unwrap();
            assert_eq!(result.hits.len(), 1);
            assert_eq!(result.hits[0].document.get("x"), Some(&Value::Int(511)));
            assert_eq!(counts.index_keys.load(Ordering::SeqCst), 65);
            assert!(counts.index_points.load(Ordering::SeqCst) <= 2);
            assert_eq!(counts.documents.load(Ordering::SeqCst), 1);
        }
    }
}

fn bounds(lo: Value, hi: Value) -> Filter {
    Filter::And(vec![
        Filter::Gte("x".into(), lo),
        Filter::Lte("x".into(), hi),
    ])
}

#[test]
fn scalar_range_narrowing_tracks_array_backfills_writes_deletes_and_legacy_metadata() {
    let (db, counts) = database();
    let col = db.collection("docs").unwrap();
    let ids = col
        .insert_many(
            (0..512)
                .map(|i| {
                    vec![
                        ("x".into(), Value::Int(i)),
                        (
                            "v".into(),
                            Value::Array(vec![Value::Float(1.0), Value::Float(i as f64 + 1.0)]),
                        ),
                    ]
                })
                .collect(),
        )
        .unwrap();
    col.create_index("x").unwrap();
    col.create_vector_index("v", 2, None, None).unwrap();
    let search = |f| {
        col.search_vectors(
            "v",
            &[1.0, 1.0],
            1000,
            Some(f),
            &options(VectorSearchMode::Exact),
        )
        .unwrap()
        .hits
    };
    counts.reset();
    assert_eq!(search(bounds(Value::Int(250), Value::Int(253))).len(), 4);
    assert_eq!(counts.index_keys.load(Ordering::SeqCst), 4);
    let id = Filter::Eq("_id".into(), Value::Str(ids[0].to_string()));
    let crossing = Value::Array(vec![Value::Int(-10), Value::Int(600)]);
    col.update_one(
        id.clone(),
        Update::Set(vec![("x".into(), crossing.clone())]),
    )
    .unwrap();
    assert_eq!(search(bounds(Value::Int(550), Value::Int(-5))).len(), 1);
    // Rebuild must backfill the array count, too.
    col.drop_index("x").unwrap();
    col.create_index("x").unwrap();
    assert_eq!(search(bounds(Value::Int(550), Value::Int(-5))).len(), 1);
    col.update_one(id.clone(), Update::Set(vec![("x".into(), Value::Int(0))]))
        .unwrap();
    counts.reset();
    assert_eq!(search(bounds(Value::Int(250), Value::Int(253))).len(), 4);
    assert_eq!(counts.index_keys.load(Ordering::SeqCst), 4);
    let added = col
        .insert(vec![
            ("x".into(), crossing.clone()),
            (
                "v".into(),
                Value::Array(vec![Value::Float(1.0), Value::Float(1.0)]),
            ),
        ])
        .unwrap();
    assert_eq!(search(bounds(Value::Int(550), Value::Int(-5))).len(), 1);
    col.delete_one(Filter::Eq("_id".into(), Value::Str(added.to_string())))
        .unwrap();
    counts.reset();
    assert_eq!(search(bounds(Value::Int(250), Value::Int(253))).len(), 4);
    assert_eq!(counts.index_keys.load(Ordering::SeqCst), 4);
    // Absent/corrupt metadata must not be reconstructed from only new writes.
    for bytes in [None, Some(vec![0, 0])] {
        let mut txn = db.backend().begin_write().unwrap();
        if let Some(bytes) = bytes {
            txn.put("meta::index_arrays", b"docs::x", &bytes).unwrap();
        } else {
            txn.delete("meta::index_arrays", b"docs::x").unwrap();
        }
        txn.commit().unwrap();
        col.update_one(
            id.clone(),
            Update::Set(vec![("x".into(), crossing.clone())]),
        )
        .unwrap();
        assert_eq!(search(bounds(Value::Int(550), Value::Int(-5))).len(), 1);
        col.update_one(id.clone(), Update::Set(vec![("x".into(), Value::Int(0))]))
            .unwrap();
    }
    // An empty array contributes no index keys but still affects metadata.
    col.update_one(
        id.clone(),
        Update::Set(vec![("x".into(), Value::Array(vec![]))]),
    )
    .unwrap();
    col.drop_index("x").unwrap();
    col.create_index("x").unwrap();
    let txn = db.backend().begin_read().unwrap();
    assert_eq!(
        txn.get("meta::index_arrays", b"docs::x").unwrap(),
        Some(1u64.to_le_bytes().to_vec())
    );
    drop(txn);
    col.update_one(id, Update::Unset(vec!["x".into()])).unwrap();
    counts.reset();
    assert_eq!(search(bounds(Value::Int(250), Value::Int(253))).len(), 4);
    assert_eq!(counts.index_keys.load(Ordering::SeqCst), 4);
}

#[test]
fn narrowed_scalar_bounds_match_document_semantics_for_numeric_extremes() {
    let (db, _) = database();
    let col = db.collection("docs").unwrap();
    let values = vec![
        Value::Int(i64::MIN),
        Value::Int(i64::MAX),
        Value::Int(9_007_199_254_740_993),
        Value::Int(-10),
        Value::Int(0),
        Value::Int(20),
        Value::Float(-0.0),
        Value::Float(0.0),
        Value::Float(9_007_199_254_740_992.0),
        Value::Float(-10.5),
        Value::Float(20.5),
        Value::Float(f64::INFINITY),
        Value::Float(f64::NEG_INFINITY),
        Value::Float(f64::NAN),
        Value::Null,
        Value::Bool(false),
        Value::Str("a\0b".into()),
        Value::Str("z".into()),
    ];
    col.insert_many(
        values
            .iter()
            .map(|x| {
                vec![
                    ("x".into(), x.clone()),
                    (
                        "v".into(),
                        Value::Array(vec![Value::Float(1.0), Value::Float(1.0)]),
                    ),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_index("x").unwrap();
    col.create_vector_index("v", 2, None, None).unwrap();
    let docs = col.find(Filter::All).unwrap();
    for lo in &values {
        for hi in &values {
            for inclusive in [true, false] {
                for upper_inclusive in [true, false] {
                    let f = Filter::And(vec![
                        if inclusive {
                            Filter::Gte("x".into(), lo.clone())
                        } else {
                            Filter::Gt("x".into(), lo.clone())
                        },
                        if upper_inclusive {
                            Filter::Lte("x".into(), hi.clone())
                        } else {
                            Filter::Lt("x".into(), hi.clone())
                        },
                        Filter::Gte("x".into(), lo.clone()), // repeated bounds must remain safe
                    ]);
                    let expected: HashSet<_> = docs
                        .iter()
                        .filter(|d| f.matches(d).unwrap())
                        .map(|d| d.id)
                        .collect();
                    let got: HashSet<_> = col
                        .search_vectors(
                            "v",
                            &[1.0, 1.0],
                            100,
                            Some(f.clone()),
                            &options(VectorSearchMode::Exact),
                        )
                        .unwrap()
                        .hits
                        .into_iter()
                        .map(|h| h.document.id)
                        .collect();
                    assert_eq!(got, expected, "filter {f:?}");
                }
            }
        }
    }
}

#[test]
fn array_metadata_is_atomic_persistent_and_rebuilt_from_documents() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filters.redb");
    {
        let db = Database::open(&path).unwrap();
        let col = db.collection("docs").unwrap();
        col.insert(vec![
            ("x".into(), Value::Int(1)),
            ("y".into(), Value::Array(vec![Value::Int(2)])),
        ])
        .unwrap();
        col.create_index("x").unwrap();
        col.create_compound_index(&["x", "y"]).unwrap();
        // Compound indexes reject two array fields, after secondary maintenance
        // has already run. Dropping the failed transaction must undo its count.
        assert!(
            col.update_many(
                Filter::All,
                Update::Set(vec![("x".into(), Value::Array(vec![Value::Int(1)]))])
            )
            .is_err()
        );
        let txn = db.backend().begin_read().unwrap();
        assert_eq!(
            txn.get("meta::index_arrays", b"docs::x").unwrap(),
            Some(0u64.to_le_bytes().to_vec())
        );
    }
    let db = Database::open(&path).unwrap();
    let col = db.collection("docs").unwrap();
    assert_eq!(
        col.find_one(Filter::All).unwrap().unwrap().get("x"),
        Some(&Value::Int(1))
    );
    col.drop_compound_index(&["x", "y"]).unwrap();
    col.update_many(
        Filter::All,
        Update::Set(vec![("x".into(), Value::Array(vec![]))]),
    )
    .unwrap();
    // Rebuilding old secondary indexes must derive counts, not trust metadata.
    let mut txn = db.backend().begin_write().unwrap();
    txn.delete("meta::index_arrays", b"docs::x").unwrap();
    taladb::migration::rebuild_secondary_indexes(txn.as_mut()).unwrap();
    txn.commit().unwrap();
    {
        let txn = db.backend().begin_read().unwrap();
        assert_eq!(
            txn.get("meta::index_arrays", b"docs::x").unwrap(),
            Some(1u64.to_le_bytes().to_vec())
        );
    }
    col.delete_many(Filter::All).unwrap();
    let txn = db.backend().begin_read().unwrap();
    assert_eq!(
        txn.get("meta::index_arrays", b"docs::x").unwrap(),
        Some(0u64.to_le_bytes().to_vec())
    );
}
