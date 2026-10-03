use std::ops::Bound;
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
};
use taladb::engine::{KvPairs, ReadTxn, WriteTxn};
use taladb::{
    Database, GraphOptions, RedbBackend, StorageBackend, TalaDbError, Value, VectorQueryOptions,
    VectorSearchMode,
};

#[derive(Default)]
struct Control {
    fail_commit: AtomicBool,
    before: AtomicU64,
    old_node_reads: AtomicUsize,
}
struct Backend {
    inner: RedbBackend,
    control: Arc<Control>,
}
impl StorageBackend for Backend {
    fn begin_read(&self) -> Result<Box<dyn ReadTxn + '_>, TalaDbError> {
        self.inner.begin_read()
    }
    fn begin_write(&self) -> Result<Box<dyn WriteTxn + '_>, TalaDbError> {
        Ok(Box::new(Writer {
            inner: self.inner.begin_write()?,
            control: self.control.clone(),
        }))
    }
}
struct Writer<'a> {
    inner: Box<dyn WriteTxn + 'a>,
    control: Arc<Control>,
}
impl WriteTxn for Writer<'_> {
    fn put(&mut self, t: &str, k: &[u8], v: &[u8]) -> Result<(), TalaDbError> {
        self.inner.put(t, k, v)
    }
    fn delete(&mut self, t: &str, k: &[u8]) -> Result<Option<Vec<u8>>, TalaDbError> {
        self.inner.delete(t, k)
    }
    fn get(&self, t: &str, k: &[u8]) -> Result<Option<Vec<u8>>, TalaDbError> {
        if t.starts_with("hnsw::")
            && k.len() == 9
            && k[0] == 1
            && u64::from_be_bytes(k[1..].try_into().unwrap())
                < self.control.before.load(Ordering::SeqCst)
        {
            self.control.old_node_reads.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.get(t, k)
    }
    fn range(&self, t: &str, s: Bound<&[u8]>, e: Bound<&[u8]>) -> Result<KvPairs, TalaDbError> {
        self.inner.range(t, s, e)
    }
    fn list_tables(&self) -> Result<Vec<String>, TalaDbError> {
        self.inner.list_tables()
    }
    fn delete_table(&mut self, t: &str) -> Result<bool, TalaDbError> {
        self.inner.delete_table(t)
    }
    fn count_entries(&self, t: &str) -> Result<u64, TalaDbError> {
        self.inner.count_entries(t)
    }
    fn commit(self: Box<Self>) -> Result<(), TalaDbError> {
        if self.control.fail_commit.swap(false, Ordering::SeqCst) {
            return Err(TalaDbError::InvalidOperation(
                "injected commit failure".into(),
            ));
        }
        self.inner.commit()
    }
}

#[test]
fn staged_nodes_are_reused_and_failed_commits_do_not_poison_a_retry() {
    let control = Arc::new(Control::default());
    let db = Database::open_with_backend(Box::new(Backend {
        inner: RedbBackend::open_in_memory().unwrap(),
        control: control.clone(),
    }))
    .unwrap();
    db.set_vector_cache_budget(1024 * 1024);
    let col = db.collection("docs").unwrap();
    col.create_vector_index("v", 8, None, None).unwrap();
    col.insert_many(
        (0..96)
            .map(|i| {
                vec![(
                    "v".into(),
                    Value::Array(
                        (0..8)
                            .map(|j| Value::Float(f64::from(i * 17 + j * 13).sin()))
                            .collect(),
                    ),
                )]
            })
            .collect(),
    )
    .unwrap();
    let build = col
        .begin_vector_build(
            "v",
            Some(GraphOptions {
                m: 4,
                ef_construction: 32,
                ..Default::default()
            }),
        )
        .unwrap();
    assert_eq!(
        col.step_vector_build("v", &build.id, 16).unwrap().processed,
        16
    );
    assert_eq!(db.vector_cache_stats().graph_indexes, 1);
    control.before.store(16, Ordering::SeqCst);
    control.old_node_reads.store(0, Ordering::SeqCst);
    // A different collection handle must see the committed cache too.
    assert_eq!(
        db.collection("docs")
            .unwrap()
            .step_vector_build("v", &build.id, 16)
            .unwrap()
            .processed,
        32
    );
    assert_eq!(control.old_node_reads.load(Ordering::SeqCst), 0);
    control.fail_commit.store(true, Ordering::SeqCst);
    assert!(col.step_vector_build("v", &build.id, 16).is_err());
    assert_eq!(db.vector_cache_stats().graph_indexes, 0);
    assert_eq!(
        col.vector_index_status("v")
            .unwrap()
            .build
            .unwrap()
            .processed,
        32
    );
    assert_eq!(
        col.step_vector_build("v", &build.id, 16).unwrap().processed,
        48
    );
    assert_eq!(db.vector_cache_stats().graph_indexes, 1);
    db.set_vector_cache_budget(0);
    assert_eq!(
        col.step_vector_build("v", &build.id, 16).unwrap().processed,
        64
    );
    assert_eq!(db.vector_cache_stats().retained_bytes, 0);
    db.set_vector_cache_budget(1024 * 1024);
    col.step_vector_build("v", &build.id, 16).unwrap();
    assert_eq!(db.vector_cache_stats().graph_indexes, 1);
    assert_eq!(
        col.step_vector_build("v", &build.id, 16).unwrap().state,
        "ready"
    );
    assert_eq!(db.vector_cache_stats().graph_indexes, 0);
    let options = VectorQueryOptions {
        mode: VectorSearchMode::Ann,
        ..Default::default()
    };
    let query = [0.5; 8];
    let before = col.search_vectors("v", &query, 5, None, &options).unwrap();
    assert_eq!(db.vector_cache_stats().graph_indexes, 1);
    let cancelled = col.begin_vector_build("v", None).unwrap();
    col.step_vector_build("v", &cancelled.id, 16).unwrap();
    assert_eq!(db.vector_cache_stats().graph_indexes, 2);
    col.cancel_vector_build("v", &cancelled.id).unwrap();
    assert_eq!(db.vector_cache_stats().graph_indexes, 1);
    let after = col.search_vectors("v", &query, 5, None, &options).unwrap();
    assert_eq!(
        before
            .hits
            .iter()
            .map(|h| h.document.id)
            .collect::<Vec<_>>(),
        after.hits.iter().map(|h| h.document.id).collect::<Vec<_>>()
    );
    let failed = col.begin_vector_build("v", None).unwrap();
    col.step_vector_build("v", &failed.id, 16).unwrap();
    col.insert(vec![(
        "v".into(),
        Value::Array(vec![Value::Float(0.25); 8]),
    )])
    .unwrap();
    assert_eq!(
        col.step_vector_build("v", &failed.id, 16).unwrap().state,
        "failed"
    );
    assert!(db.vector_cache_stats().graph_indexes <= 1);
}
