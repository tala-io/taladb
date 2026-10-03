//! Deterministic overlap: pause a graph read after its memory lease is taken.
use std::ops::Bound;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc,
};
use std::time::Duration;
use taladb::engine::{KvPairs, ReadTxn, WriteTxn};
use taladb::{
    Database, Filter, GraphOptions, Quantization, RedbBackend, StorageBackend, TalaDbError, Value,
    VectorQueryOptions, VectorSearchMode,
};

struct Gate {
    armed: AtomicBool,
    entered: mpsc::Sender<()>,
    resume: Mutex<mpsc::Receiver<()>>,
    fail: AtomicBool,
}
struct Backend {
    inner: RedbBackend,
    gate: Arc<Gate>,
}
struct Reader<'a> {
    inner: Box<dyn ReadTxn + 'a>,
    gate: Arc<Gate>,
}
impl StorageBackend for Backend {
    fn begin_read(&self) -> Result<Box<dyn ReadTxn + '_>, TalaDbError> {
        Ok(Box::new(Reader {
            inner: self.inner.begin_read()?,
            gate: self.gate.clone(),
        }))
    }
    fn begin_write(&self) -> Result<Box<dyn WriteTxn + '_>, TalaDbError> {
        self.inner.begin_write()
    }
}
impl ReadTxn for Reader<'_> {
    fn get(&self, t: &str, k: &[u8]) -> Result<Option<Vec<u8>>, TalaDbError> {
        if t.starts_with("hnsw::")
            && k.len() == 9
            && k[0] == 1
            && self.gate.armed.swap(false, Ordering::SeqCst)
        {
            self.gate.entered.send(()).unwrap();
            self.gate
                .resume
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10))
                .unwrap();
            if self.gate.fail.load(Ordering::SeqCst) {
                return Err(TalaDbError::Storage("injected read failure".into()));
            }
        }
        self.inner.get(t, k)
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
}
fn setup(count: usize) -> (Database, Arc<Gate>, mpsc::Receiver<()>, mpsc::Sender<()>) {
    let (entered, rx) = mpsc::channel();
    let (tx, resume) = mpsc::channel();
    let gate = Arc::new(Gate {
        armed: AtomicBool::new(false),
        entered,
        resume: Mutex::new(resume),
        fail: AtomicBool::new(false),
    });
    let db = Database::open_with_backend(Box::new(Backend {
        inner: RedbBackend::open_in_memory().unwrap(),
        gate: gate.clone(),
    }))
    .unwrap();
    let col = db.collection("docs").unwrap();
    col.insert_many(
        (0..count)
            .map(|i| {
                vec![
                    ("n".into(), Value::Int(i as i64)),
                    ("group".into(), Value::Int((i % 3) as i64)),
                    (
                        "v".into(),
                        Value::Array(
                            (0..8)
                                .map(|j| Value::Float(((i * 17 + j * 13) as f64).sin()))
                                .collect(),
                        ),
                    ),
                ]
            })
            .collect(),
    )
    .unwrap();
    col.create_vector_index_with_options(
        "v",
        8,
        None,
        Some(GraphOptions {
            m: 4,
            ef_construction: 32,
            quantization: Quantization::Binary,
        }),
    )
    .unwrap();
    (db, gate, rx, tx)
}
const QUERY: [f32; 8] = [0.0, 0.42, 0.76, 0.96, 0.99, 0.83, 0.51, 0.1];
fn opts(mode: VectorSearchMode) -> VectorQueryOptions {
    VectorQueryOptions {
        mode,
        ef_search: Some(64),
        ..Default::default()
    }
}
fn signature(r: &taladb::VectorQueryResult) -> Vec<(ulid::Ulid, u32)> {
    r.hits
        .iter()
        .map(|h| (h.document.id, h.score.to_bits()))
        .collect()
}

#[test]
fn overlapping_ann_uses_exact_with_filters_pagination_and_threshold_on_the_same_snapshot() {
    for budget in [1024 * 1024, 8 * 1024 * 1024] {
        let (db, gate, entered, resume) = setup(128);
        db.set_vector_cache_budget(budget);
        gate.armed.store(true, Ordering::SeqCst);
        let col = db.collection("docs").unwrap();
        let running = std::thread::spawn(move || {
            col.search_vectors("v", &QUERY, 5, None, &opts(VectorSearchMode::Ann))
                .unwrap()
        });
        entered.recv_timeout(Duration::from_secs(10)).unwrap();
        let stats = db.vector_cache_stats();
        assert_eq!(stats.active_bytes, budget);
        assert!(stats.active_bytes + stats.retained_bytes <= stats.memory_budget_bytes);
        let col = db.collection("docs").unwrap();
        let filter = Some(Filter::Eq("group".into(), Value::Int(1)));
        let mut options = opts(VectorSearchMode::Ann);
        options.offset = 2;
        options.score_threshold = Some(-0.5);
        options.group_by = Some("group".into());
        options.group_size = Some(10);
        let fallback = col
            .search_vectors("v", &QUERY, 3, filter.clone(), &options)
            .unwrap();
        assert_eq!(fallback.execution.path, "exact");
        assert_eq!(fallback.execution.reason, "memoryBudget");
        options.mode = VectorSearchMode::Exact;
        let exact = col
            .search_vectors("v", &QUERY, 3, filter, &options)
            .unwrap();
        assert_eq!(signature(&fallback), signature(&exact));
        assert_eq!(fallback.next_offset, exact.next_offset);
        resume.send(()).unwrap();
        assert_eq!(running.join().unwrap().execution.path, "hnsw");
        assert_eq!(db.vector_cache_stats().active_bytes, 0);
        assert!(db.vector_cache_stats().peak_bytes <= budget);
    }
}

#[test]
fn oversized_walk_falls_back_and_disabled_retention_still_allows_small_ann() {
    let (db, _, _, _) = setup(4096);
    db.set_vector_cache_budget(0);
    let col = db.collection("docs").unwrap();
    let small = col
        .search_vectors("v", &QUERY, 5, None, &opts(VectorSearchMode::Ann))
        .unwrap();
    assert_eq!(small.execution.path, "hnsw");
    let mut options = opts(VectorSearchMode::Ann);
    options.ef_search = Some(4096);
    let fallback = col.search_vectors("v", &QUERY, 5, None, &options).unwrap();
    assert_eq!(fallback.execution.reason, "memoryBudget");
    options.mode = VectorSearchMode::Exact;
    let exact = col.search_vectors("v", &QUERY, 5, None, &options).unwrap();
    assert_eq!(signature(&fallback), signature(&exact));
    let stats = db.vector_cache_stats();
    assert_eq!(stats.retained_bytes, 0);
    assert_eq!(stats.active_bytes, 0);
    assert!(stats.peak_bytes <= 64 * 1024);
}

#[test]
fn read_errors_and_budget_reduction_release_the_live_loan() {
    let (db, gate, entered, resume) = setup(64);
    db.set_vector_cache_budget(1024 * 1024);
    gate.armed.store(true, Ordering::SeqCst);
    gate.fail.store(true, Ordering::SeqCst);
    let col = db.collection("docs").unwrap();
    let running = std::thread::spawn(move || {
        col.search_vectors("v", &QUERY, 5, None, &opts(VectorSearchMode::Ann))
    });
    entered.recv_timeout(Duration::from_secs(10)).unwrap();
    db.set_vector_cache_budget(0);
    assert_eq!(db.vector_cache_stats().active_bytes, 1024 * 1024);
    assert_eq!(
        db.collection("docs")
            .unwrap()
            .search_vectors("v", &QUERY, 5, None, &opts(VectorSearchMode::Ann))
            .unwrap()
            .execution
            .reason,
        "memoryBudget"
    );
    resume.send(()).unwrap();
    assert!(matches!(
        running.join().unwrap(),
        Err(TalaDbError::Storage(_))
    ));
    assert_eq!(db.vector_cache_stats().active_bytes, 0);
    assert_eq!(db.vector_cache_stats().retained_bytes, 0);
}
