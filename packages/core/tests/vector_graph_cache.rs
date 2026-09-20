//! Correctness of the retained decoded-node cache on the HNSW search path.
//!
//! The node cache lives across queries, so a missed invalidation outlives the
//! query that caused it. These tests pin the observable contract for every path
//! that rewrites a graph: data writes (which move the vector revision), index
//! create and drop, and rebuild.
//!
//! What the cache can and cannot break is worth being precise about, because it
//! sets how much these tests can prove. The ANN path rescores every id the graph
//! returns against the vector table in the current snapshot, so the graph only
//! decides *which* candidates are considered — scores are always current and a
//! deleted document cannot surface through a stale node. A stale cache is
//! therefore a recall risk, not a correctness one, and attempts to make a stale
//! cache return a wrong answer did not succeed: HNSW still reaches good
//! neighbours through slightly wrong links.
//!
//! So these assertions pin the reachable contract — agreement with exact search,
//! no resurrected documents, no cross-collection bleed — rather than proving the
//! eviction in `rebuild_vector_index` is load-bearing. That eviction is
//! deliberate defence; removing it does not fail these tests.

use taladb_core::{
    Database, HnswOptions, Value, VectorMetric, VectorQueryOptions, VectorSearchMode,
};

fn vec_val(v: &[f32]) -> Value {
    Value::Array(v.iter().map(|f| Value::Float(f64::from(*f))).collect())
}

fn insert_vec(col: &taladb_core::Collection, label: &str, v: &[f32]) {
    col.insert(vec![
        ("label".into(), Value::Str(label.into())),
        ("emb".into(), vec_val(v)),
    ])
    .unwrap();
}

/// A spread of 8-D unit-ish vectors, enough nodes for the graph to have links
/// worth caching.
fn seed(col: &taladb_core::Collection, n: usize, tag: &str) {
    for i in 0..n {
        let mut v = [0.0f32; 8];
        v[i % 8] = 1.0;
        v[(i + 3) % 8] = (i as f32) * 0.01;
        insert_vec(col, &format!("{tag}-{i}"), &v);
    }
}

fn make_index(col: &taladb_core::Collection) {
    col.create_vector_index(
        "emb",
        8,
        Some(VectorMetric::Cosine),
        Some(HnswOptions {
            m: 8,
            ef_construction: 64,
        }),
    )
    .unwrap();
}

fn labels(r: taladb_core::VectorQueryResult) -> Vec<String> {
    r.hits
        .into_iter()
        .map(|h| match h.document.get("label") {
            Some(Value::Str(s)) => s.clone(),
            _ => String::new(),
        })
        .collect()
}

fn ann(col: &taladb_core::Collection, q: &[f32]) -> Vec<String> {
    let opts = VectorQueryOptions {
        mode: VectorSearchMode::Ann,
        ef_search: Some(200),
        ..Default::default()
    };
    labels(col.search_vectors("emb", q, 5, None, &opts).unwrap())
}

fn exact(col: &taladb_core::Collection, q: &[f32]) -> Vec<String> {
    let opts = VectorQueryOptions {
        mode: VectorSearchMode::Exact,
        ..Default::default()
    };
    labels(col.search_vectors("emb", q, 5, None, &opts).unwrap())
}

const Q: [f32; 8] = [1.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0];

#[test]
fn repeated_ann_queries_are_consistent() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    seed(&col, 40, "a");
    make_index(&col);

    let first = ann(&col, &Q);
    let second = ann(&col, &Q);
    let third = ann(&col, &Q);
    assert_eq!(first, second, "second query diverged from the first");
    assert_eq!(second, third, "third query diverged");
    assert!(!first.is_empty());
}

#[test]
fn a_rebuild_does_not_serve_stale_graph_nodes() {
    // The case with no revision change to hide behind: same data, so the vector
    // table revision can be identical across the rebuild, while the rebuild
    // itself reassigns node ids.
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    seed(&col, 40, "a");
    make_index(&col);

    let before = ann(&col, &Q);
    assert!(!before.is_empty());

    col.upgrade_vector_index("emb").unwrap();

    let after = ann(&col, &Q);
    assert_eq!(
        after,
        exact(&col, &Q),
        "ANN disagreed with exact after a rebuild — stale cached nodes"
    );
    assert_eq!(
        before, after,
        "rebuild changed the answer on unchanged data"
    );
}

#[test]
fn deletes_then_rebuild_are_visible_through_the_cache() {
    // Deleting leaves tombstones; the rebuild compacts them and renumbers nodes.
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    seed(&col, 40, "a");
    make_index(&col);

    let before = ann(&col, &Q);
    assert!(!before.is_empty());

    let victim = before[0].clone();
    col.delete_one(taladb_core::query::Filter::Eq(
        "label".into(),
        Value::Str(victim.clone()),
    ))
    .unwrap();
    col.upgrade_vector_index("emb").unwrap();

    let after = ann(&col, &Q);
    assert!(
        !after.contains(&victim),
        "deleted document {victim} came back from the node cache: {after:?}"
    );
    assert_eq!(after, exact(&col, &Q), "ANN disagreed with exact");
}

#[test]
fn dropping_and_recreating_the_index_does_not_serve_stale_graph_nodes() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    seed(&col, 40, "old");
    make_index(&col);

    let before = ann(&col, &Q);
    assert!(before.iter().all(|l| l.starts_with("old")));

    col.drop_vector_index("emb").unwrap();
    for d in col.find(taladb_core::query::Filter::All).unwrap() {
        col.delete_one(taladb_core::query::Filter::Eq(
            "_id".into(),
            Value::Str(d.id.to_string()),
        ))
        .unwrap();
    }
    seed(&col, 40, "new");
    make_index(&col);

    let after = ann(&col, &Q);
    assert!(
        after.iter().all(|l| l.starts_with("new")),
        "stale 'old' documents survived a drop/recreate: {after:?}"
    );
    assert_eq!(after, exact(&col, &Q));
}

#[test]
fn writes_after_a_cached_query_are_not_answered_from_the_stale_graph() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("docs").unwrap();
    seed(&col, 40, "a");
    make_index(&col);

    let _warm = ann(&col, &Q);

    // Moves the vector revision, so the graph is stale and Auto must not use it.
    insert_vec(&col, "needle", &Q);

    let opts = VectorQueryOptions::default(); // Auto
    let auto = labels(col.search_vectors("emb", &Q, 5, None, &opts).unwrap());
    assert_eq!(
        auto[0], "needle",
        "a stale graph answered instead of falling back to exact: {auto:?}"
    );
}

#[test]
fn two_handles_from_one_database_share_invalidation() {
    let db = Database::open_in_memory().unwrap();
    let a = db.collection("docs").unwrap();
    seed(&a, 40, "a");
    make_index(&a);
    let _warm = ann(&a, &Q);

    // Rebuild through a second handle; the first must not keep its nodes.
    let b = db.collection("docs").unwrap();
    b.upgrade_vector_index("emb").unwrap();

    assert_eq!(
        ann(&a, &Q),
        exact(&a, &Q),
        "handle A served nodes invalidated by a rebuild through handle B"
    );
}

#[test]
fn other_collections_are_not_invalidated_or_confused() {
    let db = Database::open_in_memory().unwrap();
    let docs = db.collection("docs").unwrap();
    let other = db.collection("other").unwrap();
    seed(&docs, 40, "d");
    seed(&other, 40, "o");
    make_index(&docs);
    make_index(&other);

    let docs_before = ann(&docs, &Q);
    let other_before = ann(&other, &Q);

    other.upgrade_vector_index("emb").unwrap();

    assert_eq!(
        ann(&docs, &Q),
        docs_before,
        "docs disturbed by other's rebuild"
    );
    assert!(ann(&docs, &Q).iter().all(|l| l.starts_with("d-")));
    assert_eq!(ann(&other, &Q), exact(&other, &Q));
    assert!(other_before.iter().all(|l| l.starts_with("o-")));
}

