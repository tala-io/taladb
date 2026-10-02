//! The serde-typed API (`Database::typed`) and the JSON query language it uses.

use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::json;
use taladb::{Database, TalaDbError};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Note {
    #[serde(rename = "_id", default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    title: String,
    done: bool,
    stars: i64,
    #[serde(default)]
    tags: Vec<String>,
}

fn note(title: &str, done: bool, stars: i64) -> Note {
    Note {
        id: None,
        title: title.into(),
        done,
        stars,
        tags: vec![],
    }
}

#[test]
fn typed_documents_round_trip_with_their_id() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    let id = notes.insert(&note("café 日本語 🎉", false, 3)).unwrap();

    let found = notes
        .find_one(json!({ "title": "café 日本語 🎉" }))
        .unwrap()
        .unwrap();
    assert_eq!(found.id.as_deref(), Some(id.to_string().as_str()));
    assert_eq!(found.stars, 3);
    assert_eq!(
        notes.find_by_id(id).unwrap().unwrap().title,
        "café 日本語 🎉"
    );
}

/// A plain `Option` id without `skip_serializing_if` serializes `None` as
/// `null`; the engine would reject that as an id, so the typed layer drops it.
#[test]
fn a_none_id_serialized_as_null_still_gets_an_assigned_id() {
    #[derive(Serialize, Deserialize)]
    struct Plain {
        #[serde(rename = "_id")]
        id: Option<String>,
        name: String,
    }
    let db = Database::open_in_memory().unwrap();
    let items = db.typed::<Plain>("items").unwrap();
    items
        .insert(&Plain {
            id: None,
            name: "a".into(),
        })
        .unwrap();
    assert!(items.find_one(json!({})).unwrap().unwrap().id.is_some());
}

#[test]
fn json_filters_and_updates_report_what_they_touched() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    notes
        .insert_many(
            &(1..=5)
                .map(|i| note(&format!("n{i}"), false, i))
                .collect::<Vec<_>>(),
        )
        .unwrap();

    assert_eq!(notes.count(json!({ "stars": { "$gte": 3 } })).unwrap(), 3);
    assert_eq!(
        notes
            .count(json!({ "$or": [{ "stars": 1 }, { "stars": 5 }] }))
            .unwrap(),
        2
    );

    assert!(
        notes
            .update_one(
                json!({ "title": "n1" }),
                json!({ "$set": { "done": true } })
            )
            .unwrap()
    );
    assert!(
        !notes
            .update_one(
                json!({ "title": "nope" }),
                json!({ "$set": { "done": true } })
            )
            .unwrap()
    );
    assert_eq!(
        notes
            .update_many(
                json!({ "stars": { "$lt": 3 } }),
                json!({ "$inc": { "stars": 10 }, "$push": { "tags": "bumped" } })
            )
            .unwrap(),
        2
    );
    let n1 = notes.find_one(json!({ "title": "n1" })).unwrap().unwrap();
    assert_eq!((n1.stars, n1.tags), (11, vec!["bumped".to_string()]));

    assert!(notes.delete_one(json!({ "title": "n5" })).unwrap());
    assert_eq!(notes.delete_many(json!({})).unwrap(), 4);
}

/// A malformed filter must fail, never degrade to match-all — that would turn
/// a typo in `delete_many` into deleting the whole collection.
#[test]
fn malformed_filters_and_updates_are_errors_not_match_all() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    notes.insert(&note("keep me", false, 1)).unwrap();

    let err = notes
        .delete_many(json!({ "stars": { "$gtee": 1 } }))
        .unwrap_err();
    assert!(matches!(err, TalaDbError::InvalidFilter(_)), "{err}");
    assert!(notes.find(json!({ "stars": {} })).is_err());
    assert!(
        notes
            .update_one(json!({}), json!({ "$sett": { "done": true } }))
            .is_err()
    );
    assert!(notes.update_one(json!({}), json!({})).is_err());
    assert_eq!(notes.count(json!({})).unwrap(), 1, "nothing was deleted");
}

#[test]
fn insert_many_writes_nothing_when_any_document_is_rejected() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    let id = notes.insert(&note("existing", false, 1)).unwrap();
    let duplicate = Note {
        id: Some(id.to_string()),
        ..note("dup", false, 1)
    };
    assert!(
        notes
            .insert_many(&[note("new", false, 1), duplicate])
            .is_err()
    );
    assert_eq!(notes.count(json!({})).unwrap(), 1);
}

#[test]
fn typed_and_raw_views_share_documents_and_indexes() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    notes.create_index("done").unwrap();
    notes.insert(&note("a", false, 1)).unwrap();
    assert_eq!(
        db.collection("notes")
            .unwrap()
            .count(taladb::Filter::All)
            .unwrap(),
        1
    );
    assert_eq!(
        notes.raw().list_indexes().unwrap().btree,
        vec!["done".to_string()]
    );
}

#[test]
fn vector_and_text_search_return_typed_hits() {
    #[derive(Serialize, Deserialize)]
    struct Doc {
        title: String,
        kind: String,
        embedding: Vec<f32>,
    }
    let db = Database::open_in_memory().unwrap();
    let docs = db.typed::<Doc>("docs").unwrap();
    docs.create_vector_index("embedding", 2, None, None)
        .unwrap();
    docs.create_fts_index("title").unwrap();
    for (title, kind, e) in [
        ("rust engine", "a", [1.0, 0.0]),
        ("swift wrapper", "b", [0.0, 1.0]),
        ("rust ffi", "b", [0.9, 0.1]),
    ] {
        docs.insert(&Doc {
            title: title.into(),
            kind: kind.into(),
            embedding: e.to_vec(),
        })
        .unwrap();
    }

    let near = docs
        .find_nearest("embedding", &[1.0, 0.0], 2, None)
        .unwrap();
    assert_eq!(
        near.iter()
            .map(|h| h.document.title.as_str())
            .collect::<Vec<_>>(),
        ["rust engine", "rust ffi"]
    );
    let only_b = docs
        .find_nearest("embedding", &[1.0, 0.0], 3, Some(json!({ "kind": "b" })))
        .unwrap();
    assert_eq!(only_b[0].document.title, "rust ffi");

    let text = docs.search_text("title", "rust", 5).unwrap();
    assert_eq!(text.len(), 2);
}

/// `Database` clones share one open database and work across threads.
#[test]
fn a_watch_on_one_thread_sees_writes_from_another() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    let watch = notes.watch(json!({ "done": false })).unwrap();
    assert!(
        watch
            .next_timeout(Duration::from_millis(10))
            .unwrap()
            .is_none()
    );

    let writer = db.clone();
    std::thread::spawn(move || {
        let notes = writer.typed::<Note>("notes").unwrap();
        notes
            .insert(&note("from another thread", false, 1))
            .unwrap();
        notes.insert(&note("finished", true, 1)).unwrap();
    })
    .join()
    .unwrap();

    let open = watch
        .next_timeout(Duration::from_secs(5))
        .unwrap()
        .expect("the writes wake the watch");
    assert_eq!(
        open.iter().map(|n| n.title.as_str()).collect::<Vec<_>>(),
        ["from another thread"]
    );
    // The original handle sees what the clone wrote: one database, not a copy.
    assert_eq!(notes.count(json!({})).unwrap(), 2);
    assert_eq!(
        db.collection("notes")
            .unwrap()
            .count(taladb::Filter::All)
            .unwrap(),
        2
    );
}

#[test]
fn open_shorthand_creates_and_reopens_a_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("app.db");
    taladb::open(&path)
        .unwrap()
        .typed::<Note>("notes")
        .unwrap()
        .insert(&note("persisted", false, 1))
        .unwrap();
    let reopened = taladb::open(&path).unwrap();
    assert_eq!(
        reopened
            .typed::<Note>("notes")
            .unwrap()
            .count(json!({}))
            .unwrap(),
        1
    );
}

/// The guide's pattern: subscribe, read the current state, then hand the
/// watch to a thread of its own. Compiles only while `TypedWatch` is `Send`.
#[test]
fn a_watch_can_run_on_its_own_thread() {
    let db = Database::open_in_memory().unwrap();
    let notes = db.typed::<Note>("notes").unwrap();
    let watch = notes.watch(json!({ "done": false })).unwrap();
    assert!(notes.find(json!({ "done": false })).unwrap().is_empty());

    let listener = std::thread::spawn(move || watch.next_timeout(Duration::from_secs(5)));
    notes
        .insert(&note("seen by the listener", false, 1))
        .unwrap();
    let seen = listener
        .join()
        .unwrap()
        .unwrap()
        .expect("woken by the insert");
    assert_eq!(seen[0].title, "seen by the listener");
}
