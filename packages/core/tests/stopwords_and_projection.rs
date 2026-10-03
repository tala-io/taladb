//! Query-time stopwords in ranked full-text search, and projection that can
//! exclude fields — on `find_with_options` and on live queries.
//!
//! Both came from building a native app on TalaDB: a search for "kind to a
//! classmate" listed every note containing "to", and a live query over notes
//! that carry 384-float embeddings sent every vector across on every write.

use taladb::Database;
use taladb::bm25::Bm25Params;
use taladb::document::{Document, Value};
use taladb::fts::HybridQuery;
use taladb::{Filter, FindOptions};

fn notes(rows: &[&str]) -> Database {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("notes").unwrap();
    for text in rows {
        col.insert(vec![("text".into(), Value::Str((*text).into()))])
            .unwrap();
    }
    col.create_fts_index("text").unwrap();
    db
}

fn text(doc: &Document) -> String {
    match doc.get("text") {
        Some(Value::Str(s)) => s.clone(),
        _ => String::new(),
    }
}

const ROWS: &[&str] = &[
    "Invited a new classmate to join her group",
    "Counted by fives to 100",
    "Reminded everyone to put goggles on",
];

#[test]
fn stopwords_do_not_match_every_document_by_default() {
    let db = notes(ROWS);
    let hits = db
        .collection("notes")
        .unwrap()
        .search_text("text", "kind to a classmate", 10)
        .unwrap();
    let texts: Vec<String> = hits.iter().map(|h| text(&h.document)).collect();
    assert_eq!(texts, vec!["Invited a new classmate to join her group"]);
}

#[test]
fn stopword_filtering_can_be_turned_off() {
    let db = notes(ROWS);
    let params = Bm25Params {
        stopwords: false,
        ..Bm25Params::default()
    };
    let hits = db
        .collection("notes")
        .unwrap()
        .search_text_with("text", "kind to a classmate", 10, &params, None)
        .unwrap();
    assert_eq!(hits.len(), 3, "every note contains \"to\"");
    // The note that also has "classmate" still ranks first.
    assert_eq!(
        text(&hits[0].document),
        "Invited a new classmate to join her group"
    );
}

#[test]
fn a_query_of_only_stopwords_is_searched_as_typed() {
    let db = notes(ROWS);
    let hits = db
        .collection("notes")
        .unwrap()
        .search_text("text", "to", 10)
        .unwrap();
    assert_eq!(hits.len(), 3);
}

#[test]
fn hybrid_search_filters_stopwords_on_its_text_side() {
    let db = Database::open_in_memory().unwrap();
    let col = db.collection("notes").unwrap();
    for (t, v) in [
        ("Invited a new classmate to join", [1.0f32, 0.0]),
        ("Counted by fives to 100", [0.0, 1.0]),
    ] {
        col.insert(vec![
            ("text".into(), Value::Str(t.into())),
            (
                "embedding".into(),
                Value::Array(v.iter().map(|x| Value::Float(f64::from(*x))).collect()),
            ),
        ])
        .unwrap();
    }
    col.create_fts_index("text").unwrap();
    col.create_vector_index("embedding", 2, None, None).unwrap();

    let hits = col
        .hybrid_search(HybridQuery::new(
            "text",
            "to a classmate",
            "embedding",
            &[0.0, 1.0],
            2,
        ))
        .unwrap();
    let counted = hits
        .iter()
        .find(|h| text(&h.document).starts_with("Counted"))
        .unwrap();
    assert_eq!(
        counted.text_rank, None,
        "\"to\" alone no longer makes a text match"
    );
    let invited = hits
        .iter()
        .find(|h| text(&h.document).starts_with("Invited"))
        .unwrap();
    assert_eq!(invited.text_rank, Some(0));
}

fn doc_with_embedding(db: &Database) -> taladb::Collection {
    let col = db.collection("notes").unwrap();
    col.insert(vec![
        ("text".into(), Value::Str("Read aloud".into())),
        (
            "tags".into(),
            Value::Array(vec![Value::Str("Literacy".into())]),
        ),
        (
            "embedding".into(),
            Value::Array(vec![Value::Float(0.1); 384]),
        ),
    ])
    .unwrap();
    col
}

#[test]
fn find_with_options_can_exclude_fields() {
    let db = Database::open_in_memory().unwrap();
    let col = doc_with_embedding(&db);
    let options = FindOptions {
        exclude: Some(vec!["embedding".into()]),
        ..FindOptions::default()
    };
    let docs = col.find_with_options(Filter::All, options).unwrap();
    assert_eq!(docs.len(), 1);
    assert!(docs[0].get("embedding").is_none());
    assert_eq!(text(&docs[0]), "Read aloud");
    assert!(docs[0].get("tags").is_some());
}

#[test]
fn fields_and_exclude_combine() {
    let db = Database::open_in_memory().unwrap();
    let col = doc_with_embedding(&db);
    let options = FindOptions {
        fields: Some(vec!["text".into(), "embedding".into()]),
        exclude: Some(vec!["embedding".into()]),
        ..FindOptions::default()
    };
    let doc = &col.find_with_options(Filter::All, options).unwrap()[0];
    assert_eq!(
        doc.fields
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>(),
        vec!["text"]
    );
}

#[test]
fn excluding_id_is_not_possible() {
    let db = Database::open_in_memory().unwrap();
    let col = doc_with_embedding(&db);
    let id = col.find(Filter::All).unwrap()[0].id;
    let options = FindOptions {
        exclude: Some(vec!["_id".into()]),
        ..FindOptions::default()
    };
    assert_eq!(
        col.find_with_options(Filter::All, options).unwrap()[0].id,
        id
    );
}

#[test]
fn live_queries_can_exclude_fields_from_every_snapshot() {
    let db = Database::open_in_memory().unwrap();
    let col = doc_with_embedding(&db);
    let watch = col.watch_with_options(
        Filter::All,
        FindOptions {
            exclude: Some(vec!["embedding".into()]),
            ..FindOptions::default()
        },
    );
    // `next` waits for a write, then returns the projected snapshot.
    col.insert(vec![
        ("text".into(), Value::Str("Counted to twenty".into())),
        (
            "embedding".into(),
            Value::Array(vec![Value::Float(0.2); 384]),
        ),
    ])
    .unwrap();
    let after = watch.next().unwrap();
    assert_eq!(after.len(), 2);
    assert!(after.iter().all(|d| d.get("embedding").is_none()));
    assert!(after.iter().all(|d| d.get("text").is_some()));
}
