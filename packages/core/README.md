# taladb

**An embedded document and vector database for Rust apps — local-first, in one file.**

Store documents, query them with MongoDB-style filters, index them, and run
vector, full-text and hybrid search, all inside your process. No server, no
network. The same engine powers TalaDB in the browser, Node.js, React Native,
Android and iOS, so data and queries behave identically everywhere.

- **Docs:** <https://taladb.dev/guide/rust> · **API:** <https://docs.rs/taladb>
- **Repository:** <https://github.com/tala-io/taladb>

> **Early release.** The Rust crate is new: its API may still change between
> minor versions before 1.0.

## Install

```sh
cargo add taladb serde --features serde/derive
cargo add serde_json
```

## Quick start

```rust,no_run
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
struct Note {
    #[serde(rename = "_id", default, skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    title: String,
    done: bool,
}

fn main() -> Result<(), taladb::TalaDbError> {
    let db = taladb::open("app.db")?;              // creates the file if needed
    let notes = db.typed::<Note>("notes")?;

    notes.create_index("done")?;
    notes.insert(&Note { id: None, title: "Buy groceries".into(), done: false })?;

    let open: Vec<Note> = notes.find(json!({ "done": false }))?;
    println!("{} open", open.len());

    notes.update_one(json!({ "title": "Buy groceries" }), json!({ "$set": { "done": true } }))?;
    Ok(())
}
```

Your types only need `serde`. Filters and updates are JSON — `$eq`, `$gt`,
`$in`, `$exists`, `$and`, `$or`, `$set`, `$inc`, `$push`, … — the same
language as every TalaDB platform ([reference](https://taladb.dev/api/filters)).
A malformed filter is an error, never a silent match-all.

## Vector and full-text search

```rust
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct Doc {
    title: String,
    embedding: Vec<f32>,
}

fn main() -> Result<(), taladb::TalaDbError> {
    let db = taladb::Database::open_in_memory()?;
    let docs = db.typed::<Doc>("docs")?;

    docs.create_vector_index("embedding", 3, None, None)?;   // exact search by default
    docs.create_fts_index("title")?;
    docs.insert(&Doc { title: "rust embedded database".into(), embedding: vec![1.0, 0.0, 0.0] })?;

    let similar = docs.find_nearest("embedding", &[1.0, 0.1, 0.0], 5, None)?;
    let matches = docs.search_text("title", "embedded", 5)?;
    println!("{} similar, {} text matches", similar.len(), matches.len());
    Ok(())
}
```

## Live queries

```rust
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize)]
struct Note {
    title: String,
    done: bool,
}

fn main() -> Result<(), taladb::TalaDbError> {
    let db = taladb::Database::open_in_memory()?;
    let notes = db.typed::<Note>("notes")?;
    let watch = notes.watch(json!({ "done": false }))?;

    let writer = db.clone();                    // clones share one open database
    std::thread::spawn(move || {
        writer.typed::<Note>("notes").unwrap()
            .insert(&Note { title: "new".into(), done: false }).unwrap();
    });

    let open: Vec<Note> = watch.next()?;        // blocks until a write changes the result
    println!("{} open", open.len());
    Ok(())
}
```

## Also included

- **Encryption at rest:** `Database::open_encrypted(path, passphrase)` (feature `encryption`).
- **Migrations:** `user_version` / `set_user_version` for your own schema steps;
  storage-format upgrades run automatically at open.
- **Sorting, pagination, projection, aggregation, HNSW, hybrid search:** on the
  untyped [`Collection`](https://docs.rs/taladb/latest/taladb/collection/struct.Collection.html),
  reachable from any typed collection with `.raw()`.

`Database` is cheap to clone and safe to share across threads. Operations are
synchronous; in async code, run them with `spawn_blocking`.

## Features

| Feature | Default | What it adds |
|---|:---:|---|
| `config-yaml` | ✅ | Reads `taladb.config.yml`. |
| `legacy-migration` | ✅ | Opens database files written by TalaDB before 0.11. Turn it off to drop a second copy of redb from the build. |
| `encryption` | | AES-GCM encryption at rest with PBKDF2 key derivation. |

## License

MIT OR Apache-2.0
