---
title: Rust Guide
description: Use TalaDB as the local database of a Rust application — serde documents, JSON filters, indexes, vector, full-text and hybrid search, live queries and encryption, in one embedded file.
---

# Rust

Use TalaDB as the local database of a Rust application: a desktop app, a CLI,
a service, an edge worker. It is the engine every other TalaDB platform runs,
used directly — documents, filters, indexes,
[vector search](/api/vector-search), [full-text search](/api/search), live
queries and encryption at rest, in one file, inside your process.

::: warning Early release
The `taladb` crate is new, and its API may still change between minor
versions before 1.0. It is published to [crates.io](https://crates.io/crates/taladb)
with each TalaDB release, starting with the next one; until then, use the git
dependency below. The full API reference is on [docs.rs](https://docs.rs/taladb).
:::

## Installation

```sh
cargo add taladb serde --features serde/derive
cargo add serde_json
```

Until a release reaches crates.io, depend on the repository instead:

```toml
taladb = { git = "https://github.com/tala-io/taladb" }
```

Requires Rust 1.90 or newer.

## Quick start

```rust
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

    notes.create_index("done")?;                   // idempotent: fine at every start
    notes.insert(&Note { id: None, title: "Buy groceries".into(), done: false })?;

    let open: Vec<Note> = notes.find(json!({ "done": false }))?;
    notes.update_one(json!({ "title": "Buy groceries" }), json!({ "$set": { "done": true } }))?;
    notes.delete_many(json!({ "done": true }))?;
    Ok(())
}
```

- **Your types** only need `serde`. Every stored document has an `_id`;
  declare it as an optional field renamed to `_id`, as above, to read it back.
  Leave it `None` on insert and the engine assigns a ULID.
- **Filters and updates** are JSON, in the same language as every TalaDB
  platform — see [Filters](/api/filters) and [Updates](/api/updates). A
  malformed filter is an error, never a silent match-all.
- **Errors** are one type, `TalaDbError`.

## Threads and async

`Database` is cheap to clone, and every clone shares the same open database —
hand one to each thread. Operations are synchronous and fast; in async code,
run them with `tokio::task::spawn_blocking` (or your runtime's equivalent) so
they do not stall the executor.

## Vector, full-text and hybrid search

```rust
#[derive(Serialize, Deserialize)]
struct Doc { title: String, embedding: Vec<f32> }

let docs = db.typed::<Doc>("docs")?;
docs.create_vector_index("embedding", 384, None, None)?;  // exact search by default
docs.create_fts_index("title")?;

let similar = docs.find_nearest("embedding", &query_embedding, 5, None)?;
let matches = docs.search_text("title", "groceries", 5)?;
```

Exact vector search is faster and always exact below tens of thousands of
vectors; pass `HnswOptions` for a persistent approximate graph beyond that.
Hybrid search, aggregation, sorting, pagination and HNSW tuning are on the
untyped collection — `docs.raw()` — documented on [docs.rs](https://docs.rs/taladb).

## Live queries

```rust
let watch = notes.watch(json!({ "done": false }))?;
let current = notes.find(json!({ "done": false }))?;   // read *after* subscribing

std::thread::spawn(move || {
    while let Ok(open) = watch.next() {                  // blocks until a write changes it
        render(&open);
    }
});
```

Writes through any clone of the database wake the watch. Rapid writes coalesce
into one snapshot of the latest state, and none is skipped. `next_timeout`
waits with a deadline, so a loop can stop cleanly. See
[Live Queries](/api/live-queries).

## Encryption

```rust
let db = taladb::Database::open_encrypted(std::path::Path::new("secret.db"), &passphrase)?;
```

Enable the `encryption` feature: `cargo add taladb --features encryption`. See
[Encryption](/api/encryption).

## Migrations

Storage-format upgrades run automatically at open. For your own schema steps,
`db.user_version()` and `db.set_user_version(n)` record which have run — the
same counter the other platforms' [migration runners](/api/migrations) use.

## Features

| Feature | Default | What it adds |
|---|:---:|---|
| `config-yaml` | ✅ | Reads `taladb.config.yml`. |
| `legacy-migration` | ✅ | Opens files written by TalaDB before 0.11. Turn it off to drop a second copy of redb from the build. |
| `encryption` | | AES-GCM encryption at rest. |
