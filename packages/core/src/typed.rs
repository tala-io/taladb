//! Serde-typed collections: store and load your own structs, query with JSON.
//!
//! ```
//! use serde::{Deserialize, Serialize};
//! use serde_json::json;
//!
//! #[derive(Serialize, Deserialize)]
//! struct Note {
//!     #[serde(rename = "_id", default, skip_serializing_if = "Option::is_none")]
//!     id: Option<String>,
//!     title: String,
//!     done: bool,
//! }
//!
//! # fn main() -> Result<(), taladb::TalaDbError> {
//! let db = taladb::Database::open_in_memory()?;
//! let notes = db.typed::<Note>("notes")?;
//!
//! notes.insert(&Note { id: None, title: "Groceries".into(), done: false })?;
//! let open: Vec<Note> = notes.find(json!({ "done": false }))?;
//! assert_eq!(open[0].title, "Groceries");
//!
//! notes.update_one(json!({ "title": "Groceries" }), json!({ "$set": { "done": true } }))?;
//! # Ok(())
//! # }
//! ```
//!
//! Filters and updates use the same JSON operators as every TalaDB binding;
//! see [`crate::json`]. Every stored document has an `_id`: declare it as an
//! optional field renamed to `_id`, as above, to read it back. Leave it `None`
//! on insert and the engine assigns a ULID.

use std::marker::PhantomData;
use std::time::Duration;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value as Json;
use ulid::Ulid;

use crate::collection::Collection;
use crate::document::Document;
use crate::error::TalaDbError;
use crate::json::{document_to_json, fields_from_json, filter_from_json, update_from_json};
use crate::vector::VectorMetric;
use crate::watch::WatchHandle;
use crate::{Database, HnswOptions};

impl Database {
    /// The collection `name`, storing and returning `T`.
    ///
    /// Collections are created on first write. The typed view and the raw
    /// [`Collection`] from [`Database::collection`] see the same documents.
    pub fn typed<T: Serialize + DeserializeOwned>(
        &self,
        name: &str,
    ) -> Result<TypedCollection<T>, TalaDbError> {
        Ok(TypedCollection {
            inner: self.collection(name)?,
            _marker: PhantomData,
        })
    }
}

/// A collection whose documents are serde values of type `T`.
pub struct TypedCollection<T> {
    inner: Collection,
    _marker: PhantomData<fn() -> T>,
}

/// A document with its similarity or relevance score. Higher is closer.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored<T> {
    pub document: T,
    pub score: f32,
}

impl<T: Serialize + DeserializeOwned> TypedCollection<T> {
    /// The underlying collection, for everything the typed view does not
    /// cover: aggregation, HNSW tuning, snapshots, audit logs.
    pub fn raw(&self) -> &Collection {
        &self.inner
    }

    // -- Documents ------------------------------------------------------------

    /// Insert `document` and return its id.
    pub fn insert(&self, document: &T) -> Result<Ulid, TalaDbError> {
        self.inner.insert(encode(document)?)
    }

    /// Insert `documents` in one transaction and return their ids in order.
    /// All or nothing: if any is rejected, none are written.
    pub fn insert_many(&self, documents: &[T]) -> Result<Vec<Ulid>, TalaDbError> {
        self.inner
            .insert_many(documents.iter().map(encode).collect::<Result<_, _>>()?)
    }

    /// Every document matching `filter`, e.g. `json!({ "done": false })`.
    pub fn find(&self, filter: Json) -> Result<Vec<T>, TalaDbError> {
        self.inner
            .find(filter_from_json(&filter)?)?
            .iter()
            .map(decode)
            .collect()
    }

    /// The first document matching `filter`, or `None`.
    pub fn find_one(&self, filter: Json) -> Result<Option<T>, TalaDbError> {
        self.inner
            .find_one(filter_from_json(&filter)?)?
            .as_ref()
            .map(decode)
            .transpose()
    }

    /// The document with this id, or `None`.
    pub fn find_by_id(&self, id: Ulid) -> Result<Option<T>, TalaDbError> {
        self.inner.find_by_id(id)?.as_ref().map(decode).transpose()
    }

    /// How many documents match `filter`.
    pub fn count(&self, filter: Json) -> Result<u64, TalaDbError> {
        self.inner.count(filter_from_json(&filter)?)
    }

    /// Apply `update`, e.g. `json!({ "$set": { "done": true } })`, to the first
    /// document matching `filter`. Returns whether one matched.
    pub fn update_one(&self, filter: Json, update: Json) -> Result<bool, TalaDbError> {
        self.inner
            .update_one(filter_from_json(&filter)?, update_from_json(&update)?)
    }

    /// Apply `update` to every document matching `filter`. Returns how many changed.
    pub fn update_many(&self, filter: Json, update: Json) -> Result<u64, TalaDbError> {
        self.inner
            .update_many(filter_from_json(&filter)?, update_from_json(&update)?)
    }

    /// Delete the first document matching `filter`. Returns whether one matched.
    pub fn delete_one(&self, filter: Json) -> Result<bool, TalaDbError> {
        self.inner.delete_one(filter_from_json(&filter)?)
    }

    /// Delete every document matching `filter`; `json!({})` empties the
    /// collection. Returns how many were deleted.
    pub fn delete_many(&self, filter: Json) -> Result<u64, TalaDbError> {
        self.inner.delete_many(filter_from_json(&filter)?)
    }

    // -- Indexes --------------------------------------------------------------

    /// Index `field` for equality and range filters. A no-op if it exists.
    pub fn create_index(&self, field: &str) -> Result<(), TalaDbError> {
        self.inner.create_index(field)
    }

    /// Index `fields` together, in order.
    pub fn create_compound_index(&self, fields: &[&str]) -> Result<(), TalaDbError> {
        self.inner.create_compound_index(fields)
    }

    /// Index `field` for [`Self::search_text`] and `$contains`. A no-op if it exists.
    pub fn create_fts_index(&self, field: &str) -> Result<(), TalaDbError> {
        self.inner.create_fts_index(field)
    }

    /// Index the float-array `field` for [`Self::find_nearest`]. `None` metric
    /// is cosine; `None` HNSW options means exact search, which is faster and
    /// always exact below tens of thousands of vectors.
    pub fn create_vector_index(
        &self,
        field: &str,
        dimensions: usize,
        metric: Option<VectorMetric>,
        hnsw: Option<HnswOptions>,
    ) -> Result<(), TalaDbError> {
        self.inner
            .create_vector_index(field, dimensions, metric, hnsw)
    }

    // -- Search ---------------------------------------------------------------

    /// The `top_k` documents whose `field` is most similar to `query`, best
    /// first. `filter` narrows the candidates before ranking.
    pub fn find_nearest(
        &self,
        field: &str,
        query: &[f32],
        top_k: usize,
        filter: Option<Json>,
    ) -> Result<Vec<Scored<T>>, TalaDbError> {
        let filter = filter.map(|f| filter_from_json(&f)).transpose()?;
        self.inner
            .find_nearest(field, query, top_k, filter)?
            .iter()
            .map(|hit| {
                Ok(Scored {
                    document: decode(&hit.document)?,
                    score: hit.score,
                })
            })
            .collect()
    }

    /// The `top_k` documents whose full-text-indexed `field` best matches
    /// `query`, ranked by BM25. Requires [`Self::create_fts_index`].
    pub fn search_text(
        &self,
        field: &str,
        query: &str,
        top_k: usize,
    ) -> Result<Vec<Scored<T>>, TalaDbError> {
        self.inner
            .search_text(field, query, top_k)?
            .iter()
            .map(|hit| {
                Ok(Scored {
                    document: decode(&hit.document)?,
                    score: hit.score,
                })
            })
            .collect()
    }

    // -- Live queries ---------------------------------------------------------

    /// A live query over the documents matching `filter`. Writes through any
    /// handle on this database wake it; see [`TypedWatch`].
    pub fn watch(&self, filter: Json) -> Result<TypedWatch<T>, TalaDbError> {
        Ok(TypedWatch {
            inner: self.inner.watch(filter_from_json(&filter)?),
            _marker: PhantomData,
        })
    }
}

/// A live query: the matching documents after each write.
///
/// It delivers no initial snapshot — read the current state with
/// [`TypedCollection::find`] *after* creating the watch, so no write can fall
/// between the two. Rapid writes coalesce into one snapshot of the latest
/// state; none is skipped. [`Self::next`] blocks, so run it on its own thread
/// (or `spawn_blocking` in async code).
pub struct TypedWatch<T> {
    inner: WatchHandle,
    _marker: PhantomData<fn() -> T>,
}

impl<T: DeserializeOwned> TypedWatch<T> {
    /// Block until the next write, then return the matching documents.
    pub fn next(&self) -> Result<Vec<T>, TalaDbError> {
        self.inner.next()?.iter().map(decode).collect()
    }

    /// Wait at most `timeout` for a write; `None` if none occurred.
    pub fn next_timeout(&self, timeout: Duration) -> Result<Option<Vec<T>>, TalaDbError> {
        self.inner
            .next_timeout(timeout)?
            .map(|docs| docs.iter().map(decode).collect())
            .transpose()
    }

    /// The new snapshot if a write occurred since the last call, without waiting.
    pub fn try_next(&self) -> Result<Option<Vec<T>>, TalaDbError> {
        self.inner
            .try_next()?
            .map(|docs| docs.iter().map(decode).collect())
            .transpose()
    }
}

fn encode<T: Serialize>(document: &T) -> Result<Vec<(String, crate::Value)>, TalaDbError> {
    let mut json =
        serde_json::to_value(document).map_err(|e| TalaDbError::Serialization(e.to_string()))?;
    // A struct whose `_id` is a plain `Option` serializes `None` as `null`,
    // which the engine rejects as an id; dropping it lets the engine assign one.
    if let Some(obj) = json.as_object_mut()
        && obj.get("_id").is_some_and(Json::is_null)
    {
        obj.remove("_id");
    }
    fields_from_json(&json)
}

fn decode<T: DeserializeOwned>(doc: &Document) -> Result<T, TalaDbError> {
    serde_json::from_value(document_to_json(doc))
        .map_err(|e| TalaDbError::Serialization(e.to_string()))
}
