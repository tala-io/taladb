//! One retained-memory budget for exact vectors and decoded graph nodes.
//! Queries borrow entries without holding the mutex. Their working memory,
//! including concurrent loans, is not a process RSS limit.
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::vector::{CachedVectors, VectorBlock};
use crate::vector_graph::CachedGraph;

#[cfg(any(target_arch = "wasm32", target_os = "android", target_os = "ios"))]
pub(crate) const DEFAULT_SEARCH_CACHE_BYTES: usize = 8 * 1024 * 1024;
#[cfg(not(any(target_arch = "wasm32", target_os = "android", target_os = "ios")))]
pub(crate) const DEFAULT_SEARCH_CACHE_BYTES: usize = 64 * 1024 * 1024;

/// Estimated retained search-cache allocations. Active query/build scratch space and
/// storage-engine caches are separate from this budget.
#[derive(Debug, Clone, Copy)]
pub struct VectorCacheStats {
    pub budget_bytes: usize,
    pub retained_bytes: usize,
    pub vector_indexes: usize,
    pub graph_indexes: usize,
}

enum Value {
    Vectors(CachedVectors),
    Graph(CachedGraph),
}
struct Entry {
    value: Value,
    touched: u64,
}
impl Entry {
    fn bytes(&self) -> usize {
        match &self.value {
            Value::Vectors(v) => v.vectors.bytes(),
            Value::Graph(g) => g.cache.bytes(),
        }
    }
}

pub(crate) type SharedSearchCache = Arc<Mutex<SearchCache>>;
pub(crate) struct SearchCache {
    entries: HashMap<String, Entry>,
    budget: usize,
    clock: u64,
    /// Invalidation also discards loans returned by queries already running.
    pub epoch: u64,
}
impl Default for SearchCache {
    fn default() -> Self {
        Self {
            entries: HashMap::new(),
            budget: DEFAULT_SEARCH_CACHE_BYTES,
            clock: 0,
            epoch: 0,
        }
    }
}
pub(crate) fn new_shared_search_cache() -> SharedSearchCache {
    Arc::new(Mutex::new(SearchCache::default()))
}
impl SearchCache {
    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn stats(&self) -> VectorCacheStats {
        VectorCacheStats {
            budget_bytes: self.budget,
            retained_bytes: self.entries.values().map(Entry::bytes).sum(),
            vector_indexes: self
                .entries
                .values()
                .filter(|e| matches!(e.value, Value::Vectors(_)))
                .count(),
            graph_indexes: self
                .entries
                .values()
                .filter(|e| matches!(e.value, Value::Graph(_)))
                .count(),
        }
    }
    pub fn set_budget(&mut self, budget: usize) {
        self.budget = budget;
        self.epoch = self.epoch.wrapping_add(1);
        self.trim(0);
    }
    fn trim(&mut self, incoming: usize) {
        while self.stats().retained_bytes.saturating_add(incoming) > self.budget {
            let Some(key) = self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.touched)
                .map(|(k, _)| k.clone())
            else {
                break;
            };
            self.entries.remove(&key);
        }
    }
    fn insert(&mut self, key: String, value: Value) {
        let entry = Entry {
            value,
            touched: self.clock,
        };
        if entry.bytes() > self.budget || self.budget == 0 {
            return;
        }
        self.entries.remove(&key);
        self.trim(entry.bytes());
        self.clock = self.clock.wrapping_add(1);
        self.entries.insert(
            key,
            Entry {
                touched: self.clock,
                ..entry
            },
        );
    }
    pub fn vectors(&mut self, key: &str, generation: u64) -> Option<Arc<VectorBlock>> {
        let key = format!("v:{key}");
        if self
            .entries
            .get(&key)
            .is_some_and(|e| matches!(&e.value, Value::Vectors(v) if v.generation < generation))
        {
            self.entries.remove(&key);
            return None;
        }
        let entry = self.entries.get_mut(&key)?;
        match &entry.value {
            Value::Vectors(v) if v.generation == generation => {
                self.clock = self.clock.wrapping_add(1);
                entry.touched = self.clock;
                Some(Arc::clone(&v.vectors))
            }
            _ => None,
        }
    }
    pub fn insert_vectors(&mut self, key: &str, generation: u64, vectors: VectorBlock, epoch: u64) {
        if self.epoch != epoch {
            return;
        }
        let key = format!("v:{key}");
        if self
            .entries
            .get(&key)
            .is_some_and(|e| matches!(&e.value, Value::Vectors(v) if v.generation > generation))
        {
            return;
        }
        self.insert(
            key,
            Value::Vectors(CachedVectors {
                generation,
                vectors: Arc::new(vectors),
            }),
        );
    }
    pub fn take_graph(&mut self, table: &str) -> Option<CachedGraph> {
        self.entries
            .remove(&format!("g:{table}"))
            .and_then(|e| match e.value {
                Value::Graph(g) => Some(g),
                _ => None,
            })
    }
    pub fn insert_graph(&mut self, table: &str, graph: CachedGraph, epoch: u64) {
        if self.epoch != epoch {
            return;
        }
        let key = format!("g:{table}");
        if self
            .entries
            .get(&key)
            .is_some_and(|e| matches!(&e.value, Value::Graph(g) if g.revision > graph.revision))
        {
            return;
        }
        self.insert(key, Value::Graph(graph));
    }
    pub fn evict_field(&mut self, collection: &str, field: &str) {
        let vector = format!("v:{collection}::{field}");
        let graph = format!("g:hnsw::{collection}::{field}::");
        self.epoch = self.epoch.wrapping_add(1);
        self.entries
            .retain(|k, _| *k != vector && !k.starts_with(&graph));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn block(n: usize) -> VectorBlock {
        VectorBlock {
            ids: vec![ulid::Ulid::from(1); n],
            values: vec![1.0; n * 2],
            dimensions: 2,
        }
    }
    #[test]
    fn eviction_preserves_recent_indexes_and_replacements_do_not_double_count() {
        let mut cache = SearchCache::default();
        cache.set_budget(block(10).bytes() * 2);
        let epoch = cache.epoch;
        cache.insert_vectors("a", 1, block(10), epoch);
        cache.insert_vectors("b", 1, block(10), epoch);
        assert!(cache.vectors("a", 1).is_some());
        cache.insert_vectors("c", 1, block(10), epoch);
        assert!(cache.vectors("a", 1).is_some());
        assert!(cache.vectors("b", 1).is_none());
        cache.insert_vectors("a", 2, block(10), epoch);
        assert!(cache.vectors("c", 1).is_some());
        assert_eq!(cache.stats().retained_bytes, cache.budget());
        cache.set_budget(0);
        assert_eq!(cache.stats().retained_bytes, 0);
        cache.insert_vectors("a", 3, block(10), epoch);
        assert_eq!(cache.stats().retained_bytes, 0);
    }

    #[test]
    fn obsolete_vectors_are_freed_without_an_old_snapshot_evicting_newer_vectors() {
        let mut cache = SearchCache::default();
        cache.insert_vectors("a", 2, block(10), cache.epoch);
        assert!(cache.vectors("a", 1).is_none());
        assert!(cache.vectors("a", 2).is_some());
        assert!(cache.vectors("a", 3).is_none());
        assert_eq!(cache.stats().retained_bytes, 0);
    }
}
