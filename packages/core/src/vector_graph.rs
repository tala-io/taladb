//! Transactional, portable HNSW. Nodes are separate storage records, not a
//! process-wide graph blob. Document writes and link changes share a transaction.
//! Algorithm: Malkov/Yashunin, https://arxiv.org/abs/1603.09320 (algorithms 1–4).
//! Deleted nodes remain traversable; rebuilding compacts tombstones.
use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::engine::{ReadTxn, WriteTxn, WriteView};
use crate::error::TalaDbError;
use crate::vector::{VectorMetric, l2_norm, norm_sq, score_with_norms};

pub(crate) const META: &str = "meta::vector_graphs";
pub(crate) const BUILDS: &str = "meta::vector_builds";
const FORMAT: u32 = 1;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Quantization {
    #[default]
    None,
    Scalar,
    Binary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct GraphOptions {
    pub m: u32,
    #[serde(alias = "ef_construction")]
    pub ef_construction: u32,
    pub quantization: Quantization,
}
impl Default for GraphOptions {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 200,
            quantization: Quantization::None,
        }
    }
}
impl GraphOptions {
    pub fn validate(&self, metric: VectorMetric) -> Result<(), TalaDbError> {
        if !(2..=128).contains(&self.m)
            || self.ef_construction < self.m
            || self.ef_construction > 100_000
        {
            return Err(invalid(
                "HNSW requires 2 <= m <= 128 and m <= efConstruction <= 100000",
            ));
        }
        if metric == VectorMetric::Dot {
            return Err(invalid(
                "HNSW requires cosine or euclidean; use exact search for dot product",
            ));
        }
        if self.quantization == Quantization::Binary && metric != VectorMetric::Cosine {
            return Err(invalid("binary quantization requires cosine similarity"));
        }
        Ok(())
    }
}
pub(crate) fn invalid(msg: &str) -> TalaDbError {
    TalaDbError::InvalidOperation(msg.into())
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Header {
    pub format: u32,
    pub table: String,
    pub revision: u64,
    pub options: GraphOptions,
    pub dimensions: usize,
    pub metric: VectorMetric,
    pub entry: Option<u64>,
    pub level: usize,
    pub next: u64,
    pub live: u64,
    pub deleted: u64,
}
impl Header {
    pub fn new(
        table: String,
        revision: u64,
        options: GraphOptions,
        dimensions: usize,
        metric: VectorMetric,
    ) -> Self {
        Self {
            format: FORMAT,
            table,
            revision,
            options,
            dimensions,
            metric,
            entry: None,
            level: 0,
            next: 0,
            live: 0,
            deleted: 0,
        }
    }
}
pub(crate) fn header(txn: &dyn ReadTxn, key: &str) -> Result<Option<Header>, TalaDbError> {
    let h: Option<Header> = txn
        .get(META, key.as_bytes())?
        .map(|b| postcard::from_bytes(&b))
        .transpose()?;
    if let Some(h) = &h {
        if h.format != FORMAT {
            return Err(invalid("unsupported HNSW format; rebuild the vector index"));
        }
        h.options.validate(h.metric)?;
        if h.level > 16 || h.dimensions == 0 {
            return Err(invalid("invalid HNSW header"));
        }
    }
    Ok(h)
}
pub(crate) fn save_header(
    txn: &mut dyn WriteTxn,
    key: &str,
    h: &Header,
) -> Result<(), TalaDbError> {
    txn.put(META, key.as_bytes(), &postcard::to_allocvec(h)?)
}

#[derive(Clone, Serialize, Deserialize)]
enum Code {
    Float(Vec<f32>),
    Scalar {
        values: Vec<u8>,
        min: f32,
        step: f32,
    },
    Binary(Vec<u8>),
}
impl Code {
    fn encode(v: &[f32], mode: Quantization) -> Self {
        match mode {
            Quantization::None => Self::Float(v.to_vec()),
            Quantization::Scalar => {
                let min = v.iter().copied().fold(f32::INFINITY, f32::min);
                let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                // f64 avoids overflow on finite, large f32 inputs.
                let step = ((f64::from(max) - f64::from(min)) / 255.0) as f32;
                let values = v
                    .iter()
                    .map(|x| {
                        if step == 0.0 {
                            0
                        } else {
                            ((f64::from(*x) - f64::from(min)) / f64::from(step))
                                .round()
                                .clamp(0.0, 255.0) as u8
                        }
                    })
                    .collect();
                Self::Scalar { values, min, step }
            }
            Quantization::Binary => {
                let mut bits = vec![0; v.len().div_ceil(8)];
                for (i, x) in v.iter().enumerate() {
                    if *x >= 0.0 {
                        bits[i / 8] |= 1 << (i % 8);
                    }
                }
                Self::Binary(bits)
            }
        }
    }
    /// Borrow the stored vector where possible.
    ///
    /// This is on the hottest path in the crate: every distance computation in
    /// both search and construction calls it once. `Quantization::None` is the
    /// default, and its vector is already `Vec<f32>` in exactly the layout the
    /// scorer wants — so returning `Vec<f32>` meant cloning 384 floats
    /// (~1.5 KB, a heap allocation plus a memcpy) per distance, purely to hand
    /// out a `&[f32]` that the caller drops immediately.
    ///
    /// The quantized variants genuinely have to reconstruct, so they still
    /// allocate. `Cow` lets the common case cost nothing without giving that up.
    fn decode(&self, dimensions: usize) -> Cow<'_, [f32]> {
        match self {
            Self::Float(v) => Cow::Borrowed(v),
            Self::Scalar { values, min, step } => Cow::Owned(
                values
                    .iter()
                    .map(|x| (f64::from(*min) + f64::from(*step) * f64::from(*x)) as f32)
                    .collect(),
            ),
            Self::Binary(v) => Cow::Owned(
                (0..dimensions)
                    .map(|i| {
                        if v[i / 8] & (1 << (i % 8)) != 0 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect(),
            ),
        }
    }
    fn valid(&self, d: usize) -> bool {
        match self {
            Self::Float(v) => v.len() == d && v.iter().all(|x| x.is_finite()),
            Self::Scalar { values, min, step } => {
                values.len() == d && min.is_finite() && step.is_finite() && *step >= 0.0
            }
            Self::Binary(v) => v.len() == d.div_ceil(8),
        }
    }
}
#[derive(Clone, Serialize, Deserialize)]
struct Node {
    doc: [u8; 16],
    code: Code,
    links: Vec<Vec<u64>>,
    deleted: bool,
}
fn node_key(id: u64) -> [u8; 9] {
    let mut key = [1; 9];
    key[1..].copy_from_slice(&id.to_be_bytes());
    key
}
fn map_key(id: &[u8; 16]) -> [u8; 17] {
    let mut key = [2; 17];
    key[1..].copy_from_slice(id);
    key
}

#[derive(Clone, Copy, PartialEq)]
struct Hit(f32, u64);
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, rhs: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(rhs))
    }
}
impl Ord for Hit {
    fn cmp(&self, rhs: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&rhs.0).then(self.1.cmp(&rhs.1))
    }
}

/// Hasher for the node cache.
///
/// Node ids are a dense sequence of `u64`s generated by this module, not
/// attacker-controlled input, so the default `SipHash` buys nothing and costs
/// real time: profiling graph construction put 3.4% of total runtime in
/// `sip::Hasher` alone. A single multiply by a 64-bit odd constant scatters
/// sequential ids across the table well enough for an in-memory cache.
#[derive(Default, Clone, Copy)]
pub(crate) struct IdHasher(u64);
impl std::hash::Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        // Only ever fed `u64` keys; this exists to satisfy the trait.
        for b in bytes {
            self.0 = (self.0 ^ u64::from(*b)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        self.0 ^= self.0 >> 32;
    }
}
#[derive(Default, Clone, Copy)]
pub(crate) struct IdHash;
impl std::hash::BuildHasher for IdHash {
    type Hasher = IdHasher;
    fn build_hasher(&self) -> IdHasher {
        IdHasher::default()
    }
}

/// A cached node and the one derived value the hot loop needs from it.
struct Cached {
    node: Node,
    /// Squared L2 norm of the decoded vector. Constant per node, needed by every
    /// cosine comparison against it.
    norm_sq: f32,
}

/// Decoded nodes, owned by the caller rather than by a `Reader`.
///
/// `insert` needs two readers — one for the descent, one for back-linking after
/// the new node is written — because the second must coexist with `&mut txn`.
/// When the cache lived inside the reader, that meant every node touched during
/// an insert was read from storage and postcard-decoded **twice**: measured at
/// 1,140 storage fetches per inserted vector against a graph holding only 2,000
/// nodes. Hoisting it here makes the second pass free.
#[derive(Default)]
pub(crate) struct NodeCache {
    nodes: HashMap<u64, Cached, IdHash>,
    bytes: usize,
}

struct Reader<'a> {
    txn: &'a dyn ReadTxn,
    h: &'a Header,
    cache: &'a mut NodeCache,
    distances: usize,
}
impl<'a> Reader<'a> {
    fn new(txn: &'a dyn ReadTxn, h: &'a Header, cache: &'a mut NodeCache) -> Self {
        Self {
            txn,
            h,
            cache,
            distances: 0,
        }
    }
    fn cached(&mut self, id: u64) -> Result<&Cached, TalaDbError> {
        if !self.cache.nodes.contains_key(&id) {
            let bytes = self
                .txn
                .get(&self.h.table, &node_key(id))?
                .ok_or_else(|| invalid("missing HNSW node; rebuild the vector index"))?;
            let node: Node = postcard::from_bytes(&bytes)?;
            if node.links.is_empty()
                || node.links.len() > 17
                || !node.code.valid(self.h.dimensions)
                || node
                    .links
                    .iter()
                    .any(|l| l.len() > self.h.options.m as usize * 2)
            {
                return Err(invalid("invalid HNSW node; rebuild the vector index"));
            }
            // Bound retained node records on phones and in WASM. The queue and
            // visited IDs remain lightweight; evicted nodes can be read again.
            if self.cache.bytes.saturating_add(bytes.len()) > 8 * 1024 * 1024 {
                self.cache.nodes.clear();
                self.cache.bytes = 0;
            }
            self.cache.bytes = self.cache.bytes.saturating_add(bytes.len());
            let norm = norm_sq(&node.code.decode(self.h.dimensions));
            self.cache.nodes.insert(
                id,
                Cached {
                    node,
                    norm_sq: norm,
                },
            );
        }
        Ok(self.cache.nodes.get(&id).unwrap())
    }
    fn node(&mut self, id: u64) -> Result<&Node, TalaDbError> {
        Ok(&self.cached(id)?.node)
    }
    fn distance_with_norm(
        &mut self,
        query: &[f32],
        query_norm: f32,
        id: u64,
    ) -> Result<Hit, TalaDbError> {
        let dimensions = self.h.dimensions;
        let metric = self.h.metric;
        // Scoped so the borrow of `self.nodes` ends before `self.distances` is
        // touched: with a borrowed `Cow` the vector points into the cached node.
        let score = {
            let cached = self.cached(id)?;
            let v = cached.node.code.decode(dimensions);
            score_with_norms(&metric, query, query_norm, &v, cached.norm_sq)
        };
        self.distances += 1;
        if !score.is_finite() {
            return Err(invalid(
                "vector magnitude overflows HNSW distance; normalize the embedding",
            ));
        }
        Ok(Hit(-score, id))
    }
    fn distance(&mut self, query: &[f32], id: u64) -> Result<Hit, TalaDbError> {
        self.distance_with_norm(query, l2_norm(query), id)
    }
    fn greedy(&mut self, q: &[f32], entry: u64, layer: usize) -> Result<u64, TalaDbError> {
        let query_norm = l2_norm(q);
        let mut best = self.distance_with_norm(q, query_norm, entry)?;
        loop {
            let before = best;
            let links = self
                .node(best.1)?
                .links
                .get(layer)
                .cloned()
                .unwrap_or_default();
            for id in links {
                let h = self.distance_with_norm(q, query_norm, id)?;
                if h < best {
                    best = h;
                }
            }
            if best == before {
                return Ok(best.1);
            }
        }
    }
    // Disallowed/tombstoned nodes are routing bridges, never returned hits.
    // Only eligible hits tighten the stopping bound, so selective filters do
    // not strand traversal at a rejected node. ANN remains approximate.
    fn layer(
        &mut self,
        q: &[f32],
        entries: &[u64],
        layer: usize,
        ef: usize,
        allowed: Option<&HashSet<[u8; 16]>>,
        live_only: bool,
    ) -> Result<Vec<Hit>, TalaDbError> {
        let query_norm = l2_norm(q);
        let mut visited = HashSet::new();
        let mut queue = BinaryHeap::new();
        let mut best = BinaryHeap::new();
        for &id in entries {
            let hit = self.distance_with_norm(q, query_norm, id)?;
            visited.insert(id);
            queue.push(Reverse(hit));
            let n = self.node(id)?;
            if (!live_only || !n.deleted) && allowed.is_none_or(|a| a.contains(&n.doc)) {
                best.push(hit);
            }
        }
        while let Some(Reverse(hit)) = queue.pop() {
            if best.len() >= ef && best.peek().is_some_and(|worst| hit > *worst) {
                break;
            }
            let links = self
                .node(hit.1)?
                .links
                .get(layer)
                .cloned()
                .unwrap_or_default();
            for id in links {
                if !visited.insert(id) {
                    continue;
                }
                let next = self.distance_with_norm(q, query_norm, id)?;
                if best.len() < ef || best.peek().is_some_and(|worst| next < *worst) {
                    queue.push(Reverse(next));
                    let n = self.node(id)?;
                    if (!live_only || !n.deleted) && allowed.is_none_or(|a| a.contains(&n.doc)) {
                        best.push(next);
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        Ok(best.into_sorted_vec())
    }
    fn select(&mut self, candidates: Vec<Hit>, limit: usize) -> Result<Vec<u64>, TalaDbError> {
        let mut selected = Vec::new();
        let mut rejected = Vec::new();
        for hit in candidates {
            let dimensions = self.h.dimensions;
            // Owned: `point` is held across `self.distance`, which reborrows mutably.
            let point = self.node(hit.1)?.code.decode(dimensions).into_owned();
            // `point` does not change across the inner loop, so its norm is
            // loop-invariant. `distance` recomputes it per comparison, which put
            // a whole extra pass over the vector inside the diversity check —
            // 15.9% of construction time lived in this function.
            let point_norm = l2_norm(&point);
            let mut diverse = true;
            for &other in &selected {
                if self.distance_with_norm(&point, point_norm, other)?.0 < hit.0 {
                    diverse = false;
                    break;
                }
            }
            if diverse {
                selected.push(hit.1);
            } else {
                rejected.push(hit.1);
            }
            if selected.len() == limit {
                return Ok(selected);
            }
        }
        selected.extend(
            rejected
                .into_iter()
                .take(limit.saturating_sub(selected.len())),
        );
        Ok(selected)
    }
}

/// Tombstone the current version of a document. Old links remain valid.
pub(crate) fn remove(
    txn: &mut dyn WriteTxn,
    h: &mut Header,
    doc: &[u8; 16],
) -> Result<(), TalaDbError> {
    if let Some(bytes) = txn.get(&h.table, &map_key(doc))? {
        txn.delete(&h.table, &map_key(doc))?;
        let id = u64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| invalid("invalid HNSW mapping"))?,
        );
        let mut cache = NodeCache::default();
        let mut n = Reader::new(&WriteView(txn), h, &mut cache)
            .node(id)?
            .clone();
        if !n.deleted {
            n.deleted = true;
            h.live = h
                .live
                .checked_sub(1)
                .ok_or_else(|| invalid("invalid HNSW live-node count; rebuild the vector index"))?;
            h.deleted = h
                .deleted
                .checked_add(1)
                .ok_or_else(|| invalid("HNSW deleted-node count overflow"))?;
        }
        txn.put(&h.table, &node_key(id), &postcard::to_allocvec(&n)?)?;
    }
    Ok(())
}
pub(crate) fn insert(
    txn: &mut dyn WriteTxn,
    h: &mut Header,
    doc: [u8; 16],
    values: &[f32],
) -> Result<(), TalaDbError> {
    if values.len() != h.dimensions || !values.iter().all(|x| x.is_finite()) {
        return Err(invalid("invalid HNSW vector"));
    }
    remove(txn, h, &doc)?;
    let id = h.next;
    h.next = h
        .next
        .checked_add(1)
        .ok_or_else(|| invalid("HNSW node ID exhausted"))?;
    // SplitMix64 gives a reproducible level distribution without native RNG or threads.
    let mut rng = id.wrapping_add(0x9e3779b97f4a7c15);
    rng = (rng ^ (rng >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    rng = (rng ^ (rng >> 27)).wrapping_mul(0x94d049bb133111eb);
    rng ^= rng >> 31;
    let mut level = 0;
    while level < 16 && rng.is_multiple_of(u64::from(h.options.m)) {
        level += 1;
        rng /= u64::from(h.options.m);
    }
    let code = Code::encode(values, h.options.quantization);
    // Owned: `code` is moved into the node below.
    let query = code.decode(h.dimensions).into_owned();
    let mut node = Node {
        doc,
        code,
        links: vec![vec![]; level + 1],
        deleted: false,
    };
    if let Some(mut entry) = h.entry {
        // One cache for the whole insert: the descent below and the back-linking
        // pass further down both traverse the same neighbourhood, so the second
        // pass should not pay to read it from storage again.
        let mut cache = NodeCache::default();
        // Scoped rather than `drop(reader)`: the reader now borrows `cache`
        // instead of owning it, so it has nothing to drop — the block is what
        // actually ends the borrow of `txn` before the writes below.
        {
            let view = WriteView(txn);
            let mut reader = Reader::new(&view, h, &mut cache);
            for layer in ((level + 1)..=h.level).rev() {
                entry = reader.greedy(&query, entry, layer)?;
            }
            // Algorithm 1 carries the *whole* result set from each layer down as
            // the entry points for the next one. This used to keep only the
            // single nearest candidate, which restarts every layer's search from
            // one point and explores a correspondingly narrower neighbourhood —
            // the new node then links to whatever that narrow search happened to
            // find. `layer` already takes a slice, so the fix is to stop
            // throwing the rest away.
            let mut entries = vec![entry];
            for layer in (0..=level.min(h.level)).rev() {
                let candidates = reader.layer(
                    &query,
                    &entries,
                    layer,
                    h.options.ef_construction as usize,
                    None,
                    false,
                )?;
                // An empty result would leave the next layer with nowhere to
                // start, so hold the previous entry set in that case.
                if !candidates.is_empty() {
                    entries = candidates.iter().map(|hit| hit.1).collect();
                }
                // Layer 0 takes twice the connections, like every other layer's
                // pruning limit a few lines down already assumes (`M_max0` in
                // Malkov/Yashunin, algorithm 1). Building it with plain `m`
                // while pruning allowed `m * 2` left the base layer — the one
                // every search actually terminates on — under-connected, and
                // the gap widens with the graph: recall@10 at efSearch 100 fell
                // from 91.5% at 2k vectors to 56% at 10k.
                let limit = h.options.m as usize * if layer == 0 { 2 } else { 1 };
                node.links[layer] = reader.select(candidates, limit)?;
            }
        }
        txn.put(&h.table, &node_key(id), &postcard::to_allocvec(&node)?)?;

        // Connect both directions; prune using the diversity heuristic.
        //
        // One `Reader` for the whole pass. This previously constructed a fresh
        // one per neighbour — because `txn.put` needs `&mut txn` and the reader
        // borrows it — which threw away the node cache on every iteration, so
        // each pruning pass re-read and re-parsed every candidate node from
        // storage. That is the dominant cost of graph construction.
        //
        // Deferring the writes lets a single reader serve the whole pass. It is
        // safe because link edits never change a node's vector and pruning reads
        // vectors only: `distance` and `select` both touch `code`, never `links`.
        // The one node whose links *are* being edited is taken from `updates`
        // rather than the cache, so a neighbour revisited on another layer sees
        // its own pending change.
        let mut updates: HashMap<u64, Node> = HashMap::new();
        {
            let view = WriteView(txn);
            let mut reader = Reader::new(&view, h, &mut cache);
            for (layer, neighbors) in node.links.iter().enumerate() {
                for &neighbor in neighbors {
                    let mut n = match updates.get(&neighbor) {
                        Some(pending) => pending.clone(),
                        None => reader.node(neighbor)?.clone(),
                    };
                    n.links[layer].push(id);
                    let limit = h.options.m as usize * if layer == 0 { 2 } else { 1 };
                    if n.links[layer].len() > limit {
                        // Owned: `n.links` is reassigned while this is still live.
                        let q = n.code.decode(h.dimensions).into_owned();
                        let mut candidates = n.links[layer]
                            .iter()
                            .map(|&v| reader.distance(&q, v))
                            .collect::<Result<Vec<_>, _>>()?;
                        candidates.sort_unstable();
                        n.links[layer] = reader.select(candidates, limit)?;
                    }
                    updates.insert(neighbor, n);
                }
            }
        }
        for (neighbor, n) in updates {
            txn.put(&h.table, &node_key(neighbor), &postcard::to_allocvec(&n)?)?;
        }
    } else {
        txn.put(&h.table, &node_key(id), &postcard::to_allocvec(&node)?)?;
    }
    if h.entry.is_none() || level > h.level {
        h.entry = Some(id);
        h.level = level;
    }
    h.live = h
        .live
        .checked_add(1)
        .ok_or_else(|| invalid("HNSW live-node count overflow"))?;
    txn.put(&h.table, &map_key(&doc), &id.to_le_bytes())?;
    Ok(())
}

/// Search the graph, reusing `cache` across calls.
///
/// The caller owns the cache because the ANN path retries with a doubled
/// `efSearch` when a filter leaves too few rows, and each retry walks the same
/// neighbourhood again. With the cache inside this function every retry re-read
/// and re-decoded nodes it had just decoded. Sharing it is unconditionally safe
/// here: a retry runs inside the same read transaction, against the same
/// snapshot and the same graph revision, so a cached node cannot be stale.
pub(crate) fn search(
    txn: &dyn ReadTxn,
    h: &Header,
    query: &[f32],
    ef: usize,
    allowed: Option<&HashSet<[u8; 16]>>,
    cache: &mut NodeCache,
) -> Result<(Vec<[u8; 16]>, usize), TalaDbError> {
    let Some(mut entry) = h.entry else {
        return Ok((vec![], 0));
    };
    if h.live == 0 {
        return Ok((vec![], 0));
    }
    let q = if h.options.quantization == Quantization::Binary {
        Code::encode(query, Quantization::Binary)
            .decode(h.dimensions)
            .into_owned()
    } else {
        query.to_vec()
    };
    let mut reader = Reader::new(txn, h, cache);
    for layer in (1..=h.level).rev() {
        entry = reader.greedy(&q, entry, layer)?;
    }
    let hits = reader.layer(&q, &[entry], 0, ef.max(1), allowed, true)?;
    let ids = hits
        .iter()
        .map(|h| reader.node(h.1).map(|n| n.doc))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((ids, reader.distances))
}
