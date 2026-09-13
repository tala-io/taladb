//! How much does a redb `open_table` cost on a write transaction?
//!
//! `engine.rs` asserts ~2.3 µs and uses that to justify the batched `get_many`
//! / `apply_batch` hooks. The read path also caches table handles per
//! transaction; the write path does not. The HNSW build runs entirely inside a
//! write transaction and does one point read per graph node, so if that number
//! is right it dominates graph construction. Measure it rather than assume it.
use redb::{Database, ReadableTable, TableDefinition};
use std::time::Instant;

const T: TableDefinition<&[u8], &[u8]> = TableDefinition::new("bench");
const N: usize = 20_000;

fn main() {
    let file = tempfile::NamedTempFile::new().unwrap();
    let db = Database::create(file.path()).unwrap();

    // Seed some rows so lookups touch a real tree.
    let txn = db.begin_write().unwrap();
    {
        let mut t = txn.open_table(T).unwrap();
        for i in 0..N {
            t.insert(&i.to_be_bytes()[..], &vec![7u8; 1600][..])
                .unwrap();
        }
    }
    txn.commit().unwrap();

    let txn = db.begin_write().unwrap();

    // A: open the table once, then N gets — what a cached handle would cost.
    let t0 = Instant::now();
    {
        let tbl = txn.open_table(T).unwrap();
        for i in 0..N {
            std::hint::black_box(
                tbl.get(&i.to_be_bytes()[..])
                    .unwrap()
                    .map(|v| v.value().len()),
            );
        }
    }
    let cached = t0.elapsed();

    // B: open the table per get — what `RedbWriteTxn::get` does today.
    let t0 = Instant::now();
    for i in 0..N {
        let tbl = txn.open_table(T).unwrap();
        std::hint::black_box(
            tbl.get(&i.to_be_bytes()[..])
                .unwrap()
                .map(|v| v.value().len()),
        );
    }
    let per_call = t0.elapsed();

    txn.abort().unwrap();

    println!("{N} point reads on a write transaction:");
    println!(
        "  one open_table, N gets   {cached:>10.2?}  ({:>6.3} µs/get)",
        cached.as_secs_f64() * 1e6 / N as f64
    );
    println!(
        "  open_table per get       {per_call:>10.2?}  ({:>6.3} µs/get)",
        per_call.as_secs_f64() * 1e6 / N as f64
    );
    println!(
        "  open_table overhead      {:>10.3} µs/get",
        (per_call.as_secs_f64() - cached.as_secs_f64()) * 1e6 / N as f64
    );
}
