//! Read a small batch with an ordered merge when IDs are near each other. Stop
//! after twice the batch size and point-fetch the remainder for scattered IDs.
use crate::TalaDbError;
use crate::engine::{ReadTxn, ScanFlow};
use std::ops::Bound;

pub(crate) fn visit<const N: usize>(
    txn: &dyn ReadTxn,
    table: &str,
    keys: &mut Vec<[u8; N]>,
    accept: &mut impl FnMut(&[u8; N], &[u8]) -> Result<(), TalaDbError>,
) -> Result<(), TalaDbError> {
    keys.sort_unstable();
    keys.dedup();
    let Some(first) = keys.first() else {
        return Ok(());
    };
    let last = keys.last().unwrap();
    let mut cursor = 0;
    let mut seen = 0;
    let mut stopped = false;
    txn.scan(
        table,
        Bound::Included(first.as_slice()),
        Bound::Included(last.as_slice()),
        &mut |key, value| {
            while cursor < keys.len() && keys[cursor].as_slice() < key {
                cursor += 1;
            }
            if cursor < keys.len() && keys[cursor].as_slice() == key {
                accept(&keys[cursor], value)?;
                cursor += 1;
            }
            seen += 1;
            stopped = cursor == keys.len() || seen >= keys.len().saturating_mul(2);
            Ok(if stopped {
                ScanFlow::Stop
            } else {
                ScanFlow::Continue
            })
        },
    )?;
    if stopped && cursor < keys.len() {
        let remaining = &keys[cursor..];
        let refs: Vec<_> = remaining.iter().map(<[u8; N]>::as_slice).collect();
        for (key, bytes) in remaining.iter().zip(txn.get_many(table, &refs)?) {
            if let Some(bytes) = bytes {
                accept(key, &bytes)?;
            }
        }
    }
    keys.clear();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ordered_batches_and_scattered_fallback_emit_each_existing_key_once() {
        let db = crate::Database::open_in_memory().unwrap();
        let mut write = db.backend().begin_write().unwrap();
        for i in 0..1000u16 {
            if i != 9 {
                write
                    .put("batch", &i.to_be_bytes(), &i.to_le_bytes())
                    .unwrap();
            }
        }
        write.commit().unwrap();
        let txn = db.backend().begin_read().unwrap();
        for values in [
            vec![],
            vec![9],
            vec![8, 9, 10],
            vec![999, 9, 0, 999, 500, 1001],
        ] {
            let mut keys: Vec<_> = values.iter().map(|i: &u16| i.to_be_bytes()).collect();
            let mut seen = Vec::new();
            visit(txn.as_ref(), "batch", &mut keys, &mut |k, v| {
                assert_eq!(
                    u16::from_be_bytes(*k),
                    u16::from_le_bytes(v.try_into().unwrap())
                );
                seen.push(u16::from_be_bytes(*k));
                Ok(())
            })
            .unwrap();
            let mut expected: Vec<_> = values.into_iter().filter(|&i| i < 1000 && i != 9).collect();
            expected.sort_unstable();
            expected.dedup();
            assert_eq!(seen, expected);
            assert!(keys.is_empty());
        }
        let error = visit(
            txn.as_ref(),
            "batch",
            &mut vec![0u16.to_be_bytes()],
            &mut |_, _| Err(TalaDbError::SearchMemoryLimit),
        );
        assert!(matches!(error, Err(TalaDbError::SearchMemoryLimit)));
    }
}
