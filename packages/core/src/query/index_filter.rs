//! Resolve positive filters from index keys. Residual predicates still require
//! documents; negation and existence cannot be answered by sparse indexes.
use std::collections::HashSet;
use std::ops::Bound;

use super::planner::plan_full;
use super::{Filter, QueryPlan};
use crate::document::Value;
use crate::engine::{ReadTxn, ScanFlow};
use crate::error::TalaDbError;
use crate::index::{
    CompoundIndexDef, IndexDef, compound_table_name, decode_value_prefix, index_table_name,
};

pub(super) struct Candidates {
    pub ids: HashSet<[u8; 16]>,
    pub exact: bool,
}

pub(super) fn resolve(
    filter: &Filter,
    indexes: &[IndexDef],
    compounds: &[CompoundIndexDef],
    txn: &dyn ReadTxn,
    collection: &str,
) -> Result<Option<Candidates>, TalaDbError> {
    match filter {
        Filter::And(children) => {
            let mut result = None;
            let mut exact = true;
            let mut covered = vec![false; children.len()];
            if !compounds.is_empty() {
                let plan = plan_full(filter, &[], &[], compounds);
                if let Some(fields) = compound_fields(&plan) {
                    let predicates: Vec<_> = fields
                        .iter()
                        .filter_map(|field| {
                            children
                                .iter()
                                .find(|f| matches!(f, Filter::Eq(name, _) if name == field))
                        })
                        .collect();
                    for (i, child) in children.iter().enumerate() {
                        covered[i] = predicates.iter().any(|p| {
                            matches!((child, p),
                            (Filter::Eq(a, x), Filter::Eq(b, y)) if a == b && x == y)
                        });
                    }
                    let mut ids = HashSet::new();
                    scan_plan(&plan, &predicates, txn, collection, &mut ids)?;
                    result = Some(Candidates { ids, exact: true });
                }
            }
            for (child, covered) in children.iter().zip(covered) {
                if covered || matches!(child, Filter::All) {
                    continue;
                }
                let next = resolve(child, indexes, compounds, txn, collection)?;
                if let Some(next) = next {
                    exact &= next.exact;
                    if let Some(current) = &mut result {
                        intersect(&mut current.ids, next.ids);
                    } else {
                        result = Some(next);
                    }
                } else {
                    exact = false;
                }
            }
            if let Some(current) = &mut result {
                // An empty indexed set proves the whole AND false even when
                // some predicates require documents.
                current.exact = current.ids.is_empty() || exact;
            }
            Ok(result)
        }
        Filter::Or(children) => {
            let mut result = Candidates {
                ids: HashSet::new(),
                exact: true,
            };
            for child in children {
                let Some(next) = resolve(child, indexes, compounds, txn, collection)? else {
                    return Ok(None);
                };
                result.ids.extend(next.ids);
                result.exact &= next.exact;
            }
            Ok(Some(result))
        }
        Filter::Eq(field, value)
        | Filter::Gt(field, value)
        | Filter::Gte(field, value)
        | Filter::Lt(field, value)
        | Filter::Lte(field, value)
            if field != "_id" && indexes.iter().any(|i| &i.field == field) && indexable(value) =>
        {
            let plan = plan_full(filter, indexes, &[], &[]);
            let mut ids = HashSet::new();
            scan_plan(&plan, &[filter], txn, collection, &mut ids)?;
            Ok(Some(Candidates { ids, exact: true }))
        }
        Filter::In(field, values)
            if field != "_id"
                && indexes.iter().any(|i| &i.field == field)
                && values.iter().all(indexable) =>
        {
            let mut ids = HashSet::new();
            if !values.is_empty() {
                let plan = plan_full(filter, indexes, &[], &[]);
                scan_plan(&plan, &[filter], txn, collection, &mut ids)?;
            }
            Ok(Some(Candidates { ids, exact: true }))
        }
        _ => Ok(None),
    }
}

fn indexable(value: &Value) -> bool {
    !matches!(value, Value::Array(_) | Value::Object(_))
}

fn intersect(a: &mut HashSet<[u8; 16]>, mut b: HashSet<[u8; 16]>) {
    if b.len() < a.len() {
        std::mem::swap(a, &mut b);
    }
    a.retain(|id| b.contains(id));
}

fn compound_fields(plan: &QueryPlan) -> Option<&[String]> {
    match plan {
        QueryPlan::CompoundIndexEq { fields, .. } => Some(fields),
        QueryPlan::IndexOr { plans } => {
            let fields = compound_fields(plans.first()?)?;
            plans
                .iter()
                .all(|p| compound_fields(p) == Some(fields))
                .then_some(fields)
        }
        _ => None,
    }
}

fn scan_plan(
    plan: &QueryPlan,
    predicates: &[&Filter],
    txn: &dyn ReadTxn,
    collection: &str,
    ids: &mut HashSet<[u8; 16]>,
) -> Result<(), TalaDbError> {
    match plan {
        QueryPlan::IndexEq { field, start, end } => scan_keys(
            txn,
            &index_table_name(collection, field),
            Bound::Included(start),
            Bound::Included(end),
            predicates,
            ids,
        ),
        QueryPlan::IndexRange { field, start, end } => scan_keys(
            txn,
            &index_table_name(collection, field),
            start.as_ref().map(Vec::as_slice),
            end.as_ref().map(Vec::as_slice),
            predicates,
            ids,
        ),
        QueryPlan::IndexIn { field, ranges } => {
            for (start, end) in ranges {
                scan_keys(
                    txn,
                    &index_table_name(collection, field),
                    Bound::Included(start),
                    Bound::Included(end),
                    predicates,
                    ids,
                )?;
            }
            Ok(())
        }
        QueryPlan::CompoundIndexEq { fields, start, end } => {
            let refs: Vec<_> = fields.iter().map(String::as_str).collect();
            scan_keys(
                txn,
                &compound_table_name(collection, &refs),
                Bound::Included(start),
                Bound::Included(end),
                predicates,
                ids,
            )
        }
        QueryPlan::IndexOr { plans } => {
            for plan in plans {
                scan_plan(plan, predicates, txn, collection, ids)?;
            }
            Ok(())
        }
        _ => Err(TalaDbError::InvalidOperation(
            "unsupported covered filter plan".into(),
        )),
    }
}

fn scan_keys(
    txn: &dyn ReadTxn,
    table: &str,
    start: Bound<&[u8]>,
    end: Bound<&[u8]>,
    predicates: &[&Filter],
    ids: &mut HashSet<[u8; 16]>,
) -> Result<(), TalaDbError> {
    txn.scan(table, start, end, &mut |key, _| {
        let malformed = || TalaDbError::InvalidOperation("invalid secondary index key".into());
        let split = key.len().checked_sub(16).ok_or_else(malformed)?;
        let id = key[split..].try_into().map_err(|_| malformed())?;
        let mut prefix = &key[..split];
        let mut matches = true;
        for filter in predicates {
            let (value, rest) = decode_value_prefix(prefix).ok_or_else(malformed)?;
            matches &= filter.matches_index_value(&value);
            prefix = rest;
        }
        if !prefix.is_empty() {
            return Err(malformed());
        }
        if matches {
            ids.insert(id);
        }
        Ok(ScanFlow::Continue)
    })
}
