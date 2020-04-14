use crate::configuration::Configuration;
use core::bag::trace::ApplyInfo;
use core::Symbol;

/// Check for e.g. 1*(1*1) or (1*1)*1 (or (1*1)*(1*1))
fn check_trivial_subtree(sub: &Symbol) -> bool {
    if sub.depth == 3 {
        let op = &sub.ident;
        let mut var: Option<&str> = None;
        for c in sub.iter_bfs() {
            if c.operator() {
                if c.ident != *op {
                    return true;
                }
            } else {
                if let Some(v) = var {
                    if v != c.ident {
                        return true;
                    }
                } else {
                    var = Some(&c.ident);
                }
            }
        }
        return false;
    }
    return true;
}

/// Filters out non sense symbols in the sene of e.g. a^0^0^0^0^0 ...
pub fn filter_interest<'a>(apply: &'a &ApplyInfo<'_>) -> bool {
    // Filter out when a pattern repeats directly
    for sub in apply.deduced.iter_bfs() {
        if sub.operator() {
            // Check for trivial repetition
            for child in sub.childs.iter() {
                let b = child.childs.iter().map(|s| &s.ident);
                if child.childs.len() == sub.childs.len() {
                    let a = sub.childs.iter().map(|s| &s.ident);
                    if a.eq(b) {
                        return false;
                    }
                }
            }
            if !check_trivial_subtree(sub) {
                return false;
            }
        }
    }
    true
}

pub fn filter_out_blacklist<'a>(config: &Configuration, apply: &'a &ApplyInfo<'_>) -> bool {
    for sub in apply.deduced.iter_bfs() {
        for black in config.blacklist_pattern.iter() {
            if black == sub {
                return false;
            }
        }
    }
    true
}

/// Should find terms which contain the same pattern (depth >= 2) is more than n times
pub fn filter_out_repeating_patterns<'a>(apply: &'a &ApplyInfo<'_>) -> bool {
    true
}
