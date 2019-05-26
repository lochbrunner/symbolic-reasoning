use crate::symbol::Symbol;
use std::collections::HashMap;
use std::hash::{BuildHasher, Hash};

/// Example
/// ```
/// #[macro_use]
/// extern crate maplit;
/// use core::common::merge;
///
/// fn main() {
///     let first = hashmap! {"a" => 1};
///     let second = hashmap! {"b" => 2};
///
///     let actual = merge(&first, &second);
///     let expected = hashmap! {"a" => 1, "b" => 2};
///     assert_eq!(actual, expected);
/// }
/// ```
pub fn merge<K: Hash + Eq + Copy, V: Copy, S: BuildHasher + Default>(
    first: &HashMap<K, V, S>,
    second: &HashMap<K, V, S>,
) -> HashMap<K, V, S> {
    let mut merged = HashMap::default();
    for (key, value) in first.iter() {
        merged.insert(*key, *value);
    }
    for (key, value) in second.iter() {
        merged.insert(*key, *value);
    }
    merged
}

pub fn merge_in_place<K: Hash + Eq + Copy, V: Copy, S: BuildHasher + Default>(
    target: &mut HashMap<K, V, S>,
    source: &HashMap<K, V, S>,
) {
    for (key, value) in source.iter() {
        target.insert(*key, *value);
    }
}

pub fn merge_from<K: Hash + Eq + Copy, V: Copy, S: BuildHasher + Default>(
    target: &mut HashMap<K, V, S>,
    source: HashMap<K, V, S>,
) {
    for (key, value) in source.into_iter() {
        target.insert(key, value);
    }
}

pub fn format_mapping<S: BuildHasher + Default>(mapping: &HashMap<&Symbol, &Symbol, S>) -> String {
    mapping
        .iter()
        .map(|(source, target)| format!("{} => {}", source, target))
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod specs {
    use super::*;

    #[test]
    fn merge_simple() {
        let first = hashmap! {"a" => 1};
        let second = hashmap! {"b" => 2};

        let actual = merge(&first, &second);
        let expected = hashmap! {"a" => 1, "b" => 2};

        assert_eq!(actual, expected);
    }

    #[test]
    fn merge_overlap() {
        let first = hashmap! {"a" => 1, "c" => 3};
        let second = hashmap! {"b" => 2, "c" => 3};

        let actual = merge(&first, &second);
        let expected = hashmap! {"a" => 1, "b" => 2, "c" => 3};

        assert_eq!(actual, expected);
    }

    #[test]
    fn merge_in_place_simple() {
        let mut first = hashmap! {"a" => 1};
        let second = hashmap! {"b" => 2};

        merge_in_place(&mut first, &second);
        let expected = hashmap! {"a" => 1, "b" => 2};

        assert_eq!(first, expected);
    }

    #[test]
    fn merge_in_place_overlap() {
        let mut first = hashmap! {"a" => 1, "c" => 3};
        let second = hashmap! {"b" => 2, "c" => 3};

        merge_in_place(&mut first, &second);
        let expected = hashmap! {"a" => 1, "b" => 2, "c" => 3};

        assert_eq!(first, expected);
    }
}
