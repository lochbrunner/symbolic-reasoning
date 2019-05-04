use crate::symbol::Symbol;
use std::collections::HashMap;
use std::hash::Hash;

pub fn merge<K: Hash + Eq + Copy, V: Copy>(
    first: &HashMap<K, V>,
    second: &HashMap<K, V>,
) -> HashMap<K, V> {
    let mut merged = HashMap::new();
    for (key, value) in first.iter() {
        merged.insert(*key, *value);
    }
    for (key, value) in second.iter() {
        merged.insert(*key, *value);
    }
    merged
}

pub fn merge_in_place<K: Hash + Eq + Copy, V: Copy>(
    target: &mut HashMap<K, V>,
    source: &HashMap<K, V>,
) {
    for (key, value) in source.iter() {
        target.insert(*key, *value);
    }
}

pub fn merge_from<K: Hash + Eq + Copy, V: Copy>(target: &mut HashMap<K, V>, source: HashMap<K, V>) {
    for (key, value) in source.into_iter() {
        target.insert(key, value);
    }
}

pub fn format_mapping(mapping: &HashMap<&Symbol, &Symbol>) -> String {
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
