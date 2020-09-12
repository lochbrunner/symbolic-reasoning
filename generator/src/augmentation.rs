use rand::prelude::*;
use std::collections::{HashMap, HashSet};

use core::bag::FitInfo;
use core::Symbol;

#[derive(Clone, Copy)]
pub enum AugmentationStrategy {
    #[cfg(test)]
    Full,
    Random(usize),
}

fn create_combinations<'a, T>(
    length: usize,
    items: &[&'a T],
    strategy: AugmentationStrategy,
) -> Vec<Vec<&'a T>> {
    let length = length as u32;
    let num_items = items.len();
    let num = num_items.pow(length);
    let mut combinations = Vec::with_capacity(num);
    let indices = match strategy {
        #[cfg(test)]
        AugmentationStrategy::Full => (0..num).collect::<Vec<_>>(),
        AugmentationStrategy::Random(size) => {
            let mut set: HashSet<usize> = HashSet::new();
            let mut rng: ThreadRng = rand::thread_rng();
            while set.len() < size {
                let proposal = rng.gen_range(0, num);
                if !set.contains(&proposal) {
                    set.insert(proposal);
                }
            }
            set.into_iter().collect::<Vec<_>>()
        }
    };

    for i in indices {
        let mut combination = vec![];
        for j in 0..length {
            let index = (i / num_items.pow(j)) % num_items;
            combination.push(items[index]);
        }
        combinations.push(combination);
    }
    combinations
}

pub fn augment_with_permuted_free_idents(
    free_idents: &HashSet<&str>,
    leafs: &[&Symbol],
    strategy: AugmentationStrategy,
    orig: (&Symbol, FitInfo),
) -> Vec<(Symbol, FitInfo)> {
    // Assume each alphabetically char is free
    let mut per_mapping: HashMap<&String, Vec<Vec<usize>>> = HashMap::new();
    for (path, sub) in orig.0.iter_dfs_path() {
        if free_idents.contains(&sub.ident[..]) {
            match per_mapping.get_mut(&sub.ident) {
                None => {
                    per_mapping.insert(&sub.ident, vec![path]);
                }
                Some(v) => v.push(path),
            }
        }
    }
    //
    create_combinations(per_mapping.len(), leafs, strategy)
        .iter()
        .map(|combination| {
            let mut new_deduced = orig.0.clone();
            for ((_, paths), leaf) in per_mapping.iter().zip(combination.iter()) {
                for path in paths.iter() {
                    new_deduced.set_ident_at(path, leaf.ident.clone()).unwrap();
                }
            }
            (new_deduced, orig.1.clone())
        })
        .collect()
}

#[cfg(test)]
mod specs {
    use super::*;
    use core::bag::Policy;
    use core::Context;

    static STRATEGY: AugmentationStrategy = AugmentationStrategy::Full;

    macro_rules! hashset {
        ( $( $x:expr ),* ) => {  // Match zero or more comma delimited items
            {
                let mut temp_set = HashSet::new();  // Create a mutable HashSet
                $(
                    temp_set.insert($x); // Insert each item matched into the HashSet
                )*
                temp_set // Return the populated HashSet
            }
        };
    }

    macro_rules! assert_contains_symbol {
        ($container:expr, $symbol_code:expr, $context:expr) => {
            let symbol = Symbol::parse($context, $symbol_code).unwrap();
            if let None = $container.iter().position(|s| s == &&symbol) {
                panic!("Container does not contain symbol {}", $symbol_code);
            }
        };
    }

    #[test]
    fn create_combinations_2_of_3() {
        let items = vec![&1, &2, &3];
        let combinations = create_combinations(2, &items, STRATEGY);
        assert_eq!(
            combinations,
            vec![
                vec![&1, &1,],
                vec![&2, &1,],
                vec![&3, &1,],
                vec![&1, &2,],
                vec![&2, &2,],
                vec![&3, &2,],
                vec![&1, &3,],
                vec![&2, &3,],
                vec![&3, &3,],
            ]
        );
    }

    #[test]
    fn create_combinations_3_of_2() {
        let items = vec![&1, &2];
        let combinations = create_combinations(3, &items, STRATEGY);
        assert_eq!(
            combinations,
            vec![
                vec![&1, &1, &1,],
                vec![&2, &1, &1,],
                vec![&1, &2, &1,],
                vec![&2, &2, &1,],
                vec![&1, &1, &2,],
                vec![&2, &1, &2,],
                vec![&1, &2, &2,],
                vec![&2, &2, &2,],
            ]
        );
    }

    #[test]
    fn permuted_free_idents_one_ident() {
        let free_idents = hashset!["a", "b"];
        let context = Context::standard();
        let leafs = vec![
            Symbol::new_variable("a", true),
            Symbol::new_variable("b", true),
        ];
        let leafs = leafs.iter().collect::<Vec<_>>();

        let orig_symbol = Symbol::parse(&context, "a=a").unwrap();
        let orig = (
            &orig_symbol,
            FitInfo {
                rule_id: 1,
                path: vec![],
                policy: Policy::Positive,
            },
        );
        let actual = augment_with_permuted_free_idents(&free_idents, &leafs, STRATEGY, orig);
        let actual = actual.iter().map(|(s, _)| s).collect::<Vec<_>>();
        assert_eq!(actual.len(), 2);
        assert_contains_symbol!(actual, "a=a", &context);
        assert_contains_symbol!(actual, "b=b", &context);
    }

    #[test]
    fn permuted_free_idents_two_idents() {
        let free_idents = hashset!["a", "b"];
        let context = Context::standard();
        let leafs = vec![
            Symbol::new_variable("a", true),
            Symbol::new_variable("b", true),
        ];
        let leafs = leafs.iter().collect::<Vec<_>>();

        let orig_symbol = Symbol::parse(&context, "a=b").unwrap();
        let orig = (
            &orig_symbol,
            FitInfo {
                rule_id: 1,
                path: vec![],
                policy: Policy::Positive,
            },
        );

        let actual = augment_with_permuted_free_idents(&free_idents, &leafs, STRATEGY, orig);
        let actual = actual.iter().map(|(s, _)| s).collect::<Vec<_>>();
        assert_eq!(actual.len(), 4);
        assert_contains_symbol!(actual, "a=a", &context);
        assert_contains_symbol!(actual, "a=b", &context);
        assert_contains_symbol!(actual, "b=a", &context);
        assert_contains_symbol!(actual, "b=b", &context);
    }
}
