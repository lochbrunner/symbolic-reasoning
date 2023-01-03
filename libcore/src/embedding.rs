use crate::common::RefEquality;
use crate::io::bag::FitInfo;
use crate::io::bag::Policy;
use crate::symbol::Symbol;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
pub struct CnnEmbedding {
    /// each items is a vector
    /// [ident, is_operator, is_fixed, is_number]
    pub embedded: Vec<Vec<i64>>,
    pub index_map: Option<Vec<Vec<i16>>>,
    pub positional_encoding: Option<Vec<Vec<i64>>>,
    pub target: Vec<Vec<f32>>,
    pub possibility_mask: Vec<Vec<bool>>,
    /// deprecated use target instead
    pub label: Vec<i64>,
    /// deprecated use target instead
    pub policy: Vec<f32>,
    pub value: i64,
}

#[derive(Debug, PartialEq)]
pub struct GraphEmbedding {
    pub nodes: Vec<Vec<i64>>,
    /// Position of the child.
    pub edges: Vec<i32>,
    pub receivers: Vec<i16>,
    pub senders: Vec<i16>,
    pub n_node: i64,
    pub n_edge: i64,
    /// The probability of rule for node.
    pub target: Vec<Vec<f32>>,
    pub possibility_mask: Vec<Vec<bool>>,
    pub value: i64,
}

pub trait Embeddable {
    fn number_of_embedded_properties() -> u32;
    fn embed_cnn(
        &self,
        dict: &HashMap<String, i16>,
        padding: i16,
        spread: usize,
        max_depth: u32,
        target_size: usize,
        fits: &[FitInfo],
        useful: bool,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> Result<CnnEmbedding, String>;

    fn embed_graph(
        &self,
        ident2id: &HashMap<String, i16>,
        target_size: usize,
        fits: &[FitInfo],
        useful: bool,
        use_additional_features: bool,
    ) -> Result<GraphEmbedding, String>;
}

fn one_encode(value: bool) -> i64 {
    if value {
        1
    } else {
        0
    }
}

impl Symbol {
    /// Needed for transformer based architecture
    /// Using the path as digits for a representative location numbering
    /// Desired properties:
    /// * `p_a - p_b` is a metric for the relative sub-graph between node `a` and `b`
    /// * `p_a - p_b` is independent of the absolute location in the graph. Only possible with vector.
    ///     Each item per row is from an other start level.
    /// * `0` indicates undefined
    fn positional_encoding(&self, spread: usize, max_depth: u32) -> Vec<Vec<i64>> {
        struct Pack {
            offset: i64,
            // scale: i64,
            depth: u32,
        }
        struct Context {
            spread: i64,
        }

        assert_eq!(spread, 2);

        (2..(max_depth + 1))
            .rev()
            .map(|depth| {
                self.iter_bfs_backpack(
                    Pack {
                        offset: spread.pow(depth) as i64 / 2,
                        depth: if depth > 0 { depth - 1 } else { 0 },
                    },
                    Context {
                        spread: spread as i64,
                    },
                    |(parent, pack), context| {
                        parent
                            .childs
                            .iter()
                            .enumerate()
                            .map(|(i, child)| {
                                let root = if pack.depth > 0 {
                                    context.spread.pow(pack.depth - 1) * (2 * (i as i64) - 1)
                                        + pack.offset
                                } else {
                                    0
                                };
                                (
                                    child,
                                    Pack {
                                        offset: root,
                                        depth: if pack.depth > 0 { pack.depth - 1 } else { 0 },
                                    },
                                )
                            })
                            .collect()
                    },
                )
                .map(|(_, pack)| pack.offset)
                .collect()
            })
            .collect()

        // vec![buffer]
    }

    /// Needed for CNN based architecture
    /// self, ..childs, parent
    fn index_map<'a>(
        &'a self,
        spread: usize,
        padding_index: i16,
        ref_to_index: &HashMap<RefEquality<'a, Self>, i16>,
    ) -> Vec<Vec<i16>> {
        let mut index_map = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                let mut row = Vec::with_capacity(spread + 2);
                row.push(i as i16);
                for child in s.childs.iter() {
                    row.push(ref_to_index[&RefEquality(child)]);
                }
                while row.len() < spread + 1 {
                    row.push(padding_index);
                }
                row
            })
            .collect::<Vec<Vec<i16>>>();

        // Append parent
        // root has no parent
        index_map[0].push(padding_index);
        for parent in self.iter_bfs() {
            let parent_index = ref_to_index[&RefEquality(parent)];
            for child in parent.childs.iter() {
                let index = ref_to_index[&RefEquality(child)] as usize;
                index_map[index].push(parent_index);
            }
        }
        index_map.push(vec![padding_index; spread + 2]);
        index_map
    }

    fn nodes_features(
        &self,
        ident2id: &HashMap<String, i16>,
        use_additional_features: bool,
    ) -> Result<Vec<Vec<i64>>, String> {
        let mut ref_to_index: HashMap<RefEquality<Self>, i16> = HashMap::new();
        let embedded = self
            .iter_bfs()
            .enumerate()
            .map(|(i, s)| {
                ref_to_index.insert(RefEquality(s), i as i16);
                ident2id
                    .get(&s.ident)
                    .map(|i| {
                        if use_additional_features {
                            vec![
                                *i as i64,
                                one_encode(s.operator()),
                                one_encode(s.fixed()),
                                one_encode(s.is_number()),
                            ]
                        } else {
                            vec![*i as i64]
                        }
                    })
                    .ok_or(format!("Unknown ident {}", s.ident))
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(embedded)
    }

    fn ref_to_index(&self) -> HashMap<RefEquality<Self>, i16> {
        self.iter_bfs()
            .enumerate()
            .map(|(i, s)| (RefEquality(s), i as i16))
            .collect()
    }
}

impl Embeddable for Symbol {
    /// Returns the number of embedded properties
    fn number_of_embedded_properties() -> u32 {
        3
    }

    /// Embeds the ident and the props (operator, fixed, number, policy)
    /// Should maybe moved to other location?
    /// If there are multiple fits per path, the last will win.
    fn embed_cnn(
        &self,
        ident2id: &HashMap<String, i16>,
        padding: i16,
        spread: usize,
        max_depth: u32,
        target_size: usize,
        fits: &[FitInfo],
        useful: bool,
        index_map: bool,
        positional_encoding: bool,
        use_additional_features: bool,
    ) -> Result<CnnEmbedding, String> {
        let mut embedded = self.nodes_features(ident2id, use_additional_features)?;
        let padding_index = embedded.len() as i16;
        embedded.push(vec![padding as i64, 0, 0, 0]);
        let ref_to_index = self.ref_to_index();
        let index_map = if index_map {
            Some(self.index_map(spread, padding_index, &ref_to_index))
        } else {
            None
        };
        let positional_encoding = if positional_encoding {
            Some(self.positional_encoding(spread, max_depth))
        } else {
            None
        };

        let mut target = vec![vec![0.; target_size]; embedded.len()];
        let mut possibility_mask = vec![vec![false; target_size]; embedded.len()];
        // Compute label
        // Deprecated
        let mut label = vec![0; embedded.len()];
        let mut policy = vec![0.0; embedded.len()];
        for fit in fits.iter() {
            let child = self
                .at(&fit.path)
                .ok_or(format!("Symbol {} has no element at {:?}", self, fit.path))?;
            let index = ref_to_index[&RefEquality(child)] as usize;
            label[index] = fit.rule_id as i64;
            policy[index] = fit.policy.value();
            // positive policy should survive after collision
            if fit.policy != Policy::NotTried
                && target[index][fit.rule_id as usize] < Policy::POSITIVE_VALUE
            {
                target[index][fit.rule_id as usize] = fit.policy.value();
            }
            possibility_mask[index][fit.rule_id as usize] = true;
        }

        Ok(CnnEmbedding {
            embedded,
            index_map,
            target,
            label,
            policy,
            positional_encoding,
            possibility_mask,
            value: if useful { 1 } else { 0 },
        })
    }

    fn embed_graph(
        &self,
        ident2id: &HashMap<String, i16>,
        target_size: usize,
        fits: &[FitInfo],
        useful: bool,
        use_additional_features: bool,
    ) -> Result<GraphEmbedding, String> {
        let nodes = self.nodes_features(ident2id, use_additional_features)?;
        let n_node = nodes.len() as i64;

        // (parents, child)
        // let mut edge_indices: Vec<(i16, i16)> = vec![];
        let mut parents = vec![];
        let mut childs = vec![];
        let mut rel_pos = vec![];
        let ref_to_index = self.ref_to_index();
        for node in self.iter_bfs() {
            if node.operator() {
                let parent_index = ref_to_index[&RefEquality(node)];
                for (pos, child) in node.childs.iter().enumerate() {
                    let child_index = ref_to_index[&RefEquality(child)];
                    parents.push(parent_index);
                    childs.push(child_index);
                    rel_pos.push(pos as i32)
                }
            }
        }
        let mut target = vec![vec![0.; target_size]; n_node as usize];
        let mut possibility_mask = vec![vec![false; target_size]; n_node as usize];
        for fit in fits.iter() {
            let child = self
                .at(&fit.path)
                .ok_or(format!("Symbol {} has no element at {:?}", self, fit.path))?;
            let index = ref_to_index[&RefEquality(child)] as usize;
            // positive policy should survive after collision
            if fit.policy != Policy::NotTried
                && target[index][fit.rule_id as usize] < Policy::POSITIVE_VALUE
            {
                target[index][fit.rule_id as usize] = fit.policy.value();
            }
            possibility_mask[index][fit.rule_id as usize] = true;
        }

        let edges = [rel_pos, vec![2; parents.len()]].concat();
        let receivers = [parents.clone(), childs.clone()].concat();
        let senders = [childs, parents].concat();
        let n_edge = receivers.len() as i64;
        Ok(GraphEmbedding {
            nodes,
            edges: edges,
            receivers,
            senders,
            n_node,
            n_edge,
            value: if useful { 1 } else { 0 },
            target,
            possibility_mask,
        })
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use crate::context::Context;

    fn fix_dict(dict: HashMap<&str, i16>) -> HashMap<String, i16> {
        dict.into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<HashMap<String, _>>()
    }

    #[test]
    fn index_map_bug_1() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b+1=(a−b)*x").unwrap();
        let ref_to_index: HashMap<RefEquality<Symbol>, i16> = symbol
            .iter_bfs()
            .enumerate()
            .map(|(i, symbol)| (RefEquality(symbol), i as i16))
            .collect();
        let index_map = symbol.index_map(2, symbol.size() as i16, &ref_to_index);
        let expected = vec![
            vec![0, 1, 2, 11],
            vec![1, 3, 4, 0],
            vec![2, 5, 6, 0],
            vec![3, 7, 8, 1],
            vec![4, 11, 11, 1],
            vec![5, 9, 10, 2],
            vec![6, 11, 11, 2],
            vec![7, 11, 11, 3],
            vec![8, 11, 11, 3],
            vec![9, 11, 11, 5],
            vec![10, 11, 11, 5],
            vec![11, 11, 11, 11],
        ];
        assert_eq!(index_map, expected);
    }

    #[test]
    fn embed_full_and_balanced() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let padding = 0;
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "*" => 3,
            "a" => 4,
            "b" => 5,
            "c" => 6,
            "d" => 7,
        };
        let dict = fix_dict(dict);
        let spread = 2;
        let CnnEmbedding {
            embedded,
            index_map,
            value,
            ..
        } = symbol
            .embed_cnn(
                &dict,
                padding,
                spread,
                symbol.depth,
                4,
                &vec![],
                true,
                true,
                false,
                true,
            )
            .unwrap();
        let embedded = embedded.iter().map(|emb| emb[0]).collect::<Vec<i64>>();

        assert_eq!(embedded, vec![1, 2, 3, 4, 5, 6, 7, 0]);

        assert!(index_map.is_some());
        let index_map = index_map.unwrap();
        assert_eq!(index_map.len(), 8);
        assert_eq!(index_map[0], vec![0, 1, 2, 7]); // *=+
        assert_eq!(index_map[1], vec![1, 3, 4, 0]); // a+b
        assert_eq!(index_map[2], vec![2, 5, 6, 0]); // c*d
        assert_eq!(index_map[3], vec![3, 7, 7, 1]); // a
        assert_eq!(index_map[4], vec![4, 7, 7, 1]); // b
        assert_eq!(index_map[5], vec![5, 7, 7, 2]); // c
        assert_eq!(index_map[6], vec![6, 7, 7, 2]); // d
        assert_eq!(index_map[7], vec![7, 7, 7, 7]); // d

        assert_eq!(value, 1);
    }

    #[test]
    fn embed_not_full() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c").unwrap();
        let padding = 0;
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "a" => 3,
            "b" => 4,
            "c" => 5,
        };
        let dict = fix_dict(dict);
        let spread = 3;
        let CnnEmbedding {
            embedded,
            index_map,
            ..
        } = symbol
            .embed_cnn(
                &dict,
                padding,
                spread,
                symbol.depth,
                4,
                &vec![],
                true,
                true,
                false,
                true,
            )
            .unwrap();
        let embedded = embedded.iter().map(|emb| emb[0]).collect::<Vec<i64>>();
        assert!(index_map.is_some());
        let index_map = index_map.unwrap();
        assert_eq!(embedded, vec![1, 2, 5, 3, 4, 0]); // =, +, c, a, b, <PAD>
        assert_eq!(index_map.len(), 6);
        assert_eq!(index_map[0], vec![0, 1, 2, 5, 5]); // *=c
        assert_eq!(index_map[1], vec![1, 3, 4, 5, 0]); // a+b
        assert_eq!(index_map[2], vec![2, 5, 5, 5, 0]); // c
        assert_eq!(index_map[3], vec![3, 5, 5, 5, 1]); // a
        assert_eq!(index_map[4], vec![4, 5, 5, 5, 1]); // b
        assert_eq!(index_map[5], vec![5, 5, 5, 5, 5]); // padding
    }

    #[test]
    fn embed_labels() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let padding = 0;
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "*" => 3,
            "a" => 4,
            "b" => 5,
            "c" => 6,
            "d" => 7,
        };
        let dict = fix_dict(dict);
        let spread = 2;
        let CnnEmbedding { label, target, .. } = symbol
            .embed_cnn(
                &dict,
                padding,
                spread,
                symbol.depth,
                3,
                &vec![
                    FitInfo {
                        rule_id: 1,
                        path: vec![0, 0],
                        policy: Policy::Positive,
                    },
                    FitInfo {
                        rule_id: 2,
                        path: vec![0, 1],
                        policy: Policy::Negative,
                    },
                ],
                true,
                true,
                false,
                true,
            )
            .unwrap();

        assert_eq!(label, vec![0, 0, 0, 1, 2, 0, 0, 0,]);
        assert_eq!(
            target,
            vec![
                vec![0.0, 0.0, 0.0,],
                vec![0.0, 0.0, 0.0,],
                vec![0.0, 0.0, 0.0,],
                vec![0.0, 1.0, 0.0,],
                vec![0.0, 0.0, -1.0,],
                vec![0.0, 0.0, 0.0,],
                vec![0.0, 0.0, 0.0,],
                vec![0.0, 0.0, 0.0,],
            ]
        )
    }

    #[test]
    fn positional_encoding_full() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let spread = 2;
        // = 4 2
        // + 2 1
        // * 6 3
        // a 1 0
        // b 3 0
        // c 5 0
        // d 7 0

        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        let expected = vec![vec![4, 2, 6, 1, 3, 5, 7], vec![2, 1, 3, 0, 0, 0, 0]];
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn positional_encoding_simple() {
        let context = Context::standard();
        let symbol = Symbol::parse(&context, "a+b=c").unwrap();

        let spread = 2;
        // = 4 2
        // + 2 1
        // c 6 3
        // a 1 0
        // b 3 0
        let expected = vec![vec![4, 2, 6, 1, 3], vec![2, 1, 3, 0, 0]];
        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn positional_encoding_deep() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+(b-d)=c").unwrap();
        let spread = 2;
        // =  8  4  2
        // +  4  2  1
        // c 12  6  3
        // a  2  1  0
        // -  6  3  0
        // b  5  0  0
        // d  7  0  0

        let positional_encoding = symbol.positional_encoding(spread, symbol.depth);
        let expected = vec![
            vec![8, 4, 12, 2, 6, 5, 7],
            vec![4, 2, 6, 1, 3, 0, 0],
            vec![2, 1, 3, 0, 0, 0, 0],
        ];
        assert_eq!(&positional_encoding, &expected);
    }

    #[test]
    fn embed_graph() {
        let context = Context::standard();

        let symbol = Symbol::parse(&context, "a+b=c*d").unwrap();
        let dict = hashmap! {
            "=" => 1,
            "+" => 2,
            "*" => 3,
            "a" => 4,
            "b" => 5,
            "c" => 6,
            "d" => 7,
        };
        let dict = fix_dict(dict);
        let embedding = symbol.embed_graph(
            &dict,
            3,
            &vec![
                FitInfo {
                    rule_id: 1,
                    path: vec![0, 0],
                    policy: Policy::Positive,
                },
                FitInfo {
                    rule_id: 2,
                    path: vec![0, 1],
                    policy: Policy::Negative,
                },
            ],
            true,
            true,
        );
        let actual = embedding.unwrap();

        let expected = GraphEmbedding {
            nodes: vec![
                vec![1, 1, 1, 0], // root
                vec![2, 1, 1, 0], // +
                vec![3, 1, 1, 0], // *
                vec![4, 0, 0, 0], // a
                vec![5, 0, 0, 0], // b
                vec![6, 0, 0, 0], // c
                vec![7, 0, 0, 0], // d
            ],
            edges: vec![0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2],
            receivers: vec![0, 0, 1, 1, 2, 2, 1, 2, 3, 4, 5, 6],
            senders: vec![1, 2, 3, 4, 5, 6, 0, 0, 1, 1, 2, 2],
            target: vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, -1.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ],
            possibility_mask: vec![
                vec![false, false, false],
                vec![false, false, false],
                vec![false, false, false],
                vec![false, true, false],
                vec![false, false, true],
                vec![false, false, false],
                vec![false, false, false],
            ],
            n_node: 7,
            n_edge: 12,
            value: 1,
        };

        assert_eq!(actual, expected)
    }
}
