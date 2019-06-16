use rand::prelude::*;
use std::collections::HashSet;
use std::iter::Iterator;
// use std::iter::{Filter, Iterator, Map};
// use std::slice::Iter;

// pub struct PickStateIter<L>
// where
//     L: Iterator + Sized,
// {
//     left_iter: L,
//     count: usize,
//     len: usize,
//     cursor: f32,
// }

// pub trait PickTraitIt: Iterator {
//     fn pick(&mut self, len: usize, count: usize) -> PickStateIter<Self>
//     where
//         Self: Sized + Clone,
//     {
//         PickStateIter {
//             left_iter: self.clone(),
//             count,
//             len,
//             cursor: 0.0,
//         }
//     }
// }

// impl<L> Iterator for PickStateIter<L>
// where
//     L: Iterator + Sized,
// {
//     type Item = L::Item;

//     #[inline]
//     fn next(&mut self) -> Option<Self::Item> {
//         let step_size = (self.len as f32) / (self.count as f32);
//         let desired_cursor = self.cursor + step_size;
//         let item = self.left_iter.next();
//         self.cursor += 1.0;

//         while self.cursor < desired_cursor {
//             self.left_iter.next();
//             self.cursor += 1.0;
//         }
//         item
//     }
// }

// impl<'a, T> PickTraitIt for Iter<'a, T> {}
// impl<B, I: Iterator, F> PickTraitIt for Map<I, F> where F: FnMut(I::Item) -> B {}
// impl<I: Iterator, P> PickTraitIt for Filter<I, P> where P: FnMut(&I::Item) -> bool {}

pub enum Strategy {
    /// Should they be unique?
    Random(bool),
    Uniform(usize),
}

pub struct PickStateVec<'a, T> {
    left_slice: &'a Vec<T>,
    strategy: Strategy,
    cursor: f32,
    seen: HashSet<usize>,
    rng: ThreadRng, // No need for rand::SeedableRng yet
}

impl<'a, T> Iterator for PickStateVec<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.strategy {
            Strategy::Uniform(count) => {
                let step_size = (self.left_slice.len() as f32 - 0.0) / (count as f32);
                let item = self.left_slice.get(self.cursor.ceil() as usize);
                self.cursor += step_size;
                item
            }
            Strategy::Random(false) => {
                let index = self.rng.gen_range(0, self.left_slice.len());
                self.left_slice.get(index)
            }
            Strategy::Random(true) => {
                if self.seen.len() >= self.left_slice.len() {
                    None
                } else {
                    let mut index = self.rng.gen_range(0, self.left_slice.len());
                    while self.seen.contains(&index) {
                        index = self.rng.gen_range(0, self.left_slice.len());
                    }
                    self.seen.insert(index);
                    self.left_slice.get(index)
                }
            }
        }
    }
}

pub trait PickTraitVec<T> {
    fn pick(&self, strategy: Strategy) -> PickStateVec<T>;
}

impl<T> PickTraitVec<T> for Vec<T> {
    fn pick(&self, strategy: Strategy) -> PickStateVec<T> {
        PickStateVec {
            left_slice: self,
            strategy,
            seen: HashSet::new(),
            cursor: 0.0,
            rng: rand::thread_rng(),
        }
    }
}

#[cfg(test)]
mod specs {
    use super::*;
    use std::collections::HashSet;

    // #[test]
    // fn pick_uniform_half_of_even() {
    //     let input = [1, 2, 3, 4];
    //     let actual = input
    //         .iter()
    //         .pick(input.len(), 2)
    //         .cloned()
    //         .collect::<Vec<i32>>();

    //     let expected = [1, 3];
    //     assert_eq!(actual, expected);
    // }

    // #[test]
    // fn pick_uniform_half_of_odd() {
    //     let input = [1, 2, 3];
    //     let actual = input
    //         .iter()
    //         .pick(input.len(), 2)
    //         .cloned()
    //         .collect::<Vec<i32>>();

    //     let expected = [1, 3];
    //     assert_eq!(actual, expected);
    // }

    // #[test]
    // fn pick_uniform_full() {
    //     let input = [1, 2, 3];
    //     let actual = input
    //         .iter()
    //         .pick(input.len(), 3)
    //         .cloned()
    //         .collect::<Vec<i32>>();

    //     let expected = [1, 2, 3];
    //     assert_eq!(actual, expected);
    // }

    #[test]
    fn pick_vec_uniform_half_of_even() {
        let input = vec![1, 2, 3, 4];
        let actual = input
            .pick(Strategy::Uniform(2))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 3];
        assert_eq!(actual, expected);
    }

    #[test]
    fn pick_vec_uniform_half_of_odd() {
        let input = vec![1, 2, 3, 4, 5];
        let actual = input
            .pick(Strategy::Uniform(3))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 3, 5];
        assert_eq!(actual, expected);
    }

    #[test]
    fn pick_vec_uniform_full() {
        let input = vec![1, 2, 3];
        let actual = input
            .pick(Strategy::Uniform(3))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 2, 3];
        assert_eq!(actual, expected);
    }

    #[test]
    fn pick_random_non_unique() {
        let input = vec![1, 2, 3, 4, 5];
        let actual = input
            .pick(Strategy::Random(false))
            .take(3)
            .cloned()
            .collect::<Vec<i32>>();

        assert_eq!(actual.len(), 3);
    }

    #[test]
    fn pick_random_unique_some() {
        let input = vec![1, 2, 3, 4, 5];
        let actual = input
            .pick(Strategy::Random(true))
            .take(5)
            .cloned()
            .collect::<Vec<i32>>();

        assert_eq!(actual.len(), 5);
        let unique: HashSet<_> = actual.into_iter().collect();
        assert_eq!(unique.len(), 5);
    }

    #[test]
    fn pick_random_unique_all() {
        let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let actual = input
            .pick(Strategy::Random(true))
            .cloned()
            .collect::<Vec<i32>>();

        assert_eq!(actual.len(), input.len());
        let unique: HashSet<_> = actual.into_iter().collect();
        assert_eq!(unique.len(), input.len());
    }
}
