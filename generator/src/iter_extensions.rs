use std::iter::{Filter, Iterator, Map};
use std::slice::Iter;

pub struct PickStateIter<L>
where
    L: Iterator + Sized,
{
    left_iter: L,
    strategy: Strategy,
    len: usize,
    cursor: f32,
}

pub enum Strategy {
    // Random(u32),
    Uniform(usize),
}

pub trait PickTraitIt: Iterator {
    fn pick(&mut self, len: usize, strategy: Strategy) -> PickStateIter<Self>
    where
        Self: Sized + Clone,
    {
        PickStateIter {
            left_iter: self.clone(),
            strategy,
            len,
            cursor: 0.0,
        }
    }
}

impl<L> Iterator for PickStateIter<L>
where
    L: Iterator + Sized,
{
    type Item = L::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self.strategy {
            Strategy::Uniform(count) => {
                let step_size = (self.len as f32) / (count as f32);
                let desired_cursor = self.cursor + step_size;
                let item = self.left_iter.next();
                self.cursor += 1.0;

                while self.cursor < desired_cursor {
                    self.left_iter.next();
                    self.cursor += 1.0;
                }
                item
            }
        }
    }
}

impl<'a, T> PickTraitIt for Iter<'a, T> {}
impl<B, I: Iterator, F> PickTraitIt for Map<I, F> where F: FnMut(I::Item) -> B {}
impl<I: Iterator, P> PickTraitIt for Filter<I, P> where P: FnMut(&I::Item) -> bool {}

pub struct PickStateVec<'a, T> {
    left_slice: &'a Vec<T>,
    strategy: Strategy,
    cursor: f32,
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
        }
    }
}

pub trait PickTraitVec<T> {
    fn pick(&self, strategy: Strategy) -> PickStateVec<T>;
}

impl<T> PickTraitVec<T> for Vec<T> {
    fn pick<'a>(&'a self, strategy: Strategy) -> PickStateVec<'a, T> {
        PickStateVec {
            left_slice: self,
            strategy,
            cursor: 0.0,
        }
    }
}

#[cfg(test)]
mod specs {
    use super::*;

    #[test]
    fn pick_uniform_half_of_even() {
        let input = [1, 2, 3, 4];
        let actual = input
            .iter()
            .pick(input.len(), Strategy::Uniform(2))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 3];
        assert_eq!(actual, expected);
    }

    #[test]
    fn pick_uniform_half_of_odd() {
        let input = [1, 2, 3];
        let actual = input
            .iter()
            .pick(input.len(), Strategy::Uniform(2))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 3];
        assert_eq!(actual, expected);
    }

    #[test]
    fn pick_uniform_full() {
        let input = [1, 2, 3];
        let actual = input
            .iter()
            .pick(input.len(), Strategy::Uniform(3))
            .cloned()
            .collect::<Vec<i32>>();

        let expected = [1, 2, 3];
        assert_eq!(actual, expected);
    }

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
}
