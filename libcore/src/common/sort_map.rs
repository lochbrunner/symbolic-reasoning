#[must_use = "iterators are lazy and do nothing unless consumed"]
pub struct SortMap<I, F> {
    iter: I,
    f: F,
}

impl<I, F> SortMap<I, F> {
    pub(super) fn new(iter: I, f: F) -> SortMap<I, F> {
        SortMap { iter, f }
    }
}

pub trait SortMapable {
    #[inline]
    fn sort_map<B, F>(self, f: F) -> SortMap<Self, F>
    where
        Self: Sized,
        F: FnMut(Self::Item) -> B,
    {
        SortMap::new(self, f)
    }
}

impl<B, I: Iterator, F> Iterator for SortMap<I, F>
where
    F: FnMut(I::Item) -> B,
{
    type item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        self.iter.next().map(&mut self.f)
    }
}

fn partition<B, F>(self, f: F) -> (B, B)
where
    Self: Sized,
    B: Default + Extend<Self::Item>,
    F: FnMut(&Self::Item) -> bool
{
    #[inline]
    fn extend<'a, T, B: Extend<T>>(
        mut f: impl FnMut(&T) -> bool + 'a,
        left: &'a mut B,
        right: &'a mut B,
    ) -> impl FnMut(T) + 'a {
        move |x| {
            if f(&x) {
                left.extend(Some(x));
            } else {
                right.extend(Some(x));
            }
        }
    }

    let mut left: B = Default::default();
    let mut right: B = Default::default();

    self.for_each(extend(f, &mut left, &mut right));

    (left, right)
}