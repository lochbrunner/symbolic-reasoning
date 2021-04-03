#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
pub enum Policy {
    Positive,
    Negative,
    NotTried,
}

impl Policy {
    pub const POSITIVE_VALUE: f32 = 1.;
    pub const NEGATIVE_VALUE: f32 = -1.;
    pub const NOT_TRIED_VALUE: f32 = 0.;

    pub fn value(&self) -> f32 {
        match self {
            Self::Positive => Self::POSITIVE_VALUE,
            Self::Negative => Self::NEGATIVE_VALUE,
            Self::NotTried => Self::NOT_TRIED_VALUE,
        }
    }

    pub fn new(positive: bool) -> Self {
        if positive {
            Self::Positive
        } else {
            Self::Negative
        }
    }
}

// Clone is needed as long sort_map is not available
#[derive(Deserialize, Serialize, Debug, PartialEq, Eq, Hash, Clone)]
pub struct FitInfo {
    /// Starting with 1 for better embedding
    pub rule_id: u32,
    pub path: Vec<usize>,
    /// For positive and negative samples
    pub policy: Policy,
}

#[derive(PartialEq, Debug)]
pub enum FitCompare {
    Unrelated,
    Contradicting,
    Matching,
}

impl FitInfo {
    pub fn compare(&self, other: &Self) -> FitCompare {
        if self.path != other.path || self.rule_id != other.rule_id {
            FitCompare::Unrelated
        } else if self.policy == other.policy {
            FitCompare::Matching
        } else {
            FitCompare::Contradicting
        }
    }

    pub fn compare_many<'a, T>(
        &self,
        others: &'a [T],
        unpack: fn(&'a T) -> &'a Self,
    ) -> FitCompare {
        for other in others.iter().map(unpack) {
            match self.compare(other) {
                FitCompare::Contradicting => {
                    return FitCompare::Contradicting;
                }
                FitCompare::Matching => {
                    return FitCompare::Matching;
                }
                _ => {}
            }
        }
        FitCompare::Unrelated
    }
}
