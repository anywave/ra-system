//! Repitan type with smart constructor validating range [1, 27].
//!
//! Repitans represent the 27 semantic sectors of the Ra System.
//! Each Repitan(n) = n/27 for n ∈ [1, 27].

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Error type for Repitan validation
#[derive(Debug, Error, PartialEq)]
pub enum RepitanError {
    #[error("Repitan index must be in range [1, 27], got {0}")]
    InvalidIndex(i32),
}

/// A validated Repitan (semantic sector index)
///
/// Invariants:
/// - Index is in range [1, 27]
/// - Value = index / 27
/// - O3: 0 < Repitan(n) ≤ 1 for all n
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Repitan(u8);

impl Repitan {
    /// Create a new Repitan with validation
    ///
    /// # Errors
    /// Returns `RepitanError::InvalidIndex` if n is not in [1, 27]
    pub fn new(n: i32) -> Result<Self, RepitanError> {
        if n >= 1 && n <= 27 {
            Ok(Repitan(n as u8))
        } else {
            Err(RepitanError::InvalidIndex(n))
        }
    }

    /// Get the index (1-27)
    pub fn index(&self) -> u8 {
        self.0
    }

    /// Get the Repitan value (n/27)
    /// Invariant I4: Repitan(n) = n/27
    pub fn value(&self) -> f64 {
        f64::from(self.0) / 27.0
    }

    /// Get theta angle in degrees (0-360)
    pub fn theta(&self) -> f64 {
        self.value() * 360.0
    }

    /// Get theta angle in radians
    pub fn theta_radians(&self) -> f64 {
        self.value() * std::f64::consts::TAU
    }

    /// Get the next Repitan (wraps from 27 to 1)
    pub fn next(&self) -> Repitan {
        if self.0 == 27 {
            Repitan(1)
        } else {
            Repitan(self.0 + 1)
        }
    }

    /// Get the previous Repitan (wraps from 1 to 27)
    pub fn prev(&self) -> Repitan {
        if self.0 == 1 {
            Repitan(27)
        } else {
            Repitan(self.0 - 1)
        }
    }

    /// Calculate angular distance to another Repitan
    /// Accounts for circular wraparound (max distance is 13)
    pub fn distance(&self, other: &Repitan) -> u8 {
        let d = (self.0 as i16 - other.0 as i16).unsigned_abs() as u8;
        d.min(27 - d)
    }

    /// First Repitan: n=1, value=1/27 (Fine Structure root)
    pub const FIRST: Repitan = Repitan(1);

    /// Ninth Repitan: n=9, value=9/27=1/3
    pub const NINTH: Repitan = Repitan(9);

    /// Unity Repitan: n=27, value=27/27=1
    pub const UNITY: Repitan = Repitan(27);
}

/// Convert theta angle (degrees) to nearest Repitan
pub fn repitan_from_theta(theta: f64) -> Repitan {
    let normalized = theta.rem_euclid(360.0) / 360.0;
    let n = (normalized * 27.0).round() as i32;
    let n = n.clamp(1, 27);
    Repitan(n as u8)
}

/// All 27 Repitans in order
pub fn all_repitans() -> [Repitan; 27] {
    let mut arr = [Repitan(1); 27];
    for i in 0..27 {
        arr[i] = Repitan((i + 1) as u8);
    }
    arr
}

/// Verify Invariant I4: Repitan(n) = n/27 for all n ∈ [1, 27]
pub fn verify_repitan_invariant() -> bool {
    (1..=27).all(|n| {
        let r = Repitan::new(n).unwrap();
        (r.value() - f64::from(n as i32) / 27.0).abs() < 1e-10
    })
}

/// Verify Invariant O3: For all n: 0 < Repitan(n) ≤ 1
pub fn verify_repitan_range_invariant() -> bool {
    (1..=27).all(|n| {
        let r = Repitan::new(n).unwrap();
        let v = r.value();
        v > 0.0 && v <= 1.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_repitan() {
        for n in 1..=27 {
            let r = Repitan::new(n).unwrap();
            assert_eq!(r.index(), n as u8);
        }
    }

    #[test]
    fn test_invalid_repitan() {
        assert_eq!(Repitan::new(0), Err(RepitanError::InvalidIndex(0)));
        assert_eq!(Repitan::new(28), Err(RepitanError::InvalidIndex(28)));
        assert_eq!(Repitan::new(-1), Err(RepitanError::InvalidIndex(-1)));
    }

    #[test]
    fn test_repitan_value() {
        assert!((Repitan::FIRST.value() - 1.0/27.0).abs() < 1e-10);
        assert!((Repitan::NINTH.value() - 1.0/3.0).abs() < 1e-10);
        assert!((Repitan::UNITY.value() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_repitan_next_prev() {
        assert_eq!(Repitan::FIRST.next(), Repitan(2));
        assert_eq!(Repitan::UNITY.next(), Repitan::FIRST);
        assert_eq!(Repitan::FIRST.prev(), Repitan::UNITY);
    }

    #[test]
    fn test_repitan_distance() {
        assert_eq!(Repitan::FIRST.distance(&Repitan::FIRST), 0);
        assert_eq!(Repitan::FIRST.distance(&Repitan::new(14).unwrap()), 13);
        assert_eq!(Repitan::FIRST.distance(&Repitan::UNITY), 1);
    }

    #[test]
    fn test_invariant_i4() {
        assert!(verify_repitan_invariant());
    }

    #[test]
    fn test_invariant_o3() {
        assert!(verify_repitan_range_invariant());
    }
}
