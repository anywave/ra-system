//! RacLevel enum (RAC1..RAC6) with validation.
//!
//! Resonant Access Constants (RACs) represent access sensitivity levels.
//! RAC1 is the highest (least restrictive), RAC6 is the lowest (most restrictive).
//!
//! Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0

use serde::{Deserialize, Serialize};
use crate::constants::ANKH;

/// The six Resonant Access Constant levels
/// RAC1 is highest access (least restricted), RAC6 is lowest (most restricted)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RacLevel {
    RAC1 = 1,
    RAC2 = 2,
    RAC3 = 3,
    RAC4 = 4,
    RAC5 = 5,
    RAC6 = 6,
}

impl RacLevel {
    /// Get RAC level from numeric index (1-6)
    pub fn from_level(n: u8) -> Option<Self> {
        match n {
            1 => Some(RacLevel::RAC1),
            2 => Some(RacLevel::RAC2),
            3 => Some(RacLevel::RAC3),
            4 => Some(RacLevel::RAC4),
            5 => Some(RacLevel::RAC5),
            6 => Some(RacLevel::RAC6),
            _ => None,
        }
    }

    /// Get the numeric level (1-6)
    pub fn level(&self) -> u8 {
        *self as u8
    }

    /// Get all RAC levels in order
    pub fn all() -> [RacLevel; 6] {
        [
            RacLevel::RAC1,
            RacLevel::RAC2,
            RacLevel::RAC3,
            RacLevel::RAC4,
            RacLevel::RAC5,
            RacLevel::RAC6,
        ]
    }
}

/// RAC value in Red Rams (must be 0 < x < 1)
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct RacValue(f64);

impl RacValue {
    /// Create a new RacValue with validation
    pub fn new(value: f64) -> Option<Self> {
        if value > 0.0 && value < 1.0 {
            Some(RacValue(value))
        } else {
            None
        }
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Get the RAC value in Red Rams for a given level
pub fn rac_value(level: RacLevel) -> RacValue {
    RacValue(match level {
        RacLevel::RAC1 => 0.6361725,      // Ankh / 8
        RacLevel::RAC2 => 0.628318519,    // 2π/10 approximation
        RacLevel::RAC3 => 0.57255525,     // φ × Hunab × 1/3
        RacLevel::RAC4 => 0.523598765,    // π/6 approximation
        RacLevel::RAC5 => 0.4580442,      // Ankh × 9 / 100
        RacLevel::RAC6 => 0.3998594565,   // RAC lattice terminus
    })
}

/// Get the RAC value in meters for a given level
pub fn rac_value_meters(level: RacLevel) -> f64 {
    match level {
        RacLevel::RAC1 => 0.639591666,
        RacLevel::RAC2 => 0.631695473,
        RacLevel::RAC3 => 0.5756325,
        RacLevel::RAC4 => 0.526412894,
        RacLevel::RAC5 => 0.460506,
        RacLevel::RAC6 => 0.4020085371,
    }
}

/// Get the RAC value normalized to RAC1 (for threshold calculations)
/// RAC1 normalized = 1.0
pub fn rac_value_normalized(level: RacLevel) -> f64 {
    rac_value(level).value() / rac_value(RacLevel::RAC1).value()
}

/// Pyramid base divided by each RAC yields key numbers
pub fn pyramid_division(level: RacLevel) -> f64 {
    match level {
        RacLevel::RAC1 => 360.0,      // Circle degrees
        RacLevel::RAC2 => 364.5,      // Balmer constant
        RacLevel::RAC3 => 400.0,
        RacLevel::RAC4 => 437.4,      // 27 × φ_green
        RacLevel::RAC5 => 500.0,
        RacLevel::RAC6 => 572.756493, // 1.125 × Green Ankh
    }
}

/// Verify Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
pub fn verify_rac_ordering() -> bool {
    let levels = RacLevel::all();
    let values: Vec<f64> = levels.iter().map(|l| rac_value(*l).value()).collect();

    // Check descending order
    let descending = values.windows(2).all(|w| w[0] > w[1]);
    // Check all positive
    let all_positive = values.iter().all(|&v| v > 0.0);

    descending && all_positive
}

/// Verify Invariant R1: 0 < RAC(i) < 1 for all i ∈ [1, 6]
pub fn verify_rac_range() -> bool {
    RacLevel::all().iter().all(|l| {
        let v = rac_value(*l).value();
        v > 0.0 && v < 1.0
    })
}

/// Verify Invariant I2: RAC1 = Ankh / 8
pub fn verify_rac1_derivation() -> bool {
    let computed = ANKH / 8.0;
    (rac_value(RacLevel::RAC1).value() - computed).abs() < 0.0001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rac_from_level() {
        assert_eq!(RacLevel::from_level(1), Some(RacLevel::RAC1));
        assert_eq!(RacLevel::from_level(6), Some(RacLevel::RAC6));
        assert_eq!(RacLevel::from_level(0), None);
        assert_eq!(RacLevel::from_level(7), None);
    }

    #[test]
    fn test_rac_value_range() {
        for level in RacLevel::all() {
            let v = rac_value(level).value();
            assert!(v > 0.0 && v < 1.0, "RAC{} value {} out of range", level.level(), v);
        }
    }

    #[test]
    fn test_invariant_o1() {
        assert!(verify_rac_ordering());
    }

    #[test]
    fn test_invariant_r1() {
        assert!(verify_rac_range());
    }

    #[test]
    fn test_invariant_i2() {
        assert!(verify_rac1_derivation());
    }

    #[test]
    fn test_rac_normalized() {
        assert!((rac_value_normalized(RacLevel::RAC1) - 1.0).abs() < 1e-10);
        assert!(rac_value_normalized(RacLevel::RAC6) < 1.0);
    }
}
