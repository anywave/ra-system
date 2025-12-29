//! AccessResult type + gating logic from spec Section 4.
//!
//! Access gating determines whether a fragment/signal can emerge based on
//! coherence and consent levels.
//!
//! From Section 4 of ra_integration_spec.md:
//!
//! ```text
//! AccessLevel(user_coherence, fragment_rac) → {FullAccess, PartialAccess(α), Blocked}
//!
//! threshold(R_f) = RAC(R_f) / RAC₁
//! C_floor = φ_green / Ankh ≈ 0.3183
//! C_ceiling = 1.0
//!
//! If C_u ≥ threshold(R_f):
//!     return FullAccess
//! Else If C_u ≥ C_floor:
//!     α = (C_u - C_floor) / (threshold(R_f) - C_floor)
//!     return PartialAccess(α)
//! Else:
//!     return Blocked
//! ```

use serde::{Deserialize, Serialize};
use crate::constants::{ANKH, GREEN_PHI};
use crate::rac::{RacLevel, rac_value_normalized};
use crate::repitans::Repitan;

/// Coherence floor: φ_green / Ankh ≈ 0.3183
pub const COHERENCE_FLOOR: f64 = GREEN_PHI / ANKH;

/// Coherence ceiling: 1.0
pub const COHERENCE_CEILING: f64 = 1.0;

/// Result of access gating check
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum AccessResult {
    /// Complete emergence allowed
    FullAccess,
    /// Partial emergence with intensity α ∈ (0, 1)
    PartialAccess(f64),
    /// No emergence allowed
    Blocked,
}

impl AccessResult {
    /// Check if result is FullAccess
    pub fn is_full_access(&self) -> bool {
        matches!(self, AccessResult::FullAccess)
    }

    /// Check if result is PartialAccess
    pub fn is_partial_access(&self) -> bool {
        matches!(self, AccessResult::PartialAccess(_))
    }

    /// Check if result is Blocked
    pub fn is_blocked(&self) -> bool {
        matches!(self, AccessResult::Blocked)
    }

    /// Extract alpha value: 1.0 for FullAccess, α for PartialAccess, 0.0 for Blocked
    pub fn alpha(&self) -> f64 {
        match self {
            AccessResult::FullAccess => 1.0,
            AccessResult::PartialAccess(a) => *a,
            AccessResult::Blocked => 0.0,
        }
    }
}

/// Get threshold for a RAC level (normalized to RAC1)
pub fn rac_threshold(rac: RacLevel) -> f64 {
    rac_value_normalized(rac)
}

/// Core gating function from spec Section 4.1
/// Determines access level based on user coherence and fragment RAC requirement
pub fn access_level(user_coherence: f64, fragment_rac: RacLevel) -> AccessResult {
    let threshold = rac_threshold(fragment_rac);

    if user_coherence >= threshold {
        AccessResult::FullAccess
    } else if user_coherence >= COHERENCE_FLOOR {
        let alpha = (user_coherence - COHERENCE_FLOOR) / (threshold - COHERENCE_FLOOR);
        AccessResult::PartialAccess(alpha.clamp(0.0, 1.0))
    } else {
        AccessResult::Blocked
    }
}

/// Simple check if access is allowed (not Blocked)
pub fn can_access(coherence: f64, rac: RacLevel) -> bool {
    !access_level(coherence, rac).is_blocked()
}

/// Calculate effective coherence given access result
/// Maps FullAccess → 1.0, PartialAccess(α) → α, Blocked → 0.0
pub fn effective_coherence(result: AccessResult) -> f64 {
    result.alpha()
}

/// Calculate partial emergence within a Repitan band
/// From spec Section 4.4
pub fn partial_emergence(current_band: &Repitan, alpha: f64) -> f64 {
    let band_low = current_band.value();
    let band_high = current_band.next().value();
    band_low + alpha * (band_high - band_low)
}

/// Weights for resonance score calculation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ResonanceWeights {
    /// θ alignment weight
    pub theta: f64,
    /// φ access weight
    pub phi: f64,
    /// h harmonic match weight
    pub harmonic: f64,
    /// r intensity weight
    pub radius: f64,
}

impl Default for ResonanceWeights {
    /// Default weights from spec Section 5.3
    /// w_θ = 0.3, w_φ = 0.4, w_h = 0.2, w_r = 0.1
    fn default() -> Self {
        ResonanceWeights {
            theta: 0.3,
            phi: 0.4,
            harmonic: 0.2,
            radius: 0.1,
        }
    }
}

/// Calculate composite resonance score
/// resonance = w_θ × θ_match + w_φ × φ_access + w_h × h_match + w_r × r_intensity
pub fn resonance_score(
    weights: &ResonanceWeights,
    theta_match: f64,
    phi_access: f64,
    harmonic_match: f64,
    intensity: f64,
) -> f64 {
    weights.theta * theta_match
        + weights.phi * phi_access
        + weights.harmonic * harmonic_match
        + weights.radius * intensity
}

/// Verify Invariant R3: Coherence bounds are [0, 1]
pub fn verify_coherence_bounds() -> bool {
    COHERENCE_FLOOR >= 0.0 && COHERENCE_FLOOR < 1.0 && COHERENCE_CEILING == 1.0
}

/// Verify coherence floor calculation: φ_green / Ankh
pub fn verify_coherence_floor() -> bool {
    (COHERENCE_FLOOR - GREEN_PHI / ANKH).abs() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_access_at_threshold() {
        // At RAC1 normalized threshold (1.0), should get full access
        let result = access_level(1.0, RacLevel::RAC1);
        assert!(result.is_full_access());
    }

    #[test]
    fn test_full_access_above_threshold() {
        // For RAC6 (lower threshold), 0.8 coherence should be full access
        let result = access_level(0.8, RacLevel::RAC6);
        assert!(result.is_full_access());
    }

    #[test]
    fn test_blocked_below_floor() {
        let result = access_level(0.1, RacLevel::RAC1);
        assert!(result.is_blocked());
    }

    #[test]
    fn test_partial_access() {
        // Between floor and threshold should be partial
        let coherence = 0.5;
        let result = access_level(coherence, RacLevel::RAC1);
        assert!(result.is_partial_access());

        let alpha = result.alpha();
        assert!(alpha > 0.0 && alpha < 1.0);
    }

    #[test]
    fn test_zero_coherence_blocked() {
        for level in RacLevel::all() {
            let result = access_level(0.0, level);
            assert!(result.is_blocked(), "Zero coherence should always block");
        }
    }

    #[test]
    fn test_full_coherence_access() {
        for level in RacLevel::all() {
            let result = access_level(1.0, level);
            assert!(result.is_full_access(), "Full coherence should always grant access");
        }
    }

    #[test]
    fn test_alpha_range() {
        for level in RacLevel::all() {
            for c in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] {
                let result = access_level(c, level);
                let alpha = result.alpha();
                assert!((0.0..=1.0).contains(&alpha), "Alpha must be in [0, 1]");
            }
        }
    }

    #[test]
    fn test_partial_emergence() {
        let r = Repitan::NINTH;
        let alpha = 0.5;
        let emergence = partial_emergence(&r, alpha);

        // Should be between current and next band values
        assert!(emergence >= r.value());
        assert!(emergence <= r.next().value());
    }

    #[test]
    fn test_resonance_score() {
        let weights = ResonanceWeights::default();
        let score = resonance_score(&weights, 1.0, 1.0, 1.0, 1.0);

        // With all 1.0 inputs and weights summing to 1.0, score should be 1.0
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_invariant_r3() {
        assert!(verify_coherence_bounds());
    }

    #[test]
    fn test_coherence_floor_derivation() {
        assert!(verify_coherence_floor());
    }
}
