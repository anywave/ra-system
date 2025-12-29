//! θ/φ/r coordinate functions for Ra System dimensional mapping.
//!
//! Coordinate transforms:
//! - θ (theta): Semantic sector ← 27 Repitans
//! - φ (phi): Access sensitivity ← 6 RACs
//! - h (harmonic): Coherence depth ← 5 Omega formats
//! - r (radius): Emergence intensity ← Ankh-normalized scalar

use serde::{Deserialize, Serialize};
use crate::constants::ANKH;
use crate::repitans::Repitan;
use crate::rac::RacLevel;
use crate::omega::OmegaFormat;

/// A complete Ra coordinate in 4-dimensional space
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RaCoordinate {
    /// Semantic sector (1-27)
    pub theta: Repitan,
    /// Access sensitivity (RAC1-RAC6)
    pub phi: RacLevel,
    /// Coherence depth (Red-Blue)
    pub harmonic: OmegaFormat,
    /// Ankh-normalized intensity [0,1]
    pub radius: f64,
}

impl RaCoordinate {
    /// Smart constructor for RaCoordinate with radius validation
    pub fn new(
        theta: Repitan,
        phi: RacLevel,
        harmonic: OmegaFormat,
        radius: f64,
    ) -> Option<Self> {
        if radius >= 0.0 && radius <= 1.0 {
            Some(RaCoordinate { theta, phi, harmonic, radius })
        } else {
            None
        }
    }

    /// Check if the coordinate is valid
    pub fn is_valid(&self) -> bool {
        self.radius >= 0.0 && self.radius <= 1.0
    }
}

/// Convert Repitan to theta angle in degrees (0-360)
pub fn theta_from_repitan(r: &Repitan) -> f64 {
    r.theta()
}

/// Convert RacLevel to phi value (0-255 encoded)
pub fn phi_from_rac(rac: RacLevel) -> u8 {
    match rac {
        RacLevel::RAC1 => 0,    // Least restrictive
        RacLevel::RAC2 => 43,
        RacLevel::RAC3 => 85,
        RacLevel::RAC4 => 128,
        RacLevel::RAC5 => 170,
        RacLevel::RAC6 => 255,  // Most restrictive
    }
}

/// Convert phi value (0-255) to RacLevel
pub fn rac_from_phi(phi: u8) -> RacLevel {
    match phi {
        0..=21   => RacLevel::RAC1,
        22..=63  => RacLevel::RAC2,
        64..=106 => RacLevel::RAC3,
        107..=148 => RacLevel::RAC4,
        149..=212 => RacLevel::RAC5,
        _ => RacLevel::RAC6,
    }
}

/// Convert OmegaFormat to harmonic index (0-4)
pub fn harmonic_from_omega(fmt: OmegaFormat) -> u8 {
    fmt.harmonic()
}

/// Convert harmonic index (0-4) to OmegaFormat
pub fn omega_from_harmonic(h: u8) -> Option<OmegaFormat> {
    OmegaFormat::from_harmonic(h)
}

/// Normalize a raw radius value to [0, 1] using Ankh
/// r_normalized = r_raw / Ankh
pub fn normalize_radius(r: f64) -> f64 {
    (r / ANKH).clamp(0.0, 1.0)
}

/// Denormalize a radius value from [0, 1] to raw scale
/// r_raw = r_normalized × Ankh
pub fn denormalize_radius(r: f64) -> f64 {
    r * ANKH
}

/// Calculate weighted distance between two coordinates
/// Returns value in [0, 1] where 0 = identical, 1 = maximally different
pub fn coordinate_distance(c1: &RaCoordinate, c2: &RaCoordinate) -> f64 {
    let theta_dist = f64::from(c1.theta.distance(&c2.theta)) / 13.5;
    let phi_dist = f64::from((phi_from_rac(c1.phi) as i16 - phi_from_rac(c2.phi) as i16).unsigned_abs()) / 255.0;
    let h_dist = f64::from((c1.harmonic.harmonic() as i8 - c2.harmonic.harmonic() as i8).unsigned_abs()) / 4.0;
    let r_dist = (c1.radius - c2.radius).abs();

    // Weighted average (from spec: w_θ=0.3, w_φ=0.4, w_h=0.2, w_r=0.1)
    0.3 * theta_dist + 0.4 * phi_dist + 0.2 * h_dist + 0.1 * r_dist
}

/// Verify Invariant O4: Omega format indices are 0-4
pub fn verify_omega_indices() -> bool {
    OmegaFormat::Red.harmonic() == 0 && OmegaFormat::Blue.harmonic() == 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordinate_creation() {
        let coord = RaCoordinate::new(
            Repitan::NINTH,
            RacLevel::RAC1,
            OmegaFormat::Green,
            0.5,
        );
        assert!(coord.is_some());

        // Invalid radius
        let invalid = RaCoordinate::new(
            Repitan::FIRST,
            RacLevel::RAC1,
            OmegaFormat::Green,
            1.5,
        );
        assert!(invalid.is_none());
    }

    #[test]
    fn test_theta_conversion() {
        let r = Repitan::NINTH;
        let theta = theta_from_repitan(&r);
        let r2 = repitan_from_theta(theta);
        assert_eq!(r.index(), r2.index());
    }

    #[test]
    fn test_phi_roundtrip() {
        for level in RacLevel::all() {
            let phi = phi_from_rac(level);
            let recovered = rac_from_phi(phi);
            assert_eq!(level, recovered);
        }
    }

    #[test]
    fn test_radius_normalization() {
        let raw = ANKH / 2.0;  // Half of Ankh
        let normalized = normalize_radius(raw);
        assert!((normalized - 0.5).abs() < 1e-10);

        let denormalized = denormalize_radius(normalized);
        assert!((denormalized - raw).abs() < 1e-10);
    }

    #[test]
    fn test_coordinate_distance() {
        let c1 = RaCoordinate::new(
            Repitan::FIRST,
            RacLevel::RAC1,
            OmegaFormat::Red,
            0.5,
        ).unwrap();

        // Same coordinate should have distance 0
        assert!((coordinate_distance(&c1, &c1)).abs() < 1e-10);

        // Different coordinate
        let c2 = RaCoordinate::new(
            Repitan::UNITY,
            RacLevel::RAC6,
            OmegaFormat::Blue,
            1.0,
        ).unwrap();

        let dist = coordinate_distance(&c1, &c2);
        assert!(dist > 0.0 && dist <= 1.0);
    }

    #[test]
    fn test_invariant_o4() {
        assert!(verify_omega_indices());
    }
}
