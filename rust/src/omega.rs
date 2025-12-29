//! OmegaFormat enum with conversion functions.
//!
//! Five-level Omega format system for frequency/precision tiers.
//! Hierarchy: Red > Omega Major > Green > Omega Minor > Blue
//!
//! Conversions use the Omega Ratio (Q-Ratio): Ω = 1.005662978

use serde::{Deserialize, Serialize};
use crate::constants::OMEGA;

/// The five Omega format levels (coherence depth tiers)
/// Index 0 = Red (highest precision), Index 4 = Blue
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OmegaFormat {
    Red = 0,
    OmegaMajor = 1,
    Green = 2,
    OmegaMinor = 3,
    Blue = 4,
}

impl OmegaFormat {
    /// Get all formats in order
    pub fn all() -> [OmegaFormat; 5] {
        [
            OmegaFormat::Red,
            OmegaFormat::OmegaMajor,
            OmegaFormat::Green,
            OmegaFormat::OmegaMinor,
            OmegaFormat::Blue,
        ]
    }

    /// Get format from harmonic index (0-4)
    pub fn from_harmonic(h: u8) -> Option<Self> {
        match h {
            0 => Some(OmegaFormat::Red),
            1 => Some(OmegaFormat::OmegaMajor),
            2 => Some(OmegaFormat::Green),
            3 => Some(OmegaFormat::OmegaMinor),
            4 => Some(OmegaFormat::Blue),
            _ => None,
        }
    }

    /// Get harmonic index (0-4)
    pub fn harmonic(&self) -> u8 {
        *self as u8
    }
}

/// Get conversion factor between formats
fn conversion_factor(from: OmegaFormat, to: OmegaFormat) -> f64 {
    use OmegaFormat::*;

    match (from, to) {
        // Identity cases
        (Red, Red) | (OmegaMajor, OmegaMajor) | (Green, Green) | (OmegaMinor, OmegaMinor) | (Blue, Blue) => 1.0,

        // From Green (reference)
        (Green, OmegaMajor) => 0.994368911,     // 1/Ω
        (Green, OmegaMinor) => 1.005662978,     // Ω
        (Green, Red)        => 0.999648641,
        (Green, Blue)       => 1.000351482,

        // To Green (reference)
        (OmegaMajor, Green) => 1.005662978,     // Ω
        (OmegaMinor, Green) => 0.994368911,     // 1/Ω
        (Red, Green)        => 1.000351482,
        (Blue, Green)       => 0.999648641,

        // Omega Major conversions
        (OmegaMajor, Red)        => 1.005309630,
        (Red, OmegaMajor)        => 0.994718414,
        (OmegaMajor, Blue)       => 1.006016451,
        (Blue, OmegaMajor)       => 0.994019530,
        (OmegaMajor, OmegaMinor) => 1.011358026,
        (OmegaMinor, OmegaMajor) => 0.988769530,

        // Red/Blue conversions
        (Red, Blue)  => 1.000703088,
        (Blue, Red)  => 0.999297406,

        // Red/Omega Minor
        (Red, OmegaMinor)  => 1.006016451,
        (OmegaMinor, Red)  => 0.994019530,

        // Blue/Omega Minor
        (Blue, OmegaMinor)  => 1.005309630,
        (OmegaMinor, Blue)  => 0.994718414,
    }
}

/// Convert a value between two Omega formats
pub fn convert_omega(from: OmegaFormat, to: OmegaFormat, x: f64) -> f64 {
    x * conversion_factor(from, to)
}

/// Green to Omega Major: x / Ω
pub fn green_to_omega_major(x: f64) -> f64 {
    convert_omega(OmegaFormat::Green, OmegaFormat::OmegaMajor, x)
}

/// Omega Major to Green: x × Ω
pub fn omega_major_to_green(x: f64) -> f64 {
    convert_omega(OmegaFormat::OmegaMajor, OmegaFormat::Green, x)
}

/// Green to Omega Minor: x × Ω
pub fn green_to_omega_minor(x: f64) -> f64 {
    convert_omega(OmegaFormat::Green, OmegaFormat::OmegaMinor, x)
}

/// Omega Minor to Green: x / Ω
pub fn omega_minor_to_green(x: f64) -> f64 {
    convert_omega(OmegaFormat::OmegaMinor, OmegaFormat::Green, x)
}

/// Red to Blue
pub fn red_to_blue(x: f64) -> f64 {
    convert_omega(OmegaFormat::Red, OmegaFormat::Blue, x)
}

/// Blue to Red
pub fn blue_to_red(x: f64) -> f64 {
    convert_omega(OmegaFormat::Blue, OmegaFormat::Red, x)
}

/// Tolerance for roundtrip conversion verification
pub const ROUNDTRIP_TOLERANCE: f64 = 1e-10;

/// Verify Invariant C1: roundtrip conversions preserve value
/// convert_omega(f, t, convert_omega(t, f, x)) ≈ x
pub fn verify_omega_roundtrip(from: OmegaFormat, to: OmegaFormat, x: f64) -> bool {
    let roundtrip = convert_omega(to, from, convert_omega(from, to, x));
    (roundtrip - x).abs() < ROUNDTRIP_TOLERANCE
}

/// Verify all omega roundtrips for a value
pub fn verify_all_omega_roundtrips(x: f64) -> bool {
    OmegaFormat::all().iter().all(|&from| {
        OmegaFormat::all().iter().all(|&to| {
            verify_omega_roundtrip(from, to, x)
        })
    })
}

/// Verify Invariant R4: Omega format index ∈ {0, 1, 2, 3, 4}
pub fn verify_omega_range() -> bool {
    OmegaFormat::all().iter().all(|f| f.harmonic() <= 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omega_harmonic() {
        assert_eq!(OmegaFormat::Red.harmonic(), 0);
        assert_eq!(OmegaFormat::Blue.harmonic(), 4);
    }

    #[test]
    fn test_omega_from_harmonic() {
        assert_eq!(OmegaFormat::from_harmonic(0), Some(OmegaFormat::Red));
        assert_eq!(OmegaFormat::from_harmonic(4), Some(OmegaFormat::Blue));
        assert_eq!(OmegaFormat::from_harmonic(5), None);
    }

    #[test]
    fn test_identity_conversion() {
        let x = 1.62;
        for fmt in OmegaFormat::all() {
            assert!((convert_omega(fmt, fmt, x) - x).abs() < 1e-10);
        }
    }

    #[test]
    fn test_green_conversions() {
        let green = 1.62;

        // Green → Omega Major: x / Ω
        let major = green_to_omega_major(green);
        assert!((major - green / OMEGA).abs() < 1e-10);

        // Green → Omega Minor: x × Ω
        let minor = green_to_omega_minor(green);
        assert!((minor - green * OMEGA).abs() < 1e-10);
    }

    #[test]
    fn test_roundtrip_all_formats() {
        let x = 1.62;
        assert!(verify_all_omega_roundtrips(x));
    }

    #[test]
    fn test_invariant_r4() {
        assert!(verify_omega_range());
    }
}
