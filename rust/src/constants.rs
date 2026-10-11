//! Ra System fundamental constants with type-safe wrappers.
//!
//! Values are Wesley H. Bateman's, from "The Rods of Amon Ra" (1992-1997), as
//! printed and as checked page by page by the lattice's TR node (Aug-Sep 2026).
//! They are the author's own units and ratios, not physics claims. Citations
//! name the lattice post (TR-###) and the archive page or printed folio; the
//! Python package (python/ra_system/constants.py) carries the full notes.

use serde::{Deserialize, Serialize};
use std::f64::consts::SQRT_2;

/// Colour ratio r = Green/Red = Blue/Green = 60000 sqrt2 / 84823 = 1.000351482
/// (TR-291, TR-300; printed at vol2a g00215_src01_p0031, TR-349). Applies as r^d,
/// d = number of lengths in play (volume_1 p.0117; TR-294, TR-301).
pub const COLOUR_RATIO: f64 = 60000.0 * SQRT_2 / 84823.0;

/// Omega Ratio (Q-Ratio) = 32 sqrt2 / 45 = 1.005662978 (TR-304; vol2a prologue, TR-302)
pub const OMEGA: f64 = 32.0 * SQRT_2 / 45.0;

/// Outer ladder rung Omega / r = 84823/84375 = 1.005309630 (TR-308, TR-309, TR-312)
pub const OMEGA_OVER_R: f64 = 84823.0 / 84375.0;

// Pi variants (chromatic)
/// Red Pi = 84823/27000 = 3.141592592... Not pi: pi rounded at a base-27 scale (TR-291, TR-349)
#[allow(clippy::approx_constant)]
pub const RED_PI: f64 = 84823.0 / 27000.0;
/// Green Pi = (20/9) sqrt2 = 3.142696805 (TR-300; named at volume_1 p.0129, TR-301)
pub const GREEN_PI: f64 = 20.0 / 9.0 * SQRT_2;
/// Blue Pi = Green Pi x r = 3.143801406 (TR-308; volume_1 p.0122)
pub const BLUE_PI: f64 = GREEN_PI * COLOUR_RATIO;

// Phi variants (chromatic): Bateman's erratum, supplement g00857_src03_p0002
// (FP-1399, TR-312, TR-316). It withdraws the Vol 2/3 Red Phi 1.619430799.
/// Omega Major Phi = 1.62, also the Ankh wheel diameter (volume_1 p.0168, TR-298)
pub const OMEGA_MAJOR_PHI: f64 = 1.62;
/// Red Phi = 1.62 x 84823/84375 = 1.6286016 (supplement p.0002; TR-312, TR-316)
pub const RED_PHI: f64 = 1.6286016;
/// Green Phi = 1.629174025 (supplement p.0002; TR-312)
pub const GREEN_PHI: f64 = 1.629174025;
/// Blue Phi = 1.629746650 (supplement p.0002; TR-312)
pub const BLUE_PHI: f64 = 1.629746650;

/// Hunab (Red) = 84823/80000 = 1.0602875 (TR-291; volume_1 p.0150; FP-1399)
pub const HUNAB: f64 = 84823.0 / 80000.0;

/// Green Hunab = (3/4) sqrt2 = Hunab x r exactly (TR-320, FP-1430, TR-327)
pub const GREEN_HUNAB: f64 = 0.75 * SQRT_2;

/// H-Bar = 135/128 = 1.0546875 = Green Hunab / Omega (TR-304: vol2a g00194_src01_p0010).
/// Bateman's value, not the physical reduced Planck constant.
/// Invariant I3: H-Bar = Green Hunab / Omega
pub const H_BAR: f64 = 135.0 / 128.0;

/// Ankh = 3 x 84823/50000 = 5.08938 = Red Pi x 1.62 (TR-329; volume_1 p.0168, TR-298)
/// Invariant I1: Ankh = Red Pi x Omega Major Phi
pub const ANKH: f64 = 3.0 * 84823.0 / 50000.0;

/// Bateman's f.s.Delta unit = 1/729 = Repitan(1)^2 (TR-323, TR-331).
/// Not the physical fine-structure constant (0.0072973526).
/// Invariant I6: Fine Structure = (1/27)^2
pub const FINE_STRUCTURE: f64 = 1.0 / 729.0;

/// Bateman's Ra Rydberg = 800/729 (volume_1 p.0126 "Green Rydberg"; TR-320, TR-322).
/// Not CODATA R_inf; do not normalise his spectral values (TR-283).
pub const RA_RYDBERG: f64 = 800.0 / 729.0;

/// Balmer constant = 729/2 = 364.5 (TR-288, TR-331)
pub const BALMER_CONSTANT: f64 = 729.0 / 2.0;

/// Typed wrapper for Ankh values
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Ankh(pub f64);

impl Default for Ankh {
    fn default() -> Self {
        Ankh(ANKH)
    }
}

impl Ankh {
    /// Create a new Ankh with the canonical value
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Derive RAC1 from Ankh (Invariant I2: RAC1 = Ankh / 8)
    pub fn derive_rac1(&self) -> f64 {
        self.0 / 8.0
    }
}

/// Typed wrapper for Omega Ratio
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OmegaRatio(pub f64);

impl Default for OmegaRatio {
    fn default() -> Self {
        OmegaRatio(OMEGA)
    }
}

impl OmegaRatio {
    /// Create a new OmegaRatio with the canonical value
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Get the reciprocal
    pub fn reciprocal(&self) -> f64 {
        1.0 / self.0
    }
}

/// Typed wrapper for Hunab
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Hunab(pub f64);

impl Default for Hunab {
    fn default() -> Self {
        Hunab(HUNAB)
    }
}

impl Hunab {
    /// Create a new Hunab with the canonical value
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// Typed wrapper for H-Bar
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HBar(pub f64);

impl Default for HBar {
    fn default() -> Self {
        HBar(H_BAR)
    }
}

impl HBar {
    /// Create a new HBar with the canonical value
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Verify I3: H-Bar = Green Hunab / Ω = 135/128 (TR-304)
    pub fn verify_invariant() -> bool {
        let computed = GREEN_HUNAB / OMEGA;
        (H_BAR - computed).abs() < 1e-12
    }
}

/// Typed wrapper for Fine Structure constant
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FineStructure(pub f64);

impl Default for FineStructure {
    fn default() -> Self {
        FineStructure(FINE_STRUCTURE)
    }
}

impl FineStructure {
    /// Create a new FineStructure with the canonical value
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the underlying value
    pub fn value(&self) -> f64 {
        self.0
    }

    /// Verify I6: Fine Structure = Repitan(1)² = (1/27)²
    pub fn verify_invariant() -> bool {
        let r1 = 1.0 / 27.0;
        (FINE_STRUCTURE - r1 * r1).abs() < 1e-10
    }
}

/// Verify Invariant I1: Ankh = π_red × 1.62 (Omega Major Phi, the Ankh wheel diameter)
pub fn verify_ankh_invariant() -> bool {
    let computed = RED_PI * OMEGA_MAJOR_PHI;
    (ANKH - computed).abs() < 1e-12
}

/// Verify Invariant I2: RAC1 = Ankh / 8
pub fn verify_rac1_invariant(rac1_value: f64) -> bool {
    let computed = ANKH / 8.0;
    (rac1_value - computed).abs() < 0.0001
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ankh_invariant() {
        assert!(verify_ankh_invariant());
    }

    #[test]
    fn test_rac1_derivation() {
        let ankh = Ankh::new();
        let rac1 = ankh.derive_rac1();
        assert!((rac1 - 0.6361725).abs() < 0.0001);
    }

    #[test]
    fn test_hbar_invariant() {
        assert!(HBar::verify_invariant());
    }

    #[test]
    fn test_fine_structure_invariant() {
        assert!(FineStructure::verify_invariant());
    }
}
