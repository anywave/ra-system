//! Ra System fundamental constants with type-safe wrappers.
//!
//! All constants are derived from "The Rods of Amon Ra" by Wesley H. Bateman.

use serde::{Deserialize, Serialize};

/// Ankh: Master harmonic constant = 5.08938
/// Invariant I1: Ankh = π_red × φ_green = 3.14159265 × 1.62
pub const ANKH: f64 = 5.08938;

/// Hunab: Natural constant = 1.05946 (12th root of 2)
pub const HUNAB: f64 = 1.05946;

/// H-Bar: Hunab / Omega = 1.05346545
/// Invariant I3: H-Bar = Hunab / Ω
pub const H_BAR: f64 = 1.05346545;

/// Omega Ratio (Q-Ratio): 1.005662978
pub const OMEGA: f64 = 1.005662978;

/// Fine Structure: Repitan(1)² = 0.0013717421
/// Invariant I6: Fine Structure = (1/27)²
pub const FINE_STRUCTURE: f64 = 0.0013717421;

// Pi variants (chromatic)
/// Red Pi: Standard π = 3.14159265
pub const RED_PI: f64 = 3.14159265;
/// Green Pi: 3.14754099
pub const GREEN_PI: f64 = 3.14754099;
/// Blue Pi: 3.15349386
pub const BLUE_PI: f64 = 3.15349386;

// Phi variants (chromatic)
/// Red Phi: 1.614
pub const RED_PHI: f64 = 1.614;
/// Green Phi: φ = 1.62
pub const GREEN_PHI: f64 = 1.62;
/// Blue Phi: 1.626
pub const BLUE_PHI: f64 = 1.626;

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

    /// Verify I3: H-Bar = Hunab / Ω
    pub fn verify_invariant() -> bool {
        let computed = HUNAB / OMEGA;
        (H_BAR - computed).abs() < 0.0001
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

/// Verify Invariant I1: Ankh = π_red × φ_green
pub fn verify_ankh_invariant() -> bool {
    let computed = RED_PI * GREEN_PHI;
    (ANKH - computed).abs() < 0.0001
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
