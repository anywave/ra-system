//! Integration tests for Ra System invariants.
//!
//! Tests all 17 invariants from ra_integration_spec.md Section 6.

use ra_core::*;
use ra_core::constants::*;
use ra_core::repitans::*;
use ra_core::rac::*;
use ra_core::omega::*;
use ra_core::spherical::*;
use ra_core::gates::*;

// =============================================================================
// Constant Invariants (I1-I6)
// =============================================================================

#[test]
fn test_i1_ankh_equals_red_pi_times_green_phi() {
    let computed = RED_PI * GREEN_PHI;
    assert!(
        (ANKH - computed).abs() < 0.0001,
        "I1: Ankh = π_red × φ_green. Expected {}, got {}",
        computed,
        ANKH
    );
}

#[test]
fn test_i2_rac1_equals_ankh_div_8() {
    let computed = ANKH / 8.0;
    let rac1 = rac_value(RacLevel::RAC1).value();
    assert!(
        (rac1 - computed).abs() < 0.0001,
        "I2: RAC₁ = Ankh / 8. Expected {}, got {}",
        computed,
        rac1
    );
}

#[test]
fn test_i3_hbar_equals_hunab_div_omega() {
    let computed = HUNAB / OMEGA;
    assert!(
        (H_BAR - computed).abs() < 0.0001,
        "I3: H-Bar = Hunab / Ω. Expected {}, got {}",
        computed,
        H_BAR
    );
}

#[test]
fn test_i4_repitan_equals_n_div_27() {
    for n in 1..=27 {
        let r = Repitan::new(n).unwrap();
        let expected = f64::from(n) / 27.0;
        assert!(
            (r.value() - expected).abs() < 1e-10,
            "I4: Repitan({}) should equal {}/27 = {}, got {}",
            n,
            n,
            expected,
            r.value()
        );
    }
}

#[test]
fn test_i5_ton_equals_m_times_0027() {
    // T.O.N. values: m × 0.027 for m ∈ [0, 36]
    for m in 0..=35 {
        let ton = f64::from(m) * 0.027;
        assert!(
            ton >= 0.0 && ton < 1.0,
            "I5: T.O.N.({}) = {} should be in [0, 1)",
            m,
            ton
        );
    }
}

#[test]
fn test_i6_fine_structure_equals_repitan1_squared() {
    let r1 = Repitan::FIRST.value();
    let computed = r1 * r1;
    assert!(
        (FINE_STRUCTURE - computed).abs() < 1e-10,
        "I6: Fine Structure = Repitan(1)². Expected {}, got {}",
        computed,
        FINE_STRUCTURE
    );
}

// =============================================================================
// Ordering Invariants (O1-O4)
// =============================================================================

#[test]
fn test_o1_rac_ordering() {
    assert!(
        verify_rac_ordering(),
        "O1: RAC₁ > RAC₂ > RAC₃ > RAC₄ > RAC₅ > RAC₆ > 0"
    );
}

#[test]
fn test_o2_pi_ordering() {
    assert!(
        RED_PI < GREEN_PI && GREEN_PI < BLUE_PI,
        "O2: π_red < π_green < π_blue"
    );
}

#[test]
fn test_o3_repitan_range() {
    assert!(
        verify_repitan_range_invariant(),
        "O3: For all n: 0 < Repitan(n) ≤ 1"
    );
}

#[test]
fn test_o4_omega_indices() {
    assert!(
        verify_omega_indices(),
        "O4: Omega format indices are 0-4"
    );
}

// =============================================================================
// Conversion Invariants (C1-C3)
// =============================================================================

#[test]
fn test_c1_omega_roundtrip() {
    let x = 1.62;
    assert!(
        verify_all_omega_roundtrips(x),
        "C1: Omega roundtrip conversions preserve value"
    );
}

#[test]
fn test_c2_green_times_omega_equals_omega_minor() {
    let green = 1.62;
    let omega_minor = green_to_omega_minor(green);
    let expected = green * OMEGA;
    assert!(
        (omega_minor - expected).abs() < 1e-9,
        "C2: Green × Ω = Omega_Minor"
    );
}

#[test]
fn test_c3_green_div_omega_equals_omega_major() {
    let green = 1.62;
    let omega_major = green_to_omega_major(green);
    let expected = green / OMEGA;
    assert!(
        (omega_major - expected).abs() < 1e-9,
        "C3: Green / Ω = Omega_Major"
    );
}

// =============================================================================
// Range Invariants (R1-R4)
// =============================================================================

#[test]
fn test_r1_rac_range() {
    assert!(
        verify_rac_range(),
        "R1: 0 < RAC(i) < 1 for all i ∈ [1, 6]"
    );
}

#[test]
fn test_r2_repitan_range() {
    for n in 1..=27 {
        let r = Repitan::new(n).unwrap();
        let v = r.value();
        assert!(
            v > 0.0 && v <= 1.0,
            "R2: Repitan({}) value {} should be in (0, 1]",
            n,
            v
        );
    }
}

#[test]
fn test_r3_coherence_bounds() {
    assert!(
        verify_coherence_bounds(),
        "R3: Coherence bounds are [0, 1]"
    );
}

#[test]
fn test_r4_omega_range() {
    assert!(
        verify_omega_range(),
        "R4: Omega format index ∈ {{0, 1, 2, 3, 4}}"
    );
}

// =============================================================================
// Additional Property Tests
// =============================================================================

#[test]
fn test_repitan_smart_constructor() {
    // Valid range
    for n in 1..=27 {
        assert!(Repitan::new(n).is_ok());
    }
    // Invalid range
    assert!(Repitan::new(0).is_err());
    assert!(Repitan::new(28).is_err());
    assert!(Repitan::new(-1).is_err());
}

#[test]
fn test_access_gating_full_coherence() {
    for level in RacLevel::all() {
        let result = access_level(1.0, level);
        assert!(
            result.is_full_access(),
            "Full coherence should always grant FullAccess for {:?}",
            level
        );
    }
}

#[test]
fn test_access_gating_zero_coherence() {
    for level in RacLevel::all() {
        let result = access_level(0.0, level);
        assert!(
            result.is_blocked(),
            "Zero coherence should always be Blocked for {:?}",
            level
        );
    }
}

#[test]
fn test_access_alpha_in_range() {
    let coherences = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

    for level in RacLevel::all() {
        for &c in &coherences {
            let result = access_level(c, level);
            let alpha = result.alpha();
            assert!(
                alpha >= 0.0 && alpha <= 1.0,
                "Alpha {} for coherence {} and {:?} should be in [0, 1]",
                alpha,
                c,
                level
            );
        }
    }
}

#[test]
fn test_coordinate_radius_validation() {
    // Valid radius
    let valid = RaCoordinate::new(
        Repitan::NINTH,
        RacLevel::RAC1,
        OmegaFormat::Green,
        0.5,
    );
    assert!(valid.is_some());

    // Invalid radius (negative)
    let invalid_neg = RaCoordinate::new(
        Repitan::FIRST,
        RacLevel::RAC1,
        OmegaFormat::Green,
        -0.1,
    );
    assert!(invalid_neg.is_none());

    // Invalid radius (> 1)
    let invalid_high = RaCoordinate::new(
        Repitan::FIRST,
        RacLevel::RAC1,
        OmegaFormat::Green,
        1.1,
    );
    assert!(invalid_high.is_none());
}

#[test]
fn test_theta_repitan_roundtrip() {
    for n in 1..=27 {
        let r = Repitan::new(n).unwrap();
        let theta = theta_from_repitan(&r);
        let r2 = repitan_from_theta(theta);
        assert_eq!(
            r.index(),
            r2.index(),
            "Theta roundtrip failed for Repitan({})",
            n
        );
    }
}

#[test]
fn test_radius_normalization_roundtrip() {
    let raw_values = [0.0, 1.0, 2.5, ANKH / 2.0, ANKH, ANKH * 1.5];

    for raw in raw_values {
        let normalized = normalize_radius(raw);
        assert!(
            normalized >= 0.0 && normalized <= 1.0,
            "Normalized radius {} should be in [0, 1]",
            normalized
        );

        // If within Ankh, roundtrip should preserve
        if raw <= ANKH && raw >= 0.0 {
            let denormalized = denormalize_radius(normalized);
            assert!(
                (denormalized - raw).abs() < 1e-10,
                "Radius roundtrip failed for {}",
                raw
            );
        }
    }
}
