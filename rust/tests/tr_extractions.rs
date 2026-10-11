//! Regression tests for the TR (Trellis) Bateman extractions, Aug-Sep 2026.
//! Bateman's values as printed and verified, not physics claims. See
//! python/tests/test_tr_extractions.py for the full set and citations.

use ra_core::constants::*;
use ra_core::gates::COHERENCE_FLOOR;
use std::f64::consts::SQRT_2;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
fn red_pi_is_84823_over_27000() {
    // TR-291, TR-349
    assert!(close(RED_PI, 84823.0 / 27000.0, 1e-15));
}

#[test]
fn green_and_blue_pi_follow_the_r_ladder() {
    // TR-300, TR-301 (volume_1 p.0129), TR-308 (volume_1 p.0122)
    assert!(close(GREEN_PI, 20.0 / 9.0 * SQRT_2, 1e-15));
    assert!(close(BLUE_PI, 3.143801406, 2e-9));
    assert!(close(GREEN_PI / RED_PI, COLOUR_RATIO, 1e-14));
}

#[test]
fn hunab_and_h_bar() {
    // TR-291 (volume_1 p.0150); TR-304 (vol2a g00194_src01_p0010)
    assert!(close(HUNAB, 84823.0 / 80000.0, 1e-15));
    assert!(close(GREEN_HUNAB, 0.75 * SQRT_2, 1e-15));
    assert!(close(H_BAR, 135.0 / 128.0, 1e-15));
    assert!(close(GREEN_HUNAB / OMEGA, H_BAR, 1e-12));
}

#[test]
fn omega_rydberg_fine_structure() {
    // TR-304, TR-320, TR-323
    assert!(close(OMEGA, 32.0 * SQRT_2 / 45.0, 1e-15));
    assert!(close(OMEGA_OVER_R, 84823.0 / 84375.0, 1e-15));
    assert!(close(RA_RYDBERG, 800.0 / 729.0, 1e-15));
    assert!(close(FINE_STRUCTURE, 1.0 / 729.0, 1e-16));
}

#[test]
fn phi_follows_the_supplement_erratum() {
    // FP-1399, TR-312, TR-316 (supplement g00857_src03_p0002)
    assert_eq!(OMEGA_MAJOR_PHI, 1.62);
    assert_eq!(RED_PHI, 1.6286016);
    assert_eq!(GREEN_PHI, 1.629174025);
    assert_eq!(BLUE_PHI, 1.629746650);
    assert!(close(OMEGA_MAJOR_PHI * OMEGA_OVER_R, RED_PHI, 1e-12));
}

#[test]
fn ankh_and_gate_floor() {
    // TR-298 (volume_1 p.0168), TR-329
    assert!(close(ANKH, 3.0 * 84823.0 / 50000.0, 1e-15));
    assert!(close(RED_PI * OMEGA_MAJOR_PHI, ANKH, 1e-12));
    // The gate floor is a modelling parameter; the erratum must not move it.
    assert!(close(COHERENCE_FLOOR, 1.62 / 5.08938, 1e-15));
}
