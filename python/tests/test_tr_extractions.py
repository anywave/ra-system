"""
Regression tests for the TR (Trellis) Bateman extractions, Aug-Sep 2026.

Each test pins a value or behaviour that TR verified against the printed page
of Wesley H. Bateman's "The Rods of Amon Ra" (1992-1997). These are Bateman's
values as printed and checked, not physics claims. Citations: lattice post id,
then the archive page id or printed folio where TR gave one.
"""

import math
import pathlib
from fractions import Fraction

import pytest

from ra_system import constants as c
from ra_system.omega import OmegaFormat, convert_omega
from ra_system.rac import (
    RAC_UNITS_OF_RED_PI,
    RacLevel,
    pyramid_division,
    rac_value,
)

SQRT2 = math.sqrt(2)
R = 60000 * SQRT2 / 84823  # colour ratio r, TR-300
OMEGA_EXACT = 32 * SQRT2 / 45  # TR-304


# ---------------------------------------------------------------------------
# Pi family
# ---------------------------------------------------------------------------


def test_red_pi_is_84823_over_27000_not_pi() -> None:
    # TR-291 / TR-349: Red Pi = round(pi x 27000) / 27000; vol2a p.167, volume_1 p.0150.
    assert c.RED_PI == pytest.approx(84823 / 27000, abs=1e-15)
    # It is NOT pi: it differs at the 8th decimal.
    assert abs(c.RED_PI - math.pi) > 5e-8


def test_green_pi_is_20_ninths_root2() -> None:
    # TR-300 / TR-301: volume_1 p.0129 (procedure 7-3) names (20/9)sqrt2 GREEN PI.
    assert c.GREEN_PI == pytest.approx(20 / 9 * SQRT2, abs=1e-15)
    # Green Pi squared = 800/81 exactly (TR-323, volume_1 p.0105).
    assert c.GREEN_PI**2 == pytest.approx(800 / 81, abs=1e-12)


def test_blue_pi_is_one_r_step_above_green() -> None:
    # TR-308: volume_1 p.0122 (printed 114) prints Blue Pi 3.143801406 on the r-ladder.
    assert c.BLUE_PI == pytest.approx(c.GREEN_PI * R, abs=1e-15)
    assert c.BLUE_PI == pytest.approx(3.143801406, abs=2e-9)


def test_omega_brackets_green_pi_with_ancient_values() -> None:
    # TR-304: Green Pi / Omega = 25/8 and Green Pi x Omega = 256/81, exactly.
    assert c.GREEN_PI / c.OMEGA == pytest.approx(25 / 8, abs=1e-12)
    assert c.GREEN_PI * c.OMEGA == pytest.approx(256 / 81, abs=1e-12)


# ---------------------------------------------------------------------------
# Colour ratio, Omega, and the ladder
# ---------------------------------------------------------------------------


def test_colour_ratio_exact_and_printed() -> None:
    # TR-291 / TR-300: r = 60000 sqrt2 / 84823.
    # Printed 1.000351482 at vol2a g00215_src01_p0031 (TR-349).
    assert c.COLOUR_RATIO == pytest.approx(R, abs=1e-15)
    assert c.COLOUR_RATIO == pytest.approx(1.000351482, abs=1e-9)
    assert c.GREEN_PI / c.RED_PI == pytest.approx(c.COLOUR_RATIO, abs=1e-14)


def test_omega_exact_form() -> None:
    # TR-304: Omega = 32 sqrt2 / 45; printed 1.005662978 (vol2a prologue, TR-302).
    assert c.OMEGA == pytest.approx(OMEGA_EXACT, abs=1e-15)
    assert c.OMEGA == pytest.approx(1.005662978, abs=1e-9)


def test_omega_over_r_is_84823_over_84375() -> None:
    # TR-308 / TR-309 / TR-312: the outer ladder rung.
    assert c.OMEGA_OVER_R == pytest.approx(84823 / 84375, abs=1e-15)
    assert c.OMEGA / c.COLOUR_RATIO == pytest.approx(c.OMEGA_OVER_R, abs=1e-14)


# ---------------------------------------------------------------------------
# Hunab, H-Bar, fine-structure unit, Rydberg
# ---------------------------------------------------------------------------


def test_hunab_is_red_hunab_not_twelfth_root_of_two() -> None:
    # TR-291: Hunab = 84823/80000 (volume_1 p.0150); FP-1399: labelled HUNAB 247 times.
    assert c.HUNAB == pytest.approx(84823 / 80000, abs=1e-15)
    assert abs(c.HUNAB - 2 ** (1 / 12)) > 5e-4


def test_green_hunab_is_three_quarters_root2() -> None:
    # TR-320 / FP-1430 s4b / TR-327: Red Hunab x r = (3/4) sqrt2 exactly.
    assert c.GREEN_HUNAB == pytest.approx(0.75 * SQRT2, abs=1e-15)
    assert c.HUNAB * c.COLOUR_RATIO == pytest.approx(c.GREEN_HUNAB, abs=1e-14)


def test_h_bar_is_135_over_128() -> None:
    # TR-304: vol2a g00194_src01_p0010 KEY NUMBERS, Omega Major column of the Hunab row.
    # TR-312: confirmed on supplement p.0002.
    assert c.H_BAR == 135 / 128
    assert c.GREEN_HUNAB / c.OMEGA == pytest.approx(c.H_BAR, abs=1e-14)
    assert c.verify_hbar_invariant()


def test_fine_structure_unit_is_one_over_729() -> None:
    # TR-281 / TR-323: Bateman's f.s.Delta = 1/729 (printed 10x small in the Supplement,
    # and with a dropped digit pair on volume_1 p.0105). Not the physical alpha.
    assert c.FINE_STRUCTURE == pytest.approx(1 / 729, abs=1e-16)
    assert abs(c.FINE_STRUCTURE - 0.0072973526) > 5e-3


def test_ra_rydberg_is_800_over_729() -> None:
    # TR-320 (volume_1 p.0126), TR-322 (volume_1 p.0130: .91125 = 729/800), TR-329.
    assert c.RA_RYDBERG == pytest.approx(800 / 729, abs=1e-15)
    assert 1 / c.RA_RYDBERG == pytest.approx(0.91125, abs=1e-15)
    assert c.RA_RYDBERG == pytest.approx(8 * c.FINE_STRUCTURE * 100, abs=1e-14)
    # 112.5 / (6561/64) = 800/729 (TR-320 s2)
    assert Fraction(225, 2) / Fraction(6561, 64) == Fraction(800, 729)


def test_balmer_constant() -> None:
    # TR-288 (BAT-BALC-001): 364.5 = 729/2, labelled in the cell.
    assert c.BALMER_CONSTANT == 729 / 2


# ---------------------------------------------------------------------------
# Phi: the author's erratum (supplement p.0002)
# ---------------------------------------------------------------------------


def test_phi_values_follow_the_supplement_erratum() -> None:
    # FP-1399 s2-s3, TR-312, TR-316: supplement g00857_src03_p0002 prints
    # OMEGA MAJOR 1.62 / RED 1.6286016 / GREEN 1.629174025 / BLUE 1.629746650 and
    # declares the Vol 2/3 Red Phi 1.619430799 erroneous.
    assert c.OMEGA_MAJOR_PHI == 1.62
    assert c.RED_PHI == 1.6286016
    assert c.GREEN_PHI == 1.629174025
    assert c.BLUE_PHI == 1.629746650
    # TR-316: the corrected Red is exactly one Omega/r rung above 1.62.
    assert c.OMEGA_MAJOR_PHI * c.OMEGA_OVER_R == pytest.approx(c.RED_PHI, abs=1e-12)
    # Inner steps are r at the printed width.
    assert c.GREEN_PHI / c.RED_PHI == pytest.approx(c.COLOUR_RATIO, abs=1e-9)
    assert c.BLUE_PHI / c.GREEN_PHI == pytest.approx(c.COLOUR_RATIO, abs=1e-9)


def test_ankh_is_red_pi_times_the_wheel_diameter() -> None:
    # TR-298: volume_1 p.0168, the Ankh measuring wheel: C = 5.08938, d = 1.62.
    # TR-329: Ankh = 3 x 84823 / 50000.
    assert c.ANKH == pytest.approx(3 * 84823 / 50000, abs=1e-15)
    assert c.RED_PI * c.OMEGA_MAJOR_PHI == pytest.approx(c.ANKH, abs=1e-12)
    assert c.verify_ankh_invariant()


# ---------------------------------------------------------------------------
# Rac units (TR-298, vol2b g00585_src09_p0038, printed 418)
# ---------------------------------------------------------------------------

_TR298 = {
    "RAC 1": Fraction(81, 400),
    "RAC 2": Fraction(1, 5),
    "RAC 3": Fraction(729, 4000),
    "RAC 4": Fraction(1, 6),
    "RAC 5": Fraction(729, 5000),
    "MIR Rac": Fraction(2, 9),
    "Block Rac 1": Fraction(27, 125),
    "Block Rac 2": Fraction(8, 45),
}


def test_rac_units_are_arcs_of_red_pi() -> None:
    for name, frac in _TR298.items():
        assert RAC_UNITS_OF_RED_PI[name] == pytest.approx(float(frac), abs=1e-15), name
    assert RAC_UNITS_OF_RED_PI["RAC 6"] == pytest.approx(0.09 * SQRT2, abs=1e-15)
    assert RAC_UNITS_OF_RED_PI["RAC 7"] == pytest.approx(SQRT2 / 8.64, abs=1e-15)


def test_rac_values_equal_red_pi_times_fraction() -> None:
    for level in RacLevel:
        frac = RAC_UNITS_OF_RED_PI[f"RAC {int(level)}"]
        assert rac_value(level) == pytest.approx(c.RED_PI * frac, abs=1e-12), level


def test_pyramid_divisions_are_base_over_rac() -> None:
    # TR-331: base = (729/10) x Red Pi, so base / RAC2 = 729/2 is forced.
    base = 72.9 * c.RED_PI
    for level in RacLevel:
        assert pyramid_division(level) == pytest.approx(base / rac_value(level), rel=1e-9), level


# ---------------------------------------------------------------------------
# Omega format ladder, with the dimensional exponent
# ---------------------------------------------------------------------------


def test_ladder_magnitude_order() -> None:
    # TR-302 / TR-312: OM < RED < GREEN < BLUE < OMEGA MINOR in magnitude.
    # The enum order follows the printed column layout, not the magnitude.
    order = [
        OmegaFormat.OMEGA_MAJOR,
        OmegaFormat.RED,
        OmegaFormat.GREEN,
        OmegaFormat.BLUE,
        OmegaFormat.OMEGA_MINOR,
    ]
    values = [convert_omega(OmegaFormat.OMEGA_MAJOR, f, 1.0) for f in order]
    assert values == sorted(values)
    assert len(set(values)) == 5


def test_ladder_steps_are_exact() -> None:
    om, red, green, blue, omin = (
        OmegaFormat.OMEGA_MAJOR,
        OmegaFormat.RED,
        OmegaFormat.GREEN,
        OmegaFormat.BLUE,
        OmegaFormat.OMEGA_MINOR,
    )
    assert convert_omega(om, red, 1.0) == pytest.approx(84823 / 84375, abs=1e-14)
    assert convert_omega(red, green, 1.0) == pytest.approx(R, abs=1e-14)
    assert convert_omega(green, blue, 1.0) == pytest.approx(R, abs=1e-14)
    assert convert_omega(blue, omin, 1.0) == pytest.approx(84823 / 84375, abs=1e-14)
    assert convert_omega(om, green, 1.0) == pytest.approx(OMEGA_EXACT, abs=1e-14)


def test_area_conversion_uses_r_squared() -> None:
    # TR-294 / TR-301: volume_1 p.0117 (printed 109) footnote: Red and Blue are
    # separated by 1.000703088 on area (degree 2) quantities.
    red, green = OmegaFormat.RED, OmegaFormat.GREEN
    assert convert_omega(red, green, 1.0, degree=2) == pytest.approx(1.000703088, abs=1e-9)
    # TR-302: K1-D area row, Green / Omega Major = Omega^2.
    om = OmegaFormat.OMEGA_MAJOR
    assert convert_omega(om, green, 46132.031235, degree=2) == pytest.approx(46656, abs=1e-4)
    # TR-309: volume rows use r^3.
    assert convert_omega(red, green, 1.0, degree=3) == pytest.approx(R**3, abs=1e-14)
    # degree 0 is the identity.
    assert convert_omega(red, OmegaFormat.BLUE, 7.0, degree=0) == 7.0


def test_roundtrip_is_exact_now() -> None:
    for a in OmegaFormat:
        for b in OmegaFormat:
            assert convert_omega(b, a, convert_omega(a, b, 123.456)) == pytest.approx(
                123.456, abs=1e-12
            )


# ---------------------------------------------------------------------------
# Guard: values with no source must not come back (TR-278 / TR-290)
# ---------------------------------------------------------------------------


def test_no_unsourced_face_angle_constant() -> None:
    # TR-278 / TR-290: 51.851651 appears nowhere in 863 pages; 51.851 (51 23/27) on
    # vol2a g00343_src02_p0035 is a count of Red Rac 3 units, not an angle.
    pkg = pathlib.Path(c.__file__).parent
    for path in pkg.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "51.851651" not in text, path
        assert "12th root of 2" not in text, path


def test_coherence_floor_unchanged_by_phi_erratum() -> None:
    # The gate floor was 1.62 / Ankh = 1 / Red Pi before the Phi erratum was applied.
    # It is a modelling parameter; the erratum must not move it.
    from ra_system.gates import COHERENCE_FLOOR, verify_coherence_floor

    assert COHERENCE_FLOOR == pytest.approx(1.62 / 5.08938, abs=1e-15)
    assert COHERENCE_FLOOR == pytest.approx(1 / c.RED_PI, abs=1e-15)
    assert verify_coherence_floor()
