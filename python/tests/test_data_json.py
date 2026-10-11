"""
Checks on data/ra_constants_v2.json against the TR (Trellis) page verification.

The JSON keeps Bateman's printed values. Where TR found a printed or extracted value
damaged, the corrected value is stored and the damaged one is kept beside it as
`printed_value`, with the TR post that established the correction.
"""

import json
import pathlib

import pytest

from ra_system.omega import OmegaFormat, convert_omega

DATA = pathlib.Path(__file__).resolve().parents[2] / "data" / "ra_constants_v2.json"


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(DATA.read_text(encoding="utf-8"))


def _fundamental(data: dict, name: str) -> dict:
    return next(e for e in data["fundamental_constants"] if e["name"] == name)


def test_fine_structure_unit_is_one_over_729(data: dict) -> None:
    # TR-281 s5 / TR-323: printed 0.0001371742 is 10x small (lost decimal, BAT-002).
    e = _fundamental(data, "Fine Structure Constant (Ra)")
    assert e["value"] == pytest.approx(1 / 729, abs=1e-13)
    assert e["printed_value"] == 0.0001371742


def test_rydberg_is_800_over_729(data: dict) -> None:
    # TR-320 / TR-322: 800/729 = 1.09739369; extraction had 1.097369.
    ryd = data["spectral_constants"]["rydberg_constant"]
    assert ryd["ra_value"] == pytest.approx(800 / 729, abs=1e-9)
    assert ryd["omega_major_value"] == 729 / 800


def test_lyman_m2_hz_is_hunab_not_dropped_zero(data: dict) -> None:
    # TR-288 (BAT-028, BAT-CONTRA-007): 1.062875 is a dropped '0' for 1.0602875.
    e = data["elf_wave_constants"]["lyman_m2_connection"]
    assert e["value"] == 1.0602875
    assert e["printed_value"] == 1.062875


def test_h_bar_derivation_uses_green_hunab(data: dict) -> None:
    # TR-304: H-Bar = Green Hunab / Omega = 135/128.
    e = _fundamental(data, "H-Bar (Omega Major)")
    assert e["value"] == 1.0546875
    assert "Green Hunab" in e["symbolic_equivalence"]


def test_phi_erratum_recorded(data: dict) -> None:
    # FP-1399 / TR-312: supplement p.0002 withdraws Red Phi 1.619430799.
    e = _fundamental(data, "Phi (full precision)")
    assert e["value"] == 1.6286016
    assert e["printed_value"] == 1.619430799


def test_lyman_m6_is_6561_over_7(data: dict) -> None:
    # TR-280 s7 / TR-286: volume_1 p.0100 prints 93.72857142 (x10 = 6561/7);
    # the Supplement's 932.7857142 is transposed.
    m6 = data["hydrogen_spectral_lines"]["lyman_series"]["M6"]
    assert m6["wavelength_omega_major_angstroms"] == pytest.approx(6561 / 7, abs=1e-6)


def test_conversion_table_matches_exact_ladder(data: dict) -> None:
    # The printed conversion table must agree with the exact ladder (TR-312) at print width.
    names = {
        "red": OmegaFormat.RED,
        "omega_major": OmegaFormat.OMEGA_MAJOR,
        "green": OmegaFormat.GREEN,
        "blue": OmegaFormat.BLUE,
        "omega_minor": OmegaFormat.OMEGA_MINOR,
    }
    for key, printed in data["omega_format_conversions"]["conversions"].items():
        if key.startswith("meters"):
            continue
        a, b = key.split("_to_")
        assert printed == pytest.approx(convert_omega(names[a], names[b], 1.0), abs=1.5e-9), key


def test_no_unsourced_face_angle(data: dict) -> None:
    # TR-278 / TR-290.
    assert "51.851651" not in DATA.read_text(encoding="utf-8")
