"""
OmegaFormat enum with conversion functions.

Bateman's five "windows" for one quantity (vol2a prologue; TR-302, TR-312).
In magnitude, a value runs

    Omega Major < Red < Green < Blue < Omega Minor

with steps Omega/r (= 84823/84375), r, r, Omega/r, where r = 60000 sqrt2 / 84823
and Omega = 32 sqrt2 / 45 (TR-300, TR-304, TR-312). The enum order below follows
the printed column layout RED | OMEGA MAJOR | GREEN | OMEGA MINOR | BLUE, which is
not the magnitude order - do not sort on it (TR-302).

Each factor applies to the power d, the number of lengths in play: d=1 for lengths,
d=2 for areas, d=3 for volumes (volume_1 p.0117, printed 109; TR-294, TR-301, TR-309).
"""

from enum import IntEnum
from typing import Final

from ra_system.constants import COLOUR_RATIO, OMEGA, OMEGA_OVER_R


class OmegaFormat(IntEnum):
    """The five Omega format levels (coherence depth tiers)."""

    RED = 0
    OMEGA_MAJOR = 1
    GREEN = 2
    OMEGA_MINOR = 3
    BLUE = 4


def all_omega_formats() -> list[OmegaFormat]:
    """Get all Omega formats in order."""
    return list(OmegaFormat)


def omega_from_harmonic(h: int) -> OmegaFormat | None:
    """Get format from harmonic index (0-4)."""
    try:
        return OmegaFormat(h)
    except ValueError:
        return None


def harmonic_from_omega(fmt: OmegaFormat) -> int:
    """Get harmonic index (0-4) from format."""
    return int(fmt)


# Position of each window on the ladder, relative to Omega Major (TR-312).
_LADDER_POSITION: Final[dict[OmegaFormat, float]] = {
    OmegaFormat.OMEGA_MAJOR: 1.0,
    OmegaFormat.RED: OMEGA_OVER_R,
    OmegaFormat.GREEN: OMEGA,
    OmegaFormat.BLUE: OMEGA * COLOUR_RATIO,
    OmegaFormat.OMEGA_MINOR: OMEGA * OMEGA,
}

# Conversion factors matrix, built from the exact ladder. It reproduces the printed
# conversion table (data/ra_constants_v2.json, omega_format_conversions) to its print width.
_CONVERSION_FACTORS: Final[dict[OmegaFormat, dict[OmegaFormat, float]]] = {
    a: {b: _LADDER_POSITION[b] / _LADDER_POSITION[a] for b in OmegaFormat} for a in OmegaFormat
}


def convert_omega(from_fmt: OmegaFormat, to_fmt: OmegaFormat, x: float, degree: int = 1) -> float:
    """Convert a value between two Omega formats.

    Args:
        from_fmt: window the value is expressed in
        to_fmt: window to express it in
        x: the value
        degree: number of lengths in play - 1 for a length, 2 for an area, 3 for a
            volume, 0 for no conversion (TR-294, TR-301)
    """
    return x * _CONVERSION_FACTORS[from_fmt][to_fmt] ** degree


def green_to_omega_major(x: float) -> float:
    """Green to Omega Major: x / Ω."""
    return convert_omega(OmegaFormat.GREEN, OmegaFormat.OMEGA_MAJOR, x)


def omega_major_to_green(x: float) -> float:
    """Omega Major to Green: x × Ω."""
    return convert_omega(OmegaFormat.OMEGA_MAJOR, OmegaFormat.GREEN, x)


def green_to_omega_minor(x: float) -> float:
    """Green to Omega Minor: x × Ω."""
    return convert_omega(OmegaFormat.GREEN, OmegaFormat.OMEGA_MINOR, x)


def omega_minor_to_green(x: float) -> float:
    """Omega Minor to Green: x / Ω."""
    return convert_omega(OmegaFormat.OMEGA_MINOR, OmegaFormat.GREEN, x)


def red_to_blue(x: float) -> float:
    """Red to Blue."""
    return convert_omega(OmegaFormat.RED, OmegaFormat.BLUE, x)


def blue_to_red(x: float) -> float:
    """Blue to Red."""
    return convert_omega(OmegaFormat.BLUE, OmegaFormat.RED, x)


ROUNDTRIP_TOLERANCE: Final[float] = 1e-9
"""Tolerance for roundtrip conversion verification (the matrix is exact up to float error)."""


def verify_omega_roundtrip(from_fmt: OmegaFormat, to_fmt: OmegaFormat, x: float) -> bool:
    """Verify Invariant C1: roundtrip conversions preserve value."""
    roundtrip = convert_omega(to_fmt, from_fmt, convert_omega(from_fmt, to_fmt, x))
    return abs(roundtrip - x) < ROUNDTRIP_TOLERANCE


def verify_all_omega_roundtrips(x: float) -> bool:
    """Verify all omega roundtrips for a value."""
    return all(
        verify_omega_roundtrip(from_fmt, to_fmt, x)
        for from_fmt in all_omega_formats()
        for to_fmt in all_omega_formats()
    )


def verify_omega_range() -> bool:
    """Verify Invariant R4: Omega format index ∈ {0, 1, 2, 3, 4}."""
    return all(harmonic_from_omega(fmt) <= 4 for fmt in all_omega_formats())
