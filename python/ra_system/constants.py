"""
Ra System fundamental constants with type-safe wrappers.

Values are Wesley H. Bateman's, from "The Rods of Amon Ra" (1992-1997), as printed
and as checked page by page by the lattice's TR node in Aug-Sep 2026. They are the
author's own system of units and ratios. They are not physics claims: where a name
borrows a physics term (Rydberg, fine structure, H-Bar), the value is Bateman's,
and the comment says how it differs from the physical constant.

Citations use the lattice post id (TR-###) and the archive page id or printed folio.
Most of Bateman's constants are exact closed forms over one numerator,
84823 = round(pi x 27000) (TR-291), so they are written that way here.
"""

import math
from dataclasses import dataclass
from typing import Final

_SQRT2: Final[float] = math.sqrt(2)

# ---------------------------------------------------------------------------
# Colour ladder ratios
# ---------------------------------------------------------------------------

COLOUR_RATIO: Final[float] = 60000 * _SQRT2 / 84823
"""Colour ratio r = Green/Red = Blue/Green = 60000 sqrt2 / 84823 = 1.000351482.

TR-291, TR-300 (closed form); printed "Red Rams x 1.000351482 = Green Rams" at
vol2a g00215_src01_p0031 (TR-349). Applies as r**d, where d is the number of
lengths in play: d=2 for areas (volume_1 p.0117, printed 109; TR-294, TR-301).
"""

OMEGA: Final[float] = 32 * _SQRT2 / 45
"""Omega Ratio (Q-Ratio) = 32 sqrt2 / 45 = 1.005662978.

TR-304 (closed form); printed 1.005662978 in the vol2a prologue (TR-302).
Omega Major = Green / Omega, Omega Minor = Green x Omega.
"""

OMEGA_OVER_R: Final[float] = 84823 / 84375
"""Outer ladder rung Omega / r = 84823/84375 = 1.005309630 (TR-308, TR-309, TR-312).

The step from Omega Major to Red and from Blue to Omega Minor.
"""

# ---------------------------------------------------------------------------
# Pi variants (chromatic)
# ---------------------------------------------------------------------------

RED_PI: Final[float] = 84823 / 27000
"""Red Pi = 84823/27000 = 3.141592592... (TR-291, TR-349).

Not pi: it is pi rounded at a base-27 scale and differs from pi at the 8th decimal.
Sites: vol2a p.167, volume_1 p.0150 (TR-291).
"""

GREEN_PI: Final[float] = 20 / 9 * _SQRT2
"""Green Pi = (20/9) sqrt2 = 3.142696805 (TR-300).

Named GREEN PI at volume_1 p.0129, procedure 7-3 (TR-301); Green Pi squared = 800/81
(volume_1 p.0105, TR-323). Green Pi / Omega = 25/8 and Green Pi x Omega = 256/81 (TR-304).
"""

BLUE_PI: Final[float] = GREEN_PI * COLOUR_RATIO
"""Blue Pi = Green Pi x r = 3.143801406 (TR-308, volume_1 p.0122, printed 114)."""

# ---------------------------------------------------------------------------
# Phi variants (chromatic) - Bateman's own erratum, supplement p.0002
# ---------------------------------------------------------------------------
# Supplement g00857_src03_p0002 prints the RA PHI VALUES row
#   OMEGA MAJOR 1.62 | RED 1.6286016 | GREEN 1.629174025 | BLUE 1.629746650
# and states that the Red Phi used "on each and every data form" of Volumes 2 and 3,
# 1.619430799 (= 1.62 / 1.000351482), was erroneous (FP-1399 s2-s3, TR-312, TR-316).
# Volume 1 p.0027 and the vol2a p.0010 master table call 1.62 the GREEN Phi; the
# erratum supersedes them (TR-312). The Omega Minor cell is blank on that sheet.

OMEGA_MAJOR_PHI: Final[float] = 1.62
"""Omega Major Phi = 1.62 (supplement p.0002; TR-312).

Also the diameter of the Ankh measuring wheel (volume_1 p.0168; TR-298), so
Ankh = Red Pi x 1.62. Not the golden ratio (1.618033989, printed beside it).
"""

RED_PHI: Final[float] = 1.6286016
"""Red Phi = 1.6286016 = 1.62 x 84823/84375 (supplement p.0002 erratum; TR-312, TR-316)."""

GREEN_PHI: Final[float] = 1.629174025
"""Green Phi = 1.629174025 (supplement p.0002 erratum; TR-312). Red Phi x r at print width.

The erratum names only Red Phi on the Vol 2/3 forms; overriding vol1 p.0027 (which calls 1.62 the
Green Phi) is TR's inference (TR-596): the withdrawn Red 1.62/r assumed 1.62 was Green, and the
correction 1.62 x 84823/84375 places 1.62 at Omega Major. Ruled to keep (TR-596)."""

BLUE_PHI: Final[float] = 1.629746650
"""Blue Phi = 1.629746650 (supplement p.0002 erratum; TR-312). Green Phi x r at print width."""

# ---------------------------------------------------------------------------
# Hunab, H-Bar and the Ankh
# ---------------------------------------------------------------------------

HUNAB: Final[float] = 84823 / 80000
"""Hunab (Red) = 84823/80000 = 1.0602875 (TR-291; volume_1 p.0150).

The corpus labels 1.0602875 HUNAB 247 times (FP-1399). It is not 2**(1/12)
(1.059463); a third-party survey he reproduces converts Hunabs at 1.05943 and he
assigns his own value (TR-308 s4).
"""

GREEN_HUNAB: Final[float] = 0.75 * _SQRT2
"""Green Hunab = (3/4) sqrt2 = 1.060660172 = Hunab x r exactly (TR-320, FP-1430, TR-327)."""

H_BAR: Final[float] = 135 / 128
"""H-Bar = 135/128 = 1.0546875 = Green Hunab / Omega, the Omega Major window of the Hunab row.

TR-304: vol2a g00194_src01_p0010 "OMEGA LEVELS: KEY NUMBERS"; TR-312: supplement p.0002.
Bateman's value, not the physical reduced Planck constant.
"""

ANKH: Final[float] = 3 * 84823 / 50000
"""Ankh = 5.08938 = 3 x 84823/50000 = Red Pi x 1.62 (TR-329; volume_1 p.0168 wheel, TR-298)."""

# ---------------------------------------------------------------------------
# Spectral / 729 family
# ---------------------------------------------------------------------------

FINE_STRUCTURE: Final[float] = 1 / 729
"""Bateman's "fine structure" unit f.s.Delta = 1/729 = Repitan(1)^2 = 0.0013717421.

Defined on volume_1 p.0105 (printed 97; TR-323) and as 1/729 of a degree on volume_1
folio 165 (TR-331). Not the physical fine-structure constant (0.0072973526, TR-323).
Printed 10x too small in the Supplement (0.0001371742, TR-281) and as .0013742113 on
volume_1 p.0105 (dropped digit pair, TR-323); both are transcription damage.
"""

RA_RYDBERG: Final[float] = 800 / 729
"""Bateman's Ra Rydberg = 800/729 = 1.097393690 (x1e5 = 109,739.369 units per Ra cm).

Volume_1 p.0126 calls it the Green Rydberg Constant (TR-320); its reciprocal .91125
(= 729/800) at volume_1 p.0130 (TR-322); vol3 p.0070 and p.0048 (TR-329). The vol2a
KEY NUMBERS sheet carries 729/800 in its Omega Major column (TR-304), so which window
owns the value is open. Not CODATA R_inf: his tables run ~545 ppm off measured lines,
and that offset is authorial - do not normalise it (TR-280 s7.3, TR-283).
"""

BALMER_CONSTANT: Final[float] = 729 / 2
"""Balmer constant = 364.5 = 729/2, labelled in the cell (TR-288, BAT-BALC-001).

Equal to pyramid base / RAC 2 with base = (729/10) x Red Pi (TR-331).
"""


@dataclass(frozen=True)
class AnkhValue:
    """Typed wrapper for Ankh values."""

    value: float = ANKH

    def derive_rac1(self) -> float:
        """Derive RAC1 from Ankh (Invariant I2: RAC1 = Ankh / 8)."""
        return self.value / 8


@dataclass(frozen=True)
class OmegaRatio:
    """Typed wrapper for Omega Ratio."""

    value: float = OMEGA

    def reciprocal(self) -> float:
        """Get the reciprocal."""
        return 1 / self.value


def verify_ankh_invariant() -> bool:
    """Verify Invariant I1: Ankh = Red Pi x 1.62 (the Ankh wheel diameter, Omega Major Phi)."""
    computed = RED_PI * OMEGA_MAJOR_PHI
    return abs(ANKH - computed) < 1e-12


def verify_rac1_invariant(rac1_value: float) -> bool:
    """Verify Invariant I2: RAC1 = Ankh / 8."""
    computed = ANKH / 8
    return abs(rac1_value - computed) < 0.0001


def verify_hbar_invariant() -> bool:
    """Verify Invariant I3: H-Bar = Green Hunab / Omega = 135/128 (TR-304)."""
    computed = GREEN_HUNAB / OMEGA
    return abs(H_BAR - computed) < 1e-12


def verify_fine_structure_invariant() -> bool:
    """Verify Invariant I6: f.s.Delta = Repitan(1)^2 = (1/27)^2 = 1/729."""
    r1 = 1 / 27
    return abs(FINE_STRUCTURE - r1 * r1) < 1e-15
