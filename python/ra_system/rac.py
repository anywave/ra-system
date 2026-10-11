"""
RacLevel enum (RAC1..RAC6) with validation.

Resonant Access Constants (RACs) represent access sensitivity levels.
RAC1 is the highest (least restrictive), RAC6 is the lowest (most restrictive).

The access-level reading is this framework's modelling layer. In Bateman's corpus a
Rac is an arc of Red Pi, a turn of a measuring wheel (TR-298); the values below are
his, the access semantics are not.

Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Final

from ra_system.constants import ANKH, RED_PI


class RacLevel(IntEnum):
    """The six Resonant Access Constant levels."""

    RAC1 = 1
    RAC2 = 2
    RAC3 = 3
    RAC4 = 4
    RAC5 = 5
    RAC6 = 6


# Rac units as fractions of Red Pi (TR-298; vol2b g00585_src09_p0038, printed 418).
# Every Rac is an ARC of Red Pi: a turn of a wheel, not a free-standing length.
# Degrees depend on the circle the arc is laid on: on the Ankh measuring wheel
# (circumference 5.08938, diameter 1.62; volume_1 p.0168) RAC 1 is a 45-degree turn.
RAC_UNITS_OF_RED_PI: Final[dict[str, float]] = {
    "RAC 1": 81 / 400,  # 3^4/400 = 1.62/8
    "RAC 2": 1 / 5,
    "RAC 3": 729 / 4000,  # 3^6/4000
    "RAC 4": 1 / 6,
    "RAC 5": 729 / 5000,  # 3^6/5000
    "RAC 6": 0.09 * math.sqrt(2),
    "RAC 7": math.sqrt(2) / 8.64,
    # MIR Rac: the printed divisor .6981131687 is a digit permutation of .6981316871 (TR-298 s5)
    "MIR Rac": 2 / 9,
    "Block Rac 1": 27 / 125,  # 3^3/125
    "Block Rac 2": 8 / 45,
}

# RAC values in Red Rams = Red Pi x fraction (TR-298). Printed values agree to their
# last digit: .6361725, .628318519, .57255525, .523598765, .4580442, .3998594565.
_RAC_VALUES: Final[dict[RacLevel, float]] = {
    level: RED_PI * RAC_UNITS_OF_RED_PI[f"RAC {int(level)}"] for level in RacLevel
}

# RAC values in meters
_RAC_VALUES_METERS: Final[dict[RacLevel, float]] = {
    RacLevel.RAC1: 0.639591666,
    RacLevel.RAC2: 0.631695473,
    RacLevel.RAC3: 0.5756325,
    RacLevel.RAC4: 0.526412894,
    RacLevel.RAC5: 0.460506,
    RacLevel.RAC6: 0.4020085371,
}

# Pyramid divisions: base / RAC, with the base side = (729/10) x Red Pi Red Rams
# (TR-331). So base / RAC 2 = 729/2 = 364.5, the Balmer constant, is forced.
_PYRAMID_DIVISIONS: Final[dict[RacLevel, float]] = {
    RacLevel.RAC1: 360.0,       # 72.9 / (81/400)
    RacLevel.RAC2: 364.5,       # 72.9 x 5 = 729/2, the Balmer constant
    RacLevel.RAC3: 400.0,       # 72.9 / (729/4000)
    RacLevel.RAC4: 437.4,       # 72.9 x 6
    RacLevel.RAC5: 500.0,       # 72.9 / (729/5000)
    RacLevel.RAC6: 572.756493,  # 72.9 / (0.09 sqrt2) = 810 / sqrt2
}


@dataclass(frozen=True)
class RacValue:
    """RAC value in Red Rams (must be 0 < x < 1)."""

    value: float

    def __post_init__(self) -> None:
        if not (0 < self.value < 1):
            raise ValueError(f"RAC value must be in (0, 1), got {self.value}")


def all_rac_levels() -> list[RacLevel]:
    """Get all RAC levels in order."""
    return list(RacLevel)


def rac_from_level(n: int) -> RacLevel | None:
    """Get RacLevel from numeric level (1-6)."""
    try:
        return RacLevel(n)
    except ValueError:
        return None


def rac_value(level: RacLevel) -> float:
    """Get the RAC value in Red Rams for a given level."""
    return _RAC_VALUES[level]


def rac_value_meters(level: RacLevel) -> float:
    """Get the RAC value in meters for a given level."""
    return _RAC_VALUES_METERS[level]


def rac_value_normalized(level: RacLevel) -> float:
    """Get the RAC value normalized to RAC1 (RAC1 normalized = 1.0)."""
    return rac_value(level) / rac_value(RacLevel.RAC1)


def pyramid_division(level: RacLevel) -> float:
    """Get pyramid division for a RAC level."""
    return _PYRAMID_DIVISIONS[level]


def is_valid_rac_value(x: float) -> bool:
    """Check if a value is valid for a RAC (0 < x < 1)."""
    return 0 < x < 1


def verify_rac_ordering() -> bool:
    """Verify Invariant O1: RAC1 > RAC2 > RAC3 > RAC4 > RAC5 > RAC6 > 0."""
    values = [rac_value(level) for level in all_rac_levels()]
    # Check descending order
    for i in range(len(values) - 1):
        if values[i] <= values[i + 1]:
            return False
    # Check all positive
    return all(v > 0 for v in values)


def verify_rac_range() -> bool:
    """Verify Invariant R1: 0 < RAC(i) < 1 for all i ∈ [1, 6]."""
    return all(is_valid_rac_value(rac_value(level)) for level in all_rac_levels())


def verify_rac1_derivation() -> bool:
    """Verify Invariant I2: RAC1 = Ankh / 8."""
    computed = ANKH / 8
    return abs(rac_value(RacLevel.RAC1) - computed) < 0.0001
