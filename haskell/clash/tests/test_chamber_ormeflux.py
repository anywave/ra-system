"""
Test Suite for Ra.Chamber.ORMEFluxTuning (Prompt 65)
ORME Flux Chamber Tuning

Maps scalar coherence zones to ORME (Orbitally Rearranged Monoatomic Elements)
stabilization states including inertial shielding, mass-phase inversion,
and superfluid field behavior.
"""

import pytest
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895

# Alpha window thresholds for ORME stabilization
ALPHA_GROUND_MAX = 0.50
ALPHA_INVERTED_MAX = 0.72
ALPHA_SUPERFLUID_MIN = 0.72

# Shield level thresholds
SHIELD_NONE_MAX = 0.10
SHIELD_LOW_MAX = 0.30
SHIELD_MEDIUM_MAX = 0.60
SHIELD_HIGH_MAX = 0.85

# Geometry multipliers
GEOMETRY_MULTIPLIERS = {
    'Dodecahedral': 1.15,
    'Spherical': 1.08,
    'Toroidal': 1.00,
    'Custom': 1.00,
}

# =============================================================================
# ENUMS
# =============================================================================

class MassPhaseState(Enum):
    """Mass phase states for ORME zones."""
    Ground = auto()
    Inverted = auto()
    Superfluid = auto()


class ShieldLevel(Enum):
    """Discrete shield levels."""
    NoneLevel = auto()
    Low = auto()
    Medium = auto()
    High = auto()
    Critical = auto()


class TorsionPhase(Enum):
    """Torsion phase states."""
    Normal = auto()
    Inverted = auto()
    Null = auto()


class GeometryType(Enum):
    """Chamber geometry types."""
    Dodecahedral = auto()
    Spherical = auto()
    Toroidal = auto()
    Custom = auto()


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class ScalarFieldPoint:
    """Point in scalar field with coherence."""
    x: float
    y: float
    z: float
    alpha: float        # coherence (0-1)
    torsion: TorsionPhase


@dataclass
class ScalarField:
    """Scalar field representation."""
    points: List[ScalarFieldPoint]
    geometry: GeometryType
    average_alpha: float


@dataclass
class ORMEFluxZone:
    """ORME flux zone with tuning parameters."""
    zone_id: int
    resonance_band: Tuple[float, float]  # alpha window (min, max)
    mass_phase: MassPhaseState
    shielding_field: ShieldLevel
    shielding_continuous: float  # 0.0-1.0


@dataclass
class ORMECoherenceMap:
    """Complete ORME coherence map."""
    zones: List[ORMEFluxZone]
    total_shielding: float
    dominant_phase: MassPhaseState


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def classify_mass_phase(alpha: float, torsion: TorsionPhase) -> MassPhaseState:
    """Classify mass phase state from alpha and torsion.

    - Ground: alpha < 0.50
    - Inverted: 0.50 <= alpha < 0.72
    - Superfluid: alpha >= 0.72 AND torsion == Inverted
    """
    if alpha < ALPHA_GROUND_MAX:
        return MassPhaseState.Ground
    elif alpha < ALPHA_SUPERFLUID_MIN:
        return MassPhaseState.Inverted
    else:
        # Superfluid requires both high coherence AND inverted torsion
        if torsion == TorsionPhase.Inverted:
            return MassPhaseState.Superfluid
        else:
            return MassPhaseState.Inverted


def continuous_to_shield_level(continuous: float) -> ShieldLevel:
    """Map continuous shield value (0-1) to discrete level."""
    if continuous < SHIELD_NONE_MAX:
        return ShieldLevel.NoneLevel
    elif continuous < SHIELD_LOW_MAX:
        return ShieldLevel.Low
    elif continuous < SHIELD_MEDIUM_MAX:
        return ShieldLevel.Medium
    elif continuous < SHIELD_HIGH_MAX:
        return ShieldLevel.High
    else:
        return ShieldLevel.Critical


def shield_level_to_continuous(level: ShieldLevel) -> Tuple[float, float]:
    """Get continuous range for a shield level."""
    ranges = {
        ShieldLevel.NoneLevel: (0.00, 0.10),
        ShieldLevel.Low: (0.10, 0.30),
        ShieldLevel.Medium: (0.30, 0.60),
        ShieldLevel.High: (0.60, 0.85),
        ShieldLevel.Critical: (0.85, 1.00),
    }
    return ranges[level]


def get_geometry_multiplier(geometry: GeometryType) -> float:
    """Get shielding multiplier for chamber geometry."""
    return GEOMETRY_MULTIPLIERS.get(geometry.name, 1.0)


def compute_zone_shielding(
    alpha: float,
    mass_phase: MassPhaseState,
    geometry: GeometryType
) -> float:
    """Compute continuous shielding value for a zone."""
    # Base shielding from alpha
    base_shielding = alpha * 0.8

    # Phase bonus
    phase_bonus = {
        MassPhaseState.Ground: 0.0,
        MassPhaseState.Inverted: 0.1,
        MassPhaseState.Superfluid: 0.25,
    }.get(mass_phase, 0.0)

    # Geometry multiplier
    geo_mult = get_geometry_multiplier(geometry)

    # Combined shielding
    total = (base_shielding + phase_bonus) * geo_mult
    return min(1.0, max(0.0, total))


def get_alpha_band_for_phase(phase: MassPhaseState) -> Tuple[float, float]:
    """Get alpha resonance band for a mass phase."""
    if phase == MassPhaseState.Ground:
        return (0.00, ALPHA_GROUND_MAX)
    elif phase == MassPhaseState.Inverted:
        return (ALPHA_GROUND_MAX, ALPHA_INVERTED_MAX)
    else:  # Superfluid
        return (ALPHA_SUPERFLUID_MIN, 1.00)


def create_orme_zone(
    zone_id: int,
    point: ScalarFieldPoint,
    geometry: GeometryType
) -> ORMEFluxZone:
    """Create an ORME flux zone from a scalar field point."""
    mass_phase = classify_mass_phase(point.alpha, point.torsion)
    resonance_band = get_alpha_band_for_phase(mass_phase)
    shielding_continuous = compute_zone_shielding(point.alpha, mass_phase, geometry)
    shielding_level = continuous_to_shield_level(shielding_continuous)

    return ORMEFluxZone(
        zone_id=zone_id,
        resonance_band=resonance_band,
        mass_phase=mass_phase,
        shielding_field=shielding_level,
        shielding_continuous=shielding_continuous
    )


def generate_orme_map(field: ScalarField) -> ORMECoherenceMap:
    """Generate ORME coherence map from scalar field."""
    zones = []

    for i, point in enumerate(field.points):
        zone = create_orme_zone(i, point, field.geometry)
        zones.append(zone)

    # Compute aggregate values
    if zones:
        total_shielding = sum(z.shielding_continuous for z in zones) / len(zones)

        # Find dominant phase (mode)
        phase_counts = {}
        for z in zones:
            phase_counts[z.mass_phase] = phase_counts.get(z.mass_phase, 0) + 1
        dominant_phase = max(phase_counts, key=phase_counts.get)
    else:
        total_shielding = 0.0
        dominant_phase = MassPhaseState.Ground

    return ORMECoherenceMap(
        zones=zones,
        total_shielding=total_shielding,
        dominant_phase=dominant_phase
    )


def is_superfluid_active(zone: ORMEFluxZone) -> bool:
    """Check if superfluid state is active in a zone."""
    return zone.mass_phase == MassPhaseState.Superfluid


def compute_inertial_shielding(orme_map: ORMECoherenceMap) -> float:
    """Compute aggregate inertial shielding from ORME map."""
    if not orme_map.zones:
        return 0.0

    # Weight by shielding contribution
    superfluid_bonus = sum(
        0.2 for z in orme_map.zones if is_superfluid_active(z)
    )

    base_shielding = orme_map.total_shielding
    return min(1.0, base_shielding + superfluid_bonus / len(orme_map.zones))


# =============================================================================
# TEST: ALPHA WINDOW RANGES
# =============================================================================

class TestAlphaWindowRanges:
    """Test alpha window classification for ORME stabilization."""

    def test_ground_state_below_0_50(self):
        """Alpha < 0.50 maps to Ground state."""
        for alpha in [0.0, 0.1, 0.25, 0.49]:
            phase = classify_mass_phase(alpha, TorsionPhase.Normal)
            assert phase == MassPhaseState.Ground

    def test_inverted_state_0_50_to_0_72(self):
        """Alpha in [0.50, 0.72) maps to Inverted state."""
        for alpha in [0.50, 0.55, 0.65, 0.71]:
            phase = classify_mass_phase(alpha, TorsionPhase.Normal)
            assert phase == MassPhaseState.Inverted

    def test_superfluid_requires_high_alpha_and_inverted_torsion(self):
        """Superfluid requires alpha >= 0.72 AND inverted torsion."""
        # High alpha with inverted torsion -> Superfluid
        phase = classify_mass_phase(0.85, TorsionPhase.Inverted)
        assert phase == MassPhaseState.Superfluid

        # High alpha with normal torsion -> NOT Superfluid
        phase = classify_mass_phase(0.85, TorsionPhase.Normal)
        assert phase == MassPhaseState.Inverted

        # High alpha with null torsion -> NOT Superfluid
        phase = classify_mass_phase(0.90, TorsionPhase.Null)
        assert phase == MassPhaseState.Inverted

    def test_superfluid_boundary_at_0_72(self):
        """Superfluid boundary is exactly at alpha = 0.72."""
        # Just below threshold
        phase = classify_mass_phase(0.719, TorsionPhase.Inverted)
        assert phase == MassPhaseState.Inverted

        # At threshold
        phase = classify_mass_phase(0.72, TorsionPhase.Inverted)
        assert phase == MassPhaseState.Superfluid


# =============================================================================
# TEST: SUPERFLUID TRIGGERING
# =============================================================================

class TestSuperfluidTriggering:
    """Test dual-condition superfluid triggering."""

    def test_superfluid_needs_both_conditions(self):
        """Superfluid requires coherence >= 0.72 AND torsion == Inverted."""
        # Both conditions met
        assert classify_mass_phase(0.80, TorsionPhase.Inverted) == MassPhaseState.Superfluid

        # Only coherence met
        assert classify_mass_phase(0.80, TorsionPhase.Normal) != MassPhaseState.Superfluid

        # Only torsion met
        assert classify_mass_phase(0.50, TorsionPhase.Inverted) != MassPhaseState.Superfluid

    def test_near_threshold_behavior(self):
        """Test behavior near superfluid threshold."""
        # Just below threshold with inverted torsion
        phase = classify_mass_phase(0.7199, TorsionPhase.Inverted)
        assert phase == MassPhaseState.Inverted

        # At threshold with inverted torsion
        phase = classify_mass_phase(0.72, TorsionPhase.Inverted)
        assert phase == MassPhaseState.Superfluid

    def test_high_coherence_normal_torsion_is_inverted(self):
        """High coherence with normal torsion gives Inverted, not Superfluid."""
        for alpha in [0.75, 0.85, 0.95, 1.0]:
            phase = classify_mass_phase(alpha, TorsionPhase.Normal)
            assert phase == MassPhaseState.Inverted


# =============================================================================
# TEST: GEOMETRY MULTIPLIERS
# =============================================================================

class TestGeometryMultipliers:
    """Test chamber geometry influence on shielding."""

    def test_dodecahedral_gives_15_percent_bonus(self):
        """Dodecahedral geometry provides +15% shielding."""
        mult = get_geometry_multiplier(GeometryType.Dodecahedral)
        assert mult == 1.15

    def test_spherical_gives_8_percent_bonus(self):
        """Spherical geometry provides +8% shielding."""
        mult = get_geometry_multiplier(GeometryType.Spherical)
        assert mult == 1.08

    def test_toroidal_is_baseline(self):
        """Toroidal geometry is baseline (1.0)."""
        mult = get_geometry_multiplier(GeometryType.Toroidal)
        assert mult == 1.00

    def test_custom_is_baseline(self):
        """Custom geometry is baseline (1.0)."""
        mult = get_geometry_multiplier(GeometryType.Custom)
        assert mult == 1.00

    def test_geometry_affects_zone_shielding(self):
        """Geometry multiplier affects computed shielding."""
        alpha = 0.6
        phase = MassPhaseState.Inverted

        shield_toroid = compute_zone_shielding(alpha, phase, GeometryType.Toroidal)
        shield_dodeca = compute_zone_shielding(alpha, phase, GeometryType.Dodecahedral)

        assert shield_dodeca > shield_toroid
        assert abs(shield_dodeca / shield_toroid - 1.15) < 0.01


# =============================================================================
# TEST: SHIELD LEVEL ENCODING
# =============================================================================

class TestShieldLevelEncoding:
    """Test shield level discrete/continuous mapping."""

    def test_none_level_0_to_0_10(self):
        """Shield 0.00-0.10 maps to None."""
        for val in [0.0, 0.05, 0.09]:
            assert continuous_to_shield_level(val) == ShieldLevel.NoneLevel

    def test_low_level_0_10_to_0_30(self):
        """Shield 0.10-0.30 maps to Low."""
        for val in [0.10, 0.15, 0.29]:
            assert continuous_to_shield_level(val) == ShieldLevel.Low

    def test_medium_level_0_30_to_0_60(self):
        """Shield 0.30-0.60 maps to Medium."""
        for val in [0.30, 0.45, 0.59]:
            assert continuous_to_shield_level(val) == ShieldLevel.Medium

    def test_high_level_0_60_to_0_85(self):
        """Shield 0.60-0.85 maps to High."""
        for val in [0.60, 0.72, 0.84]:
            assert continuous_to_shield_level(val) == ShieldLevel.High

    def test_critical_level_0_85_to_1_00(self):
        """Shield 0.85-1.00 maps to Critical."""
        for val in [0.85, 0.92, 1.0]:
            assert continuous_to_shield_level(val) == ShieldLevel.Critical

    def test_level_to_continuous_range(self):
        """Shield level maps back to continuous range."""
        for level in ShieldLevel:
            lo, hi = shield_level_to_continuous(level)
            assert lo < hi
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0


# =============================================================================
# TEST: ORME ZONE CREATION
# =============================================================================

class TestORMEZoneCreation:
    """Test ORME flux zone creation."""

    def test_creates_zone_with_correct_id(self):
        """Zone has correct ID."""
        point = ScalarFieldPoint(0, 0, 0, 0.5, TorsionPhase.Normal)
        zone = create_orme_zone(42, point, GeometryType.Toroidal)
        assert zone.zone_id == 42

    def test_zone_has_resonance_band(self):
        """Zone has valid resonance band."""
        point = ScalarFieldPoint(0, 0, 0, 0.6, TorsionPhase.Normal)
        zone = create_orme_zone(0, point, GeometryType.Toroidal)

        assert zone.resonance_band[0] < zone.resonance_band[1]
        assert 0.0 <= zone.resonance_band[0] <= 1.0
        assert 0.0 <= zone.resonance_band[1] <= 1.0

    def test_zone_mass_phase_matches_alpha(self):
        """Zone mass phase matches alpha classification."""
        # Ground state
        point_ground = ScalarFieldPoint(0, 0, 0, 0.3, TorsionPhase.Normal)
        zone_ground = create_orme_zone(0, point_ground, GeometryType.Toroidal)
        assert zone_ground.mass_phase == MassPhaseState.Ground

        # Inverted state
        point_inv = ScalarFieldPoint(0, 0, 0, 0.6, TorsionPhase.Normal)
        zone_inv = create_orme_zone(1, point_inv, GeometryType.Toroidal)
        assert zone_inv.mass_phase == MassPhaseState.Inverted

        # Superfluid state
        point_super = ScalarFieldPoint(0, 0, 0, 0.85, TorsionPhase.Inverted)
        zone_super = create_orme_zone(2, point_super, GeometryType.Toroidal)
        assert zone_super.mass_phase == MassPhaseState.Superfluid

    def test_zone_shielding_consistency(self):
        """Zone has consistent discrete/continuous shielding."""
        point = ScalarFieldPoint(0, 0, 0, 0.7, TorsionPhase.Normal)
        zone = create_orme_zone(0, point, GeometryType.Toroidal)

        # Continuous value should map to the discrete level
        expected_level = continuous_to_shield_level(zone.shielding_continuous)
        assert zone.shielding_field == expected_level


# =============================================================================
# TEST: GENERATE ORME MAP
# =============================================================================

class TestGenerateORMEMap:
    """Test ORME coherence map generation."""

    def test_generates_map_from_field(self):
        """Generates ORME map from scalar field."""
        field = create_test_scalar_field(num_points=5)
        orme_map = generate_orme_map(field)

        assert isinstance(orme_map, ORMECoherenceMap)
        assert len(orme_map.zones) == 5

    def test_zones_have_sequential_ids(self):
        """Generated zones have sequential IDs."""
        field = create_test_scalar_field(num_points=4)
        orme_map = generate_orme_map(field)

        ids = [z.zone_id for z in orme_map.zones]
        assert ids == [0, 1, 2, 3]

    def test_total_shielding_is_average(self):
        """Total shielding is average of zone shielding."""
        field = create_test_scalar_field(num_points=3, uniform_alpha=0.5)
        orme_map = generate_orme_map(field)

        expected_avg = sum(z.shielding_continuous for z in orme_map.zones) / 3
        assert abs(orme_map.total_shielding - expected_avg) < 0.01

    def test_dominant_phase_is_mode(self):
        """Dominant phase is the most common phase."""
        # Create field with mostly ground state
        points = [
            ScalarFieldPoint(0, 0, 0, 0.2, TorsionPhase.Normal),  # Ground
            ScalarFieldPoint(0, 0, 0, 0.3, TorsionPhase.Normal),  # Ground
            ScalarFieldPoint(0, 0, 0, 0.4, TorsionPhase.Normal),  # Ground
            ScalarFieldPoint(0, 0, 0, 0.6, TorsionPhase.Normal),  # Inverted
        ]
        field = ScalarField(points, GeometryType.Toroidal, 0.375)
        orme_map = generate_orme_map(field)

        assert orme_map.dominant_phase == MassPhaseState.Ground

    def test_empty_field_produces_empty_map(self):
        """Empty field produces empty map with defaults."""
        field = ScalarField([], GeometryType.Toroidal, 0.0)
        orme_map = generate_orme_map(field)

        assert len(orme_map.zones) == 0
        assert orme_map.total_shielding == 0.0
        assert orme_map.dominant_phase == MassPhaseState.Ground


# =============================================================================
# TEST: LIVE UPDATES
# =============================================================================

class TestLiveUpdates:
    """Test ORME map reacts to live scalar field updates."""

    def test_map_updates_with_new_field(self):
        """Map updates when field changes."""
        field1 = create_test_scalar_field(num_points=3, uniform_alpha=0.3)
        map1 = generate_orme_map(field1)

        field2 = create_test_scalar_field(num_points=3, uniform_alpha=0.8)
        map2 = generate_orme_map(field2)

        # Higher alpha should produce higher shielding
        assert map2.total_shielding > map1.total_shielding

    def test_phase_transition_reflected_in_map(self):
        """Phase transitions reflected in map."""
        # Low coherence field
        low_field = create_test_scalar_field(num_points=2, uniform_alpha=0.3)
        low_map = generate_orme_map(low_field)
        assert low_map.dominant_phase == MassPhaseState.Ground

        # High coherence field with inverted torsion
        points = [
            ScalarFieldPoint(0, 0, 0, 0.85, TorsionPhase.Inverted),
            ScalarFieldPoint(0, 0, 0, 0.90, TorsionPhase.Inverted),
        ]
        high_field = ScalarField(points, GeometryType.Toroidal, 0.875)
        high_map = generate_orme_map(high_field)
        assert high_map.dominant_phase == MassPhaseState.Superfluid


# =============================================================================
# TEST: INERTIAL SHIELDING
# =============================================================================

class TestInertialShielding:
    """Test inertial shielding computation."""

    def test_shielding_increases_with_coherence(self):
        """Inertial shielding increases with field coherence."""
        low_field = create_test_scalar_field(num_points=3, uniform_alpha=0.3)
        low_map = generate_orme_map(low_field)
        low_shielding = compute_inertial_shielding(low_map)

        high_field = create_test_scalar_field(num_points=3, uniform_alpha=0.7)
        high_map = generate_orme_map(high_field)
        high_shielding = compute_inertial_shielding(high_map)

        assert high_shielding > low_shielding

    def test_superfluid_zones_boost_shielding(self):
        """Superfluid zones provide shielding boost."""
        # Field without superfluid
        normal_points = [
            ScalarFieldPoint(0, 0, 0, 0.8, TorsionPhase.Normal),
            ScalarFieldPoint(0, 0, 0, 0.8, TorsionPhase.Normal),
        ]
        normal_field = ScalarField(normal_points, GeometryType.Toroidal, 0.8)
        normal_map = generate_orme_map(normal_field)
        normal_shielding = compute_inertial_shielding(normal_map)

        # Field with superfluid
        super_points = [
            ScalarFieldPoint(0, 0, 0, 0.8, TorsionPhase.Inverted),
            ScalarFieldPoint(0, 0, 0, 0.8, TorsionPhase.Inverted),
        ]
        super_field = ScalarField(super_points, GeometryType.Toroidal, 0.8)
        super_map = generate_orme_map(super_field)
        super_shielding = compute_inertial_shielding(super_map)

        assert super_shielding > normal_shielding

    def test_shielding_capped_at_1(self):
        """Inertial shielding is capped at 1.0."""
        # Very high coherence field
        points = [
            ScalarFieldPoint(0, 0, 0, 1.0, TorsionPhase.Inverted),
            ScalarFieldPoint(0, 0, 0, 1.0, TorsionPhase.Inverted),
            ScalarFieldPoint(0, 0, 0, 1.0, TorsionPhase.Inverted),
        ]
        field = ScalarField(points, GeometryType.Dodecahedral, 1.0)
        orme_map = generate_orme_map(field)
        shielding = compute_inertial_shielding(orme_map)

        assert shielding <= 1.0


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestORMEIntegration:
    """Integration tests for ORME flux tuning."""

    def test_full_pipeline_field_to_map(self):
        """Full pipeline from scalar field to ORME map."""
        points = [
            ScalarFieldPoint(1, 0, 0, 0.3, TorsionPhase.Normal),
            ScalarFieldPoint(0, 1, 0, 0.55, TorsionPhase.Normal),
            ScalarFieldPoint(0, 0, 1, 0.78, TorsionPhase.Inverted),
            ScalarFieldPoint(-1, 0, 0, 0.9, TorsionPhase.Inverted),
        ]
        field = ScalarField(points, GeometryType.Dodecahedral, 0.6325)

        orme_map = generate_orme_map(field)

        assert len(orme_map.zones) == 4
        assert orme_map.zones[0].mass_phase == MassPhaseState.Ground
        assert orme_map.zones[1].mass_phase == MassPhaseState.Inverted
        assert orme_map.zones[2].mass_phase == MassPhaseState.Superfluid
        assert orme_map.zones[3].mass_phase == MassPhaseState.Superfluid

    def test_geometry_affects_total_shielding(self):
        """Chamber geometry affects total shielding."""
        points = [
            ScalarFieldPoint(0, 0, 0, 0.7, TorsionPhase.Normal),
            ScalarFieldPoint(1, 0, 0, 0.7, TorsionPhase.Normal),
        ]

        toroid_field = ScalarField(points, GeometryType.Toroidal, 0.7)
        toroid_map = generate_orme_map(toroid_field)

        dodeca_field = ScalarField(points, GeometryType.Dodecahedral, 0.7)
        dodeca_map = generate_orme_map(dodeca_field)

        assert dodeca_map.total_shielding > toroid_map.total_shielding

    def test_all_phases_representable(self):
        """All mass phases can be represented in map."""
        points = [
            ScalarFieldPoint(0, 0, 0, 0.2, TorsionPhase.Normal),   # Ground
            ScalarFieldPoint(0, 0, 0, 0.6, TorsionPhase.Normal),   # Inverted
            ScalarFieldPoint(0, 0, 0, 0.9, TorsionPhase.Inverted), # Superfluid
        ]
        field = ScalarField(points, GeometryType.Toroidal, 0.567)
        orme_map = generate_orme_map(field)

        phases = {z.mass_phase for z in orme_map.zones}
        assert MassPhaseState.Ground in phases
        assert MassPhaseState.Inverted in phases
        assert MassPhaseState.Superfluid in phases


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_scalar_field(
    num_points: int = 4,
    uniform_alpha: float = 0.5,
    geometry: GeometryType = GeometryType.Toroidal
) -> ScalarField:
    """Create test scalar field with uniform alpha."""
    points = [
        ScalarFieldPoint(
            x=float(i),
            y=0.0,
            z=0.0,
            alpha=uniform_alpha,
            torsion=TorsionPhase.Normal
        )
        for i in range(num_points)
    ]
    return ScalarField(points, geometry, uniform_alpha)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
