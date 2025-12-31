"""
Test Suite for Ra.Energy.TapNodeLocator (Prompt 66)
Radiant Energy Tap Nodes

Identifies passive tap points in scalar fields where energy may be
extracted via harmonic convergence and geometry alignment.
"""

import pytest
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895

# Convergence thresholds
ALPHA_CONVERGENCE_MIN = 0.88
VECTOR_INWARDNESS_MIN = 0.7

# Toroidal special threshold
TOROIDAL_ALPHA_MIN = 0.82

# Dodecahedral flux amplification
DODECAHEDRAL_FLUX_AMP = 1.20

# Avatar sync thresholds
FREQUENCY_TOLERANCE = 0.05  # ±5%
HRV_COHERENCE_MIN = 0.6


# =============================================================================
# ENUMS
# =============================================================================

class ChamberForm(Enum):
    """Chamber geometry types."""
    Toroidal = auto()
    Dodecahedral = auto()
    Spherical = auto()
    Custom = auto()


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class RaCoordinate:
    """3D coordinate in Ra space."""
    x: float
    y: float
    z: float

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'RaCoordinate':
        mag = self.magnitude()
        if mag < 1e-10:
            return RaCoordinate(0, 0, 0)
        return RaCoordinate(self.x / mag, self.y / mag, self.z / mag)


@dataclass
class GradientVector:
    """Scalar field gradient vector."""
    dx: float
    dy: float
    dz: float

    def magnitude(self) -> float:
        return math.sqrt(self.dx**2 + self.dy**2 + self.dz**2)

    def normalized(self) -> 'GradientVector':
        mag = self.magnitude()
        if mag < 1e-10:
            return GradientVector(0, 0, 0)
        return GradientVector(self.dx / mag, self.dy / mag, self.dz / mag)


@dataclass
class AppendageResonance:
    """Avatar appendage resonance state."""
    frequency: float       # Hz
    amplitude: float       # 0-1
    hrv_coherence: float   # 0-1


@dataclass
class ScalarFieldPoint:
    """Point in scalar field."""
    coordinate: RaCoordinate
    alpha: float           # Coherence 0-1
    gradient: GradientVector


@dataclass
class ScalarField:
    """Scalar field representation."""
    points: List[ScalarFieldPoint]
    center: RaCoordinate


@dataclass
class Avatar:
    """Avatar with appendage resonance."""
    position: RaCoordinate
    appendage: AppendageResonance


@dataclass
class TapNode:
    """Energy tap node."""
    coordinate: RaCoordinate
    scalar_alpha: float
    flux_vector: GradientVector
    is_active: bool
    geometry_link: Optional[ChamberForm]
    avatar_sync: Optional[AppendageResonance]
    inwardness: float      # Dot product with radial


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_radial_vector(point: RaCoordinate, center: RaCoordinate) -> RaCoordinate:
    """Compute radial vector from center to point (inward is negative)."""
    return RaCoordinate(
        center.x - point.x,
        center.y - point.y,
        center.z - point.z
    ).normalized()


def compute_inwardness(gradient: GradientVector, radial: RaCoordinate) -> float:
    """Compute dot product of gradient with inward radial vector."""
    return (
        gradient.dx * radial.x +
        gradient.dy * radial.y +
        gradient.dz * radial.z
    )


def check_convergence(alpha: float, inwardness: float, chamber: Optional[ChamberForm] = None) -> bool:
    """Check if point meets convergence criteria."""
    # Toroidal has lower alpha threshold
    if chamber == ChamberForm.Toroidal:
        alpha_threshold = TOROIDAL_ALPHA_MIN
    else:
        alpha_threshold = ALPHA_CONVERGENCE_MIN

    return alpha >= alpha_threshold and inwardness >= VECTOR_INWARDNESS_MIN


def compute_flux_magnitude(gradient: GradientVector, chamber: Optional[ChamberForm] = None) -> float:
    """Compute flux magnitude with chamber amplification."""
    base_flux = gradient.magnitude()

    if chamber == ChamberForm.Dodecahedral:
        return base_flux * DODECAHEDRAL_FLUX_AMP
    else:
        return base_flux


def check_avatar_sync(
    tap_frequency: float,
    appendage: AppendageResonance
) -> bool:
    """Check if avatar appendage is in sync with tap node."""
    # Frequency within ±5%
    freq_ratio = appendage.frequency / tap_frequency if tap_frequency > 0 else 0
    freq_match = abs(freq_ratio - 1.0) <= FREQUENCY_TOLERANCE

    # HRV coherence threshold
    hrv_match = appendage.hrv_coherence >= HRV_COHERENCE_MIN

    return freq_match and hrv_match


def estimate_tap_frequency(alpha: float, gradient: GradientVector) -> float:
    """Estimate tap node harmonic frequency from field properties."""
    # Base frequency scaled by alpha and gradient magnitude
    base_freq = 7.83  # Schumann resonance base
    return base_freq * (1 + alpha) * (1 + gradient.magnitude() * 0.1)


def create_tap_node(
    point: ScalarFieldPoint,
    center: RaCoordinate,
    chamber: Optional[ChamberForm],
    avatar: Optional[Avatar]
) -> Optional[TapNode]:
    """Create tap node from field point if it meets criteria."""
    # Compute radial and inwardness
    radial = compute_radial_vector(point.coordinate, center)
    gradient_norm = point.gradient.normalized()
    inwardness = compute_inwardness(gradient_norm, radial)

    # Check convergence
    if not check_convergence(point.alpha, inwardness, chamber):
        return None

    # Compute flux
    flux_magnitude = compute_flux_magnitude(point.gradient, chamber)
    flux_vector = GradientVector(
        gradient_norm.dx * flux_magnitude,
        gradient_norm.dy * flux_magnitude,
        gradient_norm.dz * flux_magnitude
    )

    # Check avatar sync
    tap_freq = estimate_tap_frequency(point.alpha, point.gradient)
    avatar_sync = None
    is_active = True

    if avatar:
        if check_avatar_sync(tap_freq, avatar.appendage):
            avatar_sync = avatar.appendage
        else:
            is_active = False  # No sync means inactive

    return TapNode(
        coordinate=point.coordinate,
        scalar_alpha=point.alpha,
        flux_vector=flux_vector,
        is_active=is_active,
        geometry_link=chamber,
        avatar_sync=avatar_sync,
        inwardness=inwardness
    )


def locate_tap_nodes(
    field: ScalarField,
    avatar: Avatar,
    chamber: ChamberForm
) -> List[TapNode]:
    """Locate all tap nodes in scalar field."""
    tap_nodes = []

    for point in field.points:
        node = create_tap_node(point, field.center, chamber, avatar)
        if node:
            tap_nodes.append(node)

    return tap_nodes


# =============================================================================
# TEST: CONVERGENCE THRESHOLD
# =============================================================================

class TestConvergenceThreshold:
    """Test convergence threshold detection."""

    def test_alpha_above_088_and_inward_above_07_converges(self):
        """High alpha and inwardness meets convergence."""
        assert check_convergence(0.90, 0.75) is True
        assert check_convergence(0.88, 0.70) is True

    def test_alpha_below_088_does_not_converge(self):
        """Alpha below 0.88 fails convergence (non-toroidal)."""
        assert check_convergence(0.87, 0.80) is False
        assert check_convergence(0.50, 0.90) is False

    def test_inwardness_below_07_does_not_converge(self):
        """Inwardness below 0.7 fails convergence."""
        assert check_convergence(0.95, 0.65) is False
        assert check_convergence(0.90, 0.50) is False

    def test_toroidal_allows_lower_alpha(self):
        """Toroidal chamber allows α ≥ 0.82."""
        assert check_convergence(0.85, 0.75, ChamberForm.Toroidal) is True
        assert check_convergence(0.82, 0.70, ChamberForm.Toroidal) is True
        assert check_convergence(0.81, 0.70, ChamberForm.Toroidal) is False


# =============================================================================
# TEST: CHAMBER GEOMETRY MODULATION
# =============================================================================

class TestChamberGeometry:
    """Test chamber geometry effects."""

    def test_toroidal_lowers_alpha_threshold(self):
        """Toroidal allows active gate at α ≥ 0.82."""
        # Would fail with standard threshold
        assert check_convergence(0.84, 0.75, None) is False
        # Passes with toroidal
        assert check_convergence(0.84, 0.75, ChamberForm.Toroidal) is True

    def test_dodecahedral_amplifies_flux(self):
        """Dodecahedral provides +20% flux amplification."""
        gradient = GradientVector(1.0, 0.0, 0.0)

        base_flux = compute_flux_magnitude(gradient, None)
        dodeca_flux = compute_flux_magnitude(gradient, ChamberForm.Dodecahedral)

        assert abs(dodeca_flux - base_flux * 1.20) < 0.01

    def test_spherical_is_neutral(self):
        """Spherical chamber has neutral effect."""
        gradient = GradientVector(1.0, 0.0, 0.0)

        base_flux = compute_flux_magnitude(gradient, None)
        spherical_flux = compute_flux_magnitude(gradient, ChamberForm.Spherical)

        assert abs(spherical_flux - base_flux) < 0.01


# =============================================================================
# TEST: AVATAR SYNC
# =============================================================================

class TestAvatarSync:
    """Test avatar appendage synchronization."""

    def test_frequency_within_5_percent_syncs(self):
        """Frequency match within ±5% syncs."""
        tap_freq = 10.0
        appendage = AppendageResonance(frequency=10.0, amplitude=0.8, hrv_coherence=0.7)
        assert check_avatar_sync(tap_freq, appendage) is True

        appendage = AppendageResonance(frequency=10.4, amplitude=0.8, hrv_coherence=0.7)
        assert check_avatar_sync(tap_freq, appendage) is True  # +4%

        appendage = AppendageResonance(frequency=9.6, amplitude=0.8, hrv_coherence=0.7)
        assert check_avatar_sync(tap_freq, appendage) is True  # -4%

    def test_frequency_outside_5_percent_fails(self):
        """Frequency outside ±5% fails sync."""
        tap_freq = 10.0
        appendage = AppendageResonance(frequency=10.6, amplitude=0.8, hrv_coherence=0.7)
        assert check_avatar_sync(tap_freq, appendage) is False  # +6%

        appendage = AppendageResonance(frequency=9.4, amplitude=0.8, hrv_coherence=0.7)
        assert check_avatar_sync(tap_freq, appendage) is False  # -6%

    def test_hrv_coherence_below_06_fails(self):
        """HRV coherence below 0.6 fails sync."""
        tap_freq = 10.0
        appendage = AppendageResonance(frequency=10.0, amplitude=0.8, hrv_coherence=0.5)
        assert check_avatar_sync(tap_freq, appendage) is False

    def test_both_conditions_required(self):
        """Both frequency and HRV must match."""
        tap_freq = 10.0

        # Good frequency, bad HRV
        appendage = AppendageResonance(frequency=10.0, amplitude=0.8, hrv_coherence=0.4)
        assert check_avatar_sync(tap_freq, appendage) is False

        # Bad frequency, good HRV
        appendage = AppendageResonance(frequency=8.0, amplitude=0.8, hrv_coherence=0.9)
        assert check_avatar_sync(tap_freq, appendage) is False


# =============================================================================
# TEST: FLUX VECTOR
# =============================================================================

class TestFluxVector:
    """Test flux vector computation."""

    def test_gradient_normalized(self):
        """Gradient is normalized for flux direction."""
        gradient = GradientVector(3.0, 4.0, 0.0)
        norm = gradient.normalized()

        assert abs(norm.dx - 0.6) < 0.01
        assert abs(norm.dy - 0.8) < 0.01
        assert abs(norm.dz) < 0.01

    def test_flux_magnitude_computed(self):
        """Flux magnitude computed from gradient."""
        gradient = GradientVector(3.0, 4.0, 0.0)  # magnitude = 5
        flux = compute_flux_magnitude(gradient)
        assert abs(flux - 5.0) < 0.01


# =============================================================================
# TEST: TAP NODE CREATION
# =============================================================================

class TestTapNodeCreation:
    """Test tap node creation."""

    def test_creates_node_when_criteria_met(self):
        """Creates tap node when all criteria met."""
        center = RaCoordinate(0, 0, 0)
        point = ScalarFieldPoint(
            coordinate=RaCoordinate(1, 0, 0),
            alpha=0.92,
            gradient=GradientVector(-0.8, 0, 0)  # Pointing inward
        )
        # Tap freq = 7.83 * (1+0.92) * (1+0.8*0.1) ≈ 16.23, use freq within ±5%
        avatar = Avatar(
            position=RaCoordinate(0.5, 0, 0),
            appendage=AppendageResonance(frequency=16.2, amplitude=0.8, hrv_coherence=0.7)
        )

        node = create_tap_node(point, center, ChamberForm.Spherical, avatar)

        assert node is not None
        assert node.scalar_alpha == 0.92
        assert node.is_active is True

    def test_no_node_when_alpha_too_low(self):
        """No tap node when alpha below threshold."""
        center = RaCoordinate(0, 0, 0)
        point = ScalarFieldPoint(
            coordinate=RaCoordinate(1, 0, 0),
            alpha=0.50,  # Too low
            gradient=GradientVector(-0.8, 0, 0)
        )

        node = create_tap_node(point, center, None, None)
        assert node is None

    def test_no_node_when_gradient_outward(self):
        """No tap node when gradient points outward."""
        center = RaCoordinate(0, 0, 0)
        point = ScalarFieldPoint(
            coordinate=RaCoordinate(1, 0, 0),
            alpha=0.95,
            gradient=GradientVector(0.8, 0, 0)  # Pointing outward
        )

        node = create_tap_node(point, center, None, None)
        assert node is None


# =============================================================================
# TEST: LOCATE TAP NODES
# =============================================================================

class TestLocateTapNodes:
    """Test tap node location in scalar field."""

    def test_finds_tap_nodes_in_field(self):
        """Finds tap nodes in scalar field."""
        center = RaCoordinate(0, 0, 0)
        field = ScalarField(
            points=[
                ScalarFieldPoint(RaCoordinate(1, 0, 0), 0.92, GradientVector(-0.9, 0, 0)),
                ScalarFieldPoint(RaCoordinate(0, 1, 0), 0.50, GradientVector(0, -0.9, 0)),
                ScalarFieldPoint(RaCoordinate(-1, 0, 0), 0.95, GradientVector(0.9, 0, 0)),
            ],
            center=center
        )
        avatar = Avatar(
            position=RaCoordinate(0.5, 0, 0),
            appendage=AppendageResonance(frequency=15.0, amplitude=0.8, hrv_coherence=0.7)
        )

        nodes = locate_tap_nodes(field, avatar, ChamberForm.Spherical)

        # Should find 2 nodes (first and third, second has low alpha)
        assert len(nodes) == 2

    def test_empty_field_returns_empty(self):
        """Empty field returns no tap nodes."""
        field = ScalarField(points=[], center=RaCoordinate(0, 0, 0))
        avatar = Avatar(
            position=RaCoordinate(0, 0, 0),
            appendage=AppendageResonance(frequency=10.0, amplitude=0.5, hrv_coherence=0.7)
        )

        nodes = locate_tap_nodes(field, avatar, ChamberForm.Toroidal)
        assert len(nodes) == 0


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestTapNodeIntegration:
    """Integration tests for tap node location."""

    def test_full_pipeline(self):
        """Full pipeline from field to tap nodes."""
        center = RaCoordinate(0, 0, 0)

        # Create field with multiple points
        points = []
        for angle in [0, math.pi/2, math.pi, 3*math.pi/2]:
            x = math.cos(angle)
            y = math.sin(angle)
            # Gradient pointing inward
            grad = GradientVector(-x * 0.9, -y * 0.9, 0)
            points.append(ScalarFieldPoint(
                RaCoordinate(x, y, 0),
                alpha=0.90,
                gradient=grad
            ))

        field = ScalarField(points=points, center=center)
        # Tap freq = 7.83 * (1+0.90) * (1+0.9*0.1) ≈ 16.22, use freq within ±5%
        avatar = Avatar(
            position=RaCoordinate(0, 0, 0),
            appendage=AppendageResonance(frequency=16.2, amplitude=0.8, hrv_coherence=0.75)
        )

        nodes = locate_tap_nodes(field, avatar, ChamberForm.Dodecahedral)

        assert len(nodes) == 4
        for node in nodes:
            assert node.is_active is True
            assert node.geometry_link == ChamberForm.Dodecahedral

    def test_toroidal_finds_more_nodes(self):
        """Toroidal chamber finds more nodes due to lower threshold."""
        center = RaCoordinate(0, 0, 0)
        points = [
            ScalarFieldPoint(RaCoordinate(1, 0, 0), 0.85, GradientVector(-0.9, 0, 0)),
            ScalarFieldPoint(RaCoordinate(0, 1, 0), 0.90, GradientVector(0, -0.9, 0)),
        ]
        field = ScalarField(points=points, center=center)
        avatar = Avatar(
            position=RaCoordinate(0, 0, 0),
            appendage=AppendageResonance(frequency=15.0, amplitude=0.8, hrv_coherence=0.7)
        )

        # Spherical finds only 1 (α=0.90)
        spherical_nodes = locate_tap_nodes(field, avatar, ChamberForm.Spherical)
        # Toroidal finds both
        toroidal_nodes = locate_tap_nodes(field, avatar, ChamberForm.Toroidal)

        assert len(spherical_nodes) == 1
        assert len(toroidal_nodes) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
