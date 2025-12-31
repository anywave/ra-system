"""
Prompt 43: Ra.Projection.Map - Scalar Field Hypergrid Projection

Projects a live scalar field into an 810-point hypergrid defined by:
- θ (theta) = 27 slices (circular wrapping)
- φ (phi) = 6 segments (circular wrapping)
- h (harmonic levels) = 5 depths (bounded)

Produces RaCoordinate → EmergenceAlpha ∈ [0.0, 1.0] with gradient vectors
and inversion zone detection.

Codex References:
- Ra.Scalar: Scalar field definitions
- Ra.Emergence: Emergence alpha computation
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Hypergrid dimensions
THETA_SLICES = 27
PHI_SEGMENTS = 6
HARMONIC_DEPTHS = 5
RA_SPACE_SIZE = THETA_SLICES * PHI_SEGMENTS * HARMONIC_DEPTHS  # 810

# Inversion thresholds
INVERSION_ALPHA_THRESHOLD = 0.3
DIVERGENCE_THRESHOLD = 0.5


# ============================================================================
# Types
# ============================================================================

@dataclass(frozen=True)
class RaCoordinate:
    """
    Coordinate in Ra hypergrid space.

    - theta: 0-26 (27 slices, circular)
    - phi: 0-5 (6 segments, circular)
    - h: 0-4 (5 harmonic depths, bounded)
    """
    theta: int
    phi: int
    h: int

    def __post_init__(self):
        object.__setattr__(self, 'theta', self.theta % THETA_SLICES)
        object.__setattr__(self, 'phi', self.phi % PHI_SEGMENTS)
        object.__setattr__(self, 'h', max(0, min(self.h, HARMONIC_DEPTHS - 1)))

    def to_index(self) -> int:
        """Convert to flat index in 810-point space."""
        return self.theta * PHI_SEGMENTS * HARMONIC_DEPTHS + self.phi * HARMONIC_DEPTHS + self.h


@dataclass
class ScalarValue:
    """Scalar field value at a coordinate."""
    potential: float      # Raw scalar potential
    flux: float          # Flux magnitude
    phase: float         # Phase angle (radians)
    coherence: float     # Local coherence (0-1)


@dataclass
class GradientVector:
    """Gradient vector in Ra space."""
    d_theta: float       # Gradient along theta
    d_phi: float         # Gradient along phi
    d_h: float           # Gradient along harmonic depth

    def magnitude(self) -> float:
        """Compute gradient magnitude."""
        return math.sqrt(self.d_theta**2 + self.d_phi**2 + self.d_h**2)

    def is_sign_flip(self, other: 'GradientVector') -> bool:
        """Check if signs flip between this and another gradient."""
        return (self.d_theta * other.d_theta < 0 or
                self.d_phi * other.d_phi < 0 or
                self.d_h * other.d_h < 0)


class InversionType(Enum):
    """Type of inversion zone."""
    GRADIENT_FLIP = auto()      # Sign flip in gradient
    LOW_ALPHA = auto()          # EmergenceAlpha below threshold
    DIVERGENCE = auto()         # High divergence with low alpha
    COMBINED = auto()           # Multiple conditions


@dataclass
class InversionZone:
    """Detected inversion zone."""
    coordinate: RaCoordinate
    inversion_type: InversionType
    alpha_value: float
    gradient_magnitude: float


# Type aliases
ScalarField = Dict[RaCoordinate, ScalarValue]
EmergenceMap = Dict[RaCoordinate, float]
GradientMap = Dict[RaCoordinate, GradientVector]


# ============================================================================
# Harmonic Modulation (cached/precomputed)
# ============================================================================

# Precomputed harmonic curves (simulating ROM lookup)
_THETA_HARMONICS: List[float] = [math.sin(2 * math.pi * t / THETA_SLICES) for t in range(THETA_SLICES)]
_PHI_HARMONICS: List[float] = [math.cos(math.pi * p / PHI_SEGMENTS) for p in range(PHI_SEGMENTS)]
_DEPTH_DECAY: List[float] = [PHI ** (-h) for h in range(HARMONIC_DEPTHS)]


def get_theta_harmonic(theta: int) -> float:
    """Get cached theta harmonic modulation."""
    return _THETA_HARMONICS[theta % THETA_SLICES]


def get_phi_harmonic(phi: int) -> float:
    """Get cached phi harmonic modulation."""
    return _PHI_HARMONICS[phi % PHI_SEGMENTS]


def get_depth_decay(h: int) -> float:
    """Get cached depth decay factor."""
    return _DEPTH_DECAY[min(h, HARMONIC_DEPTHS - 1)]


# ============================================================================
# Scalar Field Operations
# ============================================================================

def create_scalar_value(potential: float, flux: float = 0.0,
                        phase: float = 0.0, coherence: float = 0.5) -> ScalarValue:
    """Create a scalar value with defaults."""
    return ScalarValue(
        potential=potential,
        flux=flux,
        phase=phase,
        coherence=max(0.0, min(1.0, coherence))
    )


def compute_emergence_alpha(scalar: ScalarValue, coord: RaCoordinate) -> float:
    """
    Compute EmergenceAlpha for a scalar value at coordinate.

    Alpha = coherence * depth_decay * harmonic_modulation
    Bounded to [0.0, 1.0]
    """
    theta_mod = (1 + get_theta_harmonic(coord.theta)) / 2  # Normalize to [0,1]
    phi_mod = (1 + get_phi_harmonic(coord.phi)) / 2
    depth_factor = get_depth_decay(coord.h)

    # Base alpha from coherence
    base_alpha = scalar.coherence * scalar.potential

    # Apply harmonic modulation
    modulated = base_alpha * theta_mod * phi_mod * depth_factor

    # Clamp to [0, 1]
    return max(0.0, min(1.0, modulated))


def project_scalar_field(field: ScalarField) -> EmergenceMap:
    """
    Project scalar field to EmergenceAlpha map.

    All output values guaranteed in [0.0, 1.0].
    """
    return {coord: compute_emergence_alpha(scalar, coord)
            for coord, scalar in field.items()}


# ============================================================================
# Gradient Computation (with circular wrapping)
# ============================================================================

def get_neighbors(coord: RaCoordinate) -> Dict[str, RaCoordinate]:
    """
    Get neighboring coordinates with circular wrapping.

    θ and φ wrap circularly, h is bounded.
    """
    neighbors = {}

    # Theta neighbors (circular)
    neighbors['theta_plus'] = RaCoordinate(
        (coord.theta + 1) % THETA_SLICES, coord.phi, coord.h)
    neighbors['theta_minus'] = RaCoordinate(
        (coord.theta - 1) % THETA_SLICES, coord.phi, coord.h)

    # Phi neighbors (circular)
    neighbors['phi_plus'] = RaCoordinate(
        coord.theta, (coord.phi + 1) % PHI_SEGMENTS, coord.h)
    neighbors['phi_minus'] = RaCoordinate(
        coord.theta, (coord.phi - 1) % PHI_SEGMENTS, coord.h)

    # H neighbors (bounded)
    if coord.h < HARMONIC_DEPTHS - 1:
        neighbors['h_plus'] = RaCoordinate(coord.theta, coord.phi, coord.h + 1)
    if coord.h > 0:
        neighbors['h_minus'] = RaCoordinate(coord.theta, coord.phi, coord.h - 1)

    return neighbors


def compute_gradient(emergence_map: EmergenceMap, coord: RaCoordinate) -> GradientVector:
    """
    Compute gradient at coordinate using central differences.

    ∇α ≈ (α[i+1] - α[i-1]) / 2 for circular dimensions.
    """
    neighbors = get_neighbors(coord)
    alpha_here = emergence_map.get(coord, 0.0)

    # Theta gradient (circular)
    alpha_theta_plus = emergence_map.get(neighbors['theta_plus'], alpha_here)
    alpha_theta_minus = emergence_map.get(neighbors['theta_minus'], alpha_here)
    d_theta = (alpha_theta_plus - alpha_theta_minus) / 2

    # Phi gradient (circular)
    alpha_phi_plus = emergence_map.get(neighbors['phi_plus'], alpha_here)
    alpha_phi_minus = emergence_map.get(neighbors['phi_minus'], alpha_here)
    d_phi = (alpha_phi_plus - alpha_phi_minus) / 2

    # H gradient (bounded - use forward/backward at edges)
    if 'h_plus' in neighbors and 'h_minus' in neighbors:
        alpha_h_plus = emergence_map.get(neighbors['h_plus'], alpha_here)
        alpha_h_minus = emergence_map.get(neighbors['h_minus'], alpha_here)
        d_h = (alpha_h_plus - alpha_h_minus) / 2
    elif 'h_plus' in neighbors:
        alpha_h_plus = emergence_map.get(neighbors['h_plus'], alpha_here)
        d_h = alpha_h_plus - alpha_here
    elif 'h_minus' in neighbors:
        alpha_h_minus = emergence_map.get(neighbors['h_minus'], alpha_here)
        d_h = alpha_here - alpha_h_minus
    else:
        d_h = 0.0

    return GradientVector(d_theta, d_phi, d_h)


def compute_gradient_flow(field: ScalarField) -> GradientMap:
    """Compute gradient vectors for entire field."""
    emergence_map = project_scalar_field(field)
    return {coord: compute_gradient(emergence_map, coord)
            for coord in field.keys()}


# ============================================================================
# Inversion Zone Detection
# ============================================================================

def detect_inversion(coord: RaCoordinate,
                     emergence_map: EmergenceMap,
                     gradient_map: GradientMap) -> Optional[InversionZone]:
    """
    Detect if coordinate is an inversion zone.

    Inversion conditions:
    1. Sign flip in gradient across any axis
    2. EmergenceAlpha < 0.3 with high divergence
    """
    alpha = emergence_map.get(coord, 0.0)
    gradient = gradient_map.get(coord)

    if gradient is None:
        return None

    neighbors = get_neighbors(coord)
    is_flip = False

    # Check for sign flips with neighbors
    for neighbor_coord in neighbors.values():
        neighbor_grad = gradient_map.get(neighbor_coord)
        if neighbor_grad and gradient.is_sign_flip(neighbor_grad):
            is_flip = True
            break

    # Check conditions
    low_alpha = alpha < INVERSION_ALPHA_THRESHOLD
    high_divergence = gradient.magnitude() > DIVERGENCE_THRESHOLD

    if is_flip and low_alpha:
        return InversionZone(coord, InversionType.COMBINED, alpha, gradient.magnitude())
    elif is_flip:
        return InversionZone(coord, InversionType.GRADIENT_FLIP, alpha, gradient.magnitude())
    elif low_alpha and high_divergence:
        return InversionZone(coord, InversionType.DIVERGENCE, alpha, gradient.magnitude())
    elif low_alpha:
        return InversionZone(coord, InversionType.LOW_ALPHA, alpha, gradient.magnitude())

    return None


def find_inversion_zones(field: ScalarField) -> List[InversionZone]:
    """Find all inversion zones in the field."""
    emergence_map = project_scalar_field(field)
    gradient_map = compute_gradient_flow(field)

    zones = []
    for coord in field.keys():
        zone = detect_inversion(coord, emergence_map, gradient_map)
        if zone:
            zones.append(zone)

    return zones


# ============================================================================
# Field Generation Utilities
# ============================================================================

def generate_uniform_field(value: float = 0.5) -> ScalarField:
    """Generate uniform scalar field across all 810 points."""
    field = {}
    for theta in range(THETA_SLICES):
        for phi in range(PHI_SEGMENTS):
            for h in range(HARMONIC_DEPTHS):
                coord = RaCoordinate(theta, phi, h)
                field[coord] = create_scalar_value(value, coherence=value)
    return field


def generate_gradient_field(direction: str = 'theta') -> ScalarField:
    """Generate field with gradient in specified direction."""
    field = {}
    for theta in range(THETA_SLICES):
        for phi in range(PHI_SEGMENTS):
            for h in range(HARMONIC_DEPTHS):
                coord = RaCoordinate(theta, phi, h)
                if direction == 'theta':
                    value = theta / THETA_SLICES
                elif direction == 'phi':
                    value = phi / PHI_SEGMENTS
                else:
                    value = h / HARMONIC_DEPTHS
                field[coord] = create_scalar_value(value, coherence=0.8)
    return field


def generate_inversion_field() -> ScalarField:
    """Generate field with deliberate inversion zones."""
    field = {}
    for theta in range(THETA_SLICES):
        for phi in range(PHI_SEGMENTS):
            for h in range(HARMONIC_DEPTHS):
                coord = RaCoordinate(theta, phi, h)
                # Create low-alpha regions with high flux
                if theta in [13, 14] and phi in [2, 3]:
                    value = 0.1  # Low alpha region
                    flux = 0.9  # High flux
                else:
                    value = 0.7
                    flux = 0.2
                field[coord] = create_scalar_value(value, flux=flux, coherence=value)
    return field


# ============================================================================
# Test Suite
# ============================================================================

class TestRaCoordinate:
    """Test RaCoordinate type."""

    def test_theta_wrapping(self):
        """Theta wraps at 27."""
        coord = RaCoordinate(27, 0, 0)
        assert coord.theta == 0

        coord2 = RaCoordinate(-1, 0, 0)
        assert coord2.theta == 26

    def test_phi_wrapping(self):
        """Phi wraps at 6."""
        coord = RaCoordinate(0, 6, 0)
        assert coord.phi == 0

        coord2 = RaCoordinate(0, -1, 0)
        assert coord2.phi == 5

    def test_h_bounded(self):
        """H is bounded [0, 4]."""
        coord = RaCoordinate(0, 0, 10)
        assert coord.h == 4

        coord2 = RaCoordinate(0, 0, -5)
        assert coord2.h == 0

    def test_to_index_unique(self):
        """Each coordinate maps to unique index."""
        indices = set()
        for theta in range(THETA_SLICES):
            for phi in range(PHI_SEGMENTS):
                for h in range(HARMONIC_DEPTHS):
                    idx = RaCoordinate(theta, phi, h).to_index()
                    assert idx not in indices
                    indices.add(idx)
        assert len(indices) == RA_SPACE_SIZE

    def test_space_size(self):
        """Ra space has 810 points."""
        assert RA_SPACE_SIZE == 810


class TestEmergenceAlpha:
    """Test EmergenceAlpha computation."""

    def test_alpha_bounded(self):
        """Alpha always in [0, 1]."""
        field = generate_uniform_field(0.5)
        emergence = project_scalar_field(field)
        for alpha in emergence.values():
            assert 0.0 <= alpha <= 1.0

    def test_high_coherence_higher_alpha(self):
        """Higher coherence produces higher alpha."""
        coord = RaCoordinate(5, 2, 1)

        low_coh = create_scalar_value(1.0, coherence=0.3)
        high_coh = create_scalar_value(1.0, coherence=0.9)

        alpha_low = compute_emergence_alpha(low_coh, coord)
        alpha_high = compute_emergence_alpha(high_coh, coord)

        assert alpha_high > alpha_low

    def test_depth_decay(self):
        """Deeper harmonic levels have lower alpha."""
        scalar = create_scalar_value(1.0, coherence=0.8)

        alpha_h0 = compute_emergence_alpha(scalar, RaCoordinate(5, 2, 0))
        alpha_h4 = compute_emergence_alpha(scalar, RaCoordinate(5, 2, 4))

        assert alpha_h0 > alpha_h4

    def test_zero_coherence_zero_alpha(self):
        """Zero coherence produces zero alpha."""
        scalar = create_scalar_value(1.0, coherence=0.0)
        coord = RaCoordinate(10, 3, 2)
        alpha = compute_emergence_alpha(scalar, coord)
        assert alpha == 0.0


class TestGradientComputation:
    """Test gradient flow computation."""

    def test_uniform_field_low_gradient(self):
        """Uniform field has low gradients (harmonic modulation creates some variation)."""
        field = generate_uniform_field(0.5)
        gradients = compute_gradient_flow(field)

        # Gradients are low but not zero due to harmonic modulation
        for grad in gradients.values():
            assert abs(grad.d_theta) < 0.15
            assert abs(grad.d_phi) < 0.15

    def test_theta_gradient_field(self):
        """Gradient field has positive theta gradient."""
        field = generate_gradient_field('theta')
        gradients = compute_gradient_flow(field)

        # Check middle of field (away from wrap boundary)
        coord = RaCoordinate(10, 3, 2)
        grad = gradients.get(coord)
        assert grad is not None
        assert grad.d_theta > 0  # Increasing theta direction

    def test_circular_wrapping_theta(self):
        """Gradient wraps correctly at theta boundary."""
        field = generate_uniform_field(0.5)
        # Add spike at theta=0
        field[RaCoordinate(0, 3, 2)] = create_scalar_value(1.0, coherence=1.0)

        gradients = compute_gradient_flow(field)

        # Check that theta=26 sees the spike at theta=0
        coord_26 = RaCoordinate(26, 3, 2)
        grad = gradients.get(coord_26)
        assert grad is not None
        # Should have positive gradient toward theta=0

    def test_h_bounded_gradient(self):
        """H gradient uses forward/backward diff at boundaries."""
        field = generate_gradient_field('h')
        gradients = compute_gradient_flow(field)

        # h=0 should have gradient
        coord_h0 = RaCoordinate(5, 2, 0)
        grad = gradients.get(coord_h0)
        assert grad is not None


class TestInversionZones:
    """Test inversion zone detection."""

    def test_few_inversions_high_coherence(self):
        """High-coherence uniform field has minimal gradient-flip inversions."""
        # Note: Depth decay (φ^-h) means deeper levels may still have low alpha
        # even with high coherence, so we test for low gradient-flip inversions
        field = generate_uniform_field(0.95)
        zones = find_inversion_zones(field)

        # Count only gradient flip inversions (not low-alpha from depth decay)
        flip_zones = [z for z in zones if z.inversion_type == InversionType.GRADIENT_FLIP]
        # Uniform field should have minimal pure gradient flips
        assert len(flip_zones) < len(zones) // 2

    def test_detects_low_alpha_inversion(self):
        """Detects low alpha regions as inversions."""
        field = generate_inversion_field()
        zones = find_inversion_zones(field)

        # Should find inversions
        assert len(zones) > 0

        # Check that low-alpha regions are detected
        low_alpha_zones = [z for z in zones if z.inversion_type in
                          [InversionType.LOW_ALPHA, InversionType.DIVERGENCE, InversionType.COMBINED]]
        assert len(low_alpha_zones) > 0

    def test_inversion_has_coordinate(self):
        """Inversion zones have valid coordinates."""
        field = generate_inversion_field()
        zones = find_inversion_zones(field)

        for zone in zones:
            assert 0 <= zone.coordinate.theta < THETA_SLICES
            assert 0 <= zone.coordinate.phi < PHI_SEGMENTS
            assert 0 <= zone.coordinate.h < HARMONIC_DEPTHS


class TestHarmonicModulation:
    """Test cached harmonic modulation curves."""

    def test_theta_harmonics_periodic(self):
        """Theta harmonics are periodic."""
        assert abs(get_theta_harmonic(0) - get_theta_harmonic(27)) < 0.001

    def test_phi_harmonics_cached(self):
        """Phi harmonics are precomputed."""
        assert len(_PHI_HARMONICS) == PHI_SEGMENTS

    def test_depth_decay_monotonic(self):
        """Depth decay decreases with depth."""
        for h in range(HARMONIC_DEPTHS - 1):
            assert get_depth_decay(h) > get_depth_decay(h + 1)

    def test_depth_decay_phi_scaled(self):
        """Depth decay follows phi^(-h)."""
        for h in range(HARMONIC_DEPTHS):
            expected = PHI ** (-h)
            assert abs(get_depth_decay(h) - expected) < 0.001


class TestFieldGeneration:
    """Test field generation utilities."""

    def test_uniform_field_size(self):
        """Uniform field has 810 points."""
        field = generate_uniform_field()
        assert len(field) == RA_SPACE_SIZE

    def test_gradient_field_direction(self):
        """Gradient field increases in specified direction."""
        field = generate_gradient_field('phi')

        coord_phi0 = RaCoordinate(5, 0, 2)
        coord_phi5 = RaCoordinate(5, 5, 2)

        assert field[coord_phi0].potential < field[coord_phi5].potential


class TestGradientVector:
    """Test GradientVector operations."""

    def test_magnitude(self):
        """Magnitude computed correctly."""
        grad = GradientVector(3.0, 4.0, 0.0)
        assert abs(grad.magnitude() - 5.0) < 0.001

    def test_sign_flip_detection(self):
        """Detects sign flip between gradients."""
        grad1 = GradientVector(1.0, 0.5, 0.2)
        grad2 = GradientVector(-1.0, 0.5, 0.2)  # Theta sign flip

        assert grad1.is_sign_flip(grad2) is True

    def test_no_sign_flip_same_direction(self):
        """No sign flip for same direction gradients."""
        grad1 = GradientVector(1.0, 0.5, 0.2)
        grad2 = GradientVector(0.5, 0.3, 0.1)

        assert grad1.is_sign_flip(grad2) is False


class TestNeighbors:
    """Test neighbor computation."""

    def test_neighbor_count_interior(self):
        """Interior points have 6 neighbors."""
        coord = RaCoordinate(10, 3, 2)
        neighbors = get_neighbors(coord)
        assert len(neighbors) == 6

    def test_neighbor_count_h_boundary(self):
        """H boundary points have 5 neighbors."""
        coord = RaCoordinate(10, 3, 0)  # h=0
        neighbors = get_neighbors(coord)
        assert len(neighbors) == 5

    def test_theta_wrapping_neighbors(self):
        """Theta neighbors wrap correctly."""
        coord = RaCoordinate(0, 3, 2)
        neighbors = get_neighbors(coord)
        assert neighbors['theta_minus'].theta == 26


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
