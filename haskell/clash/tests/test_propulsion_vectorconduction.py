"""
Test Harness for Prompt 51: Ra.Propulsion.VectorConduction

Tests vector impulse conduction from scalar field coherence for:
- Avatar locomotion
- Interface navigation
- Assistive overlays

Based on Hubbard coil-induced phase vectoring, Tesla directed pulses,
and Reich orgone charge movement.
"""

import pytest
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895
STABILITY_THRESHOLD = 0.05  # Default stabilization threshold
COHERENCE_MIN = 0.1  # Minimum coherence for impulse generation
DIRECTION_EPSILON = 0.01  # Minimum component magnitude


# =============================================================================
# Data Structures
# =============================================================================

@dataclass(frozen=True)
class OmegaFormat:
    """Harmonic signature format."""
    omega_l: int  # Spherical harmonic l (0-9)
    omega_m: int  # Spherical harmonic m (-l to +l)
    phase_angle: float  # Phase angle (0 to 2π)
    amplitude: float  # Field intensity (0-1)


@dataclass(frozen=True)
class RaCoordinate:
    """Position in Ra spherical coordinate space."""
    theta: float  # 0 to π
    phi: float  # 0 to 2π
    h: float  # Height/shell (0-1)


@dataclass
class ScalarField:
    """Scalar resonance field."""
    values: List[List[List[float]]]  # 3D grid of scalar values
    resolution: int  # Grid resolution per axis

    def get_value(self, theta_idx: int, phi_idx: int, h_idx: int) -> float:
        """Get scalar value at grid indices."""
        t = theta_idx % self.resolution
        p = phi_idx % self.resolution
        h = h_idx % self.resolution
        return self.values[t][p][h]


@dataclass(frozen=True)
class VectorImpulse:
    """Directional motion impulse from scalar field."""
    direction: Tuple[float, float, float]  # x, y, z normalized
    magnitude: float  # 0 to 1
    harmonic_anchor: OmegaFormat


@dataclass
class ConductionField:
    """Field configuration for impulse conduction."""
    source_field: ScalarField
    coherence_bias: float  # Intention strength (0-1)
    phase_offset: float  # φ^n time harmonic offset


# =============================================================================
# Gradient Computation
# =============================================================================

def compute_scalar_gradient(
    field: ScalarField,
    coord: RaCoordinate,
    delta: float = 0.1
) -> Tuple[float, float, float]:
    """
    Compute scalar gradient via finite difference.

    Returns gradient in spherical coordinates (dψ/dθ, dψ/dφ, dψ/dh).
    """
    res = field.resolution

    # Convert continuous coords to grid indices
    theta_idx = int((coord.theta / math.pi) * (res - 1))
    phi_idx = int((coord.phi / (2 * math.pi)) * (res - 1))
    h_idx = int(coord.h * (res - 1))

    # Finite difference with wrapping
    d_theta = (field.get_value(theta_idx + 1, phi_idx, h_idx) -
               field.get_value(theta_idx - 1, phi_idx, h_idx)) / (2 * delta)

    d_phi = (field.get_value(theta_idx, phi_idx + 1, h_idx) -
             field.get_value(theta_idx, phi_idx - 1, h_idx)) / (2 * delta)

    d_h = (field.get_value(theta_idx, phi_idx, h_idx + 1) -
           field.get_value(theta_idx, phi_idx, h_idx - 1)) / (2 * delta)

    return (d_theta, d_phi, d_h)


def spherical_to_cartesian(
    theta: float,
    phi: float,
    gradient: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Convert spherical gradient to Cartesian direction vector.

    Uses spherical coordinate unit vectors weighted by gradient.
    """
    d_theta, d_phi, d_h = gradient

    # Unit vector components in Cartesian
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    # Weighted direction from gradient
    x = d_theta * sin_theta * cos_phi + d_phi * (-sin_phi) + d_h * cos_theta * cos_phi
    y = d_theta * sin_theta * sin_phi + d_phi * cos_phi + d_h * cos_theta * sin_phi
    z = d_theta * cos_theta + d_h * (-sin_theta)

    return (x, y, z)


def normalize_vector(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector."""
    mag = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if mag < 1e-10:
        return (0.0, 0.0, 0.0)
    return (v[0]/mag, v[1]/mag, v[2]/mag)


def vector_magnitude(v: Tuple[float, float, float]) -> float:
    """Compute vector magnitude."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)


# =============================================================================
# Harmonic Alignment
# =============================================================================

def compute_harmonic_match(
    anchor: OmegaFormat,
    field_l: int,
    field_m: int
) -> float:
    """
    Compute harmonic match score between anchor and field mode.

    Returns 0-1 score where 1 = perfect match.
    """
    # L match (exact = 1, off by 1 = 0.5, etc.)
    l_diff = abs(anchor.omega_l - field_l)
    l_score = max(0, 1 - l_diff * 0.25)

    # M match (exact = 1, off by 1 = 0.7, etc.)
    m_diff = abs(anchor.omega_m - field_m)
    m_score = max(0, 1 - m_diff * 0.15)

    # Combined score weighted by amplitude
    return l_score * m_score * anchor.amplitude


def apply_harmonic_direction_bias(
    direction: Tuple[float, float, float],
    anchor: OmegaFormat
) -> Tuple[float, float, float]:
    """
    Bias direction toward harmonic pattern center based on OmegaFormat.

    Uses phase angle to rotate direction vector.
    """
    phase = anchor.phase_angle

    # Rotate direction by phase angle around z-axis
    x, y, z = direction
    cos_p = math.cos(phase)
    sin_p = math.sin(phase)

    x_rot = x * cos_p - y * sin_p
    y_rot = x * sin_p + y * cos_p

    return (x_rot, y_rot, z)


# =============================================================================
# Core Functions
# =============================================================================

def conduct_impulse(
    conduction_field: ConductionField,
    coord: RaCoordinate,
    field_l: int = 2,
    field_m: int = 0
) -> Optional[VectorImpulse]:
    """
    Calculate direction vector from scalar gradient around RaCoordinate.

    Returns None if coherence ≤ 0 or field is chaotic.
    """
    # Filter invalid: coherence too low
    if conduction_field.coherence_bias < COHERENCE_MIN:
        return None

    # Compute scalar gradient
    gradient = compute_scalar_gradient(
        conduction_field.source_field,
        coord
    )

    # Check for chaotic field (gradient too small)
    grad_mag = vector_magnitude(gradient)
    if grad_mag < 1e-6:
        return None

    # Convert to Cartesian direction
    cart_dir = spherical_to_cartesian(coord.theta, coord.phi, gradient)

    # Create harmonic anchor from field parameters
    anchor = OmegaFormat(
        omega_l=field_l,
        omega_m=field_m,
        phase_angle=conduction_field.phase_offset,
        amplitude=conduction_field.coherence_bias
    )

    # Apply harmonic direction bias
    biased_dir = apply_harmonic_direction_bias(cart_dir, anchor)

    # Normalize direction
    direction = normalize_vector(biased_dir)

    # Compute magnitude from coherence * harmonic match * gradient strength
    harmonic_match = compute_harmonic_match(anchor, field_l, field_m)
    magnitude = min(1.0, conduction_field.coherence_bias * harmonic_match * min(1.0, grad_mag))

    # Filter if magnitude too low
    if magnitude < DIRECTION_EPSILON:
        return None

    return VectorImpulse(
        direction=direction,
        magnitude=magnitude,
        harmonic_anchor=anchor
    )


def stabilize_impulse(
    impulse: VectorImpulse,
    threshold: float = STABILITY_THRESHOLD
) -> Optional[VectorImpulse]:
    """
    Suppress jitter or unstable impulses.

    Returns None if magnitude < threshold.
    """
    if impulse.magnitude < threshold:
        return None

    # Check direction components
    dir_mag = vector_magnitude(impulse.direction)
    if dir_mag < DIRECTION_EPSILON:
        return None

    return impulse


def apply_exponential_smoothing(
    current: VectorImpulse,
    previous: Optional[VectorImpulse],
    alpha: float = 0.3
) -> VectorImpulse:
    """
    Apply exponential smoothing to direction for jitter reduction.

    alpha: smoothing factor (0 = all previous, 1 = all current)
    """
    if previous is None:
        return current

    # Smooth direction components
    smoothed_dir = tuple(
        alpha * c + (1 - alpha) * p
        for c, p in zip(current.direction, previous.direction)
    )

    # Smooth magnitude
    smoothed_mag = alpha * current.magnitude + (1 - alpha) * previous.magnitude

    return VectorImpulse(
        direction=normalize_vector(smoothed_dir),
        magnitude=smoothed_mag,
        harmonic_anchor=current.harmonic_anchor
    )


# =============================================================================
# Test Field Generators
# =============================================================================

def create_uniform_field(resolution: int, value: float) -> ScalarField:
    """Create a uniform scalar field."""
    values = [[[value for _ in range(resolution)]
               for _ in range(resolution)]
              for _ in range(resolution)]
    return ScalarField(values=values, resolution=resolution)


def create_gradient_field(resolution: int, axis: str = 'theta') -> ScalarField:
    """Create a field with gradient along specified axis."""
    values = [[[0.0 for _ in range(resolution)]
               for _ in range(resolution)]
              for _ in range(resolution)]

    for t in range(resolution):
        for p in range(resolution):
            for h in range(resolution):
                if axis == 'theta':
                    values[t][p][h] = t / (resolution - 1)
                elif axis == 'phi':
                    values[t][p][h] = p / (resolution - 1)
                elif axis == 'h':
                    values[t][p][h] = h / (resolution - 1)

    return ScalarField(values=values, resolution=resolution)


def create_coherent_peak_field(
    resolution: int,
    peak_theta: float,
    peak_phi: float,
    peak_h: float
) -> ScalarField:
    """Create a field with coherent peak at specified location."""
    values = [[[0.0 for _ in range(resolution)]
               for _ in range(resolution)]
              for _ in range(resolution)]

    peak_t = int(peak_theta * (resolution - 1) / math.pi)
    peak_p = int(peak_phi * (resolution - 1) / (2 * math.pi))
    peak_hh = int(peak_h * (resolution - 1))

    for t in range(resolution):
        for p in range(resolution):
            for h in range(resolution):
                # Distance from peak
                dt = abs(t - peak_t)
                dp = abs(p - peak_p)
                dh = abs(h - peak_hh)
                dist = math.sqrt(dt**2 + dp**2 + dh**2)

                # Gaussian falloff
                values[t][p][h] = math.exp(-dist**2 / 10.0)

    return ScalarField(values=values, resolution=resolution)


# =============================================================================
# Test Cases
# =============================================================================

class TestVectorImpulse:
    """Tests for VectorImpulse data structure."""

    def test_impulse_creation(self):
        """Test basic impulse creation."""
        anchor = OmegaFormat(omega_l=2, omega_m=1, phase_angle=0.0, amplitude=0.8)
        impulse = VectorImpulse(
            direction=(0.5, 0.5, 0.707),
            magnitude=0.6,
            harmonic_anchor=anchor
        )
        assert impulse.magnitude == 0.6
        assert impulse.harmonic_anchor.omega_l == 2

    def test_impulse_direction_normalized(self):
        """Test that direction can be normalized."""
        anchor = OmegaFormat(omega_l=1, omega_m=0, phase_angle=0.0, amplitude=1.0)
        direction = normalize_vector((1.0, 1.0, 1.0))
        impulse = VectorImpulse(
            direction=direction,
            magnitude=1.0,
            harmonic_anchor=anchor
        )
        mag = vector_magnitude(impulse.direction)
        assert abs(mag - 1.0) < 0.001


class TestGradientComputation:
    """Tests for scalar gradient computation."""

    def test_uniform_field_zero_gradient(self):
        """Uniform field should have zero gradient."""
        field = create_uniform_field(10, 0.5)
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        gradient = compute_scalar_gradient(field, coord)

        assert abs(gradient[0]) < 0.001
        assert abs(gradient[1]) < 0.001
        assert abs(gradient[2]) < 0.001

    def test_theta_gradient_field(self):
        """Field with theta gradient should have positive dθ."""
        field = create_gradient_field(10, 'theta')
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        gradient = compute_scalar_gradient(field, coord)

        assert gradient[0] > 0  # Positive theta gradient

    def test_phi_gradient_field(self):
        """Field with phi gradient should have positive dφ."""
        field = create_gradient_field(10, 'phi')
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        gradient = compute_scalar_gradient(field, coord)

        assert gradient[1] > 0  # Positive phi gradient

    def test_h_gradient_field(self):
        """Field with h gradient should have positive dh."""
        field = create_gradient_field(10, 'h')
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        gradient = compute_scalar_gradient(field, coord)

        assert gradient[2] > 0  # Positive h gradient


class TestSphericalToCartesian:
    """Tests for coordinate conversion."""

    def test_equator_conversion(self):
        """Test conversion at equator (θ=π/2)."""
        gradient = (1.0, 0.0, 0.0)
        result = spherical_to_cartesian(math.pi/2, 0, gradient)
        # At equator with φ=0, gradient in θ gives x component
        assert result[0] != 0 or result[2] != 0

    def test_pole_conversion(self):
        """Test conversion at pole (θ=0)."""
        gradient = (1.0, 0.0, 0.0)  # Theta gradient at pole
        result = spherical_to_cartesian(0.01, 0, gradient)  # Near pole
        # At pole, theta gradient should give non-zero direction
        mag = vector_magnitude(result)
        assert mag > 0.001

    def test_zero_gradient_gives_zero_direction(self):
        """Zero gradient should give zero direction."""
        result = spherical_to_cartesian(math.pi/4, math.pi/4, (0, 0, 0))
        assert result == (0, 0, 0)


class TestHarmonicAlignment:
    """Tests for harmonic match computation."""

    def test_perfect_match(self):
        """Perfect l,m match should give high score."""
        anchor = OmegaFormat(omega_l=2, omega_m=1, phase_angle=0.0, amplitude=1.0)
        score = compute_harmonic_match(anchor, field_l=2, field_m=1)
        assert score == 1.0

    def test_l_mismatch(self):
        """L mismatch should reduce score."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=1.0)
        score = compute_harmonic_match(anchor, field_l=4, field_m=0)
        assert score < 1.0

    def test_m_mismatch(self):
        """M mismatch should reduce score."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=1.0)
        score = compute_harmonic_match(anchor, field_l=2, field_m=3)
        assert score < 1.0

    def test_amplitude_scales_score(self):
        """Lower amplitude should scale down score."""
        anchor_high = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=1.0)
        anchor_low = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.5)

        score_high = compute_harmonic_match(anchor_high, 2, 0)
        score_low = compute_harmonic_match(anchor_low, 2, 0)

        assert score_low < score_high
        assert abs(score_low - score_high * 0.5) < 0.001


class TestConductImpulse:
    """Tests for impulse conduction."""

    def test_coherent_peak_aligns_direction(self):
        """Field with coherent peak should align impulse toward gradient."""
        field = create_coherent_peak_field(10, math.pi/4, math.pi/4, 0.5)
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.8,
            phase_offset=0.0
        )

        # Query from position away from peak
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        impulse = conduct_impulse(conduction, coord)

        assert impulse is not None
        assert impulse.magnitude > 0

    def test_low_coherence_no_impulse(self):
        """Low coherence bias should yield no impulse."""
        field = create_gradient_field(10, 'theta')
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.05,  # Below COHERENCE_MIN
            phase_offset=0.0
        )

        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        impulse = conduct_impulse(conduction, coord)

        assert impulse is None

    def test_high_coherence_high_magnitude(self):
        """High coherence should yield higher magnitude impulse."""
        field = create_gradient_field(10, 'theta')

        conduction_low = ConductionField(
            source_field=field,
            coherence_bias=0.3,
            phase_offset=0.0
        )
        conduction_high = ConductionField(
            source_field=field,
            coherence_bias=0.9,
            phase_offset=0.0
        )

        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        impulse_low = conduct_impulse(conduction_low, coord)
        impulse_high = conduct_impulse(conduction_high, coord)

        assert impulse_low is not None
        assert impulse_high is not None
        assert impulse_high.magnitude > impulse_low.magnitude

    def test_harmonic_anchor_affects_direction(self):
        """Different harmonic anchors should produce different directions."""
        field = create_gradient_field(10, 'theta')

        conduction1 = ConductionField(
            source_field=field,
            coherence_bias=0.8,
            phase_offset=0.0
        )
        conduction2 = ConductionField(
            source_field=field,
            coherence_bias=0.8,
            phase_offset=math.pi  # Different phase
        )

        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        impulse1 = conduct_impulse(conduction1, coord, field_l=2, field_m=0)
        impulse2 = conduct_impulse(conduction2, coord, field_l=2, field_m=0)

        assert impulse1 is not None
        assert impulse2 is not None
        # Directions should differ due to phase rotation
        dir_diff = sum((a - b)**2 for a, b in zip(impulse1.direction, impulse2.direction))
        assert dir_diff > 0.01

    def test_uniform_field_no_impulse(self):
        """Uniform field (no gradient) should produce no impulse."""
        field = create_uniform_field(10, 0.5)
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.8,
            phase_offset=0.0
        )

        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)
        impulse = conduct_impulse(conduction, coord)

        assert impulse is None  # No gradient = no direction


class TestStabilizeImpulse:
    """Tests for impulse stabilization."""

    def test_below_threshold_filtered(self):
        """Impulse below threshold should be filtered."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.5)
        impulse = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.03,  # Below default threshold
            harmonic_anchor=anchor
        )

        result = stabilize_impulse(impulse)
        assert result is None

    def test_above_threshold_passes(self):
        """Impulse above threshold should pass through."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)
        impulse = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.5,
            harmonic_anchor=anchor
        )

        result = stabilize_impulse(impulse)
        assert result is not None
        assert result.magnitude == 0.5

    def test_custom_threshold(self):
        """Custom threshold should be respected."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.5)
        impulse = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.2,
            harmonic_anchor=anchor
        )

        result_low = stabilize_impulse(impulse, threshold=0.1)
        result_high = stabilize_impulse(impulse, threshold=0.3)

        assert result_low is not None
        assert result_high is None

    def test_zero_direction_filtered(self):
        """Impulse with zero direction should be filtered."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.5)
        impulse = VectorImpulse(
            direction=(0.0, 0.0, 0.0),
            magnitude=0.5,
            harmonic_anchor=anchor
        )

        result = stabilize_impulse(impulse)
        assert result is None


class TestExponentialSmoothing:
    """Tests for exponential smoothing."""

    def test_no_previous_returns_current(self):
        """No previous impulse should return current unchanged."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)
        current = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.5,
            harmonic_anchor=anchor
        )

        result = apply_exponential_smoothing(current, None)
        assert result.direction == current.direction
        assert result.magnitude == current.magnitude

    def test_smoothing_blends_directions(self):
        """Smoothing should blend current and previous directions."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)

        previous = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.5,
            harmonic_anchor=anchor
        )
        current = VectorImpulse(
            direction=(0.0, 1.0, 0.0),
            magnitude=0.5,
            harmonic_anchor=anchor
        )

        result = apply_exponential_smoothing(current, previous, alpha=0.5)

        # Should be blend of both directions
        assert abs(result.direction[0]) > 0.1  # Some x from previous
        assert abs(result.direction[1]) > 0.1  # Some y from current

    def test_alpha_1_equals_current(self):
        """Alpha=1 should return current only."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)

        previous = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.3,
            harmonic_anchor=anchor
        )
        current = VectorImpulse(
            direction=(0.0, 1.0, 0.0),
            magnitude=0.7,
            harmonic_anchor=anchor
        )

        result = apply_exponential_smoothing(current, previous, alpha=1.0)

        assert result.magnitude == 0.7

    def test_alpha_0_equals_previous(self):
        """Alpha=0 should return previous only."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)

        previous = VectorImpulse(
            direction=(1.0, 0.0, 0.0),
            magnitude=0.3,
            harmonic_anchor=anchor
        )
        current = VectorImpulse(
            direction=(0.0, 1.0, 0.0),
            magnitude=0.7,
            harmonic_anchor=anchor
        )

        result = apply_exponential_smoothing(current, previous, alpha=0.0)

        assert result.magnitude == 0.3


class TestScreenMapping:
    """Tests for UI/screen coordinate mapping."""

    def test_impulse_to_cursor_delta(self):
        """Test converting impulse to cursor movement delta."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)
        impulse = VectorImpulse(
            direction=(0.5, 0.5, 0.0),  # Diagonal in x-y plane
            magnitude=0.6,
            harmonic_anchor=anchor
        )

        # Map to screen coordinates (ignore z for 2D)
        screen_scale = 100  # pixels per unit
        dx = impulse.direction[0] * impulse.magnitude * screen_scale
        dy = impulse.direction[1] * impulse.magnitude * screen_scale

        assert abs(dx - 30) < 1  # 0.5 * 0.6 * 100 = 30
        assert abs(dy - 30) < 1

    def test_z_component_for_scroll(self):
        """Test z component for scroll/zoom behavior."""
        anchor = OmegaFormat(omega_l=2, omega_m=0, phase_angle=0.0, amplitude=0.8)
        impulse = VectorImpulse(
            direction=(0.0, 0.0, 1.0),  # Purely z direction
            magnitude=0.5,
            harmonic_anchor=anchor
        )

        # z > 0 = zoom in, z < 0 = zoom out
        scroll_delta = impulse.direction[2] * impulse.magnitude * 10  # scroll units
        assert scroll_delta > 0  # Should scroll/zoom in


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_conduction_pipeline(self):
        """Test complete pipeline from field to stabilized impulse."""
        # Create field with peak
        field = create_coherent_peak_field(10, math.pi/3, math.pi/3, 0.6)

        # Set up conduction
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.85,
            phase_offset=0.1
        )

        # Conduct impulse
        coord = RaCoordinate(theta=math.pi/2, phi=math.pi/2, h=0.5)
        impulse = conduct_impulse(conduction, coord, field_l=3, field_m=1)

        # Stabilize
        if impulse:
            stable = stabilize_impulse(impulse)
            if stable:
                # Verify valid output
                assert 0 <= stable.magnitude <= 1
                assert abs(vector_magnitude(stable.direction) - 1.0) < 0.01

    def test_sequence_with_smoothing(self):
        """Test sequence of impulses with smoothing."""
        field = create_gradient_field(10, 'theta')
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.7,
            phase_offset=0.0
        )

        coord = RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5)

        previous = None
        for i in range(5):
            impulse = conduct_impulse(conduction, coord)
            if impulse:
                smoothed = apply_exponential_smoothing(impulse, previous)
                stable = stabilize_impulse(smoothed)
                if stable:
                    previous = stable

        # After sequence, should have accumulated state
        assert previous is not None

    def test_multiple_coordinates(self):
        """Test impulses at multiple coordinates."""
        field = create_coherent_peak_field(10, math.pi/2, math.pi, 0.5)
        conduction = ConductionField(
            source_field=field,
            coherence_bias=0.8,
            phase_offset=0.0
        )

        coords = [
            RaCoordinate(theta=math.pi/4, phi=math.pi/2, h=0.3),
            RaCoordinate(theta=math.pi/2, phi=math.pi, h=0.5),
            RaCoordinate(theta=3*math.pi/4, phi=3*math.pi/2, h=0.7),
        ]

        impulses = [conduct_impulse(conduction, c) for c in coords]
        valid_impulses = [i for i in impulses if i is not None]

        # Should get valid impulses at most locations
        assert len(valid_impulses) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
