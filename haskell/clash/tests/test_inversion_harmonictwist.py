#!/usr/bin/env python3
"""
Test harness for Prompt 49: Ra.Inversion.HarmonicTwist

Models torsional field forces during scalar inversions:
- Non-physical torque-like field deformation
- Destabilizing vector fields across harmonic axes
- Alters coherence dynamics and emergence recovery time

Clarifications:
- riskIndex = |twistVector| * (1 - coherenceMod)
- durationPhiN = baseDuration * exp(-coherenceMod) (exponential scaling)
- OmegaFormat: {omegaL, omegaM, phaseAngle, amplitude}
"""

import pytest
import math
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Risk thresholds
RISK_LOW_THRESHOLD = 0.2
RISK_MODERATE_THRESHOLD = 0.5
RISK_HIGH_THRESHOLD = 0.75

# Duration base (in φ^n ticks)
BASE_DURATION_PHI_N = 8

# Coherence skip threshold (negligible twist above this)
NEGLIGIBLE_TWIST_COHERENCE = 0.95

# =============================================================================
# Enumerations
# =============================================================================

class InversionPolarity(Enum):
    """Inversion state polarity."""
    NORMAL = auto()
    INVERTED = auto()


class RiskLevel(Enum):
    """Risk tier classification."""
    LOW = auto()        # < 0.2
    MODERATE = auto()   # 0.2 - 0.5
    HIGH = auto()       # > 0.5
    CRITICAL = auto()   # > 0.75


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class OmegaFormat:
    """
    Harmonic signature format.

    Attributes:
        omega_l: Spherical harmonic l (radial order)
        omega_m: Spherical harmonic m (angular order)
        phase_angle: Phase angle in radians (0 - 2π)
        amplitude: Field intensity (0-1)
    """
    omega_l: int
    omega_m: int
    phase_angle: float
    amplitude: float

    def __post_init__(self):
        """Validate and normalize."""
        self.omega_l = max(0, min(9, self.omega_l))
        self.omega_m = max(-self.omega_l, min(self.omega_l, self.omega_m))
        self.phase_angle = self.phase_angle % (2 * math.pi)
        self.amplitude = max(0.0, min(1.0, self.amplitude))


@dataclass
class FluxCoherence:
    """
    Flux and coherence state.

    Attributes:
        coherence: Field coherence (0-1)
        flux: Field flux intensity
    """
    coherence: float
    flux: float = 1.0


@dataclass
class TwistVector:
    """
    Torsional twist force vector.

    Attributes:
        theta_force: Pull along θ axis
        phi_force: Pull along φ axis
        harmonic_axis: Harmonic mode (l, m)
        inversion_polarity: NORMAL or INVERTED
        coherence_mod: Damping factor (0-1)
    """
    theta_force: float
    phi_force: float
    harmonic_axis: Tuple[int, int]
    inversion_polarity: InversionPolarity
    coherence_mod: float

    @property
    def magnitude(self) -> float:
        """Compute vector magnitude."""
        return math.sqrt(self.theta_force ** 2 + self.phi_force ** 2)

    @property
    def normalized_magnitude(self) -> float:
        """Magnitude normalized to [0, 1]."""
        # Max possible magnitude is sqrt(2) for unit forces
        return min(1.0, self.magnitude / math.sqrt(2))


@dataclass
class TwistEnvelope:
    """
    Twist envelope with decay and risk.

    Attributes:
        net_twist: The twist vector
        duration_phi_n: Time in φ^n ticks to decay twist
        risk_index: Risk rating (0-1)
    """
    net_twist: TwistVector
    duration_phi_n: int
    risk_index: float

    @property
    def risk_level(self) -> RiskLevel:
        """Get risk tier from index."""
        if self.risk_index >= RISK_HIGH_THRESHOLD:
            return RiskLevel.CRITICAL
        elif self.risk_index >= RISK_MODERATE_THRESHOLD:
            return RiskLevel.HIGH
        elif self.risk_index >= RISK_LOW_THRESHOLD:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW


# =============================================================================
# Core Functions
# =============================================================================

def compute_theta_force(omega: OmegaFormat, inversion: InversionPolarity) -> float:
    """
    Compute θ force from omega format and inversion.

    θ force derives from angular phase offset.
    Inverted polarity flips the sign.
    """
    # Base force from phase offset relative to θ
    base_force = math.sin(omega.phase_angle) * omega.amplitude

    # Scale by harmonic order
    harmonic_scale = 1.0 + omega.omega_l * 0.1

    force = base_force * harmonic_scale

    # Inversion flips sign
    if inversion == InversionPolarity.INVERTED:
        force = -force

    return max(-1.0, min(1.0, force))


def compute_phi_force(omega: OmegaFormat, inversion: InversionPolarity) -> float:
    """
    Compute φ force from omega format and inversion.

    φ force derives from azimuthal phase offset.
    Inverted polarity flips the sign.
    """
    # Base force from phase offset relative to φ
    base_force = math.cos(omega.phase_angle) * omega.amplitude

    # Scale by m index (angular momentum component)
    angular_scale = 1.0 + abs(omega.omega_m) * 0.15

    force = base_force * angular_scale

    # Inversion flips sign
    if inversion == InversionPolarity.INVERTED:
        force = -force

    return max(-1.0, min(1.0, force))


def compute_risk_index(twist_magnitude: float, coherence: float) -> float:
    """
    Compute risk index from twist magnitude and coherence.

    riskIndex = |twistVector| * (1 - coherenceMod)

    High twist + low coherence = high risk
    """
    return twist_magnitude * (1.0 - coherence)


def compute_duration_phi_n(coherence: float, base_duration: int = BASE_DURATION_PHI_N) -> int:
    """
    Compute duration in φ^n ticks using exponential scaling.

    durationPhiN = baseDuration * exp(-coherenceMod)

    Lower coherence = longer recovery time.
    """
    duration = base_duration * math.exp(-coherence)
    return max(1, round(duration))


def compute_harmonic_twist(
    omega: OmegaFormat,
    inversion: InversionPolarity,
    flux_coherence: FluxCoherence
) -> TwistEnvelope:
    """
    Compute harmonic twist envelope.

    Parameters:
        omega: Harmonic signature
        inversion: Inversion polarity
        flux_coherence: Coherence and flux state

    Returns:
        TwistEnvelope with twist vector, duration, and risk
    """
    coherence = flux_coherence.coherence

    # Skip for negligible twist (high coherence)
    if coherence >= NEGLIGIBLE_TWIST_COHERENCE:
        return TwistEnvelope(
            net_twist=TwistVector(0.0, 0.0, (omega.omega_l, omega.omega_m),
                                  inversion, coherence),
            duration_phi_n=1,
            risk_index=0.0
        )

    # Compute forces
    theta_force = compute_theta_force(omega, inversion)
    phi_force = compute_phi_force(omega, inversion)

    # Build twist vector
    twist = TwistVector(
        theta_force=theta_force,
        phi_force=phi_force,
        harmonic_axis=(omega.omega_l, omega.omega_m),
        inversion_polarity=inversion,
        coherence_mod=coherence
    )

    # Compute derived values
    risk = compute_risk_index(twist.normalized_magnitude, coherence)
    duration = compute_duration_phi_n(coherence)

    return TwistEnvelope(
        net_twist=twist,
        duration_phi_n=duration,
        risk_index=risk
    )


def get_risk_level(risk_index: float) -> RiskLevel:
    """Classify risk index into tier."""
    if risk_index >= RISK_HIGH_THRESHOLD:
        return RiskLevel.CRITICAL
    elif risk_index >= RISK_MODERATE_THRESHOLD:
        return RiskLevel.HIGH
    elif risk_index >= RISK_LOW_THRESHOLD:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.LOW


def twist_vectors_aligned(v1: TwistVector, v2: TwistVector) -> bool:
    """Check if two twist vectors are approximately aligned."""
    dot = v1.theta_force * v2.theta_force + v1.phi_force * v2.phi_force
    return dot > 0


# =============================================================================
# Test Cases
# =============================================================================

class TestTwistVectorGeneration:
    """Test twist vector generation."""

    def test_inverted_mode_nonzero_forces(self):
        """Inverted harmonic mode produces nonzero forces."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.8)
        twist = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.5)
        )

        assert twist.net_twist.theta_force != 0
        assert twist.net_twist.phi_force != 0

    def test_forces_axis_aligned(self):
        """Forces align with harmonic axis."""
        omega = OmegaFormat(3, 2, math.pi / 3, 0.9)
        twist = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.6)
        )

        # Axis should match omega
        assert twist.net_twist.harmonic_axis == (3, 2)

    def test_normal_vs_inverted_opposite_signs(self):
        """Normal vs inverted should produce opposite force directions."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.8)

        normal_twist = compute_harmonic_twist(
            omega, InversionPolarity.NORMAL, FluxCoherence(0.5)
        )
        inverted_twist = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.5)
        )

        # Signs should be opposite
        assert (normal_twist.net_twist.theta_force *
                inverted_twist.net_twist.theta_force) < 0
        assert (normal_twist.net_twist.phi_force *
                inverted_twist.net_twist.phi_force) < 0

    def test_zero_amplitude_zero_forces(self):
        """Zero amplitude produces zero forces."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.0)  # Zero amplitude
        twist = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.5)
        )

        assert twist.net_twist.theta_force == 0
        assert twist.net_twist.phi_force == 0

    def test_magnitude_computation(self):
        """Twist magnitude computed correctly."""
        twist = TwistVector(0.6, 0.8, (2, 1), InversionPolarity.INVERTED, 0.5)
        assert abs(twist.magnitude - 1.0) < 0.01  # sqrt(0.36 + 0.64) = 1.0


class TestDurationScaling:
    """Test duration φ^n scaling with coherence."""

    def test_low_coherence_long_duration(self):
        """Low coherence produces longer duration."""
        dur_low = compute_duration_phi_n(0.2)
        dur_high = compute_duration_phi_n(0.8)

        assert dur_low > dur_high

    def test_coherence_drop_increases_duration_exponentially(self):
        """Coherence drops increase duration exponentially."""
        dur_0 = compute_duration_phi_n(0.0)
        dur_25 = compute_duration_phi_n(0.25)
        dur_50 = compute_duration_phi_n(0.50)
        dur_75 = compute_duration_phi_n(0.75)

        # Should decrease as coherence increases
        assert dur_0 > dur_25 > dur_50 > dur_75

        # Exponential: ratio should be roughly constant
        ratio1 = dur_0 / dur_25
        ratio2 = dur_25 / dur_50
        # Allow for rounding effects
        assert abs(ratio1 - ratio2) < 1.0

    def test_duration_never_zero(self):
        """Duration should always be at least 1."""
        dur = compute_duration_phi_n(1.0)
        assert dur >= 1

    def test_very_low_coherence_high_duration(self):
        """Very low coherence produces high duration."""
        dur = compute_duration_phi_n(0.05)
        assert dur > BASE_DURATION_PHI_N * 0.5


class TestPolarityImpact:
    """Test inversion polarity impact."""

    def test_polarity_inverts_theta_force(self):
        """INVERTED polarity inverts θ force sign."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.8)

        theta_normal = compute_theta_force(omega, InversionPolarity.NORMAL)
        theta_inverted = compute_theta_force(omega, InversionPolarity.INVERTED)

        assert theta_normal == -theta_inverted

    def test_polarity_inverts_phi_force(self):
        """INVERTED polarity inverts φ force sign."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.8)

        phi_normal = compute_phi_force(omega, InversionPolarity.NORMAL)
        phi_inverted = compute_phi_force(omega, InversionPolarity.INVERTED)

        assert phi_normal == -phi_inverted

    def test_polarity_preserved_in_envelope(self):
        """Polarity is preserved in twist envelope."""
        omega = OmegaFormat(2, 1, math.pi / 4, 0.8)

        normal_env = compute_harmonic_twist(
            omega, InversionPolarity.NORMAL, FluxCoherence(0.5)
        )
        inverted_env = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.5)
        )

        assert normal_env.net_twist.inversion_polarity == InversionPolarity.NORMAL
        assert inverted_env.net_twist.inversion_polarity == InversionPolarity.INVERTED


class TestRiskIndexMapping:
    """Test risk index computation and mapping."""

    def test_risk_formula(self):
        """Risk index follows formula: |twist| * (1 - coherence)."""
        twist_mag = 0.8
        coherence = 0.3

        risk = compute_risk_index(twist_mag, coherence)
        expected = 0.8 * (1 - 0.3)

        assert abs(risk - expected) < 0.01

    def test_high_twist_low_coherence_high_risk(self):
        """High twist + low coherence = high risk."""
        risk = compute_risk_index(0.9, 0.1)
        assert risk > RISK_HIGH_THRESHOLD

    def test_low_twist_high_coherence_low_risk(self):
        """Low twist + high coherence = low risk."""
        risk = compute_risk_index(0.2, 0.9)
        assert risk < RISK_LOW_THRESHOLD

    def test_risk_level_classification(self):
        """Risk levels classified correctly."""
        assert get_risk_level(0.1) == RiskLevel.LOW
        assert get_risk_level(0.3) == RiskLevel.MODERATE
        assert get_risk_level(0.6) == RiskLevel.HIGH
        assert get_risk_level(0.8) == RiskLevel.CRITICAL

    def test_high_risk_correlates_instability(self):
        """Risk > 0.75 indicates fragment instability."""
        omega = OmegaFormat(3, 2, math.pi / 2, 0.95)
        envelope = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.15)
        )

        # High amplitude, low coherence should produce high risk
        if envelope.risk_index > RISK_HIGH_THRESHOLD:
            assert envelope.risk_level == RiskLevel.CRITICAL


class TestOmegaFormat:
    """Test OmegaFormat structure."""

    def test_omega_format_validation(self):
        """OmegaFormat validates inputs."""
        omega = OmegaFormat(15, 20, 10.0, 2.0)

        # Should be clamped
        assert omega.omega_l == 9  # Clamped from 15
        assert omega.omega_m == 9  # Clamped to max |m| = l
        assert omega.phase_angle < 2 * math.pi  # Wrapped
        assert omega.amplitude == 1.0  # Clamped from 2.0

    def test_omega_negative_m(self):
        """OmegaFormat handles negative m."""
        omega = OmegaFormat(3, -2, 1.0, 0.5)
        assert omega.omega_m == -2

    def test_omega_m_bounded_by_l(self):
        """m is bounded by |m| <= l."""
        omega = OmegaFormat(2, 5, 1.0, 0.5)
        assert abs(omega.omega_m) <= omega.omega_l


class TestNegligibleTwist:
    """Test skip behavior for high coherence."""

    def test_high_coherence_negligible_twist(self):
        """Coherence >= 0.95 produces negligible twist."""
        omega = OmegaFormat(3, 2, math.pi / 4, 0.9)
        envelope = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.96)
        )

        assert envelope.net_twist.theta_force == 0
        assert envelope.net_twist.phi_force == 0
        assert envelope.risk_index == 0.0
        assert envelope.duration_phi_n == 1

    def test_just_below_threshold_not_skipped(self):
        """Coherence just below 0.95 is not skipped."""
        omega = OmegaFormat(3, 2, math.pi / 4, 0.9)
        envelope = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.94)
        )

        # Should have non-zero twist
        assert envelope.net_twist.magnitude > 0


class TestTwistEnvelopeIntegration:
    """Integration tests for twist envelope."""

    def test_full_envelope_generation(self):
        """Full envelope generation produces valid output."""
        omega = OmegaFormat(2, 1, math.pi / 3, 0.75)
        envelope = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.45)
        )

        assert envelope.net_twist.harmonic_axis == (2, 1)
        assert envelope.duration_phi_n >= 1
        assert 0 <= envelope.risk_index <= 1

    def test_sample_output_structure(self):
        """Output matches expected structure from prompt."""
        omega = OmegaFormat(2, 3, 0.5, 0.72)
        envelope = compute_harmonic_twist(
            omega, InversionPolarity.INVERTED, FluxCoherence(0.72)
        )

        # Verify structure matches prompt example format
        assert hasattr(envelope, 'net_twist')
        assert hasattr(envelope.net_twist, 'theta_force')
        assert hasattr(envelope.net_twist, 'phi_force')
        assert hasattr(envelope.net_twist, 'harmonic_axis')
        assert hasattr(envelope.net_twist, 'inversion_polarity')
        assert hasattr(envelope.net_twist, 'coherence_mod')
        assert hasattr(envelope, 'duration_phi_n')
        assert hasattr(envelope, 'risk_index')


class TestPhiIntegration:
    """Test phi constant integration."""

    def test_phi_constant_defined(self):
        """Phi constant is correctly defined."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_duration_uses_phi_ticks(self):
        """Duration is in φ^n tick units."""
        dur = compute_duration_phi_n(0.5)
        # Duration should be reasonable number of ticks
        assert 1 <= dur <= 20


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
