#!/usr/bin/env python3
"""
Test harness for Prompt 50: Ra.Fragment.EchoField

Models scalar harmonic reverberation and latent fragment echoes:
- Acoustic-style scalar reverberation
- Memory "ghosts" (deja-vu harmonics)
- Temporary coherence modulation
- Passive gating bias

Clarifications:
- Decay: intensity = baseIntensity * exp(-decayRate * dt)
- Cutoffs: < 0.10 = no influence, < 0.05 = deactivate
- triggerChance = intensity * 0.3, +0.05 harmonic match bonus
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Echo influence thresholds
INFLUENCE_THRESHOLD = 0.10       # Below this, no coherence boost
DEACTIVATION_THRESHOLD = 0.05   # Below this, deactivate echo

# Trigger chance factors
TRIGGER_BASE_FACTOR = 0.3
HARMONIC_MATCH_BONUS = 0.05

# Boost factor scaling
BOOST_FACTOR_SCALE = 0.5

# Echo age limit (skip evaluation if too old)
MAX_ECHO_AGE_PHI_N = 10

# =============================================================================
# Data Types
# =============================================================================

@dataclass
class OmegaFormat:
    """
    Harmonic signature format.

    Attributes:
        omega_l: Spherical harmonic l
        omega_m: Spherical harmonic m
        phase_angle: Phase angle in radians
        amplitude: Field intensity (0-1)
    """
    omega_l: int
    omega_m: int
    phase_angle: float = 0.0
    amplitude: float = 1.0


@dataclass
class EchoField:
    """
    Scalar harmonic echo field.

    Attributes:
        fragment_id: Origin fragment identifier
        decay_rate: Decay rate [0,1], 0 = constant, 1 = instant fade
        base_intensity: Scalar emergence alpha (0-1)
        harmonic_memory: Harmonic imprint (OmegaFormat)
        timestamp_phi_n: φ^n tick at echo creation
    """
    fragment_id: int
    decay_rate: float
    base_intensity: float
    harmonic_memory: OmegaFormat
    timestamp_phi_n: int

    def __post_init__(self):
        """Validate fields."""
        self.decay_rate = max(0.0, min(1.0, self.decay_rate))
        self.base_intensity = max(0.0, min(1.0, self.base_intensity))


@dataclass
class EchoInfluence:
    """
    Echo influence on gating and coherence.

    Attributes:
        boost_factor: Temporary coherence uplift
        harmonic_pull: Bias toward similar fragment (optional)
        trigger_chance: Probability of ghost fragment activation
    """
    boost_factor: float
    harmonic_pull: Optional[OmegaFormat]
    trigger_chance: float


@dataclass
class GateState:
    """
    Gate state for echo influence application.

    Attributes:
        coherence_threshold: Required coherence for passage
        current_coherence: User's current coherence
        next_fragment_bias: Bias toward specific fragment ID
    """
    coherence_threshold: float
    current_coherence: float
    next_fragment_bias: Optional[int] = None


# =============================================================================
# Core Functions
# =============================================================================

def compute_echo_intensity(
    echo: EchoField,
    current_phi_n: int
) -> float:
    """
    Compute current echo intensity using exponential decay.

    intensity(t) = baseIntensity * exp(-decayRate * dt)
    where dt = current_phi_n - timestamp_phi_n
    """
    dt = current_phi_n - echo.timestamp_phi_n

    if dt < 0:
        return echo.base_intensity

    intensity = echo.base_intensity * math.exp(-echo.decay_rate * dt)
    return max(0.0, intensity)


def is_echo_active(intensity: float) -> bool:
    """Check if echo is still active (above deactivation threshold)."""
    return intensity >= DEACTIVATION_THRESHOLD


def has_echo_influence(intensity: float) -> bool:
    """Check if echo has influence (above influence threshold)."""
    return intensity >= INFLUENCE_THRESHOLD


def compute_boost_factor(intensity: float) -> float:
    """
    Compute coherence boost factor from intensity.

    boostFactor = intensity * 0.5
    """
    return intensity * BOOST_FACTOR_SCALE


def compute_trigger_chance(
    intensity: float,
    harmonic_match: bool = False
) -> float:
    """
    Compute ghost fragment trigger chance.

    triggerChance = intensity * 0.3
    With harmonic match: +0.05 bonus
    """
    chance = intensity * TRIGGER_BASE_FACTOR
    if harmonic_match:
        chance += HARMONIC_MATCH_BONUS
    return min(1.0, chance)


def check_harmonic_match(
    echo_harmonic: OmegaFormat,
    field_harmonic: OmegaFormat,
    threshold: float = 0.8
) -> bool:
    """
    Check if echo harmonic matches current field harmonic.

    Match based on l, m similarity and amplitude correlation.
    """
    l_match = echo_harmonic.omega_l == field_harmonic.omega_l
    m_match = abs(echo_harmonic.omega_m - field_harmonic.omega_m) <= 1

    if l_match and m_match:
        # Check amplitude correlation
        amp_diff = abs(echo_harmonic.amplitude - field_harmonic.amplitude)
        return amp_diff < (1 - threshold)

    return False


def generate_echo_field(
    fragment_id: int,
    content_stability: float,
    content_duration: float,
    harmonic: OmegaFormat,
    current_phi_n: int
) -> EchoField:
    """
    Generate an echo field from emergent content.

    Parameters:
        fragment_id: ID of the emerged fragment
        content_stability: Stability of the content (0-1)
        content_duration: How long content was active
        harmonic: Harmonic signature of the fragment
        current_phi_n: Current φ^n tick

    Returns:
        EchoField with computed decay rate and intensity
    """
    # Decay rate inversely related to stability
    # High stability = slow decay (low rate)
    decay_rate = 1.0 - content_stability * 0.7

    # Base intensity from stability and duration
    base_intensity = content_stability * min(1.0, content_duration / 10.0)

    return EchoField(
        fragment_id=fragment_id,
        decay_rate=decay_rate,
        base_intensity=base_intensity,
        harmonic_memory=harmonic,
        timestamp_phi_n=current_phi_n
    )


def evaluate_echo_influence(
    echo: EchoField,
    current_phi_n: int,
    field_harmonic: Optional[OmegaFormat] = None
) -> Optional[EchoInfluence]:
    """
    Evaluate echo influence at current time.

    Returns None if echo has faded below threshold.
    """
    intensity = compute_echo_intensity(echo, current_phi_n)

    # Check if echo is still active
    if not is_echo_active(intensity):
        return None

    # Check if echo has influence
    if not has_echo_influence(intensity):
        return None

    # Compute influence components
    boost = compute_boost_factor(intensity)

    # Check harmonic match for pull and trigger bonus
    harmonic_match = False
    harmonic_pull = None

    if field_harmonic is not None:
        harmonic_match = check_harmonic_match(echo.harmonic_memory, field_harmonic)
        if harmonic_match:
            harmonic_pull = echo.harmonic_memory

    trigger = compute_trigger_chance(intensity, harmonic_match)

    return EchoInfluence(
        boost_factor=boost,
        harmonic_pull=harmonic_pull,
        trigger_chance=trigger
    )


def should_skip_echo_evaluation(
    echo: EchoField,
    current_phi_n: int
) -> bool:
    """
    Check if echo evaluation should be skipped for efficiency.

    Skip if:
    - Echo is too old (> 10 φ^n ticks) AND base_intensity <= 0.95
    """
    age = current_phi_n - echo.timestamp_phi_n
    return age > MAX_ECHO_AGE_PHI_N and echo.base_intensity <= 0.95


def apply_echo_influence(
    influence: EchoInfluence,
    gate_state: GateState
) -> GateState:
    """
    Apply echo influence to gate state.

    - boostFactor adds to user coherence
    - harmonicPull biases next fragment request
    """
    new_coherence = gate_state.current_coherence + influence.boost_factor
    new_coherence = min(1.0, new_coherence)

    new_bias = gate_state.next_fragment_bias
    if influence.harmonic_pull is not None:
        # Bias toward fragment with matching harmonic
        new_bias = hash(
            (influence.harmonic_pull.omega_l, influence.harmonic_pull.omega_m)
        ) % 1000

    return GateState(
        coherence_threshold=gate_state.coherence_threshold,
        current_coherence=new_coherence,
        next_fragment_bias=new_bias
    )


def merge_echo_fields(echoes: List[EchoField]) -> EchoField:
    """
    Merge multiple echo fields into one.

    - Averages decay_rate
    - Max of base_intensity
    - Harmonizes OmegaFormat (average l, m)
    - Uses oldest timestamp_phi_n
    """
    if not echoes:
        return EchoField(0, 0.5, 0.0, OmegaFormat(0, 0), 0)

    if len(echoes) == 1:
        return echoes[0]

    # Average decay rate
    avg_decay = sum(e.decay_rate for e in echoes) / len(echoes)

    # Max base intensity
    max_intensity = max(e.base_intensity for e in echoes)

    # Average harmonics
    avg_l = round(sum(e.harmonic_memory.omega_l for e in echoes) / len(echoes))
    avg_m = round(sum(e.harmonic_memory.omega_m for e in echoes) / len(echoes))
    avg_amp = sum(e.harmonic_memory.amplitude for e in echoes) / len(echoes)

    # Oldest timestamp
    oldest_timestamp = min(e.timestamp_phi_n for e in echoes)

    # Use first fragment ID (or could combine)
    fragment_id = echoes[0].fragment_id

    return EchoField(
        fragment_id=fragment_id,
        decay_rate=avg_decay,
        base_intensity=max_intensity,
        harmonic_memory=OmegaFormat(avg_l, avg_m, amplitude=avg_amp),
        timestamp_phi_n=oldest_timestamp
    )


# =============================================================================
# Test Cases
# =============================================================================

class TestEchoDecayCurve:
    """Test echo intensity decay."""

    def test_decay_over_time(self):
        """Echo decays smoothly over φ^n ticks."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.3,
            base_intensity=0.9,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        intensities = [compute_echo_intensity(echo, t) for t in range(13)]

        # Should decay monotonically
        for i in range(1, len(intensities)):
            assert intensities[i] <= intensities[i-1]

    def test_exponential_decay_formula(self):
        """Decay follows exponential formula."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.5,
            base_intensity=1.0,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # At t=2: intensity = 1.0 * exp(-0.5 * 2) = exp(-1) ≈ 0.368
        intensity = compute_echo_intensity(echo, 2)
        expected = math.exp(-1)
        assert abs(intensity - expected) < 0.01

    def test_zero_decay_rate_constant(self):
        """Zero decay rate maintains constant intensity."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.0,
            base_intensity=0.8,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        for t in range(10):
            intensity = compute_echo_intensity(echo, t)
            assert abs(intensity - 0.8) < 0.01

    def test_high_decay_rate_fast_fade(self):
        """High decay rate causes fast fade."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=1.0,
            base_intensity=0.9,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # Should be very low after a few ticks
        intensity_5 = compute_echo_intensity(echo, 5)
        assert intensity_5 < 0.01


class TestHighIntensityEchoEffect:
    """Test high intensity echo persistence."""

    def test_high_intensity_remains_influential(self):
        """High base intensity remains influential for 6+ ticks."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.15,  # Slow decay
            base_intensity=0.95,
            harmonic_memory=OmegaFormat(3, 2),
            timestamp_phi_n=0
        )

        # Check at tick 6
        intensity_6 = compute_echo_intensity(echo, 6)
        assert has_echo_influence(intensity_6)

    def test_influence_duration_with_high_intensity(self):
        """Track how long high intensity echo has influence."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.2,
            base_intensity=0.92,
            harmonic_memory=OmegaFormat(3, 2),
            timestamp_phi_n=0
        )

        # Find when influence ends
        influence_duration = 0
        for t in range(20):
            intensity = compute_echo_intensity(echo, t)
            if has_echo_influence(intensity):
                influence_duration = t
            else:
                break

        assert influence_duration >= 5


class TestTriggerChanceModel:
    """Test ghost fragment trigger chance."""

    def test_trigger_chance_formula(self):
        """Trigger chance follows formula: intensity * 0.3."""
        intensity = 0.6
        chance = compute_trigger_chance(intensity)
        expected = 0.6 * 0.3
        assert abs(chance - expected) < 0.01

    def test_harmonic_match_bonus(self):
        """Harmonic match adds 0.05 bonus."""
        intensity = 0.5
        chance_no_match = compute_trigger_chance(intensity, harmonic_match=False)
        chance_match = compute_trigger_chance(intensity, harmonic_match=True)

        assert abs(chance_match - chance_no_match - HARMONIC_MATCH_BONUS) < 0.01

    def test_trigger_chance_capped(self):
        """Trigger chance capped at 1.0."""
        chance = compute_trigger_chance(5.0, harmonic_match=True)
        assert chance <= 1.0

    def test_coherence_match_spikes_trigger(self):
        """Trigger chance spikes when chamber coherence matches echo harmonic."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.2,
            base_intensity=0.8,
            harmonic_memory=OmegaFormat(3, 2, amplitude=0.7),
            timestamp_phi_n=0
        )

        # Matching field harmonic
        matching_field = OmegaFormat(3, 2, amplitude=0.75)
        non_matching_field = OmegaFormat(1, 0, amplitude=0.5)

        influence_match = evaluate_echo_influence(echo, 2, matching_field)
        influence_no_match = evaluate_echo_influence(echo, 2, non_matching_field)

        assert influence_match is not None
        assert influence_no_match is not None
        assert influence_match.trigger_chance > influence_no_match.trigger_chance


class TestEchoGateInfluence:
    """Test echo influence on gating."""

    def test_boost_reduces_effective_threshold(self):
        """Echo boost effectively reduces gating threshold."""
        gate = GateState(
            coherence_threshold=0.72,
            current_coherence=0.65
        )

        # Gate would fail: 0.65 < 0.72
        assert gate.current_coherence < gate.coherence_threshold

        influence = EchoInfluence(
            boost_factor=0.10,
            harmonic_pull=None,
            trigger_chance=0.15
        )

        new_gate = apply_echo_influence(influence, gate)

        # After boost: 0.75 >= 0.72
        assert new_gate.current_coherence >= gate.coherence_threshold

    def test_harmonic_pull_biases_next_fragment(self):
        """Harmonic pull sets next fragment bias."""
        gate = GateState(
            coherence_threshold=0.72,
            current_coherence=0.70,
            next_fragment_bias=None
        )

        influence = EchoInfluence(
            boost_factor=0.05,
            harmonic_pull=OmegaFormat(3, 2),
            trigger_chance=0.12
        )

        new_gate = apply_echo_influence(influence, gate)

        assert new_gate.next_fragment_bias is not None

    def test_no_influence_below_threshold(self):
        """No influence returned when below threshold."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.5,
            base_intensity=0.3,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # After enough time, should have no influence
        influence = evaluate_echo_influence(echo, 20)
        assert influence is None


class TestEchoDeactivation:
    """Test echo deactivation thresholds."""

    def test_deactivation_below_005(self):
        """Echo deactivates below 0.05 intensity."""
        assert not is_echo_active(0.04)
        assert not is_echo_active(0.01)

    def test_active_above_005(self):
        """Echo active above 0.05 intensity."""
        assert is_echo_active(0.05)
        assert is_echo_active(0.06)

    def test_no_influence_below_010(self):
        """No influence below 0.10 intensity."""
        assert not has_echo_influence(0.09)
        assert not has_echo_influence(0.05)

    def test_influence_above_010(self):
        """Has influence above 0.10 intensity."""
        assert has_echo_influence(0.10)
        assert has_echo_influence(0.50)


class TestEchoFieldGeneration:
    """Test echo field generation from content."""

    def test_generate_from_stable_content(self):
        """Stable content produces slow-decay echo."""
        echo = generate_echo_field(
            fragment_id=42,
            content_stability=0.9,
            content_duration=8.0,
            harmonic=OmegaFormat(3, 1),
            current_phi_n=100
        )

        assert echo.decay_rate < 0.5  # Slow decay
        assert echo.base_intensity > 0.5  # Good intensity

    def test_generate_from_unstable_content(self):
        """Unstable content produces fast-decay echo."""
        echo = generate_echo_field(
            fragment_id=42,
            content_stability=0.2,
            content_duration=2.0,
            harmonic=OmegaFormat(1, 0),
            current_phi_n=100
        )

        assert echo.decay_rate > 0.7  # Fast decay
        assert echo.base_intensity < 0.2  # Low intensity


class TestEchoMerging:
    """Test echo field merging."""

    def test_merge_uses_max_intensity(self):
        """Merged echo uses max base intensity."""
        echoes = [
            EchoField(1, 0.3, 0.5, OmegaFormat(2, 1), 0),
            EchoField(2, 0.4, 0.8, OmegaFormat(3, 1), 5),
            EchoField(3, 0.2, 0.3, OmegaFormat(2, 0), 10),
        ]

        merged = merge_echo_fields(echoes)
        assert merged.base_intensity == 0.8

    def test_merge_averages_decay_rate(self):
        """Merged echo averages decay rate."""
        echoes = [
            EchoField(1, 0.3, 0.5, OmegaFormat(2, 1), 0),
            EchoField(2, 0.6, 0.5, OmegaFormat(2, 1), 0),
        ]

        merged = merge_echo_fields(echoes)
        assert abs(merged.decay_rate - 0.45) < 0.01

    def test_merge_uses_oldest_timestamp(self):
        """Merged echo uses oldest timestamp."""
        echoes = [
            EchoField(1, 0.3, 0.5, OmegaFormat(2, 1), 10),
            EchoField(2, 0.4, 0.5, OmegaFormat(2, 1), 5),
            EchoField(3, 0.2, 0.5, OmegaFormat(2, 1), 15),
        ]

        merged = merge_echo_fields(echoes)
        assert merged.timestamp_phi_n == 5


class TestEfficiencySkip:
    """Test efficiency skip conditions."""

    def test_skip_old_low_intensity_echo(self):
        """Skip evaluation for old, low-intensity echoes."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.5,
            base_intensity=0.6,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # Age > 10 and base_intensity <= 0.95
        assert should_skip_echo_evaluation(echo, 15)

    def test_dont_skip_old_high_intensity(self):
        """Don't skip old but high-intensity echoes."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.1,
            base_intensity=0.98,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # Age > 10 but base_intensity > 0.95
        assert not should_skip_echo_evaluation(echo, 15)

    def test_dont_skip_recent_echo(self):
        """Don't skip recent echoes."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.5,
            base_intensity=0.5,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=0
        )

        # Age <= 10
        assert not should_skip_echo_evaluation(echo, 8)


class TestHarmonicMatch:
    """Test harmonic matching logic."""

    def test_exact_match(self):
        """Exact l, m match returns true."""
        echo_h = OmegaFormat(3, 2, amplitude=0.7)
        field_h = OmegaFormat(3, 2, amplitude=0.75)

        assert check_harmonic_match(echo_h, field_h)

    def test_m_off_by_one_matches(self):
        """M off by one still matches."""
        echo_h = OmegaFormat(3, 2, amplitude=0.7)
        field_h = OmegaFormat(3, 3, amplitude=0.7)

        assert check_harmonic_match(echo_h, field_h)

    def test_l_mismatch_fails(self):
        """L mismatch fails."""
        echo_h = OmegaFormat(3, 2, amplitude=0.7)
        field_h = OmegaFormat(2, 2, amplitude=0.7)

        assert not check_harmonic_match(echo_h, field_h)


class TestPhiIntegration:
    """Test phi constant integration."""

    def test_phi_constant_defined(self):
        """Phi constant is correctly defined."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_n_tick_semantics(self):
        """φ^n ticks are used consistently."""
        echo = EchoField(
            fragment_id=1,
            decay_rate=0.2,
            base_intensity=0.8,
            harmonic_memory=OmegaFormat(2, 1),
            timestamp_phi_n=5
        )

        # At tick 5 (same as creation), intensity = base
        intensity = compute_echo_intensity(echo, 5)
        assert abs(intensity - 0.8) < 0.01


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
