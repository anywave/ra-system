"""
Prompt 37: Ra.Emitters.JoeBubble - Joe Cell Scalar Bubble Emitter

Virtual Joe Cell chamber simulation with layered scalar shells,
stage progression, and operator Y-factor integration.

Codex References:
- JOE_CELL_SPECIFICATIONS.md: Mechanical/energetic behavior
- Ra.Emergence: Fragment emergence assist
- Ra.Identity: Operator resonance
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import time


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Stage thresholds (coherence, duration in seconds)
STAGE_THRESHOLDS = {
    1: (0.0, 0.0),      # Stage 1: No requirements
    2: (0.55, 30.0),    # Stage 2: coherence ≥0.55 for 30s
    3: (0.70, 60.0),    # Stage 3: coherence ≥0.70 for 60s
    4: (0.85, 120.0),   # Stage 4: coherence ≥0.85 for 120s
    5: (0.90, 0.0),     # Stage 5: Group sync or intent lock-in
}

# Default shell radii (phi-progression)
DEFAULT_SHELL_RADII = [1.0, PHI, PHI**2, PHI**3, PHI**4]

# Polarity mismatch penalty
POLARITY_MISMATCH_PENALTY = 0.6


# ============================================================================
# Types
# ============================================================================

class JoeStage(Enum):
    """Joe Cell stage progression."""
    STAGE1 = 1  # Electrolysis-only, no scalar effects
    STAGE2 = 2  # Seeding phase
    STAGE3 = 3  # Breeding - visible field
    STAGE4 = 4  # Commercial - stable
    STAGE5 = 5  # Integration - part of environment


class Polarity(Enum):
    """Field polarity states."""
    POSITIVE = auto()
    NEGATIVE = auto()
    INVERTED = auto()


@dataclass
class ScalarShell:
    """
    Concentric scalar layer for bubble containment.

    Radii follow phi-progression: [1.0, 1.618, 2.618, 4.236, ...]
    """
    radius: float
    frequency_band: Tuple[float, float]  # Hz range
    impedance_level: float  # Orgone containment resistance


@dataclass
class OperatorResonance:
    """User/operator Y-factor resonance."""
    sync_heart_rate: float   # HRV match with cell pulse
    intent_alignment: float  # Scalar vector overlap (0-1)
    body_polarity: Polarity


@dataclass
class SpatialField:
    """Directional scalar influence field."""
    direction_theta: int   # Azimuth sector (1-13)
    harmonic_shell: int    # Depth index
    field_strength: float  # Local amplification modifier


@dataclass
class BiometricInput:
    """Biometric state for stage advancement."""
    coherence: float
    heart_rate: float
    intent_focus: float


@dataclass
class EmitterStageProfile:
    """Configurable stage transition thresholds."""
    coherence_threshold: float
    duration_threshold: float  # seconds


@dataclass
class BubbleEmitter:
    """
    Virtual Joe Cell bubble emitter.

    Simulates entrainment container with layered scalar shells.
    """
    stage: JoeStage
    harmonic_layers: List[ScalarShell]
    resonance_level: float  # 0.0-1.0
    operator_field: Optional[OperatorResonance]
    polarity_state: Polarity
    bubble_pulse_rate: float  # Hz
    emitter_zone: SpatialField
    # Stage tracking
    stage_coherence_start: float = 0.0
    stage_duration: float = 0.0


# ============================================================================
# Shell Construction
# ============================================================================

def create_default_shells() -> List[ScalarShell]:
    """
    Create default scalar shells with phi-progression radii.

    Radii: [1.0, 1.618, 2.618, 4.236, 6.854]
    """
    shells = []
    for i, radius in enumerate(DEFAULT_SHELL_RADII):
        # Frequency bands increase with radius
        base_freq = 100.0 * (i + 1)
        freq_band = (base_freq, base_freq * PHI)
        # Impedance decreases outward
        impedance = 1.0 / (1.0 + i * 0.2)
        shells.append(ScalarShell(radius, freq_band, impedance))
    return shells


def create_shell_at_radius(radius: float, index: int) -> ScalarShell:
    """Create a single shell at specified radius."""
    base_freq = 100.0 * (index + 1)
    freq_band = (base_freq, base_freq * PHI)
    impedance = 1.0 / (1.0 + index * 0.2)
    return ScalarShell(radius, freq_band, impedance)


# ============================================================================
# Stage Progression
# ============================================================================

def can_advance_stage(emitter: BubbleEmitter,
                      bio: BiometricInput,
                      elapsed_time: float) -> bool:
    """
    Check if emitter can advance to next stage.

    Stage transitions based on coherence level and duration.
    """
    current_stage = emitter.stage.value
    if current_stage >= 5:
        return False  # Already at max stage

    next_stage = current_stage + 1
    threshold = STAGE_THRESHOLDS.get(next_stage, (1.0, float('inf')))
    required_coherence, required_duration = threshold

    # Check coherence meets threshold
    if bio.coherence < required_coherence:
        return False

    # Check duration at threshold
    if elapsed_time < required_duration:
        return False

    # Stage 5 requires group sync or special conditions
    if next_stage == 5:
        # Check for intent lock-in (high alignment)
        if emitter.operator_field:
            return emitter.operator_field.intent_alignment >= 0.95
        return False

    return True


def advance_stage(emitter: BubbleEmitter,
                  bio: BiometricInput,
                  elapsed_time: float) -> BubbleEmitter:
    """
    Advance emitter to next stage if conditions met.

    Returns new emitter state (doesn't mutate original).
    """
    if not can_advance_stage(emitter, bio, elapsed_time):
        return emitter

    new_stage = JoeStage(emitter.stage.value + 1)

    # Increase resonance level with stage
    new_resonance = min(1.0, emitter.resonance_level + 0.15)

    # Add new shell layer at each stage
    new_layers = emitter.harmonic_layers.copy()
    if len(new_layers) < len(DEFAULT_SHELL_RADII):
        new_layers.append(create_shell_at_radius(
            DEFAULT_SHELL_RADII[len(new_layers)],
            len(new_layers)
        ))

    return BubbleEmitter(
        stage=new_stage,
        harmonic_layers=new_layers,
        resonance_level=new_resonance,
        operator_field=emitter.operator_field,
        polarity_state=emitter.polarity_state,
        bubble_pulse_rate=emitter.bubble_pulse_rate,
        emitter_zone=emitter.emitter_zone,
        stage_coherence_start=bio.coherence,
        stage_duration=0.0
    )


# ============================================================================
# Operator Integration
# ============================================================================

def calculate_polarity_match(emitter_polarity: Polarity,
                             operator_polarity: Polarity) -> float:
    """
    Calculate polarity match factor.

    Returns 1.0 for match, POLARITY_MISMATCH_PENALTY for mismatch.
    """
    if emitter_polarity == operator_polarity:
        return 1.0
    elif emitter_polarity == Polarity.INVERTED or operator_polarity == Polarity.INVERTED:
        return POLARITY_MISMATCH_PENALTY * 0.8  # Extra penalty for inversion
    else:
        return POLARITY_MISMATCH_PENALTY


def apply_operator_resonance(emitter: BubbleEmitter,
                             operator: OperatorResonance) -> float:
    """
    Calculate effective resonance with operator Y-factor.

    Polarity mismatch degrades efficiency but doesn't block.
    """
    base_resonance = emitter.resonance_level
    polarity_factor = calculate_polarity_match(emitter.polarity_state,
                                                operator.body_polarity)

    # Combine with intent alignment
    alignment_factor = operator.intent_alignment

    # Heart rate sync bonus
    pulse_diff = abs(emitter.bubble_pulse_rate - operator.sync_heart_rate / 60.0)
    sync_factor = max(0.5, 1.0 - pulse_diff * 0.1)

    return base_resonance * polarity_factor * alignment_factor * sync_factor


# ============================================================================
# Bubble Pulse Simulation
# ============================================================================

def calculate_pulse_rate(emitter: BubbleEmitter,
                         operators: List[OperatorResonance]) -> float:
    """
    Calculate bubble pulse rate.

    Start at 2 Hz, stabilize to heart-synced rhythm.
    Group average if multiple operators.
    """
    base_rate = 2.0

    if not operators:
        return base_rate

    # Average heart rate of operators
    avg_heart_rate = sum(op.sync_heart_rate for op in operators) / len(operators)
    target_rate = avg_heart_rate / 60.0  # Convert BPM to Hz

    # Blend based on stage (higher stages sync better)
    stage_factor = emitter.stage.value / 5.0
    blended_rate = base_rate * (1 - stage_factor) + target_rate * stage_factor

    return blended_rate


def get_emergence_boost(emitter: BubbleEmitter) -> float:
    """
    Get emergence boost factor for fragments near emitter.

    Higher stages provide stronger boost.
    """
    stage_multiplier = {
        JoeStage.STAGE1: 0.0,
        JoeStage.STAGE2: 0.1,
        JoeStage.STAGE3: 0.25,
        JoeStage.STAGE4: 0.4,
        JoeStage.STAGE5: 0.6,
    }
    return stage_multiplier.get(emitter.stage, 0.0) * emitter.resonance_level


# ============================================================================
# Factory Functions
# ============================================================================

def create_emitter(polarity: Polarity = Polarity.POSITIVE) -> BubbleEmitter:
    """Create a new Stage 1 bubble emitter."""
    return BubbleEmitter(
        stage=JoeStage.STAGE1,
        harmonic_layers=[create_default_shells()[0]],
        resonance_level=0.1,
        operator_field=None,
        polarity_state=polarity,
        bubble_pulse_rate=2.0,
        emitter_zone=SpatialField(7, 1, 0.5)
    )


# ============================================================================
# Test Suite
# ============================================================================

class TestScalarShells:
    """Test scalar shell construction."""

    def test_default_shells_phi_progression(self):
        """Default shells follow phi-progression radii."""
        shells = create_default_shells()
        assert len(shells) == 5
        assert shells[0].radius == 1.0
        assert abs(shells[1].radius - PHI) < 0.001
        assert abs(shells[2].radius - PHI**2) < 0.001

    def test_shell_frequency_bands(self):
        """Shell frequency bands increase with layer."""
        shells = create_default_shells()
        for i in range(len(shells) - 1):
            assert shells[i+1].frequency_band[0] > shells[i].frequency_band[0]

    def test_shell_impedance_decreases(self):
        """Impedance decreases outward."""
        shells = create_default_shells()
        for i in range(len(shells) - 1):
            assert shells[i+1].impedance_level < shells[i].impedance_level


class TestStageProgression:
    """Test Joe Cell stage transitions."""

    def test_stage1_no_requirements(self):
        """Stage 1 has no requirements."""
        emitter = create_emitter()
        assert emitter.stage == JoeStage.STAGE1

    def test_advance_to_stage2(self):
        """Advance to Stage 2 with coherence ≥0.55 for 30s."""
        emitter = create_emitter()
        bio = BiometricInput(coherence=0.60, heart_rate=72, intent_focus=0.5)
        new_emitter = advance_stage(emitter, bio, 35.0)
        assert new_emitter.stage == JoeStage.STAGE2

    def test_no_advance_insufficient_coherence(self):
        """No advance if coherence too low."""
        emitter = create_emitter()
        bio = BiometricInput(coherence=0.40, heart_rate=72, intent_focus=0.5)
        new_emitter = advance_stage(emitter, bio, 60.0)
        assert new_emitter.stage == JoeStage.STAGE1

    def test_no_advance_insufficient_duration(self):
        """No advance if duration too short."""
        emitter = create_emitter()
        bio = BiometricInput(coherence=0.60, heart_rate=72, intent_focus=0.5)
        new_emitter = advance_stage(emitter, bio, 10.0)
        assert new_emitter.stage == JoeStage.STAGE1

    def test_advance_to_stage3(self):
        """Advance to Stage 3 requires higher coherence."""
        emitter = create_emitter()
        emitter = BubbleEmitter(
            stage=JoeStage.STAGE2,
            harmonic_layers=create_default_shells()[:2],
            resonance_level=0.25,
            operator_field=None,
            polarity_state=Polarity.POSITIVE,
            bubble_pulse_rate=2.0,
            emitter_zone=SpatialField(7, 1, 0.5)
        )
        bio = BiometricInput(coherence=0.75, heart_rate=72, intent_focus=0.7)
        new_emitter = advance_stage(emitter, bio, 65.0)
        assert new_emitter.stage == JoeStage.STAGE3

    def test_resonance_increases_with_stage(self):
        """Resonance level increases at each stage."""
        emitter = create_emitter()
        bio = BiometricInput(coherence=0.60, heart_rate=72, intent_focus=0.5)
        new_emitter = advance_stage(emitter, bio, 35.0)
        assert new_emitter.resonance_level > emitter.resonance_level


class TestPolarityInteraction:
    """Test polarity match and mismatch effects."""

    def test_polarity_match_full_efficiency(self):
        """Matching polarity gives full efficiency."""
        factor = calculate_polarity_match(Polarity.POSITIVE, Polarity.POSITIVE)
        assert factor == 1.0

    def test_polarity_mismatch_penalty(self):
        """Mismatched polarity applies penalty."""
        factor = calculate_polarity_match(Polarity.POSITIVE, Polarity.NEGATIVE)
        assert factor == POLARITY_MISMATCH_PENALTY

    def test_inverted_extra_penalty(self):
        """Inverted polarity has extra penalty."""
        factor = calculate_polarity_match(Polarity.POSITIVE, Polarity.INVERTED)
        assert factor < POLARITY_MISMATCH_PENALTY


class TestOperatorResonance:
    """Test operator Y-factor integration."""

    def test_operator_resonance_calculation(self):
        """Operator resonance affects effective resonance."""
        emitter = create_emitter()
        emitter = BubbleEmitter(
            stage=JoeStage.STAGE3,
            harmonic_layers=create_default_shells()[:3],
            resonance_level=0.5,
            operator_field=None,
            polarity_state=Polarity.POSITIVE,
            bubble_pulse_rate=1.2,
            emitter_zone=SpatialField(7, 2, 0.6)
        )
        operator = OperatorResonance(
            sync_heart_rate=72,
            intent_alignment=0.8,
            body_polarity=Polarity.POSITIVE
        )
        effective = apply_operator_resonance(emitter, operator)
        assert 0.0 < effective < emitter.resonance_level

    def test_high_alignment_boosts_resonance(self):
        """High intent alignment boosts effective resonance."""
        emitter = create_emitter()
        emitter = BubbleEmitter(
            stage=JoeStage.STAGE3,
            harmonic_layers=create_default_shells()[:3],
            resonance_level=0.5,
            operator_field=None,
            polarity_state=Polarity.POSITIVE,
            bubble_pulse_rate=1.2,
            emitter_zone=SpatialField(7, 2, 0.6)
        )
        op_low = OperatorResonance(72, 0.3, Polarity.POSITIVE)
        op_high = OperatorResonance(72, 0.9, Polarity.POSITIVE)

        eff_low = apply_operator_resonance(emitter, op_low)
        eff_high = apply_operator_resonance(emitter, op_high)
        assert eff_high > eff_low


class TestBubblePulse:
    """Test bubble pulse rate calculation."""

    def test_base_pulse_without_operators(self):
        """Base pulse is 2 Hz without operators."""
        emitter = create_emitter()
        rate = calculate_pulse_rate(emitter, [])
        assert rate == 2.0

    def test_pulse_syncs_to_heart_rate(self):
        """Pulse syncs to operator heart rate at higher stages."""
        emitter = BubbleEmitter(
            stage=JoeStage.STAGE4,
            harmonic_layers=create_default_shells()[:4],
            resonance_level=0.6,
            operator_field=None,
            polarity_state=Polarity.POSITIVE,
            bubble_pulse_rate=2.0,
            emitter_zone=SpatialField(7, 3, 0.7)
        )
        operator = OperatorResonance(60, 0.8, Polarity.POSITIVE)
        rate = calculate_pulse_rate(emitter, [operator])
        # Should blend toward 1.0 Hz (60 BPM = 1 Hz)
        assert rate < 2.0


class TestEmergenceBoost:
    """Test fragment emergence boost from emitter."""

    def test_stage1_no_boost(self):
        """Stage 1 provides no emergence boost."""
        emitter = create_emitter()
        assert get_emergence_boost(emitter) == 0.0

    def test_higher_stages_more_boost(self):
        """Higher stages provide more boost."""
        boosts = []
        for stage in JoeStage:
            emitter = BubbleEmitter(
                stage=stage,
                harmonic_layers=create_default_shells()[:stage.value],
                resonance_level=0.8,
                operator_field=None,
                polarity_state=Polarity.POSITIVE,
                bubble_pulse_rate=2.0,
                emitter_zone=SpatialField(7, 1, 0.5)
            )
            boosts.append(get_emergence_boost(emitter))

        # Each stage should have higher boost
        for i in range(len(boosts) - 1):
            assert boosts[i+1] >= boosts[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
