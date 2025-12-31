"""
Prompt 36: Ra.Vector.Arm - Biometric to Motion Transduction (Hands-Free UI)

System for controlling UI elements through scalar-intention projection
using VectorArm polar structure with coherence-gated motion permission.

Codex References:
- Ra.Coherence: Coherence-gated activation
- Ra.Phase: Sync phase alignment
- Kaali-Beck: EEG/EMG entrainment
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895
TWO_PI = 2 * math.pi
PI = math.pi

# Coherence thresholds (configurable)
COHERENCE_FULL = 0.82
COHERENCE_PARTIAL_MIN = 0.55
COHERENCE_BLOCKED = 0.55

# Trigger thresholds
BREATH_HOLD_THRESHOLD = 4.5  # seconds
EEG_SPIKE_MULTIPLIER = 1.5   # 1.5x baseline
EEG_SPIKE_DURATION = 0.4     # 400ms

# Harmonic affinity delta
AFFINITY_DELTA = 1


# ============================================================================
# Types
# ============================================================================

class MotionPermission(Enum):
    """Coherence-gated motion permission levels."""
    FULL = auto()      # Free motion
    PARTIAL = auto()   # Damped motion with visual lag
    BLOCKED = auto()   # Frozen vector


class InteractionForm(Enum):
    """UI element interaction states."""
    IDLE = auto()
    HOVER = auto()
    ACTIVE = auto()
    LOCKED = auto()


class PhaseState(Enum):
    """Scalar rhythm phase states."""
    RISING = auto()
    PEAK = auto()
    FALLING = auto()
    TROUGH = auto()


@dataclass
class CoherenceThresholds:
    """Configurable coherence thresholds."""
    full: float = COHERENCE_FULL
    partial_min: float = COHERENCE_PARTIAL_MIN


@dataclass
class TriggerThresholds:
    """Configurable trigger detection thresholds."""
    breath_hold_seconds: float = BREATH_HOLD_THRESHOLD
    eeg_spike_multiplier: float = EEG_SPIKE_MULTIPLIER
    eeg_spike_duration: float = EEG_SPIKE_DURATION


@dataclass
class VectorArm:
    """
    Ra vector arm with polar structure for intention projection.

    - angleTheta: 1-13 (azimuth zones)
    - anglePhi: 1-12 (harmonic inclination bands)
    - reachDepth: 0.0-1.0 (phi-aligned scalar depth)
    """
    angle_theta: int           # 1-13 azimuth zone
    angle_phi: int             # 1-12 inclination band
    reach_depth: float         # 0.0-1.0
    coherence_pulse: float     # HRV/breath coherence 0.0-1.0
    sync_phase: PhaseState

    def theta_radians(self) -> float:
        """Convert discrete theta (1-13) to radians (0-2π)."""
        return ((self.angle_theta - 1) / 13) * TWO_PI

    def phi_radians(self) -> float:
        """Convert discrete phi (1-12) to radians (0-π)."""
        return ((self.angle_phi - 1) / 12) * PI

    def to_cartesian(self) -> Tuple[float, float, float]:
        """Convert polar to Cartesian coordinates."""
        theta = self.theta_radians()
        phi = self.phi_radians()
        r = self.reach_depth

        x = r * math.sin(phi) * math.cos(theta)
        y = r * math.sin(phi) * math.sin(theta)
        z = r * math.cos(phi)
        return (x, y, z)


@dataclass
class UIElement:
    """UI element with scalar resonance field."""
    element_id: str
    position: Tuple[float, float]
    scalar_resonance: Tuple[int, int]  # (l, m) harmonics
    interaction_state: InteractionForm = InteractionForm.IDLE


@dataclass
class BiometricInput:
    """Biometric input for vector intent evaluation."""
    hrv_coherence: float       # 0.0-1.0
    breath_phase: float        # 0.0-1.0 (cycle position)
    breath_hold_duration: float  # seconds
    eeg_amplitude: float       # current amplitude
    eeg_baseline: float        # baseline amplitude
    eeg_spike_duration: float  # seconds at elevated level


@dataclass
class UICommand:
    """Commands generated from VectorArm."""
    command_type: str  # "move", "activate", "drag", "scalar_mode"
    target_id: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    intensity: float = 1.0


# ============================================================================
# Core Functions
# ============================================================================

def get_motion_permission(coherence: float,
                          thresholds: CoherenceThresholds = None) -> MotionPermission:
    """
    Determine motion permission based on coherence level.

    FULL: coherence >= 0.82 -> free motion
    PARTIAL: 0.55-0.81 -> damped motion
    BLOCKED: < 0.55 -> frozen
    """
    thresholds = thresholds or CoherenceThresholds()

    if coherence >= thresholds.full:
        return MotionPermission.FULL
    elif coherence >= thresholds.partial_min:
        return MotionPermission.PARTIAL
    else:
        return MotionPermission.BLOCKED


def calculate_drag_weight(coherence: float, permission: MotionPermission) -> float:
    """
    Calculate cursor drag weight based on coherence.

    FULL: 1.0 (no drag)
    PARTIAL: interpolated (0.3-0.8)
    BLOCKED: 0.0 (frozen)
    """
    if permission == MotionPermission.FULL:
        return 1.0
    elif permission == MotionPermission.PARTIAL:
        # Interpolate drag weight
        normalized = (coherence - COHERENCE_PARTIAL_MIN) / (COHERENCE_FULL - COHERENCE_PARTIAL_MIN)
        return 0.3 + (normalized * 0.5)
    else:
        return 0.0


def check_harmonic_affinity(arm_harmonics: Tuple[int, int],
                            element_harmonics: Tuple[int, int],
                            delta: int = AFFINITY_DELTA) -> bool:
    """
    Check if VectorArm harmonics match UIElement within delta tolerance.

    Per-component matching: each component must be within ±delta.
    """
    l1, m1 = arm_harmonics
    l2, m2 = element_harmonics

    return (abs(l1 - l2) <= delta and abs(m1 - m2) <= delta)


def detect_breath_hold_trigger(bio: BiometricInput,
                               thresholds: TriggerThresholds = None) -> bool:
    """Detect breath hold trigger (≥4.5 seconds)."""
    thresholds = thresholds or TriggerThresholds()
    return bio.breath_hold_duration >= thresholds.breath_hold_seconds


def detect_eeg_spike_trigger(bio: BiometricInput,
                             thresholds: TriggerThresholds = None) -> bool:
    """Detect EEG microfocus spike (≥1.5x baseline for 400ms)."""
    thresholds = thresholds or TriggerThresholds()

    amplitude_ratio = bio.eeg_amplitude / max(bio.eeg_baseline, 0.001)
    duration_met = bio.eeg_spike_duration >= thresholds.eeg_spike_duration

    return (amplitude_ratio >= thresholds.eeg_spike_multiplier and duration_met)


def evaluate_vector_intent(bio: BiometricInput) -> VectorArm:
    """
    Map biometric input to VectorArm direction, reach, and strength.

    Synchronizes to 500ms intervals per Prompt 15 alignment.
    """
    # Derive theta from breath phase (maps to 13 zones)
    theta = int(1 + (bio.breath_phase * 12))
    theta = max(1, min(13, theta))

    # Derive phi from HRV coherence (maps to 12 bands)
    phi = int(1 + (bio.hrv_coherence * 11))
    phi = max(1, min(12, phi))

    # Reach depth from coherence
    reach = bio.hrv_coherence * PHI
    reach = min(1.0, reach)

    # Phase state from breath phase
    if bio.breath_phase < 0.25:
        phase = PhaseState.RISING
    elif bio.breath_phase < 0.5:
        phase = PhaseState.PEAK
    elif bio.breath_phase < 0.75:
        phase = PhaseState.FALLING
    else:
        phase = PhaseState.TROUGH

    return VectorArm(
        angle_theta=theta,
        angle_phi=phi,
        reach_depth=reach,
        coherence_pulse=bio.hrv_coherence,
        sync_phase=phase
    )


def vector_to_command(arm: VectorArm,
                      elements: List[UIElement],
                      bio: BiometricInput) -> Optional[UICommand]:
    """
    Translate VectorArm to UICommand based on coherence, reach, and triggers.
    """
    permission = get_motion_permission(arm.coherence_pulse)

    if permission == MotionPermission.BLOCKED:
        return None

    # Convert arm to 2D screen position
    cart = arm.to_cartesian()
    screen_x = 0.5 + (cart[0] * 0.5)  # Normalize to 0-1
    screen_y = 0.5 + (cart[1] * 0.5)

    # Find matching element
    arm_harmonics = (arm.angle_theta % 10, arm.angle_phi % 10)
    target_element = None

    for elem in elements:
        if check_harmonic_affinity(arm_harmonics, elem.scalar_resonance):
            target_element = elem
            break

    # Check for triggers
    if target_element and detect_breath_hold_trigger(bio):
        return UICommand(
            command_type="activate",
            target_id=target_element.element_id,
            intensity=arm.coherence_pulse
        )
    elif target_element and detect_eeg_spike_trigger(bio):
        return UICommand(
            command_type="activate",
            target_id=target_element.element_id,
            intensity=arm.coherence_pulse * 1.2
        )
    else:
        drag_weight = calculate_drag_weight(arm.coherence_pulse, permission)
        return UICommand(
            command_type="move",
            position=(screen_x, screen_y),
            intensity=drag_weight
        )


# ============================================================================
# Test Suite
# ============================================================================

class TestMotionPermission:
    """Test coherence-gated motion permission."""

    def test_full_permission(self):
        """High coherence grants full motion."""
        assert get_motion_permission(0.90) == MotionPermission.FULL
        assert get_motion_permission(0.82) == MotionPermission.FULL

    def test_partial_permission(self):
        """Mid coherence grants partial motion."""
        assert get_motion_permission(0.70) == MotionPermission.PARTIAL
        assert get_motion_permission(0.55) == MotionPermission.PARTIAL

    def test_blocked_permission(self):
        """Low coherence blocks motion."""
        assert get_motion_permission(0.50) == MotionPermission.BLOCKED
        assert get_motion_permission(0.20) == MotionPermission.BLOCKED

    def test_custom_thresholds(self):
        """Custom thresholds are respected."""
        custom = CoherenceThresholds(full=0.90, partial_min=0.60)
        assert get_motion_permission(0.85, custom) == MotionPermission.PARTIAL
        assert get_motion_permission(0.55, custom) == MotionPermission.BLOCKED


class TestDragWeight:
    """Test cursor drag weight calculation."""

    def test_full_no_drag(self):
        """Full permission has no drag."""
        weight = calculate_drag_weight(0.90, MotionPermission.FULL)
        assert weight == 1.0

    def test_partial_interpolated(self):
        """Partial permission has interpolated drag."""
        weight = calculate_drag_weight(0.70, MotionPermission.PARTIAL)
        assert 0.3 < weight < 0.8

    def test_blocked_frozen(self):
        """Blocked permission is frozen."""
        weight = calculate_drag_weight(0.30, MotionPermission.BLOCKED)
        assert weight == 0.0


class TestHarmonicAffinity:
    """Test harmonic affinity matching."""

    def test_exact_match(self):
        """Exact harmonics match."""
        assert check_harmonic_affinity((3, 2), (3, 2)) is True

    def test_within_delta(self):
        """Harmonics within ±1 match."""
        assert check_harmonic_affinity((3, 2), (4, 2)) is True
        assert check_harmonic_affinity((3, 2), (3, 3)) is True
        assert check_harmonic_affinity((3, 2), (4, 3)) is True

    def test_outside_delta(self):
        """Harmonics outside ±1 don't match."""
        assert check_harmonic_affinity((3, 2), (5, 2)) is False
        assert check_harmonic_affinity((3, 2), (3, 5)) is False

    def test_custom_delta(self):
        """Custom delta is respected."""
        assert check_harmonic_affinity((3, 2), (5, 4), delta=2) is True


class TestTriggers:
    """Test breath hold and EEG spike triggers."""

    def test_breath_hold_trigger(self):
        """Breath hold ≥4.5s triggers."""
        bio = BiometricInput(
            hrv_coherence=0.8,
            breath_phase=0.5,
            breath_hold_duration=5.0,
            eeg_amplitude=1.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.0
        )
        assert detect_breath_hold_trigger(bio) is True

    def test_breath_hold_no_trigger(self):
        """Breath hold <4.5s doesn't trigger."""
        bio = BiometricInput(
            hrv_coherence=0.8,
            breath_phase=0.5,
            breath_hold_duration=3.0,
            eeg_amplitude=1.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.0
        )
        assert detect_breath_hold_trigger(bio) is False

    def test_eeg_spike_trigger(self):
        """EEG spike ≥1.5x for 400ms triggers."""
        bio = BiometricInput(
            hrv_coherence=0.8,
            breath_phase=0.5,
            breath_hold_duration=0.0,
            eeg_amplitude=2.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.5
        )
        assert detect_eeg_spike_trigger(bio) is True

    def test_eeg_spike_insufficient_amplitude(self):
        """EEG spike <1.5x doesn't trigger."""
        bio = BiometricInput(
            hrv_coherence=0.8,
            breath_phase=0.5,
            breath_hold_duration=0.0,
            eeg_amplitude=1.3,
            eeg_baseline=1.0,
            eeg_spike_duration=0.5
        )
        assert detect_eeg_spike_trigger(bio) is False

    def test_eeg_spike_insufficient_duration(self):
        """EEG spike <400ms doesn't trigger."""
        bio = BiometricInput(
            hrv_coherence=0.8,
            breath_phase=0.5,
            breath_hold_duration=0.0,
            eeg_amplitude=2.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.2
        )
        assert detect_eeg_spike_trigger(bio) is False


class TestVectorArm:
    """Test VectorArm creation and conversion."""

    def test_theta_radians(self):
        """Theta converts to 0-2π range."""
        # Zone 1 maps to 0 radians
        arm = VectorArm(1, 6, 0.5, 0.8, PhaseState.PEAK)
        assert arm.theta_radians() == 0.0

        # Zone 7 maps to (6/13)*2π ≈ 2.899 radians
        arm2 = VectorArm(7, 6, 0.5, 0.8, PhaseState.PEAK)
        expected = (6 / 13) * TWO_PI
        assert abs(arm2.theta_radians() - expected) < 0.01

    def test_phi_radians(self):
        """Phi converts to 0-π range."""
        arm = VectorArm(7, 1, 0.5, 0.8, PhaseState.PEAK)
        assert arm.phi_radians() == 0.0

        arm2 = VectorArm(7, 7, 0.5, 0.8, PhaseState.PEAK)
        assert abs(arm2.phi_radians() - PI/2) < 0.1

    def test_cartesian_conversion(self):
        """Polar to Cartesian conversion."""
        # Test at zone 7, band 7: theta=(6/13)*2π≈2.899, phi=π/2
        arm = VectorArm(7, 7, 1.0, 0.8, PhaseState.PEAK)
        x, y, z = arm.to_cartesian()
        theta = (6 / 13) * TWO_PI
        # At this theta, phi=π/2, r=1:
        # x = sin(π/2)*cos(θ) ≈ cos(2.899) ≈ -0.959
        # y = sin(π/2)*sin(θ) ≈ sin(2.899) ≈ 0.283
        # z = cos(π/2) = 0
        assert abs(x - math.cos(theta)) < 0.1
        assert abs(y - math.sin(theta)) < 0.1
        assert abs(z) < 0.1


class TestEvaluateVectorIntent:
    """Test biometric to VectorArm mapping."""

    def test_high_coherence_deep_reach(self):
        """High coherence produces deep reach."""
        bio = BiometricInput(
            hrv_coherence=0.9,
            breath_phase=0.5,
            breath_hold_duration=0.0,
            eeg_amplitude=1.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.0
        )
        arm = evaluate_vector_intent(bio)
        assert arm.reach_depth > 0.8
        assert arm.coherence_pulse == 0.9

    def test_breath_phase_to_theta(self):
        """Breath phase maps to theta zone."""
        bio = BiometricInput(
            hrv_coherence=0.5,
            breath_phase=0.0,
            breath_hold_duration=0.0,
            eeg_amplitude=1.0,
            eeg_baseline=1.0,
            eeg_spike_duration=0.0
        )
        arm = evaluate_vector_intent(bio)
        assert arm.angle_theta == 1

        bio.breath_phase = 1.0
        arm2 = evaluate_vector_intent(bio)
        assert arm2.angle_theta == 13


class TestVectorToCommand:
    """Test VectorArm to UICommand translation."""

    def test_blocked_returns_none(self):
        """Blocked coherence returns no command."""
        arm = VectorArm(7, 6, 0.5, 0.30, PhaseState.PEAK)
        bio = BiometricInput(0.30, 0.5, 0.0, 1.0, 1.0, 0.0)
        cmd = vector_to_command(arm, [], bio)
        assert cmd is None

    def test_move_command(self):
        """Normal coherence produces move command."""
        arm = VectorArm(7, 6, 0.5, 0.85, PhaseState.PEAK)
        bio = BiometricInput(0.85, 0.5, 0.0, 1.0, 1.0, 0.0)
        cmd = vector_to_command(arm, [], bio)
        assert cmd is not None
        assert cmd.command_type == "move"

    def test_activate_on_breath_hold(self):
        """Breath hold activates matching element."""
        arm = VectorArm(7, 6, 0.5, 0.85, PhaseState.PEAK)
        elem = UIElement("btn1", (0.5, 0.5), (7, 6))
        bio = BiometricInput(0.85, 0.5, 5.0, 1.0, 1.0, 0.0)
        cmd = vector_to_command(arm, [elem], bio)
        assert cmd is not None
        assert cmd.command_type == "activate"
        assert cmd.target_id == "btn1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
