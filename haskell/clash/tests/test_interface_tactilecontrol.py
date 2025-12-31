"""
Test Harness for Prompt 56: Ra.Interface.TactileControl

Tests intent detection from biometric signals for tactile interface control.
Converts coherence patterns, HRV spikes, and breath rate into ControlIntents
for digital twin manipulation and interface navigation.

Based on:
- Biometric combo detection (coherence + HRV spike + breath rate)
- Intent classification mapping
- Confidence-weighted intent emission
"""

import pytest
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Biometric thresholds
COHERENCE_HIGH = 0.72  # High coherence threshold
COHERENCE_MED = 0.50   # Medium coherence threshold
COHERENCE_LOW = 0.30   # Low coherence threshold

HRV_SPIKE_THRESHOLD = 0.15  # HRV change rate threshold for spike
BREATH_FAST = 18  # breaths/min - fast breathing
BREATH_SLOW = 8   # breaths/min - slow breathing
BREATH_NORMAL_MIN = 10
BREATH_NORMAL_MAX = 16

# Intent confidence minimum
INTENT_CONFIDENCE_MIN = 0.60


# =============================================================================
# Data Structures
# =============================================================================

class ControlIntent(Enum):
    """Control intents for tactile interface."""
    NONE = auto()
    REACH = auto()      # Extend/approach
    PULL = auto()       # Attract/gather
    PUSH = auto()       # Repel/send
    GRASP = auto()      # Hold/acquire
    RELEASE = auto()    # Let go/drop
    MOVE_TO = auto()    # Navigate to position
    HOVER_AT = auto()   # Maintain position


@dataclass(frozen=True)
class BiometricState:
    """Current biometric readings."""
    coherence: float       # 0-1 coherence level
    hrv_current: float     # Current HRV (ms)
    hrv_previous: float    # Previous HRV (ms)
    breath_rate: float     # Breaths per minute
    timestamp: float       # Time in seconds

    @property
    def hrv_delta(self) -> float:
        """HRV change rate."""
        if self.hrv_previous == 0:
            return 0.0
        return (self.hrv_current - self.hrv_previous) / self.hrv_previous

    @property
    def hrv_spike(self) -> bool:
        """Whether HRV spike detected."""
        return abs(self.hrv_delta) > HRV_SPIKE_THRESHOLD


@dataclass(frozen=True)
class IntentResult:
    """Result of intent detection."""
    intent: ControlIntent
    confidence: float       # 0-1 confidence level
    coherence_component: float  # Coherence contribution
    hrv_component: float    # HRV contribution
    breath_component: float # Breath contribution
    is_valid: bool          # Meets confidence threshold


@dataclass
class IntentHistory:
    """History of recent intents for smoothing."""
    intents: List[IntentResult]
    max_history: int = 10

    def add(self, result: IntentResult):
        """Add result to history."""
        self.intents.append(result)
        if len(self.intents) > self.max_history:
            self.intents.pop(0)

    def get_dominant_intent(self) -> Optional[ControlIntent]:
        """Get most common valid intent in history."""
        valid = [r.intent for r in self.intents if r.is_valid]
        if not valid:
            return None
        # Count occurrences
        counts = {}
        for intent in valid:
            counts[intent] = counts.get(intent, 0) + 1
        return max(counts, key=counts.get)


# =============================================================================
# Core Functions
# =============================================================================

def classify_coherence_level(coherence: float) -> str:
    """Classify coherence into high/medium/low."""
    if coherence >= COHERENCE_HIGH:
        return "high"
    elif coherence >= COHERENCE_MED:
        return "medium"
    elif coherence >= COHERENCE_LOW:
        return "low"
    else:
        return "none"


def classify_breath_pattern(breath_rate: float) -> str:
    """Classify breath rate pattern."""
    if breath_rate > BREATH_FAST:
        return "fast"
    elif breath_rate < BREATH_SLOW:
        return "slow"
    elif BREATH_NORMAL_MIN <= breath_rate <= BREATH_NORMAL_MAX:
        return "normal"
    else:
        return "moderate"


def compute_coherence_component(coherence: float) -> Tuple[float, List[ControlIntent]]:
    """
    Compute coherence contribution and suggested intents.

    Returns (contribution, [possible_intents])
    """
    if coherence >= COHERENCE_HIGH:
        # High coherence - precise actions
        return (0.9, [ControlIntent.GRASP, ControlIntent.MOVE_TO, ControlIntent.REACH])
    elif coherence >= COHERENCE_MED:
        # Medium coherence - moderate actions
        return (0.6, [ControlIntent.HOVER_AT, ControlIntent.PULL, ControlIntent.PUSH])
    elif coherence >= COHERENCE_LOW:
        # Low coherence - releasing actions
        return (0.3, [ControlIntent.RELEASE, ControlIntent.HOVER_AT])
    else:
        return (0.0, [ControlIntent.NONE])


def compute_hrv_component(bio: BiometricState) -> Tuple[float, List[ControlIntent]]:
    """
    Compute HRV contribution and suggested intents.

    HRV spike indicates activation/intention change.
    """
    if not bio.hrv_spike:
        return (0.3, [ControlIntent.HOVER_AT])  # Stable - maintain

    if bio.hrv_delta > 0:
        # HRV increasing - parasympathetic activation (calming)
        return (0.7, [ControlIntent.RELEASE, ControlIntent.PULL])
    else:
        # HRV decreasing - sympathetic activation (energizing)
        return (0.7, [ControlIntent.PUSH, ControlIntent.REACH, ControlIntent.GRASP])


def compute_breath_component(breath_rate: float) -> Tuple[float, List[ControlIntent]]:
    """
    Compute breath rate contribution and suggested intents.
    """
    pattern = classify_breath_pattern(breath_rate)

    if pattern == "fast":
        # Fast breathing - urgent/active actions
        return (0.6, [ControlIntent.PUSH, ControlIntent.RELEASE])
    elif pattern == "slow":
        # Slow breathing - receptive/acquiring actions
        return (0.6, [ControlIntent.PULL, ControlIntent.GRASP, ControlIntent.REACH])
    elif pattern == "normal":
        # Normal breathing - balanced actions
        return (0.4, [ControlIntent.HOVER_AT, ControlIntent.MOVE_TO])
    else:
        # Moderate - transitional
        return (0.3, [ControlIntent.HOVER_AT])


def combine_intent_signals(
    coherence_intents: List[ControlIntent],
    hrv_intents: List[ControlIntent],
    breath_intents: List[ControlIntent],
    coherence_weight: float,
    hrv_weight: float,
    breath_weight: float
) -> Tuple[ControlIntent, float]:
    """
    Combine intent signals from all components.

    Returns (best_intent, confidence)
    """
    # Find common intents
    all_intents = set(coherence_intents) | set(hrv_intents) | set(breath_intents)

    if not all_intents or all_intents == {ControlIntent.NONE}:
        return (ControlIntent.NONE, 0.0)

    # Score each intent
    intent_scores = {}
    for intent in all_intents:
        if intent == ControlIntent.NONE:
            continue
        score = 0.0
        count = 0
        if intent in coherence_intents:
            score += coherence_weight
            count += 1
        if intent in hrv_intents:
            score += hrv_weight
            count += 1
        if intent in breath_intents:
            score += breath_weight
            count += 1
        # Boost for multi-signal agreement
        if count > 1:
            score *= 1.0 + (count - 1) * 0.2
        intent_scores[intent] = score

    if not intent_scores:
        return (ControlIntent.NONE, 0.0)

    # Get best intent
    best_intent = max(intent_scores, key=intent_scores.get)
    max_score = intent_scores[best_intent]

    # Normalize to confidence (0-1)
    # Max possible: 3 components * 0.9 * 1.4 (boost) ≈ 3.78
    confidence = min(1.0, max_score / 2.5)

    return (best_intent, confidence)


def detect_intent(bio: BiometricState) -> IntentResult:
    """
    Detect control intent from biometric state.

    Combines coherence, HRV, and breath signals into intent.
    """
    # Compute components
    coh_weight, coh_intents = compute_coherence_component(bio.coherence)
    hrv_weight, hrv_intents = compute_hrv_component(bio)
    breath_weight, breath_intents = compute_breath_component(bio.breath_rate)

    # Combine signals
    intent, confidence = combine_intent_signals(
        coh_intents, hrv_intents, breath_intents,
        coh_weight, hrv_weight, breath_weight
    )

    return IntentResult(
        intent=intent,
        confidence=confidence,
        coherence_component=coh_weight,
        hrv_component=hrv_weight,
        breath_component=breath_weight,
        is_valid=confidence >= INTENT_CONFIDENCE_MIN
    )


def apply_intent_smoothing(
    current: IntentResult,
    history: IntentHistory
) -> IntentResult:
    """
    Apply temporal smoothing to prevent jitter.

    Returns smoothed intent based on history.
    """
    if not current.is_valid:
        # Try to use history for weak signals
        dominant = history.get_dominant_intent()
        if dominant and dominant != ControlIntent.NONE:
            # Reduce confidence for historical fallback
            return IntentResult(
                intent=dominant,
                confidence=current.confidence * 0.7,
                coherence_component=current.coherence_component,
                hrv_component=current.hrv_component,
                breath_component=current.breath_component,
                is_valid=False
            )
    return current


def intent_to_vector(intent: ControlIntent) -> Tuple[float, float, float]:
    """
    Convert intent to normalized direction vector.

    Returns (x, y, z) direction for interface navigation.
    """
    vectors = {
        ControlIntent.NONE: (0.0, 0.0, 0.0),
        ControlIntent.REACH: (1.0, 0.0, 0.0),      # Forward
        ControlIntent.PULL: (-1.0, 0.0, 0.0),     # Backward
        ControlIntent.PUSH: (0.0, 1.0, 0.0),      # Outward
        ControlIntent.GRASP: (0.0, 0.0, 1.0),     # Inward/up
        ControlIntent.RELEASE: (0.0, 0.0, -1.0),  # Outward/down
        ControlIntent.MOVE_TO: (0.5, 0.5, 0.0),   # Diagonal forward
        ControlIntent.HOVER_AT: (0.0, 0.0, 0.0),  # Stationary
    }
    return vectors.get(intent, (0.0, 0.0, 0.0))


def compute_intent_magnitude(result: IntentResult) -> float:
    """
    Compute action magnitude based on confidence.

    Higher confidence = stronger action.
    """
    if not result.is_valid:
        return 0.0
    # Scale by φ for natural feel
    return result.confidence * PHI / 2.0


# =============================================================================
# Test Cases
# =============================================================================

class TestBiometricState:
    """Tests for BiometricState data structure."""

    def test_hrv_delta_calculation(self):
        """Test HRV delta is correctly calculated."""
        bio = BiometricState(
            coherence=0.7,
            hrv_current=60.0,
            hrv_previous=50.0,
            breath_rate=12.0,
            timestamp=0.0
        )
        assert bio.hrv_delta == pytest.approx(0.2)

    def test_hrv_spike_detection_positive(self):
        """Test HRV spike detected for large positive change."""
        bio = BiometricState(
            coherence=0.7,
            hrv_current=60.0,
            hrv_previous=50.0,
            breath_rate=12.0,
            timestamp=0.0
        )
        assert bio.hrv_spike is True

    def test_hrv_spike_detection_negative(self):
        """Test HRV spike detected for large negative change."""
        bio = BiometricState(
            coherence=0.7,
            hrv_current=40.0,
            hrv_previous=50.0,
            breath_rate=12.0,
            timestamp=0.0
        )
        assert bio.hrv_spike is True

    def test_hrv_no_spike_small_change(self):
        """Test no HRV spike for small change."""
        bio = BiometricState(
            coherence=0.7,
            hrv_current=51.0,
            hrv_previous=50.0,
            breath_rate=12.0,
            timestamp=0.0
        )
        assert bio.hrv_spike is False


class TestCoherenceClassification:
    """Tests for coherence classification."""

    def test_high_coherence(self):
        """Test high coherence classification."""
        assert classify_coherence_level(0.80) == "high"
        assert classify_coherence_level(0.72) == "high"

    def test_medium_coherence(self):
        """Test medium coherence classification."""
        assert classify_coherence_level(0.60) == "medium"
        assert classify_coherence_level(0.50) == "medium"

    def test_low_coherence(self):
        """Test low coherence classification."""
        assert classify_coherence_level(0.40) == "low"
        assert classify_coherence_level(0.30) == "low"

    def test_none_coherence(self):
        """Test no coherence classification."""
        assert classify_coherence_level(0.20) == "none"
        assert classify_coherence_level(0.0) == "none"


class TestBreathClassification:
    """Tests for breath pattern classification."""

    def test_fast_breathing(self):
        """Test fast breathing classification."""
        assert classify_breath_pattern(20.0) == "fast"
        assert classify_breath_pattern(19.0) == "fast"

    def test_slow_breathing(self):
        """Test slow breathing classification."""
        assert classify_breath_pattern(6.0) == "slow"
        assert classify_breath_pattern(7.0) == "slow"

    def test_normal_breathing(self):
        """Test normal breathing classification."""
        assert classify_breath_pattern(12.0) == "normal"
        assert classify_breath_pattern(14.0) == "normal"


class TestCoherenceComponent:
    """Tests for coherence component computation."""

    def test_high_coherence_precise_intents(self):
        """High coherence should suggest precise actions."""
        weight, intents = compute_coherence_component(0.80)
        assert weight == pytest.approx(0.9)
        assert ControlIntent.GRASP in intents
        assert ControlIntent.REACH in intents

    def test_low_coherence_release_intents(self):
        """Low coherence should suggest releasing actions."""
        weight, intents = compute_coherence_component(0.35)
        assert weight == pytest.approx(0.3)
        assert ControlIntent.RELEASE in intents


class TestHRVComponent:
    """Tests for HRV component computation."""

    def test_hrv_spike_positive(self):
        """Positive HRV spike suggests calming actions."""
        bio = BiometricState(0.7, 60.0, 50.0, 12.0, 0.0)
        weight, intents = compute_hrv_component(bio)
        assert weight == pytest.approx(0.7)
        assert ControlIntent.RELEASE in intents or ControlIntent.PULL in intents

    def test_hrv_spike_negative(self):
        """Negative HRV spike suggests energizing actions."""
        bio = BiometricState(0.7, 40.0, 50.0, 12.0, 0.0)
        weight, intents = compute_hrv_component(bio)
        assert weight == pytest.approx(0.7)
        assert ControlIntent.PUSH in intents or ControlIntent.GRASP in intents

    def test_hrv_stable(self):
        """Stable HRV suggests maintaining actions."""
        bio = BiometricState(0.7, 50.5, 50.0, 12.0, 0.0)
        weight, intents = compute_hrv_component(bio)
        assert ControlIntent.HOVER_AT in intents


class TestBreathComponent:
    """Tests for breath component computation."""

    def test_fast_breath_urgent_intents(self):
        """Fast breathing suggests urgent actions."""
        weight, intents = compute_breath_component(20.0)
        assert ControlIntent.PUSH in intents or ControlIntent.RELEASE in intents

    def test_slow_breath_receptive_intents(self):
        """Slow breathing suggests receptive actions."""
        weight, intents = compute_breath_component(6.0)
        assert ControlIntent.PULL in intents or ControlIntent.GRASP in intents


class TestIntentCombination:
    """Tests for intent signal combination."""

    def test_common_intent_boosted(self):
        """Intent appearing in multiple signals should be boosted."""
        intent, confidence = combine_intent_signals(
            [ControlIntent.GRASP, ControlIntent.REACH],
            [ControlIntent.GRASP, ControlIntent.PUSH],
            [ControlIntent.GRASP, ControlIntent.PULL],
            0.9, 0.7, 0.6
        )
        # GRASP appears in all three
        assert intent == ControlIntent.GRASP
        assert confidence > 0.6

    def test_no_common_intent_picks_strongest(self):
        """With no common intent, pick highest weighted."""
        intent, confidence = combine_intent_signals(
            [ControlIntent.GRASP],
            [ControlIntent.PUSH],
            [ControlIntent.PULL],
            0.9, 0.3, 0.3
        )
        assert intent == ControlIntent.GRASP

    def test_all_none_returns_none(self):
        """All NONE intents should return NONE."""
        intent, confidence = combine_intent_signals(
            [ControlIntent.NONE],
            [ControlIntent.NONE],
            [ControlIntent.NONE],
            0.0, 0.0, 0.0
        )
        assert intent == ControlIntent.NONE


class TestIntentDetection:
    """Tests for full intent detection."""

    def test_high_coherence_slow_breath_grasp(self):
        """High coherence + slow breath should suggest GRASP."""
        bio = BiometricState(
            coherence=0.85,
            hrv_current=50.0,
            hrv_previous=49.0,  # Stable HRV
            breath_rate=7.0,    # Slow breath
            timestamp=0.0
        )
        result = detect_intent(bio)
        assert result.intent in [ControlIntent.GRASP, ControlIntent.PULL, ControlIntent.REACH]
        assert result.coherence_component > 0.8

    def test_low_coherence_fast_breath_release(self):
        """Low coherence + fast breath should suggest RELEASE."""
        bio = BiometricState(
            coherence=0.35,
            hrv_current=50.0,
            hrv_previous=49.0,
            breath_rate=20.0,   # Fast breath
            timestamp=0.0
        )
        result = detect_intent(bio)
        assert result.intent == ControlIntent.RELEASE

    def test_medium_coherence_stable_hover(self):
        """Medium coherence + stable state should suggest HOVER."""
        bio = BiometricState(
            coherence=0.55,
            hrv_current=50.0,
            hrv_previous=49.5,  # Stable HRV
            breath_rate=12.0,   # Normal breath
            timestamp=0.0
        )
        result = detect_intent(bio)
        assert result.intent in [ControlIntent.HOVER_AT, ControlIntent.MOVE_TO]

    def test_valid_result_meets_threshold(self):
        """Valid result should meet confidence threshold."""
        bio = BiometricState(
            coherence=0.85,
            hrv_current=60.0,
            hrv_previous=50.0,  # HRV spike
            breath_rate=7.0,    # Slow breath
            timestamp=0.0
        )
        result = detect_intent(bio)
        if result.is_valid:
            assert result.confidence >= INTENT_CONFIDENCE_MIN


class TestIntentHistory:
    """Tests for intent history management."""

    def test_history_tracks_intents(self):
        """History should track added intents."""
        history = IntentHistory(intents=[])
        result = IntentResult(
            ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True
        )
        history.add(result)
        assert len(history.intents) == 1

    def test_history_limits_size(self):
        """History should limit to max_history."""
        history = IntentHistory(intents=[], max_history=3)
        for i in range(5):
            history.add(IntentResult(
                ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True
            ))
        assert len(history.intents) == 3

    def test_dominant_intent_detection(self):
        """Should detect most common valid intent."""
        history = IntentHistory(intents=[
            IntentResult(ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True),
            IntentResult(ControlIntent.GRASP, 0.7, 0.8, 0.6, 0.5, True),
            IntentResult(ControlIntent.PUSH, 0.6, 0.7, 0.5, 0.4, True),
        ])
        assert history.get_dominant_intent() == ControlIntent.GRASP


class TestIntentSmoothing:
    """Tests for intent smoothing."""

    def test_valid_intent_unchanged(self):
        """Valid intent should pass through unchanged."""
        current = IntentResult(
            ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True
        )
        history = IntentHistory(intents=[])
        smoothed = apply_intent_smoothing(current, history)
        assert smoothed.intent == ControlIntent.GRASP
        assert smoothed.confidence == 0.8

    def test_invalid_uses_history(self):
        """Invalid intent should use history if available."""
        current = IntentResult(
            ControlIntent.NONE, 0.3, 0.3, 0.2, 0.2, False
        )
        history = IntentHistory(intents=[
            IntentResult(ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True),
            IntentResult(ControlIntent.GRASP, 0.7, 0.8, 0.6, 0.5, True),
        ])
        smoothed = apply_intent_smoothing(current, history)
        assert smoothed.intent == ControlIntent.GRASP
        assert smoothed.confidence < current.confidence


class TestIntentVector:
    """Tests for intent to vector conversion."""

    def test_reach_forward(self):
        """REACH should map to forward."""
        vec = intent_to_vector(ControlIntent.REACH)
        assert vec[0] > 0  # Forward

    def test_pull_backward(self):
        """PULL should map to backward."""
        vec = intent_to_vector(ControlIntent.PULL)
        assert vec[0] < 0  # Backward

    def test_hover_stationary(self):
        """HOVER_AT should map to stationary."""
        vec = intent_to_vector(ControlIntent.HOVER_AT)
        assert vec == (0.0, 0.0, 0.0)

    def test_grasp_upward(self):
        """GRASP should map to upward."""
        vec = intent_to_vector(ControlIntent.GRASP)
        assert vec[2] > 0  # Upward


class TestIntentMagnitude:
    """Tests for intent magnitude computation."""

    def test_valid_intent_has_magnitude(self):
        """Valid intent should have positive magnitude."""
        result = IntentResult(
            ControlIntent.GRASP, 0.8, 0.9, 0.7, 0.6, True
        )
        mag = compute_intent_magnitude(result)
        assert mag > 0

    def test_invalid_intent_zero_magnitude(self):
        """Invalid intent should have zero magnitude."""
        result = IntentResult(
            ControlIntent.GRASP, 0.3, 0.3, 0.2, 0.2, False
        )
        mag = compute_intent_magnitude(result)
        assert mag == 0.0

    def test_magnitude_scales_with_confidence(self):
        """Magnitude should scale with confidence."""
        result_low = IntentResult(
            ControlIntent.GRASP, 0.65, 0.9, 0.7, 0.6, True
        )
        result_high = IntentResult(
            ControlIntent.GRASP, 0.95, 0.9, 0.7, 0.6, True
        )
        mag_low = compute_intent_magnitude(result_low)
        mag_high = compute_intent_magnitude(result_high)
        assert mag_high > mag_low


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_intent_pipeline(self):
        """Test complete intent detection pipeline."""
        bio = BiometricState(
            coherence=0.80,
            hrv_current=55.0,
            hrv_previous=50.0,  # Slight spike
            breath_rate=10.0,
            timestamp=0.0
        )

        # Detect intent
        result = detect_intent(bio)
        assert result.intent != ControlIntent.NONE

        # Convert to vector
        vec = intent_to_vector(result.intent)
        assert isinstance(vec, tuple)
        assert len(vec) == 3

        # Compute magnitude
        if result.is_valid:
            mag = compute_intent_magnitude(result)
            assert mag > 0

    def test_intent_sequence_with_smoothing(self):
        """Test intent sequence with temporal smoothing."""
        history = IntentHistory(intents=[])

        # Build up history with strong signals (HRV spike + high coherence + slow breath)
        for i in range(5):
            bio = BiometricState(
                coherence=0.85,
                hrv_current=60.0 + i,   # Large increase from previous
                hrv_previous=50.0 + i,  # Creates HRV spike (>15% change)
                breath_rate=7.0,        # Slow breathing
                timestamp=float(i)
            )
            result = detect_intent(bio)
            history.add(result)

        # Check history has entries
        assert len(history.intents) == 5

        # Check at least some intents are valid
        valid_count = sum(1 for r in history.intents if r.is_valid)
        assert valid_count > 0

        # Check dominant intent exists
        dominant = history.get_dominant_intent()
        assert dominant is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
