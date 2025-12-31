"""
Test Harness for Prompt 59: Ra.Shell.ResonantGestures

Tests biometric-attuned gesture access using scalar resonance,
coherence gating, and field phase alignment.

Based on:
- Default resonance threshold: 0.65
- Always-allowed gestures: HoldStill, Point, EmergencyStop
- Failure logging: RejectionReason
- Inversion whitelist: CloseHand, Release
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Default resonance threshold
DEFAULT_RESONANCE_THRESHOLD = 0.65

# Match confidence from P58
MATCH_CONFIDENCE_THRESHOLD = 0.80


# =============================================================================
# Data Structures
# =============================================================================

class GestureType(Enum):
    """Gesture types."""
    NONE = auto()
    REACH_FORWARD = auto()
    PULL_BACK = auto()
    PUSH_OUT = auto()
    GRASP_CLOSE = auto()
    RELEASE_OPEN = auto()
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    CIRCLE_CW = auto()
    CIRCLE_CCW = auto()
    HOLD_STEADY = auto()
    POINT = auto()
    EMERGENCY_STOP = auto()
    CLOSE_HAND = auto()


class ControlIntent(Enum):
    """Control intents."""
    NONE = auto()
    REACH = auto()
    PULL = auto()
    PUSH = auto()
    GRASP = auto()
    RELEASE = auto()
    MOVE_TO = auto()
    HOVER_AT = auto()
    POINT_AT = auto()
    STOP = auto()


class TorsionState(Enum):
    """Torsion field state."""
    NORMAL = auto()
    INVERTED = auto()
    NULL = auto()


class RejectionReason(Enum):
    """Reasons for gesture rejection."""
    NONE = auto()
    LOW_COHERENCE = auto()
    TORSION_INVERSION_BLOCKED = auto()
    PHASE_MISALIGNED = auto()
    NO_MATCH_FOUND = auto()
    TEMPORAL_WINDOW_CLOSED = auto()


# Always-allowed gestures (exempt from resonance check)
ALWAYS_ALLOWED_GESTURES: Set[GestureType] = {
    GestureType.HOLD_STEADY,
    GestureType.POINT,
    GestureType.EMERGENCY_STOP,
}

# Inversion whitelist (allowed under inverted torsion)
INVERSION_WHITELIST: Set[GestureType] = {
    GestureType.CLOSE_HAND,
    GestureType.RELEASE_OPEN,
}


@dataclass
class ScalarVectorTrack:
    """A single point in a gesture track."""
    x: float
    y: float
    z: float
    timestamp: float
    coherence: float = 0.5


@dataclass
class FluxCoherence:
    """Flux coherence measurement."""
    value: float  # 0-1 coherence level
    stability: float = 0.5  # How stable the coherence is
    phase: float = 0.0  # Current phase (0 to 2π)


@dataclass
class TemporalWindow:
    """φ^n temporal window state."""
    is_active: bool
    window_index: int = 0
    phase_alignment: float = 0.0  # 0-1


@dataclass
class ResonantGesture:
    """Result of resonant gesture authorization."""
    user_id: str
    gesture: GestureType
    gesture_data: List[ScalarVectorTrack]
    resonance: FluxCoherence
    torsion_phase: TorsionState
    matched: bool  # Matched user-defined pattern
    authorized: bool  # Passed all tests
    rejection_reason: Optional[RejectionReason] = None
    intent: Optional[ControlIntent] = None


@dataclass
class GestureResonanceGate:
    """Minimum resonance required per gesture per user."""
    user_id: str
    gesture: GestureType
    min_coherence: float


@dataclass
class UserBiometricState:
    """Current biometric state for a user."""
    user_id: str
    resonance: FluxCoherence
    torsion_state: TorsionState
    temporal_window: TemporalWindow


# =============================================================================
# Core Functions
# =============================================================================

def is_always_allowed(gesture: GestureType) -> bool:
    """Check if gesture is always allowed without resonance check."""
    return gesture in ALWAYS_ALLOWED_GESTURES


def is_inversion_allowed(gesture: GestureType) -> bool:
    """Check if gesture is allowed under inverted torsion."""
    return gesture in INVERSION_WHITELIST


def get_default_resonance_threshold(gesture: GestureType) -> float:
    """Get default resonance threshold for a gesture."""
    if is_always_allowed(gesture):
        return 0.0  # No threshold for always-allowed
    return DEFAULT_RESONANCE_THRESHOLD


def get_gesture_intent(gesture: GestureType) -> ControlIntent:
    """Map gesture to control intent."""
    mapping = {
        GestureType.NONE: ControlIntent.NONE,
        GestureType.REACH_FORWARD: ControlIntent.REACH,
        GestureType.PULL_BACK: ControlIntent.PULL,
        GestureType.PUSH_OUT: ControlIntent.PUSH,
        GestureType.GRASP_CLOSE: ControlIntent.GRASP,
        GestureType.RELEASE_OPEN: ControlIntent.RELEASE,
        GestureType.SWIPE_LEFT: ControlIntent.MOVE_TO,
        GestureType.SWIPE_RIGHT: ControlIntent.MOVE_TO,
        GestureType.SWIPE_UP: ControlIntent.PUSH,
        GestureType.SWIPE_DOWN: ControlIntent.PULL,
        GestureType.CIRCLE_CW: ControlIntent.HOVER_AT,
        GestureType.CIRCLE_CCW: ControlIntent.HOVER_AT,
        GestureType.HOLD_STEADY: ControlIntent.HOVER_AT,
        GestureType.POINT: ControlIntent.POINT_AT,
        GestureType.EMERGENCY_STOP: ControlIntent.STOP,
        GestureType.CLOSE_HAND: ControlIntent.GRASP,
    }
    return mapping.get(gesture, ControlIntent.NONE)


class ResonantGestureAuthorizer:
    """
    Authorizes gestures based on biometric resonance,
    coherence gating, and field phase alignment.
    """

    def __init__(self):
        # Custom resonance gates per (user_id, gesture)
        self.resonance_gates: Dict[tuple, float] = {}
        # Track user biometric states
        self.user_states: Dict[str, UserBiometricState] = {}

    def set_gesture_resonance_gate(self, user_id: str, gesture: GestureType,
                                   min_coherence: float) -> None:
        """Set minimum coherence required for a gesture for a user."""
        self.resonance_gates[(user_id, gesture)] = min_coherence

    def get_resonance_gate(self, user_id: str, gesture: GestureType) -> float:
        """Get resonance gate for user/gesture, or default."""
        key = (user_id, gesture)
        if key in self.resonance_gates:
            return self.resonance_gates[key]
        return get_default_resonance_threshold(gesture)

    def update_user_state(self, state: UserBiometricState) -> None:
        """Update biometric state for a user."""
        self.user_states[state.user_id] = state

    def get_user_state(self, user_id: str) -> Optional[UserBiometricState]:
        """Get current biometric state for a user."""
        return self.user_states.get(user_id)

    def check_coherence(self, user_id: str, gesture: GestureType,
                        resonance: FluxCoherence) -> tuple:
        """
        Check if coherence meets threshold.

        Returns (passes, rejection_reason).
        """
        # Always-allowed gestures skip coherence check
        if is_always_allowed(gesture):
            return (True, None)

        threshold = self.get_resonance_gate(user_id, gesture)
        if resonance.value >= threshold:
            return (True, None)
        else:
            return (False, RejectionReason.LOW_COHERENCE)

    def check_torsion(self, gesture: GestureType,
                      torsion_state: TorsionState) -> tuple:
        """
        Check if torsion state allows gesture.

        Returns (passes, rejection_reason).
        """
        if torsion_state == TorsionState.INVERTED:
            if not is_inversion_allowed(gesture):
                return (False, RejectionReason.TORSION_INVERSION_BLOCKED)
        return (True, None)

    def check_temporal_window(self, window: TemporalWindow) -> tuple:
        """
        Check if temporal window is active.

        Returns (passes, rejection_reason).
        """
        if not window.is_active:
            return (False, RejectionReason.TEMPORAL_WINDOW_CLOSED)
        return (True, None)

    def check_phase_alignment(self, resonance: FluxCoherence,
                              window: TemporalWindow) -> tuple:
        """
        Check if phase is aligned.

        Returns (passes, rejection_reason).
        """
        # Phase alignment check: window alignment should be > 0.3
        if window.phase_alignment < 0.3:
            return (False, RejectionReason.PHASE_MISALIGNED)
        return (True, None)

    def authorize_resonant_gesture(
        self,
        user_id: str,
        gesture: GestureType,
        gesture_data: List[ScalarVectorTrack],
        matched: bool,  # From LimbLearning match
        resonance: Optional[FluxCoherence] = None,
        torsion_state: Optional[TorsionState] = None,
        temporal_window: Optional[TemporalWindow] = None
    ) -> ResonantGesture:
        """
        Authorize a gesture based on biometric validity and field alignment.

        Checks:
        1. Gesture match (from LimbLearning)
        2. Resonance/coherence threshold
        3. Torsion state compatibility
        4. Temporal window activity
        5. Phase alignment
        """
        # Get user state if not provided
        user_state = self.get_user_state(user_id)

        if resonance is None:
            resonance = user_state.resonance if user_state else FluxCoherence(0.0)
        if torsion_state is None:
            torsion_state = user_state.torsion_state if user_state else TorsionState.NORMAL
        if temporal_window is None:
            temporal_window = user_state.temporal_window if user_state else TemporalWindow(True)

        # Start with rejection reason as None
        rejection_reason = None
        authorized = False

        # Check 1: Was gesture matched?
        if not matched:
            rejection_reason = RejectionReason.NO_MATCH_FOUND
        else:
            # Check 2: Coherence threshold
            coh_pass, coh_reject = self.check_coherence(user_id, gesture, resonance)
            if not coh_pass:
                rejection_reason = coh_reject
            else:
                # Check 3: Torsion state
                tor_pass, tor_reject = self.check_torsion(gesture, torsion_state)
                if not tor_pass:
                    rejection_reason = tor_reject
                else:
                    # Check 4: Temporal window
                    win_pass, win_reject = self.check_temporal_window(temporal_window)
                    if not win_pass:
                        rejection_reason = win_reject
                    else:
                        # Check 5: Phase alignment (only for non-always-allowed)
                        if not is_always_allowed(gesture):
                            phase_pass, phase_reject = self.check_phase_alignment(
                                resonance, temporal_window
                            )
                            if not phase_pass:
                                rejection_reason = phase_reject
                            else:
                                authorized = True
                        else:
                            authorized = True

        # Determine intent if authorized
        intent = get_gesture_intent(gesture) if authorized else None

        return ResonantGesture(
            user_id=user_id,
            gesture=gesture,
            gesture_data=gesture_data,
            resonance=resonance,
            torsion_phase=torsion_state,
            matched=matched,
            authorized=authorized,
            rejection_reason=rejection_reason,
            intent=intent
        )


# =============================================================================
# Test Helpers
# =============================================================================

def create_test_track(num_points: int = 10) -> List[ScalarVectorTrack]:
    """Create a simple test track."""
    return [
        ScalarVectorTrack(x=i*0.1, y=0, z=0, timestamp=float(i), coherence=0.7)
        for i in range(num_points)
    ]


# =============================================================================
# Test Cases
# =============================================================================

class TestAlwaysAllowedGestures:
    """Tests for always-allowed gesture check."""

    def test_hold_steady_always_allowed(self):
        """HOLD_STEADY should be always allowed."""
        assert is_always_allowed(GestureType.HOLD_STEADY) is True

    def test_point_always_allowed(self):
        """POINT should be always allowed."""
        assert is_always_allowed(GestureType.POINT) is True

    def test_emergency_stop_always_allowed(self):
        """EMERGENCY_STOP should be always allowed."""
        assert is_always_allowed(GestureType.EMERGENCY_STOP) is True

    def test_reach_not_always_allowed(self):
        """REACH_FORWARD should NOT be always allowed."""
        assert is_always_allowed(GestureType.REACH_FORWARD) is False


class TestInversionWhitelist:
    """Tests for inversion whitelist."""

    def test_close_hand_inversion_allowed(self):
        """CLOSE_HAND should be allowed under inversion."""
        assert is_inversion_allowed(GestureType.CLOSE_HAND) is True

    def test_release_inversion_allowed(self):
        """RELEASE_OPEN should be allowed under inversion."""
        assert is_inversion_allowed(GestureType.RELEASE_OPEN) is True

    def test_reach_inversion_not_allowed(self):
        """REACH_FORWARD should NOT be allowed under inversion."""
        assert is_inversion_allowed(GestureType.REACH_FORWARD) is False


class TestResonanceThreshold:
    """Tests for resonance threshold."""

    def test_default_threshold(self):
        """Default threshold should be 0.65."""
        threshold = get_default_resonance_threshold(GestureType.REACH_FORWARD)
        assert threshold == pytest.approx(DEFAULT_RESONANCE_THRESHOLD)

    def test_always_allowed_zero_threshold(self):
        """Always-allowed gestures should have 0 threshold."""
        threshold = get_default_resonance_threshold(GestureType.HOLD_STEADY)
        assert threshold == 0.0


class TestCoherenceCheck:
    """Tests for coherence checking."""

    def test_coherence_above_threshold_passes(self):
        """Coherence above threshold should pass."""
        authorizer = ResonantGestureAuthorizer()
        resonance = FluxCoherence(value=0.8)

        passes, reason = authorizer.check_coherence(
            "user1", GestureType.REACH_FORWARD, resonance
        )

        assert passes is True
        assert reason is None

    def test_coherence_below_threshold_fails(self):
        """Coherence below threshold should fail."""
        authorizer = ResonantGestureAuthorizer()
        resonance = FluxCoherence(value=0.4)

        passes, reason = authorizer.check_coherence(
            "user1", GestureType.REACH_FORWARD, resonance
        )

        assert passes is False
        assert reason == RejectionReason.LOW_COHERENCE

    def test_always_allowed_skips_coherence(self):
        """Always-allowed gestures should skip coherence check."""
        authorizer = ResonantGestureAuthorizer()
        resonance = FluxCoherence(value=0.1)  # Very low

        passes, reason = authorizer.check_coherence(
            "user1", GestureType.HOLD_STEADY, resonance
        )

        assert passes is True

    def test_custom_threshold(self):
        """Custom threshold should override default."""
        authorizer = ResonantGestureAuthorizer()
        authorizer.set_gesture_resonance_gate("user1", GestureType.REACH_FORWARD, 0.9)

        resonance = FluxCoherence(value=0.8)  # Above default but below custom

        passes, reason = authorizer.check_coherence(
            "user1", GestureType.REACH_FORWARD, resonance
        )

        assert passes is False


class TestTorsionCheck:
    """Tests for torsion state checking."""

    def test_normal_torsion_allows_all(self):
        """Normal torsion should allow all gestures."""
        authorizer = ResonantGestureAuthorizer()

        passes, reason = authorizer.check_torsion(
            GestureType.REACH_FORWARD, TorsionState.NORMAL
        )

        assert passes is True

    def test_inverted_blocks_non_whitelist(self):
        """Inverted torsion should block non-whitelisted gestures."""
        authorizer = ResonantGestureAuthorizer()

        passes, reason = authorizer.check_torsion(
            GestureType.REACH_FORWARD, TorsionState.INVERTED
        )

        assert passes is False
        assert reason == RejectionReason.TORSION_INVERSION_BLOCKED

    def test_inverted_allows_whitelist(self):
        """Inverted torsion should allow whitelisted gestures."""
        authorizer = ResonantGestureAuthorizer()

        passes, reason = authorizer.check_torsion(
            GestureType.CLOSE_HAND, TorsionState.INVERTED
        )

        assert passes is True


class TestTemporalWindowCheck:
    """Tests for temporal window checking."""

    def test_active_window_passes(self):
        """Active temporal window should pass."""
        authorizer = ResonantGestureAuthorizer()
        window = TemporalWindow(is_active=True)

        passes, reason = authorizer.check_temporal_window(window)

        assert passes is True

    def test_inactive_window_fails(self):
        """Inactive temporal window should fail."""
        authorizer = ResonantGestureAuthorizer()
        window = TemporalWindow(is_active=False)

        passes, reason = authorizer.check_temporal_window(window)

        assert passes is False
        assert reason == RejectionReason.TEMPORAL_WINDOW_CLOSED


class TestPhaseAlignmentCheck:
    """Tests for phase alignment checking."""

    def test_aligned_phase_passes(self):
        """Aligned phase should pass."""
        authorizer = ResonantGestureAuthorizer()
        resonance = FluxCoherence(value=0.8)
        window = TemporalWindow(is_active=True, phase_alignment=0.5)

        passes, reason = authorizer.check_phase_alignment(resonance, window)

        assert passes is True

    def test_misaligned_phase_fails(self):
        """Misaligned phase should fail."""
        authorizer = ResonantGestureAuthorizer()
        resonance = FluxCoherence(value=0.8)
        window = TemporalWindow(is_active=True, phase_alignment=0.1)

        passes, reason = authorizer.check_phase_alignment(resonance, window)

        assert passes is False
        assert reason == RejectionReason.PHASE_MISALIGNED


class TestAuthorizeResonantGesture:
    """Tests for full gesture authorization."""

    def test_authorize_success(self):
        """Should authorize when all checks pass."""
        authorizer = ResonantGestureAuthorizer()
        track = create_test_track()

        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.REACH_FORWARD,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )

        assert result.authorized is True
        assert result.rejection_reason is None
        assert result.intent == ControlIntent.REACH

    def test_authorize_fail_no_match(self):
        """Should reject when gesture not matched."""
        authorizer = ResonantGestureAuthorizer()
        track = create_test_track()

        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.REACH_FORWARD,
            gesture_data=track,
            matched=False,
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )

        assert result.authorized is False
        assert result.rejection_reason == RejectionReason.NO_MATCH_FOUND

    def test_authorize_fail_low_coherence(self):
        """Should reject when coherence too low."""
        authorizer = ResonantGestureAuthorizer()
        track = create_test_track()

        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.REACH_FORWARD,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.3),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )

        assert result.authorized is False
        assert result.rejection_reason == RejectionReason.LOW_COHERENCE

    def test_authorize_fail_torsion_blocked(self):
        """Should reject when torsion blocks gesture."""
        authorizer = ResonantGestureAuthorizer()
        track = create_test_track()

        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.REACH_FORWARD,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.INVERTED,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )

        assert result.authorized is False
        assert result.rejection_reason == RejectionReason.TORSION_INVERSION_BLOCKED

    def test_always_allowed_bypasses_checks(self):
        """Always-allowed gestures should bypass coherence/phase checks."""
        authorizer = ResonantGestureAuthorizer()
        track = create_test_track()

        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.HOLD_STEADY,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.1),  # Very low
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.1)  # Misaligned
        )

        assert result.authorized is True
        assert result.intent == ControlIntent.HOVER_AT


class TestUserStateManagement:
    """Tests for user state management."""

    def test_update_and_get_state(self):
        """Should store and retrieve user state."""
        authorizer = ResonantGestureAuthorizer()

        state = UserBiometricState(
            user_id="user1",
            resonance=FluxCoherence(value=0.75),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.6)
        )
        authorizer.update_user_state(state)

        retrieved = authorizer.get_user_state("user1")

        assert retrieved is not None
        assert retrieved.resonance.value == 0.75

    def test_authorize_uses_stored_state(self):
        """Authorization should use stored user state if not provided."""
        authorizer = ResonantGestureAuthorizer()

        state = UserBiometricState(
            user_id="user1",
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )
        authorizer.update_user_state(state)

        track = create_test_track()
        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.REACH_FORWARD,
            gesture_data=track,
            matched=True
            # No explicit resonance/torsion/window - should use stored state
        )

        assert result.authorized is True


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_authorization_flow(self):
        """Test complete authorization flow with custom gates."""
        authorizer = ResonantGestureAuthorizer()

        # Set custom gate for user
        authorizer.set_gesture_resonance_gate("user1", GestureType.GRASP_CLOSE, 0.7)

        # Update user state
        state = UserBiometricState(
            user_id="user1",
            resonance=FluxCoherence(value=0.75),
            torsion_state=TorsionState.NORMAL,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.6)
        )
        authorizer.update_user_state(state)

        # Authorize gesture
        track = create_test_track()
        result = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.GRASP_CLOSE,
            gesture_data=track,
            matched=True
        )

        assert result.authorized is True
        assert result.intent == ControlIntent.GRASP

    def test_inversion_whitelist_flow(self):
        """Test gesture authorization under inverted torsion."""
        authorizer = ResonantGestureAuthorizer()

        track = create_test_track()

        # Whitelisted gesture should pass under inversion
        result1 = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.CLOSE_HAND,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.INVERTED,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )
        assert result1.authorized is True

        # Non-whitelisted should fail
        result2 = authorizer.authorize_resonant_gesture(
            user_id="user1",
            gesture=GestureType.PUSH_OUT,
            gesture_data=track,
            matched=True,
            resonance=FluxCoherence(value=0.8),
            torsion_state=TorsionState.INVERTED,
            temporal_window=TemporalWindow(is_active=True, phase_alignment=0.5)
        )
        assert result2.authorized is False
        assert result2.rejection_reason == RejectionReason.TORSION_INVERSION_BLOCKED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
