"""
Test Harness for Prompt 58: Ra.Shell.LimbLearning

Tests adaptive gesture personalization engine that learns gesture patterns
per user and allows gesture → intent override mappings.

Based on:
- Hybrid matching: DTW + angle cosine + coherence (weights: 0.5, 0.3, 0.2)
- Default tolerance: 0.15
- Update strategy: Append + blend (incremental model blending)
- Conditional overrides: Always, CoherenceAbove, PhaseAligned
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Tuple
from enum import Enum, auto
import uuid

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Matching weights
DTW_WEIGHT = 0.5
COSINE_WEIGHT = 0.3
COHERENCE_WEIGHT = 0.2

# Default tolerance for pattern deviation
DEFAULT_TOLERANCE = 0.15

# Learning rate for template blending
DEFAULT_LEARNING_RATE = 0.3

# Match confidence threshold
MATCH_CONFIDENCE_THRESHOLD = 0.80

# Max users and gestures
MAX_USERS = 256
MAX_GESTURES_PER_USER = 9


# =============================================================================
# Data Structures
# =============================================================================

class GestureType(Enum):
    """Gesture types matching P57."""
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


class ControlIntent(Enum):
    """Control intents matching P56/P57."""
    NONE = auto()
    REACH = auto()
    PULL = auto()
    PUSH = auto()
    GRASP = auto()
    RELEASE = auto()
    MOVE_TO = auto()
    HOVER_AT = auto()
    # Custom intents for override
    OPEN_GATE = auto()
    CLOSE_GATE = auto()
    ACTIVATE = auto()
    DEACTIVATE = auto()


class OverrideConditionType(Enum):
    """Types of override conditions."""
    ALWAYS = auto()
    COHERENCE_ABOVE = auto()
    PHASE_ALIGNED = auto()
    CUSTOM = auto()


@dataclass
class ScalarVectorTrack:
    """A single point in a gesture track."""
    x: float
    y: float
    z: float
    timestamp: float
    coherence: float = 0.5  # Local coherence at this point


@dataclass
class OverrideCondition:
    """Condition for intent override."""
    condition_type: OverrideConditionType
    threshold: Optional[float] = None  # For COHERENCE_ABOVE
    predicate: Optional[Callable] = None  # For CUSTOM


@dataclass
class GestureModel:
    """Gesture model customized per user."""
    gesture_template: List[ScalarVectorTrack]
    tolerance: float = DEFAULT_TOLERANCE
    override_intent: Optional[ControlIntent] = None
    override_condition: Optional[OverrideCondition] = None
    sample_count: int = 1  # Number of samples used to build template


@dataclass
class GestureEvent:
    """Result of gesture matching."""
    gesture: GestureType
    confidence: float
    intent: ControlIntent
    user_id: str
    is_override: bool = False


@dataclass
class BiometricState:
    """Current biometric state for conditional checks."""
    coherence: float
    phase_aligned: bool
    torsion_state: str = "Normal"  # Normal, Inverted, Null


@dataclass
class UserGestureLibrary:
    """Gesture library for a single user."""
    user_id: str
    models: Dict[GestureType, GestureModel] = field(default_factory=dict)


# =============================================================================
# Core Functions
# =============================================================================

def compute_dtw_distance(track1: List[ScalarVectorTrack],
                         track2: List[ScalarVectorTrack]) -> float:
    """
    Compute Dynamic Time Warping distance between two tracks.

    Returns normalized distance (0 = identical, 1 = very different).
    """
    n, m = len(track1), len(track2)
    if n == 0 or m == 0:
        return 1.0

    # Initialize DTW matrix
    dtw = [[float('inf')] * (m + 1) for _ in range(n + 1)]
    dtw[0][0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Euclidean distance between points
            p1, p2 = track1[i-1], track2[j-1]
            dist = math.sqrt(
                (p1.x - p2.x)**2 +
                (p1.y - p2.y)**2 +
                (p1.z - p2.z)**2
            )
            dtw[i][j] = dist + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    # Normalize by path length
    path_length = n + m
    normalized = dtw[n][m] / path_length if path_length > 0 else 0.0

    # Convert to similarity (0 = different, 1 = identical)
    return min(1.0, normalized)


def compute_dtw_similarity(track1: List[ScalarVectorTrack],
                           track2: List[ScalarVectorTrack]) -> float:
    """Compute DTW similarity (1 = identical, 0 = very different)."""
    distance = compute_dtw_distance(track1, track2)
    return max(0.0, 1.0 - distance)


def compute_angle_cosine_similarity(track1: List[ScalarVectorTrack],
                                    track2: List[ScalarVectorTrack]) -> float:
    """
    Compute angle cosine similarity between track directions.

    Compares overall direction vectors of the two tracks.
    """
    def get_direction_vector(track: List[ScalarVectorTrack]) -> Tuple[float, float, float]:
        if len(track) < 2:
            return (0.0, 0.0, 0.0)
        dx = track[-1].x - track[0].x
        dy = track[-1].y - track[0].y
        dz = track[-1].z - track[0].z
        mag = math.sqrt(dx**2 + dy**2 + dz**2)
        if mag < 0.001:
            return (0.0, 0.0, 0.0)
        return (dx/mag, dy/mag, dz/mag)

    v1 = get_direction_vector(track1)
    v2 = get_direction_vector(track2)

    # Dot product = cosine similarity for unit vectors
    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

    # Convert from [-1, 1] to [0, 1]
    return (dot + 1.0) / 2.0


def compute_coherence_score(track: List[ScalarVectorTrack]) -> float:
    """Compute average coherence of a track."""
    if not track:
        return 0.0
    return sum(p.coherence for p in track) / len(track)


def compute_match_score(template: List[ScalarVectorTrack],
                        candidate: List[ScalarVectorTrack]) -> float:
    """
    Compute hybrid match score using DTW + angle cosine + coherence.

    Score formula: w1*dtw + w2*cosine + w3*coherence
    """
    dtw_sim = compute_dtw_similarity(template, candidate)
    cosine_sim = compute_angle_cosine_similarity(template, candidate)
    coherence = compute_coherence_score(candidate)

    score = (
        DTW_WEIGHT * dtw_sim +
        COSINE_WEIGHT * cosine_sim +
        COHERENCE_WEIGHT * coherence
    )

    return min(1.0, score)


def blend_templates(old_template: List[ScalarVectorTrack],
                    new_sample: List[ScalarVectorTrack],
                    learning_rate: float = DEFAULT_LEARNING_RATE) -> List[ScalarVectorTrack]:
    """
    Blend old template with new sample using incremental learning.

    new_template = (1 - lr) * old + lr * new
    """
    if not old_template:
        return new_sample.copy()
    if not new_sample:
        return old_template.copy()

    # Resample to same length (use shorter)
    min_len = min(len(old_template), len(new_sample))

    blended = []
    for i in range(min_len):
        old_p = old_template[i] if i < len(old_template) else old_template[-1]
        new_p = new_sample[i] if i < len(new_sample) else new_sample[-1]

        blended.append(ScalarVectorTrack(
            x=(1 - learning_rate) * old_p.x + learning_rate * new_p.x,
            y=(1 - learning_rate) * old_p.y + learning_rate * new_p.y,
            z=(1 - learning_rate) * old_p.z + learning_rate * new_p.z,
            timestamp=new_p.timestamp,
            coherence=(1 - learning_rate) * old_p.coherence + learning_rate * new_p.coherence
        ))

    return blended


def check_override_condition(condition: OverrideCondition,
                             bio_state: BiometricState) -> bool:
    """Check if override condition is met."""
    if condition.condition_type == OverrideConditionType.ALWAYS:
        return True
    elif condition.condition_type == OverrideConditionType.COHERENCE_ABOVE:
        return bio_state.coherence >= (condition.threshold or 0.6)
    elif condition.condition_type == OverrideConditionType.PHASE_ALIGNED:
        return bio_state.phase_aligned
    elif condition.condition_type == OverrideConditionType.CUSTOM:
        if condition.predicate:
            return condition.predicate(bio_state)
    return False


def get_default_intent(gesture: GestureType) -> ControlIntent:
    """Get default intent for a gesture type."""
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
    }
    return mapping.get(gesture, ControlIntent.NONE)


class GestureLearningEngine:
    """Main gesture learning and personalization engine."""

    def __init__(self):
        self.libraries: Dict[str, UserGestureLibrary] = {}

    def get_or_create_library(self, user_id: str) -> UserGestureLibrary:
        """Get or create gesture library for user."""
        if user_id not in self.libraries:
            if len(self.libraries) >= MAX_USERS:
                raise ValueError(f"Max users ({MAX_USERS}) exceeded")
            self.libraries[user_id] = UserGestureLibrary(user_id=user_id)
        return self.libraries[user_id]

    def learn_gesture(self, user_id: str, gesture: GestureType,
                      track: List[ScalarVectorTrack],
                      tolerance: Optional[float] = None) -> GestureModel:
        """
        Learn or update gesture pattern from user's motion track.

        Uses append + blend strategy for incremental model refinement.
        """
        library = self.get_or_create_library(user_id)

        if len(library.models) >= MAX_GESTURES_PER_USER and gesture not in library.models:
            raise ValueError(f"Max gestures per user ({MAX_GESTURES_PER_USER}) exceeded")

        if gesture in library.models:
            # Blend with existing model
            existing = library.models[gesture]
            new_template = blend_templates(existing.gesture_template, track)
            existing.gesture_template = new_template
            existing.sample_count += 1
            if tolerance is not None:
                existing.tolerance = tolerance
            return existing
        else:
            # Create new model
            model = GestureModel(
                gesture_template=track.copy(),
                tolerance=tolerance or DEFAULT_TOLERANCE,
                sample_count=1
            )
            library.models[gesture] = model
            return model

    def match_user_gesture(self, user_id: str,
                           track: List[ScalarVectorTrack],
                           bio_state: Optional[BiometricState] = None) -> Optional[GestureEvent]:
        """
        Match incoming track to user-known gestures.

        Returns GestureEvent if match confidence >= threshold.
        """
        if user_id not in self.libraries:
            return None

        library = self.libraries[user_id]
        if not library.models:
            return None

        best_gesture = None
        best_score = 0.0
        best_model = None

        for gesture, model in library.models.items():
            score = compute_match_score(model.gesture_template, track)

            # Apply tolerance: score must be above (1 - tolerance)
            if score >= (1.0 - model.tolerance) and score > best_score:
                best_score = score
                best_gesture = gesture
                best_model = model

        if best_gesture is None or best_score < MATCH_CONFIDENCE_THRESHOLD:
            return None

        # Determine intent (default or override)
        intent = get_default_intent(best_gesture)
        is_override = False

        if best_model.override_intent is not None:
            # Check if override condition is met
            if best_model.override_condition is None:
                # No condition = always override
                intent = best_model.override_intent
                is_override = True
            elif bio_state is not None:
                if check_override_condition(best_model.override_condition, bio_state):
                    intent = best_model.override_intent
                    is_override = True

        return GestureEvent(
            gesture=best_gesture,
            confidence=best_score,
            intent=intent,
            user_id=user_id,
            is_override=is_override
        )

    def override_gesture_intent(self, user_id: str, gesture: GestureType,
                                intent: ControlIntent,
                                condition: Optional[OverrideCondition] = None) -> bool:
        """
        Override default gesture → intent mapping for a user.

        Returns True if successful.
        """
        if user_id not in self.libraries:
            return False

        library = self.libraries[user_id]
        if gesture not in library.models:
            return False

        model = library.models[gesture]
        model.override_intent = intent
        model.override_condition = condition
        return True


# =============================================================================
# Test Helpers
# =============================================================================

def generate_linear_track(start: Tuple[float, float, float],
                          end: Tuple[float, float, float],
                          num_points: int = 10,
                          coherence: float = 0.7) -> List[ScalarVectorTrack]:
    """Generate a linear track for testing."""
    track = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        track.append(ScalarVectorTrack(
            x=start[0] + t * (end[0] - start[0]),
            y=start[1] + t * (end[1] - start[1]),
            z=start[2] + t * (end[2] - start[2]),
            timestamp=float(i),
            coherence=coherence
        ))
    return track


def generate_noisy_track(base_track: List[ScalarVectorTrack],
                         noise_level: float = 0.1) -> List[ScalarVectorTrack]:
    """Add noise to a track for testing variance."""
    import random
    noisy = []
    for p in base_track:
        noisy.append(ScalarVectorTrack(
            x=p.x + random.uniform(-noise_level, noise_level),
            y=p.y + random.uniform(-noise_level, noise_level),
            z=p.z + random.uniform(-noise_level, noise_level),
            timestamp=p.timestamp,
            coherence=p.coherence
        ))
    return noisy


# =============================================================================
# Test Cases
# =============================================================================

class TestDTWDistance:
    """Tests for DTW distance computation."""

    def test_identical_tracks(self):
        """Identical tracks should have similarity 1.0."""
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        similarity = compute_dtw_similarity(track, track)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_different_tracks(self):
        """Very different tracks should have low similarity."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (0, 0, 10), 10)  # Perpendicular
        similarity = compute_dtw_similarity(track1, track2)
        assert similarity < 0.5

    def test_similar_tracks(self):
        """Similar tracks should have high similarity."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (1.1, 0.1, 0), 10)  # Slightly different
        similarity = compute_dtw_similarity(track1, track2)
        assert similarity > 0.7


class TestAngleCosineSimilarity:
    """Tests for angle cosine similarity."""

    def test_same_direction(self):
        """Same direction should have similarity ~1.0."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (2, 0, 0), 10)  # Same direction, different length
        similarity = compute_angle_cosine_similarity(track1, track2)
        assert similarity > 0.95

    def test_opposite_direction(self):
        """Opposite direction should have similarity ~0.0."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (-1, 0, 0), 10)
        similarity = compute_angle_cosine_similarity(track1, track2)
        assert similarity < 0.1

    def test_perpendicular(self):
        """Perpendicular should have similarity ~0.5."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (0, 1, 0), 10)
        similarity = compute_angle_cosine_similarity(track1, track2)
        assert similarity == pytest.approx(0.5, abs=0.1)


class TestMatchScore:
    """Tests for hybrid match score."""

    def test_perfect_match(self):
        """Perfect match should score high."""
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.9)
        score = compute_match_score(track, track)
        assert score > 0.9

    def test_weights_applied(self):
        """Verify weights are applied correctly."""
        # Same track, different coherence
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=1.0)
        track2 = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.0)

        score1 = compute_match_score(track1, track1)
        score2 = compute_match_score(track1, track2)

        # Coherence weight is 0.2, so difference should be ~0.2
        assert score1 - score2 == pytest.approx(COHERENCE_WEIGHT, abs=0.05)


class TestBlendTemplates:
    """Tests for template blending."""

    def test_blend_same_length(self):
        """Blend two tracks of same length."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 5)
        track2 = generate_linear_track((0, 0, 0), (0, 1, 0), 5)

        blended = blend_templates(track1, track2, learning_rate=0.5)

        assert len(blended) == 5
        # End point should be average of (1,0,0) and (0,1,0)
        assert blended[-1].x == pytest.approx(0.5, abs=0.1)
        assert blended[-1].y == pytest.approx(0.5, abs=0.1)

    def test_blend_preserves_old(self):
        """Low learning rate should preserve old template."""
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 5)
        track2 = generate_linear_track((0, 0, 0), (0, 1, 0), 5)

        blended = blend_templates(track1, track2, learning_rate=0.1)

        # End should be closer to track1
        assert blended[-1].x > blended[-1].y


class TestOverrideCondition:
    """Tests for override conditions."""

    def test_always_condition(self):
        """ALWAYS should always pass."""
        condition = OverrideCondition(OverrideConditionType.ALWAYS)
        bio = BiometricState(coherence=0.1, phase_aligned=False)
        assert check_override_condition(condition, bio) is True

    def test_coherence_above_pass(self):
        """COHERENCE_ABOVE should pass when coherence is high enough."""
        condition = OverrideCondition(OverrideConditionType.COHERENCE_ABOVE, threshold=0.6)
        bio = BiometricState(coherence=0.8, phase_aligned=False)
        assert check_override_condition(condition, bio) is True

    def test_coherence_above_fail(self):
        """COHERENCE_ABOVE should fail when coherence is too low."""
        condition = OverrideCondition(OverrideConditionType.COHERENCE_ABOVE, threshold=0.6)
        bio = BiometricState(coherence=0.4, phase_aligned=False)
        assert check_override_condition(condition, bio) is False

    def test_phase_aligned(self):
        """PHASE_ALIGNED should check phase alignment."""
        condition = OverrideCondition(OverrideConditionType.PHASE_ALIGNED)
        bio_aligned = BiometricState(coherence=0.5, phase_aligned=True)
        bio_not_aligned = BiometricState(coherence=0.5, phase_aligned=False)

        assert check_override_condition(condition, bio_aligned) is True
        assert check_override_condition(condition, bio_not_aligned) is False


class TestGestureLearningEngine:
    """Tests for main gesture learning engine."""

    def test_create_library(self):
        """Engine should create library for new user."""
        engine = GestureLearningEngine()
        library = engine.get_or_create_library("user1")
        assert library.user_id == "user1"
        assert len(library.models) == 0

    def test_learn_new_gesture(self):
        """Should learn new gesture for user."""
        engine = GestureLearningEngine()
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10)

        model = engine.learn_gesture("user1", GestureType.REACH_FORWARD, track)

        assert model.sample_count == 1
        assert len(model.gesture_template) == 10

    def test_learn_incremental(self):
        """Learning same gesture should blend templates."""
        engine = GestureLearningEngine()
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        track2 = generate_linear_track((0, 0, 0), (1.2, 0.2, 0), 10)

        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track1)
        model = engine.learn_gesture("user1", GestureType.REACH_FORWARD, track2)

        assert model.sample_count == 2

    def test_match_gesture(self):
        """Should match learned gesture."""
        engine = GestureLearningEngine()
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.8)
        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track)

        # Match with similar track
        test_track = generate_linear_track((0, 0, 0), (1.05, 0.05, 0), 10, coherence=0.8)
        event = engine.match_user_gesture("user1", test_track)

        assert event is not None
        assert event.gesture == GestureType.REACH_FORWARD
        assert event.confidence >= MATCH_CONFIDENCE_THRESHOLD

    def test_match_no_library(self):
        """Should return None for unknown user."""
        engine = GestureLearningEngine()
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10)

        event = engine.match_user_gesture("unknown", track)
        assert event is None

    def test_match_below_threshold(self):
        """Should return None when confidence below threshold."""
        engine = GestureLearningEngine()
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track1)

        # Very different track
        track2 = generate_linear_track((0, 0, 0), (0, 0, 10), 10)
        event = engine.match_user_gesture("user1", track2)

        assert event is None

    def test_override_intent(self):
        """Should allow intent override."""
        engine = GestureLearningEngine()
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.8)
        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track)

        # Override REACH_FORWARD → OPEN_GATE
        success = engine.override_gesture_intent(
            "user1",
            GestureType.REACH_FORWARD,
            ControlIntent.OPEN_GATE
        )
        assert success is True

        # Match should return overridden intent
        event = engine.match_user_gesture("user1", track)
        assert event is not None
        assert event.intent == ControlIntent.OPEN_GATE
        assert event.is_override is True

    def test_conditional_override(self):
        """Override should respect conditions."""
        engine = GestureLearningEngine()
        track = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.8)
        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track)

        # Override with coherence condition
        condition = OverrideCondition(OverrideConditionType.COHERENCE_ABOVE, threshold=0.7)
        engine.override_gesture_intent(
            "user1",
            GestureType.REACH_FORWARD,
            ControlIntent.OPEN_GATE,
            condition
        )

        # With high coherence - should override
        bio_high = BiometricState(coherence=0.9, phase_aligned=False)
        event_high = engine.match_user_gesture("user1", track, bio_high)
        assert event_high.intent == ControlIntent.OPEN_GATE

        # With low coherence - should use default
        bio_low = BiometricState(coherence=0.5, phase_aligned=False)
        event_low = engine.match_user_gesture("user1", track, bio_low)
        assert event_low.intent == ControlIntent.REACH  # Default

    def test_max_users_limit(self):
        """Should enforce max users limit."""
        engine = GestureLearningEngine()

        # This would be slow with 256 users, so we'll mock the limit
        engine.libraries = {f"user{i}": UserGestureLibrary(f"user{i}")
                          for i in range(MAX_USERS)}

        with pytest.raises(ValueError):
            engine.get_or_create_library("new_user")

    def test_max_gestures_per_user(self):
        """Should enforce max gestures per user."""
        engine = GestureLearningEngine()

        # Learn max gestures
        for i, gesture in enumerate(list(GestureType)[:MAX_GESTURES_PER_USER]):
            track = generate_linear_track((0, 0, i), (1, 0, i), 10)
            engine.learn_gesture("user1", gesture, track)

        # Next gesture should fail
        with pytest.raises(ValueError):
            track = generate_linear_track((0, 0, 0), (1, 0, 0), 10)
            # Find a gesture not yet used
            unused = [g for g in GestureType if g not in engine.libraries["user1"].models][0]
            engine.learn_gesture("user1", unused, track)


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_personalization_workflow(self):
        """Test complete personalization workflow."""
        engine = GestureLearningEngine()
        user_id = "user_abc"

        # 1. User performs gesture 3 times to train
        for i in range(3):
            track = generate_linear_track(
                (0, 0, 0),
                (1 + i*0.1, i*0.05, 0),  # Slight variation each time
                12,
                coherence=0.75
            )
            engine.learn_gesture(user_id, GestureType.REACH_FORWARD, track)

        model = engine.libraries[user_id].models[GestureType.REACH_FORWARD]
        assert model.sample_count == 3

        # 2. User sets custom override
        engine.override_gesture_intent(
            user_id,
            GestureType.REACH_FORWARD,
            ControlIntent.ACTIVATE,
            OverrideCondition(OverrideConditionType.COHERENCE_ABOVE, threshold=0.6)
        )

        # 3. Recognition works with override
        test_track = generate_linear_track((0, 0, 0), (1.1, 0.08, 0), 12, coherence=0.8)
        bio = BiometricState(coherence=0.85, phase_aligned=True)

        event = engine.match_user_gesture(user_id, test_track, bio)

        assert event is not None
        assert event.gesture == GestureType.REACH_FORWARD
        assert event.intent == ControlIntent.ACTIVATE
        assert event.is_override is True

    def test_multiple_users(self):
        """Test multiple users with different gestures."""
        engine = GestureLearningEngine()

        # User 1 learns REACH
        track1 = generate_linear_track((0, 0, 0), (1, 0, 0), 10, coherence=0.8)
        engine.learn_gesture("user1", GestureType.REACH_FORWARD, track1)

        # User 2 learns PULL
        track2 = generate_linear_track((1, 0, 0), (0, 0, 0), 10, coherence=0.8)
        engine.learn_gesture("user2", GestureType.PULL_BACK, track2)

        # Each user should only match their learned gesture
        event1 = engine.match_user_gesture("user1", track1)
        event2 = engine.match_user_gesture("user2", track2)

        assert event1.gesture == GestureType.REACH_FORWARD
        assert event2.gesture == GestureType.PULL_BACK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
