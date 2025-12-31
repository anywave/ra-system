"""
Test Harness for Prompt 57: Ra.Shell.LimbGestures

Tests gesture recognition from limb motion data for avatar control.
Uses adaptive frame windows (8-21 frames) based on motion speed,
with 0.60 confidence threshold for gesture classification.

Based on:
- Adaptive frame window sizing
- Motion trajectory analysis
- Gesture-to-ControlIntent mapping
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Frame window bounds
MIN_FRAME_WINDOW = 8
MAX_FRAME_WINDOW = 21
DEFAULT_FRAME_WINDOW = 13  # ~φ * 8

# Confidence threshold for gesture recognition
CONFIDENCE_THRESHOLD = 0.60

# Motion speed thresholds (units per frame)
SPEED_SLOW = 0.05
SPEED_FAST = 0.30

# Minimum motion for gesture (vs noise)
MOTION_EPSILON = 0.01


# =============================================================================
# Data Structures
# =============================================================================

class GestureType(Enum):
    """Recognized gesture types."""
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
    """Control intents mapped from gestures."""
    NONE = auto()
    REACH = auto()
    PULL = auto()
    PUSH = auto()
    GRASP = auto()
    RELEASE = auto()
    MOVE_TO = auto()
    HOVER_AT = auto()


@dataclass(frozen=True)
class LimbPosition:
    """3D position of a limb endpoint."""
    x: float
    y: float
    z: float
    timestamp: float

    def distance_to(self, other: 'LimbPosition') -> float:
        """Euclidean distance to another position."""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


@dataclass
class MotionFrame:
    """Single frame of limb motion data."""
    position: LimbPosition
    velocity: Tuple[float, float, float]  # dx, dy, dz per frame
    frame_index: int


@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture: GestureType
    confidence: float
    frame_count: int
    motion_magnitude: float
    is_valid: bool


@dataclass
class MotionBuffer:
    """Buffer for motion frame collection."""
    frames: List[MotionFrame] = field(default_factory=list)
    target_window: int = DEFAULT_FRAME_WINDOW

    def add_frame(self, frame: MotionFrame):
        """Add frame to buffer."""
        self.frames.append(frame)
        # Trim to window size
        while len(self.frames) > self.target_window:
            self.frames.pop(0)

    def is_ready(self) -> bool:
        """Check if buffer has enough frames."""
        return len(self.frames) >= MIN_FRAME_WINDOW

    def get_speed(self) -> float:
        """Get average motion speed."""
        if len(self.frames) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(1, len(self.frames)):
            total_dist += self.frames[i].position.distance_to(
                self.frames[i-1].position
            )
        return total_dist / (len(self.frames) - 1)


# =============================================================================
# Core Functions
# =============================================================================

def compute_adaptive_window(speed: float) -> int:
    """
    Compute adaptive frame window based on motion speed.

    Slow motion = larger window (more frames needed)
    Fast motion = smaller window (fewer frames needed)
    """
    if speed <= SPEED_SLOW:
        return MAX_FRAME_WINDOW
    elif speed >= SPEED_FAST:
        return MIN_FRAME_WINDOW
    else:
        # Linear interpolation
        t = (speed - SPEED_SLOW) / (SPEED_FAST - SPEED_SLOW)
        window = MAX_FRAME_WINDOW - t * (MAX_FRAME_WINDOW - MIN_FRAME_WINDOW)
        return int(round(window))


def compute_motion_vector(frames: List[MotionFrame]) -> Tuple[float, float, float]:
    """
    Compute overall motion vector from frame sequence.

    Returns normalized (dx, dy, dz) representing primary motion direction.
    """
    if len(frames) < 2:
        return (0.0, 0.0, 0.0)

    # Sum velocities
    total_vx = sum(f.velocity[0] for f in frames)
    total_vy = sum(f.velocity[1] for f in frames)
    total_vz = sum(f.velocity[2] for f in frames)

    # Normalize
    mag = math.sqrt(total_vx**2 + total_vy**2 + total_vz**2)
    if mag < MOTION_EPSILON:
        return (0.0, 0.0, 0.0)

    return (total_vx / mag, total_vy / mag, total_vz / mag)


def compute_motion_magnitude(frames: List[MotionFrame]) -> float:
    """Compute total motion magnitude."""
    if len(frames) < 2:
        return 0.0
    return sum(
        frames[i].position.distance_to(frames[i-1].position)
        for i in range(1, len(frames))
    )


def detect_circular_motion(frames: List[MotionFrame]) -> Tuple[bool, bool]:
    """
    Detect circular motion pattern.

    Returns (is_circular, is_clockwise).
    """
    if len(frames) < 8:
        return (False, False)

    # Compute cross product of successive velocity pairs
    cross_sum = 0.0
    for i in range(1, len(frames) - 1):
        v1 = frames[i].velocity
        v2 = frames[i+1].velocity
        # 2D cross product (x-y plane)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        cross_sum += cross

    # Check if consistently curving
    avg_cross = cross_sum / (len(frames) - 2)
    is_circular = abs(avg_cross) > 0.01
    is_clockwise = avg_cross < 0

    return (is_circular, is_clockwise)


def detect_grasp_motion(frames: List[MotionFrame]) -> bool:
    """
    Detect closing/grasping motion.

    Characterized by convergent velocities toward center.
    """
    if len(frames) < 4:
        return False

    # Check if motion converges (magnitude decreases)
    first_half_mag = compute_motion_magnitude(frames[:len(frames)//2])
    second_half_mag = compute_motion_magnitude(frames[len(frames)//2:])

    # Grasp: motion slows down (closing in)
    return second_half_mag < first_half_mag * 0.5


def detect_release_motion(frames: List[MotionFrame]) -> bool:
    """
    Detect opening/release motion.

    Characterized by divergent velocities from center.
    """
    if len(frames) < 4:
        return False

    # Check if motion diverges (magnitude increases)
    first_half_mag = compute_motion_magnitude(frames[:len(frames)//2])
    second_half_mag = compute_motion_magnitude(frames[len(frames)//2:])

    # Release: motion speeds up (opening out)
    return second_half_mag > first_half_mag * 1.5


def classify_gesture(frames: List[MotionFrame]) -> Tuple[GestureType, float]:
    """
    Classify gesture from motion frames.

    Returns (gesture_type, confidence).
    """
    if len(frames) < MIN_FRAME_WINDOW:
        return (GestureType.NONE, 0.0)

    motion_mag = compute_motion_magnitude(frames)
    if motion_mag < MOTION_EPSILON:
        return (GestureType.HOLD_STEADY, 0.8)

    motion_vec = compute_motion_vector(frames)
    vx, vy, vz = motion_vec

    # Check for circular motion
    is_circular, is_clockwise = detect_circular_motion(frames)
    if is_circular:
        gesture = GestureType.CIRCLE_CW if is_clockwise else GestureType.CIRCLE_CCW
        return (gesture, 0.75)

    # Check for grasp/release
    if detect_grasp_motion(frames):
        return (GestureType.GRASP_CLOSE, 0.80)
    if detect_release_motion(frames):
        return (GestureType.RELEASE_OPEN, 0.80)

    # Classify by primary direction
    abs_vx, abs_vy, abs_vz = abs(vx), abs(vy), abs(vz)

    # Determine dominant axis
    if abs_vz > abs_vx and abs_vz > abs_vy:
        # Z dominant (forward/back)
        if vz > 0:
            return (GestureType.REACH_FORWARD, 0.85)
        else:
            return (GestureType.PULL_BACK, 0.85)
    elif abs_vy > abs_vx:
        # Y dominant (up/down)
        if vy > 0:
            return (GestureType.SWIPE_UP, 0.80)
        else:
            return (GestureType.SWIPE_DOWN, 0.80)
    else:
        # X dominant (left/right)
        if vx > 0:
            return (GestureType.SWIPE_RIGHT, 0.80)
        else:
            return (GestureType.SWIPE_LEFT, 0.80)


def recognize_gesture(buffer: MotionBuffer) -> GestureResult:
    """
    Recognize gesture from motion buffer.

    Full recognition pipeline with adaptive windowing.
    """
    if not buffer.is_ready():
        return GestureResult(
            gesture=GestureType.NONE,
            confidence=0.0,
            frame_count=len(buffer.frames),
            motion_magnitude=0.0,
            is_valid=False
        )

    # Classify gesture
    gesture, confidence = classify_gesture(buffer.frames)

    # Compute motion magnitude
    motion_mag = compute_motion_magnitude(buffer.frames)

    return GestureResult(
        gesture=gesture,
        confidence=confidence,
        frame_count=len(buffer.frames),
        motion_magnitude=motion_mag,
        is_valid=confidence >= CONFIDENCE_THRESHOLD
    )


def gesture_to_intent(gesture: GestureType) -> ControlIntent:
    """Map gesture type to control intent."""
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


def create_motion_frame(
    position: LimbPosition,
    prev_position: Optional[LimbPosition] = None,
    frame_index: int = 0
) -> MotionFrame:
    """Create motion frame with computed velocity."""
    if prev_position is None:
        velocity = (0.0, 0.0, 0.0)
    else:
        dt = position.timestamp - prev_position.timestamp
        if dt <= 0:
            dt = 1.0
        velocity = (
            (position.x - prev_position.x) / dt,
            (position.y - prev_position.y) / dt,
            (position.z - prev_position.z) / dt
        )
    return MotionFrame(position=position, velocity=velocity, frame_index=frame_index)


def generate_trajectory(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    num_frames: int
) -> List[MotionFrame]:
    """Generate linear trajectory for testing."""
    frames = []
    prev_pos = None

    for i in range(num_frames):
        t = i / (num_frames - 1) if num_frames > 1 else 0
        pos = LimbPosition(
            x=start[0] + t * (end[0] - start[0]),
            y=start[1] + t * (end[1] - start[1]),
            z=start[2] + t * (end[2] - start[2]),
            timestamp=float(i)
        )
        frame = create_motion_frame(pos, prev_pos, i)
        frames.append(frame)
        prev_pos = pos

    return frames


def generate_circular_trajectory(
    center: Tuple[float, float, float],
    radius: float,
    num_frames: int,
    clockwise: bool = True
) -> List[MotionFrame]:
    """Generate circular trajectory for testing."""
    frames = []
    prev_pos = None

    for i in range(num_frames):
        angle = 2 * math.pi * i / num_frames
        if clockwise:
            angle = -angle

        pos = LimbPosition(
            x=center[0] + radius * math.cos(angle),
            y=center[1] + radius * math.sin(angle),
            z=center[2],
            timestamp=float(i)
        )
        frame = create_motion_frame(pos, prev_pos, i)
        frames.append(frame)
        prev_pos = pos

    return frames


# =============================================================================
# Test Cases
# =============================================================================

class TestLimbPosition:
    """Tests for LimbPosition data structure."""

    def test_distance_to_same_point(self):
        """Distance to same point is zero."""
        pos = LimbPosition(1.0, 2.0, 3.0, 0.0)
        assert pos.distance_to(pos) == 0.0

    def test_distance_calculation(self):
        """Distance calculation is correct."""
        pos1 = LimbPosition(0.0, 0.0, 0.0, 0.0)
        pos2 = LimbPosition(3.0, 4.0, 0.0, 1.0)
        assert pos1.distance_to(pos2) == pytest.approx(5.0)


class TestMotionBuffer:
    """Tests for MotionBuffer."""

    def test_buffer_starts_empty(self):
        """Buffer should start empty."""
        buffer = MotionBuffer()
        assert len(buffer.frames) == 0
        assert not buffer.is_ready()

    def test_buffer_ready_at_min_frames(self):
        """Buffer ready when reaching min frames."""
        buffer = MotionBuffer()
        for i in range(MIN_FRAME_WINDOW):
            pos = LimbPosition(float(i), 0.0, 0.0, float(i))
            frame = create_motion_frame(pos, None, i)
            buffer.add_frame(frame)
        assert buffer.is_ready()

    def test_buffer_trims_to_window(self):
        """Buffer should trim to target window."""
        buffer = MotionBuffer(frames=[], target_window=10)
        for i in range(15):
            pos = LimbPosition(float(i), 0.0, 0.0, float(i))
            frame = create_motion_frame(pos, None, i)
            buffer.add_frame(frame)
        assert len(buffer.frames) == 10


class TestAdaptiveWindow:
    """Tests for adaptive window sizing."""

    def test_slow_motion_large_window(self):
        """Slow motion should use large window."""
        window = compute_adaptive_window(0.02)
        assert window == MAX_FRAME_WINDOW

    def test_fast_motion_small_window(self):
        """Fast motion should use small window."""
        window = compute_adaptive_window(0.5)
        assert window == MIN_FRAME_WINDOW

    def test_medium_motion_interpolated(self):
        """Medium motion should interpolate window."""
        window = compute_adaptive_window(0.15)
        assert MIN_FRAME_WINDOW < window < MAX_FRAME_WINDOW

    def test_window_bounds(self):
        """Window should stay in bounds."""
        for speed in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
            window = compute_adaptive_window(speed)
            assert MIN_FRAME_WINDOW <= window <= MAX_FRAME_WINDOW


class TestMotionVector:
    """Tests for motion vector computation."""

    def test_forward_motion(self):
        """Forward motion should have positive z."""
        frames = generate_trajectory(
            (0, 0, 0), (0, 0, 1), 10
        )
        vec = compute_motion_vector(frames)
        assert vec[2] > 0.9  # Dominant z

    def test_right_motion(self):
        """Right motion should have positive x."""
        frames = generate_trajectory(
            (0, 0, 0), (1, 0, 0), 10
        )
        vec = compute_motion_vector(frames)
        assert vec[0] > 0.9  # Dominant x

    def test_no_motion(self):
        """No motion should return zero vector."""
        pos = LimbPosition(0, 0, 0, 0)
        frames = [create_motion_frame(pos, pos, i) for i in range(10)]
        vec = compute_motion_vector(frames)
        assert vec == (0.0, 0.0, 0.0)


class TestCircularMotion:
    """Tests for circular motion detection."""

    def test_clockwise_circle(self):
        """Should detect clockwise circular motion."""
        frames = generate_circular_trajectory(
            (0, 0, 0), 1.0, 16, clockwise=True
        )
        is_circular, is_cw = detect_circular_motion(frames)
        assert is_circular is True
        assert is_cw is True

    def test_counter_clockwise_circle(self):
        """Should detect counter-clockwise circular motion."""
        frames = generate_circular_trajectory(
            (0, 0, 0), 1.0, 16, clockwise=False
        )
        is_circular, is_cw = detect_circular_motion(frames)
        assert is_circular is True
        assert is_cw is False

    def test_linear_not_circular(self):
        """Linear motion should not be circular."""
        frames = generate_trajectory(
            (0, 0, 0), (1, 0, 0), 10
        )
        is_circular, _ = detect_circular_motion(frames)
        assert is_circular is False


class TestGraspRelease:
    """Tests for grasp/release detection."""

    def test_grasp_motion_slowing(self):
        """Grasp motion slows down toward end."""
        # Create decelerating motion
        frames = []
        prev_pos = None
        for i in range(12):
            # Speed decreases exponentially
            t = i / 11
            decay = math.exp(-2 * t)
            pos = LimbPosition(decay * 0.5, 0, 0, float(i))
            frames.append(create_motion_frame(pos, prev_pos, i))
            prev_pos = pos

        is_grasp = detect_grasp_motion(frames)
        # This is a decelerating motion - should detect as grasp-like
        assert is_grasp is True

    def test_release_motion_accelerating(self):
        """Release motion speeds up toward end."""
        # Create accelerating motion
        frames = []
        prev_pos = None
        for i in range(12):
            t = i / 11
            accel = t ** 2  # Accelerating
            pos = LimbPosition(accel * 0.5, 0, 0, float(i))
            frames.append(create_motion_frame(pos, prev_pos, i))
            prev_pos = pos

        is_release = detect_release_motion(frames)
        assert is_release is True


class TestGestureClassification:
    """Tests for gesture classification."""

    def test_reach_forward_gesture(self):
        """Forward motion should classify as REACH_FORWARD."""
        frames = generate_trajectory(
            (0, 0, 0), (0, 0, 1), 12
        )
        gesture, confidence = classify_gesture(frames)
        assert gesture == GestureType.REACH_FORWARD
        assert confidence >= CONFIDENCE_THRESHOLD

    def test_pull_back_gesture(self):
        """Backward motion should classify as PULL_BACK."""
        frames = generate_trajectory(
            (0, 0, 1), (0, 0, 0), 12
        )
        gesture, confidence = classify_gesture(frames)
        assert gesture == GestureType.PULL_BACK
        assert confidence >= CONFIDENCE_THRESHOLD

    def test_swipe_right_gesture(self):
        """Right motion should classify as SWIPE_RIGHT."""
        frames = generate_trajectory(
            (0, 0, 0), (1, 0, 0), 12
        )
        gesture, confidence = classify_gesture(frames)
        assert gesture == GestureType.SWIPE_RIGHT
        assert confidence >= CONFIDENCE_THRESHOLD

    def test_swipe_up_gesture(self):
        """Upward motion should classify as SWIPE_UP."""
        frames = generate_trajectory(
            (0, 0, 0), (0, 1, 0), 12
        )
        gesture, confidence = classify_gesture(frames)
        assert gesture == GestureType.SWIPE_UP
        assert confidence >= CONFIDENCE_THRESHOLD

    def test_hold_steady_gesture(self):
        """No motion should classify as HOLD_STEADY."""
        pos = LimbPosition(0, 0, 0, 0)
        frames = []
        for i in range(12):
            # Tiny random noise
            p = LimbPosition(0.0001 * (i % 2), 0, 0, float(i))
            frames.append(create_motion_frame(p, pos, i))
        gesture, confidence = classify_gesture(frames)
        assert gesture == GestureType.HOLD_STEADY


class TestGestureRecognition:
    """Tests for full gesture recognition."""

    def test_recognition_needs_min_frames(self):
        """Recognition needs minimum frames."""
        buffer = MotionBuffer()
        for i in range(5):  # Less than MIN_FRAME_WINDOW
            pos = LimbPosition(float(i), 0, 0, float(i))
            buffer.add_frame(create_motion_frame(pos, None, i))

        result = recognize_gesture(buffer)
        assert not result.is_valid
        assert result.gesture == GestureType.NONE

    def test_recognition_returns_valid_result(self):
        """Valid frames should return valid result."""
        buffer = MotionBuffer()
        frames = generate_trajectory((0, 0, 0), (0, 0, 1), 12)
        for frame in frames:
            buffer.add_frame(frame)

        result = recognize_gesture(buffer)
        assert result.is_valid
        assert result.confidence >= CONFIDENCE_THRESHOLD


class TestGestureToIntent:
    """Tests for gesture to intent mapping."""

    def test_reach_forward_to_reach(self):
        """REACH_FORWARD maps to REACH."""
        intent = gesture_to_intent(GestureType.REACH_FORWARD)
        assert intent == ControlIntent.REACH

    def test_pull_back_to_pull(self):
        """PULL_BACK maps to PULL."""
        intent = gesture_to_intent(GestureType.PULL_BACK)
        assert intent == ControlIntent.PULL

    def test_grasp_close_to_grasp(self):
        """GRASP_CLOSE maps to GRASP."""
        intent = gesture_to_intent(GestureType.GRASP_CLOSE)
        assert intent == ControlIntent.GRASP

    def test_release_open_to_release(self):
        """RELEASE_OPEN maps to RELEASE."""
        intent = gesture_to_intent(GestureType.RELEASE_OPEN)
        assert intent == ControlIntent.RELEASE

    def test_hold_steady_to_hover(self):
        """HOLD_STEADY maps to HOVER_AT."""
        intent = gesture_to_intent(GestureType.HOLD_STEADY)
        assert intent == ControlIntent.HOVER_AT


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_gesture_pipeline(self):
        """Test complete gesture recognition pipeline."""
        # Generate reaching motion
        frames = generate_trajectory((0, 0, 0), (0, 0, 1), 15)

        # Add to buffer
        buffer = MotionBuffer()
        for frame in frames:
            buffer.add_frame(frame)

        # Recognize gesture
        result = recognize_gesture(buffer)
        assert result.is_valid
        assert result.gesture == GestureType.REACH_FORWARD

        # Map to intent
        intent = gesture_to_intent(result.gesture)
        assert intent == ControlIntent.REACH

    def test_adaptive_window_integration(self):
        """Test adaptive window with different speeds."""
        # Slow motion
        slow_frames = generate_trajectory((0, 0, 0), (0, 0, 0.2), 20)
        slow_buffer = MotionBuffer()
        for f in slow_frames:
            slow_buffer.add_frame(f)
        slow_speed = slow_buffer.get_speed()

        # Fast motion
        fast_frames = generate_trajectory((0, 0, 0), (0, 0, 2), 10)
        fast_buffer = MotionBuffer()
        for f in fast_frames:
            fast_buffer.add_frame(f)
        fast_speed = fast_buffer.get_speed()

        # Windows should differ
        slow_window = compute_adaptive_window(slow_speed)
        fast_window = compute_adaptive_window(fast_speed)
        assert slow_window > fast_window

    def test_gesture_sequence(self):
        """Test sequence of gestures."""
        intents_detected = []

        # Reach forward
        buffer = MotionBuffer()
        for f in generate_trajectory((0, 0, 0), (0, 0, 1), 12):
            buffer.add_frame(f)
        result = recognize_gesture(buffer)
        if result.is_valid:
            intents_detected.append(gesture_to_intent(result.gesture))

        # Pull back
        buffer = MotionBuffer()
        for f in generate_trajectory((0, 0, 1), (0, 0, 0), 12):
            buffer.add_frame(f)
        result = recognize_gesture(buffer)
        if result.is_valid:
            intents_detected.append(gesture_to_intent(result.gesture))

        assert len(intents_detected) == 2
        assert ControlIntent.REACH in intents_detected
        assert ControlIntent.PULL in intents_detected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
