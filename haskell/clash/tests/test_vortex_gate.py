"""
Prompt 38: Ra.Consent.VortexGate - Scalar Vortex Gate as Consent-Based Portal

Dynamic scalar-torsion portal that activates only when user meets
biometric coherence threshold and consent signature alignment.

Codex References:
- SCALAR_TORSION_ANTIGRAVITY.md: Torsion mechanics
- Ra.Consent.Gates: Permission protocols
- Ra.Identity: Scalar identity hashes
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import hashlib


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Default thresholds
DEFAULT_COHERENCE_THRESHOLD = 0.72
CONSENT_SIMILARITY_THRESHOLD = 0.93


# ============================================================================
# Types
# ============================================================================

class SpinDirection(Enum):
    """Torsion spin direction."""
    CLOCKWISE = auto()       # Positive phase
    COUNTER_CLOCKWISE = auto()  # Negative phase


class GateState(Enum):
    """Gate activation state."""
    LOCKED = auto()      # Conditions not met
    SPINNING = auto()    # Evaluating
    TRAVERSABLE = auto()  # Open for passage


class GatePolarity(Enum):
    """Gate polarity configuration."""
    STANDARD = auto()    # CW=Positive, CCW=Negative
    INVERTED = auto()    # CW=Negative, CCW=Positive


@dataclass
class PhaseVector:
    """Scalar phase vector for alignment checking."""
    magnitude: float
    angle: float  # radians
    spin: SpinDirection

    def to_cartesian(self) -> Tuple[float, float]:
        """Convert to 2D Cartesian for dot product."""
        return (
            self.magnitude * math.cos(self.angle),
            self.magnitude * math.sin(self.angle)
        )


@dataclass
class ConsentHash:
    """Consent signature hash."""
    hash_bytes: bytes
    timestamp: float
    identity_id: str

    @classmethod
    def from_string(cls, identity: str, timestamp: float = 0.0) -> 'ConsentHash':
        """Create hash from identity string."""
        hash_bytes = hashlib.sha256(identity.encode()).digest()
        return cls(hash_bytes, timestamp, identity)


@dataclass
class TorsionFieldSnapshot:
    """Dynamic torsion field visual state."""
    spin_velocity: float     # radians/second
    field_radius: float
    collapse_factor: float   # 0=open, 1=collapsed
    glow_intensity: float


@dataclass
class VortexGate:
    """Scalar-torsion portal with consent gating."""
    gate_id: str
    torsion_spin: SpinDirection
    required_coherence: float
    consent_signature: ConsentHash
    current_state: GateState
    field_alignment: PhaseVector
    gate_aura_field: TorsionFieldSnapshot
    polarity_config: GatePolarity = GatePolarity.STANDARD


@dataclass
class UserState:
    """User state for gate passage evaluation."""
    current_coherence: float
    consent_hash: ConsentHash
    spin_polarity: SpinDirection
    scalar_phase_vec: PhaseVector


# ============================================================================
# Consent Hash Matching
# ============================================================================

def cosine_similarity(vec1: Tuple[float, ...], vec2: Tuple[float, ...]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def hash_to_vector(hash_bytes: bytes, dimensions: int = 8) -> Tuple[float, ...]:
    """Convert hash bytes to normalized vector for similarity comparison."""
    # Take first N bytes and normalize
    values = [b / 255.0 for b in hash_bytes[:dimensions]]
    while len(values) < dimensions:
        values.append(0.0)
    return tuple(values)


def match_consent(hash1: ConsentHash, hash2: ConsentHash) -> float:
    """
    Calculate consent hash similarity using cosine similarity.

    Returns similarity score 0.0-1.0.
    """
    vec1 = hash_to_vector(hash1.hash_bytes)
    vec2 = hash_to_vector(hash2.hash_bytes)
    return cosine_similarity(vec1, vec2)


# ============================================================================
# Phase Vector Operations
# ============================================================================

def dot_product_2d(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """Calculate 2D dot product."""
    return v1[0] * v2[0] + v1[1] * v2[1]


def is_opposed(vec1: PhaseVector, vec2: PhaseVector) -> bool:
    """
    Check if phase vectors are opposed.

    Opposition defined as dot product < 0 (angular difference > 90°).
    """
    cart1 = vec1.to_cartesian()
    cart2 = vec2.to_cartesian()
    return dot_product_2d(cart1, cart2) < 0


def get_spin_polarity(spin: SpinDirection,
                      config: GatePolarity = GatePolarity.STANDARD) -> int:
    """
    Get polarity value for spin direction.

    Standard: CW=+1, CCW=-1
    Inverted: CW=-1, CCW=+1
    """
    if config == GatePolarity.STANDARD:
        return 1 if spin == SpinDirection.CLOCKWISE else -1
    else:
        return -1 if spin == SpinDirection.CLOCKWISE else 1


# ============================================================================
# Gate Evaluation
# ============================================================================

def can_pass_through(gate: VortexGate, user: UserState) -> bool:
    """
    Check if user can pass through vortex gate.

    Requires:
    1. coherence >= required threshold
    2. consent hash similarity >= 0.93
    3. phase vectors not opposed
    """
    # Check coherence
    if user.current_coherence < gate.required_coherence:
        return False

    # Check consent match
    similarity = match_consent(gate.consent_signature, user.consent_hash)
    if similarity < CONSENT_SIMILARITY_THRESHOLD:
        return False

    # Check phase opposition
    if is_opposed(user.scalar_phase_vec, gate.field_alignment):
        return False

    return True


def evaluate_gate(gate: VortexGate, user: UserState) -> Tuple[GateState, str]:
    """
    Evaluate gate state and return status message.

    Returns (new_state, message).
    """
    # Check coherence
    if user.current_coherence < gate.required_coherence:
        return (GateState.LOCKED,
                f"Coherence too low: {user.current_coherence:.2f} < {gate.required_coherence:.2f}")

    # Check consent
    similarity = match_consent(gate.consent_signature, user.consent_hash)
    if similarity < CONSENT_SIMILARITY_THRESHOLD:
        return (GateState.LOCKED,
                f"Consent mismatch: {similarity*100:.0f}% - Access denied")

    # Check phase
    if is_opposed(user.scalar_phase_vec, gate.field_alignment):
        return (GateState.LOCKED,
                "Phase rejection - scalar vectors opposed")

    # All conditions met
    spin_name = "clockwise" if gate.torsion_spin == SpinDirection.CLOCKWISE else "counter-clockwise"
    return (GateState.TRAVERSABLE,
            f"Gate opening - Vortex alignment: {spin_name}, stable")


def update_gate_visuals(gate: VortexGate, user: UserState) -> TorsionFieldSnapshot:
    """
    Update gate torsion field visuals based on state.

    Denied: Fast spin, inward collapse
    Approved: Slow spin, expanded membrane
    """
    can_pass = can_pass_through(gate, user)

    if can_pass:
        return TorsionFieldSnapshot(
            spin_velocity=0.5,     # Slow, stable
            field_radius=2.0,      # Expanded
            collapse_factor=0.0,   # Open
            glow_intensity=0.8     # Bright
        )
    else:
        return TorsionFieldSnapshot(
            spin_velocity=5.0,     # Fast, unstable
            field_radius=0.5,      # Contracted
            collapse_factor=0.8,   # Nearly collapsed
            glow_intensity=0.3     # Dim
        )


# ============================================================================
# Factory Functions
# ============================================================================

def create_gate(gate_id: str,
                identity: str,
                coherence_threshold: float = DEFAULT_COHERENCE_THRESHOLD,
                spin: SpinDirection = SpinDirection.CLOCKWISE) -> VortexGate:
    """Create a new vortex gate."""
    return VortexGate(
        gate_id=gate_id,
        torsion_spin=spin,
        required_coherence=coherence_threshold,
        consent_signature=ConsentHash.from_string(identity),
        current_state=GateState.LOCKED,
        field_alignment=PhaseVector(1.0, 0.0, spin),
        gate_aura_field=TorsionFieldSnapshot(1.0, 1.0, 0.5, 0.5)
    )


def create_user(identity: str,
                coherence: float,
                phase_angle: float = 0.0) -> UserState:
    """Create a user state for gate evaluation."""
    return UserState(
        current_coherence=coherence,
        consent_hash=ConsentHash.from_string(identity),
        spin_polarity=SpinDirection.CLOCKWISE,
        scalar_phase_vec=PhaseVector(1.0, phase_angle, SpinDirection.CLOCKWISE)
    )


# ============================================================================
# Test Suite
# ============================================================================

class TestConsentHashing:
    """Test consent hash creation and matching."""

    def test_same_identity_matches(self):
        """Same identity produces matching hashes."""
        hash1 = ConsentHash.from_string("user-123")
        hash2 = ConsentHash.from_string("user-123")
        similarity = match_consent(hash1, hash2)
        assert similarity == 1.0

    def test_different_identity_low_match(self):
        """Different identities have low similarity."""
        hash1 = ConsentHash.from_string("user-123")
        hash2 = ConsentHash.from_string("user-456")
        similarity = match_consent(hash1, hash2)
        assert similarity < CONSENT_SIMILARITY_THRESHOLD

    def test_similar_identity_partial_match(self):
        """Similar identities have partial match."""
        hash1 = ConsentHash.from_string("user-123-alpha")
        hash2 = ConsentHash.from_string("user-123-beta")
        similarity = match_consent(hash1, hash2)
        # Should have some similarity but not exact
        assert 0.0 < similarity < 1.0


class TestPhaseVectors:
    """Test phase vector operations."""

    def test_parallel_not_opposed(self):
        """Parallel vectors are not opposed."""
        v1 = PhaseVector(1.0, 0.0, SpinDirection.CLOCKWISE)
        v2 = PhaseVector(1.0, 0.0, SpinDirection.CLOCKWISE)
        assert is_opposed(v1, v2) is False

    def test_opposite_opposed(self):
        """Opposite vectors (180°) are opposed."""
        v1 = PhaseVector(1.0, 0.0, SpinDirection.CLOCKWISE)
        v2 = PhaseVector(1.0, math.pi, SpinDirection.CLOCKWISE)
        assert is_opposed(v1, v2) is True

    def test_perpendicular_not_opposed(self):
        """Perpendicular vectors (90°) are not opposed."""
        v1 = PhaseVector(1.0, 0.0, SpinDirection.CLOCKWISE)
        v2 = PhaseVector(1.0, math.pi/2, SpinDirection.CLOCKWISE)
        assert is_opposed(v1, v2) is False

    def test_obtuse_angle_opposed(self):
        """Obtuse angle (>90°) vectors are opposed."""
        v1 = PhaseVector(1.0, 0.0, SpinDirection.CLOCKWISE)
        v2 = PhaseVector(1.0, 2.5, SpinDirection.CLOCKWISE)  # ~143°
        assert is_opposed(v1, v2) is True


class TestSpinPolarity:
    """Test spin direction polarity mapping."""

    def test_standard_clockwise_positive(self):
        """Standard config: CW = positive."""
        polarity = get_spin_polarity(SpinDirection.CLOCKWISE, GatePolarity.STANDARD)
        assert polarity == 1

    def test_standard_ccw_negative(self):
        """Standard config: CCW = negative."""
        polarity = get_spin_polarity(SpinDirection.COUNTER_CLOCKWISE, GatePolarity.STANDARD)
        assert polarity == -1

    def test_inverted_clockwise_negative(self):
        """Inverted config: CW = negative."""
        polarity = get_spin_polarity(SpinDirection.CLOCKWISE, GatePolarity.INVERTED)
        assert polarity == -1


class TestGatePassage:
    """Test gate passage evaluation."""

    def test_all_conditions_met(self):
        """User passes when all conditions met."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.85)
        assert can_pass_through(gate, user) is True

    def test_coherence_too_low(self):
        """User blocked with low coherence."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.50)
        assert can_pass_through(gate, user) is False

    def test_consent_mismatch(self):
        """User blocked with consent mismatch."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-beta", 0.85)
        assert can_pass_through(gate, user) is False

    def test_phase_opposed(self):
        """User blocked with opposed phase."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.85, phase_angle=math.pi)
        assert can_pass_through(gate, user) is False


class TestGateEvaluation:
    """Test gate evaluation with messages."""

    def test_approved_message(self):
        """Approved gate shows opening message."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.85)
        state, msg = evaluate_gate(gate, user)
        assert state == GateState.TRAVERSABLE
        assert "opening" in msg.lower()

    def test_coherence_denied_message(self):
        """Low coherence shows specific message."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.50)
        state, msg = evaluate_gate(gate, user)
        assert state == GateState.LOCKED
        assert "coherence" in msg.lower()

    def test_consent_denied_message(self):
        """Consent mismatch shows percentage."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-beta", 0.85)
        state, msg = evaluate_gate(gate, user)
        assert state == GateState.LOCKED
        assert "%" in msg


class TestGateVisuals:
    """Test torsion field visual updates."""

    def test_approved_visuals(self):
        """Approved gate has expanded, slow visuals."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-alpha", 0.85)
        visuals = update_gate_visuals(gate, user)
        assert visuals.spin_velocity < 1.0
        assert visuals.field_radius > 1.0
        assert visuals.collapse_factor < 0.5

    def test_denied_visuals(self):
        """Denied gate has collapsed, fast visuals."""
        gate = create_gate("gate-1", "user-alpha", 0.72)
        user = create_user("user-beta", 0.50)
        visuals = update_gate_visuals(gate, user)
        assert visuals.spin_velocity > 1.0
        assert visuals.field_radius < 1.0
        assert visuals.collapse_factor > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
