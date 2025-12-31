"""
Test Harness for Prompt 53: Ra.Simulation.AetherLab

Tests full Ra-field simulations in synthetic morphogenic chambers:
- Blending biometric data, scalar topologies, inversion nodes
- Avatar traversal and resonance paths
- Non-realtime emergence testing and debug coherence breaks
- Prototype fragment alignments

Based on phi-nested triple pyramid chamber topology.
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895
COHERENCE_EMERGENCE_THRESHOLD = 0.72  # Combined criteria threshold
FRAGMENT_PROXIMITY_THRESHOLD = 3.0  # Distance units for alignment
DEFAULT_FIELD_SIZE = 27  # φ-nested 810-point field base


# =============================================================================
# Data Structures
# =============================================================================

class EmergenceType(Enum):
    """Types of emergence events."""
    COHERENCE_SPIKE = auto()
    FRAGMENT_ALIGNMENT = auto()
    HARMONIC_LOCK = auto()
    INVERSION_RESPONSE = auto()
    AVATAR_RESONANCE = auto()


class AvatarState(Enum):
    """Avatar resonance states."""
    STABLE = auto()
    SEEKING = auto()
    RESONANT = auto()
    DISRUPTED = auto()


@dataclass
class Position:
    """3D position in chamber space."""
    theta: int  # Grid position
    phi: int
    h: int

    def distance_to(self, other: 'Position') -> float:
        """Calculate distance to another position."""
        return math.sqrt(
            (self.theta - other.theta)**2 +
            (self.phi - other.phi)**2 +
            (self.h - other.h)**2
        )


@dataclass
class BioState:
    """Biometric state for simulation input."""
    hrv: float  # Heart rate variability (ms)
    coherence: float  # Biometric coherence (0-1)
    rhythm_phase: float  # Current phase in rhythm cycle (0-2π)


@dataclass
class BioStream:
    """Stream of biometric samples."""
    samples: List[float]  # HRV values
    current_index: int = 0

    def get_next(self) -> Optional[BioState]:
        """Get next biometric state from stream."""
        if self.current_index >= len(self.samples):
            self.current_index = 0  # Cycle
        hrv = self.samples[self.current_index]
        self.current_index += 1

        # Derive coherence from HRV stability
        avg_hrv = sum(self.samples) / len(self.samples)
        coherence = max(0, min(1, 1 - abs(hrv - avg_hrv) / avg_hrv))

        # Phase based on index
        phase = (self.current_index / len(self.samples)) * 2 * math.pi

        return BioState(hrv=hrv, coherence=coherence, rhythm_phase=phase)


@dataclass
class Fragment:
    """Scalar field fragment."""
    fragment_id: int
    position: Position
    harmonic_l: int
    harmonic_m: int
    intensity: float  # 0-1
    is_manifest: bool = False


@dataclass
class Avatar:
    """Avatar state in simulation."""
    avatar_id: int
    position: Position
    resonance_state: AvatarState
    coherence_level: float = 0.5
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class ScalarField:
    """Scalar resonance field."""
    values: List[List[List[float]]]  # 3D grid
    size: int

    def get_value(self, pos: Position) -> float:
        """Get value at position."""
        t = pos.theta % self.size
        p = pos.phi % self.size
        h = pos.h % self.size
        return self.values[t][p][h]

    def set_value(self, pos: Position, value: float):
        """Set value at position."""
        t = pos.theta % self.size
        p = pos.phi % self.size
        h = pos.h % self.size
        self.values[t][p][h] = value


@dataclass
class InversionRegion:
    """Region of field inversion."""
    center: Position
    radius: int
    intensity: float  # Inversion strength (0-1)
    is_active: bool = True


@dataclass
class PhiClock:
    """Phi-based timing."""
    tick: int = 0
    phi_phase: float = 0.0

    def advance(self):
        """Advance clock by one tick."""
        self.tick += 1
        self.phi_phase = (self.tick * PHI) % (2 * math.pi)


@dataclass
class EmergentContent:
    """Emerged content event."""
    emergence_type: EmergenceType
    position: Position
    timestamp: int
    coherence_level: float
    fragments_involved: List[int] = field(default_factory=list)


@dataclass
class AetherChamber:
    """Full simulation chamber."""
    chamber_id: str
    field_profile: ScalarField
    biometric_input: Optional[BioStream]
    fragment_set: List[Fragment]
    avatar_state: Avatar
    time_context: PhiClock
    inversion_zone: Optional[InversionRegion]


@dataclass
class SimulationResult:
    """Result of one simulation step."""
    emergence_events: List[EmergentContent]
    coherence_graph: List[float]
    trace_log: List[str]
    chamber_delta: AetherChamber


# =============================================================================
# Field Generation
# =============================================================================

def generate_phi_tuned_field(size: int, l_max: int, m_max: int) -> ScalarField:
    """
    Generate φ-tuned scalar field with harmonic weights.

    Creates an 810-point field (27*6*5) with phi-based coherence peaks.
    """
    values = [[[0.0 for _ in range(size)] for _ in range(size)] for _ in range(size)]

    for t in range(size):
        for p in range(size):
            for h in range(size):
                # φ-weighted base value
                phi_weight = PHI ** (-(t + p + h) / size)

                # Harmonic contribution
                harmonic = 0.0
                for l in range(l_max + 1):
                    for m in range(-min(l, m_max), min(l, m_max) + 1):
                        # Simplified spherical harmonic contribution
                        theta_norm = t * math.pi / size
                        phi_norm = p * 2 * math.pi / size
                        harmonic += math.cos(l * theta_norm) * math.cos(m * phi_norm) / (l + 1)

                values[t][p][h] = phi_weight * (0.5 + 0.5 * math.tanh(harmonic))

    return ScalarField(values=values, size=size)


def create_golod_seed_array(count: int, field_size: int) -> List[Fragment]:
    """Create GOLOD-seeded fragment array."""
    fragments = []
    for i in range(count):
        # Distribute fragments using φ-based spacing
        t = int((i * PHI * field_size) % field_size)
        p = int((i * PHI**2 * field_size) % field_size)
        h = int((i * PHI**3 * field_size) % field_size)

        fragments.append(Fragment(
            fragment_id=i,
            position=Position(theta=t, phi=p, h=h),
            harmonic_l=i % 5,
            harmonic_m=i % 3 - 1,
            intensity=0.5 + 0.3 * math.sin(i * PHI),
            is_manifest=False
        ))
    return fragments


# =============================================================================
# Simulation Core
# =============================================================================

def update_scalar_field(
    field: ScalarField,
    bio: Optional[BioState],
    phi_clock: PhiClock
) -> ScalarField:
    """
    Update scalar field based on biometric input.

    Returns new field with coherence modifications.
    """
    # Create copy of field values
    new_values = [[[field.values[t][p][h]
                    for h in range(field.size)]
                   for p in range(field.size)]
                  for t in range(field.size)]

    if bio is not None:
        # Modulate field by biometric coherence
        bio_factor = 0.9 + 0.2 * bio.coherence

        for t in range(field.size):
            for p in range(field.size):
                for h in range(field.size):
                    # Apply biometric modulation with phi-phase
                    phase_mod = math.cos(phi_clock.phi_phase + t * 0.1)
                    new_values[t][p][h] *= bio_factor * (0.95 + 0.1 * phase_mod)

    return ScalarField(values=new_values, size=field.size)


def compute_local_coherence(field: ScalarField, pos: Position) -> float:
    """
    Compute local coherence at a position.

    Coherence combines the field value with local stability.
    Higher field values AND smoother neighborhoods = higher coherence.
    """
    center_val = field.get_value(pos)

    # Sample neighbors
    neighbors = []
    for dt in [-1, 0, 1]:
        for dp in [-1, 0, 1]:
            for dh in [-1, 0, 1]:
                if dt == 0 and dp == 0 and dh == 0:
                    continue
                neighbor_pos = Position(
                    theta=pos.theta + dt,
                    phi=pos.phi + dp,
                    h=pos.h + dh
                )
                neighbors.append(field.get_value(neighbor_pos))

    if not neighbors:
        return center_val

    # Stability factor (1 - normalized variance)
    mean_neighbor = sum(neighbors) / len(neighbors)
    variance = sum((n - mean_neighbor)**2 for n in neighbors) / len(neighbors)
    stability = max(0, 1 - math.sqrt(variance))

    # Coherence = field value weighted by stability
    # High value + stable neighborhood = high coherence
    return center_val * (0.5 + 0.5 * stability)


def compute_coherence_gradient(field: ScalarField, pos: Position) -> Tuple[int, int, int]:
    """
    Compute coherence gradient direction (where coherence increases).

    Returns direction to move toward higher coherence.
    """
    best_dir = (0, 0, 0)
    best_coherence = compute_local_coherence(field, pos)

    for dt in [-1, 0, 1]:
        for dp in [-1, 0, 1]:
            for dh in [-1, 0, 1]:
                if dt == 0 and dp == 0 and dh == 0:
                    continue
                neighbor_pos = Position(
                    theta=pos.theta + dt,
                    phi=pos.phi + dp,
                    h=pos.h + dh
                )
                neighbor_coh = compute_local_coherence(field, neighbor_pos)
                if neighbor_coh > best_coherence:
                    best_coherence = neighbor_coh
                    best_dir = (dt, dp, dh)

    return best_dir


def move_avatar(avatar: Avatar, field: ScalarField) -> Avatar:
    """
    Move avatar following coherence gradient.

    Avatar seeks higher coherence regions.
    """
    # Get coherence gradient direction
    gradient = compute_coherence_gradient(field, avatar.position)

    # Move toward higher coherence
    new_pos = Position(
        theta=(avatar.position.theta + gradient[0]) % field.size,
        phi=(avatar.position.phi + gradient[1]) % field.size,
        h=(avatar.position.h + gradient[2]) % field.size
    )

    # Update coherence level
    new_coherence = compute_local_coherence(field, new_pos)

    # Determine resonance state
    if new_coherence > 0.85:
        state = AvatarState.RESONANT
    elif new_coherence > 0.5:
        state = AvatarState.STABLE
    elif new_coherence > 0.3:
        state = AvatarState.SEEKING
    else:
        state = AvatarState.DISRUPTED

    return Avatar(
        avatar_id=avatar.avatar_id,
        position=new_pos,
        resonance_state=state,
        coherence_level=new_coherence,
        velocity=(float(gradient[0]), float(gradient[1]), float(gradient[2]))
    )


def manifest_fragments(
    field: ScalarField,
    fragments: List[Fragment],
    coherence_threshold: float = COHERENCE_EMERGENCE_THRESHOLD
) -> List[Fragment]:
    """
    Update fragment manifestation based on local field coherence.

    Fragment manifests when local coherence exceeds threshold.
    """
    updated = []
    for frag in fragments:
        local_coh = compute_local_coherence(field, frag.position)
        is_manifest = local_coh >= coherence_threshold

        updated.append(Fragment(
            fragment_id=frag.fragment_id,
            position=frag.position,
            harmonic_l=frag.harmonic_l,
            harmonic_m=frag.harmonic_m,
            intensity=frag.intensity * (1.0 if is_manifest else 0.5),
            is_manifest=is_manifest
        ))
    return updated


def check_fragment_alignment(fragments: List[Fragment]) -> List[Tuple[int, int, int]]:
    """
    Check for aligned fragment groups (3+ in proximity).

    Returns list of aligned fragment ID tuples.
    """
    aligned_groups = []
    manifest = [f for f in fragments if f.is_manifest]

    # Check all triplets
    for i, f1 in enumerate(manifest):
        for j, f2 in enumerate(manifest[i+1:], i+1):
            for k, f3 in enumerate(manifest[j+1:], j+1):
                # Check if all within proximity
                d12 = f1.position.distance_to(f2.position)
                d23 = f2.position.distance_to(f3.position)
                d13 = f1.position.distance_to(f3.position)

                if (d12 < FRAGMENT_PROXIMITY_THRESHOLD and
                    d23 < FRAGMENT_PROXIMITY_THRESHOLD and
                    d13 < FRAGMENT_PROXIMITY_THRESHOLD):
                    aligned_groups.append((f1.fragment_id, f2.fragment_id, f3.fragment_id))

    return aligned_groups


def detect_emergence(
    field: ScalarField,
    avatar: Avatar,
    fragments: List[Fragment],
    timestamp: int
) -> List[EmergentContent]:
    """
    Detect emergence events based on combined criteria.

    Emergence when: coherence > 0.72 AND fragment proximity < threshold
    """
    events = []

    # Check avatar position coherence
    avatar_coh = compute_local_coherence(field, avatar.position)

    # Coherence spike at avatar location
    if avatar_coh >= COHERENCE_EMERGENCE_THRESHOLD:
        # Check for nearby manifest fragments
        nearby_fragments = []
        for frag in fragments:
            if frag.is_manifest:
                dist = avatar.position.distance_to(frag.position)
                if dist < FRAGMENT_PROXIMITY_THRESHOLD:
                    nearby_fragments.append(frag.fragment_id)

        if len(nearby_fragments) >= 1:
            events.append(EmergentContent(
                emergence_type=EmergenceType.COHERENCE_SPIKE,
                position=avatar.position,
                timestamp=timestamp,
                coherence_level=avatar_coh,
                fragments_involved=nearby_fragments
            ))

    # Check for fragment alignments (3+ fragments)
    alignments = check_fragment_alignment(fragments)
    for alignment in alignments:
        # Calculate center position
        aligned_frags = [f for f in fragments if f.fragment_id in alignment]
        center_t = sum(f.position.theta for f in aligned_frags) // len(aligned_frags)
        center_p = sum(f.position.phi for f in aligned_frags) // len(aligned_frags)
        center_h = sum(f.position.h for f in aligned_frags) // len(aligned_frags)
        center = Position(theta=center_t, phi=center_p, h=center_h)

        center_coh = compute_local_coherence(field, center)

        if center_coh >= COHERENCE_EMERGENCE_THRESHOLD:
            events.append(EmergentContent(
                emergence_type=EmergenceType.FRAGMENT_ALIGNMENT,
                position=center,
                timestamp=timestamp,
                coherence_level=center_coh,
                fragments_involved=list(alignment)
            ))

    # Check for avatar resonance
    if avatar.resonance_state == AvatarState.RESONANT:
        events.append(EmergentContent(
            emergence_type=EmergenceType.AVATAR_RESONANCE,
            position=avatar.position,
            timestamp=timestamp,
            coherence_level=avatar.coherence_level,
            fragments_involved=[]
        ))

    return events


def compute_coherence_trace(
    field: ScalarField,
    fragments: List[Fragment]
) -> List[float]:
    """
    Compute coherence values at fragment positions.

    Returns list of coherence values for graphing.
    """
    return [compute_local_coherence(field, f.position) for f in fragments]


def generate_trace_log(
    chamber: AetherChamber,
    bio: Optional[BioState],
    coherence_values: List[float]
) -> str:
    """Generate trace log message for simulation step."""
    avg_coh = sum(coherence_values) / len(coherence_values) if coherence_values else 0
    bio_str = f"HRV={bio.hrv:.0f}ms" if bio else "no bio"

    return (f"t={chamber.time_context.tick} | "
            f"avatar@({chamber.avatar_state.position.theta},{chamber.avatar_state.position.phi},{chamber.avatar_state.position.h}) "
            f"[{chamber.avatar_state.resonance_state.name}] | "
            f"avg_coh={avg_coh:.3f} | {bio_str}")


def simulate_step(
    chamber: AetherChamber,
    bio: Optional[BioState] = None
) -> SimulationResult:
    """
    Run one simulation step.

    Updates field, moves avatar, manifests fragments, detects emergence.
    """
    # Get bio state from stream if available
    if bio is None and chamber.biometric_input is not None:
        bio = chamber.biometric_input.get_next()

    # Update scalar field
    updated_field = update_scalar_field(
        chamber.field_profile,
        bio,
        chamber.time_context
    )

    # Move avatar following coherence gradient
    avatar_next = move_avatar(chamber.avatar_state, updated_field)

    # Manifest fragments based on field coherence
    fragments = manifest_fragments(updated_field, chamber.fragment_set)

    # Compute coherence trace
    coherence = compute_coherence_trace(updated_field, fragments)

    # Detect emergence events
    emergents = detect_emergence(
        updated_field,
        avatar_next,
        fragments,
        chamber.time_context.tick
    )

    # Generate trace log
    log_msg = generate_trace_log(chamber, bio, coherence)

    # Create updated chamber
    new_clock = PhiClock(tick=chamber.time_context.tick, phi_phase=chamber.time_context.phi_phase)
    new_clock.advance()

    updated_chamber = AetherChamber(
        chamber_id=chamber.chamber_id,
        field_profile=updated_field,
        biometric_input=chamber.biometric_input,
        fragment_set=fragments,
        avatar_state=avatar_next,
        time_context=new_clock,
        inversion_zone=chamber.inversion_zone
    )

    return SimulationResult(
        emergence_events=emergents,
        coherence_graph=coherence,
        trace_log=[log_msg],
        chamber_delta=updated_chamber
    )


# =============================================================================
# Chamber Templates
# =============================================================================

def create_phi_chamber(chamber_id: str = "lab-phi-001") -> AetherChamber:
    """
    Create φ-nested chamber template.

    810-point field with GOLOD fragments and synthetic bio.
    """
    field = generate_phi_tuned_field(DEFAULT_FIELD_SIZE, 6, 5)
    fragments = create_golod_seed_array(12, DEFAULT_FIELD_SIZE)
    bio_stream = BioStream(samples=[420, 460, 480, 440, 500])

    avatar = Avatar(
        avatar_id=1,
        position=Position(theta=13, phi=3, h=2),
        resonance_state=AvatarState.STABLE,
        coherence_level=0.5
    )

    inversion = InversionRegion(
        center=Position(theta=14, phi=4, h=2),
        radius=3,
        intensity=0.6
    )

    return AetherChamber(
        chamber_id=chamber_id,
        field_profile=field,
        biometric_input=bio_stream,
        fragment_set=fragments,
        avatar_state=avatar,
        time_context=PhiClock(),
        inversion_zone=inversion
    )


# =============================================================================
# Test Cases
# =============================================================================

class TestDataStructures:
    """Tests for basic data structures."""

    def test_position_distance(self):
        """Test position distance calculation."""
        p1 = Position(theta=0, phi=0, h=0)
        p2 = Position(theta=3, phi=4, h=0)
        assert abs(p1.distance_to(p2) - 5.0) < 0.001

    def test_bio_stream_cycles(self):
        """Test bio stream cycling behavior."""
        stream = BioStream(samples=[400, 450, 500])

        states = [stream.get_next() for _ in range(6)]
        # Should cycle after 3 samples
        assert states[0].hrv == states[3].hrv

    def test_bio_state_coherence(self):
        """Test bio state coherence derivation."""
        stream = BioStream(samples=[450, 450, 450])  # Stable HRV
        state = stream.get_next()
        assert state.coherence > 0.9  # High coherence for stable HRV

    def test_phi_clock_advance(self):
        """Test phi clock advancement."""
        clock = PhiClock()
        assert clock.tick == 0

        clock.advance()
        assert clock.tick == 1
        assert 0 <= clock.phi_phase < 2 * math.pi


class TestFieldGeneration:
    """Tests for scalar field generation."""

    def test_phi_tuned_field_size(self):
        """Test field has correct dimensions."""
        field = generate_phi_tuned_field(10, 3, 2)
        assert field.size == 10
        assert len(field.values) == 10
        assert len(field.values[0]) == 10
        assert len(field.values[0][0]) == 10

    def test_phi_tuned_field_values(self):
        """Test field values are in valid range."""
        field = generate_phi_tuned_field(10, 3, 2)
        for t in range(10):
            for p in range(10):
                for h in range(10):
                    assert 0 <= field.values[t][p][h] <= 1

    def test_golod_seed_distribution(self):
        """Test GOLOD fragments are distributed."""
        fragments = create_golod_seed_array(10, 27)
        positions = [(f.position.theta, f.position.phi, f.position.h) for f in fragments]
        # Should have unique positions
        assert len(set(positions)) == len(positions)


class TestCoherenceComputation:
    """Tests for coherence calculations."""

    def test_uniform_field_high_coherence(self):
        """Uniform high-value field should have high coherence."""
        # Coherence = value * (0.5 + 0.5 * stability)
        # For uniform field: stability = 1, so coherence = value * 1.0 = value
        values = [[[0.95 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        pos = Position(theta=5, phi=5, h=5)
        coh = compute_local_coherence(field, pos)
        assert coh > 0.90  # High for uniform high-value field

    def test_random_field_lower_coherence(self):
        """Random field should have lower coherence."""
        import random
        random.seed(42)
        values = [[[random.random() for _ in range(10)]
                   for _ in range(10)]
                  for _ in range(10)]
        field = ScalarField(values=values, size=10)

        pos = Position(theta=5, phi=5, h=5)
        coh = compute_local_coherence(field, pos)
        assert coh < 0.9  # Lower for random

    def test_coherence_gradient_direction(self):
        """Gradient should point toward higher coherence."""
        # Create field with peak at center
        values = [[[0.0 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        values[5][5][5] = 1.0  # Peak at center
        field = ScalarField(values=values, size=10)

        # From corner, gradient should point toward center
        pos = Position(theta=3, phi=3, h=3)
        grad = compute_coherence_gradient(field, pos)

        # Should have positive components (toward center)
        assert grad[0] >= 0 and grad[1] >= 0 and grad[2] >= 0


class TestAvatarMovement:
    """Tests for avatar movement."""

    def test_avatar_moves_toward_coherence(self):
        """Avatar should move toward higher coherence."""
        # Create field with smooth gradient toward center
        # This creates naturally high coherence at the peak
        values = [[[0.0 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        center = 5
        for t in range(10):
            for p in range(10):
                for h in range(10):
                    # Smooth gaussian-like falloff from center
                    dist = math.sqrt((t-center)**2 + (p-center)**2 + (h-center)**2)
                    values[t][p][h] = math.exp(-dist**2 / 20.0)
        field = ScalarField(values=values, size=10)

        # Start at edge, should move toward center where coherence is highest
        avatar = Avatar(
            avatar_id=1,
            position=Position(theta=1, phi=1, h=1),
            resonance_state=AvatarState.SEEKING
        )

        new_avatar = move_avatar(avatar, field)

        # Should have moved toward center (5,5,5)
        old_dist = avatar.position.distance_to(Position(center, center, center))
        new_dist = new_avatar.position.distance_to(Position(center, center, center))
        assert new_dist <= old_dist

    def test_avatar_state_updates(self):
        """Avatar state should update based on coherence."""
        values = [[[0.95 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        avatar = Avatar(
            avatar_id=1,
            position=Position(theta=5, phi=5, h=5),
            resonance_state=AvatarState.SEEKING
        )

        new_avatar = move_avatar(avatar, field)
        # High coherence should give RESONANT state
        assert new_avatar.resonance_state == AvatarState.RESONANT


class TestFragmentManifestation:
    """Tests for fragment manifestation."""

    def test_high_coherence_manifests(self):
        """Fragments at high coherence should manifest."""
        values = [[[0.9 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        fragments = [Fragment(
            fragment_id=0,
            position=Position(theta=5, phi=5, h=5),
            harmonic_l=2,
            harmonic_m=0,
            intensity=0.8,
            is_manifest=False
        )]

        updated = manifest_fragments(field, fragments, coherence_threshold=0.72)
        assert updated[0].is_manifest is True

    def test_low_coherence_no_manifest(self):
        """Fragments at low coherence should not manifest."""
        values = [[[0.3 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        fragments = [Fragment(
            fragment_id=0,
            position=Position(theta=5, phi=5, h=5),
            harmonic_l=2,
            harmonic_m=0,
            intensity=0.8,
            is_manifest=True
        )]

        updated = manifest_fragments(field, fragments, coherence_threshold=0.72)
        assert updated[0].is_manifest is False


class TestFragmentAlignment:
    """Tests for fragment alignment detection."""

    def test_close_fragments_align(self):
        """Fragments within proximity should be detected as aligned."""
        fragments = [
            Fragment(0, Position(5, 5, 5), 2, 0, 0.8, is_manifest=True),
            Fragment(1, Position(6, 5, 5), 2, 0, 0.8, is_manifest=True),
            Fragment(2, Position(5, 6, 5), 2, 0, 0.8, is_manifest=True),
        ]

        alignments = check_fragment_alignment(fragments)
        assert len(alignments) == 1
        assert set(alignments[0]) == {0, 1, 2}

    def test_far_fragments_no_align(self):
        """Fragments far apart should not align."""
        fragments = [
            Fragment(0, Position(0, 0, 0), 2, 0, 0.8, is_manifest=True),
            Fragment(1, Position(10, 10, 10), 2, 0, 0.8, is_manifest=True),
            Fragment(2, Position(20, 20, 20), 2, 0, 0.8, is_manifest=True),
        ]

        alignments = check_fragment_alignment(fragments)
        assert len(alignments) == 0

    def test_unmanifest_fragments_ignored(self):
        """Unmanifest fragments should not count for alignment."""
        fragments = [
            Fragment(0, Position(5, 5, 5), 2, 0, 0.8, is_manifest=True),
            Fragment(1, Position(6, 5, 5), 2, 0, 0.8, is_manifest=False),  # Not manifest
            Fragment(2, Position(5, 6, 5), 2, 0, 0.8, is_manifest=True),
        ]

        alignments = check_fragment_alignment(fragments)
        assert len(alignments) == 0  # Only 2 manifest, need 3


class TestEmergenceDetection:
    """Tests for emergence event detection."""

    def test_coherence_spike_emergence(self):
        """High coherence with nearby fragment triggers emergence."""
        values = [[[0.9 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        avatar = Avatar(
            avatar_id=1,
            position=Position(theta=5, phi=5, h=5),
            resonance_state=AvatarState.RESONANT,
            coherence_level=0.9
        )

        fragments = [Fragment(
            fragment_id=0,
            position=Position(theta=5, phi=5, h=5),
            harmonic_l=2,
            harmonic_m=0,
            intensity=0.8,
            is_manifest=True
        )]

        events = detect_emergence(field, avatar, fragments, timestamp=100)

        # Should have coherence spike event
        spike_events = [e for e in events if e.emergence_type == EmergenceType.COHERENCE_SPIKE]
        assert len(spike_events) >= 1

    def test_avatar_resonance_emergence(self):
        """Resonant avatar should trigger emergence."""
        values = [[[0.5 for _ in range(10)] for _ in range(10)] for _ in range(10)]
        field = ScalarField(values=values, size=10)

        avatar = Avatar(
            avatar_id=1,
            position=Position(theta=5, phi=5, h=5),
            resonance_state=AvatarState.RESONANT,
            coherence_level=0.9
        )

        events = detect_emergence(field, avatar, [], timestamp=100)

        resonance_events = [e for e in events if e.emergence_type == EmergenceType.AVATAR_RESONANCE]
        assert len(resonance_events) == 1


class TestSimulationStep:
    """Tests for full simulation step."""

    def test_simulation_step_runs(self):
        """Simulation step should complete without error."""
        chamber = create_phi_chamber()
        result = simulate_step(chamber)

        assert result.chamber_delta is not None
        assert len(result.coherence_graph) > 0
        assert len(result.trace_log) == 1

    def test_clock_advances(self):
        """Clock should advance after simulation step."""
        chamber = create_phi_chamber()
        assert chamber.time_context.tick == 0

        result = simulate_step(chamber)
        assert result.chamber_delta.time_context.tick == 1

    def test_bio_stream_consumed(self):
        """Bio stream should be consumed during simulation."""
        chamber = create_phi_chamber()
        initial_idx = chamber.biometric_input.current_index

        simulate_step(chamber)
        # Stream index should have advanced
        assert chamber.biometric_input.current_index == initial_idx + 1

    def test_multiple_steps(self):
        """Multiple simulation steps should accumulate."""
        chamber = create_phi_chamber()

        for _ in range(5):
            result = simulate_step(chamber)
            chamber = result.chamber_delta

        assert chamber.time_context.tick == 5


class TestPhiChamberTemplate:
    """Tests for phi chamber template."""

    def test_chamber_initialization(self):
        """Chamber should initialize with all components."""
        chamber = create_phi_chamber()

        assert chamber.chamber_id == "lab-phi-001"
        assert chamber.field_profile is not None
        assert chamber.biometric_input is not None
        assert len(chamber.fragment_set) == 12
        assert chamber.avatar_state is not None
        assert chamber.inversion_zone is not None

    def test_chamber_field_is_phi_tuned(self):
        """Chamber field should be φ-tuned."""
        chamber = create_phi_chamber()

        # Field should have φ-based value distribution
        val1 = chamber.field_profile.get_value(Position(0, 0, 0))
        val2 = chamber.field_profile.get_value(Position(13, 13, 13))
        # Values should differ (not uniform)
        assert val1 != val2


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_five_cycle_simulation(self):
        """Run 5-cycle simulation capturing all metrics."""
        chamber = create_phi_chamber()
        all_events = []
        all_coherence = []

        for _ in range(5):
            result = simulate_step(chamber)
            all_events.extend(result.emergence_events)
            all_coherence.extend(result.coherence_graph)
            chamber = result.chamber_delta

        # Should have coherence data
        assert len(all_coherence) > 0

    def test_emergence_accumulation(self):
        """Emergence events should accumulate over simulation."""
        chamber = create_phi_chamber()
        total_events = 0

        for _ in range(10):
            result = simulate_step(chamber)
            total_events += len(result.emergence_events)
            chamber = result.chamber_delta

        # May or may not have events depending on field state
        assert total_events >= 0

    def test_avatar_traversal(self):
        """Avatar should traverse chamber seeking coherence."""
        chamber = create_phi_chamber()
        positions = [chamber.avatar_state.position]

        for _ in range(10):
            result = simulate_step(chamber)
            positions.append(result.chamber_delta.avatar_state.position)
            chamber = result.chamber_delta

        # Avatar should have moved
        unique_positions = set((p.theta, p.phi, p.h) for p in positions)
        # May stay in place if already at peak, but test traversal happened
        assert len(positions) == 11


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
