"""
Test Suite for Ra.Memory.ORMEWaterSync (Prompt 72)

Models quantum coherence coupling between structured water and user
memory/emotional imprinting. Simulates loopback dynamics between
memory states, emotional signatures, and avatar-water phase resonance.

Architect Clarifications:
- Emotion intensity > 0.75 triggers imprinting, specific valence tags supported
- Phase transitions (Dormant→Entangled→Loopback) based on α coherence windows
- Entanglement: proximity-driven (spatial + α) or intent-driven (focus gesture)
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple

# Constants
PHI = 1.618033988749895
EMOTION_INTENSITY_THRESHOLD = 0.75
SUSTAINED_DURATION_TICKS = 8
ENTANGLEMENT_RADIUS = 1.0  # Spatial coherence zone radius
ALPHA_ENTANGLEMENT_MIN = 0.6
ALPHA_LOOPBACK_MIN = 0.85


class ORMEPhase(Enum):
    """ORME carrier state phases."""
    DORMANT = auto()     # Inactive, no coherence
    ENTANGLED = auto()   # Linked but not looping
    LOOPBACK = auto()    # Active memory loop


class EmotionValence(Enum):
    """Emotion valence tags for imprinting."""
    NEUTRAL = auto()
    JOY = auto()
    AWE = auto()
    GRIEF = auto()
    LOVE = auto()
    FEAR = auto()
    PEACE = auto()
    ANGER = auto()


class EntanglementMode(Enum):
    """Mode of memory entanglement."""
    PROXIMITY = auto()   # Spatial coherence zones
    INTENT = auto()      # User selection/focus gesture


@dataclass
class EmotionWaveform:
    """Emotional signature waveform."""
    valence: EmotionValence
    intensity: float      # 0-1
    frequency: float      # Hz (emotional rhythm)
    phase: float          # Radians
    duration_ticks: int   # Sustained duration


@dataclass
class FragmentID:
    """Memory fragment identifier."""
    namespace: int
    index: int


@dataclass
class RaCoordinate:
    """3D position in scalar field."""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'RaCoordinate') -> float:
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )


@dataclass
class ScalarFieldPoint:
    """Point in scalar field."""
    coordinate: RaCoordinate
    alpha: float


@dataclass
class ScalarField:
    """Scalar field for coherence evaluation."""
    points: List[ScalarFieldPoint]
    mean_alpha: float


@dataclass
class BioState:
    """Biometric state for water sync."""
    hrv: float           # 0-1 HRV coherence
    coherence: float     # 0-1 overall coherence
    focus_level: float   # 0-1 attention/focus metric
    pupil_dilation: float  # Relative dilation


@dataclass
class WaterCoherenceField:
    """Structured water coherence field."""
    molecular_lattice: List[List[float]]  # Bond angle grid
    imprint_signature: Optional[EmotionWaveform]
    orme_carrier_state: ORMEPhase
    memory_entanglement: Optional[FragmentID]
    field_position: RaCoordinate
    local_alpha: float


def compute_molecular_lattice(coherence: float, size: int = 4) -> List[List[float]]:
    """
    Generate molecular lattice representing structured water bond angles.
    Higher coherence = more ordered lattice (closer to tetrahedral 109.5°).
    """
    base_angle = 109.5  # Tetrahedral angle
    disorder = (1.0 - coherence) * 20.0  # Max 20° disorder

    lattice = []
    for i in range(size):
        row = []
        for j in range(size):
            # Add coherence-dependent variation
            variation = disorder * math.sin(i * PHI) * math.cos(j * PHI)
            angle = base_angle + variation
            row.append(angle)
        lattice.append(row)

    return lattice


def check_emotion_imprint_trigger(emotion: EmotionWaveform) -> bool:
    """Check if emotion meets imprinting criteria."""
    # Intensity threshold
    if emotion.intensity < EMOTION_INTENSITY_THRESHOLD:
        return False

    # Sustained duration
    if emotion.duration_ticks < SUSTAINED_DURATION_TICKS:
        return False

    return True


def is_imprint_valence(valence: EmotionValence) -> bool:
    """Check if valence type supports strong imprinting."""
    strong_valences = {
        EmotionValence.JOY,
        EmotionValence.AWE,
        EmotionValence.GRIEF,
        EmotionValence.LOVE,
        EmotionValence.PEACE,
    }
    return valence in strong_valences


def compute_orme_phase(
    bio_state: BioState,
    field_alpha: float,
    current_phase: ORMEPhase
) -> ORMEPhase:
    """
    Compute ORME phase based on coherence windows.
    Transitions: Dormant → Entangled → Loopback
    """
    combined_coherence = (bio_state.coherence + field_alpha) / 2.0

    # Phase transitions based on α bands
    if combined_coherence >= ALPHA_LOOPBACK_MIN:
        return ORMEPhase.LOOPBACK
    elif combined_coherence >= ALPHA_ENTANGLEMENT_MIN:
        # Can only go to Entangled from Dormant, or stay in Entangled
        if current_phase == ORMEPhase.LOOPBACK:
            return ORMEPhase.LOOPBACK  # Hysteresis - stay in loopback
        return ORMEPhase.ENTANGLED
    else:
        # Low coherence - return to dormant
        if current_phase == ORMEPhase.ENTANGLED:
            return ORMEPhase.ENTANGLED  # Brief hysteresis
        return ORMEPhase.DORMANT


def check_proximity_entanglement(
    field1: WaterCoherenceField,
    field2: WaterCoherenceField
) -> bool:
    """Check if two fields can entangle via proximity."""
    # Distance check
    distance = field1.field_position.distance_to(field2.field_position)
    if distance > ENTANGLEMENT_RADIUS:
        return False

    # Alpha alignment check
    alpha_diff = abs(field1.local_alpha - field2.local_alpha)
    if alpha_diff > 0.2:  # Must be within 0.2 of each other
        return False

    # Both must be at least entangled phase
    if field1.orme_carrier_state == ORMEPhase.DORMANT:
        return False
    if field2.orme_carrier_state == ORMEPhase.DORMANT:
        return False

    return True


def check_intent_entanglement(
    bio_state: BioState,
    target_fragment: FragmentID
) -> bool:
    """Check if intent-driven entanglement is possible."""
    # Requires high focus level
    if bio_state.focus_level < 0.7:
        return False

    # Requires elevated HRV coherence
    if bio_state.hrv < 0.6:
        return False

    return True


def create_memory_link(
    source_field: WaterCoherenceField,
    fragment_id: FragmentID,
    mode: EntanglementMode,
    bio_state: Optional[BioState] = None,
    target_field: Optional[WaterCoherenceField] = None
) -> Optional[FragmentID]:
    """Create memory entanglement link."""
    if mode == EntanglementMode.PROXIMITY:
        if target_field is None:
            return None
        if not check_proximity_entanglement(source_field, target_field):
            return None
        return fragment_id

    elif mode == EntanglementMode.INTENT:
        if bio_state is None:
            return None
        if not check_intent_entanglement(bio_state, fragment_id):
            return None
        return fragment_id

    return None


def generate_water_loopback(
    bio_state: BioState,
    emotion: EmotionWaveform,
    scalar_field: ScalarField,
    position: RaCoordinate = None,
    existing_field: Optional[WaterCoherenceField] = None
) -> WaterCoherenceField:
    """
    Generate water coherence field with loopback dynamics.
    Main function contract implementation.
    """
    if position is None:
        position = RaCoordinate(0, 0, 0)

    # Compute local alpha from scalar field
    local_alpha = scalar_field.mean_alpha
    if scalar_field.points:
        # Find closest point
        closest = min(scalar_field.points,
                     key=lambda p: position.distance_to(p.coordinate))
        local_alpha = closest.alpha

    # Generate molecular lattice based on combined coherence
    combined_coh = (bio_state.coherence + local_alpha) / 2.0
    lattice = compute_molecular_lattice(combined_coh)

    # Determine current phase
    current_phase = ORMEPhase.DORMANT
    if existing_field:
        current_phase = existing_field.orme_carrier_state

    # Compute new ORME phase
    new_phase = compute_orme_phase(bio_state, local_alpha, current_phase)

    # Check for imprint
    imprint = None
    if check_emotion_imprint_trigger(emotion) and is_imprint_valence(emotion.valence):
        if new_phase != ORMEPhase.DORMANT:
            imprint = emotion

    # Preserve memory entanglement if exists and still valid
    memory_link = None
    if existing_field and existing_field.memory_entanglement:
        if new_phase != ORMEPhase.DORMANT:
            memory_link = existing_field.memory_entanglement

    return WaterCoherenceField(
        molecular_lattice=lattice,
        imprint_signature=imprint,
        orme_carrier_state=new_phase,
        memory_entanglement=memory_link,
        field_position=position,
        local_alpha=local_alpha
    )


def compute_loopback_strength(field: WaterCoherenceField) -> float:
    """Compute strength of memory loopback."""
    if field.orme_carrier_state != ORMEPhase.LOOPBACK:
        return 0.0

    base_strength = field.local_alpha

    # Imprint amplifies loopback
    if field.imprint_signature:
        imprint_factor = 1.0 + (field.imprint_signature.intensity * 0.5)
        base_strength *= imprint_factor

    # Memory entanglement amplifies further
    if field.memory_entanglement:
        base_strength *= 1.2

    return min(1.0, base_strength)


def validate_water_field(field: WaterCoherenceField) -> bool:
    """Validate water coherence field structure."""
    # Must have lattice
    if not field.molecular_lattice:
        return False

    # Lattice must be 2D
    if not all(isinstance(row, list) for row in field.molecular_lattice):
        return False

    # Bond angles should be in reasonable range (90-130°)
    for row in field.molecular_lattice:
        for angle in row:
            if angle < 90 or angle > 130:
                return False

    return True


# ============== TESTS ==============

class TestEmotionImprinting:
    """Tests for emotional resonance imprinting."""

    def test_high_intensity_triggers_imprint(self):
        """Intensity > 0.75 should trigger imprint."""
        emotion = EmotionWaveform(
            valence=EmotionValence.JOY,
            intensity=0.8,
            frequency=7.83,
            phase=0,
            duration_ticks=10
        )
        assert check_emotion_imprint_trigger(emotion)

    def test_low_intensity_no_imprint(self):
        """Intensity < 0.75 should not trigger imprint."""
        emotion = EmotionWaveform(
            valence=EmotionValence.JOY,
            intensity=0.5,
            frequency=7.83,
            phase=0,
            duration_ticks=10
        )
        assert not check_emotion_imprint_trigger(emotion)

    def test_short_duration_no_imprint(self):
        """Duration < 8 ticks should not trigger imprint."""
        emotion = EmotionWaveform(
            valence=EmotionValence.AWE,
            intensity=0.9,
            frequency=7.83,
            phase=0,
            duration_ticks=5
        )
        assert not check_emotion_imprint_trigger(emotion)

    def test_sustained_duration_triggers(self):
        """Duration >= 8 ticks with intensity should trigger."""
        emotion = EmotionWaveform(
            valence=EmotionValence.GRIEF,
            intensity=0.85,
            frequency=7.83,
            phase=0,
            duration_ticks=8
        )
        assert check_emotion_imprint_trigger(emotion)

    def test_joy_is_imprint_valence(self):
        """Joy should be a strong imprint valence."""
        assert is_imprint_valence(EmotionValence.JOY)

    def test_awe_is_imprint_valence(self):
        """Awe should be a strong imprint valence."""
        assert is_imprint_valence(EmotionValence.AWE)

    def test_grief_is_imprint_valence(self):
        """Grief should be a strong imprint valence."""
        assert is_imprint_valence(EmotionValence.GRIEF)

    def test_neutral_not_imprint_valence(self):
        """Neutral should not be a strong imprint valence."""
        assert not is_imprint_valence(EmotionValence.NEUTRAL)

    def test_anger_not_imprint_valence(self):
        """Anger should not be a strong imprint valence."""
        assert not is_imprint_valence(EmotionValence.ANGER)


class TestORMEPhaseTransitions:
    """Tests for ORME phase state transitions."""

    def test_dormant_to_entangled_on_moderate_alpha(self):
        """Should transition to Entangled at α >= 0.6."""
        bio = BioState(hrv=0.7, coherence=0.65, focus_level=0.5, pupil_dilation=0.5)
        phase = compute_orme_phase(bio, 0.7, ORMEPhase.DORMANT)
        assert phase == ORMEPhase.ENTANGLED

    def test_entangled_to_loopback_on_high_alpha(self):
        """Should transition to Loopback at α >= 0.85."""
        bio = BioState(hrv=0.8, coherence=0.9, focus_level=0.8, pupil_dilation=0.6)
        phase = compute_orme_phase(bio, 0.9, ORMEPhase.ENTANGLED)
        assert phase == ORMEPhase.LOOPBACK

    def test_stays_dormant_on_low_alpha(self):
        """Should stay Dormant at α < 0.6."""
        bio = BioState(hrv=0.3, coherence=0.4, focus_level=0.3, pupil_dilation=0.4)
        phase = compute_orme_phase(bio, 0.3, ORMEPhase.DORMANT)
        assert phase == ORMEPhase.DORMANT

    def test_loopback_has_hysteresis(self):
        """Loopback should persist with moderate coherence drop."""
        bio = BioState(hrv=0.6, coherence=0.7, focus_level=0.6, pupil_dilation=0.5)
        # Start in loopback, moderate coherence
        phase = compute_orme_phase(bio, 0.7, ORMEPhase.LOOPBACK)
        assert phase == ORMEPhase.LOOPBACK

    def test_combined_coherence_calculation(self):
        """Phase should use combined bio + field coherence."""
        bio = BioState(hrv=0.5, coherence=0.5, focus_level=0.5, pupil_dilation=0.5)
        # Bio coherence 0.5 + field alpha 0.8 = combined 0.65
        phase = compute_orme_phase(bio, 0.8, ORMEPhase.DORMANT)
        assert phase == ORMEPhase.ENTANGLED


class TestMolecularLattice:
    """Tests for molecular lattice generation."""

    def test_lattice_is_2d_grid(self):
        """Lattice should be 2D grid."""
        lattice = compute_molecular_lattice(0.8)
        assert len(lattice) > 0
        assert all(isinstance(row, list) for row in lattice)

    def test_high_coherence_ordered_lattice(self):
        """High coherence should produce ordered lattice near 109.5°."""
        lattice = compute_molecular_lattice(0.99)
        # All angles should be close to tetrahedral
        for row in lattice:
            for angle in row:
                assert 105 < angle < 114

    def test_low_coherence_disordered_lattice(self):
        """Low coherence should produce disordered lattice."""
        lattice = compute_molecular_lattice(0.1)
        # Should have wider variation
        angles = [angle for row in lattice for angle in row]
        angle_range = max(angles) - min(angles)
        assert angle_range > 10  # Significant disorder

    def test_lattice_size_configurable(self):
        """Lattice size should be configurable."""
        lattice_small = compute_molecular_lattice(0.5, size=2)
        lattice_large = compute_molecular_lattice(0.5, size=6)
        assert len(lattice_small) == 2
        assert len(lattice_large) == 6


class TestProximityEntanglement:
    """Tests for proximity-based memory entanglement."""

    def test_nearby_aligned_fields_can_entangle(self):
        """Fields within radius with aligned α can entangle."""
        field1 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        field2 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0.5, 0, 0),  # Within radius
            local_alpha=0.75  # Within 0.2 of field1
        )
        assert check_proximity_entanglement(field1, field2)

    def test_distant_fields_cannot_entangle(self):
        """Fields beyond radius cannot entangle."""
        field1 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        field2 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(5, 0, 0),  # Beyond radius
            local_alpha=0.8
        )
        assert not check_proximity_entanglement(field1, field2)

    def test_misaligned_alpha_cannot_entangle(self):
        """Fields with misaligned α cannot entangle."""
        field1 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.9
        )
        field2 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0.5, 0, 0),
            local_alpha=0.5  # > 0.2 difference
        )
        assert not check_proximity_entanglement(field1, field2)

    def test_dormant_field_cannot_entangle(self):
        """Dormant field cannot participate in entanglement."""
        field1 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.DORMANT,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        field2 = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0.5, 0, 0),
            local_alpha=0.8
        )
        assert not check_proximity_entanglement(field1, field2)


class TestIntentEntanglement:
    """Tests for intent-driven memory entanglement."""

    def test_high_focus_enables_intent_entanglement(self):
        """High focus level enables intent entanglement."""
        bio = BioState(hrv=0.7, coherence=0.8, focus_level=0.8, pupil_dilation=0.6)
        fragment = FragmentID(namespace=1, index=42)
        assert check_intent_entanglement(bio, fragment)

    def test_low_focus_blocks_intent_entanglement(self):
        """Low focus level blocks intent entanglement."""
        bio = BioState(hrv=0.7, coherence=0.8, focus_level=0.5, pupil_dilation=0.6)
        fragment = FragmentID(namespace=1, index=42)
        assert not check_intent_entanglement(bio, fragment)

    def test_low_hrv_blocks_intent_entanglement(self):
        """Low HRV blocks intent entanglement."""
        bio = BioState(hrv=0.4, coherence=0.8, focus_level=0.8, pupil_dilation=0.6)
        fragment = FragmentID(namespace=1, index=42)
        assert not check_intent_entanglement(bio, fragment)


class TestWaterLoopbackGeneration:
    """Tests for main water loopback generation function."""

    def test_generates_valid_field(self):
        """Should generate valid water coherence field."""
        bio = BioState(hrv=0.7, coherence=0.8, focus_level=0.6, pupil_dilation=0.5)
        emotion = EmotionWaveform(
            valence=EmotionValence.PEACE,
            intensity=0.85,
            frequency=7.83,
            phase=0,
            duration_ticks=10
        )
        field = ScalarField(points=[], mean_alpha=0.8)

        water_field = generate_water_loopback(bio, emotion, field)

        assert validate_water_field(water_field)

    def test_high_coherence_produces_loopback(self):
        """High coherence should produce Loopback state."""
        bio = BioState(hrv=0.9, coherence=0.9, focus_level=0.8, pupil_dilation=0.6)
        emotion = EmotionWaveform(
            valence=EmotionValence.AWE,
            intensity=0.9,
            frequency=7.83,
            phase=0,
            duration_ticks=12
        )
        field = ScalarField(points=[], mean_alpha=0.9)

        water_field = generate_water_loopback(bio, emotion, field)

        assert water_field.orme_carrier_state == ORMEPhase.LOOPBACK

    def test_imprint_stored_when_conditions_met(self):
        """Emotion imprint should be stored when conditions met."""
        bio = BioState(hrv=0.8, coherence=0.8, focus_level=0.7, pupil_dilation=0.5)
        emotion = EmotionWaveform(
            valence=EmotionValence.JOY,
            intensity=0.9,
            frequency=10.0,
            phase=0,
            duration_ticks=10
        )
        field = ScalarField(points=[], mean_alpha=0.85)

        water_field = generate_water_loopback(bio, emotion, field)

        assert water_field.imprint_signature is not None
        assert water_field.imprint_signature.valence == EmotionValence.JOY

    def test_no_imprint_on_low_intensity(self):
        """No imprint should be stored with low intensity."""
        bio = BioState(hrv=0.8, coherence=0.8, focus_level=0.7, pupil_dilation=0.5)
        emotion = EmotionWaveform(
            valence=EmotionValence.JOY,
            intensity=0.5,  # Below threshold
            frequency=10.0,
            phase=0,
            duration_ticks=10
        )
        field = ScalarField(points=[], mean_alpha=0.85)

        water_field = generate_water_loopback(bio, emotion, field)

        assert water_field.imprint_signature is None

    def test_uses_local_alpha_from_nearest_point(self):
        """Should use alpha from nearest field point."""
        bio = BioState(hrv=0.7, coherence=0.7, focus_level=0.6, pupil_dilation=0.5)
        emotion = EmotionWaveform(
            valence=EmotionValence.PEACE,
            intensity=0.8,
            frequency=7.83,
            phase=0,
            duration_ticks=10
        )
        points = [
            ScalarFieldPoint(RaCoordinate(0, 0, 0), 0.95),
            ScalarFieldPoint(RaCoordinate(10, 10, 10), 0.3),
        ]
        field = ScalarField(points=points, mean_alpha=0.6)
        position = RaCoordinate(0.1, 0, 0)  # Closer to first point

        water_field = generate_water_loopback(bio, emotion, field, position)

        assert water_field.local_alpha == pytest.approx(0.95)


class TestLoopbackStrength:
    """Tests for loopback strength calculation."""

    def test_loopback_has_strength(self):
        """Loopback state should have positive strength."""
        field = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.LOOPBACK,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.9
        )
        strength = compute_loopback_strength(field)
        assert strength > 0

    def test_non_loopback_has_zero_strength(self):
        """Non-loopback states should have zero strength."""
        field = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.9
        )
        strength = compute_loopback_strength(field)
        assert strength == 0

    def test_imprint_amplifies_strength(self):
        """Emotion imprint should amplify loopback strength."""
        emotion = EmotionWaveform(
            valence=EmotionValence.AWE,
            intensity=0.9,
            frequency=7.83,
            phase=0,
            duration_ticks=10
        )

        field_no_imprint = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.LOOPBACK,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        field_with_imprint = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=emotion,
            orme_carrier_state=ORMEPhase.LOOPBACK,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )

        strength_no = compute_loopback_strength(field_no_imprint)
        strength_with = compute_loopback_strength(field_with_imprint)

        assert strength_with > strength_no

    def test_entanglement_amplifies_strength(self):
        """Memory entanglement should amplify loopback strength."""
        fragment = FragmentID(namespace=1, index=42)

        field_no_link = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.LOOPBACK,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        field_with_link = WaterCoherenceField(
            molecular_lattice=[[109.5]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.LOOPBACK,
            memory_entanglement=fragment,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )

        strength_no = compute_loopback_strength(field_no_link)
        strength_with = compute_loopback_strength(field_with_link)

        assert strength_with > strength_no


class TestFieldValidation:
    """Tests for water field validation."""

    def test_valid_field_passes(self):
        """Valid field should pass validation."""
        field = WaterCoherenceField(
            molecular_lattice=[[109.5, 110.0], [108.5, 109.0]],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        assert validate_water_field(field)

    def test_empty_lattice_fails(self):
        """Empty lattice should fail validation."""
        field = WaterCoherenceField(
            molecular_lattice=[],
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        assert not validate_water_field(field)

    def test_extreme_angles_fail(self):
        """Bond angles outside 90-130° should fail."""
        field = WaterCoherenceField(
            molecular_lattice=[[50.0]],  # Too low
            imprint_signature=None,
            orme_carrier_state=ORMEPhase.ENTANGLED,
            memory_entanglement=None,
            field_position=RaCoordinate(0, 0, 0),
            local_alpha=0.8
        )
        assert not validate_water_field(field)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
