"""
Prompt 21: Ra.Biotransduction Test Harness

Scalar-to-Somatic Transduction Engine.
Maps scalar harmonics, coherence levels, and chamber contexts
to real-time biophysical effects like warmth, pulses, and breath.

Codex References:
- Ra.Emergence: Field emergence modulation
- Ra.Coherence: Coherence-driven effect intensity
- P20: Chamber tuning integration
- P19: Domain safety for edge cases
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
from enum import Enum, auto


# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

PHI = 1.618033988749895
MIN_PHASE_GAIN = 0.1  # Clamped minimum to prevent effect blackout
COHERENCE_NULL_THRESHOLD = 0.2
COHERENCE_WEAK_THRESHOLD = 0.4
COHERENCE_FULL_THRESHOLD = 0.8


# ============================================================================
# Types - Basic
# ============================================================================

class BodyRegion(Enum):
    """Body regions for somatic targeting."""
    CHEST = auto()
    CROWN = auto()
    SPINE = auto()
    LEFT_HAND = auto()
    RIGHT_HAND = auto()
    FEET = auto()
    ABDOMEN = auto()


class Axis3D(Enum):
    """3D axes for proprioceptive effects."""
    X = auto()
    Y = auto()
    Z = auto()


class ChamberFlavor(Enum):
    """Unified chamber type (P20 + P21)."""
    HEALING = auto()
    RETRIEVAL = auto()
    NAVIGATION = auto()
    DREAM = auto()
    ARCHIVE = auto()
    THERAPEUTIC = auto()
    EXPLORATORY = auto()


# ============================================================================
# Types - Biophysical Effects
# ============================================================================

@dataclass
class ThermalShift:
    """Temperature change effect."""
    delta: float  # Celsius delta


@dataclass
class TactilePulse:
    """Tactile pulse at body region."""
    region: BodyRegion
    intensity: float  # 0.0-1.0


@dataclass
class ProprioceptiveWarp:
    """Stretch/tilt illusion along axis."""
    axis: Axis3D
    magnitude: float  # 0.0-1.0


@dataclass
class AffectiveOverlay:
    """Emotional valence and arousal overlay."""
    valence: float  # -1.0 to +1.0
    arousal: float  # 0.0 to 1.0


@dataclass
class BreathEntrainment:
    """Breath synchronization frequency."""
    frequency: float  # Hz


@dataclass
class InnerResonance:
    """Internal chime/resonance sensation."""
    harmonic: Tuple[int, int]  # (l, m)


@dataclass
class NullEffect:
    """No effect (coherence too low)."""
    pass


BiophysicalEffect = Union[
    ThermalShift,
    TactilePulse,
    ProprioceptiveWarp,
    AffectiveOverlay,
    BreathEntrainment,
    InnerResonance,
    NullEffect
]


# ============================================================================
# Types - Field & Context
# ============================================================================

@dataclass
class WindowPhase:
    """Phase window for modulation."""
    phi_depth: int
    phase: float  # 0.0-1.0


@dataclass
class ScalarField:
    """Scalar field parameters."""
    radius: float  # 0.0-1.0 (normalized)
    window_phase: WindowPhase


@dataclass
class RaCoordinate:
    """Ra coordinate with harmonic indices."""
    harmonic: Tuple[int, int]  # (l, m)
    angle: float  # radians


@dataclass
class EmergenceCondition:
    """Current emergence state."""
    flux: float
    shadow_threshold: float


@dataclass
class ResonanceProfile:
    """Resonance profile (placeholder)."""
    signature: str = "default"


# ============================================================================
# Coherence Interpolation
# ============================================================================

def smooth_interpolate(value: float, low: float, high: float) -> float:
    """
    Smooth interpolation using sine curve.

    Returns 0.0 below low, 1.0 above high, smooth transition between.
    """
    if value <= low:
        return 0.0
    if value >= high:
        return 1.0
    normalized = (value - low) / (high - low)
    # Sine interpolation for smooth onset
    return math.sin(normalized * math.pi / 2)


def coherence_envelope(coherence: float) -> Tuple[str, float]:
    """
    Get coherence envelope level and intensity.

    Returns (level_name, intensity_factor).
    """
    if coherence < COHERENCE_NULL_THRESHOLD:
        return ("null", 0.0)
    elif coherence < COHERENCE_WEAK_THRESHOLD:
        intensity = smooth_interpolate(coherence, 0.2, 0.4)
        return ("weak", intensity)
    elif coherence < COHERENCE_FULL_THRESHOLD:
        intensity = smooth_interpolate(coherence, 0.4, 0.8)
        return ("moderate", 0.5 + 0.5 * intensity)
    else:
        return ("full", 1.0)


# ============================================================================
# Phase Gain
# ============================================================================

def calculate_phase_gain(phase: float) -> float:
    """
    Calculate phase gain with clamped minimum.

    Uses sin(pi * phase) with minimum 0.1 to prevent blackout.
    """
    raw_gain = math.sin(math.pi * phase)
    return max(MIN_PHASE_GAIN, raw_gain)


# ============================================================================
# Body Region Mapping
# ============================================================================

def radius_to_region(radius: float) -> BodyRegion:
    """
    Map radial depth to body region.

    - 0.0: Crown (seed)
    - 0.3: Chest (heart)
    - 0.6: Abdomen
    - 1.0: Peripheral (feet)
    """
    if radius < 0.15:
        return BodyRegion.CROWN
    elif radius < 0.45:
        return BodyRegion.CHEST
    elif radius < 0.75:
        return BodyRegion.ABDOMEN
    else:
        return BodyRegion.FEET


def harmonic_to_axis(l: int) -> Optional[Axis3D]:
    """
    Map harmonic l to axis.

    - l=1: Y (vertical, breath)
    - l=2: X (bilateral, hands)
    - l=3+: Z (depth)
    """
    if l == 1:
        return Axis3D.Y
    elif l == 2:
        return Axis3D.X
    elif l >= 3:
        return Axis3D.Z
    return None


# ============================================================================
# Chamber Bias Effects
# ============================================================================

def get_chamber_effects(chamber: ChamberFlavor,
                        coherence: float) -> List[BiophysicalEffect]:
    """
    Get chamber-specific effect overlays.

    - Healing: warmth, breath
    - Retrieval: chills, cognitive pulse
    - Dream: proprioceptive warp
    - Archive: subtle harmonics
    """
    effects: List[BiophysicalEffect] = []

    if chamber == ChamberFlavor.HEALING or chamber == ChamberFlavor.THERAPEUTIC:
        effects.append(AffectiveOverlay(valence=0.8, arousal=coherence))

    elif chamber == ChamberFlavor.RETRIEVAL:
        effects.append(ThermalShift(delta=-1.0))
        effects.append(ProprioceptiveWarp(axis=Axis3D.Y, magnitude=0.3))

    elif chamber == ChamberFlavor.DREAM:
        effects.append(ProprioceptiveWarp(axis=Axis3D.Z, magnitude=0.4))

    elif chamber == ChamberFlavor.ARCHIVE:
        effects.append(InnerResonance(harmonic=(3, 0)))

    elif chamber == ChamberFlavor.NAVIGATION or chamber == ChamberFlavor.EXPLORATORY:
        effects.append(TactilePulse(region=BodyRegion.SPINE, intensity=0.3 * coherence))

    return effects


# ============================================================================
# Core Transduction Function
# ============================================================================

def transduce_field(
    field: ScalarField,
    coord: RaCoordinate,
    emergence: EmergenceCondition,
    coherence: float,
    profile: ResonanceProfile,
    chamber: ChamberFlavor
) -> List[BiophysicalEffect]:
    """
    Transduce scalar field to biophysical effects.

    Combines radial depth (body locus) with harmonic (effect type),
    applies phase gain clamping and smooth coherence interpolation.

    Args:
        field: Scalar field parameters
        coord: Ra coordinate with harmonics
        emergence: Current emergence condition
        coherence: Coherence level (0.0-1.0)
        profile: Resonance profile
        chamber: Chamber flavor for bias

    Returns:
        List of biophysical effects
    """
    # Check coherence envelope
    level, intensity = coherence_envelope(coherence)
    if level == "null":
        return [NullEffect()]

    # Calculate phase gain with clamping
    phase_gain = calculate_phase_gain(field.window_phase.phase)

    # Get body region from radius
    region = radius_to_region(field.radius)

    # Get harmonic indices
    l, m = coord.harmonic

    # Build base effects from harmonic
    base_effects: List[BiophysicalEffect] = []

    if l == 0:
        # H_{0,0}: whole-body temperature
        delta = 2.0 * phase_gain * intensity
        base_effects.append(ThermalShift(delta=delta))

    elif l == 1:
        # H_{1,m}: vertical axis, breath
        freq = 0.1 + 0.3 * intensity
        base_effects.append(BreathEntrainment(frequency=freq))

    elif l == 2:
        # H_{2,m}: bilateral, hands
        pulse_intensity = 0.5 * intensity * phase_gain
        base_effects.append(TactilePulse(region=BodyRegion.LEFT_HAND,
                                         intensity=pulse_intensity))
        base_effects.append(TactilePulse(region=BodyRegion.RIGHT_HAND,
                                         intensity=pulse_intensity))

    else:
        # H_{3+}: localized pulses and inner resonance
        base_effects.append(InnerResonance(harmonic=(l, m)))
        base_effects.append(TactilePulse(region=region,
                                         intensity=0.3 * intensity))

    # Add chamber-specific overlays
    chamber_effects = get_chamber_effects(chamber, coherence)

    # Return all effects (independent, not merged)
    return base_effects + chamber_effects


# ============================================================================
# Test Suite
# ============================================================================

class TestPhaseGain:
    """Test phase gain calculation with clamping."""

    def test_max_gain_at_half(self):
        """Phase 0.5 gives maximum gain (1.0)."""
        gain = calculate_phase_gain(0.5)
        assert abs(gain - 1.0) < 0.01

    def test_min_gain_at_zero(self):
        """Phase 0.0 clamps to minimum (0.1)."""
        gain = calculate_phase_gain(0.0)
        assert abs(gain - 0.1) < 0.01

    def test_min_gain_at_one(self):
        """Phase 1.0 clamps to minimum (0.1)."""
        gain = calculate_phase_gain(1.0)
        assert abs(gain - 0.1) < 0.01

    def test_mid_phase_gain(self):
        """Phase 0.25 gives sqrt(2)/2 ~ 0.707."""
        gain = calculate_phase_gain(0.25)
        expected = math.sin(math.pi * 0.25)
        assert abs(gain - expected) < 0.01


class TestCoherenceEnvelope:
    """Test coherence envelope with smooth interpolation."""

    def test_null_low_coherence(self):
        """Very low coherence returns null."""
        level, intensity = coherence_envelope(0.1)
        assert level == "null"
        assert intensity == 0.0

    def test_weak_coherence(self):
        """Low-mid coherence returns weak."""
        level, intensity = coherence_envelope(0.3)
        assert level == "weak"
        assert 0.0 < intensity < 1.0

    def test_moderate_coherence(self):
        """Mid coherence returns moderate."""
        level, intensity = coherence_envelope(0.6)
        assert level == "moderate"
        assert 0.5 < intensity < 1.0

    def test_full_coherence(self):
        """High coherence returns full."""
        level, intensity = coherence_envelope(0.9)
        assert level == "full"
        assert intensity == 1.0

    def test_smooth_transition(self):
        """Transitions within bands are smooth."""
        # Test within weak band (0.2-0.4)
        prev_intensity = 0.0
        for c in [0.21, 0.25, 0.30, 0.35, 0.39]:
            _, intensity = coherence_envelope(c)
            assert intensity >= prev_intensity
            prev_intensity = intensity

        # Test within moderate band (0.4-0.8)
        prev_intensity = 0.5
        for c in [0.45, 0.55, 0.65, 0.75]:
            _, intensity = coherence_envelope(c)
            assert intensity >= prev_intensity
            prev_intensity = intensity


class TestBodyRegionMapping:
    """Test radial depth to body region mapping."""

    def test_crown_at_center(self):
        """Radius 0.0 maps to crown."""
        assert radius_to_region(0.0) == BodyRegion.CROWN

    def test_chest_at_mid_low(self):
        """Radius 0.3 maps to chest."""
        assert radius_to_region(0.3) == BodyRegion.CHEST

    def test_abdomen_at_mid(self):
        """Radius 0.6 maps to abdomen."""
        assert radius_to_region(0.6) == BodyRegion.ABDOMEN

    def test_feet_at_periphery(self):
        """Radius 1.0 maps to feet."""
        assert radius_to_region(1.0) == BodyRegion.FEET


class TestTransduceField:
    """Test core transduction function."""

    def test_null_effect_low_coherence(self):
        """Low coherence returns NullEffect."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.1,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        assert len(effects) == 1
        assert isinstance(effects[0], NullEffect)

    def test_thermal_shift_h00(self):
        """H_{0,0} with high coherence produces ThermalShift."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(0, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.9,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        thermal = [e for e in effects if isinstance(e, ThermalShift)]
        assert len(thermal) >= 1
        assert thermal[0].delta > 1.0

    def test_bilateral_tactile_h21(self):
        """H_{2,1} produces bilateral tactile pulses."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(2, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.7,
                                  ResonanceProfile(), ChamberFlavor.NAVIGATION)

        tactile = [e for e in effects if isinstance(e, TactilePulse)]
        regions = [t.region for t in tactile]
        assert BodyRegion.LEFT_HAND in regions
        assert BodyRegion.RIGHT_HAND in regions

    def test_breath_entrainment_h10(self):
        """H_{1,0} produces breath entrainment."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.6,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        breath = [e for e in effects if isinstance(e, BreathEntrainment)]
        assert len(breath) >= 1
        assert breath[0].frequency > 0.1

    def test_inner_resonance_h30(self):
        """H_{3+} produces inner resonance."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(3, 2), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.8,
                                  ResonanceProfile(), ChamberFlavor.ARCHIVE)

        resonance = [e for e in effects if isinstance(e, InnerResonance)]
        assert len(resonance) >= 1


class TestChamberEffects:
    """Test chamber-specific effect overlays."""

    def test_healing_affective(self):
        """Healing chamber adds affective overlay."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(0, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.7,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        affective = [e for e in effects if isinstance(e, AffectiveOverlay)]
        assert len(affective) >= 1
        assert affective[0].valence > 0

    def test_retrieval_chills(self):
        """Retrieval chamber adds chills (negative thermal)."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.6,
                                  ResonanceProfile(), ChamberFlavor.RETRIEVAL)

        thermal = [e for e in effects if isinstance(e, ThermalShift)]
        assert any(t.delta < 0 for t in thermal)

    def test_dream_proprioceptive(self):
        """Dream chamber adds Z-axis proprioceptive warp."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.6,
                                  ResonanceProfile(), ChamberFlavor.DREAM)

        proprio = [e for e in effects if isinstance(e, ProprioceptiveWarp)]
        assert any(p.axis == Axis3D.Z for p in proprio)

    def test_archive_resonance(self):
        """Archive chamber adds inner resonance."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.6,
                                  ResonanceProfile(), ChamberFlavor.ARCHIVE)

        resonance = [e for e in effects if isinstance(e, InnerResonance)]
        assert len(resonance) >= 1


class TestPhaseModulation:
    """Test phase gain modulation effects."""

    def test_max_phase_amplifies(self):
        """Phase 0.5 amplifies effect compared to 0.0."""
        coord = RaCoordinate(harmonic=(0, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        # Max phase
        field_max = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        effects_max = transduce_field(field_max, coord, emergence, 0.9,
                                      ResonanceProfile(), ChamberFlavor.HEALING)

        # Zero phase (clamped)
        field_zero = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.0))
        effects_zero = transduce_field(field_zero, coord, emergence, 0.9,
                                       ResonanceProfile(), ChamberFlavor.HEALING)

        # Both should have thermal, but max should be stronger
        thermal_max = [e for e in effects_max if isinstance(e, ThermalShift)]
        thermal_zero = [e for e in effects_zero if isinstance(e, ThermalShift)]

        assert thermal_max[0].delta > thermal_zero[0].delta

    def test_zero_phase_not_blackout(self):
        """Phase 0.0 still produces effects (clamped to 0.1)."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.0))
        coord = RaCoordinate(harmonic=(0, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.9,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        # Should still have thermal shift (not null)
        thermal = [e for e in effects if isinstance(e, ThermalShift)]
        assert len(thermal) >= 1
        assert thermal[0].delta > 0


class TestEffectIndependence:
    """Test that effects are returned independently."""

    def test_multiple_effects_returned(self):
        """Multiple independent effects are returned."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(2, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.8,
                                  ResonanceProfile(), ChamberFlavor.HEALING)

        # Should have tactile pulses + chamber overlay
        assert len(effects) >= 3

    def test_effects_not_merged(self):
        """Same-type effects are not merged."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(2, 1), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects = transduce_field(field, coord, emergence, 0.8,
                                  ResonanceProfile(), ChamberFlavor.NAVIGATION)

        # Should have left and right hand as separate effects
        tactile = [e for e in effects if isinstance(e, TactilePulse)]
        assert len(tactile) >= 2


class TestDeterminism:
    """Test reproducibility of transduction."""

    def test_same_input_same_output(self):
        """Same input produces same output."""
        field = ScalarField(radius=0.5, window_phase=WindowPhase(1, 0.5))
        coord = RaCoordinate(harmonic=(1, 0), angle=0.0)
        emergence = EmergenceCondition(flux=0.5, shadow_threshold=0.72)

        effects1 = transduce_field(field, coord, emergence, 0.7,
                                   ResonanceProfile(), ChamberFlavor.HEALING)
        effects2 = transduce_field(field, coord, emergence, 0.7,
                                   ResonanceProfile(), ChamberFlavor.HEALING)

        assert len(effects1) == len(effects2)
        for e1, e2 in zip(effects1, effects2):
            assert type(e1) == type(e2)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
