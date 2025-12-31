"""
Prompt 44: Ra.Chamber.Morphology - Resonance-Driven Chamber Form Modulation

Integrates live scalar dynamics, user resonance, and Codex-derived harmonic
thresholds to morph chambers in real time.

Codex References:
- KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Vibratory form principles
- GOLOD_PYRAMID_SPECIFICATIONS.md: Geometric resonance
- Ra.Chamber: Base chamber definitions
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Transition timing (φ-scaled ticks)
TRANSITION_SHORT = 13
TRANSITION_MEDIUM = 21
TRANSITION_LONG = 34

# Coherence thresholds
COLLAPSE_THRESHOLD = 0.20        # Below this triggers RapidCollapse
HRV_DELTA_THRESHOLD = 0.20       # 20% HRV change in <5 ticks
GRADUAL_THRESHOLD = 0.50         # Below this triggers GradualPhaseShift
STABLE_THRESHOLD = 0.72          # Above this is stable


# ============================================================================
# Types
# ============================================================================

class ChamberForm(Enum):
    """Morphogenic chamber forms."""
    EGG = auto()                  # Embryonic, nurturing
    DODECAHEDRON = auto()         # Cosmic harmony (12 faces)
    TOROID = auto()               # Flow circulation
    HELIX_SPINDLE = auto()        # DNA-like, evolutionary
    CADUCEUS_ALIGNED = auto()     # Kundalini, healing
    HARMONIC_SHELL_1 = auto()     # 1 nested shell
    HARMONIC_SHELL_2 = auto()     # 2 nested shells
    HARMONIC_SHELL_3 = auto()     # 3 nested shells
    HARMONIC_SHELL_4 = auto()     # 4 nested shells
    HARMONIC_SHELL_5 = auto()     # 5 nested shells
    HARMONIC_SHELL_6 = auto()     # 6 nested shells
    HARMONIC_SHELL_7 = auto()     # 7 nested shells
    HARMONIC_SHELL_8 = auto()     # 8 nested shells
    HARMONIC_SHELL_9 = auto()     # 9 nested shells


class MorphEventType(Enum):
    """Morphology transition event types."""
    GRADUAL_PHASE_SHIFT = auto()  # Smooth transition
    RAPID_COLLAPSE = auto()        # Emergency destabilization
    RESONANT_EXPANSION = auto()    # Growth event


@dataclass
class MorphEvent:
    """Morphology transition event."""
    event_type: MorphEventType
    target_form: Optional[ChamberForm]
    duration_ticks: int
    coherence_trigger: float


@dataclass
class HarmonicShell:
    """Nested harmonic shell configuration."""
    shell_index: int              # 1-9
    radial_order_l: int           # Spherical harmonic l
    angular_spin_m: int           # Spherical harmonic m (optional)
    radius_factor: float          # φ^index scaling


@dataclass
class BiometricSnapshot:
    """Current biometric state for form resolution."""
    coherence: float              # 0.0-1.0
    hrv_value: float              # Heart rate variability
    hrv_delta: float              # Change from previous
    ticks_since_change: int       # Ticks since significant change
    breath_phase: float           # 0.0-1.0 breath cycle


@dataclass
class ScalarResonance:
    """Scalar field resonance state."""
    field_strength: float         # 0.0-1.0
    dominant_harmonic: Tuple[int, int]  # (l, m)
    phase_angle: float            # radians


@dataclass
class ChamberMorphState:
    """Complete chamber morphology state."""
    current_form: ChamberForm
    target_form: Optional[ChamberForm]
    transition_progress: float    # 0.0-1.0
    transition_duration: int
    shells: List[HarmonicShell]
    stability_score: float


# ============================================================================
# Form Resolution
# ============================================================================

def get_shell_for_index(index: int) -> ChamberForm:
    """Get HarmonicShell form for shell index (1-9)."""
    shell_forms = {
        1: ChamberForm.HARMONIC_SHELL_1,
        2: ChamberForm.HARMONIC_SHELL_2,
        3: ChamberForm.HARMONIC_SHELL_3,
        4: ChamberForm.HARMONIC_SHELL_4,
        5: ChamberForm.HARMONIC_SHELL_5,
        6: ChamberForm.HARMONIC_SHELL_6,
        7: ChamberForm.HARMONIC_SHELL_7,
        8: ChamberForm.HARMONIC_SHELL_8,
        9: ChamberForm.HARMONIC_SHELL_9,
    }
    return shell_forms.get(index, ChamberForm.HARMONIC_SHELL_1)


def resolve_form_from_coherence(coherence: float) -> ChamberForm:
    """
    Resolve chamber form based on coherence level.

    Lower coherence → simpler forms (Egg)
    Higher coherence → complex forms (Dodecahedron, nested shells)
    """
    if coherence < 0.2:
        return ChamberForm.EGG  # Protective, minimal
    elif coherence < 0.4:
        return ChamberForm.TOROID  # Basic circulation
    elif coherence < 0.55:
        return ChamberForm.HELIX_SPINDLE  # Growth mode
    elif coherence < 0.72:
        return ChamberForm.CADUCEUS_ALIGNED  # Healing integration
    elif coherence < 0.85:
        return ChamberForm.DODECAHEDRON  # Cosmic harmony
    else:
        # High coherence: nested shells based on level
        shell_count = int((coherence - 0.85) / 0.015) + 1
        shell_count = min(9, max(1, shell_count))
        return get_shell_for_index(shell_count)


def resolve_form_from_harmonic(harmonic: Tuple[int, int]) -> ChamberForm:
    """
    Resolve chamber form based on dominant harmonic (l, m).

    l determines shell count/complexity.
    m influences angular symmetry.
    """
    l, m = harmonic

    if l == 0:
        return ChamberForm.EGG  # Spherically symmetric
    elif l == 1:
        return ChamberForm.HELIX_SPINDLE  # Dipole
    elif l == 2:
        return ChamberForm.TOROID  # Quadrupole
    elif l == 3:
        return ChamberForm.CADUCEUS_ALIGNED  # Octupole
    elif l <= 6:
        return ChamberForm.DODECAHEDRON
    else:
        # High l: nested shells
        shell_count = min(9, l - 5)
        return get_shell_for_index(shell_count)


def resolve_chamber_form(bio: BiometricSnapshot,
                         resonance: ScalarResonance) -> ChamberForm:
    """
    Resolve optimal chamber form from biometric and scalar state.

    Combines coherence-based and harmonic-based resolution.
    """
    coherence_form = resolve_form_from_coherence(bio.coherence)
    harmonic_form = resolve_form_from_harmonic(resonance.dominant_harmonic)

    # If coherence is high, prefer coherence-based form
    if bio.coherence >= STABLE_THRESHOLD:
        return coherence_form

    # If field strength is strong, prefer harmonic form
    if resonance.field_strength >= 0.7:
        return harmonic_form

    # Default to coherence-based
    return coherence_form


# ============================================================================
# Morph Event Generation
# ============================================================================

def should_collapse(bio: BiometricSnapshot) -> bool:
    """
    Check if rapid collapse should trigger.

    Triggers on:
    - Coherence < 0.2
    - HRV delta > 20% in < 5 ticks
    """
    if bio.coherence < COLLAPSE_THRESHOLD:
        return True

    if bio.hrv_delta > HRV_DELTA_THRESHOLD and bio.ticks_since_change < 5:
        return True

    return False


def get_transition_duration(current: ChamberForm, target: ChamberForm,
                            coherence: float) -> int:
    """
    Calculate transition duration (φ-scaled ticks).

    Higher coherence → faster transitions.
    More complex form changes → longer transitions.
    """
    # Base duration on coherence
    if coherence >= 0.8:
        base = TRANSITION_SHORT
    elif coherence >= 0.5:
        base = TRANSITION_MEDIUM
    else:
        base = TRANSITION_LONG

    # Adjust for form complexity change
    # (simplified: all forms have equal "distance")
    return base


def generate_morph_event(current_state: ChamberMorphState,
                         bio: BiometricSnapshot,
                         resonance: ScalarResonance) -> Optional[MorphEvent]:
    """
    Generate morphology transition event if needed.

    Returns None if no transition required.
    """
    # Check for collapse first
    if should_collapse(bio):
        return MorphEvent(
            event_type=MorphEventType.RAPID_COLLAPSE,
            target_form=ChamberForm.EGG,
            duration_ticks=5,  # Fast collapse
            coherence_trigger=bio.coherence
        )

    # Resolve target form
    target_form = resolve_chamber_form(bio, resonance)

    # No change needed
    if target_form == current_state.current_form:
        return None

    # Already transitioning to this form
    if target_form == current_state.target_form:
        return None

    # Check for expansion (increasing complexity)
    current_idx = list(ChamberForm).index(current_state.current_form)
    target_idx = list(ChamberForm).index(target_form)

    if target_idx > current_idx and bio.coherence >= STABLE_THRESHOLD:
        event_type = MorphEventType.RESONANT_EXPANSION
    else:
        event_type = MorphEventType.GRADUAL_PHASE_SHIFT

    duration = get_transition_duration(current_state.current_form, target_form, bio.coherence)

    return MorphEvent(
        event_type=event_type,
        target_form=target_form,
        duration_ticks=duration,
        coherence_trigger=bio.coherence
    )


# ============================================================================
# Harmonic Shell Management
# ============================================================================

def create_harmonic_shell(index: int) -> HarmonicShell:
    """
    Create harmonic shell configuration.

    Shell index correlates to radial order (l).
    Radius scales by φ^index.
    """
    return HarmonicShell(
        shell_index=index,
        radial_order_l=index,
        angular_spin_m=0,  # Default no spin
        radius_factor=PHI ** index
    )


def generate_shells_for_form(form: ChamberForm) -> List[HarmonicShell]:
    """Generate shell configuration for chamber form."""
    shell_forms = {
        ChamberForm.HARMONIC_SHELL_1: 1,
        ChamberForm.HARMONIC_SHELL_2: 2,
        ChamberForm.HARMONIC_SHELL_3: 3,
        ChamberForm.HARMONIC_SHELL_4: 4,
        ChamberForm.HARMONIC_SHELL_5: 5,
        ChamberForm.HARMONIC_SHELL_6: 6,
        ChamberForm.HARMONIC_SHELL_7: 7,
        ChamberForm.HARMONIC_SHELL_8: 8,
        ChamberForm.HARMONIC_SHELL_9: 9,
    }

    if form in shell_forms:
        count = shell_forms[form]
        return [create_harmonic_shell(i + 1) for i in range(count)]

    # Non-shell forms get single shell
    return [create_harmonic_shell(1)]


# ============================================================================
# State Management
# ============================================================================

def init_chamber_state(initial_form: ChamberForm = ChamberForm.EGG) -> ChamberMorphState:
    """Initialize chamber morphology state."""
    return ChamberMorphState(
        current_form=initial_form,
        target_form=None,
        transition_progress=0.0,
        transition_duration=0,
        shells=generate_shells_for_form(initial_form),
        stability_score=0.5
    )


def update_transition(state: ChamberMorphState) -> ChamberMorphState:
    """Update transition progress by one tick."""
    if state.target_form is None or state.transition_duration == 0:
        return state

    new_progress = state.transition_progress + (1.0 / state.transition_duration)

    if new_progress >= 1.0:
        # Transition complete
        return ChamberMorphState(
            current_form=state.target_form,
            target_form=None,
            transition_progress=0.0,
            transition_duration=0,
            shells=generate_shells_for_form(state.target_form),
            stability_score=state.stability_score
        )

    return ChamberMorphState(
        current_form=state.current_form,
        target_form=state.target_form,
        transition_progress=new_progress,
        transition_duration=state.transition_duration,
        shells=state.shells,
        stability_score=state.stability_score
    )


def apply_morph_event(state: ChamberMorphState, event: MorphEvent) -> ChamberMorphState:
    """Apply morph event to chamber state."""
    return ChamberMorphState(
        current_form=state.current_form,
        target_form=event.target_form,
        transition_progress=0.0,
        transition_duration=event.duration_ticks,
        shells=state.shells,
        stability_score=event.coherence_trigger
    )


def compute_stability(bio: BiometricSnapshot, resonance: ScalarResonance) -> float:
    """Compute overall stability score."""
    coherence_factor = bio.coherence
    field_factor = resonance.field_strength
    hrv_stability = max(0.0, 1.0 - abs(bio.hrv_delta) * 5)

    return (coherence_factor * 0.5 + field_factor * 0.3 + hrv_stability * 0.2)


# ============================================================================
# Test Suite
# ============================================================================

class TestChamberForm:
    """Test chamber form enumeration."""

    def test_all_forms_exist(self):
        """All expected forms exist."""
        assert ChamberForm.EGG
        assert ChamberForm.DODECAHEDRON
        assert ChamberForm.TOROID
        assert ChamberForm.HELIX_SPINDLE
        assert ChamberForm.CADUCEUS_ALIGNED

    def test_harmonic_shells_1_to_9(self):
        """9 harmonic shell variants exist."""
        for i in range(1, 10):
            form = get_shell_for_index(i)
            assert form is not None


class TestFormResolution:
    """Test form resolution logic."""

    def test_low_coherence_egg(self):
        """Low coherence produces Egg form."""
        form = resolve_form_from_coherence(0.1)
        assert form == ChamberForm.EGG

    def test_mid_coherence_toroid(self):
        """Mid-low coherence produces Toroid."""
        form = resolve_form_from_coherence(0.3)
        assert form == ChamberForm.TOROID

    def test_healing_coherence_caduceus(self):
        """Healing-level coherence produces Caduceus."""
        form = resolve_form_from_coherence(0.6)
        assert form == ChamberForm.CADUCEUS_ALIGNED

    def test_high_coherence_dodecahedron(self):
        """High coherence produces Dodecahedron."""
        form = resolve_form_from_coherence(0.75)
        assert form == ChamberForm.DODECAHEDRON

    def test_very_high_coherence_shells(self):
        """Very high coherence produces nested shells."""
        form = resolve_form_from_coherence(0.95)
        assert "HARMONIC_SHELL" in form.name

    def test_harmonic_l0_egg(self):
        """l=0 harmonic produces Egg."""
        form = resolve_form_from_harmonic((0, 0))
        assert form == ChamberForm.EGG

    def test_harmonic_high_l_shells(self):
        """High l harmonics produce nested shells."""
        form = resolve_form_from_harmonic((8, 0))
        assert "HARMONIC_SHELL" in form.name


class TestCollapseDetection:
    """Test collapse trigger detection."""

    def test_low_coherence_triggers_collapse(self):
        """Coherence < 0.2 triggers collapse."""
        bio = BiometricSnapshot(
            coherence=0.15,
            hrv_value=50.0,
            hrv_delta=0.05,
            ticks_since_change=10,
            breath_phase=0.5
        )
        assert should_collapse(bio) is True

    def test_hrv_spike_triggers_collapse(self):
        """HRV delta > 20% in < 5 ticks triggers collapse."""
        bio = BiometricSnapshot(
            coherence=0.5,
            hrv_value=50.0,
            hrv_delta=0.25,  # 25% change
            ticks_since_change=3,
            breath_phase=0.5
        )
        assert should_collapse(bio) is True

    def test_stable_no_collapse(self):
        """Stable biometrics don't trigger collapse."""
        bio = BiometricSnapshot(
            coherence=0.7,
            hrv_value=50.0,
            hrv_delta=0.05,
            ticks_since_change=20,
            breath_phase=0.5
        )
        assert should_collapse(bio) is False


class TestTransitionTiming:
    """Test φ-scaled transition timing."""

    def test_high_coherence_fast(self):
        """High coherence produces short transitions."""
        duration = get_transition_duration(
            ChamberForm.EGG, ChamberForm.TOROID, 0.9)
        assert duration == TRANSITION_SHORT

    def test_mid_coherence_medium(self):
        """Mid coherence produces medium transitions."""
        duration = get_transition_duration(
            ChamberForm.EGG, ChamberForm.TOROID, 0.6)
        assert duration == TRANSITION_MEDIUM

    def test_low_coherence_slow(self):
        """Low coherence produces long transitions."""
        duration = get_transition_duration(
            ChamberForm.EGG, ChamberForm.TOROID, 0.3)
        assert duration == TRANSITION_LONG


class TestMorphEventGeneration:
    """Test morph event generation."""

    def test_collapse_event_on_low_coherence(self):
        """Collapse event generated for low coherence."""
        state = init_chamber_state(ChamberForm.DODECAHEDRON)
        bio = BiometricSnapshot(0.1, 50.0, 0.05, 10, 0.5)
        resonance = ScalarResonance(0.5, (2, 0), 0.0)

        event = generate_morph_event(state, bio, resonance)

        assert event is not None
        assert event.event_type == MorphEventType.RAPID_COLLAPSE
        assert event.target_form == ChamberForm.EGG

    def test_no_event_when_stable(self):
        """No event when already in optimal form."""
        state = init_chamber_state(ChamberForm.CADUCEUS_ALIGNED)
        bio = BiometricSnapshot(0.6, 50.0, 0.02, 20, 0.5)
        resonance = ScalarResonance(0.5, (3, 0), 0.0)

        event = generate_morph_event(state, bio, resonance)

        # Form matches coherence level, no change needed
        assert event is None

    def test_expansion_event_high_coherence(self):
        """Expansion event when coherence increases."""
        state = init_chamber_state(ChamberForm.TOROID)
        bio = BiometricSnapshot(0.85, 50.0, 0.02, 20, 0.5)
        resonance = ScalarResonance(0.8, (5, 0), 0.0)

        event = generate_morph_event(state, bio, resonance)

        assert event is not None
        assert event.event_type == MorphEventType.RESONANT_EXPANSION


class TestHarmonicShells:
    """Test harmonic shell generation."""

    def test_shell_index_matches(self):
        """Shell index matches creation parameter."""
        shell = create_harmonic_shell(5)
        assert shell.shell_index == 5

    def test_shell_radial_order(self):
        """Radial order l matches shell index."""
        shell = create_harmonic_shell(3)
        assert shell.radial_order_l == 3

    def test_shell_radius_phi_scaled(self):
        """Shell radius scales by φ^index."""
        shell = create_harmonic_shell(2)
        expected = PHI ** 2
        assert abs(shell.radius_factor - expected) < 0.001

    def test_shells_for_shell_form(self):
        """Shell form generates correct number of shells."""
        shells = generate_shells_for_form(ChamberForm.HARMONIC_SHELL_5)
        assert len(shells) == 5

    def test_shells_for_basic_form(self):
        """Basic forms get single shell."""
        shells = generate_shells_for_form(ChamberForm.TOROID)
        assert len(shells) == 1


class TestStateManagement:
    """Test chamber state management."""

    def test_init_state(self):
        """Initial state has correct defaults."""
        state = init_chamber_state()
        assert state.current_form == ChamberForm.EGG
        assert state.target_form is None
        assert state.transition_progress == 0.0

    def test_transition_progress(self):
        """Transition progress updates correctly."""
        state = init_chamber_state()
        state = apply_morph_event(state, MorphEvent(
            MorphEventType.GRADUAL_PHASE_SHIFT,
            ChamberForm.TOROID,
            10,
            0.5
        ))

        assert state.target_form == ChamberForm.TOROID
        assert state.transition_duration == 10

        # Update 5 ticks
        for _ in range(5):
            state = update_transition(state)

        assert abs(state.transition_progress - 0.5) < 0.1

    def test_transition_completion(self):
        """Transition completes and updates form."""
        state = init_chamber_state()
        state = apply_morph_event(state, MorphEvent(
            MorphEventType.GRADUAL_PHASE_SHIFT,
            ChamberForm.TOROID,
            5,
            0.5
        ))

        # Complete transition
        for _ in range(10):
            state = update_transition(state)

        assert state.current_form == ChamberForm.TOROID
        assert state.target_form is None


class TestStability:
    """Test stability computation."""

    def test_high_coherence_high_stability(self):
        """High coherence produces high stability."""
        bio = BiometricSnapshot(0.9, 50.0, 0.01, 20, 0.5)
        resonance = ScalarResonance(0.8, (2, 0), 0.0)

        stability = compute_stability(bio, resonance)
        assert stability > 0.7

    def test_hrv_spike_lowers_stability(self):
        """HRV spike lowers stability."""
        bio_stable = BiometricSnapshot(0.7, 50.0, 0.01, 20, 0.5)
        bio_spike = BiometricSnapshot(0.7, 50.0, 0.3, 20, 0.5)
        resonance = ScalarResonance(0.5, (2, 0), 0.0)

        stability_stable = compute_stability(bio_stable, resonance)
        stability_spike = compute_stability(bio_spike, resonance)

        assert stability_spike < stability_stable


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
