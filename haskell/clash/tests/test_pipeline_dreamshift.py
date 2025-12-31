"""
Test Suite for Ra.Pipeline.DreamShift (P76)
Somniferous chamber shift mechanism for sleep state adaptation.

Tests biometric-driven chamber morphology changes, stress intervention,
and coherence protection during vulnerable dream phases.
"""

import pytest
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# Constants
PHI = 1.618033988749895
HRV_DROP_THRESHOLD = 0.15       # 15% drop triggers stress
ALPHA_COLLAPSE_THRESHOLD = 0.3  # α < 0.3 is collapse
STRESS_INTERVENTION_THRESHOLD = 0.5
MORPH_GRADIENT_RATE = 0.1      # Per tick


class ChamberForm(Enum):
    """Chamber morphology forms for dream states."""
    EGG = auto()              # Base protective form
    TOROID = auto()           # Intermediate flow form
    CADUCEUS_ALIGNED = auto() # High coherence aligned form
    BUFFER = auto()           # Emergency protection form
    SHELL = auto()            # Collapse fallback form


class SleepPhase(Enum):
    """Sleep cycle phases."""
    AWAKE = auto()
    NREM1 = auto()    # Light sleep
    NREM2 = auto()    # Sleep spindles
    NREM3 = auto()    # Deep sleep (slow wave)
    REM = auto()      # Rapid eye movement


class MorphEventType(Enum):
    """Types of chamber morph events."""
    GENTLE_MODULATION = auto()
    STRESS_INTERVENTION = auto()
    INVERSION_BUFFERING = auto()
    COLLAPSE_FALLBACK = auto()


@dataclass
class TimeRange:
    """Time range for sleep phases."""
    start_tick: int
    end_tick: int


@dataclass
class NocturnalCoherenceProfile:
    """Profile of nocturnal coherence patterns."""
    hrv_baseline: float        # Baseline HRV (0.0-1.0 normalized)
    rem_phases: List[TimeRange]
    dream_stress: float        # Current stress level (0.0-1.0)
    shift_gradient: float      # Morph rate factor


@dataclass
class BioStream:
    """Biometric data stream."""
    hrv: float                 # Current HRV (0.0-1.0)
    hrv_baseline: float        # Baseline for comparison
    alpha: float               # Scalar coherence
    breath_coherence: float    # Breathing pattern coherence
    sleep_phase: SleepPhase
    current_tick: int


@dataclass
class ChamberState:
    """Current chamber state."""
    form: ChamberForm
    alpha: float               # Current field coherence
    stability: float           # Form stability (0.0-1.0)
    morph_progress: float      # Progress in current morph (0.0-1.0)
    target_form: Optional[ChamberForm] = None


@dataclass
class SleepMorphEvent:
    """Chamber morph event during sleep."""
    event_type: MorphEventType
    target_form: ChamberForm
    urgency: float             # 0.0-1.0, how quickly to morph
    reason: str


def compute_hrv_drop(current_hrv: float, baseline_hrv: float) -> float:
    """Compute HRV drop as percentage of baseline."""
    if baseline_hrv <= 0:
        return 0.0
    return max(0.0, (baseline_hrv - current_hrv) / baseline_hrv)


def detect_alpha_collapse(alpha: float) -> bool:
    """Check if alpha has collapsed below threshold."""
    return alpha < ALPHA_COLLAPSE_THRESHOLD


def compute_dream_stress(
    hrv_drop: float,
    alpha: float,
    breath_coherence: float,
    sleep_phase: SleepPhase
) -> float:
    """
    Compute dream stress level.
    Stress = HRV drop factor + alpha collapse factor + breath incoherence
    """
    # HRV contribution (0-0.4)
    hrv_stress = min(0.4, hrv_drop * 2.0)

    # Alpha contribution (0-0.4)
    alpha_stress = 0.0
    if alpha < ALPHA_COLLAPSE_THRESHOLD:
        alpha_stress = 0.4
    elif alpha < 0.5:
        alpha_stress = 0.2

    # Breath coherence contribution (0-0.2)
    breath_stress = (1.0 - breath_coherence) * 0.2

    # REM is more vulnerable
    phase_factor = 1.2 if sleep_phase == SleepPhase.REM else 1.0

    total = (hrv_stress + alpha_stress + breath_stress) * phase_factor
    return min(1.0, total)


def select_chamber_form_for_alpha(alpha: float, current_form: ChamberForm) -> ChamberForm:
    """
    Select appropriate chamber form based on alpha level.
    Egg → Toroid → CaduceusAligned as α rises.
    Collapse fallback if α drops below floor.
    """
    if alpha < 0.2:
        return ChamberForm.SHELL  # Emergency fallback
    elif alpha < 0.4:
        return ChamberForm.BUFFER
    elif alpha < 0.6:
        return ChamberForm.EGG
    elif alpha < 0.8:
        return ChamberForm.TOROID
    else:
        return ChamberForm.CADUCEUS_ALIGNED


def compute_shift_gradient(
    bio: BioStream,
    target_form: ChamberForm,
    current_form: ChamberForm
) -> float:
    """
    Compute shift gradient (morph rate).
    Biometric-locked "breathe with user" entrainment.
    """
    # Base rate from breath coherence
    base_rate = bio.breath_coherence * MORPH_GRADIENT_RATE

    # Emergency morph faster
    if target_form in [ChamberForm.SHELL, ChamberForm.BUFFER]:
        base_rate *= 2.0

    # Higher HRV = smoother transition
    hrv_factor = 0.5 + bio.hrv * 0.5

    return min(1.0, base_rate * hrv_factor)


def generate_sleep_shift(
    bio: BioStream,
    chamber: ChamberState
) -> Tuple[ChamberState, SleepMorphEvent]:
    """
    Generate chamber shift based on biometric state.
    Returns updated chamber state and morph event.
    """
    # Compute stress metrics
    hrv_drop = compute_hrv_drop(bio.hrv, bio.hrv_baseline)
    dream_stress = compute_dream_stress(
        hrv_drop, bio.alpha, bio.breath_coherence, bio.sleep_phase
    )

    # Determine target form based on alpha
    target_form = select_chamber_form_for_alpha(bio.alpha, chamber.form)

    # Determine event type
    if hrv_drop > HRV_DROP_THRESHOLD and bio.alpha < ALPHA_COLLAPSE_THRESHOLD:
        event_type = MorphEventType.STRESS_INTERVENTION
        urgency = 0.9
        reason = "HRV drop + alpha collapse"
    elif dream_stress > STRESS_INTERVENTION_THRESHOLD:
        event_type = MorphEventType.STRESS_INTERVENTION
        urgency = dream_stress
        reason = f"Dream stress: {dream_stress:.2f}"
    elif target_form != chamber.form:
        event_type = MorphEventType.GENTLE_MODULATION
        urgency = 0.3
        reason = f"Alpha shift to {target_form.name}"
    else:
        event_type = MorphEventType.GENTLE_MODULATION
        urgency = 0.1
        reason = "Maintain current form"

    # Compute shift gradient
    gradient = compute_shift_gradient(bio, target_form, chamber.form)

    # Update morph progress
    new_progress = chamber.morph_progress
    if target_form != chamber.form:
        new_progress = min(1.0, chamber.morph_progress + gradient)

    # Complete morph if progress reaches 1.0
    new_form = chamber.form
    if new_progress >= 1.0:
        new_form = target_form
        new_progress = 0.0

    # Update stability
    new_stability = chamber.stability
    if event_type == MorphEventType.STRESS_INTERVENTION:
        new_stability = max(0.0, chamber.stability - 0.1)
    else:
        new_stability = min(1.0, chamber.stability + 0.05)

    # Create updated state
    new_chamber = ChamberState(
        form=new_form,
        alpha=bio.alpha,
        stability=new_stability,
        morph_progress=new_progress,
        target_form=target_form if target_form != new_form else None
    )

    # Create event
    event = SleepMorphEvent(
        event_type=event_type,
        target_form=target_form,
        urgency=urgency,
        reason=reason
    )

    return new_chamber, event


def is_in_rem_phase(tick: int, rem_phases: List[TimeRange]) -> bool:
    """Check if current tick is within a REM phase."""
    for phase in rem_phases:
        if phase.start_tick <= tick <= phase.end_tick:
            return True
    return False


def create_nocturnal_profile(
    hrv_baseline: float,
    rem_phases: List[Tuple[int, int]],
    bio: BioStream
) -> NocturnalCoherenceProfile:
    """Create nocturnal coherence profile."""
    hrv_drop = compute_hrv_drop(bio.hrv, hrv_baseline)
    stress = compute_dream_stress(
        hrv_drop, bio.alpha, bio.breath_coherence, bio.sleep_phase
    )
    gradient = compute_shift_gradient(
        bio, select_chamber_form_for_alpha(bio.alpha, ChamberForm.EGG), ChamberForm.EGG
    )

    return NocturnalCoherenceProfile(
        hrv_baseline=hrv_baseline,
        rem_phases=[TimeRange(s, e) for s, e in rem_phases],
        dream_stress=stress,
        shift_gradient=gradient
    )


# ============== TEST CLASSES ==============

class TestDreamStressComputation:
    """Tests for dream stress calculation."""

    def test_no_stress_healthy_metrics(self):
        """Test no stress with healthy biometrics."""
        stress = compute_dream_stress(
            hrv_drop=0.05,       # Minimal drop
            alpha=0.8,           # Good coherence
            breath_coherence=0.9,  # Good breathing
            sleep_phase=SleepPhase.NREM2
        )
        assert stress < 0.2

    def test_high_stress_hrv_drop(self):
        """Test high stress from HRV drop."""
        stress = compute_dream_stress(
            hrv_drop=0.25,       # Significant drop
            alpha=0.7,
            breath_coherence=0.8,
            sleep_phase=SleepPhase.NREM2
        )
        assert stress > 0.3

    def test_high_stress_alpha_collapse(self):
        """Test high stress from alpha collapse."""
        stress = compute_dream_stress(
            hrv_drop=0.05,
            alpha=0.2,           # Below collapse threshold
            breath_coherence=0.8,
            sleep_phase=SleepPhase.NREM2
        )
        assert stress > 0.3

    def test_rem_amplifies_stress(self):
        """Test REM phase amplifies stress."""
        stress_nrem = compute_dream_stress(
            hrv_drop=0.15,
            alpha=0.5,
            breath_coherence=0.7,
            sleep_phase=SleepPhase.NREM3
        )
        stress_rem = compute_dream_stress(
            hrv_drop=0.15,
            alpha=0.5,
            breath_coherence=0.7,
            sleep_phase=SleepPhase.REM
        )
        assert stress_rem > stress_nrem

    def test_stress_capped_at_one(self):
        """Test stress doesn't exceed 1.0."""
        stress = compute_dream_stress(
            hrv_drop=0.5,        # Major drop
            alpha=0.1,           # Severe collapse
            breath_coherence=0.2,  # Poor breathing
            sleep_phase=SleepPhase.REM
        )
        assert stress <= 1.0

    def test_breath_incoherence_adds_stress(self):
        """Test poor breath coherence adds stress."""
        stress_good = compute_dream_stress(
            hrv_drop=0.1,
            alpha=0.6,
            breath_coherence=0.9,
            sleep_phase=SleepPhase.NREM2
        )
        stress_bad = compute_dream_stress(
            hrv_drop=0.1,
            alpha=0.6,
            breath_coherence=0.3,
            sleep_phase=SleepPhase.NREM2
        )
        assert stress_bad > stress_good


class TestChamberFormSelection:
    """Tests for chamber form selection based on alpha."""

    def test_high_alpha_caduceus(self):
        """Test high alpha selects Caduceus."""
        form = select_chamber_form_for_alpha(0.9, ChamberForm.EGG)
        assert form == ChamberForm.CADUCEUS_ALIGNED

    def test_medium_high_alpha_toroid(self):
        """Test medium-high alpha selects Toroid."""
        form = select_chamber_form_for_alpha(0.7, ChamberForm.EGG)
        assert form == ChamberForm.TOROID

    def test_medium_alpha_egg(self):
        """Test medium alpha selects Egg."""
        form = select_chamber_form_for_alpha(0.5, ChamberForm.TOROID)
        assert form == ChamberForm.EGG

    def test_low_alpha_buffer(self):
        """Test low alpha selects Buffer."""
        form = select_chamber_form_for_alpha(0.3, ChamberForm.EGG)
        assert form == ChamberForm.BUFFER

    def test_very_low_alpha_shell(self):
        """Test very low alpha triggers Shell fallback."""
        form = select_chamber_form_for_alpha(0.15, ChamberForm.EGG)
        assert form == ChamberForm.SHELL

    def test_form_progression(self):
        """Test form progresses with increasing alpha."""
        forms = []
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            forms.append(select_chamber_form_for_alpha(alpha, ChamberForm.EGG))

        # Should progress through tiers
        assert forms[0] == ChamberForm.SHELL
        assert forms[1] == ChamberForm.BUFFER
        assert forms[2] == ChamberForm.EGG
        assert forms[3] == ChamberForm.TOROID
        assert forms[4] == ChamberForm.CADUCEUS_ALIGNED


class TestHRVDropDetection:
    """Tests for HRV drop detection."""

    def test_no_drop(self):
        """Test no drop detected when HRV matches baseline."""
        drop = compute_hrv_drop(0.8, 0.8)
        assert drop == pytest.approx(0.0, abs=0.001)

    def test_normal_drop(self):
        """Test normal HRV drop calculation."""
        drop = compute_hrv_drop(0.6, 0.8)
        assert drop == pytest.approx(0.25, abs=0.001)  # 25% drop

    def test_significant_drop(self):
        """Test significant HRV drop detection."""
        drop = compute_hrv_drop(0.5, 0.8)
        assert drop > HRV_DROP_THRESHOLD

    def test_zero_baseline_handling(self):
        """Test zero baseline doesn't cause division by zero."""
        drop = compute_hrv_drop(0.5, 0.0)
        assert drop == 0.0

    def test_hrv_increase_no_drop(self):
        """Test HRV increase shows no drop."""
        drop = compute_hrv_drop(0.9, 0.7)
        assert drop == 0.0


class TestAlphaCollapseDetection:
    """Tests for alpha collapse detection."""

    def test_collapse_below_threshold(self):
        """Test collapse detected below threshold."""
        assert detect_alpha_collapse(0.2) is True
        assert detect_alpha_collapse(0.29) is True

    def test_no_collapse_above_threshold(self):
        """Test no collapse above threshold."""
        assert detect_alpha_collapse(0.3) is False
        assert detect_alpha_collapse(0.5) is False
        assert detect_alpha_collapse(0.9) is False

    def test_threshold_boundary(self):
        """Test behavior at exact threshold."""
        assert detect_alpha_collapse(ALPHA_COLLAPSE_THRESHOLD) is False
        assert detect_alpha_collapse(ALPHA_COLLAPSE_THRESHOLD - 0.01) is True


class TestShiftGradient:
    """Tests for shift gradient computation."""

    def test_high_breath_coherence_faster(self):
        """Test high breath coherence increases gradient."""
        bio_high = BioStream(
            hrv=0.7, hrv_baseline=0.8, alpha=0.6,
            breath_coherence=0.9, sleep_phase=SleepPhase.NREM2, current_tick=100
        )
        bio_low = BioStream(
            hrv=0.7, hrv_baseline=0.8, alpha=0.6,
            breath_coherence=0.3, sleep_phase=SleepPhase.NREM2, current_tick=100
        )

        grad_high = compute_shift_gradient(bio_high, ChamberForm.TOROID, ChamberForm.EGG)
        grad_low = compute_shift_gradient(bio_low, ChamberForm.TOROID, ChamberForm.EGG)

        assert grad_high > grad_low

    def test_emergency_form_faster_morph(self):
        """Test emergency forms trigger faster morphing."""
        bio = BioStream(
            hrv=0.7, hrv_baseline=0.8, alpha=0.6,
            breath_coherence=0.7, sleep_phase=SleepPhase.NREM2, current_tick=100
        )

        grad_normal = compute_shift_gradient(bio, ChamberForm.TOROID, ChamberForm.EGG)
        grad_emergency = compute_shift_gradient(bio, ChamberForm.SHELL, ChamberForm.EGG)

        assert grad_emergency > grad_normal

    def test_gradient_capped(self):
        """Test gradient doesn't exceed 1.0."""
        bio = BioStream(
            hrv=1.0, hrv_baseline=1.0, alpha=0.9,
            breath_coherence=1.0, sleep_phase=SleepPhase.NREM2, current_tick=100
        )

        grad = compute_shift_gradient(bio, ChamberForm.SHELL, ChamberForm.EGG)
        assert grad <= 1.0


class TestSleepShiftGeneration:
    """Tests for sleep shift generation."""

    def test_gentle_modulation_stable_state(self):
        """Test gentle modulation in stable state."""
        bio = BioStream(
            hrv=0.75, hrv_baseline=0.8, alpha=0.7,
            breath_coherence=0.85, sleep_phase=SleepPhase.NREM2, current_tick=100
        )
        chamber = ChamberState(
            form=ChamberForm.TOROID, alpha=0.7,
            stability=0.8, morph_progress=0.0
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        assert event.event_type == MorphEventType.GENTLE_MODULATION
        assert event.urgency < 0.5

    def test_stress_intervention_hrv_collapse(self):
        """Test stress intervention on HRV drop + alpha collapse."""
        bio = BioStream(
            hrv=0.5, hrv_baseline=0.8, alpha=0.2,
            breath_coherence=0.5, sleep_phase=SleepPhase.REM, current_tick=100
        )
        chamber = ChamberState(
            form=ChamberForm.TOROID, alpha=0.7,
            stability=0.7, morph_progress=0.0
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        assert event.event_type == MorphEventType.STRESS_INTERVENTION
        assert event.urgency > 0.7
        assert "HRV" in event.reason or "stress" in event.reason.lower()

    def test_form_change_triggers_morph_progress(self):
        """Test form change initiates morph progress."""
        bio = BioStream(
            hrv=0.8, hrv_baseline=0.8, alpha=0.9,  # High alpha
            breath_coherence=0.9, sleep_phase=SleepPhase.NREM3, current_tick=100
        )
        chamber = ChamberState(
            form=ChamberForm.EGG, alpha=0.5,  # Was at lower alpha
            stability=0.8, morph_progress=0.0
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        # Should target higher form
        assert event.target_form == ChamberForm.CADUCEUS_ALIGNED
        # Progress should increase
        assert new_chamber.morph_progress > 0 or new_chamber.form != ChamberForm.EGG

    def test_stability_decreases_during_stress(self):
        """Test stability decreases during stress intervention."""
        bio = BioStream(
            hrv=0.4, hrv_baseline=0.8, alpha=0.15,
            breath_coherence=0.3, sleep_phase=SleepPhase.REM, current_tick=100
        )
        chamber = ChamberState(
            form=ChamberForm.TOROID, alpha=0.7,
            stability=0.8, morph_progress=0.0
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        assert new_chamber.stability < chamber.stability

    def test_stability_increases_during_gentle(self):
        """Test stability increases during gentle modulation."""
        bio = BioStream(
            hrv=0.75, hrv_baseline=0.8, alpha=0.7,
            breath_coherence=0.85, sleep_phase=SleepPhase.NREM3, current_tick=100
        )
        chamber = ChamberState(
            form=ChamberForm.TOROID, alpha=0.7,
            stability=0.7, morph_progress=0.0
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        if event.event_type == MorphEventType.GENTLE_MODULATION:
            assert new_chamber.stability >= chamber.stability


class TestNocturnalProfile:
    """Tests for nocturnal coherence profile."""

    def test_create_profile(self):
        """Test creating nocturnal profile."""
        bio = BioStream(
            hrv=0.7, hrv_baseline=0.8, alpha=0.6,
            breath_coherence=0.8, sleep_phase=SleepPhase.NREM2, current_tick=100
        )

        profile = create_nocturnal_profile(
            hrv_baseline=0.8,
            rem_phases=[(100, 150), (300, 350)],
            bio=bio
        )

        assert profile.hrv_baseline == 0.8
        assert len(profile.rem_phases) == 2
        assert profile.dream_stress >= 0
        assert profile.shift_gradient > 0

    def test_rem_phase_detection(self):
        """Test REM phase detection."""
        rem_phases = [TimeRange(100, 150), TimeRange(300, 350)]

        assert is_in_rem_phase(120, rem_phases) is True
        assert is_in_rem_phase(200, rem_phases) is False
        assert is_in_rem_phase(320, rem_phases) is True
        assert is_in_rem_phase(400, rem_phases) is False

    def test_empty_rem_phases(self):
        """Test handling empty REM phases."""
        assert is_in_rem_phase(100, []) is False


class TestSleepCycleIntegration:
    """Integration tests for full sleep cycles."""

    def test_full_sleep_cycle_transitions(self):
        """Test chamber transitions through a sleep cycle."""
        chamber = ChamberState(
            form=ChamberForm.EGG, alpha=0.5,
            stability=0.8, morph_progress=0.0
        )

        # Simulate sleep cycle phases
        phases = [
            (SleepPhase.NREM1, 0.6),
            (SleepPhase.NREM2, 0.65),
            (SleepPhase.NREM3, 0.7),
            (SleepPhase.REM, 0.6),
            (SleepPhase.NREM2, 0.55),
        ]

        events = []
        for phase, alpha in phases:
            bio = BioStream(
                hrv=0.75, hrv_baseline=0.8, alpha=alpha,
                breath_coherence=0.8, sleep_phase=phase, current_tick=100
            )
            chamber, event = generate_sleep_shift(bio, chamber)
            events.append(event)

        # Should have generated events for each phase
        assert len(events) == len(phases)
        # Most should be gentle modulations in this stable scenario
        gentle_count = sum(1 for e in events if e.event_type == MorphEventType.GENTLE_MODULATION)
        assert gentle_count >= 3

    def test_nightmare_stress_intervention(self):
        """Test stress intervention during nightmare scenario."""
        chamber = ChamberState(
            form=ChamberForm.TOROID, alpha=0.7,
            stability=0.85, morph_progress=0.0
        )

        # Nightmare scenario: REM with dropping metrics
        bio = BioStream(
            hrv=0.4,           # Significant drop
            hrv_baseline=0.8,
            alpha=0.25,        # Near collapse
            breath_coherence=0.4,
            sleep_phase=SleepPhase.REM,
            current_tick=200
        )

        new_chamber, event = generate_sleep_shift(bio, chamber)

        assert event.event_type == MorphEventType.STRESS_INTERVENTION
        # Should target protective form
        assert event.target_form in [ChamberForm.SHELL, ChamberForm.BUFFER]


class TestBiometricEntrainment:
    """Tests for biometric-locked entrainment."""

    def test_breath_locked_gradient(self):
        """Test gradient follows breath coherence."""
        bio_synced = BioStream(
            hrv=0.7, hrv_baseline=0.8, alpha=0.6,
            breath_coherence=1.0,  # Perfect sync
            sleep_phase=SleepPhase.NREM3, current_tick=100
        )

        grad = compute_shift_gradient(bio_synced, ChamberForm.TOROID, ChamberForm.EGG)

        # Should be near maximum for breath coherence
        assert grad > 0.05

    def test_hrv_smooths_transition(self):
        """Test higher HRV allows smoother transitions."""
        bio_high_hrv = BioStream(
            hrv=0.9, hrv_baseline=0.9, alpha=0.6,
            breath_coherence=0.7, sleep_phase=SleepPhase.NREM2, current_tick=100
        )
        bio_low_hrv = BioStream(
            hrv=0.4, hrv_baseline=0.9, alpha=0.6,
            breath_coherence=0.7, sleep_phase=SleepPhase.NREM2, current_tick=100
        )

        grad_high = compute_shift_gradient(bio_high_hrv, ChamberForm.TOROID, ChamberForm.EGG)
        grad_low = compute_shift_gradient(bio_low_hrv, ChamberForm.TOROID, ChamberForm.EGG)

        assert grad_high > grad_low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
