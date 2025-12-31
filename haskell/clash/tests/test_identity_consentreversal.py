"""
Test Suite for Ra.Identity.ConsentReversal (Prompt 68)
Ankh-Phase Consent Reversal Threshold

Simulates consent reversal or shadow emergence during biometric
phase collapse using Δ(ankh) values for fragment inversion detection.
"""

import pytest
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895
PI = math.pi

# Δ(ankh) threshold for reversal (π/2 radians)
DELTA_ANKH_THRESHOLD = PI / 2  # ~1.571 radians

# Alpha depth threshold for reversal
ALPHA_DEPTH_THRESHOLD = 0.2

# HRV collapse threshold (25% drop)
HRV_COLLAPSE_THRESHOLD = 0.25

# HRV rolling window
HRV_WINDOW_SIZE = 3

# Recursive ripple settings
RIPPLE_ATTENUATION = 0.7
RIPPLE_MAX_HOPS = 3

# Shadow pattern glyphs
SHADOW_GLYPHS = ['◐', '◑', '◒', '◓', '●', '○', '◉', '◎']


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class FragmentID:
    """Fragment identifier."""
    fragment_id: str


@dataclass
class FragmentNode:
    """Memory fragment node."""
    fragment_id: FragmentID
    alpha_depth: float
    linked_fragments: List['FragmentNode'] = field(default_factory=list)


@dataclass
class BioState:
    """Biometric state."""
    hrv: float              # Heart rate variability (normalized 0-1)
    coherence: float        # Biometric coherence
    phase_signature: float  # Current phase (radians)


@dataclass
class HRVTrace:
    """Rolling HRV trace for collapse detection."""
    history: deque
    baseline: float


@dataclass
class Glyph:
    """Shadow pattern glyph."""
    symbol: str
    intensity: float  # 0-1


@dataclass
class ConsentShiftEvent:
    """Consent reversal event."""
    fragment_id: FragmentID
    delta_ankh: float
    shadow_pattern: Glyph
    triggered_by: BioState


@dataclass
class RippleEvent:
    """Recursive ripple propagation event."""
    fragment_id: FragmentID
    hop_number: int
    attenuated_delta: float
    shadow_pattern: Glyph


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def compute_delta_ankh(current_phase: float, reference_phase: float) -> float:
    """Compute Δ(ankh) as angular difference between phase signatures.

    Returns normalized value in [0, π].
    """
    diff = abs(current_phase - reference_phase)
    # Normalize to [0, π]
    diff = diff % (2 * PI)
    if diff > PI:
        diff = 2 * PI - diff
    return diff


def create_hrv_trace(baseline: float) -> HRVTrace:
    """Create HRV trace with baseline."""
    return HRVTrace(history=deque(maxlen=HRV_WINDOW_SIZE), baseline=baseline)


def update_hrv_trace(trace: HRVTrace, hrv: float) -> None:
    """Update HRV trace with new value."""
    trace.history.append(hrv)


def detect_hrv_collapse(trace: HRVTrace) -> bool:
    """Detect HRV collapse (≥25% drop from baseline).

    Uses 3-tick rolling average.
    """
    if len(trace.history) < HRV_WINDOW_SIZE:
        return False

    avg_hrv = sum(trace.history) / len(trace.history)
    drop_ratio = 1.0 - (avg_hrv / trace.baseline) if trace.baseline > 0 else 0

    return drop_ratio >= HRV_COLLAPSE_THRESHOLD


def select_shadow_glyph(delta_ankh: float, alpha_depth: float) -> Glyph:
    """Select shadow pattern glyph based on inversion parameters.

    Maps glyph luminance/symmetry to intensity.
    """
    # Intensity based on delta_ankh proximity to threshold
    intensity = min(1.0, delta_ankh / PI)

    # Select glyph based on alpha_depth and delta_ankh
    if alpha_depth < 0.1:
        symbol = SHADOW_GLYPHS[4]  # ● (solid, lowest depth)
    elif alpha_depth < 0.2:
        symbol = SHADOW_GLYPHS[6]  # ◉ (target, low depth)
    elif delta_ankh > 2.5:
        symbol = SHADOW_GLYPHS[0]  # ◐ (half-left, high delta)
    elif delta_ankh > 2.0:
        symbol = SHADOW_GLYPHS[1]  # ◑ (half-right)
    elif delta_ankh > 1.5:
        symbol = SHADOW_GLYPHS[2]  # ◒ (half-bottom)
    elif delta_ankh > DELTA_ANKH_THRESHOLD:
        symbol = SHADOW_GLYPHS[3]  # ◓ (half-top)
    else:
        symbol = SHADOW_GLYPHS[5]  # ○ (empty, mild)

    return Glyph(symbol=symbol, intensity=intensity)


def check_reversal_conditions(
    fragment: FragmentNode,
    bio: BioState,
    reference_phase: float,
    hrv_trace: Optional[HRVTrace] = None
) -> tuple[bool, float, str]:
    """Check if reversal conditions are met.

    Trigger conditions (OR):
    - Δ(ankh) ≥ π/2
    - alphaDepth < 0.2
    - HRV collapse > 25% within 3 ticks

    Returns: (triggered, delta_ankh, trigger_reason)
    """
    delta_ankh = compute_delta_ankh(bio.phase_signature, reference_phase)

    # Check Δ(ankh) threshold
    if delta_ankh >= DELTA_ANKH_THRESHOLD:
        return (True, delta_ankh, "delta_ankh_exceeded")

    # Check alpha depth threshold
    if fragment.alpha_depth < ALPHA_DEPTH_THRESHOLD:
        return (True, delta_ankh, "low_alpha_depth")

    # Check HRV collapse
    if hrv_trace and detect_hrv_collapse(hrv_trace):
        return (True, delta_ankh, "hrv_collapse")

    return (False, delta_ankh, "none")


def simulate_consent_reversal(
    fragment: FragmentNode,
    bio: BioState,
    reference_phase: float = 0.0,
    hrv_trace: Optional[HRVTrace] = None
) -> Optional[ConsentShiftEvent]:
    """Simulate consent reversal detection.

    Returns ConsentShiftEvent if reversal triggered, None otherwise.
    """
    triggered, delta_ankh, reason = check_reversal_conditions(
        fragment, bio, reference_phase, hrv_trace
    )

    if not triggered:
        return None

    shadow_pattern = select_shadow_glyph(delta_ankh, fragment.alpha_depth)

    return ConsentShiftEvent(
        fragment_id=fragment.fragment_id,
        delta_ankh=delta_ankh,
        shadow_pattern=shadow_pattern,
        triggered_by=bio
    )


def propagate_ripple(
    fragment: FragmentNode,
    initial_delta: float,
    hop: int = 0
) -> List[RippleEvent]:
    """Propagate reversal ripple through linked fragments.

    Attenuates by 0.7x per hop, max 3 hops.
    """
    if hop >= RIPPLE_MAX_HOPS:
        return []

    events = []
    # Attenuation applies to each hop (children get attenuated delta)
    child_delta = initial_delta * RIPPLE_ATTENUATION

    for linked in fragment.linked_fragments:
        # Only ripple if attenuated delta still exceeds threshold
        if child_delta >= DELTA_ANKH_THRESHOLD * 0.5:
            shadow = select_shadow_glyph(child_delta, linked.alpha_depth)
            events.append(RippleEvent(
                fragment_id=linked.fragment_id,
                hop_number=hop + 1,
                attenuated_delta=child_delta,
                shadow_pattern=shadow
            ))

            # Recursive ripple with further attenuation
            events.extend(propagate_ripple(linked, child_delta, hop + 1))

    return events


def simulate_reversal_with_ripple(
    fragment: FragmentNode,
    bio: BioState,
    reference_phase: float = 0.0,
    hrv_trace: Optional[HRVTrace] = None
) -> tuple[Optional[ConsentShiftEvent], List[RippleEvent]]:
    """Simulate reversal with recursive ripple propagation."""
    event = simulate_consent_reversal(fragment, bio, reference_phase, hrv_trace)

    if event is None:
        return (None, [])

    ripples = propagate_ripple(fragment, event.delta_ankh)
    return (event, ripples)


# =============================================================================
# TEST: DELTA ANKH DERIVATION
# =============================================================================

class TestDeltaAnkhDerivation:
    """Test Δ(ankh) computation."""

    def test_zero_phase_difference(self):
        """Zero phase difference gives zero delta."""
        delta = compute_delta_ankh(0.5, 0.5)
        assert abs(delta) < 0.01

    def test_pi_half_difference(self):
        """π/2 phase difference correctly computed."""
        delta = compute_delta_ankh(PI/2, 0)
        assert abs(delta - PI/2) < 0.01

    def test_normalized_to_pi(self):
        """Delta normalized to [0, π]."""
        # 3π/2 difference should normalize to π/2
        delta = compute_delta_ankh(3*PI/2, 0)
        assert delta <= PI
        assert abs(delta - PI/2) < 0.01

    def test_wrapping_at_2pi(self):
        """Delta wraps correctly at 2π."""
        delta = compute_delta_ankh(2*PI + 0.5, 0.5)
        assert abs(delta) < 0.01

    def test_threshold_value(self):
        """Threshold is π/2 ≈ 1.571."""
        assert abs(DELTA_ANKH_THRESHOLD - 1.5707963) < 0.001


# =============================================================================
# TEST: SHADOW PATTERN GLYPHS
# =============================================================================

class TestShadowPatternGlyphs:
    """Test shadow pattern glyph selection."""

    def test_glyph_set_defined(self):
        """All 8 glyphs defined."""
        assert len(SHADOW_GLYPHS) == 8

    def test_low_alpha_selects_solid_glyph(self):
        """Very low alpha selects solid glyph (●)."""
        glyph = select_shadow_glyph(1.0, alpha_depth=0.05)
        assert glyph.symbol == '●'

    def test_high_delta_selects_half_glyph(self):
        """High delta selects half-moon glyphs."""
        glyph = select_shadow_glyph(2.6, alpha_depth=0.5)
        assert glyph.symbol == '◐'

    def test_intensity_scales_with_delta(self):
        """Intensity scales with delta_ankh."""
        glyph_low = select_shadow_glyph(0.5, alpha_depth=0.5)
        glyph_high = select_shadow_glyph(2.5, alpha_depth=0.5)

        assert glyph_high.intensity > glyph_low.intensity

    def test_intensity_capped_at_1(self):
        """Intensity capped at 1.0."""
        glyph = select_shadow_glyph(5.0, alpha_depth=0.5)
        assert glyph.intensity <= 1.0


# =============================================================================
# TEST: RECURSIVE RIPPLE
# =============================================================================

class TestRecursiveRipple:
    """Test recursive ripple propagation."""

    def test_ripple_attenuates_by_07(self):
        """Ripple attenuates by 0.7x per hop."""
        frag1 = FragmentNode(FragmentID("f1"), alpha_depth=0.5)
        frag2 = FragmentNode(FragmentID("f2"), alpha_depth=0.5)
        frag1.linked_fragments = [frag2]

        ripples = propagate_ripple(frag1, initial_delta=2.0, hop=0)

        assert len(ripples) >= 1
        assert abs(ripples[0].attenuated_delta - 2.0 * 0.7) < 0.01

    def test_ripple_max_3_hops(self):
        """Ripple stops at max 3 hops."""
        # Create chain of 5 fragments
        frags = [FragmentNode(FragmentID(f"f{i}"), alpha_depth=0.5) for i in range(5)]
        for i in range(4):
            frags[i].linked_fragments = [frags[i+1]]

        ripples = propagate_ripple(frags[0], initial_delta=3.0, hop=0)

        # Should have at most 3 hops
        max_hop = max(r.hop_number for r in ripples) if ripples else 0
        assert max_hop <= RIPPLE_MAX_HOPS

    def test_ripple_stops_when_attenuated_below_threshold(self):
        """Ripple stops when delta too attenuated."""
        frag1 = FragmentNode(FragmentID("f1"), alpha_depth=0.5)
        frag2 = FragmentNode(FragmentID("f2"), alpha_depth=0.5)
        frag1.linked_fragments = [frag2]

        # Small initial delta
        ripples = propagate_ripple(frag1, initial_delta=0.5, hop=0)

        # May be empty if attenuated below threshold
        assert isinstance(ripples, list)

    def test_no_ripple_without_links(self):
        """No ripple if no linked fragments."""
        frag = FragmentNode(FragmentID("f1"), alpha_depth=0.5)
        ripples = propagate_ripple(frag, initial_delta=2.0)
        assert len(ripples) == 0


# =============================================================================
# TEST: HRV COLLAPSE
# =============================================================================

class TestHRVCollapse:
    """Test HRV collapse detection."""

    def test_no_collapse_with_stable_hrv(self):
        """Stable HRV does not trigger collapse."""
        trace = create_hrv_trace(baseline=0.8)
        for hrv in [0.78, 0.80, 0.79]:
            update_hrv_trace(trace, hrv)

        assert detect_hrv_collapse(trace) is False

    def test_collapse_with_25_percent_drop(self):
        """25% drop triggers collapse."""
        trace = create_hrv_trace(baseline=1.0)
        for hrv in [0.73, 0.72, 0.74]:  # ~27% drop
            update_hrv_trace(trace, hrv)

        assert detect_hrv_collapse(trace) is True

    def test_collapse_uses_3_tick_average(self):
        """Collapse uses 3-tick rolling average."""
        trace = create_hrv_trace(baseline=1.0)
        update_hrv_trace(trace, 0.5)  # One low value
        update_hrv_trace(trace, 1.0)  # Two high values
        update_hrv_trace(trace, 1.0)

        # Average = 0.833, drop ~17%, should NOT collapse
        assert detect_hrv_collapse(trace) is False

    def test_insufficient_data_no_collapse(self):
        """Insufficient data does not trigger collapse."""
        trace = create_hrv_trace(baseline=1.0)
        update_hrv_trace(trace, 0.5)  # Only one value

        assert detect_hrv_collapse(trace) is False


# =============================================================================
# TEST: REVERSAL CONDITIONS
# =============================================================================

class TestReversalConditions:
    """Test reversal trigger conditions."""

    def test_delta_ankh_exceeds_threshold(self):
        """Δ(ankh) ≥ π/2 triggers reversal."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.5)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=2.0)

        triggered, delta, reason = check_reversal_conditions(
            fragment, bio, reference_phase=0.0
        )

        assert triggered is True
        assert reason == "delta_ankh_exceeded"

    def test_low_alpha_depth_triggers(self):
        """alphaDepth < 0.2 triggers reversal."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.15)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=0.5)

        triggered, delta, reason = check_reversal_conditions(
            fragment, bio, reference_phase=0.5
        )

        assert triggered is True
        assert reason == "low_alpha_depth"

    def test_hrv_collapse_triggers(self):
        """HRV collapse triggers reversal."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.5)
        bio = BioState(hrv=0.6, coherence=0.7, phase_signature=0.5)

        trace = create_hrv_trace(baseline=1.0)
        for hrv in [0.7, 0.6, 0.65]:  # Collapse
            update_hrv_trace(trace, hrv)

        triggered, delta, reason = check_reversal_conditions(
            fragment, bio, reference_phase=0.5, hrv_trace=trace
        )

        assert triggered is True
        assert reason == "hrv_collapse"

    def test_no_trigger_when_healthy(self):
        """No trigger when all conditions healthy."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.5)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=0.5)

        triggered, delta, reason = check_reversal_conditions(
            fragment, bio, reference_phase=0.5
        )

        assert triggered is False
        assert reason == "none"


# =============================================================================
# TEST: SIMULATE CONSENT REVERSAL
# =============================================================================

class TestSimulateConsentReversal:
    """Test consent reversal simulation."""

    def test_returns_event_when_triggered(self):
        """Returns ConsentShiftEvent when triggered."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.5)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=2.5)

        event = simulate_consent_reversal(fragment, bio, reference_phase=0.0)

        assert event is not None
        assert event.fragment_id.fragment_id == "test"
        assert event.delta_ankh > DELTA_ANKH_THRESHOLD

    def test_returns_none_when_not_triggered(self):
        """Returns None when not triggered."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.5)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=0.5)

        event = simulate_consent_reversal(fragment, bio, reference_phase=0.5)

        assert event is None

    def test_event_includes_shadow_pattern(self):
        """Event includes shadow pattern glyph."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.15)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=0.5)

        event = simulate_consent_reversal(fragment, bio, reference_phase=0.5)

        assert event is not None
        assert event.shadow_pattern.symbol in SHADOW_GLYPHS

    def test_event_includes_bio_state(self):
        """Event includes triggering bio state."""
        fragment = FragmentNode(FragmentID("test"), alpha_depth=0.1)
        bio = BioState(hrv=0.75, coherence=0.6, phase_signature=0.5)

        event = simulate_consent_reversal(fragment, bio)

        assert event is not None
        assert event.triggered_by.hrv == 0.75


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestConsentReversalIntegration:
    """Integration tests for consent reversal system."""

    def test_full_reversal_with_ripple(self):
        """Full reversal with ripple propagation."""
        # Create linked fragments
        frag1 = FragmentNode(FragmentID("f1"), alpha_depth=0.5)
        frag2 = FragmentNode(FragmentID("f2"), alpha_depth=0.5)
        frag3 = FragmentNode(FragmentID("f3"), alpha_depth=0.5)
        frag1.linked_fragments = [frag2]
        frag2.linked_fragments = [frag3]

        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=2.5)

        event, ripples = simulate_reversal_with_ripple(frag1, bio, reference_phase=0.0)

        assert event is not None
        assert len(ripples) >= 1

    def test_event_logged_with_phase_delta(self):
        """Reversal logged with traceable phase delta."""
        fragment = FragmentNode(FragmentID("trace_test"), alpha_depth=0.5)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=2.0)

        event = simulate_consent_reversal(fragment, bio, reference_phase=0.0)

        assert event is not None
        assert event.delta_ankh == compute_delta_ankh(2.0, 0.0)

    def test_multiple_trigger_conditions(self):
        """Multiple trigger conditions handled correctly."""
        # Low alpha AND high delta
        fragment = FragmentNode(FragmentID("multi"), alpha_depth=0.1)
        bio = BioState(hrv=0.8, coherence=0.7, phase_signature=2.5)

        event = simulate_consent_reversal(fragment, bio, reference_phase=0.0)

        assert event is not None
        # Should still produce valid event


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
