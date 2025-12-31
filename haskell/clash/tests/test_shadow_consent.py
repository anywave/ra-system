#!/usr/bin/env python3
"""
Prompt 12: Consent-Gated Shadow Harmonics - Python Test Harness

Tests the RaShadowConsent Clash module for:
- Shadow fragment consent gating
- Therapeutic feedback generation
- Session state tracking
- Safety alert triggers

Codex References:
- REICH_ORGONE_ACCUMULATOR.md: DOR/armoring detection
- KAALI_BECK_BIOELECTRICAL.md: Consent gating thresholds
"""

import json
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional, List, Tuple
from datetime import datetime


# ============================================================================
# Enums matching Clash module
# ============================================================================

class ShadowType(Enum):
    REPRESSED = 0      # Hidden/denied aspects
    PROJECTED = 1      # Externalized onto others
    ANCESTRAL = 2      # Inherited patterns
    COLLECTIVE = 3     # Archetypal/collective

class ConsentState(Enum):
    NONE = 0           # No therapeutic consent
    THERAPEUTIC = 1    # Active consent for shadow work
    WITHDRAWN = 2      # Previously consented, now withdrawn
    EMERGENCY = 3      # Emergency override active

class InversionState(Enum):
    NORMAL = 0         # Standard polarity
    INVERTED = 1       # Reversed field
    OSCILLATING = 2    # Between states
    CHAOTIC = 3        # Unstable

class GatingResult(Enum):
    ALLOW = 0          # Emergence permitted
    BLOCK_NO_CONSENT = 1
    BLOCK_NO_OPERATOR = 2
    BLOCK_LOW_COHERENCE = 3
    BLOCK_NO_OVERRIDE = 4

class FeedbackIntensity(Enum):
    GENTLE = 0         # Subtle cues
    MODERATE = 1       # Standard prompts
    FIRM = 2           # Strong guidance
    URGENT = 3         # Safety-critical


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ShadowFragment:
    """Shadow fragment schema matching Clash ShadowFragment type."""
    fragment_id: int = 0
    fragment_form: int = 0       # Spherical harmonic l value
    inversion: InversionState = InversionState.NORMAL
    alpha: float = 0.5           # Normalized 0.0-1.0
    consent_state: ConsentState = ConsentState.NONE
    shadow_type: ShadowType = ShadowType.REPRESSED
    requires_override: bool = False
    origin_fragment: int = 0     # Link to originating fragment
    harmonic_mismatch: float = 0.0  # Distance from dominant mode
    emotional_charge: float = 0.5   # 0.0-1.0

@dataclass
class SessionState:
    """Session state for shadow work tracking."""
    coherence: float = 0.5       # Current coherence level
    licensed_operator: bool = False
    override_active: bool = False
    override_source: str = ""
    override_reason: str = ""
    shadow_progress: float = 0.0  # Session progress 0.0-1.0
    resonance_delta: float = 0.0  # Change since last cycle
    cycle_count: int = 0

@dataclass
class TherapeuticFeedback:
    """Therapeutic feedback output."""
    grounding_prompt: str = ""
    context_prompt: str = ""
    reflection_prompt: str = ""
    intensity: FeedbackIntensity = FeedbackIntensity.GENTLE
    audio_cue: str = "NONE"
    visual_glyph: str = "NONE"

@dataclass
class ShadowConsentOutput:
    """Complete output from shadow consent processing."""
    gating_result: GatingResult = GatingResult.BLOCK_NO_CONSENT
    emergence_allowed: bool = False
    feedback: TherapeuticFeedback = field(default_factory=TherapeuticFeedback)
    session_update: SessionState = field(default_factory=SessionState)
    safety_alert: bool = False
    cycle_count: int = 0


# ============================================================================
# ShadowModule - Core Processing Logic
# ============================================================================

class ShadowModule:
    """
    Consent-gated shadow fragment processing.

    Implements Ra.Gates consent gating with therapeutic feedback.
    """

    # Safety thresholds (matching Clash constants)
    COHERENCE_SAFE_ACCESS = 0.66    # Kaali/Beck threshold
    CHARGE_WARNING_THRESHOLD = 0.75
    COHERENCE_BOOST = 0.05
    COHERENCE_DECAY = 0.02
    PROGRESS_INCREMENT = 0.1

    @staticmethod
    def should_allow(fragment: ShadowFragment, session: SessionState) -> GatingResult:
        """
        Ra.Gates consent gating logic.

        Conditions for ALLOW:
        1. Consent state is THERAPEUTIC
        2. Licensed operator present
        3. Override active (if required)
        4. Coherence >= 0.66 (Kaali/Beck threshold)
        """
        # Condition 1: Must have therapeutic consent
        if fragment.consent_state != ConsentState.THERAPEUTIC:
            return GatingResult.BLOCK_NO_CONSENT

        # Condition 2: Must have licensed operator
        if not session.licensed_operator:
            return GatingResult.BLOCK_NO_OPERATOR

        # Condition 3: If override required, must be active
        if fragment.requires_override and not session.override_active:
            return GatingResult.BLOCK_NO_OVERRIDE

        # Condition 4: Coherence must be sufficient
        if session.coherence < ShadowModule.COHERENCE_SAFE_ACCESS:
            return GatingResult.BLOCK_LOW_COHERENCE

        return GatingResult.ALLOW

    @staticmethod
    def apply_override(session: SessionState, source: str, reason: str) -> SessionState:
        """Apply emergency override to session."""
        session.override_active = True
        session.override_source = source
        session.override_reason = reason
        return session

    @staticmethod
    def prompt(fragment: ShadowFragment, gating: GatingResult) -> TherapeuticFeedback:
        """
        Generate therapeutic feedback based on fragment and gating result.
        """
        feedback = TherapeuticFeedback()

        # Grounding prompt based on shadow type
        grounding_prompts = {
            ShadowType.REPRESSED: "Feel your feet on the ground. You are safe here.",
            ShadowType.PROJECTED: "Notice what you see in others. It may reflect within.",
            ShadowType.ANCESTRAL: "Honor what came before. You are not bound by it.",
            ShadowType.COLLECTIVE: "You are part of something larger. Breathe with it."
        }
        feedback.grounding_prompt = grounding_prompts.get(
            fragment.shadow_type,
            "Center yourself in this moment."
        )

        # Context prompt based on gating result
        context_prompts = {
            GatingResult.ALLOW: "This shadow aspect is ready for gentle exploration.",
            GatingResult.BLOCK_NO_CONSENT: "Therapeutic consent needed to proceed.",
            GatingResult.BLOCK_NO_OPERATOR: "Licensed guide required for this work.",
            GatingResult.BLOCK_LOW_COHERENCE: "Build coherence before shadow access.",
            GatingResult.BLOCK_NO_OVERRIDE: "Override authorization required."
        }
        feedback.context_prompt = context_prompts.get(
            gating,
            "Processing shadow fragment..."
        )

        # Reflection prompt - warning if high charge
        if fragment.emotional_charge >= ShadowModule.CHARGE_WARNING_THRESHOLD:
            feedback.reflection_prompt = "High emotional charge detected. Proceed gently."
            feedback.intensity = FeedbackIntensity.FIRM
        elif gating == GatingResult.ALLOW:
            feedback.reflection_prompt = "What does this shadow aspect teach you?"
            feedback.intensity = FeedbackIntensity.MODERATE
        else:
            feedback.reflection_prompt = "Preparation supports deeper work."
            feedback.intensity = FeedbackIntensity.GENTLE

        # Audio/visual cues
        if gating == GatingResult.ALLOW:
            feedback.audio_cue = "BINAURAL"
            feedback.visual_glyph = "FLOWER"
        elif fragment.emotional_charge >= ShadowModule.CHARGE_WARNING_THRESHOLD:
            feedback.audio_cue = "PULSE"
            feedback.visual_glyph = "SPIRAL"
        else:
            feedback.audio_cue = "TONE"
            feedback.visual_glyph = "MANDALA"

        return feedback

    @staticmethod
    def evaluate(fragment: ShadowFragment, session: SessionState) -> ShadowConsentOutput:
        """
        Complete shadow consent evaluation.

        Returns full output with gating, feedback, session update, and safety alerts.
        """
        output = ShadowConsentOutput()

        # Step 1: Consent gating
        output.gating_result = ShadowModule.should_allow(fragment, session)
        output.emergence_allowed = (output.gating_result == GatingResult.ALLOW)

        # Step 2: Generate feedback
        output.feedback = ShadowModule.prompt(fragment, output.gating_result)

        # Step 3: Update session state
        new_session = SessionState(
            coherence=session.coherence,
            licensed_operator=session.licensed_operator,
            override_active=session.override_active,
            override_source=session.override_source,
            override_reason=session.override_reason,
            shadow_progress=session.shadow_progress,
            resonance_delta=session.resonance_delta,
            cycle_count=session.cycle_count + 1
        )

        if output.emergence_allowed:
            # Coherence boost on successful shadow work
            coherence_boost = ShadowModule.COHERENCE_BOOST * (1.0 - fragment.emotional_charge)
            new_session.coherence = min(1.0, session.coherence + coherence_boost)
            new_session.shadow_progress = min(1.0, session.shadow_progress + ShadowModule.PROGRESS_INCREMENT)
            new_session.resonance_delta = coherence_boost
        else:
            # Slight decay when blocked
            new_session.coherence = max(0.0, session.coherence - ShadowModule.COHERENCE_DECAY)
            new_session.resonance_delta = -ShadowModule.COHERENCE_DECAY

        output.session_update = new_session
        output.cycle_count = new_session.cycle_count

        # Step 4: Safety alerts
        output.safety_alert = (
            fragment.emotional_charge >= ShadowModule.CHARGE_WARNING_THRESHOLD or
            session.coherence < 0.3 or
            fragment.inversion == InversionState.CHAOTIC
        )

        return output


# ============================================================================
# Test Scenarios
# ============================================================================

def test_therapeutic_consent_allow():
    """Test: Fragment with full therapeutic consent is allowed."""
    fragment = ShadowFragment(
        fragment_id=1,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.REPRESSED,
        emotional_charge=0.4
    )
    session = SessionState(
        coherence=0.75,
        licensed_operator=True
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.ALLOW, f"Expected ALLOW, got {output.gating_result}"
    assert output.emergence_allowed == True
    assert output.feedback.audio_cue == "BINAURAL"
    assert output.feedback.visual_glyph == "FLOWER"
    assert output.session_update.shadow_progress > 0
    print("  [PASS] therapeutic_consent_allow")
    return output

def test_no_consent_blocked():
    """Test: Fragment without consent is blocked."""
    fragment = ShadowFragment(
        fragment_id=2,
        consent_state=ConsentState.NONE,
        shadow_type=ShadowType.PROJECTED
    )
    session = SessionState(
        coherence=0.8,
        licensed_operator=True
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.BLOCK_NO_CONSENT
    assert output.emergence_allowed == False
    assert "consent" in output.feedback.context_prompt.lower()
    print("  [PASS] no_consent_blocked")
    return output

def test_no_operator_blocked():
    """Test: Fragment blocked without licensed operator."""
    fragment = ShadowFragment(
        fragment_id=3,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.ANCESTRAL
    )
    session = SessionState(
        coherence=0.7,
        licensed_operator=False
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.BLOCK_NO_OPERATOR
    assert output.emergence_allowed == False
    print("  [PASS] no_operator_blocked")
    return output

def test_low_coherence_blocked():
    """Test: Fragment blocked when coherence below threshold."""
    fragment = ShadowFragment(
        fragment_id=4,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.COLLECTIVE
    )
    session = SessionState(
        coherence=0.5,  # Below 0.66 threshold
        licensed_operator=True
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.BLOCK_LOW_COHERENCE
    assert output.emergence_allowed == False
    assert "coherence" in output.feedback.context_prompt.lower()
    print("  [PASS] low_coherence_blocked")
    return output

def test_override_required():
    """Test: Fragment requiring override is blocked without it."""
    fragment = ShadowFragment(
        fragment_id=5,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.REPRESSED,
        requires_override=True
    )
    session = SessionState(
        coherence=0.75,
        licensed_operator=True,
        override_active=False
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.BLOCK_NO_OVERRIDE
    assert output.emergence_allowed == False
    print("  [PASS] override_required_blocked")
    return output

def test_override_allows():
    """Test: Fragment with override active is allowed."""
    fragment = ShadowFragment(
        fragment_id=6,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.REPRESSED,
        requires_override=True
    )
    session = SessionState(
        coherence=0.75,
        licensed_operator=True
    )
    session = ShadowModule.apply_override(session, "THERAPIST", "Deep shadow work")

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.ALLOW
    assert output.emergence_allowed == True
    print("  [PASS] override_allows")
    return output

def test_high_charge_warning():
    """Test: High emotional charge triggers safety alert."""
    fragment = ShadowFragment(
        fragment_id=7,
        consent_state=ConsentState.THERAPEUTIC,
        shadow_type=ShadowType.PROJECTED,
        emotional_charge=0.85
    )
    session = SessionState(
        coherence=0.75,
        licensed_operator=True
    )

    output = ShadowModule.evaluate(fragment, session)

    assert output.gating_result == GatingResult.ALLOW
    assert output.safety_alert == True
    assert output.feedback.intensity == FeedbackIntensity.FIRM
    assert "charge" in output.feedback.reflection_prompt.lower()
    print("  [PASS] high_charge_warning")
    return output

def test_shadow_type_prompts():
    """Test: Different shadow types get appropriate prompts."""
    session = SessionState(coherence=0.75, licensed_operator=True)

    results = {}
    for shadow_type in ShadowType:
        fragment = ShadowFragment(
            fragment_id=10 + shadow_type.value,
            consent_state=ConsentState.THERAPEUTIC,
            shadow_type=shadow_type
        )
        output = ShadowModule.evaluate(fragment, session)
        results[shadow_type.name] = output.feedback.grounding_prompt

    # Each type should have unique grounding prompt
    assert len(set(results.values())) == len(ShadowType)
    print("  [PASS] shadow_type_prompts")
    return results


# ============================================================================
# CLI Dashboard
# ============================================================================

def print_dashboard(output: ShadowConsentOutput, fragment: ShadowFragment, cycle: int = 0):
    """Print CLI dashboard for shadow consent output."""
    width = 70

    print("=" * width)
    print(f" CONSENT-GATED SHADOW HARMONICS - Cycle {cycle}".center(width))
    print("=" * width)

    # Fragment info
    print("\n Shadow Fragment:")
    print(f"   ID: {fragment.fragment_id}")
    print(f"   Type: {fragment.shadow_type.name}")
    print(f"   Consent: {fragment.consent_state.name}")
    print(f"   Inversion: {fragment.inversion.name}")
    print(f"   Alpha: {fragment.alpha:.3f}")
    print(f"   Emotional Charge: {fragment.emotional_charge:.3f}")

    # Gating result
    print("\n Consent Gating:")
    result_icons = {
        GatingResult.ALLOW: "[ALLOW]",
        GatingResult.BLOCK_NO_CONSENT: "[BLOCK - NO CONSENT]",
        GatingResult.BLOCK_NO_OPERATOR: "[BLOCK - NO OPERATOR]",
        GatingResult.BLOCK_LOW_COHERENCE: "[BLOCK - LOW COHERENCE]",
        GatingResult.BLOCK_NO_OVERRIDE: "[BLOCK - NO OVERRIDE]"
    }
    print(f"   Result: {result_icons.get(output.gating_result, '[UNKNOWN]')}")
    print(f"   Emergence: {'PERMITTED' if output.emergence_allowed else 'BLOCKED'}")

    # Session state
    session = output.session_update
    print("\n Session State:")
    coh_bar = int(session.coherence * 30)
    prog_bar = int(session.shadow_progress * 30)
    print(f"   Coherence: [{'#' * coh_bar}{'-' * (30 - coh_bar)}] {session.coherence:.3f}")
    print(f"   Progress:  [{'#' * prog_bar}{'-' * (30 - prog_bar)}] {session.shadow_progress:.3f}")
    print(f"   Delta: {session.resonance_delta:+.3f}")
    print(f"   Operator: {'YES' if session.licensed_operator else 'NO'}")
    print(f"   Override: {'YES' if session.override_active else 'NO'}")

    # Therapeutic feedback
    feedback = output.feedback
    print("\n Therapeutic Feedback:")
    print(f"   Intensity: {feedback.intensity.name}")
    print(f"   Audio: {feedback.audio_cue}")
    print(f"   Glyph: {feedback.visual_glyph}")
    print(f"\n   Grounding: \"{feedback.grounding_prompt}\"")
    print(f"   Context: \"{feedback.context_prompt}\"")
    print(f"   Reflection: \"{feedback.reflection_prompt}\"")

    # Safety alert
    if output.safety_alert:
        print("\n" + "!" * width)
        print(" SAFETY ALERT ACTIVE".center(width))
        print("!" * width)

    print("=" * width)


def run_simulation(cycles: int = 5):
    """Run multi-cycle shadow consent simulation."""
    print("\n" + "=" * 70)
    print(" SHADOW CONSENT SIMULATION".center(70))
    print("=" * 70)

    # Start with building coherence
    session = SessionState(
        coherence=0.55,
        licensed_operator=True
    )

    fragments = [
        ShadowFragment(fragment_id=1, consent_state=ConsentState.THERAPEUTIC,
                      shadow_type=ShadowType.REPRESSED, emotional_charge=0.3),
        ShadowFragment(fragment_id=2, consent_state=ConsentState.THERAPEUTIC,
                      shadow_type=ShadowType.PROJECTED, emotional_charge=0.5),
        ShadowFragment(fragment_id=3, consent_state=ConsentState.THERAPEUTIC,
                      shadow_type=ShadowType.ANCESTRAL, emotional_charge=0.7),
        ShadowFragment(fragment_id=4, consent_state=ConsentState.THERAPEUTIC,
                      shadow_type=ShadowType.COLLECTIVE, emotional_charge=0.4),
        ShadowFragment(fragment_id=5, consent_state=ConsentState.THERAPEUTIC,
                      shadow_type=ShadowType.REPRESSED, emotional_charge=0.85),
    ]

    for i, fragment in enumerate(fragments[:cycles]):
        output = ShadowModule.evaluate(fragment, session)
        print_dashboard(output, fragment, i)
        session = output.session_update
        print()

    print(f"\nFinal Session State:")
    print(f"  Coherence: {session.coherence:.3f}")
    print(f"  Progress: {session.shadow_progress:.3f}")
    print(f"  Cycles: {session.cycle_count}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests and optional simulation."""
    print("\n" + "=" * 70)
    print(" PROMPT 12: CONSENT-GATED SHADOW HARMONICS - TEST SUITE".center(70))
    print("=" * 70)
    print()

    # Run tests
    tests = [
        ("Therapeutic Consent Allow", test_therapeutic_consent_allow),
        ("No Consent Blocked", test_no_consent_blocked),
        ("No Operator Blocked", test_no_operator_blocked),
        ("Low Coherence Blocked", test_low_coherence_blocked),
        ("Override Required Blocked", test_override_required),
        ("Override Allows", test_override_allows),
        ("High Charge Warning", test_high_charge_warning),
        ("Shadow Type Prompts", test_shadow_type_prompts),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print("\n" + "-" * 70)
    print(f" Results: {passed} passed, {failed} failed".center(70))
    print("-" * 70)

    # Run simulation if --sim flag
    if "--sim" in sys.argv:
        run_simulation()

    # Show sample dashboard
    if "--demo" in sys.argv or len(sys.argv) == 1:
        print("\n\n" + "=" * 70)
        print(" SAMPLE DASHBOARD OUTPUT".center(70))
        print("=" * 70)

        fragment = ShadowFragment(
            fragment_id=42,
            consent_state=ConsentState.THERAPEUTIC,
            shadow_type=ShadowType.ANCESTRAL,
            inversion=InversionState.NORMAL,
            alpha=0.65,
            emotional_charge=0.55
        )
        session = SessionState(
            coherence=0.72,
            licensed_operator=True,
            shadow_progress=0.3
        )
        output = ShadowModule.evaluate(fragment, session)
        print()
        print_dashboard(output, fragment, 3)

    # JSON output for integration
    if "--json" in sys.argv:
        fragment = ShadowFragment(
            fragment_id=1,
            consent_state=ConsentState.THERAPEUTIC,
            shadow_type=ShadowType.REPRESSED,
            emotional_charge=0.4
        )
        session = SessionState(coherence=0.75, licensed_operator=True)
        output = ShadowModule.evaluate(fragment, session)

        result = {
            "gating_result": output.gating_result.name,
            "emergence_allowed": output.emergence_allowed,
            "safety_alert": output.safety_alert,
            "feedback": {
                "intensity": output.feedback.intensity.name,
                "audio_cue": output.feedback.audio_cue,
                "visual_glyph": output.feedback.visual_glyph,
                "grounding_prompt": output.feedback.grounding_prompt,
                "context_prompt": output.feedback.context_prompt,
                "reflection_prompt": output.feedback.reflection_prompt
            },
            "session": {
                "coherence": output.session_update.coherence,
                "shadow_progress": output.session_update.shadow_progress,
                "resonance_delta": output.session_update.resonance_delta
            }
        }
        print(json.dumps(result, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
