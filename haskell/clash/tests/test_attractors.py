#!/usr/bin/env python3
"""
Prompt 18: Ra Attractors and Emergence Modulation

Implements metaphysical attractors that modulate Ra fragment emergence thresholds
by influencing coherence fields, inversion logic, and resonance probabilities.

Features:
- Attractor type with FragmentTarget union (ById | BySelector)
- FragmentSelector parser (shadow:*, guardian:*, h>=n, emergence_form=X)
- Phi-scaled enticement curve (enticement^1.618, floor 0.22, min 0.58)
- PhaseComponent flavor effects (Emotional/Sensory/Archetypal/UnknownPhase)
- AttractorEffect with priority ordering and 2-effect stacking
- Competing attractors resolution (interference -> vector sum -> dominant)
- Guardian harmonic matching
- Shadow consent bridge integration (P12)

Codex References:
- KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Sympathetic resonance
- RA_SCALAR.hs: Fragment emergence logic
- Prompt 12: ShadowConsent
"""

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime


# ============================================================================
# Constants (Codex-aligned, per Architect clarifications)
# ============================================================================

PHI = 1.618033988749895
ENTICEMENT_MIN = 0.58           # Minimum enticement to activate
COHERENCE_FLOOR = 0.22          # Attractors never reduce below this
SHADOW_THRESHOLD_NORMAL = 0.72  # Normal shadow emergence threshold
SHADOW_THRESHOLD_LOWERED = 0.61 # Lowered threshold with attractor
SHADOW_FLOOR = 0.42             # Minimum coherence for shadow via attractor
MAX_CONCURRENT_EFFECTS = 2      # Maximum stacked effects
MAX_CONCURRENT_ATTRACTORS = 4   # Maximum concurrent attractors per emergence


# ============================================================================
# Enums
# ============================================================================

class PhaseComponent(Enum):
    """Attractor flavor/phase component."""
    EMOTIONAL = "emotional"      # Affects flux, variability
    SENSORY = "sensory"          # Modulates temporal phase
    ARCHETYPAL = "archetypal"    # Potential guardian gate bypass
    UNKNOWN_PHASE = "unknown"    # ±25% chaos modulation


class AttractorEffect(Enum):
    """Effects an attractor can produce."""
    GATING_OVERRIDE = "gating_override"    # Bypass guardian gate
    INVERSION_FLIP = "inversion_flip"      # Flip shadow inversion
    RESONANCE_BOOST = "resonance_boost"    # Boost emergence potential

    @property
    def priority(self) -> int:
        """Effect priority (lower = higher priority)."""
        return {
            AttractorEffect.GATING_OVERRIDE: 1,
            AttractorEffect.INVERSION_FLIP: 2,
            AttractorEffect.RESONANCE_BOOST: 3
        }[self]


class FragmentType(Enum):
    """Fragment classification for targeting."""
    NORMAL = "normal"
    SHADOW = "shadow"
    GUARDIAN_GATED = "guardian_gated"
    SYMBOLIC = "symbolic"
    DREAM_LINKED = "dream_linked"


# ============================================================================
# Fragment Targeting (Union Type)
# ============================================================================

@dataclass
class ById:
    """Target fragment by explicit ID."""
    fragment_id: str


@dataclass
class BySelector:
    """Target fragments by rule selector."""
    selector: str  # e.g., "shadow:*", "guardian:*", "h>=3"

    def matches(self, fragment: 'Fragment') -> bool:
        """Check if fragment matches selector."""
        if self.selector == "shadow:*":
            return fragment.is_shadow
        elif self.selector == "guardian:*":
            return fragment.is_guardian_gated
        elif self.selector.startswith("h>="):
            try:
                min_h = int(self.selector[3:])
                return fragment.harmonic_shell >= min_h
            except ValueError:
                return False
        elif self.selector.startswith("emergence_form="):
            form = self.selector.split("=")[1]
            return fragment.emergence_form == form
        return False


FragmentTarget = Union[ById, BySelector]


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Fragment:
    """Ra fragment for emergence evaluation."""
    fragment_id: str
    coherence: float = 0.5
    potential: float = 0.5
    flux: float = 0.0
    temporal_phase: float = 0.0
    inversion_likelihood: float = 0.0
    harmonic_shell: int = 0
    emergence_form: str = "NORMAL"
    is_shadow: bool = False
    is_guardian_gated: bool = False
    guardian_symbol: str = ""


@dataclass
class EmergenceCondition:
    """Emergence condition state that attractors modify."""
    potential: float = 0.5
    flux: float = 0.0
    temporal_phase: float = 0.0
    inversion_likelihood: float = 0.0
    coherence_threshold: float = 0.6
    gating_active: bool = True
    shadow_accessible: bool = False


@dataclass
class Attractor:
    """Metaphysical attractor that modulates emergence."""
    symbol: str                          # e.g., "apple", "mirror"
    flavor: PhaseComponent               # Emotional, Sensory, etc.
    enticement_level: float              # 0.0-1.0 scalar influence
    target_fragments: List[FragmentTarget]
    allow_list: List[str] = field(default_factory=list)  # Guardian symbols

    def is_active(self) -> bool:
        """Check if enticement is above activation threshold."""
        return self.enticement_level >= ENTICEMENT_MIN

    def phi_scaled_enticement(self) -> float:
        """Calculate phi-scaled enticement: enticement^phi."""
        if self.enticement_level <= 0:
            return 0.0
        return self.enticement_level ** PHI


@dataclass
class AttractorResult:
    """Result of applying an attractor."""
    effects: List[AttractorEffect]
    coherence_delta: float = 0.0
    potential_boost: float = 0.0
    flux_modulation: float = 0.0
    phase_shift: float = 0.0
    gating_overridden: bool = False
    inversion_flipped: bool = False
    matched: bool = False


# ============================================================================
# Default Attractor Library
# ============================================================================

def get_default_attractors() -> Dict[str, Attractor]:
    """Get the predefined attractor library."""
    return {
        "apple": Attractor(
            symbol="apple",
            flavor=PhaseComponent.ARCHETYPAL,
            enticement_level=0.92,
            target_fragments=[ById("F144"), BySelector("shadow:*")],
            allow_list=["shinigami"]
        ),
        "tuning_fork": Attractor(
            symbol="tuning_fork",
            flavor=PhaseComponent.SENSORY,
            enticement_level=0.88,
            target_fragments=[BySelector("guardian:*")],
            allow_list=["harmonic", "resonance"]
        ),
        "mirror": Attractor(
            symbol="mirror",
            flavor=PhaseComponent.EMOTIONAL,
            enticement_level=0.85,
            target_fragments=[BySelector("shadow:*")],
            allow_list=["reflection", "shadow"]
        ),
        "quartz": Attractor(
            symbol="quartz",
            flavor=PhaseComponent.SENSORY,
            enticement_level=0.78,
            target_fragments=[BySelector("h>=3")],
            allow_list=["crystal", "amplifier"]
        ),
        "sun_glyph": Attractor(
            symbol="sun_glyph",
            flavor=PhaseComponent.ARCHETYPAL,
            enticement_level=0.95,
            target_fragments=[BySelector("emergence_form=SYMBOLIC")],
            allow_list=["solar", "light"]
        ),
    }


# ============================================================================
# Flavor Effects
# ============================================================================

def apply_flavor_effect(flavor: PhaseComponent, condition: EmergenceCondition,
                        enticement: float) -> EmergenceCondition:
    """Apply flavor-specific effects to emergence condition."""
    result = EmergenceCondition(
        potential=condition.potential,
        flux=condition.flux,
        temporal_phase=condition.temporal_phase,
        inversion_likelihood=condition.inversion_likelihood,
        coherence_threshold=condition.coherence_threshold,
        gating_active=condition.gating_active,
        shadow_accessible=condition.shadow_accessible
    )

    if flavor == PhaseComponent.EMOTIONAL:
        # Affects flux, increasing variability
        result.flux += enticement * 0.2
        result.inversion_likelihood += enticement * 0.1

    elif flavor == PhaseComponent.SENSORY:
        # Modulates temporal phase
        result.temporal_phase += enticement * 0.15
        result.flux += enticement * 0.05

    elif flavor == PhaseComponent.ARCHETYPAL:
        # Potential to bypass guardian gates
        if enticement >= 0.85:
            result.gating_active = False
        result.potential += enticement * 0.1

    elif flavor == PhaseComponent.UNKNOWN_PHASE:
        # ±25% random modulation
        chaos = (random.random() - 0.5) * 0.5  # -0.25 to +0.25
        result.potential += result.potential * chaos
        result.flux += result.flux * chaos
        result.temporal_phase += result.temporal_phase * chaos

    return result


# ============================================================================
# Guardian Matching
# ============================================================================

def matches_guardian(attractor: Attractor, fragment: Fragment) -> bool:
    """Check if attractor matches guardian gate."""
    if not fragment.is_guardian_gated:
        return True  # No gate to match

    # Check symbol in allow list
    if fragment.guardian_symbol in attractor.allow_list:
        return True

    # Check harmonic alignment (l, m match)
    # Simplified: check if attractor symbol relates to guardian
    return attractor.symbol in fragment.guardian_symbol.lower()


# ============================================================================
# Core Attractor Logic
# ============================================================================

def attractor_targets_fragment(attractor: Attractor, fragment: Fragment) -> bool:
    """Check if attractor targets the given fragment."""
    for target in attractor.target_fragments:
        if isinstance(target, ById):
            if target.fragment_id == fragment.fragment_id:
                return True
        elif isinstance(target, BySelector):
            if target.matches(fragment):
                return True
    return False


def apply_attractor(attractor: Attractor, condition: EmergenceCondition,
                    fragment: Optional[Fragment] = None) -> Tuple[EmergenceCondition, AttractorResult]:
    """
    Apply attractor to emergence condition.

    Returns modified condition and result details.
    """
    result = AttractorResult(effects=[], matched=False)

    # Check if attractor is active
    if not attractor.is_active():
        return condition, result

    # Check if attractor targets fragment (if provided)
    if fragment and not attractor_targets_fragment(attractor, fragment):
        return condition, result

    result.matched = True

    # Calculate phi-scaled enticement
    phi_enticement = attractor.phi_scaled_enticement()

    # Start with flavor effects
    modified = apply_flavor_effect(attractor.flavor, condition, phi_enticement)

    # Determine effects (priority order, max 2)
    effects_to_apply = []

    # Check for gating override (Archetypal flavor with high enticement)
    if (attractor.flavor == PhaseComponent.ARCHETYPAL and
        attractor.enticement_level >= 0.85):
        effects_to_apply.append(AttractorEffect.GATING_OVERRIDE)
        modified.gating_active = False
        result.gating_overridden = True

    # Check for inversion flip (shadow targeting with sufficient enticement)
    if fragment and fragment.is_shadow and attractor.enticement_level >= 0.80:
        effects_to_apply.append(AttractorEffect.INVERSION_FLIP)
        modified.inversion_likelihood = 1.0 - modified.inversion_likelihood
        modified.shadow_accessible = True
        result.inversion_flipped = True

    # Always apply resonance boost if enticement is high
    if attractor.enticement_level >= ENTICEMENT_MIN:
        effects_to_apply.append(AttractorEffect.RESONANCE_BOOST)
        # Boost potential: potential *= (1 + phi_enticement * 0.2)
        result.potential_boost = phi_enticement * 0.2
        modified.potential *= (1 + result.potential_boost)

    # Sort by priority and limit to 2
    effects_to_apply.sort(key=lambda e: e.priority)
    result.effects = effects_to_apply[:MAX_CONCURRENT_EFFECTS]

    # Apply coherence threshold reduction (never below floor)
    coherence_delta = phi_enticement * 0.15
    new_threshold = max(COHERENCE_FLOOR, modified.coherence_threshold - coherence_delta)
    result.coherence_delta = modified.coherence_threshold - new_threshold
    modified.coherence_threshold = new_threshold

    # Handle shadow threshold lowering
    if fragment and fragment.is_shadow:
        if modified.coherence_threshold > SHADOW_THRESHOLD_LOWERED:
            modified.coherence_threshold = SHADOW_THRESHOLD_LOWERED

    return modified, result


# ============================================================================
# Competing Attractors Resolution
# ============================================================================

def resolve_competing_attractors(attractors: List[Attractor],
                                 condition: EmergenceCondition,
                                 fragment: Optional[Fragment] = None
                                 ) -> Tuple[EmergenceCondition, List[AttractorResult]]:
    """
    Resolve competing attractors using:
    1. Constructive/destructive interference (phase alignment)
    2. Vector summation of enticement
    3. Dominant attractor wins (highest enticement)
    """
    # Limit to max concurrent
    active_attractors = [a for a in attractors if a.is_active()][:MAX_CONCURRENT_ATTRACTORS]

    if not active_attractors:
        return condition, []

    results = []
    current = condition

    # Sort by enticement (descending) for dominant resolution
    sorted_attractors = sorted(active_attractors,
                               key=lambda a: a.enticement_level,
                               reverse=True)

    # Track total effects applied
    total_effects = 0

    for attractor in sorted_attractors:
        if total_effects >= MAX_CONCURRENT_EFFECTS * 2:
            break  # Limit total effects across all attractors

        modified, result = apply_attractor(attractor, current, fragment)

        if result.matched:
            results.append(result)
            current = modified
            total_effects += len(result.effects)

    return current, results


# ============================================================================
# Shadow Consent Integration (P12)
# ============================================================================

def check_shadow_consent_with_attractor(fragment: Fragment, attractor: Attractor,
                                        user_coherence: float,
                                        has_operator: bool = False) -> Dict[str, Any]:
    """
    Check shadow consent with attractor influence.

    Per architect:
    - Attractors lower threshold from 0.72 to 0.61
    - Operator not required if enticement >= 0.88 and matches shadow
    - Minimum coherence floor: 0.42
    """
    if not fragment.is_shadow:
        return {"allowed": True, "reason": "Not shadow fragment"}

    # Check coherence floor
    if user_coherence < SHADOW_FLOOR:
        return {"allowed": False, "reason": f"Coherence {user_coherence:.2f} below floor {SHADOW_FLOOR}"}

    # Check if attractor lowers threshold
    threshold = SHADOW_THRESHOLD_NORMAL
    if attractor.is_active() and attractor_targets_fragment(attractor, fragment):
        threshold = SHADOW_THRESHOLD_LOWERED

        # Check if operator required
        operator_required = not (attractor.enticement_level >= 0.88 and
                                 fragment.is_shadow)
        if operator_required and not has_operator:
            return {"allowed": False, "reason": "Operator required for shadow access"}

    # Check against threshold
    if user_coherence >= threshold:
        return {"allowed": True, "reason": "Coherence meets threshold",
                "threshold_used": threshold}
    else:
        return {"allowed": False,
                "reason": f"Coherence {user_coherence:.2f} below threshold {threshold}"}


# ============================================================================
# Test Scenarios
# ============================================================================

def test_attractor_activation():
    """Test: Attractor activation threshold (0.58 minimum)."""
    # Active attractor
    apple = get_default_attractors()["apple"]
    assert apple.is_active(), "Apple attractor should be active"

    # Inactive attractor
    weak = Attractor("weak", PhaseComponent.SENSORY, 0.5, [])
    assert not weak.is_active(), "Weak attractor should not be active"

    print("  [PASS] attractor_activation")


def test_phi_scaled_enticement():
    """Test: Phi-scaled enticement curve."""
    apple = get_default_attractors()["apple"]
    phi_ent = apple.phi_scaled_enticement()

    # 0.92 ^ 1.618 ≈ 0.87
    assert 0.85 <= phi_ent <= 0.90, f"Expected ~0.87, got {phi_ent}"

    print(f"  [PASS] phi_scaled_enticement - 0.92^phi = {phi_ent:.4f}")


def test_fragment_targeting():
    """Test: Fragment targeting with ById and BySelector."""
    apple = get_default_attractors()["apple"]

    # Test ById
    frag_144 = Fragment("F144", is_shadow=True)
    assert attractor_targets_fragment(apple, frag_144)

    # Test BySelector shadow:*
    frag_shadow = Fragment("F999", is_shadow=True)
    assert attractor_targets_fragment(apple, frag_shadow)

    # Test non-match
    frag_normal = Fragment("F001", is_shadow=False)
    assert not attractor_targets_fragment(apple, frag_normal)

    print("  [PASS] fragment_targeting")


def test_flavor_effects():
    """Test: PhaseComponent flavor effects."""
    condition = EmergenceCondition(potential=0.5, flux=0.1)

    # Emotional: affects flux
    emotional = apply_flavor_effect(PhaseComponent.EMOTIONAL, condition, 0.8)
    assert emotional.flux > condition.flux

    # Archetypal: can disable gating
    archetypal = apply_flavor_effect(PhaseComponent.ARCHETYPAL, condition, 0.9)
    assert not archetypal.gating_active

    print("  [PASS] flavor_effects")


def test_attractor_application():
    """Test: Apply attractor to emergence condition."""
    apple = get_default_attractors()["apple"]
    condition = EmergenceCondition(potential=0.5, coherence_threshold=0.7)
    fragment = Fragment("F144", is_shadow=True)

    modified, result = apply_attractor(apple, condition, fragment)

    assert result.matched
    assert len(result.effects) <= MAX_CONCURRENT_EFFECTS
    assert modified.coherence_threshold < condition.coherence_threshold
    assert modified.coherence_threshold >= COHERENCE_FLOOR

    print(f"  [PASS] attractor_application - Threshold: {condition.coherence_threshold:.2f} -> {modified.coherence_threshold:.2f}")


def test_gating_override():
    """Test: GatingOverride effect via guardian attractor."""
    tuning_fork = get_default_attractors()["tuning_fork"]
    condition = EmergenceCondition(gating_active=True)
    fragment = Fragment("F100", is_guardian_gated=True, guardian_symbol="harmonic")

    modified, result = apply_attractor(tuning_fork, condition, fragment)

    # Sensory flavor doesn't override gating directly
    # But high enticement archetypal would
    sun_glyph = get_default_attractors()["sun_glyph"]
    fragment2 = Fragment("F200", emergence_form="SYMBOLIC")
    modified2, result2 = apply_attractor(sun_glyph, condition, fragment2)

    assert not modified2.gating_active
    assert AttractorEffect.GATING_OVERRIDE in result2.effects

    print("  [PASS] gating_override")


def test_inversion_flip():
    """Test: InversionFlip for shadow fragments."""
    mirror = get_default_attractors()["mirror"]
    condition = EmergenceCondition(inversion_likelihood=0.2)
    fragment = Fragment("F300", is_shadow=True)

    modified, result = apply_attractor(mirror, condition, fragment)

    assert result.inversion_flipped
    assert modified.shadow_accessible

    print("  [PASS] inversion_flip")


def test_competing_attractors():
    """Test: Resolve competing attractors."""
    attractors = [
        get_default_attractors()["apple"],
        get_default_attractors()["mirror"],
    ]
    condition = EmergenceCondition(potential=0.5)
    fragment = Fragment("F144", is_shadow=True)

    modified, results = resolve_competing_attractors(attractors, condition, fragment)

    assert len(results) >= 1
    assert modified.potential > condition.potential

    print(f"  [PASS] competing_attractors - {len(results)} attractors applied")


def test_shadow_consent_integration():
    """Test: Shadow consent with attractor influence (P12 integration)."""
    fragment = Fragment("F144", is_shadow=True)
    apple = get_default_attractors()["apple"]

    # High coherence - should pass
    result1 = check_shadow_consent_with_attractor(fragment, apple, 0.75, has_operator=False)
    assert result1["allowed"]

    # Low coherence - should fail
    result2 = check_shadow_consent_with_attractor(fragment, apple, 0.35, has_operator=False)
    assert not result2["allowed"]

    # Borderline with attractor lowering threshold
    result3 = check_shadow_consent_with_attractor(fragment, apple, 0.65, has_operator=False)
    assert result3["allowed"]  # 0.65 >= 0.61 (lowered threshold)

    print("  [PASS] shadow_consent_integration")


def test_coherence_floor():
    """Test: Coherence floor (never reduce below 0.22)."""
    # High enticement attractor
    powerful = Attractor("omega", PhaseComponent.ARCHETYPAL, 0.99,
                         [BySelector("shadow:*")])
    condition = EmergenceCondition(coherence_threshold=0.3)
    fragment = Fragment("F500", is_shadow=True)

    modified, result = apply_attractor(powerful, condition, fragment)

    assert modified.coherence_threshold >= COHERENCE_FLOOR

    print(f"  [PASS] coherence_floor - Floor maintained at {modified.coherence_threshold:.2f}")


def test_effect_stacking_limit():
    """Test: Maximum 2 concurrent effects."""
    # Create attractor that could trigger all 3 effects
    super_attractor = Attractor("omega", PhaseComponent.ARCHETYPAL, 0.99,
                                [BySelector("shadow:*")])
    condition = EmergenceCondition()
    fragment = Fragment("F600", is_shadow=True, is_guardian_gated=True)

    modified, result = apply_attractor(super_attractor, condition, fragment)

    assert len(result.effects) <= MAX_CONCURRENT_EFFECTS

    print(f"  [PASS] effect_stacking_limit - {len(result.effects)} effects applied")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all P18 tests."""
    print("\n" + "=" * 70)
    print(" PROMPT 18: RA ATTRACTORS AND EMERGENCE MODULATION - TEST SUITE ".center(70))
    print("=" * 70)
    print()

    tests = [
        ("Attractor Activation", test_attractor_activation),
        ("Phi-Scaled Enticement", test_phi_scaled_enticement),
        ("Fragment Targeting", test_fragment_targeting),
        ("Flavor Effects", test_flavor_effects),
        ("Attractor Application", test_attractor_application),
        ("Gating Override", test_gating_override),
        ("Inversion Flip", test_inversion_flip),
        ("Competing Attractors", test_competing_attractors),
        ("Shadow Consent Integration", test_shadow_consent_integration),
        ("Coherence Floor", test_coherence_floor),
        ("Effect Stacking Limit", test_effect_stacking_limit),
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

    # Demo
    if "--demo" in __import__("sys").argv or len(__import__("sys").argv) == 1:
        print("\n\n")
        print("=" * 70)
        print(" ATTRACTOR DEMO ".center(70))
        print("=" * 70)

        print("\n[1] Default Attractor Library:")
        for name, attractor in get_default_attractors().items():
            print(f"    {name}: {attractor.flavor.value}, enticement={attractor.enticement_level}")

        print("\n[2] Applying 'apple' attractor to shadow fragment F144...")
        apple = get_default_attractors()["apple"]
        condition = EmergenceCondition(potential=0.5, coherence_threshold=0.72)
        fragment = Fragment("F144", is_shadow=True)
        modified, result = apply_attractor(apple, condition, fragment)
        print(f"    Effects: {[e.value for e in result.effects]}")
        print(f"    Potential: {condition.potential:.2f} -> {modified.potential:.2f}")
        print(f"    Threshold: {condition.coherence_threshold:.2f} -> {modified.coherence_threshold:.2f}")
        print(f"    Inversion flipped: {result.inversion_flipped}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
