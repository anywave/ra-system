#!/usr/bin/env python3
"""
Prompt 16: Quantum Radionics Overlay for Scalar Pathway Tuning

Implements intention-guided scalar emergence modulation using radionics overlays.

Features:
- Overlay data model with hybrid rate encoding (l, m, shell + decimal skew)
- Intention vectorization with 20 core intentions + fallback
- Radius-scaled mode modifiers (constructive, diffusive, directive)
- Overlay stacking with weighted modulation (phi_boost clamped <= 0.5)
- Consent gate integration (P12 + P15)
- Visualization-ready output (mode colors, WebSocket JSON)

Codex References:
- Ra.Gates: Consent gating
- Ra.Emergence: Fragment emergence
- Prompt 12: ShadowConsent
- Prompt 15: ConsentState
"""

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


# ============================================================================
# Constants (Codex-aligned, per Architect clarifications)
# ============================================================================

PHI = 1.618033988749895
COHERENCE_MIN_FOR_PARTIAL = 0.4  # Partial overlay if coherence > 0.4
PHI_BOOST_MAX = 0.5              # Maximum phi_boost after stacking
RATE_MAX = 1440.0                # Core rate range for symbolic calendar alignment


# ============================================================================
# Enums
# ============================================================================

class OverlayMode(Enum):
    """Overlay modes determine how overlays affect emergence."""
    CONSTRUCTIVE = "constructive"  # phi potential boost
    DIFFUSIVE = "diffusive"        # fragment soft dispersion
    DIRECTIVE = "directive"        # targeted emergence vector bias


class ConsentState(Enum):
    """Unified consent states (from P12 + P15)."""
    NONE = "NONE"
    PRIVATE = "PRIVATE"
    THERAPEUTIC = "THERAPEUTIC"
    ENTANGLED = "ENTANGLED"
    WITHDRAWN = "WITHDRAWN"
    EMERGENCY = "EMERGENCY"


# ============================================================================
# Mode Colors for Visualization
# ============================================================================

MODE_COLORS = {
    OverlayMode.CONSTRUCTIVE: {"name": "gold", "hex": "#FFD700", "rgb": (255, 215, 0)},
    OverlayMode.DIFFUSIVE: {"name": "cyan", "hex": "#00FFFF", "rgb": (0, 255, 255)},
    OverlayMode.DIRECTIVE: {"name": "violet", "hex": "#8B00FF", "rgb": (139, 0, 255)},
}


# ============================================================================
# Core Intentions (20 keywords per architect spec)
# ============================================================================

CORE_INTENTIONS = {
    "healing": {"zone": (2, 1), "behavior": "stabilize_flux"},
    "insight": {"zone": (2, 2), "behavior": "boost_coherence"},
    "protection": {"zone": (1, 0), "behavior": "gate_strengthen"},
    "integration": {"zone": (3, 2), "behavior": "merge_fragments"},
    "shadow_work": {"zone": (4, 3), "behavior": "lower_threshold"},
    "focus": {"zone": (1, 1), "behavior": "direct_flux"},
    "clarity": {"zone": (2, 0), "behavior": "reduce_entropy"},
    "connection": {"zone": (3, 3), "behavior": "bridge_enable"},
    "release": {"zone": (4, 1), "behavior": "disperse_flux"},
    "guidance": {"zone": (2, 3), "behavior": "vector_align"},
    "manifestation": {"zone": (3, 0), "behavior": "amplify_potential"},
    "purification": {"zone": (1, 2), "behavior": "filter_noise"},
    "joy": {"zone": (2, 4), "behavior": "elevate_phase"},
    "calm": {"zone": (0, 1), "behavior": "dampen_flux"},
    "strength": {"zone": (3, 1), "behavior": "fortify_field"},
    "ancestral": {"zone": (4, 4), "behavior": "deep_resonance"},
    "alignment": {"zone": (0, 0), "behavior": "center_field"},
    "harmony": {"zone": (1, 3), "behavior": "balance_vectors"},
    "lucidity": {"zone": (5, 2), "behavior": "dream_enhance"},
    "memory_recovery": {"zone": (4, 2), "behavior": "fragment_recall"},
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HarmonicIndices:
    """Harmonic indices derived from rate encoding."""
    l: int = 0          # Primary harmonic (from integer part)
    m: int = 0          # Secondary harmonic
    shell: int = 0      # Shell zone
    skew: float = 0.0   # Decimal skew for fine tuning


@dataclass
class RaCoordinate:
    """Ra coordinate in normalized space."""
    x: float = 0.5
    y: float = 0.5
    theta: int = 1
    phi: int = 1
    h: int = 0
    r: float = 0.0


@dataclass
class Overlay:
    """Radionics overlay definition."""
    overlay_id: str
    rate: str                           # e.g., "568.12"
    intention: str                      # Natural language intention
    radius: float                       # 0.0-1.0 in normalized RaCoord space
    mode: OverlayMode                   # constructive/diffusive/directive
    center: RaCoordinate                # Overlay center
    harmonic_coupling: Tuple[int, int]  # (l, m) indices
    identity_gate: str = ""             # Ra.Gate.F### format
    timestamp: str = ""
    active: bool = True

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "overlay_id": self.overlay_id,
            "rate": self.rate,
            "intention": self.intention,
            "radius": self.radius,
            "mode": self.mode.value,
            "center": {"x": self.center.x, "y": self.center.y},
            "harmonic_coupling": list(self.harmonic_coupling),
            "identity_gate": self.identity_gate,
            "timestamp": self.timestamp,
            "active": self.active
        }


@dataclass
class Modifiers:
    """Field modifiers resulting from overlay application."""
    phi_boost: float = 0.0
    flux_stabilization: float = 0.0
    ankh_shift: int = 0  # +1, 0, -1
    modulated: bool = False
    vetoed: bool = False
    partial: bool = False


@dataclass
class UserState:
    """User state for consent gating."""
    user_id: str
    coherence: float = 0.5
    consent_state: ConsentState = ConsentState.NONE


# ============================================================================
# Rate Encoding (Hybrid l, m, shell + decimal skew)
# ============================================================================

def encode_rate(rate_str: str) -> HarmonicIndices:
    """
    Encode rate string into harmonic indices.

    Hybrid model per architect:
    - Integer part: l=first digit, m=second digit, shell=third digit
    - Decimal part: skew = 0.0X (e.g., .12 -> +0.02)
    """
    try:
        rate = float(rate_str)
    except ValueError:
        return HarmonicIndices()

    # Clamp to valid range
    rate = max(0.0, min(9999.99, rate))

    # Integer part
    int_part = int(rate)
    digits = str(int_part).zfill(3)

    l = int(digits[0]) if len(digits) > 0 else 0
    m = int(digits[1]) if len(digits) > 1 else 0
    shell = int(digits[2]) if len(digits) > 2 else 0

    # Decimal skew
    decimal_part = rate - int_part
    skew = decimal_part * 0.1  # Scale down for fine tuning

    return HarmonicIndices(l=l, m=m, shell=shell, skew=skew)


# ============================================================================
# Intention Vectorization
# ============================================================================

def vectorize_intention(intention: str) -> Dict[str, Any]:
    """
    Map intention to harmonic zone and behavior.

    Primary: Keyword lookup from CORE_INTENTIONS
    Fallback: Default zone assignment
    """
    intention_lower = intention.lower().strip()

    # Check for exact or partial match
    for keyword, mapping in CORE_INTENTIONS.items():
        if keyword in intention_lower:
            return {
                "matched": keyword,
                "zone": mapping["zone"],
                "behavior": mapping["behavior"],
                "confidence": 1.0
            }

    # Fallback: assign to neutral zone
    return {
        "matched": None,
        "zone": (1, 1),
        "behavior": "neutral",
        "confidence": 0.3
    }


# ============================================================================
# Mode Modifiers (Radius-Scaled)
# ============================================================================

def get_base_modifiers(mode: OverlayMode) -> Tuple[float, float, int]:
    """Get base modifier values for each mode."""
    if mode == OverlayMode.CONSTRUCTIVE:
        return (0.15, 0.05, 1)   # phi_boost, flux, ankh
    elif mode == OverlayMode.DIFFUSIVE:
        return (0.05, 0.10, -1)
    else:  # DIRECTIVE
        return (0.10, 0.02, 1)


def scale_modifiers(base: Tuple[float, float, int], radius: float,
                    rate_indices: HarmonicIndices) -> Modifiers:
    """
    Scale modifiers by normalized radius.

    - Larger radius = wider but diluted influence
    - Rate skew adds fine adjustment
    """
    phi_boost, flux, ankh = base

    # Scale by radius (inverse - smaller radius = stronger local effect)
    scale = 1.0 - (radius * 0.5)  # 0.5 at r=1.0, 1.0 at r=0.0

    # Apply rate skew (logarithmic for higher rates)
    rate_factor = 1.0 + rate_indices.skew

    scaled_phi = phi_boost * scale * rate_factor
    scaled_flux = flux * scale * rate_factor

    return Modifiers(
        phi_boost=round(scaled_phi, 4),
        flux_stabilization=round(scaled_flux, 4),
        ankh_shift=ankh,
        modulated=True
    )


# ============================================================================
# Overlay Intersection Detection
# ============================================================================

def euclidean_distance(p1: RaCoordinate, p2: RaCoordinate) -> float:
    """Calculate Euclidean distance in normalized space."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return math.sqrt(dx * dx + dy * dy)


def check_intersection(fragment_coord: RaCoordinate, overlay: Overlay) -> bool:
    """Check if fragment coordinate is within overlay radius."""
    dist = euclidean_distance(fragment_coord, overlay.center)
    return dist <= overlay.radius


# ============================================================================
# Consent Gate Integration
# ============================================================================

def check_consent_gate(overlay: Overlay, user: UserState) -> Tuple[bool, str]:
    """
    Check consent gate for overlay application.

    Per architect:
    - Partial overlay allowed if coherence > 0.4
    - Full overlay requires appropriate consent state
    """
    # Check consent state
    if user.consent_state in [ConsentState.NONE, ConsentState.WITHDRAWN]:
        return False, "Consent not granted"

    if user.consent_state == ConsentState.EMERGENCY:
        return False, "Emergency override active"

    # Check coherence threshold
    if user.coherence < COHERENCE_MIN_FOR_PARTIAL:
        return False, f"Coherence {user.coherence:.2f} below minimum {COHERENCE_MIN_FOR_PARTIAL}"

    return True, "OK"


# ============================================================================
# Overlay Stacking with Weighted Modulation
# ============================================================================

def stack_overlays(overlays: List[Overlay], fragment_coord: RaCoordinate,
                   user: UserState) -> Modifiers:
    """
    Stack multiple overlays with weighted modulation.

    Per architect:
    - Weight = modifier * (1 - distance/radius)
    - Closer overlays have stronger weight
    - phi_boost clamped to <= 0.5
    - Max 4 concurrent overlays
    """
    # Limit to 4 overlays
    active_overlays = [o for o in overlays if o.active][:4]

    total_phi = 0.0
    total_flux = 0.0
    ankh_votes = []

    for overlay in active_overlays:
        if not check_intersection(fragment_coord, overlay):
            continue

        # Check consent
        allowed, _ = check_consent_gate(overlay, user)
        if not allowed:
            continue

        # Calculate weight based on distance
        dist = euclidean_distance(fragment_coord, overlay.center)
        weight = 1.0 - (dist / overlay.radius) if overlay.radius > 0 else 1.0
        weight = max(0.0, min(1.0, weight))

        # Get scaled modifiers
        rate_indices = encode_rate(overlay.rate)
        base = get_base_modifiers(overlay.mode)
        mods = scale_modifiers(base, overlay.radius, rate_indices)

        # Apply weight
        total_phi += mods.phi_boost * weight
        total_flux += mods.flux_stabilization * weight
        ankh_votes.append(mods.ankh_shift)

    # Clamp phi_boost
    total_phi = min(total_phi, PHI_BOOST_MAX)

    # Resolve ankh (dominant vote or oscillation if tied)
    ankh_result = 0
    if ankh_votes:
        pos = sum(1 for a in ankh_votes if a > 0)
        neg = sum(1 for a in ankh_votes if a < 0)
        if pos > neg:
            ankh_result = 1
        elif neg > pos:
            ankh_result = -1

    return Modifiers(
        phi_boost=round(total_phi, 4),
        flux_stabilization=round(total_flux, 4),
        ankh_shift=ankh_result,
        modulated=total_phi > 0 or total_flux > 0
    )


# ============================================================================
# Single Overlay Application
# ============================================================================

def apply_overlay(fragment_coord: RaCoordinate, overlay: Overlay,
                  user: UserState) -> Modifiers:
    """Apply single overlay to fragment coordinate."""
    # Check intersection
    if not check_intersection(fragment_coord, overlay):
        return Modifiers(modulated=False)

    # Check consent
    allowed, reason = check_consent_gate(overlay, user)
    if not allowed:
        return Modifiers(modulated=False, vetoed=True)

    # Partial application if coherence is borderline
    partial = user.coherence < 0.6

    # Calculate modifiers
    rate_indices = encode_rate(overlay.rate)
    base = get_base_modifiers(overlay.mode)
    mods = scale_modifiers(base, overlay.radius, rate_indices)

    # Reduce effect for partial application
    if partial:
        mods.phi_boost *= 0.5
        mods.flux_stabilization *= 0.5
        mods.partial = True

    return mods


# ============================================================================
# Visualization Output
# ============================================================================

def generate_visualization_json(overlay: Overlay, modifiers: Modifiers) -> Dict[str, Any]:
    """Generate WebSocket-ready JSON for overlay visualization."""
    color = MODE_COLORS.get(overlay.mode, MODE_COLORS[OverlayMode.CONSTRUCTIVE])

    return {
        "overlay_id": overlay.overlay_id,
        "mode": overlay.mode.value,
        "mode_color": color,
        "coherence_gate": not modifiers.vetoed,
        "radius": overlay.radius,
        "phi_boost": modifiers.phi_boost,
        "flux_stabilization": modifiers.flux_stabilization,
        "ankh_shift": modifiers.ankh_shift,
        "intention": overlay.intention,
        "rate": overlay.rate,
        "timestamp": int(datetime.utcnow().timestamp())
    }


# ============================================================================
# Test Scenarios
# ============================================================================

def test_rate_encoding():
    """Test: Hybrid rate encoding (l, m, shell + decimal skew)."""
    # Test "568.12" -> l=5, m=6, shell=8, skew=+0.012
    indices = encode_rate("568.12")
    assert indices.l == 5, f"Expected l=5, got {indices.l}"
    assert indices.m == 6, f"Expected m=6, got {indices.m}"
    assert indices.shell == 8, f"Expected shell=8, got {indices.shell}"
    assert 0.01 <= indices.skew <= 0.02, f"Expected skew ~0.012, got {indices.skew}"

    # Test edge cases
    indices2 = encode_rate("0.0")
    assert indices2.l == 0

    indices3 = encode_rate("1440.00")
    assert indices3.l == 1

    print("  [PASS] rate_encoding")


def test_intention_vectorization():
    """Test: Intention vectorization with 20 core intentions."""
    # Test exact match
    result = vectorize_intention("healing")
    assert result["matched"] == "healing"
    assert result["zone"] == (2, 1)
    assert result["confidence"] == 1.0

    # Test partial match
    result2 = vectorize_intention("deep ancestral healing")
    assert result2["matched"] in ["ancestral", "healing"]

    # Test fallback
    result3 = vectorize_intention("xyzzy unknown")
    assert result3["matched"] is None
    assert result3["confidence"] < 1.0

    print("  [PASS] intention_vectorization")


def test_mode_modifiers():
    """Test: Mode modifiers with radius scaling."""
    # Constructive mode
    base = get_base_modifiers(OverlayMode.CONSTRUCTIVE)
    assert base == (0.15, 0.05, 1)

    # Scale by radius
    indices = encode_rate("568.12")
    mods = scale_modifiers(base, radius=0.6, rate_indices=indices)
    assert mods.phi_boost > 0
    assert mods.phi_boost < 0.15  # Should be scaled down
    assert mods.modulated

    print("  [PASS] mode_modifiers")


def test_overlay_intersection():
    """Test: Euclidean distance intersection detection."""
    center = RaCoordinate(x=0.5, y=0.5)
    overlay = Overlay(
        overlay_id="test-1",
        rate="568.12",
        intention="healing",
        radius=0.1,
        mode=OverlayMode.CONSTRUCTIVE,
        center=center,
        harmonic_coupling=(5, 6)
    )

    # Inside radius
    inside = RaCoordinate(x=0.52, y=0.48)
    assert check_intersection(inside, overlay)

    # Outside radius
    outside = RaCoordinate(x=0.8, y=0.8)
    assert not check_intersection(outside, overlay)

    print("  [PASS] overlay_intersection")


def test_consent_gating():
    """Test: Consent gate integration (P12 + P15)."""
    overlay = Overlay(
        overlay_id="test-1",
        rate="568.12",
        intention="healing",
        radius=0.1,
        mode=OverlayMode.CONSTRUCTIVE,
        center=RaCoordinate(x=0.5, y=0.5),
        harmonic_coupling=(5, 6)
    )

    # Allowed: THERAPEUTIC consent, good coherence
    user_ok = UserState("user1", coherence=0.75, consent_state=ConsentState.THERAPEUTIC)
    allowed, _ = check_consent_gate(overlay, user_ok)
    assert allowed

    # Denied: No consent
    user_no = UserState("user2", coherence=0.75, consent_state=ConsentState.NONE)
    allowed2, reason = check_consent_gate(overlay, user_no)
    assert not allowed2

    # Denied: Low coherence
    user_low = UserState("user3", coherence=0.2, consent_state=ConsentState.THERAPEUTIC)
    allowed3, reason = check_consent_gate(overlay, user_low)
    assert not allowed3

    print("  [PASS] consent_gating")


def test_overlay_stacking():
    """Test: Overlay stacking with weighted modulation."""
    center = RaCoordinate(x=0.5, y=0.5)
    fragment = RaCoordinate(x=0.52, y=0.48)

    overlays = [
        Overlay("o1", "568.12", "healing", 0.2, OverlayMode.CONSTRUCTIVE,
                center, (5, 6)),
        Overlay("o2", "396.00", "release", 0.3, OverlayMode.DIFFUSIVE,
                RaCoordinate(x=0.55, y=0.5), (3, 9)),
        Overlay("o3", "639.50", "connection", 0.15, OverlayMode.DIRECTIVE,
                RaCoordinate(x=0.48, y=0.52), (6, 3)),
    ]

    user = UserState("user1", coherence=0.8, consent_state=ConsentState.ENTANGLED)

    result = stack_overlays(overlays, fragment, user)

    # Should have combined effects
    assert result.modulated
    assert result.phi_boost > 0
    assert result.phi_boost <= PHI_BOOST_MAX  # Clamped

    print(f"  [PASS] overlay_stacking - Combined phi_boost: {result.phi_boost}")


def test_single_overlay_application():
    """Test: Single overlay application."""
    overlay = Overlay(
        overlay_id="test-1",
        rate="568.12",
        intention="emotional unblocking",
        radius=0.1,
        mode=OverlayMode.CONSTRUCTIVE,
        center=RaCoordinate(x=0.5, y=0.5),
        harmonic_coupling=(5, 6)
    )

    fragment = RaCoordinate(x=0.52, y=0.48)
    user = UserState("user1", coherence=0.85, consent_state=ConsentState.THERAPEUTIC)

    mods = apply_overlay(fragment, overlay, user)

    assert mods.modulated
    assert mods.phi_boost > 0
    assert not mods.vetoed
    assert not mods.partial

    print("  [PASS] single_overlay_application")


def test_visualization_output():
    """Test: Visualization JSON output."""
    overlay = Overlay(
        overlay_id="r-56812",
        rate="568.12",
        intention="healing",
        radius=0.42,
        mode=OverlayMode.CONSTRUCTIVE,
        center=RaCoordinate(x=0.5, y=0.5),
        harmonic_coupling=(5, 6)
    )

    mods = Modifiers(phi_boost=0.093, flux_stabilization=0.03, ankh_shift=1, modulated=True)

    viz = generate_visualization_json(overlay, mods)

    assert viz["overlay_id"] == "r-56812"
    assert viz["mode"] == "constructive"
    assert viz["mode_color"]["name"] == "gold"
    assert viz["coherence_gate"] == True
    assert "timestamp" in viz

    print("  [PASS] visualization_output")


def test_partial_application():
    """Test: Partial application at borderline coherence."""
    overlay = Overlay(
        overlay_id="test-1",
        rate="568.12",
        intention="healing",
        radius=0.2,
        mode=OverlayMode.CONSTRUCTIVE,
        center=RaCoordinate(x=0.5, y=0.5),
        harmonic_coupling=(5, 6)
    )

    fragment = RaCoordinate(x=0.52, y=0.48)

    # Borderline coherence (> 0.4 but < 0.6)
    user = UserState("user1", coherence=0.5, consent_state=ConsentState.THERAPEUTIC)

    mods = apply_overlay(fragment, overlay, user)

    assert mods.modulated
    assert mods.partial  # Should be partial application

    print("  [PASS] partial_application")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all P16 tests."""
    print("\n" + "=" * 70)
    print(" PROMPT 16: QUANTUM RADIONICS OVERLAY - TEST SUITE ".center(70))
    print("=" * 70)
    print()

    tests = [
        ("Rate Encoding", test_rate_encoding),
        ("Intention Vectorization", test_intention_vectorization),
        ("Mode Modifiers", test_mode_modifiers),
        ("Overlay Intersection", test_overlay_intersection),
        ("Consent Gating", test_consent_gating),
        ("Overlay Stacking", test_overlay_stacking),
        ("Single Overlay Application", test_single_overlay_application),
        ("Visualization Output", test_visualization_output),
        ("Partial Application", test_partial_application),
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
        print(" RADIONICS OVERLAY DEMO ".center(70))
        print("=" * 70)

        print("\n[1] Creating overlay with rate 568.12...")
        overlay = Overlay(
            overlay_id="demo-1",
            rate="568.12",
            intention="emotional unblocking",
            radius=0.15,
            mode=OverlayMode.CONSTRUCTIVE,
            center=RaCoordinate(x=0.5, y=0.5),
            harmonic_coupling=(5, 6),
            identity_gate="Ra.Gate.F13"
        )
        print(f"    Rate encoding: {encode_rate('568.12')}")
        print(f"    Intention: {vectorize_intention('emotional unblocking')}")

        print("\n[2] Applying to fragment at (0.52, 0.48)...")
        fragment = RaCoordinate(x=0.52, y=0.48)
        user = UserState("demo_user", coherence=0.85, consent_state=ConsentState.THERAPEUTIC)
        mods = apply_overlay(fragment, overlay, user)
        print(f"    Modulated: {mods.modulated}")
        print(f"    phi_boost: {mods.phi_boost}")
        print(f"    flux_stabilization: {mods.flux_stabilization}")

        print("\n[3] Generating visualization JSON...")
        viz = generate_visualization_json(overlay, mods)
        print(json.dumps(viz, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
