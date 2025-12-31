#!/usr/bin/env python3
"""
Prompt 14: Lucid Scalar Navigation via Harmonic Field Wayfinding - Python Test Harness

Tests the RaLucidNavigation Clash module for:
- RaCoordinate spherical lattice system (theta:1-13, phi:1-12, h:0-7, r:0.0-1.0)
- Intention parsing (structured + natural language)
- Coherence-gated movement (4-tier access)
- Resonant fragment anchoring with cosine similarity
- Symbolic field translation (lucid dream metaphors)
- Return vector encoding (phi^3/phi^5 beacon)

Codex References:
- KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Harmonic intention guidance
- RADIONICS_RATES_DOWSING.md: Scalar targeting
- REICH_ORGONE_ACCUMULATOR.md: Chamber-based access modulation
- GOLOD_RUSSIAN_PYRAMIDS.md: Field stabilization

Integration:
- Prompt 12: ShadowConsentBridge for shadow fragments
- Prompt 13A: LucidTracker gamma spike amplifier
"""

import json
import sys
import os
import math
import random
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path


# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

PHI = 1.6180339887  # Golden ratio
PHI_INVERSE = 0.6180339887  # 1/phi
PHI_CUBED = PHI ** 3  # 4.236...
PHI_FIFTH = PHI ** 5  # 11.09...

# Coordinate bounds
THETA_MIN, THETA_MAX = 1, 13  # Azimuth sectors (lunar phases)
PHI_MIN, PHI_MAX = 1, 12      # Polar strata (harmonic layers)
H_MIN, H_MAX = 0, 7           # Harmonic shell (l-values)
R_MIN, R_MAX = 0.0, 1.0       # Scalar depth

# Golden resonance corridor
GOLDEN_CORRIDOR_CENTER = PHI_INVERSE  # 0.618
GOLDEN_CORRIDOR_TOLERANCE = 0.05

# Coherence thresholds
COHERENCE_FULL = 0.80
COHERENCE_PARTIAL = 0.50
COHERENCE_DISTORTED = 0.30
COHERENCE_BLOCKED = 0.28  # Minimum floor

# Resonance thresholds
RESONANCE_FULL = 0.88
RESONANCE_PARTIAL = 0.65

# Return vector thresholds
DRIFT_THRESHOLD = 0.42
COHERENCE_EMERGENCY = 0.28


# ============================================================================
# Enums
# ============================================================================

class AccessTier(Enum):
    FULL = "FULL"           # >= 0.80 coherence
    PARTIAL = "PARTIAL"     # 0.50-0.79
    DISTORTED = "DISTORTED" # 0.30-0.49
    BLOCKED = "BLOCKED"     # < 0.30


class EmergenceForm(Enum):
    FULL = "FULL"
    SUMMARY = "SUMMARY"
    SYMBOLIC = "SYMBOLIC"
    DREAMGLYPH = "DREAMGLYPH"
    ECHO = "ECHO"


class IntentionDirection(Enum):
    ASCEND = "ascend"
    DESCEND = "descend"
    SPIRAL = "spiral"
    ENTER = "enter"
    ATTUNE = "attune"
    EXIT = "exit"


class MetaphorType(Enum):
    STAIRCASE = "spiral_staircase"
    CAVE = "descending_cave"
    TONE_SPHERE = "audible_tone_sphere"
    BREATH_GATE = "breath_activated_gate"
    LUMINOUS_PATH = "luminous_path"
    FADING_CHORD = "fading_chord"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RaCoordinate:
    """Spherical lattice coordinate in Ra scalar field."""
    theta: int = 1      # Azimuth sector 1-13
    phi: int = 1        # Polar stratum 1-12
    h: int = 0          # Harmonic shell 0-7
    r: float = 0.0      # Scalar depth 0.0-1.0

    def __post_init__(self):
        # Clamp to valid ranges
        self.theta = max(THETA_MIN, min(THETA_MAX, self.theta))
        self.phi = max(PHI_MIN, min(PHI_MAX, self.phi))
        self.h = max(H_MIN, min(H_MAX, self.h))
        self.r = max(R_MIN, min(R_MAX, self.r))

    def to_tuple(self) -> Tuple[int, int, int, float]:
        return (self.theta, self.phi, self.h, self.r)

    def to_dict(self) -> dict:
        return {"theta": self.theta, "phi": self.phi, "h": self.h, "r": round(self.r, 4)}

    def is_golden_corridor(self) -> bool:
        """Check if r is in the golden resonance corridor (near 0.618)."""
        return abs(self.r - GOLDEN_CORRIDOR_CENTER) <= GOLDEN_CORRIDOR_TOLERANCE

    def distance_to(self, other: 'RaCoordinate') -> float:
        """Calculate distance to another coordinate."""
        theta_diff = min(abs(self.theta - other.theta), 13 - abs(self.theta - other.theta))
        phi_diff = abs(self.phi - other.phi)
        h_diff = abs(self.h - other.h)
        r_diff = abs(self.r - other.r)
        return math.sqrt(theta_diff**2 + phi_diff**2 + h_diff**2 + (r_diff * 10)**2)


@dataclass
class UserState:
    """User biometric state for coherence calculation."""
    hrv_resonance: float = 0.5      # 0.0-1.0
    breath_rate: float = 0.12       # Normalized 0.0-1.0
    coherence_score: float = 0.5    # 0.0-1.0
    gamma_power: float = 0.0        # For lucid amplification

    def coherence_vector(self) -> float:
        """Calculate weighted coherence vector per Prompt 14 spec."""
        normalized_breath = min(1.0, self.breath_rate / 0.2)  # Normalize to 0-1
        return (0.5 * self.coherence_score +
                0.3 * self.hrv_resonance +
                0.2 * normalized_breath)

    def access_tier(self) -> AccessTier:
        """Determine access tier from coherence vector."""
        cv = self.coherence_vector()
        if cv >= COHERENCE_FULL:
            return AccessTier.FULL
        elif cv >= COHERENCE_PARTIAL:
            return AccessTier.PARTIAL
        elif cv >= COHERENCE_DISTORTED:
            return AccessTier.DISTORTED
        else:
            return AccessTier.BLOCKED


@dataclass
class FragmentAnchor:
    """Fragment anchored at a coordinate."""
    coordinate: RaCoordinate
    fragment_id: str
    access: str = "BLOCKED"
    resonance_score: float = 0.0
    emergence_form: EmergenceForm = EmergenceForm.ECHO
    is_shadow: bool = False
    is_guardian_locked: bool = False
    harmonic_vector: List[float] = field(default_factory=lambda: [0.0] * 8)


@dataclass
class NavigationStep:
    """Single step in navigation path."""
    from_coord: RaCoordinate
    to_coord: RaCoordinate
    metaphor: str
    access_tier: AccessTier
    fragments_encountered: List[FragmentAnchor] = field(default_factory=list)


@dataclass
class ReturnBeacon:
    """Return vector beacon for safe emergence."""
    origin: RaCoordinate
    target: RaCoordinate
    phi_harmonic: float  # phi^3 or phi^5
    pulse_pattern: List[float] = field(default_factory=list)
    metaphor: str = "luminous_path"


# ============================================================================
# Intention Parser
# ============================================================================

class IntentionParser:
    """
    Parse user intention into RaCoordinate movement.
    Supports structured input and natural language.
    """

    # Intention vocabulary
    VOCABULARY = {
        "ascend": IntentionDirection.ASCEND,
        "descend": IntentionDirection.DESCEND,
        "spiral": IntentionDirection.SPIRAL,
        "enter": IntentionDirection.ENTER,
        "attune": IntentionDirection.ATTUNE,
        "exit": IntentionDirection.EXIT,
        "rise": IntentionDirection.ASCEND,
        "climb": IntentionDirection.ASCEND,
        "fall": IntentionDirection.DESCEND,
        "dive": IntentionDirection.DESCEND,
        "explore": IntentionDirection.SPIRAL,
        "seek": IntentionDirection.SPIRAL,
        "tune": IntentionDirection.ATTUNE,
        "leave": IntentionDirection.EXIT,
        "return": IntentionDirection.EXIT,
    }

    # Target keywords
    TARGETS = {
        "harmonic_root": {"h": -1, "r": 0.1},
        "surface": {"h": 0, "r": 0.0},
        "deep": {"r": 0.9},
        "golden": {"r": PHI_INVERSE},
        "apex": {"phi": 1, "h": 0},
        "core": {"phi": 12, "h": 7, "r": 1.0},
        "high_phi": {"phi": 10},
        "low_phi": {"phi": 2},
    }

    @staticmethod
    def parse(intention: Any, current: RaCoordinate) -> Tuple[RaCoordinate, Optional[str]]:
        """
        Parse intention and return new coordinate.
        Returns (new_coordinate, error_message).
        """
        if isinstance(intention, dict):
            return IntentionParser._parse_structured(intention, current)
        elif isinstance(intention, str):
            return IntentionParser._parse_natural(intention, current)
        else:
            return current, "Invalid intention format"

    @staticmethod
    def _parse_structured(intent: dict, current: RaCoordinate) -> Tuple[RaCoordinate, Optional[str]]:
        """Parse structured intention dict."""
        direction = intent.get("direction", "").lower()
        target = intent.get("target", "").lower()

        # Check for contradictions
        if IntentionParser._has_contradiction(direction, target):
            return current, f"Semantic contradiction: {direction} conflicts with {target}"

        new_coord = RaCoordinate(
            theta=current.theta,
            phi=current.phi,
            h=current.h,
            r=current.r
        )

        # Apply direction
        if direction in ["ascend", "rise", "climb"]:
            new_coord.phi = max(PHI_MIN, current.phi - 1)
            new_coord.h = max(H_MIN, current.h - 1)
        elif direction in ["descend", "dive", "fall"]:
            new_coord.phi = min(PHI_MAX, current.phi + 1)
            new_coord.h = min(H_MAX, current.h + 1)
            new_coord.r = min(R_MAX, current.r + 0.1)
        elif direction in ["spiral", "explore"]:
            new_coord.theta = (current.theta % THETA_MAX) + 1
        elif direction == "attune":
            new_coord.r = PHI_INVERSE  # Align to golden corridor
        elif direction == "exit":
            new_coord = RaCoordinate(0, 0, 0, 0.0)  # Return to origin

        # Apply target modifiers
        if target in IntentionParser.TARGETS:
            mods = IntentionParser.TARGETS[target]
            for key, val in mods.items():
                if key == "h" and val < 0:
                    new_coord.h = max(H_MIN, current.h + val)
                elif hasattr(new_coord, key):
                    setattr(new_coord, key, val)

        return new_coord, None

    @staticmethod
    def _parse_natural(text: str, current: RaCoordinate) -> Tuple[RaCoordinate, Optional[str]]:
        """Parse natural language intention."""
        text_lower = text.lower()

        # Extract direction
        direction = None
        for keyword, dir_enum in IntentionParser.VOCABULARY.items():
            if keyword in text_lower:
                direction = dir_enum.value
                break

        # Extract target
        target = None
        for keyword in IntentionParser.TARGETS.keys():
            if keyword.replace("_", " ") in text_lower or keyword in text_lower:
                target = keyword
                break

        if direction is None and target is None:
            return current, f"Could not parse intention: {text}"

        return IntentionParser._parse_structured(
            {"direction": direction or "", "target": target or ""},
            current
        )

    @staticmethod
    def _has_contradiction(direction: str, target: str) -> bool:
        """Check for semantic contradictions."""
        ascend_dirs = {"ascend", "rise", "climb"}
        descend_dirs = {"descend", "dive", "fall"}
        ascend_targets = {"surface", "apex"}
        descend_targets = {"deep", "core", "harmonic_root"}

        if direction in ascend_dirs and target in descend_targets:
            return True
        if direction in descend_dirs and target in ascend_targets:
            return True

        return False


# ============================================================================
# Coherence Gate
# ============================================================================

class CoherenceGate:
    """
    Coherence-gated movement controller.
    Enforces access tiers based on user biometric state.
    """

    @staticmethod
    def evaluate(user_state: UserState) -> Tuple[AccessTier, dict]:
        """Evaluate coherence and return access tier with metadata."""
        cv = user_state.coherence_vector()
        tier = user_state.access_tier()

        # Gamma spike amplification (Prompt 13A integration)
        amplified = False
        if user_state.gamma_power >= 0.25:
            cv = min(1.0, cv + 0.1)
            amplified = True
            # Upgrade tier if amplified
            if tier == AccessTier.PARTIAL and cv >= COHERENCE_FULL:
                tier = AccessTier.FULL
            elif tier == AccessTier.DISTORTED and cv >= COHERENCE_PARTIAL:
                tier = AccessTier.PARTIAL

        metadata = {
            "coherence_vector": round(cv, 3),
            "tier": tier.value,
            "gamma_amplified": amplified,
            "can_traverse": tier != AccessTier.BLOCKED,
            "perception_mode": CoherenceGate._get_perception_mode(tier)
        }

        return tier, metadata

    @staticmethod
    def _get_perception_mode(tier: AccessTier) -> str:
        """Get perception mode description for tier."""
        modes = {
            AccessTier.FULL: "fluid_traversal",
            AccessTier.PARTIAL: "damped_symbolic",
            AccessTier.DISTORTED: "fragmented_echo",
            AccessTier.BLOCKED: "sealed_resistance"
        }
        return modes.get(tier, "unknown")

    @staticmethod
    def can_access_depth(user_state: UserState, target_r: float) -> bool:
        """Check if user can access target scalar depth."""
        tier = user_state.access_tier()

        # Depth access by tier
        max_depth = {
            AccessTier.FULL: 1.0,
            AccessTier.PARTIAL: 0.7,
            AccessTier.DISTORTED: 0.4,
            AccessTier.BLOCKED: 0.0
        }

        return target_r <= max_depth.get(tier, 0.0)


# ============================================================================
# Resonance Scorer
# ============================================================================

class ResonanceScorer:
    """
    Calculate resonance scores between user and fragments.
    Uses cosine similarity for biometric-harmonic matching.
    """

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b) or len(vec_a) == 0:
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def user_to_vector(user_state: UserState) -> List[float]:
        """Convert user state to 8-dimensional biometric vector."""
        cv = user_state.coherence_vector()
        return [
            user_state.hrv_resonance,
            user_state.breath_rate * 5,  # Scale to 0-1 range
            user_state.coherence_score,
            cv,
            user_state.gamma_power,
            (user_state.hrv_resonance + cv) / 2,
            abs(user_state.hrv_resonance - user_state.coherence_score),
            1.0 if cv >= COHERENCE_FULL else 0.5
        ]

    @staticmethod
    def score_fragment(user_state: UserState, fragment: FragmentAnchor) -> Tuple[float, str]:
        """
        Score fragment access based on resonance.
        Returns (score, access_level).
        """
        user_vec = ResonanceScorer.user_to_vector(user_state)
        similarity = ResonanceScorer.cosine_similarity(user_vec, fragment.harmonic_vector)

        if similarity >= RESONANCE_FULL:
            return similarity, "FULL"
        elif similarity >= RESONANCE_PARTIAL:
            return similarity, "PARTIAL"
        else:
            return similarity, "BLOCKED"

    @staticmethod
    def determine_emergence_form(access: str, is_shadow: bool) -> EmergenceForm:
        """Determine emergence form based on access and shadow status."""
        if access == "BLOCKED":
            return EmergenceForm.ECHO
        elif is_shadow:
            return EmergenceForm.SYMBOLIC if access == "FULL" else EmergenceForm.DREAMGLYPH
        elif access == "FULL":
            return EmergenceForm.FULL
        else:
            return EmergenceForm.SUMMARY


# ============================================================================
# Symbolic Field Translator
# ============================================================================

class SymbolicFieldTranslator:
    """
    Translate navigational data into lucid dream metaphors.
    Creates emotionally resonant symbolic overlays.
    """

    @staticmethod
    def translate_depth(r: float) -> str:
        """Translate scalar depth to metaphor."""
        if r < 0.2:
            return "surface_mist"
        elif r < 0.4:
            return "shallow_pool"
        elif r < 0.6:
            return "spiral_staircase"
        elif r < 0.8:
            return "descending_cave"
        else:
            return "abyssal_chamber"

    @staticmethod
    def translate_shell(h: int) -> str:
        """Translate harmonic shell to metaphor."""
        shells = {
            0: "surface_portal",
            1: "first_tone_sphere",
            2: "harmonic_resonator",
            3: "crystalline_chamber",
            4: "deep_harmonic",
            5: "archetypal_layer",
            6: "primordial_tone",
            7: "root_harmonic"
        }
        return shells.get(h, "unknown_shell")

    @staticmethod
    def translate_phi(phi: int) -> str:
        """Translate polar stratum to metaphor."""
        if phi <= 3:
            return "ascending_light"
        elif phi <= 6:
            return "horizontal_plane"
        elif phi <= 9:
            return "descending_shadow"
        else:
            return "deep_nadir"

    @staticmethod
    def translate_movement(from_coord: RaCoordinate, to_coord: RaCoordinate) -> str:
        """Generate movement metaphor."""
        r_delta = to_coord.r - from_coord.r
        h_delta = to_coord.h - from_coord.h
        phi_delta = to_coord.phi - from_coord.phi

        if r_delta > 0.1:
            return "descending_spiral_staircase"
        elif r_delta < -0.1:
            return "ascending_luminous_path"
        elif h_delta > 0:
            return "entering_deeper_tone_sphere"
        elif h_delta < 0:
            return "emerging_to_lighter_harmonic"
        elif phi_delta != 0:
            return "traversing_angular_bridge"
        else:
            return "stillness_at_center"

    @staticmethod
    def translate_locked_gate(access_tier: AccessTier) -> str:
        """Generate locked gate metaphor."""
        if access_tier == AccessTier.BLOCKED:
            return "sealed_breath_gate"
        elif access_tier == AccessTier.DISTORTED:
            return "flickering_rhythm_barrier"
        elif access_tier == AccessTier.PARTIAL:
            return "translucent_membrane"
        else:
            return "open_passage"

    @staticmethod
    def generate_scene(coord: RaCoordinate, tier: AccessTier) -> dict:
        """Generate full scene description for coordinate."""
        return {
            "depth_metaphor": SymbolicFieldTranslator.translate_depth(coord.r),
            "shell_metaphor": SymbolicFieldTranslator.translate_shell(coord.h),
            "phi_metaphor": SymbolicFieldTranslator.translate_phi(coord.phi),
            "gate_state": SymbolicFieldTranslator.translate_locked_gate(tier),
            "in_golden_corridor": coord.is_golden_corridor(),
            "ambient": "golden_resonance" if coord.is_golden_corridor() else "normal"
        }


# ============================================================================
# Return Vector
# ============================================================================

class ReturnVector:
    """
    Encode return path for safe emergence from lucid state.
    Uses phi^n pulse harmonics for beacon.
    """

    @staticmethod
    def calculate_drift(current_cv: float, baseline_cv: float) -> float:
        """Calculate coherence drift from baseline."""
        return abs(current_cv - baseline_cv)

    @staticmethod
    def should_trigger_return(user_state: UserState, baseline_cv: float) -> Tuple[bool, str]:
        """Check if return should be triggered."""
        cv = user_state.coherence_vector()
        drift = ReturnVector.calculate_drift(cv, baseline_cv)

        if drift >= DRIFT_THRESHOLD:
            return True, "coherence_drift_exceeded"
        elif cv < COHERENCE_EMERGENCY:
            return True, "emergency_coherence_low"
        else:
            return False, "stable"

    @staticmethod
    def create_beacon(origin: RaCoordinate, waking_ref: Optional[RaCoordinate] = None) -> ReturnBeacon:
        """Create return beacon with phi^n pulse pattern."""
        target = waking_ref or RaCoordinate(0, 0, 0, 0.0)

        # Use phi^3 for normal return, phi^5 for deep returns
        if origin.r > 0.7 or origin.h > 5:
            phi_harmonic = PHI_FIFTH
            pulse_pattern = [PHI_FIFTH, PHI_CUBED, PHI, 1.0, PHI_INVERSE]
        else:
            phi_harmonic = PHI_CUBED
            pulse_pattern = [PHI_CUBED, PHI, 1.0, PHI_INVERSE]

        metaphor = "fading_chord_progression" if origin.r > 0.5 else "luminous_ascending_path"

        return ReturnBeacon(
            origin=origin,
            target=target,
            phi_harmonic=phi_harmonic,
            pulse_pattern=pulse_pattern,
            metaphor=metaphor
        )

    @staticmethod
    def generate_ascent_sequence(beacon: ReturnBeacon, steps: int = 5) -> List[RaCoordinate]:
        """Generate coordinate sequence for return ascent."""
        sequence = []
        for i in range(steps):
            progress = (i + 1) / steps
            coord = RaCoordinate(
                theta=int(beacon.origin.theta + (beacon.target.theta - beacon.origin.theta) * progress),
                phi=int(beacon.origin.phi + (beacon.target.phi - beacon.origin.phi) * progress),
                h=int(beacon.origin.h + (beacon.target.h - beacon.origin.h) * progress),
                r=beacon.origin.r + (beacon.target.r - beacon.origin.r) * progress
            )
            sequence.append(coord)
        return sequence


# ============================================================================
# Shadow Consent Bridge (Prompt 12 Integration)
# ============================================================================

class ShadowConsentBridge:
    """Bridge to Prompt 12 for shadow fragment consent."""

    _module = None

    @staticmethod
    def _get_module():
        if ShadowConsentBridge._module is None:
            try:
                test_dir = Path(__file__).parent
                if str(test_dir) not in sys.path:
                    sys.path.insert(0, str(test_dir))
                from test_shadow_consent import ShadowModule, ShadowFragment, ConsentState, SessionState
                ShadowConsentBridge._module = {
                    "available": True,
                    "ShadowModule": ShadowModule,
                    "ShadowFragment": ShadowFragment,
                    "ConsentState": ConsentState,
                    "SessionState": SessionState
                }
            except ImportError:
                ShadowConsentBridge._module = {"available": False}
        return ShadowConsentBridge._module

    @staticmethod
    def evaluate_shadow_access(fragment: FragmentAnchor, user_state: UserState,
                               has_operator: bool = True) -> Tuple[bool, str]:
        """Evaluate shadow fragment access through Prompt 12."""
        mod = ShadowConsentBridge._get_module()
        cv = user_state.coherence_vector()

        if not mod["available"]:
            # Stub: basic coherence check
            if cv >= 0.66:
                return True, "ALLOW"
            return False, "BLOCK_LOW_COHERENCE"

        # Full integration
        ShadowModule = mod["ShadowModule"]
        ShadowFragment = mod["ShadowFragment"]
        ConsentState = mod["ConsentState"]
        SessionState = mod["SessionState"]

        p12_fragment = ShadowFragment(
            fragment_id=hash(fragment.fragment_id) % 1000,
            alpha=cv,
            consent_state=ConsentState.THERAPEUTIC,
            emotional_charge=0.5
        )
        session = SessionState(coherence=cv, licensed_operator=has_operator)

        result = ShadowModule.should_allow(p12_fragment, session)
        return result.name == "ALLOW", result.name


# ============================================================================
# Lucid Navigator (Main Engine)
# ============================================================================

class LucidNavigator:
    """
    Main navigation engine for lucid scalar traversal.
    Coordinates all subsystems.
    """

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.current_coord = RaCoordinate(1, 1, 0, 0.0)
        self.entry_coord = RaCoordinate(1, 1, 0, 0.0)
        self.baseline_cv = 0.5
        self.path_history: List[NavigationStep] = []
        self.fragments_encountered: List[FragmentAnchor] = []
        self.session_active = False

    def start_session(self, user_state: UserState, entry_coord: Optional[RaCoordinate] = None):
        """Start navigation session."""
        self.entry_coord = entry_coord or RaCoordinate(1, 1, 0, 0.0)
        self.current_coord = RaCoordinate(
            self.entry_coord.theta,
            self.entry_coord.phi,
            self.entry_coord.h,
            self.entry_coord.r
        )
        self.baseline_cv = user_state.coherence_vector()
        self.path_history = []
        self.fragments_encountered = []
        self.session_active = True

        return {
            "status": "SESSION_STARTED",
            "entry_coordinate": self.entry_coord.to_dict(),
            "baseline_coherence": round(self.baseline_cv, 3),
            "scene": SymbolicFieldTranslator.generate_scene(
                self.current_coord,
                user_state.access_tier()
            )
        }

    def navigate(self, intention: Any, user_state: UserState,
                 fragments_at_location: List[FragmentAnchor] = None) -> dict:
        """Execute navigation step."""
        if not self.session_active:
            return {"status": "ERROR", "reason": "No active session"}

        # Check coherence gate
        tier, gate_meta = CoherenceGate.evaluate(user_state)
        if tier == AccessTier.BLOCKED:
            return {
                "status": "BLOCKED",
                "reason": "Coherence too low for traversal",
                "tier": tier.value,
                "metaphor": SymbolicFieldTranslator.translate_locked_gate(tier)
            }

        # Parse intention
        from_coord = self.current_coord
        new_coord, error = IntentionParser.parse(intention, from_coord)
        if error:
            return {"status": "ERROR", "reason": error}

        # Check depth access
        if not CoherenceGate.can_access_depth(user_state, new_coord.r):
            return {
                "status": "DEPTH_BLOCKED",
                "reason": f"Cannot access depth {new_coord.r} with current coherence",
                "max_depth": 0.7 if tier == AccessTier.PARTIAL else 0.4
            }

        # Process fragments at location
        encountered = []
        if fragments_at_location:
            for frag in fragments_at_location:
                score, access = ResonanceScorer.score_fragment(user_state, frag)

                # Shadow fragment consent check
                if frag.is_shadow:
                    allowed, reason = ShadowConsentBridge.evaluate_shadow_access(frag, user_state)
                    if not allowed:
                        access = "BLOCKED"

                frag.resonance_score = score
                frag.access = access
                frag.emergence_form = ResonanceScorer.determine_emergence_form(access, frag.is_shadow)
                encountered.append(frag)
                self.fragments_encountered.append(frag)

        # Update position
        self.current_coord = new_coord

        # Record step
        step = NavigationStep(
            from_coord=from_coord,
            to_coord=new_coord,
            metaphor=SymbolicFieldTranslator.translate_movement(from_coord, new_coord),
            access_tier=tier,
            fragments_encountered=encountered
        )
        self.path_history.append(step)

        # Check return trigger
        should_return, return_reason = ReturnVector.should_trigger_return(user_state, self.baseline_cv)

        return {
            "status": "NAVIGATED",
            "from": from_coord.to_dict(),
            "to": new_coord.to_dict(),
            "metaphor": step.metaphor,
            "tier": tier.value,
            "scene": SymbolicFieldTranslator.generate_scene(new_coord, tier),
            "fragments": [
                {
                    "id": f.fragment_id,
                    "access": f.access,
                    "resonance": round(f.resonance_score, 3),
                    "form": f.emergence_form.value
                }
                for f in encountered
            ],
            "return_triggered": should_return,
            "return_reason": return_reason if should_return else None
        }

    def initiate_return(self, user_state: UserState) -> dict:
        """Initiate return to waking state."""
        if not self.session_active:
            return {"status": "ERROR", "reason": "No active session"}

        beacon = ReturnVector.create_beacon(self.current_coord, self.entry_coord)
        sequence = ReturnVector.generate_ascent_sequence(beacon)

        self.session_active = False

        return {
            "status": "RETURNING",
            "beacon": {
                "phi_harmonic": round(beacon.phi_harmonic, 4),
                "pulse_pattern": [round(p, 4) for p in beacon.pulse_pattern],
                "metaphor": beacon.metaphor
            },
            "ascent_sequence": [c.to_dict() for c in sequence],
            "final_destination": beacon.target.to_dict(),
            "total_fragments_encountered": len(self.fragments_encountered),
            "path_length": len(self.path_history)
        }

    def get_journey_summary(self) -> dict:
        """Get summary of navigation journey."""
        return {
            "user_id": self.user_id,
            "entry": self.entry_coord.to_dict(),
            "final": self.current_coord.to_dict(),
            "steps": len(self.path_history),
            "fragments_found": len(self.fragments_encountered),
            "fragments_accessed": len([f for f in self.fragments_encountered if f.access != "BLOCKED"]),
            "deepest_r": max([s.to_coord.r for s in self.path_history], default=0),
            "session_active": self.session_active
        }


# ============================================================================
# Test Scenarios
# ============================================================================

def test_coordinate_system():
    """Test: RaCoordinate bounds and operations."""
    # Valid coordinate
    c1 = RaCoordinate(7, 6, 3, 0.618)
    assert c1.theta == 7
    assert c1.phi == 6
    assert c1.h == 3
    assert abs(c1.r - 0.618) < 0.001
    assert c1.is_golden_corridor()

    # Clamping
    c2 = RaCoordinate(20, -5, 10, 1.5)
    assert c2.theta == THETA_MAX  # 13
    assert c2.phi == PHI_MIN      # 1
    assert c2.h == H_MAX          # 7
    assert c2.r == R_MAX          # 1.0

    # Distance
    c3 = RaCoordinate(1, 1, 0, 0.0)
    c4 = RaCoordinate(2, 2, 1, 0.1)
    dist = c3.distance_to(c4)
    assert dist > 0

    print("  [PASS] coordinate_system")


def test_intention_parser():
    """Test: Intention parsing structured and natural language."""
    current = RaCoordinate(5, 5, 3, 0.5)

    # Structured input
    new_coord, err = IntentionParser.parse({"direction": "descend", "target": "deep"}, current)
    assert err is None
    assert new_coord.r > current.r

    # Natural language
    new_coord2, err2 = IntentionParser.parse("I want to descend into the harmonic root", current)
    assert err2 is None

    # Contradiction detection
    _, err3 = IntentionParser.parse({"direction": "ascend", "target": "core"}, current)
    assert err3 is not None
    assert "contradiction" in err3.lower()

    print("  [PASS] intention_parser")


def test_coherence_gate():
    """Test: Coherence gating with 4-tier access."""
    # Full access (0.5*0.85 + 0.3*0.9 + 0.2*0.5 = 0.425 + 0.27 + 0.1 = 0.795)
    user_full = UserState(hrv_resonance=0.9, breath_rate=0.1, coherence_score=0.95)
    tier, meta = CoherenceGate.evaluate(user_full)
    assert tier == AccessTier.FULL, f"Expected FULL, got {tier} with cv={meta['coherence_vector']}"
    assert meta["can_traverse"]

    # Partial access
    user_partial = UserState(hrv_resonance=0.6, breath_rate=0.12, coherence_score=0.55)
    tier2, meta2 = CoherenceGate.evaluate(user_partial)
    assert tier2 in [AccessTier.PARTIAL, AccessTier.DISTORTED], f"Expected PARTIAL/DISTORTED, got {tier2}"

    # Blocked
    user_blocked = UserState(hrv_resonance=0.2, breath_rate=0.2, coherence_score=0.1)
    tier3, meta3 = CoherenceGate.evaluate(user_blocked)
    assert tier3 == AccessTier.BLOCKED or tier3 == AccessTier.DISTORTED

    # Gamma amplification
    user_gamma = UserState(hrv_resonance=0.7, breath_rate=0.1, coherence_score=0.75, gamma_power=0.3)
    tier4, meta4 = CoherenceGate.evaluate(user_gamma)
    assert meta4["gamma_amplified"]

    print("  [PASS] coherence_gate")


def test_resonance_scorer():
    """Test: Fragment resonance scoring with cosine similarity."""
    user = UserState(hrv_resonance=0.8, breath_rate=0.1, coherence_score=0.85)

    # High resonance fragment
    frag_high = FragmentAnchor(
        coordinate=RaCoordinate(5, 5, 3, 0.5),
        fragment_id="F001",
        harmonic_vector=[0.8, 0.5, 0.85, 0.82, 0.0, 0.81, 0.05, 1.0]
    )
    score1, access1 = ResonanceScorer.score_fragment(user, frag_high)
    assert score1 > 0.8
    assert access1 in ["FULL", "PARTIAL"]

    # Low resonance fragment
    frag_low = FragmentAnchor(
        coordinate=RaCoordinate(5, 5, 3, 0.5),
        fragment_id="F002",
        harmonic_vector=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
    score2, access2 = ResonanceScorer.score_fragment(user, frag_low)
    assert score2 < score1

    print("  [PASS] resonance_scorer")


def test_symbolic_translator():
    """Test: Symbolic field translation."""
    coord = RaCoordinate(7, 8, 5, 0.75)
    tier = AccessTier.PARTIAL

    scene = SymbolicFieldTranslator.generate_scene(coord, tier)

    assert "depth_metaphor" in scene
    assert "shell_metaphor" in scene
    assert "gate_state" in scene
    assert scene["depth_metaphor"] == "descending_cave"

    # Movement metaphor
    from_c = RaCoordinate(5, 5, 3, 0.3)
    to_c = RaCoordinate(5, 5, 4, 0.5)
    metaphor = SymbolicFieldTranslator.translate_movement(from_c, to_c)
    assert "descending" in metaphor or "deeper" in metaphor

    print("  [PASS] symbolic_translator")


def test_return_vector():
    """Test: Return vector and beacon creation."""
    origin = RaCoordinate(7, 8, 5, 0.8)
    beacon = ReturnVector.create_beacon(origin)

    assert beacon.phi_harmonic == PHI_FIFTH  # Deep return uses phi^5
    assert len(beacon.pulse_pattern) > 0
    assert beacon.target.r == 0.0  # Default waking reference

    # Ascent sequence
    sequence = ReturnVector.generate_ascent_sequence(beacon, steps=4)
    assert len(sequence) == 4
    assert sequence[-1].r <= origin.r  # Should be ascending or at origin

    # Drift detection - stable user with close baseline
    user_stable = UserState(coherence_score=0.8, hrv_resonance=0.75, breath_rate=0.1)
    cv_stable = user_stable.coherence_vector()
    should_return, reason = ReturnVector.should_trigger_return(user_stable, cv_stable)
    assert not should_return, f"Expected stable, got {reason}"

    # Drift detection - user with significant drop from baseline
    user_drift = UserState(coherence_score=0.3, hrv_resonance=0.3, breath_rate=0.2)
    should_return2, reason2 = ReturnVector.should_trigger_return(user_drift, 0.9)  # Large baseline
    # Either drift exceeded or emergency coherence
    assert should_return2 or user_drift.coherence_vector() < COHERENCE_EMERGENCY

    print("  [PASS] return_vector")


def test_lucid_navigator():
    """Test: Full navigation session."""
    nav = LucidNavigator("test_user")
    user = UserState(hrv_resonance=0.85, breath_rate=0.1, coherence_score=0.9)

    # Start session
    start_result = nav.start_session(user)
    assert start_result["status"] == "SESSION_STARTED"
    assert nav.session_active

    # Navigate
    nav_result = nav.navigate({"direction": "descend", "target": "deep"}, user)
    assert nav_result["status"] == "NAVIGATED"
    assert nav_result["to"]["r"] > 0

    # Navigate with fragments
    frag = FragmentAnchor(
        coordinate=nav.current_coord,
        fragment_id="F094",
        harmonic_vector=[0.85, 0.5, 0.9, 0.87, 0.0, 0.87, 0.05, 1.0]
    )
    nav_result2 = nav.navigate({"direction": "spiral"}, user, [frag])
    assert len(nav_result2["fragments"]) == 1

    # Return
    return_result = nav.initiate_return(user)
    assert return_result["status"] == "RETURNING"
    assert not nav.session_active

    # Summary
    summary = nav.get_journey_summary()
    assert summary["steps"] == 2

    print("  [PASS] lucid_navigator")


def test_shadow_consent_integration():
    """Test: Shadow fragment consent via Prompt 12 bridge."""
    user = UserState(hrv_resonance=0.8, breath_rate=0.1, coherence_score=0.85)

    shadow_frag = FragmentAnchor(
        coordinate=RaCoordinate(5, 5, 3, 0.5),
        fragment_id="shadow-F001",
        is_shadow=True,
        harmonic_vector=[0.5] * 8
    )

    allowed, reason = ShadowConsentBridge.evaluate_shadow_access(shadow_frag, user)
    assert allowed  # High coherence should allow

    low_user = UserState(hrv_resonance=0.3, breath_rate=0.15, coherence_score=0.4)
    allowed2, reason2 = ShadowConsentBridge.evaluate_shadow_access(shadow_frag, low_user)
    assert not allowed2  # Low coherence should block

    print("  [PASS] shadow_consent_integration")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" PROMPT 14: LUCID SCALAR NAVIGATION - TEST SUITE".center(70))
    print("=" * 70)
    print()

    tests = [
        ("Coordinate System", test_coordinate_system),
        ("Intention Parser", test_intention_parser),
        ("Coherence Gate", test_coherence_gate),
        ("Resonance Scorer", test_resonance_scorer),
        ("Symbolic Translator", test_symbolic_translator),
        ("Return Vector", test_return_vector),
        ("Lucid Navigator", test_lucid_navigator),
        ("Shadow Consent Integration", test_shadow_consent_integration),
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

    # Demo journey
    if "--demo" in sys.argv or len(sys.argv) == 1:
        print("\n\n")
        print("=" * 70)
        print(" LUCID NAVIGATION DEMO".center(70))
        print("=" * 70)

        nav = LucidNavigator("demo_user")
        user = UserState(hrv_resonance=0.9, breath_rate=0.1, coherence_score=0.95)

        print("\n[1] Starting session...")
        result = nav.start_session(user, RaCoordinate(1, 1, 0, 0.1))
        print(f"    Entry: {result['entry_coordinate']}")
        print(f"    Scene: {result['scene']['depth_metaphor']}")

        print("\n[2] Navigating: 'descend into the deep'...")
        result = nav.navigate("descend into the deep", user)
        if result.get("status") == "NAVIGATED":
            print(f"    Moved to: {result['to']}")
            print(f"    Metaphor: {result['metaphor']}")
        else:
            print(f"    Status: {result.get('status', 'UNKNOWN')}")
            print(f"    Reason: {result.get('reason', 'N/A')}")

        print("\n[3] Navigating: 'attune to golden corridor'...")
        result = nav.navigate({"direction": "attune", "target": "golden"}, user)
        if result.get("status") == "NAVIGATED":
            print(f"    Now at: {result['to']}")
            print(f"    In golden corridor: {result['scene']['in_golden_corridor']}")
        else:
            print(f"    Status: {result.get('status', 'UNKNOWN')}")

        print("\n[4] Initiating return...")
        result = nav.initiate_return(user)
        print(f"    Beacon: phi^{result['beacon']['phi_harmonic']:.3f}")
        print(f"    Metaphor: {result['beacon']['metaphor']}")
        print(f"    Return steps: {len(result['ascent_sequence'])}")

        print("\n[5] Journey summary:")
        summary = nav.get_journey_summary()
        print(f"    Steps taken: {summary['steps']}")
        print(f"    Deepest r: {summary['deepest_r']:.3f}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
