#!/usr/bin/env python3
"""
Prompt 13: Scalar Dream Induction & Symbolic Fragment Integration - Python Test Harness
(Patch 13A - Architect Clarifications Implemented)

Tests the RaDreamPhaseScheduler Clash module for:
- Sleep phase state machine (WAKE -> THETA -> DELTA -> REM)
- phi-cycle timing (~90 min REM alignment)
- Hybrid EEG/HRV phase detection with confidence scoring (13A)
- Symbolic fragment emergence during REM
- Coherence-gated fragment surfacing
- Shadow consent gating via ShadowModule.evaluate() (13A)
- Symbol-to-fragment dynamic linking with semantic similarity (13A)
- Fragment storage backend (JSON per session) (13A)
- Lucid dream marker tracking (13A)
- Hardware interface stubs (13A)

Codex References:
- ra_constants_v2.json: phi^n timing modulation
- Prompt 12: Shadow consent gating for symbols
"""

import json
import sys
import os
import random
import math
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path


# ============================================================================
# Enums matching Clash module
# ============================================================================

class SleepPhase(Enum):
    WAKE = 0
    THETA = 1   # Light sleep (3.5-7 Hz)
    DELTA = 2   # Deep sleep (0.5-3.5 Hz)
    REM = 3     # REM (7-12 Hz)


class ArchetypalSymbol(Enum):
    OWL = "owl"           # Wisdom, insight
    SPIRAL = "spiral"     # Searching, journey
    MIRROR = "mirror"     # Self-reflection
    RIVER = "river"       # Flow, change
    LABYRINTH = "labyrinth"  # Confusion, complexity
    LIGHT = "light"       # Insight, clarity
    FLAME = "flame"       # Transformation
    CAVE = "cave"         # Hidden, unconscious
    TREE = "tree"         # Growth, connection
    MOON = "moon"         # Cycles, feminine
    STAR = "star"         # Guidance, aspiration
    WATER = "water"       # Emotion, depth


class EmotionFlag(Enum):
    JOY = "joy"
    CONFUSION = "confusion"
    AWE = "awe"
    GRIEF = "grief"
    CURIOSITY = "curiosity"
    FEAR = "fear"
    PEACE = "peace"
    NONE = "none"


class FrequencyBand(Enum):
    DELTA = 0  # 0.5-3.5 Hz
    THETA = 1  # 3.5-7 Hz
    ALPHA = 2  # 7-12 Hz (REM)
    BETA = 3   # 12-30 Hz (wake)


class AudioType(Enum):
    BINAURAL = "binaural"
    ISOCHRONIC = "isochronic"
    GOLDEN_STACK = "golden_stack"
    NONE = "none"


class VisualType(Enum):
    FLOWER_OF_LIFE = "flower_of_life"
    PHI_SPIRAL = "phi_spiral"
    LED_PULSE = "led_pulse"
    NONE = "none"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SymbolMapping:
    """Symbol to fragment/concept mapping."""
    symbol: ArchetypalSymbol
    mapped_fragment: Optional[str] = None  # e.g., "F13"
    mapped_concept: Optional[str] = None   # e.g., "wisdom"


@dataclass
class EmotionalRegister:
    """Combined emotional register."""
    primary: EmotionFlag = EmotionFlag.NONE
    secondary: EmotionFlag = EmotionFlag.NONE


@dataclass
class EmergentDreamFragment:
    """Dream fragment with symbolic content."""
    fragment_id: str = ""
    form: str = "SYMBOLIC"
    emergence_phase: SleepPhase = SleepPhase.REM
    coherence_trace: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    emotional_register: EmotionalRegister = field(default_factory=EmotionalRegister)
    symbol_map: List[SymbolMapping] = field(default_factory=list)
    shadow_detected: bool = False
    timestamp: float = 0.0


@dataclass
class ResonanceOutput:
    """Entrainment settings for current phase."""
    target_band: FrequencyBand = FrequencyBand.DELTA
    audio_type: AudioType = AudioType.NONE
    visual_type: VisualType = VisualType.NONE
    base_frequency: float = 0.0  # Hz
    amplitude_mod: float = 0.0   # 0.0-1.0
    phi_multiplier: int = 0


@dataclass
class PhaseOutput:
    """Current phase information."""
    in_rem: bool = False
    sleep_phase: SleepPhase = SleepPhase.WAKE
    phase_depth: float = 0.0


@dataclass
class IntegrationSummary:
    """Post-sleep integration summary."""
    fragments_surfaced: int = 0
    dominant_symbols: List[ArchetypalSymbol] = field(default_factory=list)
    coherence_delta: float = 0.0
    shadow_gated: bool = False
    journal_prompts: List[str] = field(default_factory=list)


# ============================================================================
# Symbol Database
# ============================================================================

SYMBOL_CONCEPTS = {
    ArchetypalSymbol.OWL: ("wisdom", "F13"),
    ArchetypalSymbol.SPIRAL: ("searching", None),
    ArchetypalSymbol.MIRROR: ("self-reflection", "F21"),
    ArchetypalSymbol.RIVER: ("flow", None),
    ArchetypalSymbol.LABYRINTH: ("confusion", "F42"),
    ArchetypalSymbol.LIGHT: ("insight", None),
    ArchetypalSymbol.FLAME: ("transformation", "F08"),
    ArchetypalSymbol.CAVE: ("hidden", None),
    ArchetypalSymbol.TREE: ("growth", "F55"),
    ArchetypalSymbol.MOON: ("cycles", None),
    ArchetypalSymbol.STAR: ("guidance", "F77"),
    ArchetypalSymbol.WATER: ("emotion", None),
}

SHADOW_SYMBOLS = {ArchetypalSymbol.CAVE, ArchetypalSymbol.LABYRINTH, ArchetypalSymbol.MIRROR}

# Symbol semantic vectors for dynamic linking (Patch 13A)
# Each symbol has a 8-dimensional semantic embedding
SYMBOL_VECTORS = {
    ArchetypalSymbol.OWL: [0.9, 0.2, 0.1, 0.8, 0.3, 0.1, 0.7, 0.4],
    ArchetypalSymbol.SPIRAL: [0.2, 0.9, 0.6, 0.3, 0.7, 0.2, 0.4, 0.5],
    ArchetypalSymbol.MIRROR: [0.4, 0.3, 0.9, 0.2, 0.5, 0.8, 0.3, 0.6],
    ArchetypalSymbol.RIVER: [0.3, 0.7, 0.2, 0.6, 0.8, 0.1, 0.5, 0.3],
    ArchetypalSymbol.LABYRINTH: [0.2, 0.8, 0.7, 0.1, 0.4, 0.6, 0.3, 0.9],
    ArchetypalSymbol.LIGHT: [0.8, 0.1, 0.3, 0.9, 0.2, 0.1, 0.8, 0.2],
    ArchetypalSymbol.FLAME: [0.6, 0.4, 0.5, 0.7, 0.3, 0.9, 0.2, 0.4],
    ArchetypalSymbol.CAVE: [0.1, 0.6, 0.8, 0.2, 0.5, 0.7, 0.4, 0.8],
    ArchetypalSymbol.TREE: [0.5, 0.3, 0.2, 0.4, 0.9, 0.3, 0.6, 0.5],
    ArchetypalSymbol.MOON: [0.4, 0.5, 0.4, 0.3, 0.6, 0.2, 0.5, 0.7],
    ArchetypalSymbol.STAR: [0.7, 0.2, 0.3, 0.8, 0.4, 0.1, 0.9, 0.3],
    ArchetypalSymbol.WATER: [0.3, 0.6, 0.5, 0.2, 0.7, 0.4, 0.3, 0.8],
}


# ============================================================================
# Patch 13A: Biometric Input with EEG/HRV Hybrid Detection
# ============================================================================

@dataclass
class BiometricInput:
    """EEG band powers and HRV coherence for hybrid phase detection (13A)."""
    # EEG band powers (0.0-1.0 normalized)
    delta_power: float = 0.0   # 0.5-3.5 Hz
    theta_power: float = 0.0   # 3.5-7 Hz
    alpha_power: float = 0.0   # 7-12 Hz
    beta_power: float = 0.0    # 12-30 Hz
    gamma_power: float = 0.0   # 30-100 Hz (lucid marker)
    # HRV metrics
    hrv_coherence: float = 0.0
    hrv_rmssd: float = 0.0     # Root mean square of successive differences


@dataclass
class PhaseDetection:
    """Phase detection result with confidence scoring (13A)."""
    detected_phase: SleepPhase = SleepPhase.WAKE
    confidence: float = 0.0
    eeg_dominant: str = "beta"
    hrv_aligned: bool = False


class PhaseDetector:
    """Hybrid EEG/HRV phase detector with confidence scoring (Patch 13A)."""

    CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for phase transition

    # EEG thresholds for phase detection
    DELTA_THRESHOLD = 0.60   # >60% delta power -> deep sleep
    THETA_THRESHOLD = 0.45   # >45% theta power -> light sleep
    ALPHA_THRESHOLD = 0.40   # >40% alpha power -> REM
    BETA_THRESHOLD = 0.50    # >50% beta power -> awake
    GAMMA_SPIKE_THRESHOLD = 0.25  # Lucid dream marker

    @staticmethod
    def detect_phase(bio: BiometricInput) -> PhaseDetection:
        """Detect sleep phase from biometric input with confidence scoring."""
        # Determine dominant EEG band
        bands = [
            ("delta", bio.delta_power, SleepPhase.DELTA),
            ("theta", bio.theta_power, SleepPhase.THETA),
            ("alpha", bio.alpha_power, SleepPhase.REM),
            ("beta", bio.beta_power, SleepPhase.WAKE),
        ]
        bands.sort(key=lambda x: -x[1])  # Sort by power descending
        dominant_band, dominant_power, dominant_phase = bands[0]

        # Calculate confidence based on band separation and HRV alignment
        second_power = bands[1][1] if len(bands) > 1 else 0
        band_separation = dominant_power - second_power

        # HRV alignment check
        hrv_aligned = False
        if dominant_phase == SleepPhase.DELTA and bio.hrv_coherence > 0.6:
            hrv_aligned = True
        elif dominant_phase == SleepPhase.THETA and 0.4 < bio.hrv_coherence < 0.7:
            hrv_aligned = True
        elif dominant_phase == SleepPhase.REM and bio.hrv_rmssd > 30:
            hrv_aligned = True
        elif dominant_phase == SleepPhase.WAKE and bio.hrv_coherence < 0.5:
            hrv_aligned = True

        # Confidence = band separation + HRV alignment bonus
        confidence = min(1.0, band_separation + (0.15 if hrv_aligned else 0))

        # Apply phase-specific thresholds
        if dominant_phase == SleepPhase.DELTA and bio.delta_power < PhaseDetector.DELTA_THRESHOLD:
            confidence *= 0.7
        elif dominant_phase == SleepPhase.THETA and bio.theta_power < PhaseDetector.THETA_THRESHOLD:
            confidence *= 0.7
        elif dominant_phase == SleepPhase.REM and bio.alpha_power < PhaseDetector.ALPHA_THRESHOLD:
            confidence *= 0.7

        return PhaseDetection(
            detected_phase=dominant_phase,
            confidence=confidence,
            eeg_dominant=dominant_band,
            hrv_aligned=hrv_aligned
        )

    @staticmethod
    def is_lucid_spike(bio: BiometricInput) -> bool:
        """Check for gamma spike indicating potential lucid state (13A)."""
        return bio.gamma_power >= PhaseDetector.GAMMA_SPIKE_THRESHOLD


# ============================================================================
# Patch 13A: Shadow Consent Bridge (Integration with Prompt 12)
# ============================================================================

@dataclass
class ShadowConsentResult:
    """Result from shadow consent evaluation."""
    allowed: bool = False
    gating_reason: str = "none"
    feedback_prompt: str = ""


class ShadowConsentBridge:
    """
    Bridge to Prompt 12 ShadowModule for consent-gated shadow symbols (Patch 13A).

    Routes shadow fragments through ShadowModule.evaluate() with synthetic
    ShadowFragment format.
    """

    # Imports ShadowModule from Prompt 12 (or provides stub)
    _shadow_module = None

    @staticmethod
    def _get_shadow_module():
        """Lazy import of ShadowModule from Prompt 12."""
        if ShadowConsentBridge._shadow_module is None:
            try:
                # Try to import from Prompt 12
                import sys
                test_dir = Path(__file__).parent
                if str(test_dir) not in sys.path:
                    sys.path.insert(0, str(test_dir))
                from test_shadow_consent import ShadowModule, ShadowFragment as P12Fragment
                from test_shadow_consent import ConsentState, SessionState, GatingResult
                ShadowConsentBridge._shadow_module = {
                    "module": ShadowModule,
                    "fragment": P12Fragment,
                    "consent": ConsentState,
                    "session": SessionState,
                    "gating": GatingResult
                }
            except ImportError:
                # Stub implementation
                ShadowConsentBridge._shadow_module = {"stub": True}
        return ShadowConsentBridge._shadow_module

    @staticmethod
    def evaluate_shadow_symbol(symbol: ArchetypalSymbol, coherence: float,
                               has_operator: bool = True) -> ShadowConsentResult:
        """
        Evaluate shadow symbol through Prompt 12 consent gating.

        Creates synthetic ShadowFragment:
        {
            "id": "shadow-F13",
            "origin": "dream",
            "symbol": "owl (reversed)",
            "coherence": 0.72,
            "consent_state": "THERAPEUTIC"
        }
        """
        mod = ShadowConsentBridge._get_shadow_module()

        if "stub" in mod:
            # Stub implementation - basic coherence gating
            if coherence >= 0.66:
                return ShadowConsentResult(
                    allowed=True,
                    gating_reason="ALLOW",
                    feedback_prompt="This shadow aspect is ready for gentle exploration."
                )
            else:
                return ShadowConsentResult(
                    allowed=False,
                    gating_reason="BLOCK_LOW_COHERENCE",
                    feedback_prompt="Build coherence before shadow access."
                )

        # Full integration with Prompt 12
        ShadowModule = mod["module"]
        P12Fragment = mod["fragment"]
        ConsentState = mod["consent"]
        SessionState = mod["session"]

        # Create synthetic ShadowFragment
        fragment = P12Fragment(
            fragment_id=13,  # Dream module ID
            fragment_form=1,
            alpha=coherence,
            consent_state=ConsentState.THERAPEUTIC,
            emotional_charge=0.5,
            last_session_token=f"dream-{symbol.value}"
        )

        # Create session state
        session = SessionState(
            coherence=coherence,
            licensed_operator=has_operator,
            session_token=f"dream-session-{datetime.now().timestamp()}"
        )

        # Evaluate through ShadowModule
        result = ShadowModule.should_allow(fragment, session)
        feedback = ShadowModule.prompt(fragment, result)

        return ShadowConsentResult(
            allowed=(result.name == "ALLOW"),
            gating_reason=result.name,
            feedback_prompt=feedback.context_prompt
        )


# ============================================================================
# Patch 13A: Fragment Storage Backend
# ============================================================================

class FragmentStorage:
    """
    JSON-based fragment storage per session (Patch 13A).

    Storage path: dream_fragments/{user_id}/{session_timestamp}.json
    """

    def __init__(self, base_path: Optional[str] = None, user_id: str = "default"):
        self.base_path = Path(base_path) if base_path else Path("dream_fragments")
        self.user_id = user_id
        self.session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Create directory structure if needed."""
        user_dir = self.base_path / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session_path(self) -> Path:
        """Get current session file path."""
        return self.base_path / self.user_id / f"{self.session_id}.json"

    def save_fragment(self, fragment: EmergentDreamFragment) -> bool:
        """Save fragment to session file."""
        try:
            # Load existing session data
            session_data = self._load_session()

            # Add fragment
            frag_dict = {
                "fragment_id": fragment.fragment_id,
                "form": fragment.form,
                "emergence_phase": fragment.emergence_phase.name,
                "coherence_trace": fragment.coherence_trace,
                "emotional_register": {
                    "primary": fragment.emotional_register.primary.value,
                    "secondary": fragment.emotional_register.secondary.value
                },
                "symbols": [
                    {
                        "symbol": sm.symbol.value,
                        "concept": sm.mapped_concept,
                        "fragment_link": sm.mapped_fragment
                    }
                    for sm in fragment.symbol_map
                ],
                "shadow_detected": fragment.shadow_detected,
                "timestamp": fragment.timestamp
            }
            session_data["fragments"].append(frag_dict)

            # Save
            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)

            return True
        except Exception as e:
            print(f"[FragmentStorage] Save error: {e}")
            return False

    def _load_session(self) -> dict:
        """Load or initialize session data."""
        if self.session_path.exists():
            with open(self.session_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "started": datetime.now().isoformat(),
            "fragments": [],
            "metadata": {}
        }

    def save_summary(self, summary: IntegrationSummary) -> bool:
        """Save post-sleep integration summary."""
        try:
            session_data = self._load_session()
            session_data["summary"] = {
                "fragments_surfaced": summary.fragments_surfaced,
                "dominant_symbols": [s.value for s in summary.dominant_symbols],
                "coherence_delta": summary.coherence_delta,
                "shadow_gated": summary.shadow_gated,
                "journal_prompts": summary.journal_prompts
            }
            session_data["ended"] = datetime.now().isoformat()

            with open(self.session_path, "w", encoding="utf-8") as f:
                json.dump(session_data, f, indent=2)

            return True
        except Exception as e:
            print(f"[FragmentStorage] Summary save error: {e}")
            return False


# ============================================================================
# Patch 13A: Symbol-to-Fragment Dynamic Linking
# ============================================================================

class SymbolLinker:
    """
    Dynamic symbol-to-fragment linking using semantic similarity (Patch 13A).

    Uses cosine similarity with threshold > 0.72 for linking.
    """

    SIMILARITY_THRESHOLD = 0.72

    @staticmethod
    def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def find_linked_fragments(symbol: ArchetypalSymbol,
                              fragment_vectors: Dict[str, List[float]]) -> List[Tuple[str, float]]:
        """
        Find fragments linked to symbol via semantic similarity.

        Returns list of (fragment_id, similarity) tuples above threshold.
        """
        symbol_vec = SYMBOL_VECTORS.get(symbol, [])
        if not symbol_vec:
            return []

        matches = []
        for frag_id, frag_vec in fragment_vectors.items():
            sim = SymbolLinker.cosine_similarity(symbol_vec, frag_vec)
            if sim >= SymbolLinker.SIMILARITY_THRESHOLD:
                matches.append((frag_id, sim))

        # Sort by similarity descending
        matches.sort(key=lambda x: -x[1])
        return matches

    @staticmethod
    def link_symbol_to_concept(symbol: ArchetypalSymbol,
                               concept_vectors: Dict[str, List[float]]) -> Optional[str]:
        """Find best matching concept for symbol."""
        symbol_vec = SYMBOL_VECTORS.get(symbol, [])
        if not symbol_vec:
            return None

        best_match = None
        best_sim = 0.0

        for concept, concept_vec in concept_vectors.items():
            sim = SymbolLinker.cosine_similarity(symbol_vec, concept_vec)
            if sim > best_sim:
                best_sim = sim
                best_match = concept

        return best_match if best_sim >= SymbolLinker.SIMILARITY_THRESHOLD else None


# ============================================================================
# Patch 13A: Lucid Dream Marker Tracking
# ============================================================================

@dataclass
class LucidMarker:
    """Lucid dream detection marker (metadata only, no induction) (Patch 13A)."""
    timestamp: float = 0.0
    gamma_power: float = 0.0
    phase_when_detected: SleepPhase = SleepPhase.REM
    coherence_at_detection: float = 0.0
    duration_seconds: float = 0.0


class LucidTracker:
    """
    Track lucid dream markers (gamma spikes) - metadata only (Patch 13A).

    Per architect: "Lucid Mode is NOT actively induced by this module."
    Only detects and logs gamma spike events.
    """

    GAMMA_SPIKE_THRESHOLD = 0.25
    MIN_SPIKE_DURATION = 3.0  # seconds

    def __init__(self):
        self.markers: List[LucidMarker] = []
        self._current_spike_start: Optional[float] = None
        self._current_spike_power: float = 0.0

    def process_sample(self, bio: BiometricInput, phase: SleepPhase,
                      coherence: float, timestamp: float) -> Optional[LucidMarker]:
        """Process biometric sample, return marker if spike detected."""
        is_spike = bio.gamma_power >= self.GAMMA_SPIKE_THRESHOLD

        if is_spike:
            if self._current_spike_start is None:
                # Start of new spike
                self._current_spike_start = timestamp
                self._current_spike_power = bio.gamma_power
            else:
                # Continuing spike - track max power
                self._current_spike_power = max(self._current_spike_power, bio.gamma_power)
            return None
        else:
            if self._current_spike_start is not None:
                # End of spike
                duration = timestamp - self._current_spike_start
                if duration >= self.MIN_SPIKE_DURATION:
                    marker = LucidMarker(
                        timestamp=self._current_spike_start,
                        gamma_power=self._current_spike_power,
                        phase_when_detected=phase,
                        coherence_at_detection=coherence,
                        duration_seconds=duration
                    )
                    self.markers.append(marker)
                    self._current_spike_start = None
                    self._current_spike_power = 0.0
                    return marker
                else:
                    # Too short, reset
                    self._current_spike_start = None
                    self._current_spike_power = 0.0
            return None

    def get_session_markers(self) -> List[LucidMarker]:
        """Get all lucid markers for session."""
        return self.markers.copy()


# ============================================================================
# Patch 13A: Hardware Interface Stubs
# ============================================================================

class HardwareInterface:
    """
    Hardware interface stubs for future EEG/audio/visual devices (Patch 13A).

    All methods marked as future_linked per architect directive.
    """

    @staticmethod
    def read_eeg_bands() -> BiometricInput:
        """[STUB] Read EEG band powers from hardware.
        future_linked: EEG headband driver (OpenBCI, Muse, etc.)
        """
        # Return simulated data
        return BiometricInput(
            delta_power=random.uniform(0.1, 0.3),
            theta_power=random.uniform(0.2, 0.4),
            alpha_power=random.uniform(0.15, 0.35),
            beta_power=random.uniform(0.1, 0.25),
            gamma_power=random.uniform(0.0, 0.15),
            hrv_coherence=random.uniform(0.4, 0.7),
            hrv_rmssd=random.uniform(20, 50)
        )

    @staticmethod
    def read_hrv_coherence() -> Tuple[float, float]:
        """[STUB] Read HRV coherence and RMSSD from hardware.
        future_linked: Heart rate sensor (Polar, Garmin, etc.)
        """
        return (random.uniform(0.4, 0.7), random.uniform(20, 50))

    @staticmethod
    def output_audio_entrainment(freq: float, audio_type: AudioType) -> bool:
        """[STUB] Output audio entrainment signal.
        future_linked: Audio DAC/PWM driver
        """
        print(f"[HW_STUB] Audio: {freq:.1f}Hz {audio_type.value}")
        return True

    @staticmethod
    def output_visual_entrainment(visual_type: VisualType, brightness: float) -> bool:
        """[STUB] Output visual entrainment pattern.
        future_linked: LED driver / display controller
        """
        print(f"[HW_STUB] Visual: {visual_type.value} @ {brightness:.0%}")
        return True

    @staticmethod
    def trigger_haptic_pulse(intensity: float, duration_ms: int) -> bool:
        """[STUB] Trigger haptic feedback pulse.
        future_linked: Haptic motor driver
        """
        print(f"[HW_STUB] Haptic: {intensity:.0%} for {duration_ms}ms")
        return True


# ============================================================================
# DreamScheduler - Core Processing Logic
# ============================================================================

class DreamScheduler:
    """
    Dream phase scheduler with symbolic emergence.

    Implements φ-aligned REM cycles and symbolic fragment surfacing.
    """

    # Timing constants (in seconds)
    PHI_CYCLE = 5400         # ~90 mins between REM
    THETA_MIN = 600          # 10 min minimum
    DELTA_MIN = 1800         # 30 min minimum
    REM_BASE_DURATION = 900  # 15 min base, grows per cycle

    # Coherence thresholds
    COHERENCE_THRESHOLD = 0.6
    SHADOW_COHERENCE_MIN = 0.7  # Higher threshold for shadow

    def __init__(self):
        self.sleep_mode = False
        self.current_phase = SleepPhase.WAKE
        self.phase_counter = 0
        self.rem_counter = 0
        self.cycle_number = 0
        self.fragments: List[EmergentDreamFragment] = []
        self.session_start = None

    def start_session(self) -> None:
        """Start sleep session."""
        self.sleep_mode = True
        self.current_phase = SleepPhase.WAKE
        self.phase_counter = 0
        self.rem_counter = 0
        self.cycle_number = 0
        self.fragments = []
        self.session_start = datetime.now()

    def end_session(self) -> IntegrationSummary:
        """End session and generate integration summary."""
        self.sleep_mode = False

        # Collect dominant symbols
        symbol_counts: Dict[ArchetypalSymbol, int] = {}
        for frag in self.fragments:
            for sm in frag.symbol_map:
                symbol_counts[sm.symbol] = symbol_counts.get(sm.symbol, 0) + 1

        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: -x[1])
        dominant = [s for s, _ in sorted_symbols[:3]]

        # Calculate coherence delta
        if self.fragments:
            first_coh = self.fragments[0].coherence_trace[-1]
            last_coh = self.fragments[-1].coherence_trace[-1]
            delta = last_coh - first_coh
        else:
            delta = 0.0

        # Check for shadow content
        shadow_gated = any(f.shadow_detected for f in self.fragments)

        # Generate journal prompts
        prompts = self._generate_journal_prompts(dominant, shadow_gated)

        return IntegrationSummary(
            fragments_surfaced=len(self.fragments),
            dominant_symbols=dominant,
            coherence_delta=delta,
            shadow_gated=shadow_gated,
            journal_prompts=prompts
        )

    def _generate_journal_prompts(self, symbols: List[ArchetypalSymbol],
                                  shadow: bool) -> List[str]:
        """Generate journaling prompts based on symbols."""
        prompts = []

        if ArchetypalSymbol.OWL in symbols:
            prompts.append("What wisdom emerged that you didn't expect?")
        if ArchetypalSymbol.SPIRAL in symbols:
            prompts.append("What journey or search was represented?")
        if ArchetypalSymbol.MIRROR in symbols:
            prompts.append("What aspect of yourself was reflected back?")
        if ArchetypalSymbol.FLAME in symbols:
            prompts.append("What transformation is calling to you?")

        if shadow:
            prompts.append("Some deeper content surfaced. Would you like to explore it with support?")

        if not prompts:
            prompts.append("What feelings arose as you awakened?")

        return prompts

    def step(self, coherence_trace: List[float]) -> Tuple[PhaseOutput, Optional[EmergentDreamFragment]]:
        """
        Step the scheduler by one cycle.

        Returns current phase output and any emerged fragment.
        """
        if not self.sleep_mode:
            return PhaseOutput(), None

        self.phase_counter += 1

        # Phase transitions
        old_phase = self.current_phase
        self._update_phase()

        # Check for fragment emergence
        fragment = None
        if self.current_phase == SleepPhase.REM:
            fragment = self._check_emergence(coherence_trace)

        # Update REM counter
        if self.current_phase == SleepPhase.DELTA:
            self.rem_counter += 1
        elif self.current_phase == SleepPhase.REM and old_phase != SleepPhase.REM:
            self.rem_counter = 0
            self.cycle_number += 1

        phase_out = PhaseOutput(
            in_rem=(self.current_phase == SleepPhase.REM),
            sleep_phase=self.current_phase,
            phase_depth=min(1.0, self.phase_counter / 1000)
        )

        return phase_out, fragment

    def _update_phase(self) -> None:
        """Update sleep phase based on timing."""
        if self.current_phase == SleepPhase.WAKE:
            self.current_phase = SleepPhase.THETA
            self.phase_counter = 0
        elif self.current_phase == SleepPhase.THETA:
            if self.phase_counter >= self.THETA_MIN:
                self.current_phase = SleepPhase.DELTA
                self.phase_counter = 0
        elif self.current_phase == SleepPhase.DELTA:
            if self.rem_counter >= self.PHI_CYCLE:
                self.current_phase = SleepPhase.REM
                self.phase_counter = 0
        elif self.current_phase == SleepPhase.REM:
            rem_duration = self.REM_BASE_DURATION + (self.cycle_number * 300)
            if self.phase_counter >= rem_duration:
                self.current_phase = SleepPhase.WAKE
                self.phase_counter = 0

    def _check_emergence(self, trace: List[float]) -> Optional[EmergentDreamFragment]:
        """Check if fragment should emerge."""
        if len(trace) < 3:
            return None

        # Check rising coherence pattern
        if not (trace[2] > trace[0] and trace[1] >= trace[0]):
            return None

        # Check threshold
        if trace[2] < self.COHERENCE_THRESHOLD:
            return None

        # Generate fragment
        return self._generate_fragment(trace)

    def _generate_fragment(self, trace: List[float]) -> EmergentDreamFragment:
        """Generate an emergent dream fragment."""
        # Select 2-3 symbols
        num_symbols = random.randint(2, 3)
        symbols = random.sample(list(ArchetypalSymbol), num_symbols)

        # Create symbol mappings
        mappings = []
        shadow_detected = False
        for sym in symbols:
            concept, frag_id = SYMBOL_CONCEPTS.get(sym, ("unknown", None))
            mappings.append(SymbolMapping(
                symbol=sym,
                mapped_fragment=frag_id,
                mapped_concept=concept
            ))
            if sym in SHADOW_SYMBOLS:
                shadow_detected = True

        # Select emotions
        emotions = random.sample(list(EmotionFlag)[:-1], 2)

        fragment = EmergentDreamFragment(
            fragment_id=f"dream-{random.randint(1000, 9999)}",
            form="SYMBOLIC",
            emergence_phase=SleepPhase.REM,
            coherence_trace=trace.copy(),
            emotional_register=EmotionalRegister(emotions[0], emotions[1]),
            symbol_map=mappings,
            shadow_detected=shadow_detected,
            timestamp=datetime.now().timestamp()
        )

        self.fragments.append(fragment)
        return fragment

    def get_resonance(self) -> ResonanceOutput:
        """Get current resonance settings for phase."""
        if self.current_phase == SleepPhase.WAKE:
            return ResonanceOutput(FrequencyBand.BETA, AudioType.NONE, VisualType.NONE)
        elif self.current_phase == SleepPhase.THETA:
            return ResonanceOutput(
                FrequencyBand.THETA, AudioType.BINAURAL, VisualType.PHI_SPIRAL,
                base_frequency=5.0, amplitude_mod=0.5, phi_multiplier=1
            )
        elif self.current_phase == SleepPhase.DELTA:
            return ResonanceOutput(
                FrequencyBand.DELTA, AudioType.GOLDEN_STACK, VisualType.LED_PULSE,
                base_frequency=2.0, amplitude_mod=0.3, phi_multiplier=2
            )
        else:  # REM
            return ResonanceOutput(
                FrequencyBand.ALPHA, AudioType.GOLDEN_STACK, VisualType.FLOWER_OF_LIFE,
                base_frequency=10.0, amplitude_mod=0.7, phi_multiplier=3
            )


# ============================================================================
# Dream Symbol Prompt Generator
# ============================================================================

class DreamPromptGenerator:
    """Generates Claude-ready dream prompts for testing."""

    SYMBOLS = list(ArchetypalSymbol)
    EMOTIONS = [e for e in EmotionFlag if e != EmotionFlag.NONE]

    @staticmethod
    def generate() -> str:
        """Generate a random dream prompt."""
        chosen = random.sample(DreamPromptGenerator.SYMBOLS, 2)
        coherence = [round(random.uniform(0.35, 0.65), 2) for _ in range(3)]
        coherence.sort()  # Ensure rising pattern
        emo = random.choice(DreamPromptGenerator.EMOTIONS)

        s1_concept, s1_frag = SYMBOL_CONCEPTS.get(chosen[0], ("unknown", None))
        s2_concept, s2_frag = SYMBOL_CONCEPTS.get(chosen[1], ("unknown", None))

        prompt = f"""
You are entering REM phase with biometric coherence trace: {coherence}.
Emotional register: {emo.value}.
Generate an EmergentDreamFragment containing these symbols:
- {chosen[0].value} (→ {s1_concept}, mapped to {s1_frag or 'new concept'})
- {chosen[1].value} (→ {s2_concept}, mapped to {s2_frag or 'new concept'})
Format output in symbolic mapping for post-sleep integration.
"""
        return prompt

    @staticmethod
    def generate_expected(coherence: List[float], symbols: List[ArchetypalSymbol],
                         emotion: EmotionFlag) -> dict:
        """Generate expected output schema."""
        symbol_map = []
        for sym in symbols:
            concept, frag = SYMBOL_CONCEPTS.get(sym, ("unknown", None))
            entry = {"symbol": sym.value, "mapped_concept": concept}
            if frag:
                entry["mapped_fragment"] = frag
            symbol_map.append(entry)

        return {
            "fragment_id": f"dream-{random.randint(1000, 9999)}",
            "form": "SYMBOLIC",
            "emergence_phase": "REM",
            "coherence_trace": coherence,
            "emotional_register": emotion.value,
            "symbol_map": symbol_map
        }


# ============================================================================
# Test Scenarios
# ============================================================================

def test_phase_transitions():
    """Test: Sleep phase transitions follow correct order."""
    scheduler = DreamScheduler()
    scheduler.start_session()

    # Should start in WAKE, immediately transition to THETA
    trace = [0.4, 0.45, 0.5]
    phase, _ = scheduler.step(trace)
    assert scheduler.current_phase == SleepPhase.THETA

    # Advance past THETA minimum
    for _ in range(DreamScheduler.THETA_MIN + 1):
        scheduler.step(trace)

    assert scheduler.current_phase == SleepPhase.DELTA
    print("  [PASS] phase_transitions")


def test_rem_timing():
    """Test: REM occurs after φ-cycle (~90 min)."""
    scheduler = DreamScheduler()
    scheduler.start_session()

    trace = [0.4, 0.45, 0.5]

    # Advance through THETA
    for _ in range(DreamScheduler.THETA_MIN + 1):
        scheduler.step(trace)

    # Advance through DELTA (simulate with direct counter set)
    scheduler.rem_counter = DreamScheduler.PHI_CYCLE
    scheduler.step(trace)

    assert scheduler.current_phase == SleepPhase.REM
    print("  [PASS] rem_timing")


def test_fragment_emergence():
    """Test: Fragment emerges during REM with rising coherence."""
    scheduler = DreamScheduler()
    scheduler.start_session()

    # Force into REM
    scheduler.current_phase = SleepPhase.REM
    scheduler.phase_counter = 0

    # Rising coherence above threshold
    trace = [0.41, 0.47, 0.61]
    phase, fragment = scheduler.step(trace)

    assert phase.in_rem == True
    assert fragment is not None
    assert fragment.form == "SYMBOLIC"
    assert len(fragment.symbol_map) >= 2
    print("  [PASS] fragment_emergence")


def test_coherence_gating():
    """Test: Fragment blocked when coherence below threshold."""
    scheduler = DreamScheduler()
    scheduler.start_session()
    scheduler.current_phase = SleepPhase.REM

    # Low coherence trace
    trace = [0.3, 0.35, 0.4]  # Below 0.6 threshold
    _, fragment = scheduler.step(trace)

    assert fragment is None
    print("  [PASS] coherence_gating")


def test_shadow_detection():
    """Test: Shadow symbols trigger shadow_detected flag."""
    scheduler = DreamScheduler()
    scheduler.start_session()
    scheduler.current_phase = SleepPhase.REM

    # Generate fragment and check shadow detection
    trace = [0.5, 0.6, 0.7]

    # Generate multiple times to catch shadow symbol
    shadow_found = False
    for _ in range(20):
        scheduler.phase_counter = 0  # Reset to allow emergence
        _, fragment = scheduler.step(trace)
        if fragment and fragment.shadow_detected:
            shadow_found = True
            break

    # At least one shadow symbol should appear in 20 tries
    assert shadow_found, "Expected shadow symbol in 20 attempts"
    print("  [PASS] shadow_detection")


def test_integration_summary():
    """Test: Post-sleep integration summary generated correctly."""
    scheduler = DreamScheduler()
    scheduler.start_session()
    scheduler.current_phase = SleepPhase.REM

    # Generate a few fragments
    for i in range(3):
        trace = [0.4 + i*0.05, 0.5 + i*0.05, 0.65 + i*0.05]
        scheduler.phase_counter = 0
        scheduler.step(trace)

    summary = scheduler.end_session()

    assert summary.fragments_surfaced >= 1
    assert len(summary.journal_prompts) > 0
    print("  [PASS] integration_summary")


def test_resonance_output():
    """Test: Resonance settings match current phase."""
    scheduler = DreamScheduler()
    scheduler.start_session()

    # THETA phase
    scheduler.current_phase = SleepPhase.THETA
    reso = scheduler.get_resonance()
    assert reso.target_band == FrequencyBand.THETA
    assert reso.audio_type == AudioType.BINAURAL

    # DELTA phase
    scheduler.current_phase = SleepPhase.DELTA
    reso = scheduler.get_resonance()
    assert reso.target_band == FrequencyBand.DELTA
    assert reso.audio_type == AudioType.GOLDEN_STACK

    # REM phase
    scheduler.current_phase = SleepPhase.REM
    reso = scheduler.get_resonance()
    assert reso.target_band == FrequencyBand.ALPHA
    assert reso.visual_type == VisualType.FLOWER_OF_LIFE

    print("  [PASS] resonance_output")


def test_prompt_generator():
    """Test: Dream prompt generator produces valid prompts."""
    prompt = DreamPromptGenerator.generate()

    assert "REM phase" in prompt
    assert "coherence trace" in prompt
    assert "Emotional register" in prompt
    print("  [PASS] prompt_generator")


def test_symbol_mapping():
    """Test: All symbols have valid mappings."""
    for symbol in ArchetypalSymbol:
        concept, _ = SYMBOL_CONCEPTS.get(symbol, (None, None))
        assert concept is not None, f"Missing concept for {symbol}"

    print("  [PASS] symbol_mapping")


# ============================================================================
# Patch 13A Test Scenarios
# ============================================================================

def test_hybrid_phase_detection():
    """Test: Hybrid EEG/HRV phase detection with confidence scoring (13A)."""
    # Test Delta phase detection
    delta_bio = BiometricInput(
        delta_power=0.7,
        theta_power=0.15,
        alpha_power=0.1,
        beta_power=0.05,
        hrv_coherence=0.65
    )
    result = PhaseDetector.detect_phase(delta_bio)
    assert result.detected_phase == SleepPhase.DELTA
    assert result.eeg_dominant == "delta"
    assert result.confidence >= PhaseDetector.CONFIDENCE_THRESHOLD

    # Test REM phase detection
    rem_bio = BiometricInput(
        delta_power=0.1,
        theta_power=0.2,
        alpha_power=0.55,
        beta_power=0.15,
        hrv_rmssd=45
    )
    result = PhaseDetector.detect_phase(rem_bio)
    assert result.detected_phase == SleepPhase.REM
    assert result.eeg_dominant == "alpha"

    # Test low confidence detection
    ambiguous_bio = BiometricInput(
        delta_power=0.25,
        theta_power=0.28,
        alpha_power=0.24,
        beta_power=0.23
    )
    result = PhaseDetector.detect_phase(ambiguous_bio)
    assert result.confidence < PhaseDetector.CONFIDENCE_THRESHOLD

    print("  [PASS] hybrid_phase_detection (13A)")


def test_shadow_consent_bridge():
    """Test: Shadow consent gating through Prompt 12 bridge (13A)."""
    # Test high coherence allows shadow access
    result = ShadowConsentBridge.evaluate_shadow_symbol(
        ArchetypalSymbol.MIRROR,
        coherence=0.72,
        has_operator=True
    )
    assert result.allowed == True
    assert result.gating_reason == "ALLOW"

    # Test low coherence blocks shadow access
    result = ShadowConsentBridge.evaluate_shadow_symbol(
        ArchetypalSymbol.CAVE,
        coherence=0.5,
        has_operator=True
    )
    assert result.allowed == False
    assert "COHERENCE" in result.gating_reason

    print("  [PASS] shadow_consent_bridge (13A)")


def test_fragment_storage():
    """Test: Fragment storage backend saves and loads correctly (13A)."""
    import tempfile
    import shutil

    # Create temp directory for test
    temp_dir = tempfile.mkdtemp()
    try:
        storage = FragmentStorage(base_path=temp_dir, user_id="test_user")

        # Create test fragment
        fragment = EmergentDreamFragment(
            fragment_id="dream-test-001",
            form="SYMBOLIC",
            emergence_phase=SleepPhase.REM,
            coherence_trace=[0.4, 0.5, 0.65],
            emotional_register=EmotionalRegister(EmotionFlag.AWE, EmotionFlag.PEACE),
            symbol_map=[
                SymbolMapping(ArchetypalSymbol.OWL, "F13", "wisdom"),
                SymbolMapping(ArchetypalSymbol.STAR, "F77", "guidance")
            ],
            shadow_detected=False,
            timestamp=datetime.now().timestamp()
        )

        # Save fragment
        assert storage.save_fragment(fragment) == True

        # Verify file exists
        assert storage.session_path.exists()

        # Load and verify content
        with open(storage.session_path) as f:
            data = json.load(f)

        assert len(data["fragments"]) == 1
        assert data["fragments"][0]["fragment_id"] == "dream-test-001"

        print("  [PASS] fragment_storage (13A)")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_symbol_linker():
    """Test: Symbol-to-fragment dynamic linking with cosine similarity (13A)."""
    # Test cosine similarity calculation
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    assert abs(SymbolLinker.cosine_similarity(vec_a, vec_b) - 1.0) < 0.001

    vec_c = [1.0, 0.0, 0.0]
    vec_d = [0.0, 1.0, 0.0]
    assert abs(SymbolLinker.cosine_similarity(vec_c, vec_d)) < 0.001

    # Test self-similarity of symbols
    owl_vec = SYMBOL_VECTORS[ArchetypalSymbol.OWL]
    self_sim = SymbolLinker.cosine_similarity(owl_vec, owl_vec)
    assert abs(self_sim - 1.0) < 0.001

    # Test finding linked fragments
    fragment_vectors = {
        "F13": [0.85, 0.25, 0.15, 0.75, 0.35, 0.15, 0.65, 0.45],  # Similar to OWL
        "F99": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]  # Dissimilar
    }
    matches = SymbolLinker.find_linked_fragments(ArchetypalSymbol.OWL, fragment_vectors)

    # F13 should match, F99 should not
    match_ids = [m[0] for m in matches]
    assert "F13" in match_ids

    print("  [PASS] symbol_linker (13A)")


def test_lucid_tracker():
    """Test: Lucid dream marker tracking (13A)."""
    tracker = LucidTracker()
    timestamp = 0.0

    # Simulate gamma spike sequence
    for i in range(5):
        bio = BiometricInput(gamma_power=0.30)  # Above threshold
        marker = tracker.process_sample(bio, SleepPhase.REM, 0.7, timestamp)
        timestamp += 1.0
        assert marker is None  # Still in spike

    # End spike
    bio = BiometricInput(gamma_power=0.1)  # Below threshold
    marker = tracker.process_sample(bio, SleepPhase.REM, 0.7, timestamp)

    # Should have registered marker (5s > 3s minimum)
    assert marker is not None
    assert marker.gamma_power == 0.30
    assert marker.duration_seconds == 5.0
    assert marker.phase_when_detected == SleepPhase.REM

    # Verify session markers
    markers = tracker.get_session_markers()
    assert len(markers) == 1

    print("  [PASS] lucid_tracker (13A)")


def test_hardware_interface_stubs():
    """Test: Hardware interface stubs return valid data (13A)."""
    # Test EEG reading stub
    bio = HardwareInterface.read_eeg_bands()
    assert 0 <= bio.delta_power <= 1
    assert 0 <= bio.theta_power <= 1
    assert 0 <= bio.alpha_power <= 1
    assert 0 <= bio.beta_power <= 1
    assert 0 <= bio.gamma_power <= 1

    # Test HRV reading stub
    coherence, rmssd = HardwareInterface.read_hrv_coherence()
    assert 0 <= coherence <= 1
    assert rmssd > 0

    # Test output stubs (should return True)
    assert HardwareInterface.output_audio_entrainment(10.0, AudioType.BINAURAL) == True
    assert HardwareInterface.output_visual_entrainment(VisualType.FLOWER_OF_LIFE, 0.5) == True
    assert HardwareInterface.trigger_haptic_pulse(0.7, 100) == True

    print("  [PASS] hardware_interface_stubs (13A)")


# ============================================================================
# CLI Dashboard
# ============================================================================

def print_dashboard(scheduler: DreamScheduler, fragment: Optional[EmergentDreamFragment] = None):
    """Print CLI dashboard for dream scheduler."""
    width = 70

    print("=" * width)
    print(" SCALAR DREAM INDUCTION - Session Active".center(width))
    print("=" * width)

    # Current phase
    phase_icons = {
        SleepPhase.WAKE: "WAKE",
        SleepPhase.THETA: "THETA (Light Sleep)",
        SleepPhase.DELTA: "DELTA (Deep Sleep)",
        SleepPhase.REM: "REM (Dreaming)"
    }
    print(f"\n Current Phase: {phase_icons.get(scheduler.current_phase, 'UNKNOWN')}")
    print(f" Cycle: {scheduler.cycle_number}")
    print(f" Phase Duration: {scheduler.phase_counter}s")
    print(f" REM Counter: {scheduler.rem_counter}/{scheduler.PHI_CYCLE}s")

    # Resonance
    reso = scheduler.get_resonance()
    print(f"\n Resonance Entrainment:")
    print(f"   Band: {reso.target_band.name} ({reso.base_frequency:.1f} Hz)")
    print(f"   Audio: {reso.audio_type.value}")
    print(f"   Visual: {reso.visual_type.value}")
    print(f"   phi^{reso.phi_multiplier} modulation")

    # Fragment (if any)
    if fragment:
        print(f"\n [FRAGMENT] Emerged: {fragment.fragment_id}")
        print(f"   Coherence: {fragment.coherence_trace}")
        print(f"   Emotions: {fragment.emotional_register.primary.value} + {fragment.emotional_register.secondary.value}")
        print(f"   Symbols:")
        for sm in fragment.symbol_map:
            frag_str = f" -> {sm.mapped_fragment}" if sm.mapped_fragment else ""
            print(f"     {sm.symbol.value}: {sm.mapped_concept}{frag_str}")
        if fragment.shadow_detected:
            print("   [!] Shadow content detected (consent-gated)")

    # Fragments surfaced
    print(f"\n Fragments this session: {len(scheduler.fragments)}")

    print("=" * width)


def run_simulation(cycles: int = 10):
    """Run dream session simulation."""
    print("\n" + "=" * 70)
    print(" DREAM SESSION SIMULATION".center(70))
    print("=" * 70)

    scheduler = DreamScheduler()
    scheduler.start_session()

    for i in range(cycles):
        # Generate rising coherence trace
        base = 0.35 + (i * 0.03)
        trace = [base, base + 0.05, base + 0.1 + random.uniform(0, 0.15)]

        phase, fragment = scheduler.step(trace)

        # Print status every few cycles or on fragment
        if i % 3 == 0 or fragment:
            print(f"\n--- Cycle {i} ---")
            print(f"Phase: {scheduler.current_phase.name}")
            print(f"Coherence: {[round(t, 2) for t in trace]}")
            if fragment:
                print(f"* Fragment emerged: {fragment.fragment_id}")
                for sm in fragment.symbol_map:
                    print(f"   {sm.symbol.value} -> {sm.mapped_concept}")

    # End session
    summary = scheduler.end_session()

    print("\n" + "=" * 70)
    print(" POST-SLEEP INTEGRATION SUMMARY".center(70))
    print("=" * 70)
    print(f"\n Fragments surfaced: {summary.fragments_surfaced}")
    print(f" Dominant symbols: {[s.value for s in summary.dominant_symbols]}")
    print(f" Coherence delta: {summary.coherence_delta:+.3f}")
    print(f" Shadow content: {'Yes (gated)' if summary.shadow_gated else 'No'}")
    print(f"\n Journal prompts:")
    for prompt in summary.journal_prompts:
        print(f"   • {prompt}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests and optional simulation."""
    print("\n" + "=" * 70)
    print(" PROMPT 13: SCALAR DREAM INDUCTION - TEST SUITE (Patch 13A)".center(70))
    print("=" * 70)
    print()

    tests = [
        # Core tests
        ("Phase Transitions", test_phase_transitions),
        ("REM Timing (phi-cycle)", test_rem_timing),
        ("Fragment Emergence", test_fragment_emergence),
        ("Coherence Gating", test_coherence_gating),
        ("Shadow Detection", test_shadow_detection),
        ("Integration Summary", test_integration_summary),
        ("Resonance Output", test_resonance_output),
        ("Prompt Generator", test_prompt_generator),
        ("Symbol Mapping", test_symbol_mapping),
        # Patch 13A tests
        ("Hybrid Phase Detection (13A)", test_hybrid_phase_detection),
        ("Shadow Consent Bridge (13A)", test_shadow_consent_bridge),
        ("Fragment Storage (13A)", test_fragment_storage),
        ("Symbol Linker (13A)", test_symbol_linker),
        ("Lucid Tracker (13A)", test_lucid_tracker),
        ("Hardware Interface Stubs (13A)", test_hardware_interface_stubs),
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
        run_simulation(20)

    # Demo dashboard
    if "--demo" in sys.argv or len(sys.argv) == 1:
        print("\n\n")
        scheduler = DreamScheduler()
        scheduler.start_session()
        scheduler.current_phase = SleepPhase.REM

        trace = [0.41, 0.47, 0.61]
        _, fragment = scheduler.step(trace)

        print_dashboard(scheduler, fragment)

    # JSON output
    if "--json" in sys.argv:
        scheduler = DreamScheduler()
        scheduler.start_session()
        scheduler.current_phase = SleepPhase.REM

        trace = [0.41, 0.47, 0.61]
        _, fragment = scheduler.step(trace)

        if fragment:
            result = {
                "fragment_id": fragment.fragment_id,
                "form": fragment.form,
                "emergence_phase": fragment.emergence_phase.name,
                "coherence_trace": fragment.coherence_trace,
                "emotional_register": f"{fragment.emotional_register.primary.value} + {fragment.emotional_register.secondary.value}",
                "symbol_map": [
                    {"symbol": sm.symbol.value, "mapped_concept": sm.mapped_concept,
                     "mapped_fragment": sm.mapped_fragment}
                    for sm in fragment.symbol_map
                ],
                "shadow_detected": fragment.shadow_detected
            }
            print(json.dumps(result, indent=2))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
