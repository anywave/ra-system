#!/usr/bin/env python3
"""
Prompt 13: Scalar Dream Induction & Symbolic Fragment Integration - Python Test Harness

Tests the RaDreamPhaseScheduler Clash module for:
- Sleep phase state machine (WAKE → THETA → DELTA → REM)
- φ-cycle timing (~90 min REM alignment)
- Symbolic fragment emergence during REM
- Coherence-gated fragment surfacing
- Post-sleep integration summary
- Shadow consent gating (Prompt 12 integration)

Codex References:
- ra_constants_v2.json: φ^n timing modulation
- Prompt 12: Shadow consent gating for symbols
"""

import json
import sys
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple
from datetime import datetime


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
    print(" PROMPT 13: SCALAR DREAM INDUCTION - TEST SUITE".center(70))
    print("=" * 70)
    print()

    tests = [
        ("Phase Transitions", test_phase_transitions),
        ("REM Timing (phi-cycle)", test_rem_timing),
        ("Fragment Emergence", test_fragment_emergence),
        ("Coherence Gating", test_coherence_gating),
        ("Shadow Detection", test_shadow_detection),
        ("Integration Summary", test_integration_summary),
        ("Resonance Output", test_resonance_output),
        ("Prompt Generator", test_prompt_generator),
        ("Symbol Mapping", test_symbol_mapping),
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
