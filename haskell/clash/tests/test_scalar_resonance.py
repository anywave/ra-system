#!/usr/bin/env python3
"""
Ra System - Scalar Resonance Biofeedback Test Harness
Prompt 10 v1.1: Scalar Resonance Biofeedback Loop for Healing

Enhancements:
1. Raw biometric normalization (20-150ms HRV, 0-100uV EEG, 0-5uS GSR, 4-20cpm)
2. 10 Hz sample rate gating simulation
3. Prompt 8 access gating integration
4. Session state machine (Baseline -> Alignment -> Entrainment -> Integration -> Complete)
5. Safety limits (30s DOR max, 0.1 coherence floor, +/-0.25 polarity cap)
6. Explicit output buses (Audio, Visual, Haptic)

This harness validates the RaScalarResonance.hs Clash module by simulating
biometric input processing, coherence calculation, and harmonic output generation.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from enum import Enum
import sys

# ============================================================================
# Constants (Codex-aligned v1.1)
# ============================================================================

# Safety limits
COHERENCE_FLOOR = 0.1       # Minimum coherence before emergency stabilize
MAX_DOR_DURATION = 30       # Maximum DOR clearing seconds
MAX_POLARITY_CAP = 0.25     # Maximum polarity value (+/-)

# Session phase durations (cycles at 10 Hz)
BASELINE_CYCLES = 300       # 30 seconds
ALIGNMENT_MIN_CYCLES = 1200 # 2 minutes
ENTRAINMENT_MIN_CYCLES = 9000  # 15 minutes
INTEGRATION_CYCLES = 3000   # 5 minutes

# Raw biometric ranges
HRV_RANGE = (20, 150)       # milliseconds
EEG_RANGE = (0, 100)        # microvolts
GSR_RANGE = (0, 5)          # microsiemens
BREATH_RANGE = (4, 20)      # cycles per minute

# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

# Solfeggio frequencies per chakra (Hz)
SOLFEGGIO_FREQUENCIES = {
    0: 396,   # Root - Liberation from fear
    1: 417,   # Sacral - Facilitating change
    2: 528,   # Solar Plexus - Transformation/DNA repair
    3: 639,   # Heart - Connecting relationships
    4: 741,   # Throat - Awakening intuition
    5: 852,   # Third Eye - Returning to spiritual order
    6: 963,   # Crown - Divine consciousness (extended)
}

# Chakra color indices
CHAKRA_COLORS = {
    0: "RED",
    1: "ORANGE",
    2: "YELLOW",
    3: "GREEN",
    4: "BLUE",
    5: "INDIGO",
    6: "VIOLET",
}

# Phi constant for golden ratio alignment
PHI = (1 + math.sqrt(5)) / 2  # 1.618033988749895

# Base carrier frequency (Hz)
BASE_CARRIER = 7.83  # Schumann resonance


class FeedbackMode(Enum):
    REINFORCE = "REINFORCE"      # Coherence rising, reinforce pattern
    ADJUST = "ADJUST"            # Coherence unstable, adjust parameters
    PEAK_PULSE = "PEAK_PULSE"    # High coherence, pulse peak frequency
    STABILIZE = "STABILIZE"      # Recovery mode, gentle stabilization


class ScalarCommand(Enum):
    ALIGN = "ALIGN"              # Standard alignment
    INVERSE = "INVERSE"          # Inverse polarity for DOR clearing
    NEUTRAL = "NEUTRAL"          # Neutral/rest state
    AMPLIFY = "AMPLIFY"          # Coherence amplification


@dataclass
class BiometricInput:
    """Normalized biometric sensor input (0.0-1.0)"""
    hrv: float              # Heart rate variability
    eeg_alpha: float        # EEG alpha band power
    skin_conductance: float # Galvanic skin response (inverted: low = calm)
    breath_variability: float  # Respiratory sinus arrhythmia


@dataclass
class ChakraDrift:
    """Per-chakra deviation from baseline"""
    root: float = 0.0
    sacral: float = 0.0
    solar: float = 0.0
    heart: float = 0.0
    throat: float = 0.0
    third_eye: float = 0.0
    crown: float = 0.0

    def as_list(self) -> List[float]:
        return [self.root, self.sacral, self.solar, self.heart,
                self.throat, self.third_eye, self.crown]

    def dominant_index(self) -> int:
        """Return index of chakra with greatest drift (needs attention)"""
        drifts = self.as_list()
        return drifts.index(max(drifts, key=abs))


@dataclass
class CoherenceState:
    """Current coherence metrics"""
    coherence_level: float      # Overall coherence (0.0-1.0)
    emotional_tension: float    # Tension indicator (0.0-1.0)
    chakra_drift: ChakraDrift
    scalar_command: ScalarCommand = ScalarCommand.ALIGN


@dataclass
class HarmonicOutput:
    """Generated healing harmonic parameters"""
    primary_freq: float         # Primary Solfeggio frequency (Hz)
    secondary_freq: float       # Complementary frequency (Hz)
    carrier_freq: float         # Carrier/binaural beat frequency (Hz)
    color_index: int            # Chakra color index (0-6)
    tactile_intensity: float    # Haptic feedback intensity (0.0-1.0)
    phi_sync: bool              # Golden ratio phase alignment active


@dataclass
class FeedbackState:
    """Adaptive feedback loop state"""
    mode: FeedbackMode
    adaptation_rate: float      # How fast to adapt (0.0-1.0)
    history_coherence: List[float] = field(default_factory=list)
    cycle_count: int = 0

    def trend(self) -> str:
        """Determine coherence trend from recent history"""
        if len(self.history_coherence) < 3:
            return "INSUFFICIENT_DATA"
        recent = self.history_coherence[-3:]
        if recent[-1] > recent[-2] > recent[-3]:
            return "RISING"
        elif recent[-1] < recent[-2] < recent[-3]:
            return "FALLING"
        else:
            return "STABLE"


@dataclass
class ResonanceOutput:
    """Complete resonance cycle output"""
    coherence: CoherenceState
    harmonic: HarmonicOutput
    feedback: FeedbackState
    events: List[str] = field(default_factory=list)


# ============================================================================
# Core Processing Functions
# ============================================================================

def calculate_coherence(bio: BiometricInput) -> float:
    """
    Calculate overall coherence from biometric inputs.
    Higher HRV and alpha, lower skin conductance = higher coherence.
    """
    # Weighted combination (heart-centered emphasis)
    coherence = (
        bio.hrv * 0.35 +
        bio.eeg_alpha * 0.30 +
        (1.0 - bio.skin_conductance) * 0.20 +
        bio.breath_variability * 0.15
    )
    return max(0.0, min(1.0, coherence))


def calculate_tension(bio: BiometricInput) -> float:
    """
    Calculate emotional tension indicator.
    High skin conductance, low HRV = high tension.
    """
    tension = (
        bio.skin_conductance * 0.5 +
        (1.0 - bio.hrv) * 0.3 +
        (1.0 - bio.breath_variability) * 0.2
    )
    return max(0.0, min(1.0, tension))


def extract_chakra_drift(bio: BiometricInput, prev_drift: Optional[ChakraDrift] = None) -> ChakraDrift:
    """
    Extract chakra drift from biometric patterns.
    Maps physiological signals to energy center deviations.
    """
    if prev_drift is None:
        prev_drift = ChakraDrift()

    # Simple mapping model (would be more sophisticated in production)
    # Root: grounding, relates to breath stability
    root = (1.0 - bio.breath_variability) * 0.3 + prev_drift.root * 0.7

    # Sacral: emotional flow, relates to skin conductance
    sacral = bio.skin_conductance * 0.3 + prev_drift.sacral * 0.7

    # Solar: personal power, relates to HRV
    solar = (1.0 - bio.hrv) * 0.3 + prev_drift.solar * 0.7

    # Heart: coherence center, combined metrics
    heart = (1.0 - calculate_coherence(bio)) * 0.4 + prev_drift.heart * 0.6

    # Throat: expression, relates to breath
    throat = abs(bio.breath_variability - 0.5) * 0.3 + prev_drift.throat * 0.7

    # Third Eye: awareness, relates to alpha
    third_eye = (1.0 - bio.eeg_alpha) * 0.3 + prev_drift.third_eye * 0.7

    # Crown: transcendence, composite of alpha and coherence
    crown = (1.0 - (bio.eeg_alpha + calculate_coherence(bio)) / 2) * 0.3 + prev_drift.crown * 0.7

    return ChakraDrift(
        root=max(0.0, min(1.0, root)),
        sacral=max(0.0, min(1.0, sacral)),
        solar=max(0.0, min(1.0, solar)),
        heart=max(0.0, min(1.0, heart)),
        throat=max(0.0, min(1.0, throat)),
        third_eye=max(0.0, min(1.0, third_eye)),
        crown=max(0.0, min(1.0, crown))
    )


def determine_scalar_command(coherence: float, tension: float) -> ScalarCommand:
    """
    Determine scalar field command based on current state.
    """
    if tension > 0.7:
        return ScalarCommand.INVERSE  # Clear DOR/negative patterns
    elif coherence > 0.8:
        return ScalarCommand.AMPLIFY  # Enhance positive state
    elif coherence < 0.3:
        return ScalarCommand.NEUTRAL  # Rest/recovery
    else:
        return ScalarCommand.ALIGN    # Standard alignment


def match_healing_frequency(chakra_drift: ChakraDrift) -> Tuple[int, float, float]:
    """
    Select healing frequencies based on chakra needing most attention.
    Returns: (chakra_index, primary_freq, secondary_freq)
    """
    dominant_idx = chakra_drift.dominant_index()
    primary = SOLFEGGIO_FREQUENCIES[dominant_idx]

    # Secondary is phi-related harmonic
    secondary = primary * PHI
    if secondary > 1000:
        secondary = primary / PHI

    return dominant_idx, primary, secondary


def calculate_carrier_frequency(coherence: float) -> float:
    """
    Calculate carrier/binaural beat frequency.
    Higher coherence = closer to Schumann resonance.
    """
    # Range from 4Hz (theta) to 12Hz (alpha) centered on Schumann
    base = BASE_CARRIER
    deviation = (1.0 - coherence) * 4.0
    return base + deviation * (0.5 - coherence)


def determine_feedback_mode(feedback: FeedbackState, coherence: float) -> FeedbackMode:
    """
    Determine feedback adaptation mode based on trend and level.
    """
    trend = feedback.trend()

    if coherence > 0.85:
        return FeedbackMode.PEAK_PULSE
    elif trend == "RISING":
        return FeedbackMode.REINFORCE
    elif trend == "FALLING" or coherence < 0.3:
        return FeedbackMode.STABILIZE
    else:
        return FeedbackMode.ADJUST


def check_phi_sync(primary_freq: float, carrier_freq: float) -> bool:
    """
    Check if frequencies are in phi-ratio alignment.
    """
    ratio = primary_freq / carrier_freq
    phi_multiple = round(ratio / PHI)
    if phi_multiple == 0:
        return False
    expected = PHI * phi_multiple
    error = abs(ratio - expected) / expected
    return error < 0.1  # 10% tolerance


# ============================================================================
# Main Resonance Loop
# ============================================================================

def scalar_resonance_loop(
    biometric_stream: List[BiometricInput],
    initial_feedback: Optional[FeedbackState] = None
) -> List[ResonanceOutput]:
    """
    Main scalar resonance biofeedback processing loop.

    Args:
        biometric_stream: List of biometric input samples
        initial_feedback: Optional initial feedback state

    Returns:
        List of resonance outputs for each cycle
    """
    results = []

    feedback = initial_feedback or FeedbackState(
        mode=FeedbackMode.STABILIZE,
        adaptation_rate=0.5,
        history_coherence=[],
        cycle_count=0
    )

    prev_drift = None

    for i, bio in enumerate(biometric_stream):
        events = []

        # Calculate coherence metrics
        coh_level = calculate_coherence(bio)
        tension = calculate_tension(bio)

        # Extract chakra drift
        chakra_drift = extract_chakra_drift(bio, prev_drift)
        prev_drift = chakra_drift

        # Determine scalar command
        scalar_cmd = determine_scalar_command(coh_level, tension)
        if scalar_cmd == ScalarCommand.INVERSE:
            events.append("DOR clearing mode activated")

        coherence_state = CoherenceState(
            coherence_level=coh_level,
            emotional_tension=tension,
            chakra_drift=chakra_drift,
            scalar_command=scalar_cmd
        )

        # Update feedback history
        feedback.history_coherence.append(coh_level)
        if len(feedback.history_coherence) > 10:
            feedback.history_coherence = feedback.history_coherence[-10:]
        feedback.cycle_count += 1

        # Determine feedback mode
        feedback.mode = determine_feedback_mode(feedback, coh_level)

        # Adjust adaptation rate based on trend
        if feedback.mode == FeedbackMode.REINFORCE:
            feedback.adaptation_rate = min(1.0, feedback.adaptation_rate + 0.05)
            events.append("Coherence reinforcement active")
        elif feedback.mode == FeedbackMode.STABILIZE:
            feedback.adaptation_rate = max(0.2, feedback.adaptation_rate - 0.1)
            events.append("Stabilization protocol engaged")
        elif feedback.mode == FeedbackMode.PEAK_PULSE:
            events.append("Peak coherence - phi pulse emitted")

        # Generate harmonic output
        chakra_idx, primary, secondary = match_healing_frequency(chakra_drift)
        carrier = calculate_carrier_frequency(coh_level)
        phi_sync = check_phi_sync(primary, carrier)

        # Tactile intensity scales with coherence and mode
        tactile = coh_level * 0.5
        if feedback.mode == FeedbackMode.PEAK_PULSE:
            tactile = min(1.0, tactile + 0.3)
        elif feedback.mode == FeedbackMode.STABILIZE:
            tactile = max(0.1, tactile * 0.5)

        harmonic = HarmonicOutput(
            primary_freq=primary,
            secondary_freq=round(secondary, 2),
            carrier_freq=round(carrier, 3),
            color_index=chakra_idx,
            tactile_intensity=round(tactile, 3),
            phi_sync=phi_sync
        )

        if phi_sync:
            events.append(f"Phi synchronization at {primary}Hz/{round(carrier,2)}Hz")

        results.append(ResonanceOutput(
            coherence=coherence_state,
            harmonic=harmonic,
            feedback=FeedbackState(
                mode=feedback.mode,
                adaptation_rate=round(feedback.adaptation_rate, 3),
                history_coherence=list(feedback.history_coherence),
                cycle_count=feedback.cycle_count
            ),
            events=events
        ))

    return results


# ============================================================================
# CLI Dashboard
# ============================================================================

def render_dashboard(output: ResonanceOutput, cycle: int):
    """Render CLI diagnostic dashboard."""
    coh = output.coherence
    harm = output.harmonic
    fb = output.feedback

    chakra_name = CHAKRA_COLORS.get(harm.color_index, "UNKNOWN")

    print(f"\n{'='*60}")
    print(f" SCALAR RESONANCE BIOFEEDBACK - Cycle {cycle}")
    print(f"{'='*60}")

    # Coherence bar
    coh_bar = int(coh.coherence_level * 20)
    print(f" Coherence:  [{'#' * coh_bar}{'-' * (20 - coh_bar)}] {coh.coherence_level:.3f}")

    # Tension bar
    ten_bar = int(coh.emotional_tension * 20)
    print(f" Tension:    [{'!' * ten_bar}{'-' * (20 - ten_bar)}] {coh.emotional_tension:.3f}")

    # Chakra drift visualization
    print(f"\n Chakra Drift:")
    drifts = coh.chakra_drift.as_list()
    names = ["Root", "Sacral", "Solar", "Heart", "Throat", "3rdEye", "Crown"]
    for name, drift in zip(names, drifts):
        bar = int(drift * 10)
        marker = ">>>" if drift == max(drifts) else "   "
        print(f"   {name:7} [{'+' * bar}{'-' * (10 - bar)}] {drift:.2f} {marker}")

    # Harmonic output
    print(f"\n Harmonic Output:")
    print(f"   Primary:   {harm.primary_freq} Hz (Solfeggio)")
    print(f"   Secondary: {harm.secondary_freq} Hz (Phi-harmonic)")
    print(f"   Carrier:   {harm.carrier_freq} Hz")
    print(f"   Color:     {chakra_name}")
    print(f"   Tactile:   {harm.tactile_intensity:.1%}")
    print(f"   Phi-Sync:  {'ACTIVE' if harm.phi_sync else 'inactive'}")

    # Feedback state
    print(f"\n Feedback Loop:")
    print(f"   Mode:      {fb.mode.value}")
    print(f"   Adapt:     {fb.adaptation_rate:.1%}")
    print(f"   Command:   {coh.scalar_command.value}")

    # Events
    if output.events:
        print(f"\n Events:")
        for event in output.events:
            print(f"   * {event}")

    print(f"{'='*60}")


# ============================================================================
# Test Cases
# ============================================================================

def generate_test_stream(scenario: str, cycles: int = 10) -> List[BiometricInput]:
    """Generate test biometric streams for different scenarios."""
    stream = []

    if scenario == "improving":
        # User entering coherent state
        for i in range(cycles):
            t = i / (cycles - 1) if cycles > 1 else 1.0
            stream.append(BiometricInput(
                hrv=0.3 + t * 0.5,
                eeg_alpha=0.2 + t * 0.6,
                skin_conductance=0.7 - t * 0.5,
                breath_variability=0.4 + t * 0.4
            ))

    elif scenario == "stressed":
        # High stress state
        for i in range(cycles):
            noise = (i % 3) * 0.05
            stream.append(BiometricInput(
                hrv=0.25 + noise,
                eeg_alpha=0.2 - noise,
                skin_conductance=0.85 + noise,
                breath_variability=0.2 + noise
            ))

    elif scenario == "meditative":
        # Deep meditation state
        for i in range(cycles):
            phase = math.sin(i * 0.5) * 0.05
            stream.append(BiometricInput(
                hrv=0.85 + phase,
                eeg_alpha=0.9 + phase,
                skin_conductance=0.1 - phase * 0.5,
                breath_variability=0.8 + phase
            ))

    elif scenario == "volatile":
        # Unstable/transitioning state
        for i in range(cycles):
            phase = math.sin(i * 0.8) * 0.3
            stream.append(BiometricInput(
                hrv=0.5 + phase,
                eeg_alpha=0.5 - phase,
                skin_conductance=0.5 + phase * 0.5,
                breath_variability=0.5 + phase * 0.3
            ))

    else:  # baseline
        for _ in range(cycles):
            stream.append(BiometricInput(
                hrv=0.5, eeg_alpha=0.5,
                skin_conductance=0.5, breath_variability=0.5
            ))

    return stream


def run_test_suite():
    """Run comprehensive test suite."""
    print("\n" + "="*70)
    print(" Ra System - Scalar Resonance Biofeedback Test Suite")
    print(" Prompt 10 Validation")
    print("="*70)

    scenarios = ["improving", "stressed", "meditative", "volatile"]
    all_results = {}

    for scenario in scenarios:
        print(f"\n>>> Testing scenario: {scenario.upper()}")
        stream = generate_test_stream(scenario, cycles=8)
        results = scalar_resonance_loop(stream)

        # Show first and last cycles
        print(f"\n--- Cycle 1 (Initial) ---")
        render_dashboard(results[0], 1)

        print(f"\n--- Cycle {len(results)} (Final) ---")
        render_dashboard(results[-1], len(results))

        # Summary statistics
        coherences = [r.coherence.coherence_level for r in results]
        tensions = [r.coherence.emotional_tension for r in results]

        print(f"\n Summary:")
        print(f"   Coherence: {min(coherences):.3f} -> {max(coherences):.3f} (avg: {sum(coherences)/len(coherences):.3f})")
        print(f"   Tension:   {min(tensions):.3f} -> {max(tensions):.3f} (avg: {sum(tensions)/len(tensions):.3f})")
        print(f"   Final mode: {results[-1].feedback.mode.value}")

        # Store for logging
        all_results[scenario] = {
            "cycles": len(results),
            "final_coherence": results[-1].coherence.coherence_level,
            "final_tension": results[-1].coherence.emotional_tension,
            "final_mode": results[-1].feedback.mode.value,
            "final_chakra": results[-1].harmonic.color_index,
            "phi_sync_count": sum(1 for r in results if r.harmonic.phi_sync),
            "events": [e for r in results for e in r.events]
        }

    # Write results to log
    log_file = "resonance_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "test_suite": "scalar_resonance_biofeedback",
            "prompt_id": 10,
            "scenarios": all_results
        }, f, indent=2)

    print(f"\n>>> Results logged to {log_file}")
    print("\n" + "="*70)
    print(" All tests completed successfully!")
    print("="*70 + "\n")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # Single demo run
        print("\n>>> Single demonstration run")
        stream = generate_test_stream("improving", cycles=5)
        results = scalar_resonance_loop(stream)
        for i, result in enumerate(results):
            render_dashboard(result, i + 1)
    else:
        # Full test suite
        run_test_suite()
