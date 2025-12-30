#!/usr/bin/env python3
"""
Orgone Scalar Stability Test Harness
=====================================

Python harness for testing RaOrgoneScalar Clash module.
Validates Reichian orgone dynamics with scalar field modulation.

Prompt 9: Orgone Field Influence on Ra Scalar Stability

Clarifications Implemented:
- Accumulation loop over N cycles with saturation curves
- Emotion-to-stress DOR injection from biometric input
- Access gating precondition from Prompt 8 (RaSympatheticHarmonic)
- CLI trace output for phase visualization
- Base inversion: 13/256 ≈ 0.05078 (Codex-aligned)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import json
import math


# =============================================================================
# Constants (Codex-Aligned)
# =============================================================================

BASE_INVERSION = 13 / 256  # 0.05078 - Codex harmonic constant
STRESS_WEIGHT = 0.3        # DOR injection weight from emotional stress
DECAY_RATE = 0.02          # OR decay rate per cycle (optional)
LUMINESCENCE_THRESHOLD = 0.85
SHADOW_THRESHOLD = 0.60


# =============================================================================
# Access Gate (Prompt 8 Integration)
# =============================================================================

@dataclass
class AccessResult:
    """Result from Prompt 8: Sympathetic Harmonic Fragment Access."""
    valid: bool
    access_type: str  # "FULL", "PARTIAL", "BLOCKED", "SHADOW"
    match_score: float
    resonance_locked: bool = False


def check_access_gate(access: Optional[AccessResult]) -> Tuple[bool, str]:
    """
    Precondition check: Prompt 8 access must be valid before Prompt 9 evaluation.

    Returns:
        (can_proceed, message)
    """
    if access is None:
        return True, "No access gate configured - proceeding"

    if not access.valid:
        return False, "Scalar evaluation deferred: access gate not passed"

    if access.access_type == "BLOCKED":
        return False, "Scalar evaluation blocked: fragment access denied"

    if access.access_type == "SHADOW":
        return True, "Warning: Shadow access - emergence may be unstable"

    return True, f"Access gate passed: {access.access_type} (score={access.match_score:.2f})"


# =============================================================================
# Orgone Field Model
# =============================================================================

@dataclass
class OrgoneField:
    """Orgone field state with geometry modifiers."""
    or_level: float
    dor_level: float
    accumulation_rate: float = 0.04
    chamber_geometry: str = "rectangular"
    _geometry_applied: bool = field(default=False, repr=False)

    def __post_init__(self):
        if not self._geometry_applied:
            self.apply_geometry_modifiers()
            self._geometry_applied = True

    def apply_geometry_modifiers(self):
        """Apply chamber geometry OR boost and DOR shield."""
        modifiers = {
            "pyramidal": {"or_boost": 0.15, "dor_penalty": -0.10},
            "dome": {"or_boost": 0.10, "dor_penalty": -0.05},
            "rectangular": {"or_boost": 0.0, "dor_penalty": 0.0},
            "spherical": {"or_boost": 0.12, "dor_penalty": -0.08}
        }
        geo = modifiers.get(self.chamber_geometry, modifiers["rectangular"])
        self.or_level = min(1.0, max(0.0, self.or_level + geo["or_boost"]))
        self.dor_level = min(1.0, max(0.0, self.dor_level + geo["dor_penalty"]))

    def inject_stress(self, emotional_stress: float):
        """
        Inject emotional stress into DOR field.

        Input: emotional_stress normalized 0-1 (from HRV, GSR, EEG)
        Formula: dor_level += stress_weight * emotional_stress
        """
        dor_injection = STRESS_WEIGHT * emotional_stress
        self.dor_level = min(1.0, self.dor_level + dor_injection)

    def accumulate(self, delta_t: float = 1.0) -> float:
        """
        Accumulate OR over time with exponential approach to saturation.

        Formula: or_level_t+1 = min(1.0, or_level_t + accumulation_rate * delta_t * headroom)
        Returns: new OR level
        """
        headroom = 1.0 - self.or_level
        increment = self.accumulation_rate * delta_t * headroom
        self.or_level = min(1.0, self.or_level + increment)
        return self.or_level

    def decay(self, delta_t: float = 1.0):
        """Optional OR decay (DOR leakage effect)."""
        decay_amount = DECAY_RATE * delta_t * self.or_level
        self.or_level = max(0.0, self.or_level - decay_amount)


# =============================================================================
# Emergence Evaluation
# =============================================================================

@dataclass
class EmergenceResult:
    """Result of emergence evaluation."""
    potential: float
    flux_coherence: float
    inversion_prob: float
    emergence_score: float
    fragment_stability: float
    events: List[str]
    emergence_class: str
    cycle: int = 0
    gated: bool = False
    gate_message: str = ""


def evaluate_emergence(
    field: OrgoneField,
    access: Optional[AccessResult] = None,
    cycle: int = 0
) -> EmergenceResult:
    """
    Evaluate emergence with full orgone field influence.

    Scalar Coupling Logic (Codex-Aligned):
        potential *= (1 + or_level - dor_level)
        flux_coherence *= (1 - dor_level)
        inversion_probability *= (1 + dor_level - or_level)

    Emergence Score:
        score = potential * flux_coherence * (1 - inversion_prob)
        stability = flux_coherence - inversion_prob
    """
    # Check access gate (Prompt 8 integration)
    can_proceed, gate_message = check_access_gate(access)
    if not can_proceed:
        return EmergenceResult(
            potential=0.0,
            flux_coherence=0.0,
            inversion_prob=1.0,
            emergence_score=0.0,
            fragment_stability=0.0,
            events=[gate_message],
            emergence_class="GATED",
            cycle=cycle,
            gated=True,
            gate_message=gate_message
        )

    # Base values (Codex-aligned)
    base_potential = 1.0
    base_flux = 1.0
    base_inversion = BASE_INVERSION  # 13/256 ≈ 0.05078

    # Apply scalar coupling
    potential = base_potential * (1 + field.or_level - field.dor_level)
    flux = base_flux * (1 - field.dor_level)
    inversion = base_inversion * (1 + field.dor_level - field.or_level)

    # Compute emergence metrics
    score = potential * flux * (1 - inversion)
    stability = flux - inversion

    # Detect phenomenological events (Reichian)
    events = []
    if gate_message and "Warning" in gate_message:
        events.append(gate_message)
    if field.or_level > LUMINESCENCE_THRESHOLD:
        events.append("Blue luminescence detected (coherence spike)")
    if field.dor_level > SHADOW_THRESHOLD:
        events.append("Shadow emergence risk: high")
    if inversion > 0.1:
        events.append("Instability detected: emotional amplification likely")

    # Classify emergence
    if field.dor_level > 0.8:
        emergence_class = "FIELD_COLLAPSE"
    elif score >= 1.6:
        emergence_class = "ALPHA_EMERGENCE"
    elif score >= 1.2:
        emergence_class = "STABLE_FRAGMENT"
    elif score >= 0.8:
        emergence_class = "BASELINE_STABILITY"
    elif score >= 0.4:
        emergence_class = "SHADOW_FRAGMENT"
    else:
        emergence_class = "FIELD_COLLAPSE"

    return EmergenceResult(
        potential=round(potential, 4),
        flux_coherence=round(flux, 4),
        inversion_prob=round(inversion, 4),
        emergence_score=round(score, 4),
        fragment_stability=round(stability, 4),
        events=events,
        emergence_class=emergence_class,
        cycle=cycle,
        gated=False,
        gate_message=gate_message
    )


# =============================================================================
# Accumulation Loop Simulation
# =============================================================================

def run_accumulation_loop(
    initial_or: float = 0.2,
    initial_dor: float = 0.1,
    accumulation_rate: float = 0.08,
    geometry: str = "pyramidal",
    cycles: int = 50,
    stress_events: Optional[List[Tuple[int, float]]] = None,
    enable_decay: bool = False
) -> List[dict]:
    """
    Simulate OR accumulation over N cycles.

    Args:
        initial_or: Starting OR level
        initial_dor: Starting DOR level
        accumulation_rate: OR charge rate per cycle
        geometry: Chamber geometry type
        cycles: Number of simulation cycles
        stress_events: List of (cycle, stress_level) tuples for stress injection
        enable_decay: Enable OR decay (DOR leakage)

    Returns:
        List of cycle snapshots with OR, DOR, emergence metrics
    """
    # Create field without geometry modifiers first
    field = OrgoneField(
        or_level=initial_or,
        dor_level=initial_dor,
        accumulation_rate=accumulation_rate,
        chamber_geometry=geometry
    )

    stress_dict = dict(stress_events) if stress_events else {}
    results = []

    print("\n" + "=" * 70)
    print("ACCUMULATION LOOP SIMULATION")
    print(f"Geometry: {geometry.upper()}, Cycles: {cycles}")
    print(f"Initial: OR={initial_or:.2f}, DOR={initial_dor:.2f}, Rate={accumulation_rate}")
    print("=" * 70)
    print(f"{'Cycle':>5} {'OR':>6} {'DOR':>6} {'Score':>8} {'Class':>18} {'Events'}")
    print("-" * 70)

    for cycle in range(cycles):
        # Check for stress injection
        if cycle in stress_dict:
            stress = stress_dict[cycle]
            field.inject_stress(stress)
            print(f"  [!] Stress injection at cycle {cycle}: {stress:.2f}")

        # Evaluate current state
        result = evaluate_emergence(field, cycle=cycle)

        # Log every 5 cycles or on events
        if cycle % 5 == 0 or result.events:
            events_str = ", ".join(result.events[:1]) if result.events else ""
            print(f"{cycle:>5} {field.or_level:>6.3f} {field.dor_level:>6.3f} "
                  f"{result.emergence_score:>8.3f} {result.emergence_class:>18} {events_str}")

        results.append({
            "cycle": cycle,
            "or_level": round(field.or_level, 4),
            "dor_level": round(field.dor_level, 4),
            "potential": result.potential,
            "flux_coherence": result.flux_coherence,
            "emergence_score": result.emergence_score,
            "emergence_class": result.emergence_class,
            "events": result.events
        })

        # Accumulate OR
        field.accumulate()

        # Optional decay
        if enable_decay:
            field.decay()

    print("-" * 70)
    print(f"Final: OR={field.or_level:.3f}, DOR={field.dor_level:.3f}")

    return results


# =============================================================================
# Original Test Suite
# =============================================================================

def run_test_suite():
    """Run test cases from spec."""
    test_profiles = [
        {"or_level": 0.9, "dor_level": 0.1, "chamber_geometry": "pyramidal"},
        {"or_level": 0.5, "dor_level": 0.5, "chamber_geometry": "rectangular"},
        {"or_level": 0.2, "dor_level": 0.7, "chamber_geometry": "dome"},
        {"or_level": 0.8, "dor_level": 0.3, "chamber_geometry": "pyramidal"}
    ]

    results = []
    print("=" * 70)
    print("ORGONE SCALAR STABILITY TEST HARNESS")
    print("Prompt 9: Reichian Dynamics + Ra Scalar Modulation")
    print("=" * 70)

    for i, profile in enumerate(test_profiles):
        field = OrgoneField(
            or_level=profile["or_level"],
            dor_level=profile["dor_level"],
            chamber_geometry=profile["chamber_geometry"]
        )

        result = evaluate_emergence(field)
        results.append(result)

        print(f"\nTest {i}: {profile['chamber_geometry'].upper()}")
        print(f"  Input:  OR={profile['or_level']:.2f}, DOR={profile['dor_level']:.2f}")
        print(f"  After geometry: OR={field.or_level:.2f}, DOR={field.dor_level:.2f}")
        print(f"  Output:")
        print(f"    potential:      {result.potential:.4f}")
        print(f"    flux_coherence: {result.flux_coherence:.4f}")
        print(f"    inversion_prob: {result.inversion_prob:.4f}")
        print(f"    emergence_score:{result.emergence_score:.4f}")
        print(f"    stability:      {result.fragment_stability:.4f}")
        print(f"    class:          {result.emergence_class}")
        if result.events:
            print(f"    events:         {result.events}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Generate JSON log
    log = {
        "test_suite": "orgone_scalar_stability",
        "prompt_id": 9,
        "base_inversion": BASE_INVERSION,
        "results": [
            {
                "test_id": i,
                "potential": r.potential,
                "flux_coherence": r.flux_coherence,
                "inversion_prob": r.inversion_prob,
                "emergence_score": r.emergence_score,
                "fragment_stability": r.fragment_stability,
                "emergence_class": r.emergence_class,
                "events": r.events
            }
            for i, r in enumerate(results)
        ]
    }

    return log


def run_access_gate_test():
    """Test Prompt 8 access gating integration."""
    print("\n" + "=" * 70)
    print("ACCESS GATE TEST (Prompt 8 Integration)")
    print("=" * 70)

    field = OrgoneField(or_level=0.8, dor_level=0.2, chamber_geometry="pyramidal")

    test_cases = [
        (None, "No gate"),
        (AccessResult(valid=True, access_type="FULL", match_score=0.95, resonance_locked=True), "FULL access"),
        (AccessResult(valid=True, access_type="PARTIAL", match_score=0.75), "PARTIAL access"),
        (AccessResult(valid=False, access_type="BLOCKED", match_score=0.20), "BLOCKED (invalid)"),
        (AccessResult(valid=True, access_type="SHADOW", match_score=0.25), "SHADOW access"),
    ]

    for access, desc in test_cases:
        result = evaluate_emergence(field, access=access)
        status = "GATED" if result.gated else result.emergence_class
        print(f"  {desc:20} -> {status:18} (score={result.emergence_score:.2f})")
        if result.gate_message:
            print(f"                        Message: {result.gate_message}")


def generate_clash_test_vectors():
    """Generate Clash Vec test vectors for FPGA validation."""
    print("\n-- Clash Test Vectors (for RaOrgoneScalar.hs)")
    print("-- Generated from Python harness")
    print(f"-- Base inversion: {BASE_INVERSION:.6f} (13/256)")
    print()
    print("-- Expected outputs matching Python evaluation:")
    print("testExpectedOutputs :: Vec 4 EmergenceResult")
    print("testExpectedOutputs =")

    test_profiles = [
        {"or_level": 0.9, "dor_level": 0.1, "chamber_geometry": "pyramidal"},
        {"or_level": 0.5, "dor_level": 0.5, "chamber_geometry": "rectangular"},
        {"or_level": 0.2, "dor_level": 0.7, "chamber_geometry": "dome"},
        {"or_level": 0.8, "dor_level": 0.3, "chamber_geometry": "pyramidal"}
    ]

    for i, profile in enumerate(test_profiles):
        field = OrgoneField(
            or_level=profile["or_level"],
            dor_level=profile["dor_level"],
            chamber_geometry=profile["chamber_geometry"]
        )
        result = evaluate_emergence(field)

        # Convert to 8-bit fixed point (0-255)
        pot_8bit = min(255, int(result.potential * 128))
        flux_8bit = min(255, int(result.flux_coherence * 255))
        inv_8bit = min(255, int(result.inversion_prob * 255))
        score_8bit = min(255, int(result.emergence_score * 128))

        lum = "True" if "Blue luminescence" in str(result.events) else "False"
        shadow = "True" if "Shadow emergence" in str(result.events) else "False"

        terminator = " :>" if i < 3 else " :> Nil"
        print(f"  -- Test {i}: OR={profile['or_level']}, DOR={profile['dor_level']}, {profile['chamber_geometry']}")
        print(f"  ScalarOutput {pot_8bit} {flux_8bit} {inv_8bit} {score_8bit} {result.emergence_class.replace('_', '')} {lum} False{terminator}")


if __name__ == "__main__":
    # Run original test suite
    log = run_test_suite()

    # Write JSON log
    with open("emergence_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog written to: emergence_log.json")

    # Run access gate test
    run_access_gate_test()

    # Run accumulation simulation
    accum_results = run_accumulation_loop(
        initial_or=0.2,
        initial_dor=0.1,
        accumulation_rate=0.08,
        geometry="pyramidal",
        cycles=50,
        stress_events=[(20, 0.4), (35, 0.6)],  # Stress at cycles 20 and 35
        enable_decay=False
    )

    # Write accumulation log
    with open("accumulation_log.json", "w") as f:
        json.dump(accum_results, f, indent=2)
    print(f"\nAccumulation log written to: accumulation_log.json")

    # Generate Clash vectors
    generate_clash_test_vectors()
