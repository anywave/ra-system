#!/usr/bin/env python3
"""
Orgone Scalar Stability Test Harness
=====================================

Python harness for testing RaOrgoneScalar Clash module.
Validates Reichian orgone dynamics with scalar field modulation.

Prompt 9: Orgone Field Influence on Ra Scalar Stability
"""

from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class OrgoneField:
    """Orgone field state with geometry modifiers."""
    or_level: float
    dor_level: float
    accumulation_rate: float = 0.04
    chamber_geometry: str = "rectangular"

    def __post_init__(self):
        self.apply_geometry_modifiers()

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


def evaluate_emergence(field: OrgoneField) -> EmergenceResult:
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
    # Base values
    base_potential = 1.0
    base_flux = 1.0
    base_inversion = 0.05

    # Apply scalar coupling
    potential = base_potential * (1 + field.or_level - field.dor_level)
    flux = base_flux * (1 - field.dor_level)
    inversion = base_inversion * (1 + field.dor_level - field.or_level)

    # Compute emergence metrics
    score = potential * flux * (1 - inversion)
    stability = flux - inversion

    # Detect phenomenological events (Reichian)
    events = []
    if field.or_level > 0.85:
        events.append("Blue luminescence detected (coherence spike)")
    if field.dor_level > 0.6:
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
        emergence_class=emergence_class
    )


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


def generate_clash_test_vectors():
    """Generate Clash Vec test vectors for FPGA validation."""
    print("\n-- Clash Test Vectors (for RaOrgoneScalar.hs)")
    print("-- Generated from Python harness")
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
    log = run_test_suite()

    # Write JSON log
    with open("emergence_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog written to: emergence_log.json")

    generate_clash_test_vectors()
