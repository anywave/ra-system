#!/usr/bin/env python3
"""
Ra System - Multi-Avatar Group Coherence Test Harness
Prompt 11: Multi-Avatar Scalar Entrainment & Group Coherence Field

This harness validates the RaGroupCoherence.hs Clash module by simulating
multiple avatars harmonizing their scalar coherence fields.

Codex References:
- GOLOD_RUSSIAN_PYRAMIDS.md: Group field amplification
- REICH_ORGONE_ACCUMULATOR.md: Orgone coherence scaling
- KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Harmonic transfer
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from enum import Enum
import random
import sys

# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

# Golod pyramid amplification factor (~15% boost for aligned groups)
GOLOD_AMPLIFICATION = 0.15

# Reich orgone threshold for stable field
REICH_THRESHOLD = 0.7

# Keely triadic resonance multiplier (for 3+ aligned)
KEELY_TRIADIC = 0.3

# Minimum avatars for group entrainment
MIN_GROUP_SIZE = 3

# Harmonic distance threshold for clustering
CLUSTER_THRESHOLD = 2

# Solfeggio frequencies per L value
SOLFEGGIO_BY_L = {
    0: 396,  # Root
    1: 417,  # Sacral
    2: 528,  # Solar
    3: 639,  # Heart
    4: 741,  # Throat
    5: 852,  # Third Eye
    6: 963,  # Crown
    7: 963,  # Crown (extended)
}

# ============================================================================
# Enums
# ============================================================================

class InversionState(Enum):
    NORMAL = "Normal"
    INVERTED = "Inverted"
    SHADOW = "Shadow"
    CLEARING = "Clearing"


class SymmetryStatus(Enum):
    STABLE = "Stable"
    UNSTABLE = "Unstable"
    TRANSITIONING = "Transitioning"
    COLLAPSED = "Collapsed"


class FeedbackAction(Enum):
    HOLD_BREATH = "HOLD_BREATH"
    RESUME_BREATH = "RESUME_BREATH"
    RECENTER_FIELD = "RECENTER_FIELD"
    SHADOW_WARNING = "SHADOW_WARNING"
    COHERENCE_ACHIEVED = "COHERENCE_ACHIEVED"
    STABILIZING = "STABILIZING"


class GlyphType(Enum):
    MANDALA = 0
    FLOWER = 1
    SPIRAL = 2


class AudioCueType(Enum):
    TONE = 0
    BINAURAL = 1
    PULSE = 2


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HarmonicSignature:
    """Spherical harmonic signature (l, m)"""
    l: int  # Degree (0-7)
    m: int  # Order (-l to +l)

    def distance(self, other: 'HarmonicSignature') -> int:
        """Calculate harmonic distance"""
        return abs(self.l - other.l) + abs(self.m - other.m)

    def __str__(self):
        return f"(l={self.l}, m={self.m})"


@dataclass
class TemporalWindow:
    """Temporal window for emergence"""
    start: float  # Start time (seconds)
    end: float    # End time (seconds)

    def overlaps(self, other: 'TemporalWindow') -> bool:
        return self.start < other.end and other.start < self.end

    def intersection(self, other: 'TemporalWindow') -> Optional['TemporalWindow']:
        if not self.overlaps(other):
            return None
        return TemporalWindow(max(self.start, other.start), min(self.end, other.end))


@dataclass
class AvatarInput:
    """Single avatar coherence input"""
    user_id: str
    coherence: float              # 0.0-1.0
    inversion: InversionState
    harmonic_signature: HarmonicSignature
    scalar_depth: float           # 0.0-1.0
    temporal_window: TemporalWindow
    active: bool = True

    def weight(self) -> float:
        """Calculate avatar weight for field contribution"""
        if self.inversion == InversionState.INVERTED:
            return -self.coherence * self.scalar_depth * 0.5
        return self.coherence * self.scalar_depth


@dataclass
class HarmonicCluster:
    """Cluster of harmonically aligned avatars"""
    signature: HarmonicSignature
    members: List[str]            # User IDs
    coherence: float              # Average coherence
    size: int

    def __str__(self):
        return f"Cluster {self.signature}: {self.size} avatars aligned"


@dataclass
class ClusterResult:
    """Result of avatar clustering"""
    clusters: List[HarmonicCluster]
    outliers: List[str]
    inverted: List[str]
    dominant_cluster: int


@dataclass
class GroupScalarField:
    """Collective scalar field"""
    dominant_mode: HarmonicSignature
    delta_ankh: float             # Symmetry correction
    symmetry_status: SymmetryStatus
    coherence_vector: float       # Group coherence
    inversion_flags: List[str]    # Inverted avatars
    field_strength: float


@dataclass
class EmergenceWindow:
    """Optimal emergence window result"""
    optimal_start: float
    optimal_end: float
    breath_initiate: float
    valid: bool
    participants: List[str]


@dataclass
class AudioCue:
    """Audio feedback cue"""
    frequency: float
    duration: float
    cue_type: AudioCueType


@dataclass
class VisualGlyph:
    """Visual feedback glyph"""
    glyph_type: GlyphType
    phase: float
    scale: float
    rotation: float


@dataclass
class EntrainmentFeedback:
    """Complete entrainment feedback"""
    action: FeedbackAction
    audio: AudioCue
    glyph: VisualGlyph
    message: str


@dataclass
class GroupCoherenceOutput:
    """Complete output from group coherence processing"""
    group_field: GroupScalarField
    clusters: ClusterResult
    emergence: EmergenceWindow
    feedback: EntrainmentFeedback
    cycle: int
    safety_alert: bool
    events: List[str] = field(default_factory=list)


# ============================================================================
# Core Functions: Clustering
# ============================================================================

def cluster_avatars(avatars: List[AvatarInput]) -> ClusterResult:
    """Cluster avatars by harmonic signature"""
    active = [a for a in avatars if a.active]
    inverted = [a.user_id for a in active if a.inversion == InversionState.INVERTED]

    # Group by L value (simplified clustering)
    clusters_by_l: Dict[int, List[AvatarInput]] = {}
    for av in active:
        l_val = av.harmonic_signature.l
        if l_val not in clusters_by_l:
            clusters_by_l[l_val] = []
        clusters_by_l[l_val].append(av)

    # Build clusters
    clusters = []
    for l_val, members in clusters_by_l.items():
        if len(members) >= 1:
            avg_coh = sum(a.coherence for a in members) / len(members)
            # Use most common m value
            m_counts = {}
            for a in members:
                m = a.harmonic_signature.m
                m_counts[m] = m_counts.get(m, 0) + 1
            dominant_m = max(m_counts, key=m_counts.get)

            clusters.append(HarmonicCluster(
                signature=HarmonicSignature(l_val, dominant_m),
                members=[a.user_id for a in members],
                coherence=avg_coh,
                size=len(members)
            ))

    # Sort by size then coherence
    clusters.sort(key=lambda c: (c.size, c.coherence), reverse=True)

    # Find outliers (avatars not fitting well in any cluster)
    outliers = []
    for av in active:
        if av.inversion == InversionState.INVERTED:
            continue
        in_cluster = False
        for c in clusters:
            if av.user_id in c.members and c.size >= 2:
                in_cluster = True
                break
        if not in_cluster and any(c.size >= 2 for c in clusters):
            outliers.append(av.user_id)

    dominant_idx = 0 if clusters else -1

    return ClusterResult(
        clusters=clusters,
        outliers=outliers,
        inverted=inverted,
        dominant_cluster=dominant_idx
    )


# ============================================================================
# Core Functions: Emergence Window
# ============================================================================

def find_emergence_window(avatars: List[AvatarInput]) -> EmergenceWindow:
    """Find optimal emergence window from avatar temporal windows"""
    active = [a for a in avatars if a.active]

    if len(active) < MIN_GROUP_SIZE:
        return EmergenceWindow(0, 0, 0, False, [])

    # Find intersection of all windows
    windows = [a.temporal_window for a in active]
    intersection = windows[0]

    for w in windows[1:]:
        intersection = intersection.intersection(w)
        if intersection is None:
            # No common window, use highest coherence avatar's window
            best = max(active, key=lambda a: a.coherence)
            return EmergenceWindow(
                optimal_start=best.temporal_window.start,
                optimal_end=best.temporal_window.end,
                breath_initiate=best.temporal_window.start + 0.5,
                valid=True,
                participants=[a.user_id for a in active]
            )

    # Group breath starts 500ms after window start
    breath_start = intersection.start + 0.5

    return EmergenceWindow(
        optimal_start=intersection.start,
        optimal_end=intersection.end,
        breath_initiate=breath_start,
        valid=True,
        participants=[a.user_id for a in active]
    )


# ============================================================================
# Core Functions: Group Field Construction
# ============================================================================

def construct_group_field(
    avatars: List[AvatarInput],
    clusters: ClusterResult
) -> GroupScalarField:
    """Construct group scalar field from avatars"""
    active = [a for a in avatars if a.active]

    if not active:
        return GroupScalarField(
            dominant_mode=HarmonicSignature(0, 0),
            delta_ankh=0,
            symmetry_status=SymmetryStatus.COLLAPSED,
            coherence_vector=0,
            inversion_flags=[],
            field_strength=0
        )

    # Get dominant signature
    if clusters.clusters:
        dom_sig = clusters.clusters[clusters.dominant_cluster].signature
        dom_cluster = clusters.clusters[clusters.dominant_cluster]
    else:
        dom_sig = HarmonicSignature(0, 0)
        dom_cluster = None

    # Calculate weighted coherence
    total_weight = 0
    for av in active:
        if av.inversion != InversionState.INVERTED:
            total_weight += av.weight()

    base_coh = total_weight / len(active) if active else 0

    # Apply Golod amplification for 3+ aligned avatars
    golod_boost = 0
    if dom_cluster and dom_cluster.size >= 3:
        golod_boost = GOLOD_AMPLIFICATION
        # print(f"  [Golod] {dom_cluster.size} aligned -> +{golod_boost:.1%} boost")

    group_coh = min(1.0, base_coh + golod_boost)

    # Apply Keely triadic boost
    keely_boost = 0
    if dom_cluster and dom_cluster.size >= 3:
        keely_boost = KEELY_TRIADIC * (dom_cluster.size / 8)
        # print(f"  [Keely] Triadic resonance -> +{keely_boost:.2f} strength")

    field_strength = min(1.0, group_coh + keely_boost)

    # Calculate Delta(ankh)
    inv_count = len(clusters.inverted)
    delta_ankh = (inv_count * 0.2) - 0.4  # Negative if many inverted

    # Determine symmetry status
    if inv_count == 0 and group_coh > REICH_THRESHOLD:
        sym_status = SymmetryStatus.STABLE
    elif inv_count > 2:
        sym_status = SymmetryStatus.COLLAPSED
    elif inv_count > 0:
        sym_status = SymmetryStatus.UNSTABLE
    else:
        sym_status = SymmetryStatus.TRANSITIONING

    return GroupScalarField(
        dominant_mode=dom_sig,
        delta_ankh=delta_ankh,
        symmetry_status=sym_status,
        coherence_vector=group_coh,
        inversion_flags=clusters.inverted,
        field_strength=field_strength
    )


# ============================================================================
# Core Functions: Feedback Generation
# ============================================================================

def generate_feedback(
    prev_field: Optional[GroupScalarField],
    curr_field: GroupScalarField
) -> EntrainmentFeedback:
    """Generate entrainment feedback based on field state"""

    # Calculate coherence delta
    if prev_field:
        coh_delta = curr_field.coherence_vector - prev_field.coherence_vector
        inv_increased = len(curr_field.inversion_flags) > len(prev_field.inversion_flags)
    else:
        coh_delta = 0
        inv_increased = False

    # Determine action
    if inv_increased:
        action = FeedbackAction.SHADOW_WARNING
        message = f"Shadow harmonic detected: {curr_field.inversion_flags}"
    elif curr_field.symmetry_status == SymmetryStatus.COLLAPSED:
        action = FeedbackAction.RECENTER_FIELD
        message = "Re-centering field... standby..."
    elif coh_delta > 0.05:
        action = FeedbackAction.HOLD_BREATH
        message = "Group coherence rising... hold breath..."
    elif curr_field.coherence_vector > REICH_THRESHOLD:
        action = FeedbackAction.COHERENCE_ACHIEVED
        message = "Peak coherence achieved!"
    elif coh_delta < -0.05:
        action = FeedbackAction.RESUME_BREATH
        message = "Resume synchronized breath in 3, 2, 1..."
    else:
        action = FeedbackAction.STABILIZING
        message = "Stabilizing group field..."

    # Generate audio cue
    l_val = curr_field.dominant_mode.l
    freq = SOLFEGGIO_BY_L.get(l_val, 528)

    cue_type = {
        FeedbackAction.HOLD_BREATH: AudioCueType.BINAURAL,
        FeedbackAction.RECENTER_FIELD: AudioCueType.PULSE,
        FeedbackAction.SHADOW_WARNING: AudioCueType.PULSE,
        FeedbackAction.COHERENCE_ACHIEVED: AudioCueType.BINAURAL,
    }.get(action, AudioCueType.TONE)

    audio = AudioCue(frequency=freq, duration=1.0, cue_type=cue_type)

    # Generate visual glyph
    glyph_type = {
        FeedbackAction.HOLD_BREATH: GlyphType.MANDALA,
        FeedbackAction.RECENTER_FIELD: GlyphType.SPIRAL,
        FeedbackAction.SHADOW_WARNING: GlyphType.SPIRAL,
        FeedbackAction.COHERENCE_ACHIEVED: GlyphType.FLOWER,
    }.get(action, GlyphType.MANDALA)

    glyph = VisualGlyph(
        glyph_type=glyph_type,
        phase=curr_field.coherence_vector,
        scale=curr_field.field_strength,
        rotation=0
    )

    return EntrainmentFeedback(
        action=action,
        audio=audio,
        glyph=glyph,
        message=message
    )


# ============================================================================
# Main Processing Loop
# ============================================================================

def process_group_coherence(
    avatars: List[AvatarInput],
    prev_field: Optional[GroupScalarField] = None,
    cycle: int = 0
) -> GroupCoherenceOutput:
    """Process group coherence for one cycle"""
    events = []

    # Step 1: Cluster avatars
    clusters = cluster_avatars(avatars)

    for c in clusters.clusters:
        if c.size >= 2:
            events.append(f"Cluster {c.signature}: {c.size} avatars aligned")

    for inv in clusters.inverted:
        events.append(f"User {inv} is inverted - contributes counterpoint phase")

    for out in clusters.outliers:
        events.append(f"User {out} is an outlier")

    # Step 2: Find emergence window
    emergence = find_emergence_window(avatars)

    if emergence.valid:
        events.append(f"Optimal emergence window: T+{emergence.optimal_start:.2f}s to T+{emergence.optimal_end:.2f}s")
        events.append(f"Begin group breath at T+{emergence.breath_initiate:.2f}s")

    # Step 3: Construct group field
    field = construct_group_field(avatars, clusters)

    # Step 4: Generate feedback
    feedback = generate_feedback(prev_field, field)
    events.append(f"-> {feedback.message}")

    # Step 5: Safety check
    safety = (
        field.symmetry_status == SymmetryStatus.COLLAPSED
        or len(field.inversion_flags) > 4
    )

    if safety:
        events.append("!!! SAFETY ALERT: Field collapse or excessive inversions !!!")

    return GroupCoherenceOutput(
        group_field=field,
        clusters=clusters,
        emergence=emergence,
        feedback=feedback,
        cycle=cycle,
        safety_alert=safety,
        events=events
    )


# ============================================================================
# CLI Dashboard
# ============================================================================

def render_dashboard(output: GroupCoherenceOutput):
    """Render CLI diagnostic dashboard"""
    field = output.group_field
    clusters = output.clusters
    emergence = output.emergence
    feedback = output.feedback

    print(f"\n{'='*70}")
    print(f" MULTI-AVATAR GROUP COHERENCE - Cycle {output.cycle}")
    print(f"{'='*70}")

    # Cluster info
    print(f"\n Harmonic Clusters:")
    for i, c in enumerate(clusters.clusters):
        marker = ">>>" if i == clusters.dominant_cluster else "   "
        print(f"   {marker} Cluster {i}: {c.size} avatars at {c.signature}, coh={c.coherence:.3f}")
        print(f"       Members: {', '.join(c.members)}")

    if clusters.inverted:
        print(f"\n Inverted Avatars: {', '.join(clusters.inverted)}")
    if clusters.outliers:
        print(f" Outliers: {', '.join(clusters.outliers)}")

    # Emergence window
    print(f"\n Emergence Window:")
    print(f"   Valid: {'YES' if emergence.valid else 'NO'}")
    if emergence.valid:
        print(f"   Window: T+{emergence.optimal_start:.2f}s to T+{emergence.optimal_end:.2f}s")
        print(f"   Breath Init: T+{emergence.breath_initiate:.2f}s")

    # Group field
    print(f"\n Group Scalar Field:")
    print(f"   Dominant Mode: {field.dominant_mode}")
    print(f"   Delta(ankh): {field.delta_ankh:+.3f}")
    print(f"   Symmetry: {field.symmetry_status.value}")

    # Coherence bar
    coh_bar = int(field.coherence_vector * 30)
    print(f"   Coherence: [{'#' * coh_bar}{'-' * (30 - coh_bar)}] {field.coherence_vector:.3f}")

    str_bar = int(field.field_strength * 30)
    print(f"   Strength:  [{'#' * str_bar}{'-' * (30 - str_bar)}] {field.field_strength:.3f}")

    # Feedback
    print(f"\n Entrainment Feedback:")
    print(f"   Action: {feedback.action.value}")
    print(f"   Audio: {feedback.audio.frequency} Hz ({feedback.audio.cue_type.name})")
    print(f"   Glyph: {feedback.glyph.glyph_type.name}")
    print(f"   >>> {feedback.message}")

    if output.safety_alert:
        print(f"\n {'!'*30}")
        print(f" !!! SAFETY ALERT TRIGGERED !!!")
        print(f" {'!'*30}")

    print(f"{'='*70}")


# ============================================================================
# Test Scenarios
# ============================================================================

def create_test_scenario(name: str) -> List[AvatarInput]:
    """Create test avatar configurations"""

    if name == "aligned_triad":
        # Three avatars perfectly aligned at (3, 0)
        return [
            AvatarInput("A1X2", 0.57, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.63, TemporalWindow(4.5, 8.0)),
            AvatarInput("B3Y4", 0.72, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.71, TemporalWindow(4.0, 7.5)),
            AvatarInput("C5Z6", 0.65, InversionState.NORMAL,
                       HarmonicSignature(3, 1), 0.68, TemporalWindow(5.0, 9.0)),
        ]

    elif name == "with_inverted":
        # Group with one inverted avatar
        return [
            AvatarInput("A1X2", 0.57, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.63, TemporalWindow(4.5, 8.0)),
            AvatarInput("B3Y4", 0.72, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.71, TemporalWindow(4.0, 7.5)),
            AvatarInput("C5Z6", 0.65, InversionState.NORMAL,
                       HarmonicSignature(3, 1), 0.68, TemporalWindow(5.0, 9.0)),
            AvatarInput("A7X5", 0.40, InversionState.INVERTED,
                       HarmonicSignature(2, -1), 0.30, TemporalWindow(3.0, 6.0)),
        ]

    elif name == "mixed_clusters":
        # Multiple clusters with different harmonics
        return [
            AvatarInput("A1", 0.60, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.65, TemporalWindow(4.5, 8.0)),
            AvatarInput("A2", 0.75, InversionState.NORMAL,
                       HarmonicSignature(3, 1), 0.70, TemporalWindow(4.0, 7.5)),
            AvatarInput("A3", 0.55, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.60, TemporalWindow(5.0, 9.0)),
            AvatarInput("B1", 0.50, InversionState.NORMAL,
                       HarmonicSignature(2, 0), 0.55, TemporalWindow(4.0, 7.0)),
            AvatarInput("B2", 0.48, InversionState.NORMAL,
                       HarmonicSignature(2, 1), 0.50, TemporalWindow(4.5, 8.5)),
            AvatarInput("C1", 0.80, InversionState.NORMAL,
                       HarmonicSignature(5, 0), 0.85, TemporalWindow(3.5, 6.5)),
        ]

    elif name == "high_coherence":
        # High coherence group for peak state
        return [
            AvatarInput("A1", 0.85, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.90, TemporalWindow(4.0, 8.0)),
            AvatarInput("A2", 0.88, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.92, TemporalWindow(4.0, 8.0)),
            AvatarInput("A3", 0.82, InversionState.NORMAL,
                       HarmonicSignature(3, 1), 0.88, TemporalWindow(4.0, 8.0)),
            AvatarInput("A4", 0.79, InversionState.NORMAL,
                       HarmonicSignature(3, 0), 0.85, TemporalWindow(4.0, 8.0)),
        ]

    elif name == "collapse":
        # Scenario leading to field collapse
        return [
            AvatarInput("A1", 0.30, InversionState.NORMAL,
                       HarmonicSignature(1, 0), 0.35, TemporalWindow(4.0, 8.0)),
            AvatarInput("A2", 0.25, InversionState.INVERTED,
                       HarmonicSignature(2, -1), 0.20, TemporalWindow(4.0, 8.0)),
            AvatarInput("A3", 0.28, InversionState.INVERTED,
                       HarmonicSignature(3, 2), 0.25, TemporalWindow(4.0, 8.0)),
            AvatarInput("A4", 0.22, InversionState.INVERTED,
                       HarmonicSignature(4, -2), 0.18, TemporalWindow(4.0, 8.0)),
        ]

    else:  # default
        return [
            AvatarInput("U1", 0.50, InversionState.NORMAL,
                       HarmonicSignature(2, 0), 0.50, TemporalWindow(4.0, 8.0)),
            AvatarInput("U2", 0.55, InversionState.NORMAL,
                       HarmonicSignature(2, 0), 0.55, TemporalWindow(4.0, 8.0)),
            AvatarInput("U3", 0.52, InversionState.NORMAL,
                       HarmonicSignature(2, 1), 0.53, TemporalWindow(4.0, 8.0)),
        ]


def run_multi_cycle_simulation(
    avatars: List[AvatarInput],
    cycles: int = 5,
    coherence_drift: float = 0.02
) -> List[GroupCoherenceOutput]:
    """Run multi-cycle simulation with coherence drift"""
    results = []
    prev_field = None

    for cycle in range(cycles):
        # Apply coherence drift (simulate natural fluctuation)
        for av in avatars:
            drift = random.uniform(-coherence_drift, coherence_drift)
            av.coherence = max(0, min(1, av.coherence + drift))

        output = process_group_coherence(avatars, prev_field, cycle)
        results.append(output)
        prev_field = output.group_field

    return results


# ============================================================================
# Test Suite
# ============================================================================

def run_test_suite():
    """Run comprehensive test suite"""
    print("\n" + "="*70)
    print(" Ra System - Multi-Avatar Group Coherence Test Suite")
    print(" Prompt 11 Validation")
    print("="*70)

    scenarios = ["aligned_triad", "with_inverted", "mixed_clusters", "high_coherence", "collapse"]
    all_results = {}

    for scenario in scenarios:
        print(f"\n>>> Testing scenario: {scenario.upper()}")
        avatars = create_test_scenario(scenario)

        print(f"\nInput Avatars:")
        for av in avatars:
            print(f"  {av.user_id}: coh={av.coherence:.2f}, inv={av.inversion.value}, "
                  f"sig={av.harmonic_signature}, depth={av.scalar_depth:.2f}")

        output = process_group_coherence(avatars, cycle=0)
        render_dashboard(output)

        # Store results
        all_results[scenario] = {
            "avatars": len(avatars),
            "clusters": len(output.clusters.clusters),
            "dominant_mode": str(output.group_field.dominant_mode),
            "coherence_vector": output.group_field.coherence_vector,
            "field_strength": output.group_field.field_strength,
            "symmetry": output.group_field.symmetry_status.value,
            "inverted_count": len(output.clusters.inverted),
            "feedback_action": output.feedback.action.value,
            "safety_alert": output.safety_alert,
            "events": output.events
        }

    # Write results to log
    log_file = "group_coherence_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "test_suite": "group_coherence",
            "prompt_id": 11,
            "scenarios": all_results
        }, f, indent=2)

    print(f"\n>>> Results logged to {log_file}")
    print("\n" + "="*70)
    print(" All tests completed!")
    print("="*70 + "\n")

    return all_results


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sim":
        # Multi-cycle simulation
        print("\n>>> Running 5-cycle simulation with coherence drift")
        avatars = create_test_scenario("with_inverted")
        results = run_multi_cycle_simulation(avatars, cycles=5)
        for r in results:
            render_dashboard(r)
    else:
        # Full test suite
        run_test_suite()
