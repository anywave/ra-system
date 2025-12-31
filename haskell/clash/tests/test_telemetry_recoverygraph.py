#!/usr/bin/env python3
"""
Test harness for Prompt 48: Ra.Telemetry.RecoveryGraph

Temporal topological map of field coherence disruption and healing:
- Emergence transitions and inversion resolutions
- Harmonic axis realignment
- φ^n loop closure detection

Clarifications:
- EmergenceResult: INCOHERENT, REINTEGRATING, HARMONICALLY_STABLE, COLLAPSED, PHASE_LOCKED, GHOST_EMERGENT
- Loop closure: coherence within ±0.05 of baseline AND full 2π phase cycle
- HarmonicAxis: AXIS_THETA, AXIS_PHI, AXIS_THETA_PHI, AXIS_H, AXIS_LM
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto
from datetime import datetime, timedelta

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Recovery thresholds
COHERENCE_DELTA_THRESHOLD = 0.05   # Minimum delta to log event
LOOP_CLOSURE_TOLERANCE = 0.05     # Coherence must return within ±0.05
PHASE_CYCLE_2PI = 2 * math.pi     # Full phase cycle

# =============================================================================
# Enumerations
# =============================================================================

class EmergenceResult(Enum):
    """States of chamber/fragment during scalar emergence."""
    INCOHERENT = auto()          # Unstable or fragmented field
    REINTEGRATING = auto()       # Moving toward harmonic balance
    HARMONICALLY_STABLE = auto() # Coherent and aligned
    COLLAPSED = auto()           # Field disintegrated or retreated
    PHASE_LOCKED = auto()        # Temporarily coherent, externally entrained
    GHOST_EMERGENT = auto()      # Residual/pseudo emergence (echo-state)


class HarmonicAxis(Enum):
    """Axes for tracking recovery trajectory."""
    AXIS_THETA = auto()      # Polar angular component
    AXIS_PHI = auto()        # Azimuthal rotation
    AXIS_THETA_PHI = auto()  # Combined orbital plane
    AXIS_H = auto()          # Scalar field harmonic depth
    AXIS_LM = auto()         # Spherical harmonic (l,m) space


class InversionPolarity(Enum):
    """Inversion state."""
    NORMAL = auto()
    INVERTED = auto()


# =============================================================================
# Data Types
# =============================================================================

@dataclass(frozen=True)
class RaCoordinate:
    """Ra hypergrid coordinate."""
    theta: float   # 0-2π
    phi: float     # 0-π
    h: float       # Harmonic depth


@dataclass
class InversionShift:
    """Inversion shift event."""
    axis_flipped: bool
    torsion_intensity: float
    polarity: InversionPolarity = InversionPolarity.INVERTED


@dataclass
class ScalarFieldSnapshot:
    """Snapshot of scalar field state."""
    timestamp: datetime
    coherence: float
    flux: float
    phase_angle: float
    dominant_l: int = 0
    dominant_m: int = 0


@dataclass
class RecoveryEvent:
    """
    A recovery event in the telemetry trail.

    Attributes:
        timestamp: When the event occurred
        location: Ra coordinate of the event
        coherence_value: Coherence at this point
        flux_value: Flux at this point
        state_before: Emergence state before transition
        state_after: Emergence state after transition
        inversion_shift: Optional inversion shift data
    """
    timestamp: datetime
    location: RaCoordinate
    coherence_value: float
    flux_value: float
    state_before: EmergenceResult
    state_after: EmergenceResult
    inversion_shift: Optional[InversionShift] = None


@dataclass
class RecoveryGraph:
    """
    Complete recovery graph tracking emergence healing.

    Attributes:
        event_trail: List of recovery events
        loop_closed: Whether harmonic loop completed
        dominant_axis: Primary axis of recovery
        coherence_delta: Total coherence change
        baseline_coherence: Initial coherence before disruption
        final_phase: Final phase angle
    """
    event_trail: List[RecoveryEvent]
    loop_closed: bool
    dominant_axis: HarmonicAxis
    coherence_delta: float
    baseline_coherence: float = 0.0
    final_phase: float = 0.0


# =============================================================================
# Core Functions
# =============================================================================

def determine_emergence_state(coherence: float, previous_state: Optional[EmergenceResult] = None) -> EmergenceResult:
    """Determine emergence state from coherence value."""
    if coherence < 0.20:
        return EmergenceResult.COLLAPSED
    elif coherence < 0.40:
        return EmergenceResult.INCOHERENT
    elif coherence < 0.72:
        if previous_state == EmergenceResult.INCOHERENT:
            return EmergenceResult.REINTEGRATING
        return EmergenceResult.REINTEGRATING
    elif coherence < 0.90:
        return EmergenceResult.PHASE_LOCKED
    else:
        return EmergenceResult.HARMONICALLY_STABLE


def detect_inversion_shift(
    prev_snapshot: ScalarFieldSnapshot,
    curr_snapshot: ScalarFieldSnapshot
) -> Optional[InversionShift]:
    """Detect if an inversion shift occurred between snapshots."""
    # Phase flip detection (crossing π boundary)
    phase_diff = abs(curr_snapshot.phase_angle - prev_snapshot.phase_angle)
    phase_flip = phase_diff > math.pi * 0.8

    # Coherence drop with phase change indicates inversion
    coherence_drop = prev_snapshot.coherence - curr_snapshot.coherence > 0.15

    if phase_flip or (coherence_drop and phase_diff > 0.5):
        torsion = abs(phase_diff / math.pi) * (1 - curr_snapshot.coherence)
        return InversionShift(
            axis_flipped=phase_flip,
            torsion_intensity=min(1.0, torsion),
            polarity=InversionPolarity.INVERTED
        )
    return None


def compute_dominant_axis(events: List[RecoveryEvent]) -> HarmonicAxis:
    """Determine dominant axis from recovery events."""
    if len(events) < 2:
        return HarmonicAxis.AXIS_THETA

    # Compute movement along each axis
    theta_movement = 0.0
    phi_movement = 0.0
    h_movement = 0.0

    for i in range(1, len(events)):
        prev = events[i-1].location
        curr = events[i].location
        theta_movement += abs(curr.theta - prev.theta)
        phi_movement += abs(curr.phi - prev.phi)
        h_movement += abs(curr.h - prev.h)

    # Determine dominant
    max_movement = max(theta_movement, phi_movement, h_movement)

    if max_movement == 0:
        return HarmonicAxis.AXIS_THETA

    if theta_movement == max_movement and phi_movement > max_movement * 0.5:
        return HarmonicAxis.AXIS_THETA_PHI
    elif theta_movement == max_movement:
        return HarmonicAxis.AXIS_THETA
    elif phi_movement == max_movement:
        return HarmonicAxis.AXIS_PHI
    else:
        return HarmonicAxis.AXIS_H


def check_loop_closure(
    baseline_coherence: float,
    final_coherence: float,
    initial_phase: float,
    final_phase: float
) -> bool:
    """
    Check if recovery loop is closed.

    Requires:
    1. Coherence within ±0.05 of baseline
    2. Full 2π phase cycle completed
    """
    coherence_restored = abs(final_coherence - baseline_coherence) <= LOOP_CLOSURE_TOLERANCE

    # Check if phase completed full cycle (allowing for wraparound)
    phase_delta = abs(final_phase - initial_phase)
    phase_cycle_complete = (
        phase_delta >= PHASE_CYCLE_2PI * 0.95 or  # Nearly full cycle
        phase_delta <= 0.1  # Returned to start (with wraparound)
    )

    return coherence_restored and phase_cycle_complete


def build_recovery_graph(
    emergence_results: List[EmergenceResult],
    field_snapshots: List[ScalarFieldSnapshot],
    inversion_shifts: Optional[List[InversionShift]] = None
) -> RecoveryGraph:
    """
    Build a recovery graph from emergence results and field snapshots.

    Parameters:
        emergence_results: List of emergence states
        field_snapshots: List of scalar field snapshots
        inversion_shifts: Optional pre-detected inversions

    Returns:
        RecoveryGraph with event trail and analysis
    """
    if not field_snapshots:
        return RecoveryGraph(
            event_trail=[],
            loop_closed=False,
            dominant_axis=HarmonicAxis.AXIS_THETA,
            coherence_delta=0.0
        )

    events = []
    baseline_coherence = field_snapshots[0].coherence
    initial_phase = field_snapshots[0].phase_angle

    prev_state = emergence_results[0] if emergence_results else EmergenceResult.INCOHERENT
    prev_coherence = baseline_coherence

    for i, snapshot in enumerate(field_snapshots):
        # Determine current state
        if i < len(emergence_results):
            curr_state = emergence_results[i]
        else:
            curr_state = determine_emergence_state(snapshot.coherence, prev_state)

        # Check if we should log this event
        coherence_delta = abs(snapshot.coherence - prev_coherence)
        state_changed = curr_state != prev_state

        if coherence_delta > COHERENCE_DELTA_THRESHOLD or state_changed:
            # Detect inversion
            inversion = None
            if i > 0:
                inversion = detect_inversion_shift(field_snapshots[i-1], snapshot)

            # Override with provided inversions if available
            if inversion_shifts and i < len(inversion_shifts):
                inversion = inversion_shifts[i]

            event = RecoveryEvent(
                timestamp=snapshot.timestamp,
                location=RaCoordinate(
                    theta=snapshot.phase_angle % (2 * math.pi),
                    phi=snapshot.phase_angle / 2 % math.pi,
                    h=float(snapshot.dominant_l)
                ),
                coherence_value=snapshot.coherence,
                flux_value=snapshot.flux,
                state_before=prev_state,
                state_after=curr_state,
                inversion_shift=inversion
            )
            events.append(event)
            prev_state = curr_state
            prev_coherence = snapshot.coherence

    # Compute final metrics
    final_coherence = field_snapshots[-1].coherence if field_snapshots else 0.0
    final_phase = field_snapshots[-1].phase_angle if field_snapshots else 0.0
    total_delta = final_coherence - baseline_coherence

    loop_closed = check_loop_closure(
        baseline_coherence, final_coherence,
        initial_phase, final_phase
    )

    dominant_axis = compute_dominant_axis(events)

    return RecoveryGraph(
        event_trail=events,
        loop_closed=loop_closed,
        dominant_axis=dominant_axis,
        coherence_delta=total_delta,
        baseline_coherence=baseline_coherence,
        final_phase=final_phase
    )


def find_coherence_minima(snapshots: List[ScalarFieldSnapshot]) -> List[int]:
    """Find indices of coherence local minima."""
    minima = []
    for i in range(1, len(snapshots) - 1):
        if (snapshots[i].coherence < snapshots[i-1].coherence and
            snapshots[i].coherence < snapshots[i+1].coherence):
            minima.append(i)
    return minima


def track_upward_recovery(
    snapshots: List[ScalarFieldSnapshot],
    min_idx: int
) -> List[ScalarFieldSnapshot]:
    """Track upward recovery from a coherence minimum."""
    recovery = [snapshots[min_idx]]
    for i in range(min_idx + 1, len(snapshots)):
        if snapshots[i].coherence >= snapshots[i-1].coherence:
            recovery.append(snapshots[i])
        else:
            break
    return recovery


# =============================================================================
# Test Cases
# =============================================================================

class TestEmergenceStates:
    """Test emergence state classification."""

    def test_collapsed_state(self):
        """Low coherence yields COLLAPSED."""
        assert determine_emergence_state(0.10) == EmergenceResult.COLLAPSED
        assert determine_emergence_state(0.19) == EmergenceResult.COLLAPSED

    def test_incoherent_state(self):
        """Low-mid coherence yields INCOHERENT."""
        assert determine_emergence_state(0.25) == EmergenceResult.INCOHERENT
        assert determine_emergence_state(0.39) == EmergenceResult.INCOHERENT

    def test_reintegrating_state(self):
        """Mid coherence yields REINTEGRATING."""
        assert determine_emergence_state(0.50) == EmergenceResult.REINTEGRATING
        assert determine_emergence_state(0.70) == EmergenceResult.REINTEGRATING

    def test_phase_locked_state(self):
        """High coherence yields PHASE_LOCKED."""
        assert determine_emergence_state(0.75) == EmergenceResult.PHASE_LOCKED
        assert determine_emergence_state(0.89) == EmergenceResult.PHASE_LOCKED

    def test_harmonically_stable_state(self):
        """Very high coherence yields HARMONICALLY_STABLE."""
        assert determine_emergence_state(0.92) == EmergenceResult.HARMONICALLY_STABLE
        assert determine_emergence_state(1.0) == EmergenceResult.HARMONICALLY_STABLE


class TestCoherenceDropAndRecovery:
    """Test coherence drop tracking and recovery."""

    def test_coherence_dip_tracked(self):
        """Synthetic dip is tracked with correct delta."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.85, 1.0, 0.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=1), 0.60, 1.1, 0.5),
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.42, 1.2, 1.0),  # Dip
            ScalarFieldSnapshot(base_time + timedelta(seconds=3), 0.55, 1.1, 1.5),
            ScalarFieldSnapshot(base_time + timedelta(seconds=4), 0.75, 1.0, 2.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=5), 0.88, 0.9, 2.5),
        ]

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        # Should have events logged
        assert len(graph.event_trail) > 0
        # Delta should be positive (recovery)
        assert graph.coherence_delta > 0 or abs(graph.coherence_delta) < 0.1

    def test_baseline_restoration(self):
        """Check that baseline coherence can be restored."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.80, 1.0, 0.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=1), 0.40, 1.2, 1.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.60, 1.1, 3.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=3), 0.78, 1.0, 5.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=4), 0.82, 0.9, 6.28),  # ~2π
        ]

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        # Coherence returned close to baseline
        final_coh = snapshots[-1].coherence
        assert abs(final_coh - graph.baseline_coherence) < 0.1


class TestEmergenceTransitionTrail:
    """Test emergence transition alignment."""

    def test_state_transitions_temporal(self):
        """State before/after pairs align temporally."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.30, 1.0, 0.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.50, 1.0, 1.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=4), 0.75, 1.0, 2.0),
        ]

        states = [
            EmergenceResult.INCOHERENT,
            EmergenceResult.REINTEGRATING,
            EmergenceResult.PHASE_LOCKED
        ]

        graph = build_recovery_graph(states, snapshots)

        # Check transitions are sequential
        for i, event in enumerate(graph.event_trail):
            if i > 0:
                prev_event = graph.event_trail[i-1]
                assert event.timestamp > prev_event.timestamp

    def test_state_before_after_consistency(self):
        """State before matches previous state after."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.25, 1.0, 0.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=1), 0.45, 1.0, 1.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.65, 1.0, 2.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=3), 0.85, 1.0, 3.0),
        ]

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        # State continuity: each event's state_before should match prev event's state_after
        for i in range(1, len(graph.event_trail)):
            assert graph.event_trail[i].state_before == graph.event_trail[i-1].state_after


class TestDominantAxisDetection:
    """Test harmonic axis detection."""

    def test_theta_dominant(self):
        """Detect theta-dominant recovery."""
        events = [
            RecoveryEvent(datetime.now(), RaCoordinate(0.0, 0.5, 1.0), 0.5, 1.0,
                         EmergenceResult.INCOHERENT, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(1.0, 0.5, 1.0), 0.6, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(2.0, 0.6, 1.0), 0.7, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.PHASE_LOCKED),
        ]

        axis = compute_dominant_axis(events)
        assert axis == HarmonicAxis.AXIS_THETA

    def test_phi_dominant(self):
        """Detect phi-dominant recovery."""
        events = [
            RecoveryEvent(datetime.now(), RaCoordinate(0.5, 0.0, 1.0), 0.5, 1.0,
                         EmergenceResult.INCOHERENT, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(0.5, 1.0, 1.0), 0.6, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(0.6, 2.0, 1.0), 0.7, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.PHASE_LOCKED),
        ]

        axis = compute_dominant_axis(events)
        assert axis == HarmonicAxis.AXIS_PHI

    def test_combined_axis(self):
        """Detect combined theta-phi recovery."""
        events = [
            RecoveryEvent(datetime.now(), RaCoordinate(0.0, 0.0, 1.0), 0.5, 1.0,
                         EmergenceResult.INCOHERENT, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(1.0, 0.8, 1.0), 0.6, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.REINTEGRATING),
            RecoveryEvent(datetime.now(), RaCoordinate(2.0, 1.6, 1.0), 0.7, 1.0,
                         EmergenceResult.REINTEGRATING, EmergenceResult.PHASE_LOCKED),
        ]

        axis = compute_dominant_axis(events)
        assert axis == HarmonicAxis.AXIS_THETA_PHI


class TestLoopClosure:
    """Test φ^n harmonic loop closure detection."""

    def test_loop_closed_full_recovery(self):
        """Loop closes when coherence and phase both complete."""
        closed = check_loop_closure(
            baseline_coherence=0.85,
            final_coherence=0.87,
            initial_phase=0.0,
            final_phase=6.28  # ~2π
        )
        assert closed is True

    def test_loop_not_closed_coherence_low(self):
        """Loop not closed if coherence not restored."""
        closed = check_loop_closure(
            baseline_coherence=0.85,
            final_coherence=0.60,  # Not restored
            initial_phase=0.0,
            final_phase=6.28
        )
        assert closed is False

    def test_loop_not_closed_phase_incomplete(self):
        """Loop not closed if phase cycle incomplete."""
        closed = check_loop_closure(
            baseline_coherence=0.85,
            final_coherence=0.86,
            initial_phase=0.0,
            final_phase=3.14  # Only π
        )
        assert closed is False

    def test_loop_closed_phase_wraparound(self):
        """Loop closes with phase wraparound to start."""
        closed = check_loop_closure(
            baseline_coherence=0.85,
            final_coherence=0.84,
            initial_phase=0.1,
            final_phase=0.05  # Returned to start
        )
        assert closed is True

    def test_phi_harmonic_field_closure(self):
        """Full φ-harmonic field produces closed loop."""
        base_time = datetime.now()
        # Simulate φ-harmonic recovery over 2π
        snapshots = []
        for i in range(13):
            phase = i * (2 * math.pi / 12)
            coherence = 0.80 - 0.3 * math.sin(phase / 2) + 0.35 * (i / 12)
            coherence = max(0.4, min(0.95, coherence))
            snapshots.append(ScalarFieldSnapshot(
                base_time + timedelta(seconds=i),
                coherence, 1.0, phase
            ))

        # Last snapshot should be at ~2π with coherence near baseline
        snapshots[-1] = ScalarFieldSnapshot(
            snapshots[-1].timestamp, 0.82, 1.0, 6.28
        )

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        assert graph.loop_closed is True


class TestInversionLogging:
    """Test inversion shift event logging."""

    def test_inversion_detected_phase_flip(self):
        """Inversion detected on phase flip."""
        prev = ScalarFieldSnapshot(datetime.now(), 0.70, 1.0, 0.5)
        curr = ScalarFieldSnapshot(datetime.now(), 0.65, 1.1, 3.5)  # >π phase jump

        inversion = detect_inversion_shift(prev, curr)

        assert inversion is not None
        assert inversion.axis_flipped is True

    def test_inversion_embedded_in_graph(self):
        """Inversion events embedded at correct timestamps."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.75, 1.0, 0.5),
            ScalarFieldSnapshot(base_time + timedelta(seconds=1), 0.55, 1.2, 3.7),  # Phase flip
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.70, 1.0, 4.0),
        ]

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        # Should have inversion in one of the events
        inversions = [e for e in graph.event_trail if e.inversion_shift is not None]
        assert len(inversions) >= 1

    def test_no_inversion_smooth_transition(self):
        """No inversion on smooth coherence transition."""
        prev = ScalarFieldSnapshot(datetime.now(), 0.70, 1.0, 1.0)
        curr = ScalarFieldSnapshot(datetime.now(), 0.72, 1.0, 1.2)

        inversion = detect_inversion_shift(prev, curr)
        assert inversion is None


class TestCoherenceMinima:
    """Test coherence minima detection."""

    def test_find_single_minimum(self):
        """Find single coherence minimum."""
        snapshots = [
            ScalarFieldSnapshot(datetime.now(), 0.80, 1.0, 0.0),
            ScalarFieldSnapshot(datetime.now(), 0.60, 1.0, 1.0),
            ScalarFieldSnapshot(datetime.now(), 0.40, 1.0, 2.0),  # Minimum
            ScalarFieldSnapshot(datetime.now(), 0.55, 1.0, 3.0),
            ScalarFieldSnapshot(datetime.now(), 0.75, 1.0, 4.0),
        ]

        minima = find_coherence_minima(snapshots)
        assert len(minima) == 1
        assert minima[0] == 2

    def test_find_multiple_minima(self):
        """Find multiple coherence minima."""
        snapshots = [
            ScalarFieldSnapshot(datetime.now(), 0.80, 1.0, 0.0),
            ScalarFieldSnapshot(datetime.now(), 0.50, 1.0, 1.0),  # Min 1
            ScalarFieldSnapshot(datetime.now(), 0.70, 1.0, 2.0),
            ScalarFieldSnapshot(datetime.now(), 0.45, 1.0, 3.0),  # Min 2
            ScalarFieldSnapshot(datetime.now(), 0.85, 1.0, 4.0),
        ]

        minima = find_coherence_minima(snapshots)
        assert len(minima) == 2


class TestUpwardRecovery:
    """Test upward recovery tracking."""

    def test_track_recovery_from_minimum(self):
        """Track upward recovery from minimum."""
        snapshots = [
            ScalarFieldSnapshot(datetime.now(), 0.80, 1.0, 0.0),
            ScalarFieldSnapshot(datetime.now(), 0.50, 1.0, 1.0),
            ScalarFieldSnapshot(datetime.now(), 0.35, 1.0, 2.0),  # Min at idx 2
            ScalarFieldSnapshot(datetime.now(), 0.50, 1.0, 3.0),
            ScalarFieldSnapshot(datetime.now(), 0.65, 1.0, 4.0),
            ScalarFieldSnapshot(datetime.now(), 0.80, 1.0, 5.0),
        ]

        recovery = track_upward_recovery(snapshots, 2)

        assert len(recovery) == 4
        assert recovery[0].coherence == 0.35
        assert recovery[-1].coherence == 0.80


class TestRecoveryGraphIntegration:
    """Integration tests for full recovery graph."""

    def test_full_recovery_cycle(self):
        """Complete recovery cycle produces valid graph."""
        base_time = datetime.now()
        snapshots = [
            ScalarFieldSnapshot(base_time, 0.85, 1.0, 0.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=1), 0.70, 1.1, 0.5),
            ScalarFieldSnapshot(base_time + timedelta(seconds=2), 0.42, 1.2, 1.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=3), 0.38, 1.3, 1.5),  # Bottom
            ScalarFieldSnapshot(base_time + timedelta(seconds=4), 0.50, 1.2, 2.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=5), 0.65, 1.1, 3.0),
            ScalarFieldSnapshot(base_time + timedelta(seconds=6), 0.78, 1.0, 4.5),
            ScalarFieldSnapshot(base_time + timedelta(seconds=7), 0.88, 0.9, 6.0),
        ]

        states = [determine_emergence_state(s.coherence) for s in snapshots]
        graph = build_recovery_graph(states, snapshots)

        assert len(graph.event_trail) > 0
        assert graph.dominant_axis in HarmonicAxis
        assert graph.baseline_coherence == 0.85

    def test_empty_snapshots(self):
        """Empty snapshots produce empty graph."""
        graph = build_recovery_graph([], [])

        assert len(graph.event_trail) == 0
        assert graph.loop_closed is False
        assert graph.coherence_delta == 0.0


class TestPhiIntegration:
    """Test phi constant integration."""

    def test_phi_constant_defined(self):
        """Phi constant is correctly defined."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phase_cycle_2pi(self):
        """2π phase cycle constant is correct."""
        assert abs(PHASE_CYCLE_2PI - 6.283185307) < 0.001


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
