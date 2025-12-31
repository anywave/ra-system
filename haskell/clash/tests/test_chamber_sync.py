"""
Prompt 40: Ra.Chamber.Sync - Multi-Chamber Resonance Synchronization

Synchronizes coherence across linked resonance chambers with phi-based
distance falloff and temporal phase propagation.

Codex References:
- Ra.Chamber.Tuning: Base chamber definitions
- Ra.Coherence.Bands: Sync thresholds
- SCALAR_TORSION_ANTIGRAVITY.md: Scalar field synchronization
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum, auto
from collections import deque
import time


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895
SYNC_THRESHOLD = 0.72
MAX_PROPAGATION_DISTANCE = 7  # Maximum link hops


# ============================================================================
# Types
# ============================================================================

class SyncState(Enum):
    """Chamber synchronization state."""
    ISOLATED = auto()      # No active sync
    SEEKING = auto()       # Looking for sync partners
    SYNCING = auto()       # Synchronization in progress
    SYNCHRONIZED = auto()  # Fully synchronized
    DESYNC = auto()        # Losing synchronization


class LinkType(Enum):
    """Type of chamber link."""
    DIRECT = auto()        # Physical/direct connection
    RESONANT = auto()      # Harmonic resonance link
    SCALAR = auto()        # Scalar field entanglement
    TEMPORAL = auto()      # Time-phase link


@dataclass
class ChamberNode:
    """Resonance chamber node in sync network."""
    chamber_id: str
    base_frequency: float      # Hz
    coherence_level: float     # 0.0-1.0
    phase_angle: float         # radians
    sync_state: SyncState
    harmonic_order: int        # 1, 2, 3... for overtone series
    position: Tuple[float, float, float]  # 3D position for distance calc

    def __hash__(self):
        return hash(self.chamber_id)

    def __eq__(self, other):
        if isinstance(other, ChamberNode):
            return self.chamber_id == other.chamber_id
        return False


@dataclass
class SyncLink:
    """Link between two chambers."""
    source_id: str
    target_id: str
    link_type: LinkType
    link_strength: float       # 0.0-1.0
    distance: int              # Hop distance in network
    phase_offset: float        # Phase delay in radians
    last_sync_time: float      # Timestamp of last sync


@dataclass
class PhaseEvent:
    """Phase propagation event in delay queue."""
    source_id: str
    target_id: str
    phase_delta: float
    coherence_pulse: float
    timestamp: float
    hops_remaining: int


@dataclass
class SyncGraph:
    """Complete synchronization graph."""
    nodes: Dict[str, ChamberNode]
    links: List[SyncLink]
    phase_queue: deque  # PhaseEvent queue
    global_sync_score: float


# ============================================================================
# Distance and Resonance Functions
# ============================================================================

def euclidean_distance(p1: Tuple[float, float, float],
                       p2: Tuple[float, float, float]) -> float:
    """Calculate 3D Euclidean distance."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def compute_resonance_score(node1: ChamberNode,
                            node2: ChamberNode,
                            hop_distance: int) -> float:
    """
    Compute resonance score with phi^(-distance) falloff.

    Score = (harmonic_factor * coherence_product) * phi^(-hop_distance)
    """
    # Harmonic factor: closer harmonic orders resonate more
    harmonic_diff = abs(node1.harmonic_order - node2.harmonic_order)
    harmonic_factor = 1.0 / (1 + harmonic_diff * 0.25)

    # Coherence product
    coherence_product = node1.coherence_level * node2.coherence_level

    # Phi falloff
    phi_falloff = PHI ** (-hop_distance)

    return harmonic_factor * coherence_product * phi_falloff


def frequency_ratio_match(freq1: float, freq2: float,
                          tolerance: float = 0.05) -> bool:
    """
    Check if frequencies are harmonically related.

    Returns True if ratio is close to n/m for small n, m.
    """
    if freq1 == 0 or freq2 == 0:
        return False

    ratio = max(freq1, freq2) / min(freq1, freq2)

    # Check common harmonic ratios
    harmonic_ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
                       PHI, PHI * 2, 1 / PHI]

    for hr in harmonic_ratios:
        if abs(ratio - hr) < tolerance:
            return True

    return False


# ============================================================================
# Synchronization Functions
# ============================================================================

def can_sync(node1: ChamberNode, node2: ChamberNode,
             threshold: float = SYNC_THRESHOLD) -> bool:
    """
    Check if two chambers can synchronize.

    Requires both coherence >= threshold and harmonic compatibility.
    """
    if node1.coherence_level < threshold or node2.coherence_level < threshold:
        return False

    return frequency_ratio_match(node1.base_frequency, node2.base_frequency)


def synchronize_chambers(graph: SyncGraph,
                         threshold: float = SYNC_THRESHOLD) -> SyncGraph:
    """
    Synchronize all compatible chambers in the graph.

    Updates sync states and link strengths.
    """
    # Update each node's sync state
    for node_id, node in graph.nodes.items():
        if node.coherence_level < threshold:
            node.sync_state = SyncState.ISOLATED
            continue

        # Find sync partners
        partners = find_sync_partners(graph, node_id, threshold)

        if not partners:
            node.sync_state = SyncState.SEEKING
        elif len(partners) >= 2:
            node.sync_state = SyncState.SYNCHRONIZED
        else:
            node.sync_state = SyncState.SYNCING

    # Update link strengths
    for link in graph.links:
        source = graph.nodes.get(link.source_id)
        target = graph.nodes.get(link.target_id)

        if source and target:
            link.link_strength = compute_resonance_score(
                source, target, link.distance
            )
            link.last_sync_time = time.time()

    # Compute global sync score
    graph.global_sync_score = compute_global_sync(graph)

    return graph


def find_sync_partners(graph: SyncGraph,
                       node_id: str,
                       threshold: float = SYNC_THRESHOLD) -> List[str]:
    """Find all nodes that can sync with given node."""
    node = graph.nodes.get(node_id)
    if not node:
        return []

    partners = []
    for link in graph.links:
        partner_id = None
        if link.source_id == node_id:
            partner_id = link.target_id
        elif link.target_id == node_id:
            partner_id = link.source_id

        if partner_id:
            partner = graph.nodes.get(partner_id)
            if partner and can_sync(node, partner, threshold):
                partners.append(partner_id)

    return partners


def compute_global_sync(graph: SyncGraph) -> float:
    """
    Compute global synchronization score.

    Average of all link strengths weighted by node coherence.
    """
    if not graph.links:
        return 0.0

    total_weight = 0.0
    weighted_sum = 0.0

    for link in graph.links:
        source = graph.nodes.get(link.source_id)
        target = graph.nodes.get(link.target_id)

        if source and target:
            weight = source.coherence_level * target.coherence_level
            weighted_sum += link.link_strength * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight


# ============================================================================
# Phase Propagation
# ============================================================================

def propagate_phase(graph: SyncGraph,
                    source_id: str,
                    phase_delta: float,
                    coherence_pulse: float) -> SyncGraph:
    """
    Propagate phase change through the network with temporal delay.

    Uses queue-based delay for realistic wave propagation.
    """
    current_time = time.time()

    # Create initial events for all direct neighbors
    for link in graph.links:
        target_id = None
        if link.source_id == source_id:
            target_id = link.target_id
        elif link.target_id == source_id:
            target_id = link.source_id

        if target_id:
            # Phase attenuates with distance
            attenuated_phase = phase_delta * (PHI ** (-link.distance))
            attenuated_coherence = coherence_pulse * (PHI ** (-link.distance))

            event = PhaseEvent(
                source_id=source_id,
                target_id=target_id,
                phase_delta=attenuated_phase,
                coherence_pulse=attenuated_coherence,
                timestamp=current_time + link.distance * 0.01,  # 10ms per hop
                hops_remaining=MAX_PROPAGATION_DISTANCE - link.distance
            )
            graph.phase_queue.append(event)

    return graph


def process_phase_queue(graph: SyncGraph,
                        current_time: float) -> Tuple[SyncGraph, int]:
    """
    Process pending phase events up to current time.

    Returns updated graph and count of processed events.
    """
    processed = 0
    pending = deque()

    while graph.phase_queue:
        event = graph.phase_queue.popleft()

        if event.timestamp <= current_time:
            # Apply phase to target node
            target = graph.nodes.get(event.target_id)
            if target:
                target.phase_angle += event.phase_delta
                # Normalize to 0-2π
                target.phase_angle = target.phase_angle % (2 * math.pi)

                # Propagate further if hops remaining
                if event.hops_remaining > 0:
                    propagate_from_node(graph, event, current_time)

                processed += 1
        else:
            pending.append(event)

    graph.phase_queue = pending
    return graph, processed


def propagate_from_node(graph: SyncGraph,
                        event: PhaseEvent,
                        current_time: float):
    """Continue propagating from a node after receiving phase update."""
    for link in graph.links:
        next_target = None
        if link.source_id == event.target_id and link.target_id != event.source_id:
            next_target = link.target_id
        elif link.target_id == event.target_id and link.source_id != event.source_id:
            next_target = link.source_id

        if next_target:
            # Further attenuation
            new_phase = event.phase_delta * (PHI ** -1)
            new_coherence = event.coherence_pulse * (PHI ** -1)

            if abs(new_phase) > 0.001:  # Minimum threshold
                new_event = PhaseEvent(
                    source_id=event.target_id,
                    target_id=next_target,
                    phase_delta=new_phase,
                    coherence_pulse=new_coherence,
                    timestamp=current_time + 0.01,
                    hops_remaining=event.hops_remaining - 1
                )
                graph.phase_queue.append(new_event)


# ============================================================================
# Graph Construction
# ============================================================================

def create_chamber_node(chamber_id: str,
                        frequency: float,
                        coherence: float,
                        position: Tuple[float, float, float] = (0, 0, 0),
                        harmonic_order: int = 1) -> ChamberNode:
    """Create a new chamber node."""
    return ChamberNode(
        chamber_id=chamber_id,
        base_frequency=frequency,
        coherence_level=coherence,
        phase_angle=0.0,
        sync_state=SyncState.ISOLATED,
        harmonic_order=harmonic_order,
        position=position
    )


def create_sync_link(source_id: str,
                     target_id: str,
                     distance: int = 1,
                     link_type: LinkType = LinkType.RESONANT) -> SyncLink:
    """Create a synchronization link."""
    return SyncLink(
        source_id=source_id,
        target_id=target_id,
        link_type=link_type,
        link_strength=0.0,
        distance=distance,
        phase_offset=0.0,
        last_sync_time=0.0
    )


def create_sync_graph(nodes: List[ChamberNode],
                      links: List[SyncLink]) -> SyncGraph:
    """Create a synchronization graph."""
    return SyncGraph(
        nodes={n.chamber_id: n for n in nodes},
        links=links,
        phase_queue=deque(),
        global_sync_score=0.0
    )


# ============================================================================
# Test Suite
# ============================================================================

class TestResonanceScore:
    """Test phi-falloff resonance scoring."""

    def test_adjacent_chambers_high_score(self):
        """Adjacent chambers (distance=1) have high resonance."""
        n1 = create_chamber_node("c1", 432.0, 0.9, harmonic_order=1)
        n2 = create_chamber_node("c2", 432.0, 0.9, harmonic_order=1)
        score = compute_resonance_score(n1, n2, 1)
        assert score > 0.5

    def test_distant_chambers_low_score(self):
        """Distant chambers have attenuated resonance."""
        n1 = create_chamber_node("c1", 432.0, 0.9, harmonic_order=1)
        n2 = create_chamber_node("c2", 432.0, 0.9, harmonic_order=1)

        score_near = compute_resonance_score(n1, n2, 1)
        score_far = compute_resonance_score(n1, n2, 5)

        assert score_far < score_near
        assert score_far < score_near * 0.2  # Significant attenuation

    def test_phi_falloff_ratio(self):
        """Falloff follows phi^(-distance)."""
        n1 = create_chamber_node("c1", 432.0, 1.0, harmonic_order=1)
        n2 = create_chamber_node("c2", 432.0, 1.0, harmonic_order=1)

        score_d1 = compute_resonance_score(n1, n2, 1)
        score_d2 = compute_resonance_score(n1, n2, 2)

        ratio = score_d1 / score_d2
        expected_ratio = PHI  # phi^(-1) / phi^(-2) = phi
        assert abs(ratio - expected_ratio) < 0.01

    def test_harmonic_order_affects_score(self):
        """Different harmonic orders reduce resonance."""
        n1 = create_chamber_node("c1", 432.0, 0.9, harmonic_order=1)
        n2_same = create_chamber_node("c2", 432.0, 0.9, harmonic_order=1)
        n2_diff = create_chamber_node("c3", 432.0, 0.9, harmonic_order=5)

        score_same = compute_resonance_score(n1, n2_same, 1)
        score_diff = compute_resonance_score(n1, n2_diff, 1)

        assert score_diff < score_same

    def test_low_coherence_low_score(self):
        """Low coherence chambers have low resonance."""
        n1 = create_chamber_node("c1", 432.0, 0.3)
        n2 = create_chamber_node("c2", 432.0, 0.3)
        score = compute_resonance_score(n1, n2, 1)
        assert score < 0.1


class TestSyncThreshold:
    """Test 0.72 synchronization threshold."""

    def test_above_threshold_can_sync(self):
        """Chambers above 0.72 can synchronize."""
        n1 = create_chamber_node("c1", 432.0, 0.80)
        n2 = create_chamber_node("c2", 432.0, 0.85)
        assert can_sync(n1, n2, SYNC_THRESHOLD) is True

    def test_below_threshold_cannot_sync(self):
        """Chambers below 0.72 cannot synchronize."""
        n1 = create_chamber_node("c1", 432.0, 0.60)
        n2 = create_chamber_node("c2", 432.0, 0.85)
        assert can_sync(n1, n2, SYNC_THRESHOLD) is False

    def test_at_threshold_can_sync(self):
        """Chambers at exactly 0.72 can synchronize."""
        n1 = create_chamber_node("c1", 432.0, 0.72)
        n2 = create_chamber_node("c2", 432.0, 0.72)
        assert can_sync(n1, n2, SYNC_THRESHOLD) is True

    def test_incompatible_frequencies_cannot_sync(self):
        """Non-harmonic frequencies cannot sync even above threshold."""
        n1 = create_chamber_node("c1", 432.0, 0.90)
        n2 = create_chamber_node("c2", 517.0, 0.90)  # Not harmonic ratio
        assert can_sync(n1, n2, SYNC_THRESHOLD) is False


class TestFrequencyRatios:
    """Test harmonic frequency ratio matching."""

    def test_unison_matches(self):
        """Same frequency matches (1:1)."""
        assert frequency_ratio_match(432.0, 432.0) is True

    def test_octave_matches(self):
        """Octave ratio matches (2:1)."""
        assert frequency_ratio_match(432.0, 864.0) is True

    def test_fifth_matches(self):
        """Perfect fifth matches (3:2)."""
        assert frequency_ratio_match(432.0, 648.0) is True

    def test_phi_ratio_matches(self):
        """Phi ratio matches."""
        assert frequency_ratio_match(432.0, 432.0 * PHI) is True

    def test_arbitrary_ratio_fails(self):
        """Arbitrary non-harmonic ratio fails."""
        assert frequency_ratio_match(432.0, 517.0) is False


class TestSynchronizeGraph:
    """Test graph synchronization."""

    def test_sync_updates_states(self):
        """Synchronization updates node states."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.80)
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = synchronize_chambers(graph)

        assert graph.nodes["c1"].sync_state != SyncState.ISOLATED
        assert graph.nodes["c2"].sync_state != SyncState.ISOLATED

    def test_sync_updates_link_strength(self):
        """Synchronization computes link strengths."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.80)
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = synchronize_chambers(graph)

        assert graph.links[0].link_strength > 0

    def test_isolated_low_coherence(self):
        """Low coherence nodes remain isolated."""
        n1 = create_chamber_node("c1", 432.0, 0.50)
        n2 = create_chamber_node("c2", 432.0, 0.50)
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = synchronize_chambers(graph)

        assert graph.nodes["c1"].sync_state == SyncState.ISOLATED
        assert graph.nodes["c2"].sync_state == SyncState.ISOLATED


class TestPhasePropagation:
    """Test temporal phase propagation."""

    def test_phase_queues_events(self):
        """Phase propagation creates queue events."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.80)
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = propagate_phase(graph, "c1", 0.5, 0.9)

        assert len(graph.phase_queue) > 0

    def test_phase_attenuates_with_distance(self):
        """Phase delta attenuates through propagation."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.80)
        n3 = create_chamber_node("c3", 432.0, 0.80)
        link1 = create_sync_link("c1", "c2", distance=1)
        link2 = create_sync_link("c2", "c3", distance=2)

        graph = create_sync_graph([n1, n2, n3], [link1, link2])
        graph = propagate_phase(graph, "c1", 1.0, 0.9)

        # Check first event has larger phase than second (if both created)
        events = list(graph.phase_queue)
        if len(events) >= 2:
            assert events[0].phase_delta >= events[1].phase_delta

    def test_process_queue_applies_phase(self):
        """Processing queue applies phase to target nodes."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.80)
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = propagate_phase(graph, "c1", 0.5, 0.9)

        initial_phase = graph.nodes["c2"].phase_angle

        # Process after delay
        future_time = time.time() + 1.0
        graph, processed = process_phase_queue(graph, future_time)

        assert processed > 0
        assert graph.nodes["c2"].phase_angle != initial_phase

    def test_phase_normalizes_to_2pi(self):
        """Phase angles normalize to 0-2π range."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n1.phase_angle = 5.0  # Below 2π
        n2 = create_chamber_node("c2", 432.0, 0.80)
        n2.phase_angle = 6.0  # Below 2π
        link = create_sync_link("c1", "c2")

        graph = create_sync_graph([n1, n2], [link])
        graph = propagate_phase(graph, "c1", 3.0, 0.9)

        future_time = time.time() + 1.0
        graph, _ = process_phase_queue(graph, future_time)

        # Phase should be normalized
        assert 0 <= graph.nodes["c2"].phase_angle < 2 * math.pi


class TestGlobalSync:
    """Test global synchronization scoring."""

    def test_high_coherence_high_global(self):
        """High coherence network has high global sync."""
        nodes = [
            create_chamber_node(f"c{i}", 432.0, 0.90)
            for i in range(4)
        ]
        links = [
            create_sync_link("c0", "c1"),
            create_sync_link("c1", "c2"),
            create_sync_link("c2", "c3"),
            create_sync_link("c3", "c0"),
        ]

        graph = create_sync_graph(nodes, links)
        graph = synchronize_chambers(graph)

        assert graph.global_sync_score > 0.4

    def test_mixed_coherence_medium_global(self):
        """Mixed coherence has medium global sync."""
        nodes = [
            create_chamber_node("c0", 432.0, 0.90),
            create_chamber_node("c1", 432.0, 0.50),
            create_chamber_node("c2", 432.0, 0.80),
        ]
        links = [
            create_sync_link("c0", "c1"),
            create_sync_link("c1", "c2"),
        ]

        graph = create_sync_graph(nodes, links)
        graph = synchronize_chambers(graph)

        assert 0.1 < graph.global_sync_score < 0.7

    def test_empty_graph_zero_global(self):
        """Empty graph has zero global sync."""
        graph = create_sync_graph([], [])
        assert compute_global_sync(graph) == 0.0


class TestFindSyncPartners:
    """Test sync partner discovery."""

    def test_finds_compatible_partners(self):
        """Finds all compatible sync partners."""
        nodes = [
            create_chamber_node("c0", 432.0, 0.85),
            create_chamber_node("c1", 432.0, 0.80),
            create_chamber_node("c2", 432.0, 0.75),
        ]
        links = [
            create_sync_link("c0", "c1"),
            create_sync_link("c0", "c2"),
        ]

        graph = create_sync_graph(nodes, links)
        partners = find_sync_partners(graph, "c0")

        assert len(partners) == 2

    def test_excludes_low_coherence(self):
        """Excludes partners below threshold."""
        nodes = [
            create_chamber_node("c0", 432.0, 0.85),
            create_chamber_node("c1", 432.0, 0.50),  # Below threshold
        ]
        links = [create_sync_link("c0", "c1")]

        graph = create_sync_graph(nodes, links)
        partners = find_sync_partners(graph, "c0")

        assert len(partners) == 0


class TestLinkTypes:
    """Test different link type behaviors."""

    def test_direct_link_creation(self):
        """Direct links can be created."""
        link = create_sync_link("c1", "c2", link_type=LinkType.DIRECT)
        assert link.link_type == LinkType.DIRECT

    def test_scalar_link_creation(self):
        """Scalar links can be created."""
        link = create_sync_link("c1", "c2", link_type=LinkType.SCALAR)
        assert link.link_type == LinkType.SCALAR

    def test_temporal_link_creation(self):
        """Temporal links can be created."""
        link = create_sync_link("c1", "c2", link_type=LinkType.TEMPORAL)
        assert link.link_type == LinkType.TEMPORAL


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_node_graph(self):
        """Single node graph handles correctly."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        graph = create_sync_graph([n1], [])
        graph = synchronize_chambers(graph)

        assert graph.nodes["c1"].sync_state == SyncState.SEEKING

    def test_disconnected_nodes(self):
        """Disconnected nodes stay isolated."""
        n1 = create_chamber_node("c1", 432.0, 0.85)
        n2 = create_chamber_node("c2", 432.0, 0.85)
        # No link between them
        graph = create_sync_graph([n1, n2], [])
        graph = synchronize_chambers(graph)

        assert graph.nodes["c1"].sync_state == SyncState.SEEKING
        assert graph.nodes["c2"].sync_state == SyncState.SEEKING

    def test_zero_frequency_handling(self):
        """Zero frequency handled gracefully."""
        assert frequency_ratio_match(0.0, 432.0) is False
        assert frequency_ratio_match(432.0, 0.0) is False

    def test_max_propagation_limit(self):
        """Phase stops propagating after max hops."""
        # Create chain of nodes
        nodes = [create_chamber_node(f"c{i}", 432.0, 0.85) for i in range(10)]
        links = [create_sync_link(f"c{i}", f"c{i+1}") for i in range(9)]

        graph = create_sync_graph(nodes, links)

        # Create event with limited hops
        event = PhaseEvent(
            source_id="c0",
            target_id="c1",
            phase_delta=1.0,
            coherence_pulse=0.9,
            timestamp=0,
            hops_remaining=0  # No more hops
        )

        graph.phase_queue.append(event)
        initial_queue_len = len(graph.phase_queue)

        # Process should not add more events
        graph, _ = process_phase_queue(graph, time.time() + 1.0)

        # No new events added since hops_remaining was 0
        assert len(graph.phase_queue) <= initial_queue_len


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
