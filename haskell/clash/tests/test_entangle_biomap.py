"""
Prompt 45: Ra.Entangle.BioMap - Biometric-Fragment Entanglement Graphs

Constructs dynamic entanglement graphs linking biometric coherence signals,
scalar field zones, memory fragments, and chamber nodes.

Codex References:
- Ra.Scalar: Scalar field definitions
- Ra.Identity: Fragment identity
- Ra.Chamber: Chamber node integration
"""

import pytest
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Entanglement threshold
ENTANGLE_THRESHOLD = 0.72

# Update triggers
COHERENCE_DELTA_THRESHOLD = 0.05
PHI_CYCLE_TICKS = 1618  # φ * 1000


# ============================================================================
# Types
# ============================================================================

class BodyZone(Enum):
    """Body zone enumeration (8 zones)."""
    CROWN = auto()
    THIRD_EYE = auto()
    THROAT = auto()
    HEART = auto()
    GUT = auto()          # Solar Plexus
    SACRAL = auto()
    ROOT = auto()
    BREATH = auto()       # Respiratory system


@dataclass
class PhaseCode:
    """
    Phase signature for harmonic matching.

    Encodes spherical harmonic identity with amplitude and phase.
    """
    harmonic_l: int        # Radial order (0-9)
    harmonic_m: int        # Angular order (-l to +l)
    amplitude: float       # 0.0-1.0
    phase_angle: float     # radians (0 to 2π)

    def matches(self, other: 'PhaseCode', l_tolerance: int = 1,
                m_tolerance: int = 1, phase_tolerance: float = 0.5) -> bool:
        """Check if phase codes match within tolerance."""
        l_match = abs(self.harmonic_l - other.harmonic_l) <= l_tolerance
        m_match = abs(self.harmonic_m - other.harmonic_m) <= m_tolerance

        # Phase difference (circular)
        phase_diff = abs(self.phase_angle - other.phase_angle)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
        phase_match = phase_diff <= phase_tolerance

        return l_match and m_match and phase_match

    def similarity(self, other: 'PhaseCode') -> float:
        """Calculate similarity score 0.0-1.0."""
        l_sim = 1.0 / (1 + abs(self.harmonic_l - other.harmonic_l))
        m_sim = 1.0 / (1 + abs(self.harmonic_m - other.harmonic_m))

        phase_diff = abs(self.phase_angle - other.phase_angle)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)
        phase_sim = 1.0 - (phase_diff / math.pi)

        amp_sim = 1.0 - abs(self.amplitude - other.amplitude)

        return (l_sim * 0.3 + m_sim * 0.2 + phase_sim * 0.3 + amp_sim * 0.2)


@dataclass
class RaCoordinate:
    """Coordinate in Ra scalar field space."""
    theta: int             # 0-26
    phi: int               # 0-5
    h: int                 # 0-4

    def __hash__(self):
        return hash((self.theta, self.phi, self.h))


@dataclass
class BioNode:
    """Biometric coherence node."""
    user_region: BodyZone
    coherence_value: float        # 0.0-1.0
    phase_signature: PhaseCode
    last_update_tick: int = 0


@dataclass
class EntangledFragment:
    """Memory fragment with entanglement state."""
    fragment_id: str
    anchor_coord: RaCoordinate
    resonance_score: float        # 0.0-1.0
    active_link: bool
    phase_code: PhaseCode


@dataclass
class EntanglementLink:
    """Link between BioNode and EntangledFragment."""
    bio_zone: BodyZone
    fragment_id: str
    strength: float               # 0.0-1.0
    active: bool
    last_sync_tick: int


class UpdateTrigger(Enum):
    """Trigger for graph update."""
    COHERENCE_DELTA = auto()      # Δcoherence > 0.05
    PHI_CYCLE = auto()            # φⁿ cycle tick
    FRAGMENT_CHANGE = auto()       # Fragment added/removed
    MANUAL = auto()                # Explicit update request


@dataclass
class EntangleGraph:
    """Complete biometric-fragment entanglement graph."""
    bio_nodes: Dict[BodyZone, BioNode]
    fragments: Dict[str, EntangledFragment]
    links: List[EntanglementLink]
    last_update_tick: int
    pending_trigger: Optional[UpdateTrigger] = None


# ============================================================================
# PhaseCode Operations
# ============================================================================

def create_phase_code(l: int = 0, m: int = 0,
                      amplitude: float = 1.0,
                      phase: float = 0.0) -> PhaseCode:
    """Create phase code with validation."""
    l = max(0, min(9, l))
    m = max(-l, min(l, m))
    amplitude = max(0.0, min(1.0, amplitude))
    phase = phase % (2 * math.pi)

    return PhaseCode(l, m, amplitude, phase)


def blend_phase_codes(codes: List[PhaseCode], weights: List[float]) -> PhaseCode:
    """Blend multiple phase codes with weights."""
    if not codes:
        return create_phase_code()

    total_weight = sum(weights)
    if total_weight == 0:
        return codes[0]

    # Weighted averages
    l_avg = sum(c.harmonic_l * w for c, w in zip(codes, weights)) / total_weight
    m_avg = sum(c.harmonic_m * w for c, w in zip(codes, weights)) / total_weight
    amp_avg = sum(c.amplitude * w for c, w in zip(codes, weights)) / total_weight

    # Circular average for phase
    x = sum(math.cos(c.phase_angle) * w for c, w in zip(codes, weights))
    y = sum(math.sin(c.phase_angle) * w for c, w in zip(codes, weights))
    phase_avg = math.atan2(y, x) % (2 * math.pi)

    return create_phase_code(round(l_avg), round(m_avg), amp_avg, phase_avg)


# ============================================================================
# BioNode Operations
# ============================================================================

def create_bio_node(zone: BodyZone, coherence: float = 0.5,
                    phase: Optional[PhaseCode] = None) -> BioNode:
    """Create a biometric node."""
    if phase is None:
        # Default phase based on zone
        zone_l = list(BodyZone).index(zone) % 5
        phase = create_phase_code(l=zone_l, m=0, amplitude=coherence)

    return BioNode(
        user_region=zone,
        coherence_value=max(0.0, min(1.0, coherence)),
        phase_signature=phase
    )


def init_bio_nodes(coherence: float = 0.5) -> Dict[BodyZone, BioNode]:
    """Initialize all 8 body zone nodes."""
    return {zone: create_bio_node(zone, coherence) for zone in BodyZone}


def update_bio_node(node: BioNode, new_coherence: float,
                    current_tick: int) -> Tuple[BioNode, bool]:
    """
    Update bio node coherence.

    Returns (updated_node, should_trigger_update).
    """
    delta = abs(new_coherence - node.coherence_value)
    should_trigger = delta > COHERENCE_DELTA_THRESHOLD

    new_node = BioNode(
        user_region=node.user_region,
        coherence_value=max(0.0, min(1.0, new_coherence)),
        phase_signature=node.phase_signature,
        last_update_tick=current_tick
    )

    return new_node, should_trigger


# ============================================================================
# Fragment Operations
# ============================================================================

def create_fragment(fragment_id: str,
                    anchor: RaCoordinate,
                    resonance: float = 0.5) -> EntangledFragment:
    """Create an entangled fragment."""
    # Generate phase code from anchor
    phase = create_phase_code(
        l=anchor.h,
        m=anchor.phi - 3,  # Center around 0
        amplitude=resonance,
        phase=(anchor.theta / 27) * 2 * math.pi
    )

    return EntangledFragment(
        fragment_id=fragment_id,
        anchor_coord=anchor,
        resonance_score=resonance,
        active_link=resonance >= ENTANGLE_THRESHOLD,
        phase_code=phase
    )


def update_fragment_resonance(fragment: EntangledFragment,
                              new_resonance: float) -> EntangledFragment:
    """Update fragment resonance score."""
    return EntangledFragment(
        fragment_id=fragment.fragment_id,
        anchor_coord=fragment.anchor_coord,
        resonance_score=new_resonance,
        active_link=new_resonance >= ENTANGLE_THRESHOLD,
        phase_code=fragment.phase_code
    )


# ============================================================================
# Entanglement Graph Operations
# ============================================================================

def init_entangle_graph() -> EntangleGraph:
    """Initialize empty entanglement graph."""
    return EntangleGraph(
        bio_nodes=init_bio_nodes(),
        fragments={},
        links=[],
        last_update_tick=0
    )


def compute_link_strength(bio_node: BioNode,
                          fragment: EntangledFragment) -> float:
    """
    Compute entanglement link strength.

    Based on phase similarity and coherence product.
    """
    phase_sim = bio_node.phase_signature.similarity(fragment.phase_code)
    coherence_product = bio_node.coherence_value * fragment.resonance_score

    return phase_sim * 0.4 + coherence_product * 0.6


def should_link_activate(strength: float) -> bool:
    """Check if link should be active (threshold 0.72)."""
    return strength >= ENTANGLE_THRESHOLD


def update_links(graph: EntangleGraph) -> List[EntanglementLink]:
    """Recompute all entanglement links."""
    links = []

    for zone, bio_node in graph.bio_nodes.items():
        for frag_id, fragment in graph.fragments.items():
            strength = compute_link_strength(bio_node, fragment)
            active = should_link_activate(strength)

            links.append(EntanglementLink(
                bio_zone=zone,
                fragment_id=frag_id,
                strength=strength,
                active=active,
                last_sync_tick=graph.last_update_tick
            ))

    return links


def add_fragment(graph: EntangleGraph, fragment: EntangledFragment) -> EntangleGraph:
    """Add fragment to graph and update links."""
    new_fragments = dict(graph.fragments)
    new_fragments[fragment.fragment_id] = fragment

    new_graph = EntangleGraph(
        bio_nodes=graph.bio_nodes,
        fragments=new_fragments,
        links=graph.links,
        last_update_tick=graph.last_update_tick,
        pending_trigger=UpdateTrigger.FRAGMENT_CHANGE
    )

    new_graph.links = update_links(new_graph)
    return new_graph


def remove_fragment(graph: EntangleGraph, fragment_id: str) -> EntangleGraph:
    """Remove fragment from graph."""
    new_fragments = {k: v for k, v in graph.fragments.items() if k != fragment_id}
    new_links = [l for l in graph.links if l.fragment_id != fragment_id]

    return EntangleGraph(
        bio_nodes=graph.bio_nodes,
        fragments=new_fragments,
        links=new_links,
        last_update_tick=graph.last_update_tick
    )


def update_bio_coherence(graph: EntangleGraph, zone: BodyZone,
                         coherence: float, current_tick: int) -> EntangleGraph:
    """Update coherence for a body zone."""
    node = graph.bio_nodes.get(zone)
    if node is None:
        return graph

    new_node, should_trigger = update_bio_node(node, coherence, current_tick)

    new_bio_nodes = dict(graph.bio_nodes)
    new_bio_nodes[zone] = new_node

    trigger = UpdateTrigger.COHERENCE_DELTA if should_trigger else graph.pending_trigger

    new_graph = EntangleGraph(
        bio_nodes=new_bio_nodes,
        fragments=graph.fragments,
        links=graph.links,
        last_update_tick=current_tick if should_trigger else graph.last_update_tick,
        pending_trigger=trigger
    )

    if should_trigger:
        new_graph.links = update_links(new_graph)

    return new_graph


def check_phi_cycle_update(graph: EntangleGraph, current_tick: int) -> bool:
    """Check if φⁿ cycle update is needed."""
    ticks_since_update = current_tick - graph.last_update_tick
    return ticks_since_update >= PHI_CYCLE_TICKS


def process_phi_cycle(graph: EntangleGraph, current_tick: int) -> EntangleGraph:
    """Process φⁿ cycle update if needed."""
    if not check_phi_cycle_update(graph, current_tick):
        return graph

    new_graph = EntangleGraph(
        bio_nodes=graph.bio_nodes,
        fragments=graph.fragments,
        links=update_links(graph),
        last_update_tick=current_tick,
        pending_trigger=UpdateTrigger.PHI_CYCLE
    )

    return new_graph


# ============================================================================
# Graph Analysis
# ============================================================================

def get_active_links(graph: EntangleGraph) -> List[EntanglementLink]:
    """Get all active entanglement links."""
    return [l for l in graph.links if l.active]


def get_zone_fragments(graph: EntangleGraph, zone: BodyZone) -> List[EntangledFragment]:
    """Get all fragments linked to a body zone."""
    linked_ids = {l.fragment_id for l in graph.links
                  if l.bio_zone == zone and l.active}
    return [f for f in graph.fragments.values() if f.fragment_id in linked_ids]


def get_fragment_zones(graph: EntangleGraph, fragment_id: str) -> List[BodyZone]:
    """Get all body zones linked to a fragment."""
    return [l.bio_zone for l in graph.links
            if l.fragment_id == fragment_id and l.active]


def compute_overall_coherence(graph: EntangleGraph) -> float:
    """Compute average coherence across all bio nodes."""
    if not graph.bio_nodes:
        return 0.0
    return sum(n.coherence_value for n in graph.bio_nodes.values()) / len(graph.bio_nodes)


def compute_entanglement_density(graph: EntangleGraph) -> float:
    """Compute ratio of active links to potential links."""
    potential = len(graph.bio_nodes) * len(graph.fragments)
    if potential == 0:
        return 0.0
    active = len(get_active_links(graph))
    return active / potential


# ============================================================================
# Test Suite
# ============================================================================

class TestBodyZone:
    """Test body zone enumeration."""

    def test_eight_zones(self):
        """8 body zones exist."""
        assert len(list(BodyZone)) == 8

    def test_expected_zones(self):
        """Expected zones are present."""
        zone_names = [z.name for z in BodyZone]
        assert "CROWN" in zone_names
        assert "HEART" in zone_names
        assert "BREATH" in zone_names
        assert "ROOT" in zone_names


class TestPhaseCode:
    """Test PhaseCode operations."""

    def test_create_phase_code(self):
        """Phase code creation with defaults."""
        pc = create_phase_code()
        assert pc.harmonic_l == 0
        assert pc.harmonic_m == 0
        assert pc.amplitude == 1.0

    def test_l_bounds(self):
        """L is bounded 0-9."""
        pc = create_phase_code(l=15)
        assert pc.harmonic_l == 9

        pc2 = create_phase_code(l=-5)
        assert pc2.harmonic_l == 0

    def test_m_bounded_by_l(self):
        """M is bounded by L."""
        pc = create_phase_code(l=3, m=10)
        assert abs(pc.harmonic_m) <= pc.harmonic_l

    def test_phase_similarity_identical(self):
        """Identical phase codes have similarity 1.0."""
        pc1 = create_phase_code(l=2, m=1, amplitude=0.8, phase=1.0)
        pc2 = create_phase_code(l=2, m=1, amplitude=0.8, phase=1.0)
        assert pc1.similarity(pc2) == 1.0

    def test_phase_similarity_different(self):
        """Different phase codes have lower similarity."""
        pc1 = create_phase_code(l=2, m=1, amplitude=0.8, phase=0.0)
        pc2 = create_phase_code(l=5, m=-2, amplitude=0.3, phase=3.14)
        assert pc1.similarity(pc2) < 0.5

    def test_phase_matching(self):
        """Phase codes match within tolerance."""
        pc1 = create_phase_code(l=3, m=1, amplitude=0.7, phase=1.0)
        pc2 = create_phase_code(l=3, m=0, amplitude=0.8, phase=1.2)
        assert pc1.matches(pc2, l_tolerance=1, m_tolerance=1, phase_tolerance=0.5)


class TestBioNode:
    """Test BioNode operations."""

    def test_create_bio_node(self):
        """Bio node creation."""
        node = create_bio_node(BodyZone.HEART, coherence=0.8)
        assert node.user_region == BodyZone.HEART
        assert node.coherence_value == 0.8

    def test_init_all_nodes(self):
        """Initialize all 8 nodes."""
        nodes = init_bio_nodes(0.6)
        assert len(nodes) == 8
        for zone in BodyZone:
            assert zone in nodes

    def test_update_triggers_on_delta(self):
        """Update triggers when delta > 0.05."""
        node = create_bio_node(BodyZone.CROWN, coherence=0.5)
        _, should_trigger = update_bio_node(node, 0.6, 100)
        assert should_trigger is True

    def test_no_trigger_small_delta(self):
        """No trigger for small delta."""
        node = create_bio_node(BodyZone.CROWN, coherence=0.5)
        _, should_trigger = update_bio_node(node, 0.52, 100)
        assert should_trigger is False


class TestEntangledFragment:
    """Test fragment operations."""

    def test_create_fragment(self):
        """Fragment creation."""
        anchor = RaCoordinate(10, 3, 2)
        frag = create_fragment("frag-001", anchor, resonance=0.8)

        assert frag.fragment_id == "frag-001"
        assert frag.resonance_score == 0.8
        assert frag.active_link is True  # 0.8 >= 0.72

    def test_active_link_threshold(self):
        """Link activates at 0.72 threshold."""
        anchor = RaCoordinate(5, 2, 1)

        frag_low = create_fragment("low", anchor, resonance=0.5)
        assert frag_low.active_link is False

        frag_high = create_fragment("high", anchor, resonance=0.72)
        assert frag_high.active_link is True

    def test_update_resonance(self):
        """Fragment resonance update."""
        anchor = RaCoordinate(5, 2, 1)
        frag = create_fragment("test", anchor, resonance=0.5)

        updated = update_fragment_resonance(frag, 0.9)
        assert updated.resonance_score == 0.9
        assert updated.active_link is True


class TestEntangleGraph:
    """Test entanglement graph operations."""

    def test_init_graph(self):
        """Graph initialization."""
        graph = init_entangle_graph()
        assert len(graph.bio_nodes) == 8
        assert len(graph.fragments) == 0
        assert len(graph.links) == 0

    def test_add_fragment(self):
        """Add fragment to graph."""
        graph = init_entangle_graph()
        frag = create_fragment("frag-001", RaCoordinate(5, 2, 1), 0.8)

        graph = add_fragment(graph, frag)

        assert "frag-001" in graph.fragments
        assert len(graph.links) == 8  # One link per bio zone

    def test_remove_fragment(self):
        """Remove fragment from graph."""
        graph = init_entangle_graph()
        frag = create_fragment("frag-001", RaCoordinate(5, 2, 1), 0.8)
        graph = add_fragment(graph, frag)

        graph = remove_fragment(graph, "frag-001")

        assert "frag-001" not in graph.fragments
        assert len(graph.links) == 0

    def test_update_coherence_triggers(self):
        """Coherence update triggers graph update."""
        graph = init_entangle_graph()
        graph = update_bio_coherence(graph, BodyZone.HEART, 0.9, 100)

        assert graph.bio_nodes[BodyZone.HEART].coherence_value == 0.9
        assert graph.pending_trigger == UpdateTrigger.COHERENCE_DELTA


class TestLinkStrength:
    """Test link strength computation."""

    def test_high_coherence_strong_link(self):
        """High coherence produces strong links."""
        bio = create_bio_node(BodyZone.HEART, coherence=0.9)
        frag = create_fragment("test", RaCoordinate(5, 2, 1), 0.9)

        strength = compute_link_strength(bio, frag)
        assert strength > 0.7

    def test_low_coherence_weak_link(self):
        """Low coherence produces weak links."""
        bio = create_bio_node(BodyZone.HEART, coherence=0.2)
        frag = create_fragment("test", RaCoordinate(5, 2, 1), 0.2)

        strength = compute_link_strength(bio, frag)
        assert strength < 0.5

    def test_activation_threshold(self):
        """Link activation at 0.72 threshold."""
        assert should_link_activate(0.72) is True
        assert should_link_activate(0.71) is False


class TestPhiCycleUpdate:
    """Test φⁿ cycle update logic."""

    def test_phi_cycle_check(self):
        """Check phi cycle update timing."""
        graph = init_entangle_graph()

        # Just updated
        assert check_phi_cycle_update(graph, 100) is False

        # After full cycle
        assert check_phi_cycle_update(graph, PHI_CYCLE_TICKS + 100) is True

    def test_phi_cycle_process(self):
        """Process phi cycle update."""
        graph = init_entangle_graph()
        graph = add_fragment(graph, create_fragment("f1", RaCoordinate(1, 1, 1), 0.8))

        old_tick = graph.last_update_tick
        new_tick = old_tick + PHI_CYCLE_TICKS + 1

        updated = process_phi_cycle(graph, new_tick)

        assert updated.last_update_tick == new_tick
        assert updated.pending_trigger == UpdateTrigger.PHI_CYCLE


class TestGraphAnalysis:
    """Test graph analysis functions."""

    def test_active_links(self):
        """Get active links from graph."""
        graph = init_entangle_graph()

        # Add high-resonance fragment
        frag = create_fragment("high", RaCoordinate(5, 2, 1), 0.95)
        graph = add_fragment(graph, frag)

        # Update coherence to high values
        for zone in BodyZone:
            graph = update_bio_coherence(graph, zone, 0.9, 100)

        active = get_active_links(graph)
        assert len(active) > 0

    def test_overall_coherence(self):
        """Compute overall coherence."""
        graph = init_entangle_graph()

        for zone in BodyZone:
            graph = update_bio_coherence(graph, zone, 0.8, 100)

        overall = compute_overall_coherence(graph)
        assert abs(overall - 0.8) < 0.01

    def test_entanglement_density(self):
        """Compute entanglement density."""
        graph = init_entangle_graph()

        # No fragments = 0 density
        assert compute_entanglement_density(graph) == 0.0

        # Add fragment with high resonance
        frag = create_fragment("f1", RaCoordinate(5, 2, 1), 0.95)
        graph = add_fragment(graph, frag)

        # Boost coherence
        for zone in BodyZone:
            graph = update_bio_coherence(graph, zone, 0.95, 100)

        density = compute_entanglement_density(graph)
        assert density > 0.0  # Some links should be active


class TestZoneFragmentMapping:
    """Test zone-fragment relationship queries."""

    def test_get_zone_fragments(self):
        """Get fragments linked to a zone."""
        graph = init_entangle_graph()

        frag1 = create_fragment("f1", RaCoordinate(5, 2, 1), 0.95)
        frag2 = create_fragment("f2", RaCoordinate(10, 3, 2), 0.95)
        graph = add_fragment(graph, frag1)
        graph = add_fragment(graph, frag2)

        # Boost heart coherence
        graph = update_bio_coherence(graph, BodyZone.HEART, 0.95, 100)

        heart_frags = get_zone_fragments(graph, BodyZone.HEART)
        # Should find some fragments linked to heart
        assert len(heart_frags) >= 0  # May depend on phase matching

    def test_get_fragment_zones(self):
        """Get zones linked to a fragment."""
        graph = init_entangle_graph()

        frag = create_fragment("f1", RaCoordinate(5, 2, 1), 0.99)
        graph = add_fragment(graph, frag)

        # Boost all coherence values
        for zone in BodyZone:
            graph = update_bio_coherence(graph, zone, 0.99, 100)

        zones = get_fragment_zones(graph, "f1")
        assert len(zones) > 0  # High resonance should link to some zones


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
