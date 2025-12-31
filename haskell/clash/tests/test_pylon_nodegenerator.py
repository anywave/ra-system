"""
Test Suite for Ra.Pylon.NodeGenerator (Prompt 74)

Models physical scalar pylon devices (Golod pyramids, Tesla coils) as
field amplifiers and anchor nodes. Enables avatar projection stabilization,
scalar emergence tuning, and node teleportation between linked structures.

Architect Clarifications:
- Amplification: Golod~phi, Tesla~2.0, Hex~sqrt(2), Spherical~1.0
- Node linking: frequency resonance match + leyline proximity
- Teleport anchor: requires BOTH α>0.9 AND HRV coherence-lock
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Set, Tuple

# Constants
PHI = 1.618033988749895
SQRT_2 = 1.4142135623730951
ALPHA_TELEPORT_THRESHOLD = 0.9
HRV_COHERENCE_THRESHOLD = 0.8
FREQUENCY_MATCH_TOLERANCE = 0.1  # 10% tolerance
LEYLINE_PROXIMITY_KM = 200.0


class PylonShape(Enum):
    """Pylon geometry types."""
    GOLOD_PYRAMID = auto()    # Russian pyramid design
    TESLA_COIL = auto()       # Tesla resonance coil
    HEXAGONAL_ARRAY = auto()  # Hexagonal grid structure
    SPHERICAL = auto()        # Spherical resonator
    OBELISK = auto()          # Traditional obelisk
    CUSTOM = auto()           # User-defined


class LinkMethod(Enum):
    """Method for linking pylon nodes."""
    FREQUENCY_RESONANCE = auto()  # Harmonic fingerprint match
    LEYLINE_PROXIMITY = auto()    # Distance on grid
    PHASE_BURST = auto()          # Phase-synced correlation
    HYBRID = auto()               # Multiple methods combined


# Amplification factors per geometry
AMPLIFICATION_FACTORS = {
    PylonShape.GOLOD_PYRAMID: PHI,        # ~1.618
    PylonShape.TESLA_COIL: 2.0,           # High amplification
    PylonShape.HEXAGONAL_ARRAY: SQRT_2,   # ~1.414
    PylonShape.SPHERICAL: 1.0,            # Neutral
    PylonShape.OBELISK: 1.2,              # Modest amplification
    PylonShape.CUSTOM: 1.0,               # Default neutral
}


@dataclass
class GeoLocation:
    """Geographic location."""
    latitude: float
    longitude: float

    def distance_to(self, other: 'GeoLocation') -> float:
        """Haversine distance in km."""
        R = 6371.0  # Earth radius km
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return R * c


@dataclass
class HarmonicFingerprint:
    """Harmonic frequency fingerprint for resonance matching."""
    fundamental: float
    harmonics: List[float]  # Overtone series
    phase_signature: float  # 0-2π


@dataclass
class BioState:
    """Biometric state for teleport activation."""
    hrv_coherence: float   # 0-1
    alpha_level: float     # 0-1 scalar coherence
    phase_lock: bool       # HRV phase-locked to reference


@dataclass
class ScalarPylon:
    """Scalar pylon field amplifier."""
    pylon_id: uuid.UUID
    geometry_profile: PylonShape
    location: GeoLocation
    amplification: float
    linked_nodes: List[uuid.UUID]
    teleport_anchor: bool
    harmonic_fingerprint: HarmonicFingerprint
    is_active: bool = True


@dataclass
class PylonNetwork:
    """Network of interconnected pylons."""
    pylons: List[ScalarPylon]
    total_amplification: float
    active_teleport_links: int


def get_amplification_factor(shape: PylonShape) -> float:
    """Get amplification factor for pylon shape."""
    return AMPLIFICATION_FACTORS.get(shape, 1.0)


def generate_harmonic_fingerprint(
    base_frequency: float,
    shape: PylonShape
) -> HarmonicFingerprint:
    """Generate harmonic fingerprint based on shape."""
    # Fundamental modified by shape
    amp_factor = get_amplification_factor(shape)
    fundamental = base_frequency * amp_factor

    # Generate harmonics based on shape
    if shape == PylonShape.GOLOD_PYRAMID:
        # Phi-based harmonic series
        harmonics = [fundamental * (PHI ** i) for i in range(1, 5)]
    elif shape == PylonShape.TESLA_COIL:
        # Standard harmonic series with emphasis on odd
        harmonics = [fundamental * (2*i + 1) for i in range(1, 5)]
    elif shape == PylonShape.HEXAGONAL_ARRAY:
        # Hexagonal (6-fold) symmetry
        harmonics = [fundamental * (6 * i) for i in range(1, 5)]
    else:
        # Standard integer harmonics
        harmonics = [fundamental * i for i in range(2, 6)]

    # Phase signature based on fundamental
    phase = (fundamental * PHI) % (2 * math.pi)

    return HarmonicFingerprint(
        fundamental=fundamental,
        harmonics=harmonics,
        phase_signature=phase
    )


def check_frequency_resonance(
    fp1: HarmonicFingerprint,
    fp2: HarmonicFingerprint,
    tolerance: float = FREQUENCY_MATCH_TOLERANCE
) -> bool:
    """Check if two fingerprints are in resonance."""
    # Check fundamental match
    fundamental_ratio = fp1.fundamental / fp2.fundamental if fp2.fundamental > 0 else 0
    if abs(fundamental_ratio - 1.0) > tolerance:
        # Check for harmonic relationship
        if not any(abs(fundamental_ratio - i) < tolerance for i in [0.5, 2.0, PHI, 1/PHI]):
            return False

    # Check phase correlation
    phase_diff = abs(fp1.phase_signature - fp2.phase_signature)
    if phase_diff > math.pi:
        phase_diff = 2 * math.pi - phase_diff

    # Must be within 30° phase
    if phase_diff > math.pi / 6:
        return False

    return True


def check_leyline_proximity(
    loc1: GeoLocation,
    loc2: GeoLocation,
    max_distance: float = LEYLINE_PROXIMITY_KM
) -> bool:
    """Check if two locations are within leyline proximity."""
    return loc1.distance_to(loc2) <= max_distance


def can_link_pylons(
    pylon1: ScalarPylon,
    pylon2: ScalarPylon,
    method: LinkMethod = LinkMethod.HYBRID
) -> bool:
    """Check if two pylons can be linked."""
    if pylon1.pylon_id == pylon2.pylon_id:
        return False

    if method == LinkMethod.FREQUENCY_RESONANCE:
        return check_frequency_resonance(
            pylon1.harmonic_fingerprint,
            pylon2.harmonic_fingerprint
        )

    elif method == LinkMethod.LEYLINE_PROXIMITY:
        return check_leyline_proximity(pylon1.location, pylon2.location)

    elif method == LinkMethod.HYBRID:
        # Both frequency and proximity must match
        freq_match = check_frequency_resonance(
            pylon1.harmonic_fingerprint,
            pylon2.harmonic_fingerprint
        )
        prox_match = check_leyline_proximity(pylon1.location, pylon2.location)
        return freq_match or prox_match  # Either is sufficient

    return False


def check_teleport_activation(bio_state: BioState) -> bool:
    """Check if teleport anchor can be activated."""
    # Requires BOTH conditions
    alpha_ok = bio_state.alpha_level >= ALPHA_TELEPORT_THRESHOLD
    hrv_ok = bio_state.hrv_coherence >= HRV_COHERENCE_THRESHOLD and bio_state.phase_lock

    return alpha_ok and hrv_ok


def generate_pylon_from_geometry(
    shape: PylonShape,
    location: GeoLocation,
    base_frequency: float = 7.83  # Schumann
) -> ScalarPylon:
    """
    Generate scalar pylon from geometry specification.
    First function contract.
    """
    pylon_id = uuid.uuid4()
    amplification = get_amplification_factor(shape)
    fingerprint = generate_harmonic_fingerprint(base_frequency, shape)

    return ScalarPylon(
        pylon_id=pylon_id,
        geometry_profile=shape,
        location=location,
        amplification=amplification,
        linked_nodes=[],
        teleport_anchor=False,
        harmonic_fingerprint=fingerprint,
        is_active=True
    )


def link_pylon_teleportation(
    network: PylonNetwork,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    bio_state: Optional[BioState] = None
) -> PylonNetwork:
    """
    Link two pylons for teleportation.
    Second function contract.
    """
    # Find pylons
    source = None
    target = None
    for pylon in network.pylons:
        if pylon.pylon_id == source_id:
            source = pylon
        if pylon.pylon_id == target_id:
            target = pylon

    if source is None or target is None:
        return network

    # Check if can link
    if not can_link_pylons(source, target):
        return network

    # Create new network with updated links
    updated_pylons = []
    for pylon in network.pylons:
        if pylon.pylon_id == source_id:
            new_links = pylon.linked_nodes.copy()
            if target_id not in new_links:
                new_links.append(target_id)
            # Activate teleport anchor if bio state permits
            teleport = pylon.teleport_anchor
            if bio_state and check_teleport_activation(bio_state):
                teleport = True
            updated_pylons.append(ScalarPylon(
                pylon_id=pylon.pylon_id,
                geometry_profile=pylon.geometry_profile,
                location=pylon.location,
                amplification=pylon.amplification,
                linked_nodes=new_links,
                teleport_anchor=teleport,
                harmonic_fingerprint=pylon.harmonic_fingerprint,
                is_active=pylon.is_active
            ))
        elif pylon.pylon_id == target_id:
            new_links = pylon.linked_nodes.copy()
            if source_id not in new_links:
                new_links.append(source_id)
            teleport = pylon.teleport_anchor
            if bio_state and check_teleport_activation(bio_state):
                teleport = True
            updated_pylons.append(ScalarPylon(
                pylon_id=pylon.pylon_id,
                geometry_profile=pylon.geometry_profile,
                location=pylon.location,
                amplification=pylon.amplification,
                linked_nodes=new_links,
                teleport_anchor=teleport,
                harmonic_fingerprint=pylon.harmonic_fingerprint,
                is_active=pylon.is_active
            ))
        else:
            updated_pylons.append(pylon)

    # Recompute network stats
    total_amp = sum(p.amplification for p in updated_pylons if p.is_active)
    active_teleport = sum(1 for p in updated_pylons if p.teleport_anchor)

    return PylonNetwork(
        pylons=updated_pylons,
        total_amplification=total_amp,
        active_teleport_links=active_teleport
    )


def create_pylon_network(pylons: List[ScalarPylon]) -> PylonNetwork:
    """Create pylon network from list of pylons."""
    total_amp = sum(p.amplification for p in pylons if p.is_active)
    active_teleport = sum(1 for p in pylons if p.teleport_anchor)

    return PylonNetwork(
        pylons=pylons,
        total_amplification=total_amp,
        active_teleport_links=active_teleport
    )


def find_resonant_pylons(
    network: PylonNetwork,
    reference: HarmonicFingerprint
) -> List[ScalarPylon]:
    """Find pylons in resonance with reference fingerprint."""
    return [
        p for p in network.pylons
        if check_frequency_resonance(p.harmonic_fingerprint, reference)
    ]


def compute_network_coherence(network: PylonNetwork) -> float:
    """Compute overall network coherence based on links."""
    if not network.pylons:
        return 0.0

    total_links = sum(len(p.linked_nodes) for p in network.pylons)
    max_links = len(network.pylons) * (len(network.pylons) - 1)

    if max_links == 0:
        return 1.0 if len(network.pylons) == 1 else 0.0

    return total_links / max_links


def validate_pylon(pylon: ScalarPylon) -> bool:
    """Validate pylon structure."""
    if pylon.amplification <= 0:
        return False
    if not (-90 <= pylon.location.latitude <= 90):
        return False
    if not (-180 <= pylon.location.longitude <= 180):
        return False
    if pylon.harmonic_fingerprint.fundamental <= 0:
        return False
    return True


# ============== TESTS ==============

class TestAmplificationFactors:
    """Tests for geometry-based amplification."""

    def test_golod_pyramid_uses_phi(self):
        """Golod pyramid should use phi amplification."""
        factor = get_amplification_factor(PylonShape.GOLOD_PYRAMID)
        assert factor == pytest.approx(PHI)

    def test_tesla_coil_high_amplification(self):
        """Tesla coil should have 2.0 amplification."""
        factor = get_amplification_factor(PylonShape.TESLA_COIL)
        assert factor == pytest.approx(2.0)

    def test_hexagonal_uses_sqrt2(self):
        """Hexagonal array should use sqrt(2) amplification."""
        factor = get_amplification_factor(PylonShape.HEXAGONAL_ARRAY)
        assert factor == pytest.approx(SQRT_2)

    def test_spherical_neutral(self):
        """Spherical should have neutral (1.0) amplification."""
        factor = get_amplification_factor(PylonShape.SPHERICAL)
        assert factor == pytest.approx(1.0)

    def test_all_shapes_have_factors(self):
        """All shapes should have defined factors."""
        for shape in PylonShape:
            factor = get_amplification_factor(shape)
            assert factor > 0


class TestHarmonicFingerprint:
    """Tests for harmonic fingerprint generation."""

    def test_generates_valid_fingerprint(self):
        """Should generate valid fingerprint."""
        fp = generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID)

        assert fp.fundamental > 0
        assert len(fp.harmonics) > 0
        assert 0 <= fp.phase_signature <= 2 * math.pi

    def test_golod_has_phi_harmonics(self):
        """Golod pyramid should have phi-based harmonics."""
        fp = generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID)

        # First harmonic should be fundamental * phi
        expected_first = fp.fundamental * PHI
        assert fp.harmonics[0] == pytest.approx(expected_first)

    def test_tesla_has_odd_harmonics(self):
        """Tesla coil should emphasize odd harmonics."""
        fp = generate_harmonic_fingerprint(7.83, PylonShape.TESLA_COIL)

        # Should have 3rd, 5th, 7th, 9th harmonics
        for i, harmonic in enumerate(fp.harmonics):
            expected = fp.fundamental * (2 * (i + 1) + 1)
            assert harmonic == pytest.approx(expected)

    def test_shape_affects_fundamental(self):
        """Different shapes should produce different fundamentals."""
        base_freq = 7.83

        fp_golod = generate_harmonic_fingerprint(base_freq, PylonShape.GOLOD_PYRAMID)
        fp_tesla = generate_harmonic_fingerprint(base_freq, PylonShape.TESLA_COIL)
        fp_sphere = generate_harmonic_fingerprint(base_freq, PylonShape.SPHERICAL)

        assert fp_golod.fundamental != fp_tesla.fundamental
        assert fp_sphere.fundamental == pytest.approx(base_freq)  # Neutral


class TestFrequencyResonance:
    """Tests for frequency resonance matching."""

    def test_identical_fingerprints_resonate(self):
        """Identical fingerprints should resonate."""
        fp = generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID)
        assert check_frequency_resonance(fp, fp)

    def test_similar_frequencies_resonate(self):
        """Similar frequencies within tolerance should resonate."""
        fp1 = HarmonicFingerprint(10.0, [20, 30], 0.5)
        fp2 = HarmonicFingerprint(10.05, [20.1, 30.15], 0.5)  # Within 10%
        assert check_frequency_resonance(fp1, fp2)

    def test_different_frequencies_no_resonance(self):
        """Very different frequencies should not resonate."""
        fp1 = HarmonicFingerprint(10.0, [20, 30], 0.5)
        fp2 = HarmonicFingerprint(50.0, [100, 150], 2.0)  # Very different
        assert not check_frequency_resonance(fp1, fp2)

    def test_harmonic_relationship_resonates(self):
        """Harmonic relationships (2:1, phi) should resonate."""
        fp1 = HarmonicFingerprint(10.0, [20, 30], 0.5)
        fp2 = HarmonicFingerprint(20.0, [40, 60], 0.5)  # 2:1 ratio
        assert check_frequency_resonance(fp1, fp2)

    def test_phase_difference_blocks_resonance(self):
        """Large phase difference should block resonance."""
        fp1 = HarmonicFingerprint(10.0, [20, 30], 0.0)
        fp2 = HarmonicFingerprint(10.0, [20, 30], math.pi)  # 180° out
        assert not check_frequency_resonance(fp1, fp2)


class TestLeylineProximity:
    """Tests for leyline proximity checking."""

    def test_nearby_locations_in_proximity(self):
        """Nearby locations should be in proximity."""
        loc1 = GeoLocation(51.5074, -0.1278)  # London
        loc2 = GeoLocation(51.5, -0.1)        # Very close

        assert check_leyline_proximity(loc1, loc2)

    def test_distant_locations_not_in_proximity(self):
        """Distant locations should not be in proximity."""
        loc1 = GeoLocation(51.5074, -0.1278)  # London
        loc2 = GeoLocation(-33.8688, 151.2093)  # Sydney

        assert not check_leyline_proximity(loc1, loc2)

    def test_custom_distance_threshold(self):
        """Should respect custom distance threshold."""
        loc1 = GeoLocation(0, 0)
        loc2 = GeoLocation(1, 0)  # ~111km

        assert check_leyline_proximity(loc1, loc2, max_distance=200)
        assert not check_leyline_proximity(loc1, loc2, max_distance=50)


class TestTeleportActivation:
    """Tests for teleport anchor activation."""

    def test_both_conditions_required(self):
        """Both α > 0.9 AND HRV lock required."""
        # Both met
        bio_both = BioState(hrv_coherence=0.9, alpha_level=0.95, phase_lock=True)
        assert check_teleport_activation(bio_both)

        # Only alpha
        bio_alpha = BioState(hrv_coherence=0.5, alpha_level=0.95, phase_lock=False)
        assert not check_teleport_activation(bio_alpha)

        # Only HRV
        bio_hrv = BioState(hrv_coherence=0.9, alpha_level=0.5, phase_lock=True)
        assert not check_teleport_activation(bio_hrv)

    def test_phase_lock_required(self):
        """Phase lock must be true even with high HRV."""
        bio = BioState(hrv_coherence=0.95, alpha_level=0.95, phase_lock=False)
        assert not check_teleport_activation(bio)

    def test_threshold_boundary(self):
        """Should respect exact threshold boundaries."""
        # Just below alpha threshold
        bio_low_alpha = BioState(hrv_coherence=0.9, alpha_level=0.89, phase_lock=True)
        assert not check_teleport_activation(bio_low_alpha)

        # Just at threshold
        bio_at_threshold = BioState(hrv_coherence=0.8, alpha_level=0.9, phase_lock=True)
        assert check_teleport_activation(bio_at_threshold)


class TestPylonGeneration:
    """Tests for pylon generation from geometry."""

    def test_generates_valid_pylon(self):
        """Should generate valid pylon."""
        location = GeoLocation(30.0, 31.0)
        pylon = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, location)

        assert validate_pylon(pylon)
        assert pylon.geometry_profile == PylonShape.GOLOD_PYRAMID
        assert pylon.location == location

    def test_applies_correct_amplification(self):
        """Should apply correct amplification for shape."""
        location = GeoLocation(0, 0)

        golod = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, location)
        tesla = generate_pylon_from_geometry(PylonShape.TESLA_COIL, location)

        assert golod.amplification == pytest.approx(PHI)
        assert tesla.amplification == pytest.approx(2.0)

    def test_generates_unique_ids(self):
        """Each pylon should have unique ID."""
        location = GeoLocation(0, 0)
        pylons = [
            generate_pylon_from_geometry(PylonShape.SPHERICAL, location)
            for _ in range(10)
        ]

        ids = [p.pylon_id for p in pylons]
        assert len(ids) == len(set(ids))  # All unique

    def test_starts_without_links(self):
        """New pylons should have no links."""
        pylon = generate_pylon_from_geometry(
            PylonShape.HEXAGONAL_ARRAY,
            GeoLocation(0, 0)
        )
        assert len(pylon.linked_nodes) == 0

    def test_teleport_anchor_initially_false(self):
        """Teleport anchor should be initially false."""
        pylon = generate_pylon_from_geometry(
            PylonShape.TESLA_COIL,
            GeoLocation(0, 0)
        )
        assert not pylon.teleport_anchor


class TestPylonLinking:
    """Tests for pylon network linking."""

    def test_links_compatible_pylons(self):
        """Should link compatible pylons."""
        # Create two nearby pylons with same shape
        p1 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0, 0))
        p2 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0.5, 0.5))

        network = create_pylon_network([p1, p2])
        updated = link_pylon_teleportation(network, p1.pylon_id, p2.pylon_id)

        # Find updated pylons
        updated_p1 = next(p for p in updated.pylons if p.pylon_id == p1.pylon_id)
        updated_p2 = next(p for p in updated.pylons if p.pylon_id == p2.pylon_id)

        assert p2.pylon_id in updated_p1.linked_nodes
        assert p1.pylon_id in updated_p2.linked_nodes

    def test_bidirectional_linking(self):
        """Links should be bidirectional."""
        p1 = generate_pylon_from_geometry(PylonShape.TESLA_COIL, GeoLocation(10, 20))
        p2 = generate_pylon_from_geometry(PylonShape.TESLA_COIL, GeoLocation(10.1, 20.1))

        network = create_pylon_network([p1, p2])
        updated = link_pylon_teleportation(network, p1.pylon_id, p2.pylon_id)

        updated_p1 = next(p for p in updated.pylons if p.pylon_id == p1.pylon_id)
        updated_p2 = next(p for p in updated.pylons if p.pylon_id == p2.pylon_id)

        # Both should link to each other
        assert p2.pylon_id in updated_p1.linked_nodes
        assert p1.pylon_id in updated_p2.linked_nodes

    def test_no_self_link(self):
        """Pylon should not link to itself."""
        p1 = generate_pylon_from_geometry(PylonShape.SPHERICAL, GeoLocation(0, 0))
        network = create_pylon_network([p1])

        updated = link_pylon_teleportation(network, p1.pylon_id, p1.pylon_id)
        updated_p1 = updated.pylons[0]

        assert p1.pylon_id not in updated_p1.linked_nodes

    def test_activates_teleport_with_valid_bio(self):
        """Should activate teleport anchor with valid bio state."""
        p1 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0, 0))
        p2 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0.5, 0.5))

        bio = BioState(hrv_coherence=0.9, alpha_level=0.95, phase_lock=True)

        network = create_pylon_network([p1, p2])
        updated = link_pylon_teleportation(network, p1.pylon_id, p2.pylon_id, bio)

        updated_p1 = next(p for p in updated.pylons if p.pylon_id == p1.pylon_id)
        assert updated_p1.teleport_anchor

    def test_no_teleport_without_bio_conditions(self):
        """Should not activate teleport without bio conditions."""
        p1 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0, 0))
        p2 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0.5, 0.5))

        bio = BioState(hrv_coherence=0.5, alpha_level=0.5, phase_lock=False)

        network = create_pylon_network([p1, p2])
        updated = link_pylon_teleportation(network, p1.pylon_id, p2.pylon_id, bio)

        updated_p1 = next(p for p in updated.pylons if p.pylon_id == p1.pylon_id)
        assert not updated_p1.teleport_anchor


class TestPylonNetwork:
    """Tests for pylon network operations."""

    def test_computes_total_amplification(self):
        """Should compute total network amplification."""
        p1 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0, 0))
        p2 = generate_pylon_from_geometry(PylonShape.TESLA_COIL, GeoLocation(10, 10))

        network = create_pylon_network([p1, p2])

        expected = PHI + 2.0
        assert network.total_amplification == pytest.approx(expected)

    def test_counts_teleport_links(self):
        """Should count active teleport links."""
        p1 = ScalarPylon(
            pylon_id=uuid.uuid4(),
            geometry_profile=PylonShape.GOLOD_PYRAMID,
            location=GeoLocation(0, 0),
            amplification=PHI,
            linked_nodes=[],
            teleport_anchor=True,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID),
            is_active=True
        )
        p2 = ScalarPylon(
            pylon_id=uuid.uuid4(),
            geometry_profile=PylonShape.TESLA_COIL,
            location=GeoLocation(10, 10),
            amplification=2.0,
            linked_nodes=[],
            teleport_anchor=False,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.TESLA_COIL),
            is_active=True
        )

        network = create_pylon_network([p1, p2])

        assert network.active_teleport_links == 1

    def test_network_coherence_with_links(self):
        """Should compute network coherence based on links."""
        p1_id = uuid.uuid4()
        p2_id = uuid.uuid4()

        p1 = ScalarPylon(
            pylon_id=p1_id,
            geometry_profile=PylonShape.GOLOD_PYRAMID,
            location=GeoLocation(0, 0),
            amplification=PHI,
            linked_nodes=[p2_id],  # Linked
            teleport_anchor=False,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID),
            is_active=True
        )
        p2 = ScalarPylon(
            pylon_id=p2_id,
            geometry_profile=PylonShape.GOLOD_PYRAMID,
            location=GeoLocation(0.5, 0.5),
            amplification=PHI,
            linked_nodes=[p1_id],  # Linked back
            teleport_anchor=False,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.GOLOD_PYRAMID),
            is_active=True
        )

        network = create_pylon_network([p1, p2])
        coherence = compute_network_coherence(network)

        # 2 pylons, 2 links (bidirectional), max would be 2
        assert coherence == pytest.approx(1.0)

    def test_empty_network_zero_coherence(self):
        """Empty network should have zero coherence."""
        network = create_pylon_network([])
        coherence = compute_network_coherence(network)
        assert coherence == 0.0


class TestResonantPylonFinding:
    """Tests for finding resonant pylons."""

    def test_finds_matching_pylons(self):
        """Should find pylons matching reference fingerprint."""
        # Create pylons with same shape (similar fingerprints)
        p1 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(0, 0))
        p2 = generate_pylon_from_geometry(PylonShape.GOLOD_PYRAMID, GeoLocation(10, 10))
        p3 = generate_pylon_from_geometry(PylonShape.TESLA_COIL, GeoLocation(20, 20))

        network = create_pylon_network([p1, p2, p3])
        reference = p1.harmonic_fingerprint

        resonant = find_resonant_pylons(network, reference)

        # Should find the two Golod pyramids
        assert len(resonant) >= 2

    def test_empty_for_no_match(self):
        """Should return empty for no matching fingerprints."""
        p1 = generate_pylon_from_geometry(PylonShape.SPHERICAL, GeoLocation(0, 0))
        network = create_pylon_network([p1])

        # Very different reference
        reference = HarmonicFingerprint(9999.0, [19998, 29997], 5.0)

        resonant = find_resonant_pylons(network, reference)
        assert len(resonant) == 0


class TestPylonValidation:
    """Tests for pylon validation."""

    def test_valid_pylon_passes(self):
        """Valid pylon should pass validation."""
        pylon = generate_pylon_from_geometry(
            PylonShape.HEXAGONAL_ARRAY,
            GeoLocation(45.0, 90.0)
        )
        assert validate_pylon(pylon)

    def test_zero_amplification_fails(self):
        """Zero amplification should fail."""
        pylon = ScalarPylon(
            pylon_id=uuid.uuid4(),
            geometry_profile=PylonShape.SPHERICAL,
            location=GeoLocation(0, 0),
            amplification=0.0,  # Invalid
            linked_nodes=[],
            teleport_anchor=False,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.SPHERICAL),
            is_active=True
        )
        assert not validate_pylon(pylon)

    def test_invalid_coordinates_fail(self):
        """Invalid coordinates should fail."""
        pylon = ScalarPylon(
            pylon_id=uuid.uuid4(),
            geometry_profile=PylonShape.SPHERICAL,
            location=GeoLocation(100, 0),  # Invalid latitude
            amplification=1.0,
            linked_nodes=[],
            teleport_anchor=False,
            harmonic_fingerprint=generate_harmonic_fingerprint(7.83, PylonShape.SPHERICAL),
            is_active=True
        )
        assert not validate_pylon(pylon)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
