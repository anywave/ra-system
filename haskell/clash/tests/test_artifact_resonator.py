"""
Test Suite for Ra.Fragment.ArtifactResonator (P78)
Physical object memory fragment system.

Tests material-based scalar signatures, coherence-match activation,
non-user fragment emission, and grid/envelope tethering.
"""

import pytest
import math
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# Constants
PHI = 1.618033988749895
ACTIVATION_TOLERANCE = 0.15     # 15% coherence match tolerance
DECAY_RATE_PER_CYCLE = 0.05     # 5% decay per field cycle
MUTATION_THRESHOLD = 0.3        # Misalignment threshold for mutation


class MaterialType(Enum):
    """Physical material types."""
    QUARTZ = auto()
    OBSIDIAN = auto()
    GRANITE = auto()
    COPPER = auto()
    GOLD = auto()
    SILVER = auto()
    IRON = auto()
    LIMESTONE = auto()


class FragmentOrigin(Enum):
    """Origin type for fragments."""
    USER = auto()
    ARTIFACT = auto()
    GRID_NODE = auto()
    CONTACT = auto()


class TetherType(Enum):
    """Types of artifact tethering."""
    NONE = auto()
    EARTH_NODE = auto()         # Leyline tethering
    CONTACT_ENVELOPE = auto()   # ET contact tethering
    AVATAR_LINEAGE = auto()     # User lineage binding


# Material → Frequency mapping (static table)
MATERIAL_FREQUENCIES: Dict[MaterialType, float] = {
    MaterialType.QUARTZ: 32768.0,      # High resonance crystal
    MaterialType.OBSIDIAN: 8192.0,     # Volcanic glass
    MaterialType.GRANITE: 4096.0,      # Earth stone
    MaterialType.COPPER: 16384.0,      # Conductive metal
    MaterialType.GOLD: 24576.0,        # Noble metal
    MaterialType.SILVER: 20480.0,      # Reflective metal
    MaterialType.IRON: 12288.0,        # Magnetic metal
    MaterialType.LIMESTONE: 2048.0,    # Sedimentary
}

# Material → Alpha affinity
MATERIAL_ALPHA_AFFINITY: Dict[MaterialType, float] = {
    MaterialType.QUARTZ: 0.9,
    MaterialType.OBSIDIAN: 0.7,
    MaterialType.GRANITE: 0.5,
    MaterialType.COPPER: 0.8,
    MaterialType.GOLD: 0.95,
    MaterialType.SILVER: 0.85,
    MaterialType.IRON: 0.6,
    MaterialType.LIMESTONE: 0.4,
}


@dataclass
class ScalarSignature:
    """Material-based scalar signature."""
    base_frequency: float       # Hz
    harmonic_series: List[float]
    alpha_affinity: float       # 0.0-1.0
    conductivity: float         # 0.0-1.0


@dataclass
class ScalarField:
    """Current scalar field state."""
    alpha: float
    frequency: float
    phase: float


@dataclass
class FragmentNode:
    """Memory fragment within artifact."""
    fragment_id: str
    origin: FragmentOrigin
    coherence: float           # 0.0-1.0
    payload_hash: int          # Content identifier
    decay_cycles: int          # Cycles until decay
    mutated: bool


@dataclass
class LeylineAnchor:
    """Leyline anchor point for tethering."""
    node_id: str
    latitude: float
    longitude: float
    phase_lock: float          # 0.0-1.0


@dataclass
class ContactEnvelopeRef:
    """Reference to a contact envelope for tethering."""
    envelope_id: str
    modulation_lock: float     # 0.0-1.0


@dataclass
class ArtifactResonator:
    """Physical object with embedded memory fragments."""
    artifact_id: str
    material_type: MaterialType
    material_profile: ScalarSignature
    embedded_fragments: List[FragmentNode]
    activation_window: Tuple[float, float]  # (min_alpha, max_alpha)
    currently_active: bool
    tether_type: TetherType
    tether_data: Optional[dict] = None


def compute_scalar_signature(material: MaterialType) -> ScalarSignature:
    """Compute scalar signature from material type (static frequency table)."""
    base_freq = MATERIAL_FREQUENCIES.get(material, 1000.0)
    alpha_aff = MATERIAL_ALPHA_AFFINITY.get(material, 0.5)

    # Generate harmonic series (powers of φ)
    harmonics = [base_freq * (PHI ** i) for i in range(5)]

    # Conductivity based on material
    conductivity = {
        MaterialType.GOLD: 0.95,
        MaterialType.SILVER: 0.9,
        MaterialType.COPPER: 0.85,
        MaterialType.IRON: 0.5,
        MaterialType.QUARTZ: 0.3,      # Piezoelectric, not conductive
        MaterialType.OBSIDIAN: 0.1,
        MaterialType.GRANITE: 0.05,
        MaterialType.LIMESTONE: 0.02,
    }.get(material, 0.1)

    return ScalarSignature(
        base_frequency=base_freq,
        harmonic_series=harmonics,
        alpha_affinity=alpha_aff,
        conductivity=conductivity
    )


def compute_activation_window(signature: ScalarSignature) -> Tuple[float, float]:
    """Compute activation alpha window from material signature."""
    center = signature.alpha_affinity
    # Window width based on conductivity
    width = 0.1 + signature.conductivity * 0.2

    min_alpha = max(0.0, center - width / 2)
    max_alpha = min(1.0, center + width / 2)

    return (min_alpha, max_alpha)


def check_coherence_match(
    field: ScalarField,
    artifact: ArtifactResonator
) -> bool:
    """Check if field coherence matches artifact activation window."""
    min_alpha, max_alpha = artifact.activation_window
    return min_alpha <= field.alpha <= max_alpha


def check_frequency_resonance(
    field: ScalarField,
    signature: ScalarSignature
) -> bool:
    """Check if field frequency resonates with material."""
    # Check if field frequency is near any harmonic
    for harmonic in signature.harmonic_series:
        ratio = field.frequency / harmonic if harmonic > 0 else 0
        # Accept if ratio is near a simple fraction
        if abs(ratio - 1.0) < ACTIVATION_TOLERANCE:
            return True
        if abs(ratio - PHI) < ACTIVATION_TOLERANCE:
            return True
        if abs(ratio - 0.5) < ACTIVATION_TOLERANCE:
            return True
        if abs(ratio - 2.0) < ACTIVATION_TOLERANCE:
            return True
    return False


def create_artifact_resonator(
    material: MaterialType,
    tether_type: TetherType = TetherType.NONE,
    tether_data: Optional[dict] = None
) -> ArtifactResonator:
    """Create a new artifact resonator from material."""
    artifact_id = str(uuid.uuid4())[:8]
    signature = compute_scalar_signature(material)
    window = compute_activation_window(signature)

    return ArtifactResonator(
        artifact_id=artifact_id,
        material_type=material,
        material_profile=signature,
        embedded_fragments=[],
        activation_window=window,
        currently_active=False,
        tether_type=tether_type,
        tether_data=tether_data
    )


def embed_fragment_in_artifact(
    artifact: ArtifactResonator,
    fragment: FragmentNode
) -> ArtifactResonator:
    """Embed a fragment into an artifact."""
    # Mark fragment as artifact origin
    new_fragment = FragmentNode(
        fragment_id=fragment.fragment_id,
        origin=FragmentOrigin.ARTIFACT,
        coherence=fragment.coherence,
        payload_hash=fragment.payload_hash,
        decay_cycles=fragment.decay_cycles,
        mutated=fragment.mutated
    )

    new_fragments = artifact.embedded_fragments + [new_fragment]

    return ArtifactResonator(
        artifact_id=artifact.artifact_id,
        material_type=artifact.material_type,
        material_profile=artifact.material_profile,
        embedded_fragments=new_fragments,
        activation_window=artifact.activation_window,
        currently_active=artifact.currently_active,
        tether_type=artifact.tether_type,
        tether_data=artifact.tether_data
    )


def scan_artifact_proximity(
    field: ScalarField,
    artifacts: List[ArtifactResonator]
) -> List[FragmentNode]:
    """
    Scan artifacts in field proximity and emit activated fragments.
    Fragments are tagged as non-user origin.
    """
    emitted_fragments = []

    for artifact in artifacts:
        # Check activation conditions
        coherence_match = check_coherence_match(field, artifact)
        freq_resonance = check_frequency_resonance(field, artifact.material_profile)

        if coherence_match and freq_resonance:
            # Emit all embedded fragments
            for fragment in artifact.embedded_fragments:
                if fragment.coherence > 0 and fragment.decay_cycles > 0:
                    emitted_fragments.append(fragment)

    return emitted_fragments


def apply_field_decay(
    artifact: ArtifactResonator,
    cycles: int = 1
) -> ArtifactResonator:
    """Apply field cycle decay to embedded fragments."""
    new_fragments = []

    for fragment in artifact.embedded_fragments:
        new_decay = max(0, fragment.decay_cycles - cycles)
        new_coherence = fragment.coherence * (1.0 - DECAY_RATE_PER_CYCLE * cycles)
        new_coherence = max(0.0, new_coherence)

        new_fragment = FragmentNode(
            fragment_id=fragment.fragment_id,
            origin=fragment.origin,
            coherence=new_coherence,
            payload_hash=fragment.payload_hash,
            decay_cycles=new_decay,
            mutated=fragment.mutated
        )
        new_fragments.append(new_fragment)

    return ArtifactResonator(
        artifact_id=artifact.artifact_id,
        material_type=artifact.material_type,
        material_profile=artifact.material_profile,
        embedded_fragments=new_fragments,
        activation_window=artifact.activation_window,
        currently_active=artifact.currently_active,
        tether_type=artifact.tether_type,
        tether_data=artifact.tether_data
    )


def check_field_alignment(field: ScalarField, artifact: ArtifactResonator) -> float:
    """Check alignment between field and artifact signature."""
    alpha_diff = abs(field.alpha - artifact.material_profile.alpha_affinity)
    return 1.0 - alpha_diff


def apply_misaligned_reactivation(
    artifact: ArtifactResonator,
    field: ScalarField
) -> ArtifactResonator:
    """Apply mutation to fragments on misaligned reactivation."""
    alignment = check_field_alignment(field, artifact)

    if alignment < MUTATION_THRESHOLD:
        # Mutate fragments
        new_fragments = []
        for fragment in artifact.embedded_fragments:
            mutated_fragment = FragmentNode(
                fragment_id=fragment.fragment_id,
                origin=fragment.origin,
                coherence=fragment.coherence * 0.8,  # Reduce coherence
                payload_hash=fragment.payload_hash ^ 0xFF,  # Corrupt hash
                decay_cycles=fragment.decay_cycles,
                mutated=True
            )
            new_fragments.append(mutated_fragment)

        return ArtifactResonator(
            artifact_id=artifact.artifact_id,
            material_type=artifact.material_type,
            material_profile=artifact.material_profile,
            embedded_fragments=new_fragments,
            activation_window=artifact.activation_window,
            currently_active=artifact.currently_active,
            tether_type=artifact.tether_type,
            tether_data=artifact.tether_data
        )

    return artifact


def tether_to_leyline(
    artifact: ArtifactResonator,
    anchor: LeylineAnchor
) -> ArtifactResonator:
    """Tether artifact to leyline anchor."""
    return ArtifactResonator(
        artifact_id=artifact.artifact_id,
        material_type=artifact.material_type,
        material_profile=artifact.material_profile,
        embedded_fragments=artifact.embedded_fragments,
        activation_window=artifact.activation_window,
        currently_active=artifact.currently_active,
        tether_type=TetherType.EARTH_NODE,
        tether_data={
            "node_id": anchor.node_id,
            "lat": anchor.latitude,
            "lon": anchor.longitude,
            "phase_lock": anchor.phase_lock
        }
    )


def tether_to_contact_envelope(
    artifact: ArtifactResonator,
    envelope_ref: ContactEnvelopeRef
) -> ArtifactResonator:
    """Tether artifact to contact envelope."""
    return ArtifactResonator(
        artifact_id=artifact.artifact_id,
        material_type=artifact.material_type,
        material_profile=artifact.material_profile,
        embedded_fragments=artifact.embedded_fragments,
        activation_window=artifact.activation_window,
        currently_active=artifact.currently_active,
        tether_type=TetherType.CONTACT_ENVELOPE,
        tether_data={
            "envelope_id": envelope_ref.envelope_id,
            "modulation_lock": envelope_ref.modulation_lock
        }
    )


def prune_decayed_fragments(artifact: ArtifactResonator) -> ArtifactResonator:
    """Remove fully decayed fragments from artifact."""
    active_fragments = [
        f for f in artifact.embedded_fragments
        if f.decay_cycles > 0 and f.coherence > 0.01
    ]

    return ArtifactResonator(
        artifact_id=artifact.artifact_id,
        material_type=artifact.material_type,
        material_profile=artifact.material_profile,
        embedded_fragments=active_fragments,
        activation_window=artifact.activation_window,
        currently_active=artifact.currently_active,
        tether_type=artifact.tether_type,
        tether_data=artifact.tether_data
    )


# ============== TEST CLASSES ==============

class TestMaterialSignatures:
    """Tests for material scalar signature computation."""

    def test_quartz_signature(self):
        """Test quartz has high frequency and affinity."""
        sig = compute_scalar_signature(MaterialType.QUARTZ)

        assert sig.base_frequency == 32768.0
        assert sig.alpha_affinity == 0.9
        assert len(sig.harmonic_series) == 5

    def test_granite_signature(self):
        """Test granite has lower frequency and affinity."""
        sig = compute_scalar_signature(MaterialType.GRANITE)

        assert sig.base_frequency == 4096.0
        assert sig.alpha_affinity == 0.5

    def test_gold_highest_affinity(self):
        """Test gold has highest alpha affinity."""
        gold_sig = compute_scalar_signature(MaterialType.GOLD)
        silver_sig = compute_scalar_signature(MaterialType.SILVER)

        assert gold_sig.alpha_affinity > silver_sig.alpha_affinity

    def test_harmonic_series_phi_based(self):
        """Test harmonic series uses φ scaling."""
        sig = compute_scalar_signature(MaterialType.COPPER)

        ratio = sig.harmonic_series[1] / sig.harmonic_series[0]
        assert ratio == pytest.approx(PHI, rel=0.001)

    def test_conductivity_metals_higher(self):
        """Test metals have higher conductivity."""
        gold = compute_scalar_signature(MaterialType.GOLD)
        granite = compute_scalar_signature(MaterialType.GRANITE)

        assert gold.conductivity > granite.conductivity

    def test_all_materials_have_signatures(self):
        """Test all material types have valid signatures."""
        for material in MaterialType:
            sig = compute_scalar_signature(material)
            assert sig.base_frequency > 0
            assert 0.0 <= sig.alpha_affinity <= 1.0
            assert len(sig.harmonic_series) > 0


class TestActivationWindow:
    """Tests for activation window computation."""

    def test_window_centered_on_affinity(self):
        """Test window is centered on material affinity."""
        sig = compute_scalar_signature(MaterialType.QUARTZ)
        window = compute_activation_window(sig)

        center = (window[0] + window[1]) / 2
        assert center == pytest.approx(sig.alpha_affinity, abs=0.1)

    def test_conductive_materials_wider_window(self):
        """Test conductive materials have wider activation window."""
        gold_sig = compute_scalar_signature(MaterialType.GOLD)
        granite_sig = compute_scalar_signature(MaterialType.GRANITE)

        gold_window = compute_activation_window(gold_sig)
        granite_window = compute_activation_window(granite_sig)

        gold_width = gold_window[1] - gold_window[0]
        granite_width = granite_window[1] - granite_window[0]

        assert gold_width > granite_width

    def test_window_clamped_to_valid_range(self):
        """Test window stays in 0.0-1.0 range."""
        for material in MaterialType:
            sig = compute_scalar_signature(material)
            window = compute_activation_window(sig)

            assert window[0] >= 0.0
            assert window[1] <= 1.0
            assert window[0] < window[1]


class TestCoherenceMatch:
    """Tests for coherence match detection."""

    def test_match_within_window(self):
        """Test field within window matches."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        field = ScalarField(alpha=0.85, frequency=32768.0, phase=0.0)

        assert check_coherence_match(field, artifact) is True

    def test_no_match_below_window(self):
        """Test field below window doesn't match."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        field = ScalarField(alpha=0.3, frequency=32768.0, phase=0.0)

        assert check_coherence_match(field, artifact) is False

    def test_no_match_above_window(self):
        """Test field above window doesn't match."""
        artifact = create_artifact_resonator(MaterialType.LIMESTONE)
        # Limestone has low affinity ~0.4
        field = ScalarField(alpha=0.95, frequency=2048.0, phase=0.0)

        assert check_coherence_match(field, artifact) is False


class TestFrequencyResonance:
    """Tests for frequency resonance detection."""

    def test_exact_frequency_resonates(self):
        """Test exact frequency match resonates."""
        sig = compute_scalar_signature(MaterialType.COPPER)
        field = ScalarField(alpha=0.8, frequency=sig.base_frequency, phase=0.0)

        assert check_frequency_resonance(field, sig) is True

    def test_phi_harmonic_resonates(self):
        """Test φ harmonic resonates."""
        sig = compute_scalar_signature(MaterialType.COPPER)
        field = ScalarField(alpha=0.8, frequency=sig.base_frequency * PHI, phase=0.0)

        assert check_frequency_resonance(field, sig) is True

    def test_octave_resonates(self):
        """Test octave (2x) resonates."""
        sig = compute_scalar_signature(MaterialType.COPPER)
        field = ScalarField(alpha=0.8, frequency=sig.base_frequency * 2, phase=0.0)

        assert check_frequency_resonance(field, sig) is True

    def test_non_harmonic_no_resonance(self):
        """Test non-harmonic frequency doesn't resonate."""
        sig = compute_scalar_signature(MaterialType.COPPER)
        # Use 0.3x which avoids 1.0, PHI, 0.5, and 2.0 ratios with all harmonics
        field = ScalarField(alpha=0.8, frequency=sig.base_frequency * 0.3, phase=0.0)

        assert check_frequency_resonance(field, sig) is False


class TestArtifactCreation:
    """Tests for artifact resonator creation."""

    def test_create_basic_artifact(self):
        """Test creating basic artifact."""
        artifact = create_artifact_resonator(MaterialType.OBSIDIAN)

        assert artifact.material_type == MaterialType.OBSIDIAN
        assert len(artifact.embedded_fragments) == 0
        assert artifact.currently_active is False
        assert artifact.tether_type == TetherType.NONE

    def test_create_with_tether(self):
        """Test creating artifact with tether."""
        artifact = create_artifact_resonator(
            MaterialType.QUARTZ,
            tether_type=TetherType.EARTH_NODE,
            tether_data={"node_id": "sedona_01"}
        )

        assert artifact.tether_type == TetherType.EARTH_NODE
        assert artifact.tether_data["node_id"] == "sedona_01"

    def test_unique_artifact_ids(self):
        """Test each artifact gets unique ID."""
        art1 = create_artifact_resonator(MaterialType.GOLD)
        art2 = create_artifact_resonator(MaterialType.GOLD)

        assert art1.artifact_id != art2.artifact_id


class TestFragmentEmbedding:
    """Tests for fragment embedding in artifacts."""

    def test_embed_single_fragment(self):
        """Test embedding single fragment."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        fragment = FragmentNode(
            fragment_id="frag_001",
            origin=FragmentOrigin.USER,
            coherence=0.8,
            payload_hash=12345,
            decay_cycles=100,
            mutated=False
        )

        new_artifact = embed_fragment_in_artifact(artifact, fragment)

        assert len(new_artifact.embedded_fragments) == 1
        assert new_artifact.embedded_fragments[0].origin == FragmentOrigin.ARTIFACT

    def test_embed_multiple_fragments(self):
        """Test embedding multiple fragments."""
        artifact = create_artifact_resonator(MaterialType.GOLD)

        for i in range(3):
            fragment = FragmentNode(
                fragment_id=f"frag_{i:03d}",
                origin=FragmentOrigin.USER,
                coherence=0.7 + i * 0.1,
                payload_hash=1000 + i,
                decay_cycles=50,
                mutated=False
            )
            artifact = embed_fragment_in_artifact(artifact, fragment)

        assert len(artifact.embedded_fragments) == 3

    def test_original_artifact_unchanged(self):
        """Test embedding doesn't modify original artifact."""
        original = create_artifact_resonator(MaterialType.SILVER)
        fragment = FragmentNode(
            fragment_id="test",
            origin=FragmentOrigin.USER,
            coherence=0.5,
            payload_hash=999,
            decay_cycles=10,
            mutated=False
        )

        new_artifact = embed_fragment_in_artifact(original, fragment)

        assert len(original.embedded_fragments) == 0
        assert len(new_artifact.embedded_fragments) == 1


class TestProximityScan:
    """Tests for artifact proximity scanning."""

    def test_scan_emits_fragments_on_match(self):
        """Test scan emits fragments when conditions match."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        fragment = FragmentNode(
            fragment_id="emit_test",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.9,
            payload_hash=7777,
            decay_cycles=50,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # Field matches quartz activation window
        field = ScalarField(alpha=0.88, frequency=32768.0, phase=0.0)

        emitted = scan_artifact_proximity(field, [artifact])

        assert len(emitted) == 1
        assert emitted[0].fragment_id == "emit_test"

    def test_scan_no_emission_on_mismatch(self):
        """Test scan doesn't emit on condition mismatch."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        fragment = FragmentNode(
            fragment_id="no_emit",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.9,
            payload_hash=8888,
            decay_cycles=50,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # Field doesn't match quartz (low alpha)
        field = ScalarField(alpha=0.3, frequency=32768.0, phase=0.0)

        emitted = scan_artifact_proximity(field, [artifact])

        assert len(emitted) == 0

    def test_scan_multiple_artifacts(self):
        """Test scanning multiple artifacts."""
        artifacts = []
        for i, material in enumerate([MaterialType.QUARTZ, MaterialType.GOLD]):
            art = create_artifact_resonator(material)
            frag = FragmentNode(
                fragment_id=f"multi_{i}",
                origin=FragmentOrigin.ARTIFACT,
                coherence=0.85,
                payload_hash=i * 1000,
                decay_cycles=50,
                mutated=False
            )
            art = embed_fragment_in_artifact(art, frag)
            artifacts.append(art)

        # Field that matches both high-affinity materials
        field = ScalarField(alpha=0.92, frequency=24576.0, phase=0.0)

        emitted = scan_artifact_proximity(field, artifacts)

        # Should get at least one emission
        assert len(emitted) >= 1

    def test_decayed_fragments_not_emitted(self):
        """Test decayed fragments aren't emitted."""
        artifact = create_artifact_resonator(MaterialType.COPPER)
        decayed_frag = FragmentNode(
            fragment_id="decayed",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.0,  # No coherence
            payload_hash=5555,
            decay_cycles=0,  # No cycles left
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, decayed_frag)

        field = ScalarField(alpha=0.8, frequency=16384.0, phase=0.0)

        emitted = scan_artifact_proximity(field, [artifact])

        assert len(emitted) == 0


class TestFragmentDecay:
    """Tests for fragment decay with field cycles."""

    def test_single_cycle_decay(self):
        """Test decay after single cycle."""
        artifact = create_artifact_resonator(MaterialType.IRON)
        fragment = FragmentNode(
            fragment_id="decay_test",
            origin=FragmentOrigin.ARTIFACT,
            coherence=1.0,
            payload_hash=4444,
            decay_cycles=100,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        decayed = apply_field_decay(artifact, 1)

        assert decayed.embedded_fragments[0].decay_cycles == 99
        assert decayed.embedded_fragments[0].coherence < 1.0

    def test_multiple_cycle_decay(self):
        """Test decay over multiple cycles."""
        artifact = create_artifact_resonator(MaterialType.SILVER)
        fragment = FragmentNode(
            fragment_id="multi_decay",
            origin=FragmentOrigin.ARTIFACT,
            coherence=1.0,
            payload_hash=3333,
            decay_cycles=50,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        decayed = apply_field_decay(artifact, 10)

        assert decayed.embedded_fragments[0].decay_cycles == 40
        expected_coherence = 1.0 * (1.0 - DECAY_RATE_PER_CYCLE * 10)
        assert decayed.embedded_fragments[0].coherence == pytest.approx(expected_coherence, abs=0.01)

    def test_decay_floors_at_zero(self):
        """Test decay doesn't go below zero."""
        artifact = create_artifact_resonator(MaterialType.LIMESTONE)
        fragment = FragmentNode(
            fragment_id="floor_test",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.1,
            payload_hash=2222,
            decay_cycles=5,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        decayed = apply_field_decay(artifact, 100)

        assert decayed.embedded_fragments[0].decay_cycles == 0
        assert decayed.embedded_fragments[0].coherence >= 0.0


class TestMisalignedReactivation:
    """Tests for fragment mutation on misaligned reactivation."""

    def test_mutation_on_severe_misalignment(self):
        """Test mutation occurs on severe field misalignment."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)  # High affinity
        fragment = FragmentNode(
            fragment_id="mutate_test",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.8,
            payload_hash=1111,
            decay_cycles=50,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # Very low alpha - severe misalignment with quartz
        field = ScalarField(alpha=0.1, frequency=32768.0, phase=0.0)

        mutated = apply_misaligned_reactivation(artifact, field)

        assert mutated.embedded_fragments[0].mutated is True
        assert mutated.embedded_fragments[0].coherence < 0.8

    def test_no_mutation_on_aligned_field(self):
        """Test no mutation when field is aligned."""
        artifact = create_artifact_resonator(MaterialType.QUARTZ)
        fragment = FragmentNode(
            fragment_id="no_mutate",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.8,
            payload_hash=9999,
            decay_cycles=50,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # High alpha - aligned with quartz
        field = ScalarField(alpha=0.9, frequency=32768.0, phase=0.0)

        result = apply_misaligned_reactivation(artifact, field)

        assert result.embedded_fragments[0].mutated is False


class TestTethering:
    """Tests for artifact tethering."""

    def test_tether_to_leyline(self):
        """Test tethering artifact to leyline."""
        artifact = create_artifact_resonator(MaterialType.GRANITE)
        anchor = LeylineAnchor(
            node_id="stonehenge_01",
            latitude=51.1789,
            longitude=-1.8262,
            phase_lock=0.85
        )

        tethered = tether_to_leyline(artifact, anchor)

        assert tethered.tether_type == TetherType.EARTH_NODE
        assert tethered.tether_data["node_id"] == "stonehenge_01"
        assert tethered.tether_data["lat"] == 51.1789

    def test_tether_to_contact_envelope(self):
        """Test tethering artifact to contact envelope."""
        artifact = create_artifact_resonator(MaterialType.GOLD)
        envelope = ContactEnvelopeRef(
            envelope_id="contact_777",
            modulation_lock=0.92
        )

        tethered = tether_to_contact_envelope(artifact, envelope)

        assert tethered.tether_type == TetherType.CONTACT_ENVELOPE
        assert tethered.tether_data["envelope_id"] == "contact_777"

    def test_tether_preserves_fragments(self):
        """Test tethering preserves embedded fragments."""
        artifact = create_artifact_resonator(MaterialType.OBSIDIAN)
        fragment = FragmentNode(
            fragment_id="preserve_test",
            origin=FragmentOrigin.ARTIFACT,
            coherence=0.7,
            payload_hash=6666,
            decay_cycles=30,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        anchor = LeylineAnchor("giza_01", 29.9792, 31.1342, 0.95)
        tethered = tether_to_leyline(artifact, anchor)

        assert len(tethered.embedded_fragments) == 1
        assert tethered.embedded_fragments[0].fragment_id == "preserve_test"


class TestFragmentPruning:
    """Tests for pruning decayed fragments."""

    def test_prune_fully_decayed(self):
        """Test pruning removes fully decayed fragments."""
        artifact = create_artifact_resonator(MaterialType.IRON)

        # Active fragment
        active = FragmentNode("active", FragmentOrigin.ARTIFACT, 0.8, 1000, 50, False)
        # Decayed fragment
        decayed = FragmentNode("decayed", FragmentOrigin.ARTIFACT, 0.005, 0, 0, False)

        artifact = embed_fragment_in_artifact(artifact, active)
        artifact = embed_fragment_in_artifact(artifact, decayed)

        pruned = prune_decayed_fragments(artifact)

        assert len(pruned.embedded_fragments) == 1
        assert pruned.embedded_fragments[0].fragment_id == "active"

    def test_prune_low_coherence(self):
        """Test pruning removes low coherence fragments."""
        artifact = create_artifact_resonator(MaterialType.COPPER)

        good = FragmentNode("good", FragmentOrigin.ARTIFACT, 0.5, 2000, 25, False)
        bad = FragmentNode("bad", FragmentOrigin.ARTIFACT, 0.005, 100, 10, False)

        artifact = embed_fragment_in_artifact(artifact, good)
        artifact = embed_fragment_in_artifact(artifact, bad)

        pruned = prune_decayed_fragments(artifact)

        assert len(pruned.embedded_fragments) == 1


class TestArtifactIntegration:
    """Integration tests for artifact system."""

    def test_full_artifact_lifecycle(self):
        """Test full artifact lifecycle: create → embed → activate → decay."""
        # Create artifact
        artifact = create_artifact_resonator(MaterialType.QUARTZ)

        # Embed fragment
        fragment = FragmentNode(
            fragment_id="lifecycle_test",
            origin=FragmentOrigin.USER,
            coherence=1.0,
            payload_hash=42424,
            decay_cycles=20,
            mutated=False
        )
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # Activate with matching field
        field = ScalarField(alpha=0.88, frequency=32768.0, phase=0.0)
        emitted = scan_artifact_proximity(field, [artifact])
        assert len(emitted) == 1

        # Apply decay
        artifact = apply_field_decay(artifact, 5)
        assert artifact.embedded_fragments[0].decay_cycles == 15

        # Continue decay until exhausted
        artifact = apply_field_decay(artifact, 20)
        artifact = prune_decayed_fragments(artifact)

        # Fragment should be pruned
        assert len(artifact.embedded_fragments) == 0

    def test_tethered_artifact_scanning(self):
        """Test tethered artifacts can still be scanned."""
        artifact = create_artifact_resonator(MaterialType.GOLD)
        fragment = FragmentNode("tether_scan", FragmentOrigin.ARTIFACT, 0.9, 8888, 100, False)
        artifact = embed_fragment_in_artifact(artifact, fragment)

        # Tether to leyline
        anchor = LeylineAnchor("sedona_vortex", 34.8697, -111.7609, 0.88)
        artifact = tether_to_leyline(artifact, anchor)

        # Should still emit when scanned
        field = ScalarField(alpha=0.93, frequency=24576.0, phase=0.0)
        emitted = scan_artifact_proximity(field, [artifact])

        assert len(emitted) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
