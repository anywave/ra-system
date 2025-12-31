"""
Test Harness for Prompt 60: Ra.Chamber.Synthesis

Tests multi-layer scalar chamber generator for fragment emergence,
avatar stability, and scalar-based experience rendering.

Based on:
- φ^n depth sequencing: 8 layers, baseDepth=0.618
- OmegaFormat mapping: l∈[0..7], m∈[-l..l]
- FragmentRequirement: fragId, minCoherence, preferredShells, torsionAllowed
- Rendering metaphors: pulse rings, nodal circles, spiral arcs, color swirl
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Chamber synthesis defaults
BASE_DEPTH = 0.618
MAX_CHAMBER_LAYERS = 8

# Coherence thresholds
MIN_COHERENCE_FOR_EMERGENCE = 0.65
DEFAULT_RESONANCE_PEAK = 0.72


# =============================================================================
# Data Structures
# =============================================================================

class TorsionState(Enum):
    """Torsion field state."""
    NORMAL = auto()
    INVERTED = auto()
    NULL = auto()


class CoherenceBand(Enum):
    """Coherence band classification."""
    DORMANT = auto()      # < 0.3
    AWAKENING = auto()    # 0.3 - 0.5
    ACTIVE = auto()       # 0.5 - 0.72
    RESONANT = auto()     # 0.72 - 0.9
    EMERGENCE = auto()    # > 0.9


class VisualizationType(Enum):
    """Types of visualization elements."""
    PULSE_RING = auto()       # Expanding/contracting rings
    NODAL_GEOMETRY = auto()   # Radial nodes (small circles)
    HARMONIC_LOOP = auto()    # Spiral arcs
    TORSION_SWIRL = auto()    # Color swirl direction
    SYMMETRY_RING = auto()    # Δ(ankh) balance intensity


@dataclass(frozen=True)
class OmegaFormat:
    """Spherical harmonic format (l, m)."""
    l: int  # 0-7
    m: int  # -l to +l

    def __post_init__(self):
        if not 0 <= self.l <= 7:
            raise ValueError(f"l must be in [0,7], got {self.l}")
        if not -self.l <= self.m <= self.l:
            raise ValueError(f"m must be in [-{self.l},{self.l}], got {self.m}")


@dataclass
class ScalarLayer:
    """A single layer in the scalar chamber."""
    depth: float
    amplitude: float
    phase: float
    omega: OmegaFormat
    layer_index: int


@dataclass
class TorsionSignature:
    """Torsion field signature."""
    state: TorsionState
    bias: float  # -1 to 1
    rotation_rate: float


@dataclass
class BiometricField:
    """User's biometric field input."""
    coherence: float
    hrv: float
    breath_rate: float
    phase_alignment: float


@dataclass
class FragmentRequirement:
    """Requirements for fragment placement in chamber."""
    frag_id: str
    min_coherence: float
    preferred_shells: List[OmegaFormat]
    torsion_allowed: List[TorsionState]


@dataclass
class VisualizationElement:
    """A single visualization element."""
    vis_type: VisualizationType
    position: Tuple[float, float, float]
    intensity: float
    phase: float
    layer_index: int


@dataclass
class ChamberVisualization:
    """Chamber visualization export."""
    elements: List[VisualizationElement]
    pulse_layers: List[float]  # Pulse radii
    nodal_positions: List[Tuple[float, float]]
    torsion_direction: int  # +1 CW, -1 CCW


@dataclass
class ScalarChamber:
    """Complete scalar chamber structure."""
    chamber_id: str
    depth_profile: List[ScalarLayer]
    coherence_zone: CoherenceBand
    resonance_peak: float
    torsion_tune: TorsionSignature
    harmonic_roots: List[OmegaFormat]
    safe_for: List[str]  # FragmentIDs


@dataclass
class SynthesisResult:
    """Result of chamber synthesis."""
    chamber: Optional[ScalarChamber]
    success: bool
    unplaceable_fragments: List[str]
    error_message: Optional[str] = None


# =============================================================================
# Core Functions
# =============================================================================

def compute_phi_depths(base_depth: float = BASE_DEPTH,
                       num_layers: int = MAX_CHAMBER_LAYERS) -> List[float]:
    """
    Compute φ^n depth progression for layers.

    Returns depths: [base, base*φ, base*φ², ...]
    """
    depths = []
    current = base_depth
    for i in range(num_layers):
        depths.append(current)
        current *= PHI
    return depths


def classify_coherence_band(coherence: float) -> CoherenceBand:
    """Classify coherence into a band."""
    if coherence >= 0.9:
        return CoherenceBand.EMERGENCE
    elif coherence >= 0.72:
        return CoherenceBand.RESONANT
    elif coherence >= 0.5:
        return CoherenceBand.ACTIVE
    elif coherence >= 0.3:
        return CoherenceBand.AWAKENING
    else:
        return CoherenceBand.DORMANT


def generate_harmonic_roots(num_roots: int = 5) -> List[OmegaFormat]:
    """
    Generate standard harmonic roots.

    Uses primary spherical harmonics as base selectors.
    """
    roots = []
    for l in range(min(num_roots, 8)):
        # Use m=0 for primary harmonics
        roots.append(OmegaFormat(l=l, m=0))
    return roots


def compute_layer_amplitude(layer_index: int, coherence: float,
                            torsion_bias: float) -> float:
    """
    Compute amplitude for a layer based on coherence and torsion.

    Inner layers (lower index) have higher base amplitude.
    """
    base_amp = 1.0 - (layer_index / MAX_CHAMBER_LAYERS) * 0.5

    # Coherence boost
    coherence_factor = 0.5 + 0.5 * coherence

    # Torsion modulation based on layer parity
    parity = 1 if layer_index % 2 == 0 else -1
    torsion_factor = 1.0 + torsion_bias * parity * 0.2

    return max(0.0, min(1.0, base_amp * coherence_factor * torsion_factor))


def compute_layer_phase(layer_index: int, bio_phase: float) -> float:
    """
    Compute phase for a layer.

    Uses φ-based phase offset per layer.
    """
    phase_offset = (layer_index * PHI * 0.5) % (2 * math.pi)
    return (bio_phase + phase_offset) % (2 * math.pi)


def select_omega_for_layer(layer_index: int,
                           fragment_prefs: List[OmegaFormat]) -> OmegaFormat:
    """
    Select OmegaFormat for a layer.

    Tries to match fragment preferences if available.
    """
    # Use layer index as l, m=0 as default
    if fragment_prefs:
        # Try to find a matching preference
        for omega in fragment_prefs:
            if omega.l == layer_index % 8:
                return omega
    return OmegaFormat(l=layer_index % 8, m=0)


def check_fragment_compatible(fragment: FragmentRequirement,
                              chamber: ScalarChamber) -> bool:
    """
    Check if a fragment is compatible with the chamber.

    Fragment is compatible if:
    1. Chamber coherence >= fragment min_coherence
    2. Chamber has at least one matching shell
    3. Torsion state is allowed
    """
    # Check coherence
    coherence = chamber.resonance_peak
    if coherence < fragment.min_coherence:
        return False

    # Check shells
    shell_match = False
    for omega in fragment.preferred_shells:
        if omega in chamber.harmonic_roots:
            shell_match = True
            break
    if not shell_match and fragment.preferred_shells:
        return False

    # Check torsion
    if chamber.torsion_tune.state not in fragment.torsion_allowed:
        return False

    return True


def create_torsion_signature(bio_field: BiometricField) -> TorsionSignature:
    """Create torsion signature from biometric field."""
    # Determine state based on coherence and phase
    if bio_field.coherence < 0.3:
        state = TorsionState.NULL
    elif bio_field.phase_alignment < 0.3:
        state = TorsionState.INVERTED
    else:
        state = TorsionState.NORMAL

    # Bias from HRV
    bias = (bio_field.hrv - 50) / 100  # Normalize around 50ms HRV
    bias = max(-1.0, min(1.0, bias))

    # Rotation rate from breath
    rotation_rate = bio_field.breath_rate / 12.0  # Normalize around 12 bpm

    return TorsionSignature(state=state, bias=bias, rotation_rate=rotation_rate)


def synthesize_chamber(
    user_id: str,
    bio_field: BiometricField,
    fragment_requirements: List[FragmentRequirement]
) -> SynthesisResult:
    """
    Synthesize a scalar chamber tailored to biometric field and fragment requirements.

    Returns chamber if successful, or list of unplaceable fragments.
    """
    chamber_id = str(uuid.uuid4())

    # Compute depth profile
    depths = compute_phi_depths()

    # Create torsion signature
    torsion = create_torsion_signature(bio_field)

    # Generate harmonic roots
    harmonic_roots = generate_harmonic_roots()

    # Collect preferred shells from fragments
    all_preferred = []
    for frag in fragment_requirements:
        all_preferred.extend(frag.preferred_shells)

    # Build layers
    layers = []
    for i, depth in enumerate(depths):
        omega = select_omega_for_layer(i, all_preferred)
        amplitude = compute_layer_amplitude(i, bio_field.coherence, torsion.bias)
        phase = compute_layer_phase(i, bio_field.phase_alignment * 2 * math.pi)

        layers.append(ScalarLayer(
            depth=depth,
            amplitude=amplitude,
            phase=phase,
            omega=omega,
            layer_index=i
        ))

    # Compute resonance peak
    # Based on average amplitude weighted by coherence
    avg_amplitude = sum(l.amplitude for l in layers) / len(layers)
    resonance_peak = bio_field.coherence * avg_amplitude

    # Classify coherence zone
    coherence_zone = classify_coherence_band(resonance_peak)

    # Create chamber
    chamber = ScalarChamber(
        chamber_id=chamber_id,
        depth_profile=layers,
        coherence_zone=coherence_zone,
        resonance_peak=resonance_peak,
        torsion_tune=torsion,
        harmonic_roots=harmonic_roots,
        safe_for=[]
    )

    # Check which fragments are compatible
    unplaceable = []
    for frag in fragment_requirements:
        if check_fragment_compatible(frag, chamber):
            chamber.safe_for.append(frag.frag_id)
        else:
            unplaceable.append(frag.frag_id)

    if unplaceable:
        return SynthesisResult(
            chamber=chamber,
            success=False,
            unplaceable_fragments=unplaceable,
            error_message=f"{len(unplaceable)} fragments cannot be placed"
        )

    return SynthesisResult(
        chamber=chamber,
        success=True,
        unplaceable_fragments=[]
    )


def export_chamber_visualization(chamber: ScalarChamber) -> ChamberVisualization:
    """
    Export chamber as visualization structure.

    Creates visualization elements for each layer.
    """
    elements = []

    for layer in chamber.depth_profile:
        # Pulse ring for each layer
        elements.append(VisualizationElement(
            vis_type=VisualizationType.PULSE_RING,
            position=(0, 0, layer.depth),
            intensity=layer.amplitude,
            phase=layer.phase,
            layer_index=layer.layer_index
        ))

        # Nodal geometry at l nodes
        for m in range(layer.omega.l + 1):
            angle = 2 * math.pi * m / (layer.omega.l + 1)
            elements.append(VisualizationElement(
                vis_type=VisualizationType.NODAL_GEOMETRY,
                position=(math.cos(angle) * layer.depth,
                         math.sin(angle) * layer.depth,
                         layer.depth),
                intensity=layer.amplitude * 0.5,
                phase=layer.phase,
                layer_index=layer.layer_index
            ))

    # Create pulse layers (radii)
    pulse_layers = [l.depth for l in chamber.depth_profile]

    # Create nodal positions (2D projection)
    nodal_positions = []
    for layer in chamber.depth_profile:
        for m in range(layer.omega.l + 1):
            angle = 2 * math.pi * m / (layer.omega.l + 1)
            nodal_positions.append((
                math.cos(angle) * layer.depth,
                math.sin(angle) * layer.depth
            ))

    # Torsion direction
    torsion_dir = 1 if chamber.torsion_tune.bias >= 0 else -1

    return ChamberVisualization(
        elements=elements,
        pulse_layers=pulse_layers,
        nodal_positions=nodal_positions,
        torsion_direction=torsion_dir
    )


# =============================================================================
# Test Cases
# =============================================================================

class TestPhiDepths:
    """Tests for φ^n depth computation."""

    def test_base_depth_first(self):
        """First depth should be base depth."""
        depths = compute_phi_depths(0.618, 8)
        assert depths[0] == pytest.approx(0.618)

    def test_phi_progression(self):
        """Each depth should be φ times previous."""
        depths = compute_phi_depths(0.618, 8)
        for i in range(1, len(depths)):
            ratio = depths[i] / depths[i-1]
            assert ratio == pytest.approx(PHI, rel=0.001)

    def test_layer_count(self):
        """Should generate correct number of layers."""
        depths = compute_phi_depths(0.618, 8)
        assert len(depths) == 8


class TestOmegaFormat:
    """Tests for OmegaFormat validation."""

    def test_valid_omega(self):
        """Valid omega should create successfully."""
        omega = OmegaFormat(l=3, m=2)
        assert omega.l == 3
        assert omega.m == 2

    def test_l_bounds(self):
        """l must be in [0, 7]."""
        with pytest.raises(ValueError):
            OmegaFormat(l=8, m=0)
        with pytest.raises(ValueError):
            OmegaFormat(l=-1, m=0)

    def test_m_bounds(self):
        """m must be in [-l, l]."""
        with pytest.raises(ValueError):
            OmegaFormat(l=2, m=3)
        with pytest.raises(ValueError):
            OmegaFormat(l=2, m=-3)

    def test_m_negative_valid(self):
        """Negative m within bounds should work."""
        omega = OmegaFormat(l=3, m=-2)
        assert omega.m == -2


class TestCoherenceBand:
    """Tests for coherence band classification."""

    def test_emergence_band(self):
        """High coherence should be EMERGENCE."""
        band = classify_coherence_band(0.95)
        assert band == CoherenceBand.EMERGENCE

    def test_resonant_band(self):
        """0.72-0.9 should be RESONANT."""
        band = classify_coherence_band(0.8)
        assert band == CoherenceBand.RESONANT

    def test_active_band(self):
        """0.5-0.72 should be ACTIVE."""
        band = classify_coherence_band(0.6)
        assert band == CoherenceBand.ACTIVE

    def test_dormant_band(self):
        """Low coherence should be DORMANT."""
        band = classify_coherence_band(0.2)
        assert band == CoherenceBand.DORMANT


class TestHarmonicRoots:
    """Tests for harmonic root generation."""

    def test_generates_correct_count(self):
        """Should generate requested number of roots."""
        roots = generate_harmonic_roots(5)
        assert len(roots) == 5

    def test_l_values_sequential(self):
        """l values should be sequential from 0."""
        roots = generate_harmonic_roots(5)
        for i, root in enumerate(roots):
            assert root.l == i

    def test_m_zero_default(self):
        """m should be 0 for default roots."""
        roots = generate_harmonic_roots(5)
        for root in roots:
            assert root.m == 0


class TestLayerAmplitude:
    """Tests for layer amplitude computation."""

    def test_inner_layers_higher(self):
        """Inner layers should have higher base amplitude."""
        amp0 = compute_layer_amplitude(0, 0.5, 0.0)
        amp7 = compute_layer_amplitude(7, 0.5, 0.0)
        assert amp0 > amp7

    def test_coherence_increases_amplitude(self):
        """Higher coherence should increase amplitude."""
        amp_low = compute_layer_amplitude(0, 0.3, 0.0)
        amp_high = compute_layer_amplitude(0, 0.9, 0.0)
        assert amp_high > amp_low

    def test_amplitude_bounded(self):
        """Amplitude should be in [0, 1]."""
        for i in range(8):
            for coh in [0.0, 0.5, 1.0]:
                for bias in [-1.0, 0.0, 1.0]:
                    amp = compute_layer_amplitude(i, coh, bias)
                    assert 0.0 <= amp <= 1.0


class TestTorsionSignature:
    """Tests for torsion signature creation."""

    def test_normal_state(self):
        """High coherence and alignment should give NORMAL."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.7)
        torsion = create_torsion_signature(bio)
        assert torsion.state == TorsionState.NORMAL

    def test_inverted_state(self):
        """Low phase alignment should give INVERTED."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.1)
        torsion = create_torsion_signature(bio)
        assert torsion.state == TorsionState.INVERTED

    def test_null_state(self):
        """Very low coherence should give NULL."""
        bio = BiometricField(coherence=0.2, hrv=50, breath_rate=12, phase_alignment=0.5)
        torsion = create_torsion_signature(bio)
        assert torsion.state == TorsionState.NULL


class TestFragmentCompatibility:
    """Tests for fragment compatibility checking."""

    def test_compatible_fragment(self):
        """Compatible fragment should pass all checks."""
        chamber = ScalarChamber(
            chamber_id="test",
            depth_profile=[],
            coherence_zone=CoherenceBand.RESONANT,
            resonance_peak=0.8,
            torsion_tune=TorsionSignature(TorsionState.NORMAL, 0.0, 1.0),
            harmonic_roots=[OmegaFormat(0, 0), OmegaFormat(1, 0)],
            safe_for=[]
        )

        fragment = FragmentRequirement(
            frag_id="frag1",
            min_coherence=0.6,
            preferred_shells=[OmegaFormat(1, 0)],
            torsion_allowed=[TorsionState.NORMAL, TorsionState.INVERTED]
        )

        assert check_fragment_compatible(fragment, chamber) is True

    def test_incompatible_coherence(self):
        """Fragment requiring higher coherence should fail."""
        chamber = ScalarChamber(
            chamber_id="test",
            depth_profile=[],
            coherence_zone=CoherenceBand.ACTIVE,
            resonance_peak=0.5,
            torsion_tune=TorsionSignature(TorsionState.NORMAL, 0.0, 1.0),
            harmonic_roots=[OmegaFormat(0, 0)],
            safe_for=[]
        )

        fragment = FragmentRequirement(
            frag_id="frag1",
            min_coherence=0.7,  # Higher than chamber
            preferred_shells=[],
            torsion_allowed=[TorsionState.NORMAL]
        )

        assert check_fragment_compatible(fragment, chamber) is False

    def test_incompatible_torsion(self):
        """Fragment not allowing chamber torsion should fail."""
        chamber = ScalarChamber(
            chamber_id="test",
            depth_profile=[],
            coherence_zone=CoherenceBand.RESONANT,
            resonance_peak=0.8,
            torsion_tune=TorsionSignature(TorsionState.INVERTED, 0.0, 1.0),
            harmonic_roots=[OmegaFormat(0, 0)],
            safe_for=[]
        )

        fragment = FragmentRequirement(
            frag_id="frag1",
            min_coherence=0.6,
            preferred_shells=[],
            torsion_allowed=[TorsionState.NORMAL]  # Doesn't allow INVERTED
        )

        assert check_fragment_compatible(fragment, chamber) is False


class TestSynthesizeChamber:
    """Tests for chamber synthesis."""

    def test_synthesis_success(self):
        """Should synthesize chamber successfully."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.7)

        result = synthesize_chamber("user1", bio, [])

        assert result.success is True
        assert result.chamber is not None
        assert len(result.chamber.depth_profile) == MAX_CHAMBER_LAYERS

    def test_synthesis_with_fragment(self):
        """Should mark compatible fragments as safe."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.7)

        fragment = FragmentRequirement(
            frag_id="frag1",
            min_coherence=0.5,
            preferred_shells=[OmegaFormat(0, 0)],
            torsion_allowed=[TorsionState.NORMAL]
        )

        result = synthesize_chamber("user1", bio, [fragment])

        assert "frag1" in result.chamber.safe_for

    def test_synthesis_unplaceable_fragment(self):
        """Should report unplaceable fragments."""
        bio = BiometricField(coherence=0.5, hrv=50, breath_rate=12, phase_alignment=0.7)

        fragment = FragmentRequirement(
            frag_id="frag1",
            min_coherence=0.9,  # Too high
            preferred_shells=[],
            torsion_allowed=[TorsionState.NORMAL]
        )

        result = synthesize_chamber("user1", bio, [fragment])

        assert result.success is False
        assert "frag1" in result.unplaceable_fragments


class TestChamberVisualization:
    """Tests for chamber visualization export."""

    def test_export_creates_elements(self):
        """Should create visualization elements."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.7)
        result = synthesize_chamber("user1", bio, [])

        vis = export_chamber_visualization(result.chamber)

        assert len(vis.elements) > 0

    def test_pulse_layers_match_depths(self):
        """Pulse layers should match depth profile."""
        bio = BiometricField(coherence=0.8, hrv=50, breath_rate=12, phase_alignment=0.7)
        result = synthesize_chamber("user1", bio, [])

        vis = export_chamber_visualization(result.chamber)

        assert len(vis.pulse_layers) == len(result.chamber.depth_profile)

    def test_torsion_direction(self):
        """Torsion direction should reflect bias."""
        bio_pos = BiometricField(coherence=0.8, hrv=70, breath_rate=12, phase_alignment=0.7)
        bio_neg = BiometricField(coherence=0.8, hrv=30, breath_rate=12, phase_alignment=0.7)

        result_pos = synthesize_chamber("user1", bio_pos, [])
        result_neg = synthesize_chamber("user1", bio_neg, [])

        vis_pos = export_chamber_visualization(result_pos.chamber)
        vis_neg = export_chamber_visualization(result_neg.chamber)

        assert vis_pos.torsion_direction == 1
        assert vis_neg.torsion_direction == -1


class TestIntegrationScenarios:
    """End-to-end integration tests."""

    def test_full_synthesis_workflow(self):
        """Test complete chamber synthesis workflow."""
        # Create biometric field
        bio = BiometricField(
            coherence=0.85,
            hrv=55,
            breath_rate=10,
            phase_alignment=0.75
        )

        # Define fragment requirements
        fragments = [
            FragmentRequirement(
                frag_id="avatar_core",
                min_coherence=0.6,
                preferred_shells=[OmegaFormat(0, 0), OmegaFormat(1, 0)],
                torsion_allowed=[TorsionState.NORMAL, TorsionState.NULL]
            ),
            FragmentRequirement(
                frag_id="memory_shard",
                min_coherence=0.5,
                preferred_shells=[OmegaFormat(2, 0)],
                torsion_allowed=[TorsionState.NORMAL]
            )
        ]

        # Synthesize chamber
        result = synthesize_chamber("user123", bio, fragments)

        assert result.success is True
        assert result.chamber is not None
        assert len(result.chamber.safe_for) == 2
        # Resonance peak is coherence * avg_amplitude, which may be ACTIVE or higher
        assert result.chamber.coherence_zone in [CoherenceBand.ACTIVE, CoherenceBand.RESONANT]

        # Export visualization
        vis = export_chamber_visualization(result.chamber)
        assert len(vis.elements) > 0
        assert len(vis.pulse_layers) == MAX_CHAMBER_LAYERS

    def test_multiple_fragment_placement(self):
        """Test chamber with multiple fragments, some placeable."""
        bio = BiometricField(coherence=0.7, hrv=50, breath_rate=12, phase_alignment=0.6)

        fragments = [
            FragmentRequirement(
                frag_id="easy",
                min_coherence=0.4,
                preferred_shells=[],
                torsion_allowed=[TorsionState.NORMAL, TorsionState.INVERTED, TorsionState.NULL]
            ),
            FragmentRequirement(
                frag_id="hard",
                min_coherence=0.95,  # Too high
                preferred_shells=[],
                torsion_allowed=[TorsionState.NORMAL]
            )
        ]

        result = synthesize_chamber("user1", bio, fragments)

        assert "easy" in result.chamber.safe_for
        assert "hard" in result.unplaceable_fragments


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
