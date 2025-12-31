"""
Test Suite for Ra.Healing.CellularCoilMatrix (Prompt 70)

Coil arrangements for localized cellular healing based on harmonic field data.
References HUBBARD_COIL_GENERATOR.md and ELECTROCULTURE_PARAMETERS.md.

Architect Clarifications:
- Coil tuning from α: α<0.3→Wide spacing, α>0.7→Dense (linear interpolation)
- Mirror coil sync: real-time biometric (HRV latency) or quantum sync (zero-latency)
- Tissue targeting: static templates with dynamic override from BioField.Signature
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple, Dict

# Constants
PHI = 1.618033988749895
ALPHA_WIDE_THRESHOLD = 0.3    # Below this: wide, diffuse spacing
ALPHA_DENSE_THRESHOLD = 0.7   # Above this: dense, focused configuration


class SyncMethod(Enum):
    BIOMETRIC_FEEDBACK = auto()  # Real-time HRV latency cycles
    QUANTUM_SYNC = auto()        # Zero-latency entangled pairing


class PhaseEnvelope(Enum):
    CONSTANT = auto()
    RISING = auto()
    FALLING = auto()
    OSCILLATING = auto()
    PULSED = auto()


class TissueType(Enum):
    HEART = auto()
    LIVER = auto()
    KIDNEY = auto()
    BRAIN = auto()
    LUNGS = auto()
    MUSCLE = auto()
    BONE = auto()
    SKIN = auto()
    BLOOD = auto()
    NERVE = auto()


# Static tissue templates (from anatomical models)
TISSUE_TEMPLATES: Dict[TissueType, Dict] = {
    TissueType.HEART: {
        "base_turns": 12,
        "spacing_factor": PHI,
        "orientation": "toroidal",
        "resonance_freq": 7.83,
    },
    TissueType.LIVER: {
        "base_turns": 8,
        "spacing_factor": 1.2,
        "orientation": "planar",
        "resonance_freq": 40.0,
    },
    TissueType.BRAIN: {
        "base_turns": 21,  # Fibonacci
        "spacing_factor": PHI,
        "orientation": "spherical",
        "resonance_freq": 10.0,
    },
    TissueType.KIDNEY: {
        "base_turns": 8,
        "spacing_factor": 1.0,
        "orientation": "paired",
        "resonance_freq": 20.0,
    },
    TissueType.LUNGS: {
        "base_turns": 13,  # Fibonacci
        "spacing_factor": 1.414,  # sqrt(2)
        "orientation": "bilateral",
        "resonance_freq": 72.0,
    },
    TissueType.MUSCLE: {
        "base_turns": 5,
        "spacing_factor": 1.0,
        "orientation": "linear",
        "resonance_freq": 50.0,
    },
    TissueType.BONE: {
        "base_turns": 3,
        "spacing_factor": 0.8,
        "orientation": "spiral",
        "resonance_freq": 25.0,
    },
    TissueType.SKIN: {
        "base_turns": 8,
        "spacing_factor": 1.1,
        "orientation": "planar",
        "resonance_freq": 30.0,
    },
    TissueType.BLOOD: {
        "base_turns": 13,
        "spacing_factor": PHI,
        "orientation": "vortex",
        "resonance_freq": 1550.0,
    },
    TissueType.NERVE: {
        "base_turns": 21,
        "spacing_factor": PHI,
        "orientation": "axial",
        "resonance_freq": 100.0,
    },
}


@dataclass
class RaCoordinate:
    """3D coordinate in Ra field."""
    x: float
    y: float
    z: float


@dataclass
class GradientVector:
    """Field gradient vector."""
    dx: float
    dy: float
    dz: float


@dataclass
class ScalarFieldPoint:
    """Point in scalar field with alpha value."""
    coordinate: RaCoordinate
    alpha: float  # 0-1 coherence
    gradient: GradientVector


@dataclass
class ScalarField:
    """Scalar field representation."""
    points: List[ScalarFieldPoint]
    mean_alpha: float
    max_gradient: float


@dataclass
class BioFieldSignature:
    """User-specific biofield signature for dynamic targeting."""
    user_id: uuid.UUID
    tissue_resonances: Dict[TissueType, float]  # Tissue -> resonance modifier
    coherence_map: Dict[TissueType, float]      # Tissue -> local coherence
    active_zones: List[TissueType]


@dataclass
class BioState:
    """Real-time biometric state."""
    hrv: float
    pulse: float
    coherence: float
    hrv_latency_ms: float  # For biometric sync


@dataclass
class CoilMatrix:
    """Coil matrix for cellular healing."""
    coil_id: uuid.UUID
    harmonic_layout: List[List[float]]  # 2D grid of turns/spacing
    target_tissue: Optional[TissueType]
    entangled_pair: Optional['CoilMatrix']
    modulation_phase: PhaseEnvelope
    stabilization_gain: float
    sync_method: SyncMethod
    spacing_density: float  # 0-1, computed from alpha


def compute_spacing_from_alpha(alpha: float) -> float:
    """
    Compute coil spacing density from scalar alpha.
    α < 0.3 → Wide (low density ~0.2)
    α > 0.7 → Dense (high density ~1.0)
    Linear interpolation between.
    """
    if alpha <= ALPHA_WIDE_THRESHOLD:
        return 0.2  # Wide spacing
    elif alpha >= ALPHA_DENSE_THRESHOLD:
        return 1.0  # Dense configuration
    else:
        # Linear interpolation
        t = (alpha - ALPHA_WIDE_THRESHOLD) / (ALPHA_DENSE_THRESHOLD - ALPHA_WIDE_THRESHOLD)
        return 0.2 + (t * 0.8)


def compute_turn_density(base_turns: int, spacing_density: float) -> int:
    """Compute actual turn count based on spacing density."""
    # Higher density = more turns
    multiplier = 0.5 + (spacing_density * 1.0)  # 0.5x to 1.5x
    return max(1, int(base_turns * multiplier))


def generate_harmonic_layout(
    turns: int,
    spacing_factor: float,
    orientation: str,
    spacing_density: float
) -> List[List[float]]:
    """
    Generate 2D harmonic layout grid for coil matrix.
    Values represent turn spacing at each grid position.
    """
    # Grid size based on turns
    grid_size = max(3, int(math.sqrt(turns)) + 1)

    layout = []
    for i in range(grid_size):
        row = []
        for j in range(grid_size):
            # Base spacing
            base_spacing = spacing_factor / (1.0 + spacing_density)

            # Orientation-specific modulation
            if orientation == "toroidal":
                # Radial falloff from center
                dist = math.sqrt((i - grid_size/2)**2 + (j - grid_size/2)**2)
                mod = 1.0 / (1.0 + dist * 0.1)
            elif orientation == "spherical":
                # Uniform with phi modulation
                mod = 1.0 + 0.1 * math.sin(i * PHI) * math.cos(j * PHI)
            elif orientation == "planar":
                # Flat, uniform
                mod = 1.0
            elif orientation == "bilateral":
                # Mirror symmetry
                mod = 1.0 + 0.05 * abs(i - grid_size/2)
            elif orientation == "vortex":
                # Spiral pattern
                angle = math.atan2(j - grid_size/2, i - grid_size/2)
                mod = 1.0 + 0.1 * math.sin(angle * 3)
            else:
                mod = 1.0

            row.append(base_spacing * mod)
        layout.append(row)

    return layout


def select_phase_envelope(coherence: float, tissue: TissueType) -> PhaseEnvelope:
    """Select modulation phase envelope based on coherence and tissue."""
    # High coherence -> constant
    if coherence > 0.8:
        return PhaseEnvelope.CONSTANT

    # Specific tissue patterns
    if tissue in [TissueType.HEART, TissueType.BLOOD]:
        return PhaseEnvelope.PULSED
    elif tissue == TissueType.BRAIN:
        return PhaseEnvelope.OSCILLATING
    elif tissue in [TissueType.MUSCLE, TissueType.BONE]:
        return PhaseEnvelope.RISING

    # Default based on coherence
    if coherence < 0.4:
        return PhaseEnvelope.RISING
    return PhaseEnvelope.OSCILLATING


def compute_stabilization_gain(
    bio_state: BioState,
    spacing_density: float
) -> float:
    """Compute stabilization gain from biometrics and density."""
    # Base gain from HRV (higher HRV = better stability = higher gain)
    hrv_factor = 0.5 + (bio_state.hrv * 0.5)

    # Density affects gain (denser = more focused = higher gain)
    density_factor = 0.8 + (spacing_density * 0.4)

    # Coherence boost
    coherence_factor = 0.7 + (bio_state.coherence * 0.6)

    return hrv_factor * density_factor * coherence_factor


def select_sync_method(bio_state: BioState, has_entangled_pair: bool) -> SyncMethod:
    """Select synchronization method based on state and entanglement."""
    if not has_entangled_pair:
        return SyncMethod.BIOMETRIC_FEEDBACK

    # Quantum sync for high coherence + low latency
    if bio_state.coherence > 0.7 and bio_state.hrv_latency_ms < 50:
        return SyncMethod.QUANTUM_SYNC

    return SyncMethod.BIOMETRIC_FEEDBACK


def get_tissue_template(tissue: TissueType) -> Dict:
    """Get static template for tissue type."""
    return TISSUE_TEMPLATES.get(tissue, TISSUE_TEMPLATES[TissueType.MUSCLE])


def apply_biofield_override(
    template: Dict,
    signature: Optional[BioFieldSignature],
    tissue: TissueType
) -> Dict:
    """Apply dynamic override from biofield signature to template."""
    if signature is None:
        return template

    modified = template.copy()

    # Apply resonance modifier
    if tissue in signature.tissue_resonances:
        modifier = signature.tissue_resonances[tissue]
        modified["resonance_freq"] = template["resonance_freq"] * modifier

    # Apply coherence-based spacing adjustment
    if tissue in signature.coherence_map:
        local_coherence = signature.coherence_map[tissue]
        modified["spacing_factor"] = template["spacing_factor"] * (0.8 + local_coherence * 0.4)

    return modified


def generate_coil_matrix(
    scalar_field: ScalarField,
    bio_state: BioState,
    target_tissue: Optional[TissueType] = None,
    biofield_signature: Optional[BioFieldSignature] = None,
    create_entangled_pair: bool = False
) -> CoilMatrix:
    """
    Generate coil matrix from scalar field and biometric state.

    Uses α and field gradients to modulate coil turn density and spatial harmonics.
    Biometric sync enables mirror coil entanglement.
    """
    coil_id = uuid.uuid4()

    # Compute spacing from mean alpha
    spacing_density = compute_spacing_from_alpha(scalar_field.mean_alpha)

    # Get tissue template (static or dynamically overridden)
    if target_tissue:
        template = get_tissue_template(target_tissue)
        template = apply_biofield_override(template, biofield_signature, target_tissue)
    else:
        template = {
            "base_turns": 8,
            "spacing_factor": 1.0,
            "orientation": "planar",
            "resonance_freq": 10.0,
        }

    # Compute turn density
    turns = compute_turn_density(template["base_turns"], spacing_density)

    # Generate harmonic layout
    layout = generate_harmonic_layout(
        turns,
        template["spacing_factor"],
        template["orientation"],
        spacing_density
    )

    # Select phase envelope
    phase = select_phase_envelope(bio_state.coherence, target_tissue or TissueType.MUSCLE)

    # Compute stabilization gain
    gain = compute_stabilization_gain(bio_state, spacing_density)

    # Determine sync method
    sync_method = select_sync_method(bio_state, create_entangled_pair)

    # Create main coil
    main_coil = CoilMatrix(
        coil_id=coil_id,
        harmonic_layout=layout,
        target_tissue=target_tissue,
        entangled_pair=None,
        modulation_phase=phase,
        stabilization_gain=gain,
        sync_method=sync_method,
        spacing_density=spacing_density
    )

    # Create entangled pair if requested
    if create_entangled_pair:
        entangled = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=layout,  # Same layout
            target_tissue=target_tissue,
            entangled_pair=None,  # Will be linked
            modulation_phase=phase,
            stabilization_gain=gain,
            sync_method=sync_method,
            spacing_density=spacing_density
        )
        main_coil.entangled_pair = entangled
        entangled.entangled_pair = main_coil

    return main_coil


def create_mirror_stabilization_pulse(
    source_coil: CoilMatrix,
    bio_state: BioState
) -> Dict:
    """Create stabilization pulse to send to mirror coil."""
    pulse = {
        "source_id": source_coil.coil_id,
        "gain": source_coil.stabilization_gain,
        "phase": source_coil.modulation_phase,
        "timestamp_offset": 0.0,
    }

    if source_coil.sync_method == SyncMethod.BIOMETRIC_FEEDBACK:
        # Add HRV latency delay
        pulse["timestamp_offset"] = bio_state.hrv_latency_ms / 1000.0
    # Quantum sync: zero offset

    return pulse


def validate_coil_alignment(coil: CoilMatrix, target_zone: RaCoordinate) -> bool:
    """Validate that coil matrix aligns with healing vector zone."""
    # Check that layout is non-empty
    if not coil.harmonic_layout or not coil.harmonic_layout[0]:
        return False

    # Check spacing density is reasonable
    if coil.spacing_density < 0.1 or coil.spacing_density > 1.0:
        return False

    # Check gain is positive
    if coil.stabilization_gain <= 0:
        return False

    return True


# ============== TESTS ==============

class TestSpacingFromAlpha:
    """Tests for α-based coil spacing computation."""

    def test_low_alpha_gives_wide_spacing(self):
        """α < 0.3 should give wide (low density) spacing."""
        spacing = compute_spacing_from_alpha(0.2)
        assert spacing == pytest.approx(0.2)

    def test_high_alpha_gives_dense_spacing(self):
        """α > 0.7 should give dense (high density) spacing."""
        spacing = compute_spacing_from_alpha(0.8)
        assert spacing == pytest.approx(1.0)

    def test_threshold_alpha_030(self):
        """α = 0.3 should be at wide threshold."""
        spacing = compute_spacing_from_alpha(0.3)
        assert spacing == pytest.approx(0.2)

    def test_threshold_alpha_070(self):
        """α = 0.7 should be at dense threshold."""
        spacing = compute_spacing_from_alpha(0.7)
        assert spacing == pytest.approx(1.0)

    def test_midpoint_interpolation(self):
        """α = 0.5 should give interpolated spacing."""
        spacing = compute_spacing_from_alpha(0.5)
        # t = (0.5 - 0.3) / (0.7 - 0.3) = 0.5
        # spacing = 0.2 + (0.5 * 0.8) = 0.6
        assert spacing == pytest.approx(0.6)

    def test_linear_interpolation(self):
        """Interpolation should be linear between thresholds."""
        spacing_40 = compute_spacing_from_alpha(0.4)
        spacing_50 = compute_spacing_from_alpha(0.5)
        spacing_60 = compute_spacing_from_alpha(0.6)

        # Check linear progression
        delta1 = spacing_50 - spacing_40
        delta2 = spacing_60 - spacing_50
        assert delta1 == pytest.approx(delta2, rel=0.01)

    def test_extreme_low_alpha(self):
        """α = 0 should give minimum spacing."""
        spacing = compute_spacing_from_alpha(0.0)
        assert spacing == pytest.approx(0.2)

    def test_extreme_high_alpha(self):
        """α = 1 should give maximum spacing."""
        spacing = compute_spacing_from_alpha(1.0)
        assert spacing == pytest.approx(1.0)


class TestTurnDensity:
    """Tests for turn count computation."""

    def test_low_density_reduces_turns(self):
        """Low density should reduce turn count."""
        turns = compute_turn_density(10, 0.2)
        assert turns < 10

    def test_high_density_increases_turns(self):
        """High density should increase turn count."""
        turns = compute_turn_density(10, 1.0)
        assert turns > 10

    def test_minimum_one_turn(self):
        """Should always have at least one turn."""
        turns = compute_turn_density(1, 0.0)
        assert turns >= 1

    def test_turn_scaling_range(self):
        """Turn multiplier should be in expected range (0.5x to 1.5x)."""
        low_turns = compute_turn_density(10, 0.0)
        high_turns = compute_turn_density(10, 1.0)

        assert low_turns >= 5   # 0.5x of 10
        assert high_turns <= 15  # 1.5x of 10


class TestHarmonicLayoutGeneration:
    """Tests for 2D harmonic layout generation."""

    def test_layout_is_2d_grid(self):
        """Layout should be a 2D grid."""
        layout = generate_harmonic_layout(9, 1.0, "planar", 0.5)

        assert len(layout) > 0
        assert all(isinstance(row, list) for row in layout)
        assert all(len(row) == len(layout[0]) for row in layout)

    def test_all_values_positive(self):
        """All spacing values should be positive."""
        layout = generate_harmonic_layout(16, PHI, "toroidal", 0.7)

        for row in layout:
            for val in row:
                assert val > 0

    def test_toroidal_has_radial_pattern(self):
        """Toroidal orientation should have radial falloff."""
        layout = generate_harmonic_layout(16, 1.0, "toroidal", 0.5)

        # Center values should differ from edges
        center = len(layout) // 2
        center_val = layout[center][center]
        edge_val = layout[0][0]

        # Toroidal should have higher values at center
        assert center_val != edge_val

    def test_planar_is_uniform(self):
        """Planar orientation should be uniform."""
        layout = generate_harmonic_layout(9, 1.0, "planar", 0.5)

        # All values should be equal (modifier = 1.0)
        first_val = layout[0][0]
        for row in layout:
            for val in row:
                assert val == pytest.approx(first_val, rel=0.01)


class TestPhaseEnvelopeSelection:
    """Tests for modulation phase selection."""

    def test_high_coherence_constant_phase(self):
        """High coherence should give constant phase."""
        phase = select_phase_envelope(0.9, TissueType.MUSCLE)
        assert phase == PhaseEnvelope.CONSTANT

    def test_heart_gets_pulsed_phase(self):
        """Heart tissue should get pulsed phase."""
        phase = select_phase_envelope(0.5, TissueType.HEART)
        assert phase == PhaseEnvelope.PULSED

    def test_blood_gets_pulsed_phase(self):
        """Blood tissue should get pulsed phase."""
        phase = select_phase_envelope(0.5, TissueType.BLOOD)
        assert phase == PhaseEnvelope.PULSED

    def test_brain_gets_oscillating_phase(self):
        """Brain tissue should get oscillating phase."""
        phase = select_phase_envelope(0.5, TissueType.BRAIN)
        assert phase == PhaseEnvelope.OSCILLATING

    def test_low_coherence_rising_phase(self):
        """Low coherence should give rising phase."""
        phase = select_phase_envelope(0.3, TissueType.SKIN)
        assert phase == PhaseEnvelope.RISING


class TestStabilizationGain:
    """Tests for stabilization gain computation."""

    def test_high_hrv_increases_gain(self):
        """High HRV should increase gain."""
        bio_high = BioState(hrv=0.9, pulse=60, coherence=0.5, hrv_latency_ms=50)
        bio_low = BioState(hrv=0.1, pulse=60, coherence=0.5, hrv_latency_ms=50)

        gain_high = compute_stabilization_gain(bio_high, 0.5)
        gain_low = compute_stabilization_gain(bio_low, 0.5)

        assert gain_high > gain_low

    def test_high_density_increases_gain(self):
        """High spacing density should increase gain."""
        bio = BioState(hrv=0.5, pulse=60, coherence=0.5, hrv_latency_ms=50)

        gain_dense = compute_stabilization_gain(bio, 1.0)
        gain_sparse = compute_stabilization_gain(bio, 0.2)

        assert gain_dense > gain_sparse

    def test_coherence_boosts_gain(self):
        """High coherence should boost gain."""
        bio_coherent = BioState(hrv=0.5, pulse=60, coherence=0.9, hrv_latency_ms=50)
        bio_incoherent = BioState(hrv=0.5, pulse=60, coherence=0.1, hrv_latency_ms=50)

        gain_coh = compute_stabilization_gain(bio_coherent, 0.5)
        gain_incoh = compute_stabilization_gain(bio_incoherent, 0.5)

        assert gain_coh > gain_incoh

    def test_gain_always_positive(self):
        """Gain should always be positive."""
        bio = BioState(hrv=0.0, pulse=60, coherence=0.0, hrv_latency_ms=100)
        gain = compute_stabilization_gain(bio, 0.0)
        assert gain > 0


class TestSyncMethodSelection:
    """Tests for sync method selection."""

    def test_no_pair_uses_biometric(self):
        """Without entangled pair, use biometric feedback."""
        bio = BioState(hrv=0.9, pulse=60, coherence=0.9, hrv_latency_ms=10)
        method = select_sync_method(bio, has_entangled_pair=False)
        assert method == SyncMethod.BIOMETRIC_FEEDBACK

    def test_high_coherence_low_latency_uses_quantum(self):
        """High coherence + low latency with pair uses quantum sync."""
        bio = BioState(hrv=0.5, pulse=60, coherence=0.8, hrv_latency_ms=30)
        method = select_sync_method(bio, has_entangled_pair=True)
        assert method == SyncMethod.QUANTUM_SYNC

    def test_low_coherence_uses_biometric(self):
        """Low coherence with pair uses biometric feedback."""
        bio = BioState(hrv=0.5, pulse=60, coherence=0.5, hrv_latency_ms=30)
        method = select_sync_method(bio, has_entangled_pair=True)
        assert method == SyncMethod.BIOMETRIC_FEEDBACK

    def test_high_latency_uses_biometric(self):
        """High latency with pair uses biometric feedback."""
        bio = BioState(hrv=0.5, pulse=60, coherence=0.9, hrv_latency_ms=100)
        method = select_sync_method(bio, has_entangled_pair=True)
        assert method == SyncMethod.BIOMETRIC_FEEDBACK


class TestTissueTemplates:
    """Tests for static tissue templates."""

    def test_all_tissues_have_templates(self):
        """All tissue types should have templates."""
        for tissue in TissueType:
            template = get_tissue_template(tissue)
            assert "base_turns" in template
            assert "spacing_factor" in template
            assert "orientation" in template
            assert "resonance_freq" in template

    def test_heart_uses_phi_spacing(self):
        """Heart template should use phi spacing."""
        template = get_tissue_template(TissueType.HEART)
        assert template["spacing_factor"] == pytest.approx(PHI)

    def test_brain_uses_fibonacci_turns(self):
        """Brain template should use Fibonacci turn count."""
        template = get_tissue_template(TissueType.BRAIN)
        assert template["base_turns"] == 21  # Fibonacci

    def test_heart_schumann_resonance(self):
        """Heart should resonate at Schumann frequency."""
        template = get_tissue_template(TissueType.HEART)
        assert template["resonance_freq"] == pytest.approx(7.83)


class TestBiofieldOverride:
    """Tests for dynamic biofield signature override."""

    def test_override_applies_resonance_modifier(self):
        """Biofield signature should modify resonance frequency."""
        template = {"base_turns": 10, "spacing_factor": 1.0,
                   "orientation": "planar", "resonance_freq": 100.0}

        signature = BioFieldSignature(
            user_id=uuid.uuid4(),
            tissue_resonances={TissueType.HEART: 1.5},
            coherence_map={},
            active_zones=[]
        )

        modified = apply_biofield_override(template, signature, TissueType.HEART)
        assert modified["resonance_freq"] == pytest.approx(150.0)

    def test_override_applies_coherence_spacing(self):
        """Biofield signature should modify spacing based on coherence."""
        template = {"base_turns": 10, "spacing_factor": 1.0,
                   "orientation": "planar", "resonance_freq": 100.0}

        signature = BioFieldSignature(
            user_id=uuid.uuid4(),
            tissue_resonances={},
            coherence_map={TissueType.LIVER: 0.5},
            active_zones=[]
        )

        modified = apply_biofield_override(template, signature, TissueType.LIVER)
        # spacing = 1.0 * (0.8 + 0.5 * 0.4) = 1.0
        assert modified["spacing_factor"] == pytest.approx(1.0)

    def test_no_signature_returns_original(self):
        """No signature should return original template."""
        template = {"base_turns": 10, "spacing_factor": 1.0,
                   "orientation": "planar", "resonance_freq": 100.0}

        modified = apply_biofield_override(template, None, TissueType.HEART)
        assert modified == template


class TestCoilMatrixGeneration:
    """Tests for complete coil matrix generation."""

    def test_generates_valid_matrix(self):
        """Should generate valid coil matrix."""
        field = ScalarField(
            points=[ScalarFieldPoint(RaCoordinate(0, 0, 0), 0.5, GradientVector(1, 0, 0))],
            mean_alpha=0.5,
            max_gradient=1.0
        )
        bio = BioState(hrv=0.5, pulse=60, coherence=0.5, hrv_latency_ms=50)

        matrix = generate_coil_matrix(field, bio, TissueType.HEART)

        assert matrix.coil_id is not None
        assert matrix.target_tissue == TissueType.HEART
        assert len(matrix.harmonic_layout) > 0
        assert matrix.stabilization_gain > 0

    def test_high_alpha_dense_layout(self):
        """High alpha should produce dense layout."""
        field_dense = ScalarField(points=[], mean_alpha=0.9, max_gradient=1.0)
        field_sparse = ScalarField(points=[], mean_alpha=0.2, max_gradient=1.0)
        bio = BioState(hrv=0.5, pulse=60, coherence=0.5, hrv_latency_ms=50)

        matrix_dense = generate_coil_matrix(field_dense, bio)
        matrix_sparse = generate_coil_matrix(field_sparse, bio)

        assert matrix_dense.spacing_density > matrix_sparse.spacing_density

    def test_creates_entangled_pair(self):
        """Should create entangled pair when requested."""
        field = ScalarField(points=[], mean_alpha=0.5, max_gradient=1.0)
        bio = BioState(hrv=0.5, pulse=60, coherence=0.8, hrv_latency_ms=30)

        matrix = generate_coil_matrix(field, bio, TissueType.HEART,
                                     create_entangled_pair=True)

        assert matrix.entangled_pair is not None
        assert matrix.entangled_pair.entangled_pair == matrix  # Bidirectional link
        assert matrix.entangled_pair.target_tissue == matrix.target_tissue

    def test_entangled_pair_shares_layout(self):
        """Entangled pairs should share harmonic layout."""
        field = ScalarField(points=[], mean_alpha=0.5, max_gradient=1.0)
        bio = BioState(hrv=0.5, pulse=60, coherence=0.8, hrv_latency_ms=30)

        matrix = generate_coil_matrix(field, bio, TissueType.BRAIN,
                                     create_entangled_pair=True)

        assert matrix.harmonic_layout == matrix.entangled_pair.harmonic_layout


class TestMirrorStabilization:
    """Tests for mirror coil stabilization pulses."""

    def test_biometric_sync_has_latency_offset(self):
        """Biometric sync should include HRV latency offset."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[[1.0]],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.PULSED,
            stabilization_gain=1.0,
            sync_method=SyncMethod.BIOMETRIC_FEEDBACK,
            spacing_density=0.5
        )
        bio = BioState(hrv=0.5, pulse=60, coherence=0.5, hrv_latency_ms=100)

        pulse = create_mirror_stabilization_pulse(matrix, bio)

        assert pulse["timestamp_offset"] == pytest.approx(0.1)  # 100ms

    def test_quantum_sync_zero_offset(self):
        """Quantum sync should have zero latency offset."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[[1.0]],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.PULSED,
            stabilization_gain=1.0,
            sync_method=SyncMethod.QUANTUM_SYNC,
            spacing_density=0.5
        )
        bio = BioState(hrv=0.5, pulse=60, coherence=0.9, hrv_latency_ms=100)

        pulse = create_mirror_stabilization_pulse(matrix, bio)

        assert pulse["timestamp_offset"] == pytest.approx(0.0)

    def test_pulse_carries_gain_and_phase(self):
        """Pulse should carry gain and phase information."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[[1.0]],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.OSCILLATING,
            stabilization_gain=1.5,
            sync_method=SyncMethod.QUANTUM_SYNC,
            spacing_density=0.5
        )
        bio = BioState(hrv=0.5, pulse=60, coherence=0.9, hrv_latency_ms=50)

        pulse = create_mirror_stabilization_pulse(matrix, bio)

        assert pulse["gain"] == 1.5
        assert pulse["phase"] == PhaseEnvelope.OSCILLATING


class TestCoilValidation:
    """Tests for coil alignment validation."""

    def test_valid_coil_passes(self):
        """Valid coil should pass validation."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[[1.0, 1.0], [1.0, 1.0]],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.PULSED,
            stabilization_gain=1.0,
            sync_method=SyncMethod.BIOMETRIC_FEEDBACK,
            spacing_density=0.5
        )

        assert validate_coil_alignment(matrix, RaCoordinate(0, 0, 0))

    def test_empty_layout_fails(self):
        """Empty layout should fail validation."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.PULSED,
            stabilization_gain=1.0,
            sync_method=SyncMethod.BIOMETRIC_FEEDBACK,
            spacing_density=0.5
        )

        assert not validate_coil_alignment(matrix, RaCoordinate(0, 0, 0))

    def test_zero_gain_fails(self):
        """Zero gain should fail validation."""
        matrix = CoilMatrix(
            coil_id=uuid.uuid4(),
            harmonic_layout=[[1.0]],
            target_tissue=TissueType.HEART,
            entangled_pair=None,
            modulation_phase=PhaseEnvelope.PULSED,
            stabilization_gain=0.0,
            sync_method=SyncMethod.BIOMETRIC_FEEDBACK,
            spacing_density=0.5
        )

        assert not validate_coil_alignment(matrix, RaCoordinate(0, 0, 0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
