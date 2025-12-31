"""
Prompt 20: Ra.Chamber.Tuning Test Harness

Biometric Lattice & Coherence Tensor for Chamber Tuning.
Multi-vector biometric input system with unified coherence tensor
derivation and resonance shell activation.

Codex References:
- Ra.Emergence: Shell activation based on coherence
- Ra.Coherence: Tensor derivation from biometric vectors
- P16: Overlay integration
- P19: Domain safety for edge cases
"""

import pytest
import math
import json
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from enum import Enum, auto
from datetime import datetime


# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

PHI = 1.618033988749895
BASE_FREQUENCY = 432.0  # Hz - coherence-scaled base
FREQUENCY_SCALE = 88.0  # Hz per coherence unit
OCTAVE_ANCHORS = [111.0, 222.0, 444.0, 888.0]  # Fallback frequencies
TORSION_THRESHOLD = 0.2  # Threshold for torsion polarity


# ============================================================================
# Types - Biometric Vectors
# ============================================================================

@dataclass
class CardiovascularVector:
    """Cardiovascular biometric vector."""
    heart_rate: int  # bpm
    hrv_lf_hf_ratio: float  # LowFreq/HighFreq HRV band
    heart_coherence: float  # 0.0-1.0


@dataclass
class NeuroVector:
    """Neuroelectrical biometric vector."""
    eeg_alpha: float
    eeg_theta: float
    eeg_delta: float
    neuro_coherence: float  # 0.0-1.0


@dataclass
class RespirationVector:
    """Respiratory biometric vector."""
    breath_rate: float  # breaths per minute
    inhale_exhale_sym: float  # [-1 to +1] symmetry
    breath_coherence: float  # 0.0-1.0


@dataclass
class ElectrodermalVector:
    """Electrodermal activity vector."""
    gsr_level: float
    tension_index: float  # 0-1 scale
    skin_coherence: float  # 0.0-1.0


@dataclass
class BiometricLattice:
    """Unified biometric input lattice with 4 vectors."""
    timestamp: datetime
    cardio: CardiovascularVector
    neuro: NeuroVector
    breath: RespirationVector
    eda: ElectrodermalVector
    lattice_seed: str


@dataclass
class BiometricState:
    """Legacy flat biometric state (adapter target)."""
    coherence_level: float
    heart_rate: int
    respiration_rate: float
    gsr_level: float
    biometric_seed: str


# ============================================================================
# Types - Coherence Tensor
# ============================================================================

@dataclass
class CoherenceTensor:
    """Derived coherence tensor from biometric lattice."""
    vector_scores: List[float]  # [heart, neuro, breath, skin]
    overall_score: float
    torsion_bias: float  # -1, 0, or +1
    harmonic_skew: float  # EEG/Breath phase angle shift


# ============================================================================
# Types - Chamber & Shell
# ============================================================================

class ChamberFlavor(Enum):
    """Unified chamber type (P20 + P21 combined)."""
    HEALING = auto()
    RETRIEVAL = auto()
    NAVIGATION = auto()
    DREAM = auto()
    ARCHIVE = auto()
    THERAPEUTIC = auto()
    EXPLORATORY = auto()


@dataclass
class ResonanceShell:
    """Single scalar resonance field layer."""
    layer_id: int
    harmonic_carrier: float  # Hz
    coherence_band: Tuple[float, float]  # (low, high)
    torsion_factor: float  # -1 to +1
    fluid_memory_seed: Optional[str] = None


@dataclass
class ChamberTuningProfile:
    """Complete chamber tuning profile."""
    timestamp: datetime
    user_id: str
    biometric_hash: str
    active_shells: List[ResonanceShell]
    ambient_mode: ChamberFlavor
    feedback_tone: Optional[float]  # Hz
    tensor: Optional[CoherenceTensor] = None


# ============================================================================
# Types - Symbolic Overlay
# ============================================================================

@dataclass
class SymbolicOverlay:
    """Symbolic overlay derived from coherence tensor."""
    theme: str
    symbol: str
    color: str
    sound: str
    placement: str
    derived_from: List[str]
    note: Optional[str] = None


# Default archetype mapping (configurable via JSON)
DEFAULT_ARCHETYPES = {
    "heart": {"symbol": "lion", "color": "#FF6B6B", "element": "fire"},
    "neuro": {"symbol": "owl", "color": "#9B59B6", "element": "air"},
    "breath": {"symbol": "wave", "color": "#3498DB", "element": "water"},
    "skin": {"symbol": "obsidian", "color": "#2C3E50", "element": "earth"},
}

TONE_FAMILIES = {
    "theta": {"range": (0.0, 0.2), "frequency": 432.0},
    "alpha": {"range": (0.2, 0.4), "frequency": 528.0},
    "gamma": {"range": (0.4, 1.0), "frequency": 639.0},
}


# ============================================================================
# Core Functions
# ============================================================================

def generate_coherence_tensor(lattice: BiometricLattice) -> CoherenceTensor:
    """
    Derive coherence tensor from biometric lattice.

    Uses tensor-sum for torsion calculation with threshold mapping.
    """
    hc = lattice.cardio.heart_coherence
    nc = lattice.neuro.neuro_coherence
    bc = lattice.breath.breath_coherence
    sc = lattice.eda.skin_coherence

    vector_scores = [hc, nc, bc, sc]
    overall_score = sum(vector_scores) / len(vector_scores)

    # Tensor-derived torsion with threshold
    tensor_sum = sum(vector_scores) - 2.0  # Centered around 2.0 (average 0.5 per vector)
    if tensor_sum < -TORSION_THRESHOLD:
        torsion_bias = -1.0
    elif tensor_sum > TORSION_THRESHOLD:
        torsion_bias = 1.0
    else:
        torsion_bias = 0.0

    # Harmonic skew from EEG and breath symmetry
    harmonic_skew = (lattice.neuro.eeg_alpha - lattice.neuro.eeg_delta) + \
                    lattice.breath.inhale_exhale_sym

    return CoherenceTensor(
        vector_scores=vector_scores,
        overall_score=overall_score,
        torsion_bias=torsion_bias,
        harmonic_skew=harmonic_skew
    )


def coherence_scaled_frequency(coherence: float) -> float:
    """Calculate coherence-scaled frequency: 432.0 + (c * 88.0)"""
    return BASE_FREQUENCY + (coherence * FREQUENCY_SCALE)


def get_octave_anchor(layer_id: int) -> float:
    """Get octave anchor frequency for fallback."""
    if 0 <= layer_id < len(OCTAVE_ANCHORS):
        return OCTAVE_ANCHORS[layer_id]
    return OCTAVE_ANCHORS[-1]


def shell_for_vector(vector_idx: int, coherence: float,
                     torsion: float, use_octave: bool = False) -> ResonanceShell:
    """
    Construct a resonance shell for a biometric vector.

    Args:
        vector_idx: Index of vector (0=heart, 1=neuro, 2=breath, 3=skin)
        coherence: Vector coherence value
        torsion: Global torsion bias
        use_octave: Use octave anchors instead of coherence-scaled
    """
    if use_octave:
        freq = get_octave_anchor(vector_idx)
    else:
        freq = coherence_scaled_frequency(coherence)

    # Coherence band with overlap allowance (±0.15)
    band = (max(0.0, coherence - 0.15), min(1.0, coherence + 0.15))

    # Fluid memory seed for high coherence
    seed = f"memory-v{vector_idx}" if coherence > 0.85 else None

    return ResonanceShell(
        layer_id=vector_idx,
        harmonic_carrier=freq,
        coherence_band=band,
        torsion_factor=torsion,
        fluid_memory_seed=seed
    )


def coherence_in_band(coherence: float, shell: ResonanceShell) -> bool:
    """Check if coherence activates a shell (allows overlap)."""
    low, high = shell.coherence_band
    return low <= coherence <= high


def generate_tuning_profile(lattice: BiometricLattice,
                            user_id: str = "default") -> ChamberTuningProfile:
    """
    Generate chamber tuning profile from biometric lattice.

    Supports shell overlap and coherence-scaled frequencies.
    """
    tensor = generate_coherence_tensor(lattice)

    # Generate shells for each vector
    shells = []
    for i, score in enumerate(tensor.vector_scores):
        shell = shell_for_vector(i, score, tensor.torsion_bias)
        # Allow overlap - check if overall coherence activates this shell
        if coherence_in_band(tensor.overall_score, shell):
            shells.append(shell)

    # If no shells activated, use fallback with octave anchors
    if not shells:
        for i, score in enumerate(tensor.vector_scores):
            shells.append(shell_for_vector(i, score, tensor.torsion_bias, use_octave=True))

    # Determine ambient mode
    if tensor.overall_score < 0.3183:  # 1/pi threshold
        mode = ChamberFlavor.THERAPEUTIC
    else:
        mode = ChamberFlavor.EXPLORATORY

    # Feedback tone based on dominant vector
    max_idx = tensor.vector_scores.index(max(tensor.vector_scores))
    feedback_tone = coherence_scaled_frequency(tensor.vector_scores[max_idx])

    return ChamberTuningProfile(
        timestamp=lattice.timestamp,
        user_id=user_id,
        biometric_hash=lattice.lattice_seed,
        active_shells=shells,
        ambient_mode=mode,
        feedback_tone=feedback_tone,
        tensor=tensor
    )


def lattice_to_state(lattice: BiometricLattice) -> BiometricState:
    """Adapter: Convert BiometricLattice to legacy BiometricState."""
    tensor = generate_coherence_tensor(lattice)
    return BiometricState(
        coherence_level=tensor.overall_score,
        heart_rate=lattice.cardio.heart_rate,
        respiration_rate=lattice.breath.breath_rate,
        gsr_level=lattice.eda.gsr_level,
        biometric_seed=lattice.lattice_seed
    )


def state_to_lattice(state: BiometricState,
                     timestamp: Optional[datetime] = None) -> BiometricLattice:
    """Adapter: Convert legacy BiometricState to BiometricLattice."""
    ts = timestamp or datetime.utcnow()
    coh = state.coherence_level

    return BiometricLattice(
        timestamp=ts,
        cardio=CardiovascularVector(
            heart_rate=state.heart_rate,
            hrv_lf_hf_ratio=1.5,  # Default
            heart_coherence=coh
        ),
        neuro=NeuroVector(
            eeg_alpha=0.4,
            eeg_theta=0.3,
            eeg_delta=0.2,
            neuro_coherence=coh
        ),
        breath=RespirationVector(
            breath_rate=state.respiration_rate,
            inhale_exhale_sym=0.0,
            breath_coherence=coh
        ),
        eda=ElectrodermalVector(
            gsr_level=state.gsr_level,
            tension_index=0.5,
            skin_coherence=coh
        ),
        lattice_seed=state.biometric_seed
    )


# ============================================================================
# Symbolic Overlay Functions
# ============================================================================

def get_tone_family(harmonic_skew: float) -> Tuple[str, float]:
    """Get tone family based on harmonic skew."""
    for name, info in TONE_FAMILIES.items():
        low, high = info["range"]
        if low <= abs(harmonic_skew) < high:
            return name, info["frequency"]
    return "gamma", 639.0


def generate_symbolic_overlay(tensor: CoherenceTensor,
                              user_intent: str = "",
                              archetypes: Optional[Dict] = None) -> SymbolicOverlay:
    """
    Generate symbolic overlay from coherence tensor.

    Maps biometric vectors to archetypal symbols based on coherence.
    """
    archetypes = archetypes or DEFAULT_ARCHETYPES

    # Find dominant vectors (top 2 by coherence)
    vector_names = ["heart", "neuro", "breath", "skin"]
    scored = list(zip(vector_names, tensor.vector_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    dominant = [s[0] for s in scored[:2]]

    # Get primary archetype
    primary = archetypes.get(dominant[0], DEFAULT_ARCHETYPES["heart"])

    # Determine symbol based on torsion
    if tensor.torsion_bias > 0:
        symbol = "spiral"
        placement = "layer_2"
    elif tensor.torsion_bias < 0:
        symbol = "cave"
        placement = "layer_3"
    else:
        symbol = primary["symbol"]
        placement = "layer_1"

    # Get tone family
    tone_name, tone_freq = get_tone_family(tensor.harmonic_skew)
    sound = f"{int(tone_freq)}Hz - {tone_name} pulse"

    # Theme from intent or default
    theme = user_intent if user_intent else "resonance"

    # Note for inverse torsion
    note = None
    if tensor.torsion_bias < 0:
        note = "Symbol reflects unresolved shadow vector. Consent required for reintegration."

    return SymbolicOverlay(
        theme=theme,
        symbol=symbol,
        color=primary["color"],
        sound=sound,
        placement=placement,
        derived_from=dominant,
        note=note
    )


def symbolic_overlay_to_json(overlay: SymbolicOverlay) -> Dict[str, Any]:
    """Convert symbolic overlay to JSON-conforming dict."""
    result = {
        "symbolic_overlay": {
            "theme": overlay.theme,
            "symbol": overlay.symbol,
            "color": overlay.color,
            "sound": overlay.sound,
            "placement": overlay.placement,
            "derived_from": overlay.derived_from,
        }
    }
    if overlay.note:
        result["symbolic_overlay"]["note"] = overlay.note
    return result


# ============================================================================
# Test Suite
# ============================================================================

class TestBiometricLattice:
    """Test biometric lattice creation and validation."""

    def test_create_lattice(self):
        """Create a valid biometric lattice."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.42),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.51),
            breath=RespirationVector(13.0, 0.05, 0.48),
            eda=ElectrodermalVector(0.7, 0.3, 0.43),
            lattice_seed="test-seed"
        )
        assert lattice.cardio.heart_rate == 72
        assert lattice.neuro.neuro_coherence == 0.51

    def test_vector_coherence_range(self):
        """Vector coherence values are in valid range."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(65, 1.5, 0.95),
            neuro=NeuroVector(0.5, 0.3, 0.2, 0.88),
            breath=RespirationVector(12.0, 0.1, 0.92),
            eda=ElectrodermalVector(0.5, 0.2, 0.90),
            lattice_seed="high-coherence"
        )
        for vec in [lattice.cardio.heart_coherence,
                    lattice.neuro.neuro_coherence,
                    lattice.breath.breath_coherence,
                    lattice.eda.skin_coherence]:
            assert 0.0 <= vec <= 1.0


class TestCoherenceTensor:
    """Test coherence tensor derivation."""

    def test_tensor_from_lattice(self):
        """Derive coherence tensor from lattice."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.42),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.51),
            breath=RespirationVector(13.0, 0.05, 0.48),
            eda=ElectrodermalVector(0.7, 0.3, 0.43),
            lattice_seed="test-seed"
        )
        tensor = generate_coherence_tensor(lattice)

        assert tensor.vector_scores == [0.42, 0.51, 0.48, 0.43]
        assert abs(tensor.overall_score - 0.46) < 0.01

    def test_torsion_positive(self):
        """High coherence produces positive torsion."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(65, 1.5, 0.85),
            neuro=NeuroVector(0.5, 0.3, 0.2, 0.88),
            breath=RespirationVector(12.0, 0.1, 0.82),
            eda=ElectrodermalVector(0.5, 0.2, 0.80),
            lattice_seed="high"
        )
        tensor = generate_coherence_tensor(lattice)
        assert tensor.torsion_bias == 1.0

    def test_torsion_negative(self):
        """Low coherence produces negative torsion."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(90, 2.5, 0.15),
            neuro=NeuroVector(0.2, 0.5, 0.4, 0.18),
            breath=RespirationVector(18.0, -0.3, 0.12),
            eda=ElectrodermalVector(1.2, 0.8, 0.20),
            lattice_seed="low"
        )
        tensor = generate_coherence_tensor(lattice)
        assert tensor.torsion_bias == -1.0

    def test_torsion_neutral(self):
        """Mid coherence produces neutral torsion."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.50),
            neuro=NeuroVector(0.4, 0.35, 0.25, 0.50),
            breath=RespirationVector(14.0, 0.0, 0.50),
            eda=ElectrodermalVector(0.6, 0.4, 0.50),
            lattice_seed="mid"
        )
        tensor = generate_coherence_tensor(lattice)
        assert tensor.torsion_bias == 0.0

    def test_harmonic_skew(self):
        """Harmonic skew calculated from EEG and breath."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.50),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.50),  # alpha=0.45, delta=0.22
            breath=RespirationVector(14.0, 0.05, 0.50),  # sym=0.05
            eda=ElectrodermalVector(0.6, 0.4, 0.50),
            lattice_seed="skew-test"
        )
        tensor = generate_coherence_tensor(lattice)
        # skew = (0.45 - 0.22) + 0.05 = 0.28
        assert abs(tensor.harmonic_skew - 0.28) < 0.01


class TestResonanceShell:
    """Test resonance shell generation."""

    def test_coherence_scaled_frequency(self):
        """Coherence-scaled frequency calculation."""
        freq = coherence_scaled_frequency(0.5)
        assert abs(freq - 476.0) < 0.1  # 432 + (0.5 * 88)

    def test_octave_anchor_fallback(self):
        """Octave anchor frequencies for fallback."""
        assert get_octave_anchor(0) == 111.0
        assert get_octave_anchor(1) == 222.0
        assert get_octave_anchor(2) == 444.0
        assert get_octave_anchor(3) == 888.0

    def test_shell_overlap_bands(self):
        """Shell bands allow overlap (±0.15)."""
        shell = shell_for_vector(0, 0.5, 0.0)
        assert shell.coherence_band == (0.35, 0.65)

    def test_fluid_memory_seed_high_coherence(self):
        """Fluid memory seed set for high coherence."""
        shell = shell_for_vector(0, 0.90, 1.0)
        assert shell.fluid_memory_seed == "memory-v0"

    def test_no_fluid_memory_low_coherence(self):
        """No fluid memory seed for low coherence."""
        shell = shell_for_vector(0, 0.50, 0.0)
        assert shell.fluid_memory_seed is None


class TestTuningProfile:
    """Test chamber tuning profile generation."""

    def test_generate_profile(self):
        """Generate tuning profile from lattice."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.55),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.60),
            breath=RespirationVector(13.0, 0.05, 0.58),
            eda=ElectrodermalVector(0.7, 0.3, 0.52),
            lattice_seed="profile-test"
        )
        profile = generate_tuning_profile(lattice, "user-123")

        assert profile.user_id == "user-123"
        assert len(profile.active_shells) > 0
        assert profile.tensor is not None

    def test_therapeutic_mode_low_coherence(self):
        """Low coherence triggers therapeutic mode."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(85, 2.0, 0.20),
            neuro=NeuroVector(0.3, 0.4, 0.3, 0.22),
            breath=RespirationVector(16.0, -0.1, 0.18),
            eda=ElectrodermalVector(0.9, 0.6, 0.25),
            lattice_seed="low-coh"
        )
        profile = generate_tuning_profile(lattice)
        assert profile.ambient_mode == ChamberFlavor.THERAPEUTIC

    def test_exploratory_mode_high_coherence(self):
        """High coherence triggers exploratory mode."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(65, 1.5, 0.75),
            neuro=NeuroVector(0.5, 0.3, 0.2, 0.80),
            breath=RespirationVector(12.0, 0.1, 0.78),
            eda=ElectrodermalVector(0.5, 0.2, 0.72),
            lattice_seed="high-coh"
        )
        profile = generate_tuning_profile(lattice)
        assert profile.ambient_mode == ChamberFlavor.EXPLORATORY

    def test_deterministic_profile(self):
        """Same input produces same profile."""
        ts = datetime(2025, 12, 30, 12, 0, 0)
        lattice = BiometricLattice(
            timestamp=ts,
            cardio=CardiovascularVector(72, 1.8, 0.55),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.60),
            breath=RespirationVector(13.0, 0.05, 0.58),
            eda=ElectrodermalVector(0.7, 0.3, 0.52),
            lattice_seed="stable"
        )
        p1 = generate_tuning_profile(lattice)
        p2 = generate_tuning_profile(lattice)

        assert len(p1.active_shells) == len(p2.active_shells)
        for s1, s2 in zip(p1.active_shells, p2.active_shells):
            assert s1.harmonic_carrier == s2.harmonic_carrier


class TestLegacyAdapter:
    """Test BiometricState <-> BiometricLattice adapters."""

    def test_lattice_to_state(self):
        """Convert lattice to legacy state."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.55),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.60),
            breath=RespirationVector(13.0, 0.05, 0.58),
            eda=ElectrodermalVector(0.7, 0.3, 0.52),
            lattice_seed="adapter-test"
        )
        state = lattice_to_state(lattice)

        assert state.heart_rate == 72
        assert state.respiration_rate == 13.0
        assert state.gsr_level == 0.7
        assert state.biometric_seed == "adapter-test"

    def test_state_to_lattice(self):
        """Convert legacy state to lattice."""
        state = BiometricState(
            coherence_level=0.65,
            heart_rate=68,
            respiration_rate=14.0,
            gsr_level=0.55,
            biometric_seed="legacy-seed"
        )
        lattice = state_to_lattice(state)

        assert lattice.cardio.heart_rate == 68
        assert lattice.cardio.heart_coherence == 0.65
        assert lattice.breath.breath_rate == 14.0

    def test_roundtrip(self):
        """Roundtrip conversion preserves key values."""
        original = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(72, 1.8, 0.55),
            neuro=NeuroVector(0.45, 0.33, 0.22, 0.55),
            breath=RespirationVector(13.0, 0.05, 0.55),
            eda=ElectrodermalVector(0.7, 0.3, 0.55),
            lattice_seed="roundtrip"
        )
        state = lattice_to_state(original)
        restored = state_to_lattice(state)

        assert restored.cardio.heart_rate == original.cardio.heart_rate
        assert restored.lattice_seed == original.lattice_seed


class TestSymbolicOverlay:
    """Test symbolic overlay generation."""

    def test_constructive_torsion_overlay(self):
        """Constructive torsion produces upward spiral."""
        tensor = CoherenceTensor(
            vector_scores=[0.75, 0.80, 0.78, 0.72],
            overall_score=0.76,
            torsion_bias=1.0,
            harmonic_skew=0.25
        )
        overlay = generate_symbolic_overlay(tensor, "clarity")

        assert overlay.symbol == "spiral"
        assert overlay.placement == "layer_2"
        assert overlay.note is None

    def test_inverse_torsion_overlay(self):
        """Inverse torsion produces cave symbol with note."""
        tensor = CoherenceTensor(
            vector_scores=[0.20, 0.18, 0.22, 0.25],
            overall_score=0.21,
            torsion_bias=-1.0,
            harmonic_skew=0.53
        )
        overlay = generate_symbolic_overlay(tensor)

        assert overlay.symbol == "cave"
        assert overlay.placement == "layer_3"
        assert "shadow" in overlay.note.lower()

    def test_tone_family_theta(self):
        """Low skew produces theta tone."""
        tensor = CoherenceTensor(
            vector_scores=[0.50, 0.50, 0.50, 0.50],
            overall_score=0.50,
            torsion_bias=0.0,
            harmonic_skew=0.11
        )
        overlay = generate_symbolic_overlay(tensor)

        assert "432Hz" in overlay.sound
        assert "theta" in overlay.sound

    def test_tone_family_alpha(self):
        """Mid skew produces alpha tone."""
        tensor = CoherenceTensor(
            vector_scores=[0.50, 0.50, 0.50, 0.50],
            overall_score=0.50,
            torsion_bias=0.0,
            harmonic_skew=0.30
        )
        overlay = generate_symbolic_overlay(tensor)

        assert "528Hz" in overlay.sound
        assert "alpha" in overlay.sound

    def test_overlay_json_output(self):
        """Overlay converts to valid JSON structure."""
        tensor = CoherenceTensor(
            vector_scores=[0.60, 0.65, 0.58, 0.62],
            overall_score=0.61,
            torsion_bias=0.0,
            harmonic_skew=0.22
        )
        overlay = generate_symbolic_overlay(tensor, "emotional clarity")
        json_out = symbolic_overlay_to_json(overlay)

        assert "symbolic_overlay" in json_out
        assert json_out["symbolic_overlay"]["theme"] == "emotional clarity"
        assert len(json_out["symbolic_overlay"]["derived_from"]) == 2


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline(self):
        """Full pipeline from lattice to symbolic overlay."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(68, 1.6, 0.72),
            neuro=NeuroVector(0.48, 0.32, 0.20, 0.78),
            breath=RespirationVector(12.0, 0.08, 0.75),
            eda=ElectrodermalVector(0.55, 0.25, 0.70),
            lattice_seed="integration-test"
        )

        # Generate profile
        profile = generate_tuning_profile(lattice)
        assert profile.ambient_mode == ChamberFlavor.EXPLORATORY

        # Generate overlay
        overlay = generate_symbolic_overlay(profile.tensor, "insight")
        assert overlay.symbol == "spiral"  # High coherence = positive torsion

    def test_shell_frequency_visualization(self):
        """Extract shell frequencies for visualization."""
        lattice = BiometricLattice(
            timestamp=datetime.utcnow(),
            cardio=CardiovascularVector(70, 1.7, 0.65),
            neuro=NeuroVector(0.45, 0.35, 0.22, 0.70),
            breath=RespirationVector(13.0, 0.05, 0.68),
            eda=ElectrodermalVector(0.6, 0.3, 0.62),
            lattice_seed="viz-test"
        )
        profile = generate_tuning_profile(lattice)

        frequencies = [s.harmonic_carrier for s in profile.active_shells]
        assert all(f > 400.0 for f in frequencies)  # All above base


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
