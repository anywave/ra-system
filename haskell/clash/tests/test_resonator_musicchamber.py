#!/usr/bin/env python3
"""
Test harness for Prompt 47: Ra.Resonator.MusicChamber

A biometric-driven music/sound chamber that:
- Maps coherence to rhythm/BPM
- Uses φ^(1+coherence) loop timing
- Applies avatar tones (Metallic, Crystalline, Hollow, Warm)
- Implements inversion state (waveform polarity + stereo flip)
- Generates φ-harmonic sound fields

Clarifications:
- Biometric bands: heartRate 50-140, hrv 0-100ms, respiration 6-18rpm
- Inversion: both waveform polarity AND stereo panning
- Loop timing: φ^(1+coherence) seconds for smooth resolution
- Use provided Haskell as functional reference
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Biometric input ranges
HEART_RATE_MIN = 50
HEART_RATE_MAX = 140
HRV_MIN = 0
HRV_MAX = 100
RESPIRATION_MIN = 6
RESPIRATION_MAX = 18

# BPM range (derived from coherence)
BPM_BASE = 60
BPM_RANGE = 80  # BPM = 60 + coherence * 80

# =============================================================================
# Enumerations
# =============================================================================

class HarmonicScale(Enum):
    """Musical scale types."""
    PHI_MAJOR = auto()       # φ-based major scale
    ROOT_10_MINOR = auto()   # √10 tuning minor
    CUSTOM = auto()          # Custom frequency list


class SoundSpatialProfile(Enum):
    """Spatial audio profiles."""
    STEREO_SWIRL = auto()    # Rotating stereo field
    VORTEX_SPIRAL = auto()   # Spiraling vortex pattern
    CENTER_PULSE = auto()    # Centered pulsing


class InversionState(Enum):
    """Waveform/stereo inversion state."""
    NORMAL = auto()
    INVERTED = auto()


class AvatarTone(Enum):
    """Avatar timbre characteristics."""
    METALLIC = auto()     # Sharp, metallic overtones
    CRYSTALLINE = auto()  # Clear, crystalline shimmer
    HOLLOW = auto()       # Deep, hollow resonance
    WARM = auto()         # Soft, warm harmonics


class ResonanceModType(Enum):
    """Types of resonance modulation."""
    COHERENCE_PULSE = auto()
    FLUX_DRIFT = auto()
    INVERSION_SHIFT = auto()
    AVATAR_OVERLAY = auto()


# =============================================================================
# Data Types
# =============================================================================

@dataclass
class BiometricInput:
    """
    Biometric sensor input.

    Attributes:
        heart_rate: Heart rate in BPM (50-140)
        hrv: Heart rate variability in ms (0-100)
        respiration: Respiration rate in rpm (6-18)
        skin_conductance: Skin conductance (0-1)
        coherence: Overall coherence score (0-1)
    """
    heart_rate: float
    hrv: float
    respiration: float
    skin_conductance: float
    coherence: float

    def __post_init__(self):
        """Validate and clamp inputs."""
        self.heart_rate = max(HEART_RATE_MIN, min(HEART_RATE_MAX, self.heart_rate))
        self.hrv = max(HRV_MIN, min(HRV_MAX, self.hrv))
        self.respiration = max(RESPIRATION_MIN, min(RESPIRATION_MAX, self.respiration))
        self.skin_conductance = max(0.0, min(1.0, self.skin_conductance))
        self.coherence = max(0.0, min(1.0, self.coherence))


@dataclass
class TemporalEnvelope:
    """
    Timing envelope for sound field.

    Attributes:
        bpm: Beats per minute
        loop_duration: Loop length in seconds (φ-scaled)
    """
    bpm: float
    loop_duration: float


@dataclass
class ResonanceMod:
    """
    A resonance modulation effect.

    Attributes:
        mod_type: Type of modulation
        value: Modulation value (float for Pulse/Drift, InversionState, AvatarTone)
    """
    mod_type: ResonanceModType
    value: any


@dataclass
class AvatarProfile:
    """
    Avatar audio configuration.

    Attributes:
        tone: Avatar timbre
        invert: Whether inversion is active
    """
    tone: AvatarTone
    invert: bool = False


@dataclass
class ChamberConfig:
    """
    Chamber sound configuration.

    Attributes:
        scale: Harmonic scale type
        spatial: Spatial audio profile
        custom_frequencies: Custom frequencies for CUSTOM scale
    """
    scale: HarmonicScale
    spatial: SoundSpatialProfile
    custom_frequencies: Optional[List[float]] = None


@dataclass
class ScalarField:
    """Scalar field state (placeholder for integration)."""
    coherence: float = 0.5
    dominant_l: int = 0
    phase_angle: float = 0.0


@dataclass
class ChamberSoundField:
    """
    Generated sound field output.

    Attributes:
        base_scale: The harmonic scale used
        pulse_envelope: Timing envelope
        spatial_profile: Spatial audio profile
        modulations: Active resonance modulations
        waveform_inverted: Whether waveform is inverted
        stereo_inverted: Whether stereo is inverted
    """
    base_scale: HarmonicScale
    pulse_envelope: TemporalEnvelope
    spatial_profile: SoundSpatialProfile
    modulations: List[ResonanceMod]
    waveform_inverted: bool = False
    stereo_inverted: bool = False


@dataclass
class DiagnosticFrame:
    """
    Runtime diagnostic snapshot.

    Attributes:
        bpm_out: Output BPM
        loop_out: Loop duration in seconds
        mods_active: List of active modulation descriptions
        avatar_tone: Current avatar tone
        inversion_active: Whether inversion is active
    """
    bpm_out: float
    loop_out: float
    mods_active: List[str]
    avatar_tone: Optional[AvatarTone]
    inversion_active: bool


# =============================================================================
# Core Functions
# =============================================================================

def compute_bpm(coherence: float) -> float:
    """
    Compute BPM from coherence.
    BPM = 60 + coherence * 80
    Range: 60 (low coherence) to 140 (high coherence)
    """
    return BPM_BASE + coherence * BPM_RANGE


def compute_phi_loop(coherence: float) -> float:
    """
    Compute loop duration using φ^(1+coherence).
    This gives smooth, non-discrete phase resolution.

    Examples:
    - coherence 0.0: φ^1 ≈ 1.618s
    - coherence 0.5: φ^1.5 ≈ 2.058s
    - coherence 1.0: φ^2 ≈ 2.618s
    """
    return PHI ** (1 + coherence)


def compute_phi_loop_quantized(coherence: float) -> Tuple[float, int]:
    """
    Compute quantized loop duration snapped to φ^n.
    Returns (loop_duration, n)
    """
    # Determine nearest integer n based on coherence
    n = round(1 + coherence * 2)  # n ranges from 1 to 3 for coherence 0-1
    n = max(1, min(8, n))  # Clamp to valid range
    return (PHI ** n, n)


def normalize_hrv(hrv: float) -> float:
    """Normalize HRV to 0-1 range for modulation index."""
    return hrv / HRV_MAX


def normalize_respiration(respiration: float) -> float:
    """Normalize respiration to 0-1 range for LFO modulation."""
    return (respiration - RESPIRATION_MIN) / (RESPIRATION_MAX - RESPIRATION_MIN)


def apply_inversion(sample: float, inverted: bool) -> float:
    """Apply waveform inversion (polarity flip)."""
    return -sample if inverted else sample


def apply_stereo_inversion(left: float, right: float, inverted: bool) -> Tuple[float, float]:
    """Apply stereo inversion (channel swap)."""
    return (right, left) if inverted else (left, right)


def generate_chamber_sound(
    bio: BiometricInput,
    scalar_field: ScalarField,
    avatar: AvatarProfile,
    config: ChamberConfig
) -> ChamberSoundField:
    """
    Generate a ChamberSoundField from biometric input.

    Parameters:
        bio: Biometric sensor input
        scalar_field: Current scalar field state
        avatar: Avatar configuration
        config: Chamber configuration

    Returns:
        ChamberSoundField with computed parameters
    """
    # Compute timing from coherence
    bpm = compute_bpm(bio.coherence)
    loop_duration = compute_phi_loop(bio.coherence)
    envelope = TemporalEnvelope(bpm, loop_duration)

    # Build modulation list
    modulations = [
        ResonanceMod(ResonanceModType.COHERENCE_PULSE, bio.coherence),
        ResonanceMod(ResonanceModType.FLUX_DRIFT, normalize_hrv(bio.hrv)),
        ResonanceMod(ResonanceModType.AVATAR_OVERLAY, avatar.tone),
    ]

    # Add inversion if active
    waveform_inverted = False
    stereo_inverted = False
    if avatar.invert:
        modulations.append(
            ResonanceMod(ResonanceModType.INVERSION_SHIFT, InversionState.INVERTED)
        )
        waveform_inverted = True
        stereo_inverted = True

    return ChamberSoundField(
        base_scale=config.scale,
        pulse_envelope=envelope,
        spatial_profile=config.spatial,
        modulations=modulations,
        waveform_inverted=waveform_inverted,
        stereo_inverted=stereo_inverted
    )


def extract_tempo(sound_field: ChamberSoundField) -> float:
    """Extract BPM from sound field."""
    return sound_field.pulse_envelope.bpm


def extract_loop_duration(sound_field: ChamberSoundField) -> float:
    """Extract loop duration from sound field."""
    return sound_field.pulse_envelope.loop_duration


def is_waveform_inverted(sound_field: ChamberSoundField) -> bool:
    """Check if waveform is inverted."""
    return sound_field.waveform_inverted


def is_stereo_inverted(sound_field: ChamberSoundField) -> bool:
    """Check if stereo is inverted."""
    return sound_field.stereo_inverted


def get_avatar_tone(sound_field: ChamberSoundField) -> Optional[AvatarTone]:
    """Extract avatar tone from sound field modulations."""
    for mod in sound_field.modulations:
        if mod.mod_type == ResonanceModType.AVATAR_OVERLAY:
            return mod.value
    return None


def has_inversion_shift(sound_field: ChamberSoundField) -> bool:
    """Check if inversion shift modulation is present."""
    return any(
        mod.mod_type == ResonanceModType.INVERSION_SHIFT
        for mod in sound_field.modulations
    )


def simulate_diagnostics(
    bio: BiometricInput,
    avatar: AvatarProfile,
    config: ChamberConfig
) -> DiagnosticFrame:
    """
    Generate diagnostic frame from inputs.
    """
    sf = generate_chamber_sound(bio, ScalarField(), avatar, config)

    return DiagnosticFrame(
        bpm_out=sf.pulse_envelope.bpm,
        loop_out=sf.pulse_envelope.loop_duration,
        mods_active=[f"{m.mod_type.name}: {m.value}" for m in sf.modulations],
        avatar_tone=get_avatar_tone(sf),
        inversion_active=has_inversion_shift(sf)
    )


# =============================================================================
# Scale and Frequency Functions
# =============================================================================

def get_phi_major_frequencies(base_freq: float = 432.0) -> List[float]:
    """
    Generate φ-based major scale frequencies.
    Each successive note is base * φ^(n/12) for harmonic spacing.
    """
    return [base_freq * (PHI ** (n / 12)) for n in range(12)]


def get_root_10_minor_frequencies(base_freq: float = 220.0) -> List[float]:
    """
    Generate √10 tuning minor scale frequencies.
    Uses √10 ≈ 3.162 as the tuning ratio.
    """
    root_10 = math.sqrt(10)
    return [base_freq * (root_10 ** (n / 12)) for n in range(12)]


# =============================================================================
# Test Cases
# =============================================================================

class TestBPMComputation:
    """Test BPM computation from coherence."""

    def test_low_coherence_low_bpm(self):
        """Coherence 0 gives BPM 60."""
        bpm = compute_bpm(0.0)
        assert bpm == 60.0

    def test_high_coherence_high_bpm(self):
        """Coherence 1 gives BPM 140."""
        bpm = compute_bpm(1.0)
        assert bpm == 140.0

    def test_mid_coherence_mid_bpm(self):
        """Coherence 0.5 gives BPM 100."""
        bpm = compute_bpm(0.5)
        assert bpm == 100.0

    def test_coherence_modulation_expected_range(self):
        """From prompt: coherence 0.9 should give BPM 110-140."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.9)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.METALLIC, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )
        tempo = extract_tempo(sf)
        assert 110 < tempo < 140, f"Expected tempo 110-140, got {tempo}"

    def test_bpm_linear_interpolation(self):
        """BPM interpolates linearly with coherence."""
        bpm_0 = compute_bpm(0.0)
        bpm_25 = compute_bpm(0.25)
        bpm_50 = compute_bpm(0.5)
        bpm_75 = compute_bpm(0.75)
        bpm_100 = compute_bpm(1.0)

        # Check linear spacing
        assert abs(bpm_25 - bpm_0 - 20) < 0.01
        assert abs(bpm_50 - bpm_25 - 20) < 0.01
        assert abs(bpm_75 - bpm_50 - 20) < 0.01
        assert abs(bpm_100 - bpm_75 - 20) < 0.01


class TestPhiLoopTiming:
    """Test φ-based loop timing."""

    def test_phi_loop_coherence_0(self):
        """Coherence 0 gives φ^1 ≈ 1.618s."""
        loop = compute_phi_loop(0.0)
        assert abs(loop - PHI) < 0.001

    def test_phi_loop_coherence_1(self):
        """Coherence 1 gives φ^2 ≈ 2.618s."""
        loop = compute_phi_loop(1.0)
        expected = PHI ** 2
        assert abs(loop - expected) < 0.001

    def test_phi_loop_coherence_half(self):
        """Coherence 0.5 gives φ^1.5."""
        loop = compute_phi_loop(0.5)
        expected = PHI ** 1.5
        assert abs(loop - expected) < 0.001

    def test_phi_loop_increases_with_coherence(self):
        """Loop duration increases with coherence."""
        loop_0 = compute_phi_loop(0.0)
        loop_25 = compute_phi_loop(0.25)
        loop_50 = compute_phi_loop(0.5)
        loop_75 = compute_phi_loop(0.75)
        loop_100 = compute_phi_loop(1.0)

        assert loop_0 < loop_25 < loop_50 < loop_75 < loop_100

    def test_phi_loop_quantized_coherence_0(self):
        """Quantized loop at coherence 0 snaps to φ^1."""
        loop, n = compute_phi_loop_quantized(0.0)
        assert n == 1
        assert abs(loop - PHI) < 0.001

    def test_phi_loop_quantized_coherence_1(self):
        """Quantized loop at coherence 1 snaps to φ^3."""
        loop, n = compute_phi_loop_quantized(1.0)
        assert n == 3
        assert abs(loop - PHI ** 3) < 0.001

    def test_loop_duration_in_sound_field(self):
        """Loop duration computed correctly in sound field."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.88)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.CRYSTALLINE, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )
        loop = extract_loop_duration(sf)
        expected = PHI ** (1 + 0.88)

        assert abs(loop - expected) < 0.01


class TestInversionState:
    """Test inversion state handling."""

    def test_inversion_shift_triggers_waveform_flip(self):
        """Inversion triggers waveform polarity flip."""
        bio = BiometricInput(60, 45, 10, 0.3, 0.4)
        avatar = AvatarProfile(AvatarTone.CRYSTALLINE, invert=True)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.ROOT_10_MINOR, SoundSpatialProfile.VORTEX_SPIRAL)
        )

        assert is_waveform_inverted(sf) is True
        assert has_inversion_shift(sf) is True

    def test_inversion_shift_triggers_stereo_flip(self):
        """Inversion triggers stereo channel swap."""
        bio = BiometricInput(60, 45, 10, 0.3, 0.4)
        avatar = AvatarProfile(AvatarTone.HOLLOW, invert=True)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.ROOT_10_MINOR, SoundSpatialProfile.VORTEX_SPIRAL)
        )

        assert is_stereo_inverted(sf) is True

    def test_no_inversion_by_default(self):
        """No inversion when avatar.invert is False."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.9)
        avatar = AvatarProfile(AvatarTone.METALLIC, invert=False)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        assert is_waveform_inverted(sf) is False
        assert is_stereo_inverted(sf) is False
        assert has_inversion_shift(sf) is False

    def test_apply_waveform_inversion(self):
        """Waveform inversion flips sample polarity."""
        sample = 0.5
        inverted = apply_inversion(sample, True)
        normal = apply_inversion(sample, False)

        assert inverted == -0.5
        assert normal == 0.5

    def test_apply_stereo_inversion(self):
        """Stereo inversion swaps left/right channels."""
        left, right = 0.3, 0.7
        inv_left, inv_right = apply_stereo_inversion(left, right, True)
        norm_left, norm_right = apply_stereo_inversion(left, right, False)

        assert inv_left == 0.7  # Was right
        assert inv_right == 0.3  # Was left
        assert norm_left == 0.3
        assert norm_right == 0.7


class TestAvatarToneMapping:
    """Test avatar tone mapping to overlay."""

    def test_avatar_tone_metallic(self):
        """Metallic tone correctly mapped."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.9)
        avatar = AvatarProfile(AvatarTone.METALLIC, False)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        assert get_avatar_tone(sf) == AvatarTone.METALLIC

    def test_avatar_tone_crystalline(self):
        """Crystalline tone correctly mapped."""
        bio = BiometricInput(72, 60, 14, 0.45, 0.88)
        avatar = AvatarProfile(AvatarTone.CRYSTALLINE, False)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        assert get_avatar_tone(sf) == AvatarTone.CRYSTALLINE

    def test_avatar_tone_hollow(self):
        """Hollow tone correctly mapped."""
        bio = BiometricInput(68, 42, 11, 0.35, 0.7)
        avatar = AvatarProfile(AvatarTone.HOLLOW, False)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.CUSTOM, SoundSpatialProfile.CENTER_PULSE,
                         [220, 440])
        )

        assert get_avatar_tone(sf) == AvatarTone.HOLLOW

    def test_avatar_tone_warm(self):
        """Warm tone correctly mapped."""
        bio = BiometricInput(65, 55, 10, 0.5, 0.8)
        avatar = AvatarProfile(AvatarTone.WARM, False)
        sf = generate_chamber_sound(
            bio, ScalarField(), avatar,
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        assert get_avatar_tone(sf) == AvatarTone.WARM


class TestBiometricMapping:
    """Test biometric input mapping to modulations."""

    def test_hrv_maps_to_flux_drift(self):
        """HRV maps to flux drift modulation."""
        bio = BiometricInput(70, 60, 12, 0.4, 0.8)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.METALLIC, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        flux_mods = [m for m in sf.modulations if m.mod_type == ResonanceModType.FLUX_DRIFT]
        assert len(flux_mods) == 1
        assert abs(flux_mods[0].value - 0.6) < 0.01  # 60/100 = 0.6

    def test_coherence_maps_to_pulse(self):
        """Coherence maps to coherence pulse modulation."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.75)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.METALLIC, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        pulse_mods = [m for m in sf.modulations if m.mod_type == ResonanceModType.COHERENCE_PULSE]
        assert len(pulse_mods) == 1
        assert pulse_mods[0].value == 0.75

    def test_respiration_normalization(self):
        """Respiration normalized correctly."""
        # Min respiration
        norm_min = normalize_respiration(6)
        assert norm_min == 0.0

        # Max respiration
        norm_max = normalize_respiration(18)
        assert norm_max == 1.0

        # Mid respiration
        norm_mid = normalize_respiration(12)
        assert abs(norm_mid - 0.5) < 0.01

    def test_biometric_input_clamping(self):
        """Biometric inputs are clamped to valid ranges."""
        # Values outside range should be clamped
        bio = BiometricInput(
            heart_rate=200,    # Over max
            hrv=-10,           # Under min
            respiration=30,    # Over max
            skin_conductance=2.0,  # Over max
            coherence=1.5      # Over max
        )

        assert bio.heart_rate == 140  # Clamped to max
        assert bio.hrv == 0           # Clamped to min
        assert bio.respiration == 18  # Clamped to max
        assert bio.skin_conductance == 1.0
        assert bio.coherence == 1.0


class TestScaleGeneration:
    """Test harmonic scale frequency generation."""

    def test_phi_major_scale_base_frequency(self):
        """φ-major scale starts at base frequency."""
        freqs = get_phi_major_frequencies(432.0)
        assert freqs[0] == 432.0

    def test_phi_major_scale_phi_spacing(self):
        """φ-major scale has φ-based spacing."""
        freqs = get_phi_major_frequencies(432.0)

        # Each note should be previous * φ^(1/12)
        for i in range(1, len(freqs)):
            expected = 432.0 * (PHI ** (i / 12))
            assert abs(freqs[i] - expected) < 0.01

    def test_root_10_minor_scale(self):
        """√10 minor scale has correct spacing."""
        freqs = get_root_10_minor_frequencies(220.0)
        root_10 = math.sqrt(10)

        for i in range(1, len(freqs)):
            expected = 220.0 * (root_10 ** (i / 12))
            assert abs(freqs[i] - expected) < 0.01


class TestDiagnostics:
    """Test diagnostic frame generation."""

    def test_diagnostic_frame_creation(self):
        """Diagnostic frame captures all fields."""
        bio = BiometricInput(72, 60, 14, 0.45, 0.88)
        avatar = AvatarProfile(AvatarTone.CRYSTALLINE, False)
        config = ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)

        diag = simulate_diagnostics(bio, avatar, config)

        assert diag.bpm_out > 120  # High coherence = high BPM
        assert diag.loop_out > 0
        assert len(diag.mods_active) >= 3
        assert diag.avatar_tone == AvatarTone.CRYSTALLINE
        assert diag.inversion_active is False

    def test_diagnostic_with_inversion(self):
        """Diagnostic frame shows inversion state."""
        bio = BiometricInput(64, 40, 11, 0.30, 0.42)
        avatar = AvatarProfile(AvatarTone.HOLLOW, invert=True)
        config = ChamberConfig(HarmonicScale.ROOT_10_MINOR, SoundSpatialProfile.VORTEX_SPIRAL)

        diag = simulate_diagnostics(bio, avatar, config)

        assert diag.inversion_active is True
        assert diag.avatar_tone == AvatarTone.HOLLOW


class TestSessionSimulation:
    """Test session simulation scenarios from the prompt."""

    def test_harmonic_observer_session(self):
        """Test 'Harmonic Observer' session from prompt."""
        bio = BiometricInput(72, 60, 14, 0.45, 0.88)
        avatar = AvatarProfile(AvatarTone.CRYSTALLINE, invert=False)
        config = ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)

        diag = simulate_diagnostics(bio, avatar, config)

        # BPM = 60 + 0.88 * 80 = 130.4
        # Loop = φ^(1+0.88) = φ^1.88 ≈ 2.47
        assert 125 < diag.bpm_out < 135  # ≈ 130.4
        expected_loop = PHI ** 1.88
        assert abs(diag.loop_out - expected_loop) < 0.1
        assert diag.avatar_tone == AvatarTone.CRYSTALLINE
        assert diag.inversion_active is False

    def test_phase_inverter_session(self):
        """Test 'Phase Inverter' session from prompt."""
        bio = BiometricInput(64, 40, 11, 0.30, 0.42)
        avatar = AvatarProfile(AvatarTone.HOLLOW, invert=True)
        config = ChamberConfig(HarmonicScale.ROOT_10_MINOR, SoundSpatialProfile.VORTEX_SPIRAL)

        diag = simulate_diagnostics(bio, avatar, config)

        # BPM = 60 + 0.42 * 80 = 93.6
        # Loop = φ^(1+0.42) = φ^1.42 ≈ 1.95
        assert 90 < diag.bpm_out < 100  # ≈ 93.6
        expected_loop = PHI ** 1.42
        assert abs(diag.loop_out - expected_loop) < 0.1
        assert diag.avatar_tone == AvatarTone.HOLLOW
        assert diag.inversion_active is True

    def test_scalar_mirror_session(self):
        """Test 'Scalar Mirror' session from prompt."""
        bio = BiometricInput(78, 70, 16, 0.5, 0.98)
        avatar = AvatarProfile(AvatarTone.METALLIC, invert=False)
        config = ChamberConfig(
            HarmonicScale.CUSTOM,
            SoundSpatialProfile.CENTER_PULSE,
            [144, 288, 432]
        )

        diag = simulate_diagnostics(bio, avatar, config)

        # BPM = 60 + 0.98 * 80 = 138.4
        # Loop = φ^(1+0.98) = φ^1.98 ≈ 2.59
        assert 135 < diag.bpm_out < 142  # ≈ 138.4
        expected_loop = PHI ** 1.98
        assert abs(diag.loop_out - expected_loop) < 0.1
        assert diag.avatar_tone == AvatarTone.METALLIC
        assert diag.inversion_active is False


class TestSpatialProfiles:
    """Test spatial audio profile handling."""

    def test_stereo_swirl_profile(self):
        """Stereo swirl profile is preserved."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.8)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.WARM, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.STEREO_SWIRL)
        )

        assert sf.spatial_profile == SoundSpatialProfile.STEREO_SWIRL

    def test_vortex_spiral_profile(self):
        """Vortex spiral profile is preserved."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.8)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.WARM, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.VORTEX_SPIRAL)
        )

        assert sf.spatial_profile == SoundSpatialProfile.VORTEX_SPIRAL

    def test_center_pulse_profile(self):
        """Center pulse profile is preserved."""
        bio = BiometricInput(70, 50, 12, 0.4, 0.8)
        sf = generate_chamber_sound(
            bio, ScalarField(),
            AvatarProfile(AvatarTone.WARM, False),
            ChamberConfig(HarmonicScale.PHI_MAJOR, SoundSpatialProfile.CENTER_PULSE)
        )

        assert sf.spatial_profile == SoundSpatialProfile.CENTER_PULSE


class TestPhiIntegration:
    """Test phi constant integration."""

    def test_phi_constant_defined(self):
        """Phi constant is correctly defined."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_powers(self):
        """Phi powers computed correctly."""
        assert abs(PHI ** 1 - 1.618) < 0.001
        assert abs(PHI ** 2 - 2.618) < 0.001
        assert abs(PHI ** 3 - 4.236) < 0.001

    def test_phi_in_loop_timing(self):
        """Loop timing uses phi correctly."""
        loop_0 = compute_phi_loop(0.0)
        loop_1 = compute_phi_loop(1.0)

        assert abs(loop_0 - PHI ** 1) < 0.001
        assert abs(loop_1 - PHI ** 2) < 0.001


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
