"""
Test Suite for Ra.BioField.TuningMap (Prompt 69)

Per-user healing frequency tuning maps based on therapeutic frequency
profiles and biometric input. References RIFE, Lakhovsky, Kaali-Beck sources.

Architect Clarifications:
- Fixed organ frequency maps as base (Liver→40Hz, Heart→7.83Hz), with dynamic modulation
- Waveform selection: Sine→gentle, Square→RIFE-pulsed, Triangle→ramping
- Chamber modulation: both closed-loop and read-only modes supported
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict

# Constants
PHI = 1.618033988749895
SCHUMANN_FUNDAMENTAL = 7.83  # Hz

# Reference frequency maps (from RIFE_LAKHOVSKY_SPECIFICATIONS)
ORGAN_FREQUENCY_MAP: Dict[str, List[float]] = {
    "heart": [7.83, 10.0, 40.0],           # Schumann + cardiac bands
    "liver": [40.0, 317.0, 727.0],         # Rife liver frequencies
    "kidney": [20.0, 440.0, 625.0],        # Rife kidney frequencies
    "brain": [7.83, 10.0, 14.0, 40.0],     # Theta, Alpha, Beta, Gamma
    "lungs": [72.0, 125.0, 727.0],         # Rife respiratory
    "spine": [9.6, 10.0, 96.0],            # Spinal resonance
    "blood": [1550.0, 880.0, 787.0],       # Kaali-Beck + Rife pathogen
    "immune": [727.0, 787.0, 880.0],       # Rife immune support
}

# Waveform types
class Waveform(Enum):
    SINE = auto()       # Gentle regulation
    SQUARE = auto()     # RIFE/radionic pulsed delivery
    TRIANGLE = auto()   # Ramping field tests
    SAWTOOTH = auto()   # Asymmetric charge/discharge


class PhaseLock(Enum):
    FREE = auto()           # No phase lock
    SCHUMANN = auto()       # Lock to 7.83Hz fundamental
    CARDIAC = auto()        # Lock to heart rate
    RESPIRATORY = auto()    # Lock to breath cycle
    CUSTOM = auto()         # Custom phase reference


class ChamberForm(Enum):
    TOROIDAL = auto()
    DODECAHEDRAL = auto()
    SPHERICAL = auto()
    PYRAMIDAL = auto()


class ChamberMode(Enum):
    CLOSED_LOOP = auto()    # Modulates frequency delivery
    READ_ONLY = auto()      # Reflects alignment phase only


class WaveformSelector(Enum):
    USER_SPECIFIED = auto()
    PER_ORGAN = auto()
    COHERENCE_DRIVEN = auto()


@dataclass
class BioState:
    """Real-time biometric state."""
    hrv: float              # Heart rate variability (0-1)
    pulse: float            # BPM
    skin_conductance: float # Microsiemens
    coherence: float        # Overall coherence (0-1)
    breath_rate: float      # Breaths per minute
    stress_index: float     # 0-1 stress level


@dataclass
class BioTuningWave:
    """Single tuning wave in the map."""
    user_id: uuid.UUID
    target_organ: Optional[str]
    frequency_set: List[float]
    waveform_type: Waveform
    resonance_phase: PhaseLock
    amplitude: float = 1.0
    duty_cycle: float = 0.5  # For square waves


@dataclass
class BioTuningMap:
    """Collection of tuning waves for a user."""
    waves: List[BioTuningWave]
    user_id: uuid.UUID
    chamber_form: ChamberForm
    chamber_mode: ChamberMode
    total_power: float = 0.0


def get_organ_base_frequencies(organ: str) -> List[float]:
    """Get base frequencies for an organ from reference maps."""
    return ORGAN_FREQUENCY_MAP.get(organ.lower(), [SCHUMANN_FUNDAMENTAL])


def modulate_frequency_by_biometrics(
    base_freq: float,
    bio_state: BioState
) -> float:
    """Modulate base frequency based on biometric state."""
    # Coherence scales frequency slightly (±10%)
    coherence_factor = 0.9 + (bio_state.coherence * 0.2)

    # HRV influences stability - low HRV reduces modulation
    hrv_stability = 0.5 + (bio_state.hrv * 0.5)

    # Stress reduces effective frequency
    stress_damping = 1.0 - (bio_state.stress_index * 0.1)

    return base_freq * coherence_factor * hrv_stability * stress_damping


def select_waveform_for_organ(organ: str, bio_state: BioState) -> Waveform:
    """Select appropriate waveform based on organ and state."""
    # High coherence -> sine (gentle)
    if bio_state.coherence > 0.8:
        return Waveform.SINE

    # Specific organ mappings
    organ_waveforms = {
        "heart": Waveform.SINE,      # Always gentle for heart
        "blood": Waveform.SQUARE,    # RIFE-style for blood purification
        "immune": Waveform.SQUARE,   # Pulsed for immune activation
        "brain": Waveform.SINE,      # Gentle for neural
        "liver": Waveform.TRIANGLE,  # Ramping for detox
    }

    if organ.lower() in organ_waveforms:
        return organ_waveforms[organ.lower()]

    # Default based on coherence
    if bio_state.coherence < 0.4:
        return Waveform.TRIANGLE  # Ramping to build coherence
    return Waveform.SINE


def select_waveform_coherence_driven(bio_state: BioState) -> Waveform:
    """Select waveform purely based on coherence level."""
    if bio_state.coherence >= 0.8:
        return Waveform.SINE
    elif bio_state.coherence >= 0.5:
        return Waveform.TRIANGLE
    else:
        return Waveform.SQUARE


def select_phase_lock(bio_state: BioState, chamber_form: ChamberForm) -> PhaseLock:
    """Select phase lock mode based on state and chamber."""
    # High HRV -> cardiac lock
    if bio_state.hrv > 0.7:
        return PhaseLock.CARDIAC

    # Toroidal chambers work well with Schumann
    if chamber_form == ChamberForm.TOROIDAL:
        return PhaseLock.SCHUMANN

    # Low stress -> respiratory lock
    if bio_state.stress_index < 0.3:
        return PhaseLock.RESPIRATORY

    return PhaseLock.FREE


def compute_chamber_modulation(
    frequency: float,
    bio_state: BioState,
    chamber_form: ChamberForm
) -> float:
    """Compute chamber modulation factor for closed-loop mode."""
    # Chamber geometry affects resonance
    geometry_factors = {
        ChamberForm.TOROIDAL: PHI,
        ChamberForm.DODECAHEDRAL: 1.2,
        ChamberForm.SPHERICAL: 1.0,
        ChamberForm.PYRAMIDAL: 1.414,  # sqrt(2)
    }

    geo_factor = geometry_factors.get(chamber_form, 1.0)

    # Modulate based on coherence alignment
    alignment = bio_state.coherence * bio_state.hrv

    return frequency * (1.0 + (alignment * (geo_factor - 1.0) * 0.1))


def create_tuning_wave(
    user_id: uuid.UUID,
    organ: Optional[str],
    bio_state: BioState,
    chamber_form: ChamberForm,
    waveform_selector: WaveformSelector = WaveformSelector.PER_ORGAN
) -> BioTuningWave:
    """Create a single tuning wave for an organ."""
    # Get base frequencies
    if organ:
        base_freqs = get_organ_base_frequencies(organ)
    else:
        base_freqs = [SCHUMANN_FUNDAMENTAL]

    # Modulate frequencies
    modulated_freqs = [
        modulate_frequency_by_biometrics(f, bio_state)
        for f in base_freqs
    ]

    # Select waveform
    if waveform_selector == WaveformSelector.USER_SPECIFIED:
        waveform = Waveform.SINE  # Default for user-specified
    elif waveform_selector == WaveformSelector.COHERENCE_DRIVEN:
        waveform = select_waveform_coherence_driven(bio_state)
    else:  # PER_ORGAN
        waveform = select_waveform_for_organ(organ or "general", bio_state)

    # Select phase lock
    phase_lock = select_phase_lock(bio_state, chamber_form)

    # Calculate amplitude based on coherence
    amplitude = 0.5 + (bio_state.coherence * 0.5)

    return BioTuningWave(
        user_id=user_id,
        target_organ=organ,
        frequency_set=modulated_freqs,
        waveform_type=waveform,
        resonance_phase=phase_lock,
        amplitude=amplitude
    )


def generate_bio_tuning_map(
    bio_state: BioState,
    chamber_form: ChamberForm,
    target_organs: Optional[List[str]] = None,
    chamber_mode: ChamberMode = ChamberMode.CLOSED_LOOP
) -> BioTuningMap:
    """Generate complete bio tuning map for a user."""
    user_id = uuid.uuid4()

    # Default to full-body if no specific organs
    if target_organs is None:
        target_organs = ["heart", "brain", "immune"]

    waves = []
    total_power = 0.0

    for organ in target_organs:
        wave = create_tuning_wave(
            user_id=user_id,
            organ=organ,
            bio_state=bio_state,
            chamber_form=chamber_form
        )

        # Apply chamber modulation if closed-loop
        if chamber_mode == ChamberMode.CLOSED_LOOP:
            wave.frequency_set = [
                compute_chamber_modulation(f, bio_state, chamber_form)
                for f in wave.frequency_set
            ]

        waves.append(wave)
        total_power += wave.amplitude * len(wave.frequency_set)

    return BioTuningMap(
        waves=waves,
        user_id=user_id,
        chamber_form=chamber_form,
        chamber_mode=chamber_mode,
        total_power=total_power
    )


def validate_frequency_in_therapeutic_range(freq: float) -> bool:
    """Check if frequency is within known therapeutic ranges."""
    therapeutic_ranges = [
        (0.5, 4.0),      # Delta
        (4.0, 8.0),      # Theta
        (7.0, 14.0),     # Alpha + Schumann
        (14.0, 30.0),    # Beta
        (30.0, 100.0),   # Gamma
        (100.0, 1000.0), # RIFE low
        (1000.0, 2000.0), # RIFE high
    ]
    return any(low <= freq <= high for low, high in therapeutic_ranges)


# ============== TESTS ==============

class TestOrganFrequencyMapping:
    """Tests for organ-specific frequency extraction."""

    def test_heart_frequencies_include_schumann(self):
        """Heart frequencies should include Schumann fundamental."""
        freqs = get_organ_base_frequencies("heart")
        assert SCHUMANN_FUNDAMENTAL in freqs

    def test_liver_frequencies_include_rife_band(self):
        """Liver frequencies should include RIFE liver frequency."""
        freqs = get_organ_base_frequencies("liver")
        assert 40.0 in freqs

    def test_blood_frequencies_include_kaali_beck(self):
        """Blood frequencies should include Kaali-Beck range."""
        freqs = get_organ_base_frequencies("blood")
        # Kaali-Beck blood electrification around 1550Hz
        assert any(f > 1000 for f in freqs)

    def test_unknown_organ_defaults_to_schumann(self):
        """Unknown organs default to Schumann fundamental."""
        freqs = get_organ_base_frequencies("unknown_organ")
        assert freqs == [SCHUMANN_FUNDAMENTAL]

    def test_case_insensitive_lookup(self):
        """Organ lookup should be case-insensitive."""
        freqs_lower = get_organ_base_frequencies("heart")
        freqs_upper = get_organ_base_frequencies("HEART")
        assert freqs_lower == freqs_upper

    def test_brain_includes_brainwave_bands(self):
        """Brain frequencies should span brainwave bands."""
        freqs = get_organ_base_frequencies("brain")
        # Should include theta (7.83), alpha (10), beta (14), gamma (40)
        assert len(freqs) >= 4
        assert any(7 <= f <= 8 for f in freqs)   # Theta/Schumann
        assert any(38 <= f <= 42 for f in freqs) # Gamma


class TestBiometricModulation:
    """Tests for frequency modulation based on biometrics."""

    def test_high_coherence_increases_frequency(self):
        """High coherence should increase effective frequency."""
        bio_high = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                           coherence=0.9, breath_rate=12, stress_index=0.1)
        bio_low = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                          coherence=0.1, breath_rate=12, stress_index=0.1)

        freq_high = modulate_frequency_by_biometrics(100.0, bio_high)
        freq_low = modulate_frequency_by_biometrics(100.0, bio_low)

        assert freq_high > freq_low

    def test_high_stress_reduces_frequency(self):
        """High stress should reduce effective frequency."""
        bio_calm = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                           coherence=0.5, breath_rate=12, stress_index=0.1)
        bio_stressed = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                               coherence=0.5, breath_rate=12, stress_index=0.9)

        freq_calm = modulate_frequency_by_biometrics(100.0, bio_calm)
        freq_stressed = modulate_frequency_by_biometrics(100.0, bio_stressed)

        assert freq_calm > freq_stressed

    def test_low_hrv_reduces_modulation_range(self):
        """Low HRV should reduce the modulation stability factor."""
        bio_stable = BioState(hrv=0.9, pulse=60, skin_conductance=5.0,
                             coherence=0.5, breath_rate=12, stress_index=0.5)
        bio_unstable = BioState(hrv=0.1, pulse=60, skin_conductance=5.0,
                               coherence=0.5, breath_rate=12, stress_index=0.5)

        freq_stable = modulate_frequency_by_biometrics(100.0, bio_stable)
        freq_unstable = modulate_frequency_by_biometrics(100.0, bio_unstable)

        assert freq_stable > freq_unstable

    def test_modulation_keeps_frequency_positive(self):
        """Modulation should never produce negative frequencies."""
        bio_worst = BioState(hrv=0.0, pulse=60, skin_conductance=5.0,
                            coherence=0.0, breath_rate=12, stress_index=1.0)

        freq = modulate_frequency_by_biometrics(100.0, bio_worst)
        assert freq > 0


class TestWaveformSelection:
    """Tests for waveform type selection logic."""

    def test_heart_always_gets_sine(self):
        """Heart should always use gentle sine waves."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        waveform = select_waveform_for_organ("heart", bio)
        assert waveform == Waveform.SINE

    def test_blood_uses_square_for_rife(self):
        """Blood purification should use RIFE-style square waves."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        waveform = select_waveform_for_organ("blood", bio)
        assert waveform == Waveform.SQUARE

    def test_high_coherence_overrides_to_sine(self):
        """Very high coherence should override to sine."""
        bio = BioState(hrv=0.9, pulse=60, skin_conductance=5.0,
                      coherence=0.85, breath_rate=12, stress_index=0.1)

        waveform = select_waveform_for_organ("blood", bio)
        assert waveform == Waveform.SINE

    def test_liver_uses_triangle_for_ramping(self):
        """Liver detox uses triangle (ramping) waves."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        waveform = select_waveform_for_organ("liver", bio)
        assert waveform == Waveform.TRIANGLE

    def test_coherence_driven_selection(self):
        """Coherence-driven selection follows coherence thresholds."""
        bio_high = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                           coherence=0.85, breath_rate=12, stress_index=0.5)
        bio_mid = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                          coherence=0.6, breath_rate=12, stress_index=0.5)
        bio_low = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                          coherence=0.3, breath_rate=12, stress_index=0.5)

        assert select_waveform_coherence_driven(bio_high) == Waveform.SINE
        assert select_waveform_coherence_driven(bio_mid) == Waveform.TRIANGLE
        assert select_waveform_coherence_driven(bio_low) == Waveform.SQUARE


class TestPhaseLockSelection:
    """Tests for phase lock mode selection."""

    def test_high_hrv_selects_cardiac_lock(self):
        """High HRV should select cardiac phase lock."""
        bio = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        phase = select_phase_lock(bio, ChamberForm.SPHERICAL)
        assert phase == PhaseLock.CARDIAC

    def test_toroidal_chamber_prefers_schumann(self):
        """Toroidal chambers should prefer Schumann lock."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        phase = select_phase_lock(bio, ChamberForm.TOROIDAL)
        assert phase == PhaseLock.SCHUMANN

    def test_low_stress_allows_respiratory_lock(self):
        """Low stress enables respiratory phase lock."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.2)

        phase = select_phase_lock(bio, ChamberForm.SPHERICAL)
        assert phase == PhaseLock.RESPIRATORY


class TestChamberModulation:
    """Tests for chamber feedback modulation."""

    def test_toroidal_applies_phi_factor(self):
        """Toroidal chamber should apply phi-based modulation."""
        bio = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                      coherence=0.8, breath_rate=12, stress_index=0.2)

        base_freq = 100.0
        modulated = compute_chamber_modulation(base_freq, bio, ChamberForm.TOROIDAL)

        # Should be increased by phi-influenced factor
        assert modulated > base_freq

    def test_spherical_minimal_modulation(self):
        """Spherical chamber should have minimal geometry factor."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        base_freq = 100.0
        modulated = compute_chamber_modulation(base_freq, bio, ChamberForm.SPHERICAL)

        # Spherical has factor 1.0, so less modulation
        assert abs(modulated - base_freq) < abs(
            compute_chamber_modulation(base_freq, bio, ChamberForm.TOROIDAL) - base_freq
        )

    def test_low_coherence_reduces_modulation(self):
        """Low coherence should reduce modulation effect."""
        bio_high = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                           coherence=0.9, breath_rate=12, stress_index=0.1)
        bio_low = BioState(hrv=0.2, pulse=60, skin_conductance=5.0,
                          coherence=0.1, breath_rate=12, stress_index=0.9)

        base_freq = 100.0
        mod_high = compute_chamber_modulation(base_freq, bio_high, ChamberForm.TOROIDAL)
        mod_low = compute_chamber_modulation(base_freq, bio_low, ChamberForm.TOROIDAL)

        assert abs(mod_high - base_freq) > abs(mod_low - base_freq)


class TestTuningWaveCreation:
    """Tests for individual tuning wave creation."""

    def test_creates_wave_with_user_id(self):
        """Wave should have assigned user ID."""
        user_id = uuid.uuid4()
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        wave = create_tuning_wave(user_id, "heart", bio, ChamberForm.TOROIDAL)

        assert wave.user_id == user_id

    def test_wave_frequencies_are_modulated(self):
        """Wave frequencies should be modulated from base."""
        user_id = uuid.uuid4()
        bio = BioState(hrv=0.9, pulse=60, skin_conductance=5.0,
                      coherence=0.9, breath_rate=12, stress_index=0.1)

        base_freqs = get_organ_base_frequencies("heart")
        wave = create_tuning_wave(user_id, "heart", bio, ChamberForm.TOROIDAL)

        # Frequencies should be different from base (modulated)
        assert wave.frequency_set != base_freqs

    def test_amplitude_scales_with_coherence(self):
        """Wave amplitude should scale with coherence."""
        user_id = uuid.uuid4()
        bio_high = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                           coherence=0.9, breath_rate=12, stress_index=0.5)
        bio_low = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                          coherence=0.1, breath_rate=12, stress_index=0.5)

        wave_high = create_tuning_wave(user_id, "heart", bio_high, ChamberForm.TOROIDAL)
        wave_low = create_tuning_wave(user_id, "heart", bio_low, ChamberForm.TOROIDAL)

        assert wave_high.amplitude > wave_low.amplitude


class TestBioTuningMapGeneration:
    """Tests for complete tuning map generation."""

    def test_generates_map_for_multiple_organs(self):
        """Map should contain waves for all target organs."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        tuning_map = generate_bio_tuning_map(
            bio, ChamberForm.TOROIDAL,
            target_organs=["heart", "liver", "brain"]
        )

        assert len(tuning_map.waves) == 3
        organs = [w.target_organ for w in tuning_map.waves]
        assert "heart" in organs
        assert "liver" in organs
        assert "brain" in organs

    def test_closed_loop_applies_chamber_modulation(self):
        """Closed-loop mode should apply chamber modulation."""
        bio = BioState(hrv=0.8, pulse=60, skin_conductance=5.0,
                      coherence=0.8, breath_rate=12, stress_index=0.2)

        map_closed = generate_bio_tuning_map(
            bio, ChamberForm.TOROIDAL,
            target_organs=["heart"],
            chamber_mode=ChamberMode.CLOSED_LOOP
        )
        map_readonly = generate_bio_tuning_map(
            bio, ChamberForm.TOROIDAL,
            target_organs=["heart"],
            chamber_mode=ChamberMode.READ_ONLY
        )

        # Closed loop should have different (modulated) frequencies
        assert map_closed.waves[0].frequency_set != map_readonly.waves[0].frequency_set

    def test_total_power_calculated(self):
        """Total power should be sum of wave amplitudes * frequency counts."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        tuning_map = generate_bio_tuning_map(
            bio, ChamberForm.SPHERICAL,
            target_organs=["heart", "brain"]
        )

        expected_power = sum(
            w.amplitude * len(w.frequency_set)
            for w in tuning_map.waves
        )
        assert abs(tuning_map.total_power - expected_power) < 0.01

    def test_default_organs_if_none_specified(self):
        """Should use default organ set if none specified."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        tuning_map = generate_bio_tuning_map(bio, ChamberForm.SPHERICAL)

        assert len(tuning_map.waves) > 0

    def test_all_frequencies_in_therapeutic_range(self):
        """All generated frequencies should be in therapeutic ranges."""
        bio = BioState(hrv=0.5, pulse=60, skin_conductance=5.0,
                      coherence=0.5, breath_rate=12, stress_index=0.5)

        tuning_map = generate_bio_tuning_map(
            bio, ChamberForm.TOROIDAL,
            target_organs=["heart", "liver", "brain", "blood"]
        )

        for wave in tuning_map.waves:
            for freq in wave.frequency_set:
                assert validate_frequency_in_therapeutic_range(freq), \
                    f"Frequency {freq} not in therapeutic range"


class TestTherapeuticRangeValidation:
    """Tests for frequency range validation."""

    def test_schumann_in_range(self):
        """Schumann frequency should be valid."""
        assert validate_frequency_in_therapeutic_range(7.83)

    def test_rife_frequencies_in_range(self):
        """RIFE frequencies should be valid."""
        rife_freqs = [727.0, 787.0, 880.0, 1550.0]
        for freq in rife_freqs:
            assert validate_frequency_in_therapeutic_range(freq)

    def test_brainwave_bands_in_range(self):
        """Brainwave band frequencies should be valid."""
        brainwave_freqs = [2.0, 6.0, 10.0, 20.0, 40.0]  # Delta, Theta, Alpha, Beta, Gamma
        for freq in brainwave_freqs:
            assert validate_frequency_in_therapeutic_range(freq)

    def test_out_of_range_rejected(self):
        """Frequencies outside therapeutic ranges should be rejected."""
        assert not validate_frequency_in_therapeutic_range(0.1)   # Too low
        assert not validate_frequency_in_therapeutic_range(5000.0)  # Too high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
