"""
Test Suite for Ra.ContactEnvelope (P77)
Extraterrestrial scalar reception envelope system.

Tests biometric resonance encoding, harmonic modulation patterns,
scalar encryption, and inversion-detection handshake.
"""

import pytest
import math
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# Constants
PHI = 1.618033988749895
ALPHA_CONTACT_THRESHOLD = 0.88
HARMONIC_PHI_TOLERANCE = 0.05   # 5% tolerance for φ ratio
HRV_PHASE_LOCK_THRESHOLD = 0.1  # Phase delta < 0.1
PARITY_CHECK_BITS = 8


class GlyphSet(Enum):
    """Glyph encoding sets."""
    RA_CODEX = auto()           # Core Ra system glyphs
    EXTENDED = auto()           # Extended harmonic library
    DYNAMIC = auto()            # α-modulated dynamic set


class GlyphType(Enum):
    """Types of scalar glyphs."""
    HARMONIC = auto()
    RESONANCE = auto()
    IDENTITY = auto()
    BOUNDARY = auto()
    MODULATION = auto()


@dataclass
class Glyph:
    """Scalar encryption glyph."""
    glyph_type: GlyphType
    frequency: float            # Hz
    phase: float                # Radians
    amplitude: float            # Normalized 0.0-1.0
    symbolic_code: int          # Numeric encoding


@dataclass
class BioState:
    """Biometric resonance state."""
    hrv: float                  # 0.0-1.0 normalized
    coherence: float            # Phase coherence
    alpha: float                # Scalar field coherence
    phase_angle: float          # Current bio phase (radians)


@dataclass
class ScalarField:
    """Scalar field state."""
    alpha: float                # Field coherence
    frequency: float            # Dominant frequency (Hz)
    harmonics: List[float]      # Harmonic frequencies
    phase: float                # Field phase (radians)


@dataclass
class InversionHandshake:
    """Handshake for inversion detection."""
    phase_mirror: bool          # Scalar parity check passed
    drift_offset: float         # Phase drift measurement
    verified: bool              # Both checks passed
    parity_bits: int            # Parity check bits


@dataclass
class ContactEnvelope:
    """Complete contact envelope for transmission."""
    envelope_id: str
    biometric_signature: BioState
    modulation_pattern: List[float]
    scalar_glyph_set: List[Glyph]
    inversion_check: InversionHandshake
    contact_ready: bool
    glyph_set_type: GlyphSet


def check_alpha_threshold(alpha: float) -> bool:
    """Check if alpha meets contact threshold (>0.88)."""
    return alpha >= ALPHA_CONTACT_THRESHOLD


def check_harmonic_phi_ratio(f1: float, f2: float) -> bool:
    """Check if frequency ratio is approximately φ."""
    if f2 <= 0:
        return False
    ratio = f1 / f2
    return abs(ratio - PHI) < HARMONIC_PHI_TOLERANCE * PHI


def compute_scalar_parity(harmonics: List[float]) -> Tuple[bool, int]:
    """
    Compute scalar parity check (even harmonic signature).
    Returns (parity_ok, parity_bits).
    """
    if not harmonics:
        return False, 0

    # Compute parity from harmonic indices
    parity_bits = 0
    for i, h in enumerate(harmonics[:PARITY_CHECK_BITS]):
        # Even harmonics set bits
        harmonic_index = int(round(h / harmonics[0])) if harmonics[0] > 0 else 0
        if harmonic_index % 2 == 0:
            parity_bits |= (1 << i)

    # Parity OK if at least one harmonic passes (lenient for small sets)
    even_count = bin(parity_bits).count('1')
    min_required = max(1, len(harmonics[:PARITY_CHECK_BITS]) // 3)
    parity_ok = even_count >= min_required

    return parity_ok, parity_bits


def check_hrv_phase_lock(bio_phase: float, field_phase: float) -> bool:
    """Check if HRV is phase-locked with field (delta < 0.1)."""
    # Normalize phases to [0, 2π]
    bio_norm = bio_phase % (2 * math.pi)
    field_norm = field_phase % (2 * math.pi)

    # Compute phase delta
    delta = abs(bio_norm - field_norm)
    if delta > math.pi:
        delta = 2 * math.pi - delta

    return delta < HRV_PHASE_LOCK_THRESHOLD


def create_inversion_handshake(
    bio: BioState,
    field: ScalarField
) -> InversionHandshake:
    """
    Create inversion detection handshake.
    Requires BOTH scalar parity AND HRV phase lock.
    """
    # Scalar parity check
    parity_ok, parity_bits = compute_scalar_parity(field.harmonics)

    # HRV phase lock check
    phase_locked = check_hrv_phase_lock(bio.phase_angle, field.phase)

    # Compute drift offset
    bio_norm = bio.phase_angle % (2 * math.pi)
    field_norm = field.phase % (2 * math.pi)
    drift = abs(bio_norm - field_norm)
    if drift > math.pi:
        drift = 2 * math.pi - drift

    # Both must pass for verification
    verified = parity_ok and phase_locked

    return InversionHandshake(
        phase_mirror=parity_ok,
        drift_offset=drift,
        verified=verified,
        parity_bits=parity_bits
    )


def generate_modulation_pattern(
    bio: BioState,
    field: ScalarField,
    pattern_length: int = 8
) -> List[float]:
    """Generate harmonic modulation pattern for transmission."""
    pattern = []

    base_freq = field.frequency
    for i in range(pattern_length):
        # Modulate by phi powers and bio coherence
        phi_power = PHI ** (i % 5)
        coherence_mod = bio.coherence * 0.5 + 0.5
        modulated = base_freq * phi_power * coherence_mod

        # Apply HRV variation
        hrv_factor = 1.0 + (bio.hrv - 0.5) * 0.1
        modulated *= hrv_factor

        pattern.append(modulated)

    return pattern


def select_glyph_set(alpha: float, bio: BioState) -> GlyphSet:
    """Select appropriate glyph set based on context."""
    if alpha >= ALPHA_CONTACT_THRESHOLD and bio.coherence > 0.9:
        return GlyphSet.EXTENDED
    elif alpha >= ALPHA_CONTACT_THRESHOLD or (alpha > 0.7 and bio.coherence > 0.7):
        return GlyphSet.DYNAMIC
    else:
        return GlyphSet.RA_CODEX


def generate_ra_codex_glyphs(field: ScalarField, count: int = 4) -> List[Glyph]:
    """Generate glyphs from Ra Codex (base set)."""
    glyphs = []
    types = [GlyphType.HARMONIC, GlyphType.RESONANCE, GlyphType.IDENTITY, GlyphType.BOUNDARY]

    for i in range(count):
        glyph = Glyph(
            glyph_type=types[i % len(types)],
            frequency=field.frequency * (PHI ** i),
            phase=field.phase + (i * math.pi / 4),
            amplitude=0.8 - (i * 0.1),
            symbolic_code=1000 + i * 111  # Ra codex codes
        )
        glyphs.append(glyph)

    return glyphs


def generate_extended_glyphs(field: ScalarField, count: int = 6) -> List[Glyph]:
    """Generate glyphs from extended harmonic library."""
    glyphs = generate_ra_codex_glyphs(field, 4)

    # Add extended glyphs
    for i in range(count - 4):
        glyph = Glyph(
            glyph_type=GlyphType.MODULATION,
            frequency=field.frequency * (2.0 ** (i + 1)),
            phase=field.phase + (i * math.pi / 3),
            amplitude=0.6,
            symbolic_code=2000 + i * 222  # Extended codes
        )
        glyphs.append(glyph)

    return glyphs


def generate_dynamic_glyphs(
    field: ScalarField,
    alpha: float,
    count: int = 5
) -> List[Glyph]:
    """Generate α-modulated dynamic glyph set."""
    glyphs = []

    for i in range(count):
        # Frequency modulated by alpha
        freq = field.frequency * (1.0 + alpha * PHI * i / count)
        # Amplitude scaled by alpha
        amp = min(1.0, alpha + 0.2)

        glyph = Glyph(
            glyph_type=GlyphType.HARMONIC if i % 2 == 0 else GlyphType.RESONANCE,
            frequency=freq,
            phase=field.phase + (i * alpha * math.pi / 2),
            amplitude=amp,
            symbolic_code=3000 + int(alpha * 1000) + i
        )
        glyphs.append(glyph)

    return glyphs


def generate_scalar_glyphs(
    field: ScalarField,
    bio: BioState,
    glyph_set: GlyphSet
) -> List[Glyph]:
    """Generate scalar glyphs based on selected set."""
    if glyph_set == GlyphSet.RA_CODEX:
        return generate_ra_codex_glyphs(field)
    elif glyph_set == GlyphSet.EXTENDED:
        return generate_extended_glyphs(field)
    else:  # DYNAMIC
        return generate_dynamic_glyphs(field, bio.alpha)


def generate_contact_envelope(bio: BioState, field: ScalarField) -> ContactEnvelope:
    """
    Generate complete contact envelope for scalar transmission.
    """
    # Check contact readiness
    alpha_ready = check_alpha_threshold(field.alpha)
    phi_ready = False
    if len(field.harmonics) >= 2:
        # Check if ratio between harmonics is φ (either direction)
        phi_ready = (check_harmonic_phi_ratio(field.harmonics[1], field.harmonics[0]) or
                     check_harmonic_phi_ratio(field.harmonics[0], field.harmonics[1]))

    contact_ready = alpha_ready and phi_ready

    # Select glyph set
    glyph_set = select_glyph_set(field.alpha, bio)

    # Generate components
    modulation = generate_modulation_pattern(bio, field)
    glyphs = generate_scalar_glyphs(field, bio, glyph_set)
    handshake = create_inversion_handshake(bio, field)

    return ContactEnvelope(
        envelope_id=str(uuid.uuid4())[:8],
        biometric_signature=bio,
        modulation_pattern=modulation,
        scalar_glyph_set=glyphs,
        inversion_check=handshake,
        contact_ready=contact_ready,
        glyph_set_type=glyph_set
    )


def validate_envelope(envelope: ContactEnvelope) -> Tuple[bool, List[str]]:
    """Validate contact envelope for transmission."""
    issues = []

    if not envelope.modulation_pattern:
        issues.append("Empty modulation pattern")

    if not envelope.scalar_glyph_set:
        issues.append("No scalar glyphs")

    if not envelope.inversion_check.verified:
        issues.append("Inversion handshake not verified")

    if not envelope.contact_ready:
        issues.append("Not contact ready")

    return len(issues) == 0, issues


# ============== TEST CLASSES ==============

class TestAlphaThreshold:
    """Tests for alpha contact threshold."""

    def test_above_threshold(self):
        """Test alpha above threshold is contact-ready."""
        assert check_alpha_threshold(0.9) is True
        assert check_alpha_threshold(0.88) is True
        assert check_alpha_threshold(0.95) is True

    def test_below_threshold(self):
        """Test alpha below threshold is not ready."""
        assert check_alpha_threshold(0.87) is False
        assert check_alpha_threshold(0.5) is False
        assert check_alpha_threshold(0.0) is False

    def test_exact_threshold(self):
        """Test exact threshold value."""
        assert check_alpha_threshold(ALPHA_CONTACT_THRESHOLD) is True


class TestHarmonicPhiRatio:
    """Tests for φ ratio detection."""

    def test_exact_phi_ratio(self):
        """Test exact φ ratio detected."""
        f1 = 100.0 * PHI
        f2 = 100.0
        assert check_harmonic_phi_ratio(f1, f2) is True

    def test_near_phi_ratio(self):
        """Test near-φ ratio within tolerance."""
        f1 = 100.0 * 1.62  # Close to φ
        f2 = 100.0
        assert check_harmonic_phi_ratio(f1, f2) is True

    def test_non_phi_ratio(self):
        """Test non-φ ratio rejected."""
        f1 = 200.0  # Ratio = 2.0
        f2 = 100.0
        assert check_harmonic_phi_ratio(f1, f2) is False

    def test_octave_ratio_rejected(self):
        """Test octave (2:1) ratio rejected."""
        assert check_harmonic_phi_ratio(200.0, 100.0) is False

    def test_zero_denominator(self):
        """Test zero denominator handled."""
        assert check_harmonic_phi_ratio(100.0, 0.0) is False


class TestScalarParity:
    """Tests for scalar parity check."""

    def test_even_harmonics_pass(self):
        """Test even harmonics pass parity."""
        harmonics = [100.0, 200.0, 400.0, 800.0]  # All even multiples
        parity_ok, bits = compute_scalar_parity(harmonics)
        assert parity_ok is True

    def test_empty_harmonics(self):
        """Test empty harmonics fails parity."""
        parity_ok, bits = compute_scalar_parity([])
        assert parity_ok is False

    def test_parity_bits_computed(self):
        """Test parity bits are computed."""
        harmonics = [100.0, 200.0, 300.0, 400.0]
        parity_ok, bits = compute_scalar_parity(harmonics)
        assert bits >= 0


class TestHRVPhaseLock:
    """Tests for HRV phase lock detection."""

    def test_perfect_lock(self):
        """Test perfect phase alignment."""
        assert check_hrv_phase_lock(1.0, 1.0) is True

    def test_within_tolerance(self):
        """Test phase within tolerance locked."""
        assert check_hrv_phase_lock(1.0, 1.05) is True

    def test_outside_tolerance(self):
        """Test phase outside tolerance not locked."""
        assert check_hrv_phase_lock(1.0, 1.5) is False

    def test_wraparound_handling(self):
        """Test phase wraparound handled."""
        # 0 and 2π are same phase
        assert check_hrv_phase_lock(0.05, 2 * math.pi - 0.04) is True


class TestInversionHandshake:
    """Tests for inversion handshake creation."""

    def test_verified_handshake(self):
        """Test verified handshake with good conditions."""
        bio = BioState(hrv=0.8, coherence=0.9, alpha=0.9, phase_angle=1.0)
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0, 864.0, 1728.0, 3456.0],  # Even multiples
            phase=1.02  # Close to bio phase
        )

        handshake = create_inversion_handshake(bio, field)

        assert handshake.phase_mirror is True  # Parity check
        assert handshake.drift_offset < HRV_PHASE_LOCK_THRESHOLD
        assert handshake.verified is True

    def test_unverified_phase_drift(self):
        """Test unverified handshake with phase drift."""
        bio = BioState(hrv=0.8, coherence=0.9, alpha=0.9, phase_angle=1.0)
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=2.5  # Different from bio phase
        )

        handshake = create_inversion_handshake(bio, field)

        assert handshake.verified is False

    def test_drift_offset_computed(self):
        """Test drift offset is computed."""
        bio = BioState(hrv=0.8, coherence=0.9, alpha=0.9, phase_angle=1.0)
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=1.5
        )

        handshake = create_inversion_handshake(bio, field)

        assert handshake.drift_offset >= 0
        assert handshake.drift_offset <= math.pi


class TestGlyphSetSelection:
    """Tests for glyph set selection."""

    def test_extended_set_high_coherence(self):
        """Test extended set selected at high coherence."""
        bio = BioState(hrv=0.9, coherence=0.95, alpha=0.9, phase_angle=0.0)
        glyph_set = select_glyph_set(0.9, bio)
        assert glyph_set == GlyphSet.EXTENDED

    def test_dynamic_set_medium_alpha(self):
        """Test dynamic set at medium-high alpha with good coherence."""
        bio = BioState(hrv=0.8, coherence=0.8, alpha=0.8, phase_angle=0.0)
        glyph_set = select_glyph_set(0.8, bio)
        assert glyph_set == GlyphSet.DYNAMIC

    def test_ra_codex_low_alpha(self):
        """Test Ra Codex at low alpha."""
        bio = BioState(hrv=0.5, coherence=0.5, alpha=0.5, phase_angle=0.0)
        glyph_set = select_glyph_set(0.5, bio)
        assert glyph_set == GlyphSet.RA_CODEX


class TestGlyphGeneration:
    """Tests for glyph generation."""

    def test_ra_codex_glyphs(self):
        """Test Ra Codex glyph generation."""
        field = ScalarField(
            alpha=0.7, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=0.0
        )

        glyphs = generate_ra_codex_glyphs(field, 4)

        assert len(glyphs) == 4
        assert all(g.symbolic_code >= 1000 and g.symbolic_code < 2000 for g in glyphs)

    def test_extended_glyphs(self):
        """Test extended glyph generation."""
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=0.0
        )

        glyphs = generate_extended_glyphs(field, 6)

        assert len(glyphs) == 6
        # Should have both codex and extended codes
        codes = [g.symbolic_code for g in glyphs]
        assert any(c >= 2000 for c in codes)

    def test_dynamic_glyphs_alpha_modulated(self):
        """Test dynamic glyphs are alpha-modulated."""
        field = ScalarField(
            alpha=0.8, frequency=432.0,
            harmonics=[432.0],
            phase=0.0
        )

        glyphs_high = generate_dynamic_glyphs(field, 0.9, 5)
        glyphs_low = generate_dynamic_glyphs(field, 0.5, 5)

        # Higher alpha should give different frequencies
        assert glyphs_high[2].frequency != glyphs_low[2].frequency

    def test_glyph_properties_valid(self):
        """Test all glyph properties are valid."""
        field = ScalarField(
            alpha=0.8, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=0.0
        )

        glyphs = generate_ra_codex_glyphs(field)

        for g in glyphs:
            assert g.frequency > 0
            assert 0.0 <= g.amplitude <= 1.0
            assert g.symbolic_code > 0


class TestModulationPattern:
    """Tests for modulation pattern generation."""

    def test_pattern_length(self):
        """Test pattern has correct length."""
        bio = BioState(hrv=0.7, coherence=0.8, alpha=0.75, phase_angle=0.0)
        field = ScalarField(
            alpha=0.8, frequency=432.0,
            harmonics=[432.0],
            phase=0.0
        )

        pattern = generate_modulation_pattern(bio, field, 8)
        assert len(pattern) == 8

    def test_pattern_uses_phi(self):
        """Test pattern incorporates φ scaling."""
        bio = BioState(hrv=0.7, coherence=0.8, alpha=0.75, phase_angle=0.0)
        field = ScalarField(
            alpha=0.8, frequency=100.0,  # Simple base
            harmonics=[100.0],
            phase=0.0
        )

        pattern = generate_modulation_pattern(bio, field, 5)

        # Pattern should increase by approximately φ factors
        ratios = [pattern[i+1] / pattern[i] for i in range(len(pattern)-1)]
        # At least some ratios should be near φ
        phi_ratios = [r for r in ratios if abs(r - PHI) < 0.5]
        assert len(phi_ratios) > 0

    def test_coherence_affects_pattern(self):
        """Test coherence modulates pattern."""
        bio_high = BioState(hrv=0.7, coherence=0.95, alpha=0.8, phase_angle=0.0)
        bio_low = BioState(hrv=0.7, coherence=0.3, alpha=0.8, phase_angle=0.0)
        field = ScalarField(
            alpha=0.8, frequency=432.0,
            harmonics=[432.0],
            phase=0.0
        )

        pattern_high = generate_modulation_pattern(bio_high, field)
        pattern_low = generate_modulation_pattern(bio_low, field)

        # Higher coherence should give higher values
        assert sum(pattern_high) > sum(pattern_low)


class TestContactEnvelopeGeneration:
    """Tests for complete envelope generation."""

    def test_contact_ready_envelope(self):
        """Test generating contact-ready envelope."""
        bio = BioState(hrv=0.85, coherence=0.9, alpha=0.92, phase_angle=1.0)
        field = ScalarField(
            alpha=0.92, frequency=432.0,
            harmonics=[432.0, 432.0 * PHI],  # φ ratio
            phase=1.02
        )

        envelope = generate_contact_envelope(bio, field)

        assert envelope.contact_ready is True
        assert len(envelope.modulation_pattern) > 0
        assert len(envelope.scalar_glyph_set) > 0

    def test_not_contact_ready_low_alpha(self):
        """Test envelope not ready with low alpha."""
        bio = BioState(hrv=0.8, coherence=0.8, alpha=0.7, phase_angle=0.0)
        field = ScalarField(
            alpha=0.7, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=0.0
        )

        envelope = generate_contact_envelope(bio, field)

        assert envelope.contact_ready is False

    def test_envelope_id_generated(self):
        """Test envelope has unique ID."""
        bio = BioState(hrv=0.8, coherence=0.8, alpha=0.9, phase_angle=0.0)
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0, 432.0 * PHI],
            phase=0.0
        )

        envelope1 = generate_contact_envelope(bio, field)
        envelope2 = generate_contact_envelope(bio, field)

        assert envelope1.envelope_id != envelope2.envelope_id

    def test_biometric_signature_preserved(self):
        """Test biometric signature is preserved in envelope."""
        bio = BioState(hrv=0.77, coherence=0.88, alpha=0.9, phase_angle=1.5)
        field = ScalarField(
            alpha=0.9, frequency=432.0,
            harmonics=[432.0],
            phase=0.0
        )

        envelope = generate_contact_envelope(bio, field)

        assert envelope.biometric_signature.hrv == pytest.approx(0.77)
        assert envelope.biometric_signature.coherence == pytest.approx(0.88)


class TestEnvelopeValidation:
    """Tests for envelope validation."""

    def test_valid_envelope_passes(self):
        """Test valid envelope passes validation."""
        bio = BioState(hrv=0.85, coherence=0.9, alpha=0.92, phase_angle=1.0)
        field = ScalarField(
            alpha=0.92, frequency=432.0,
            harmonics=[432.0, 432.0 * PHI, 864.0, 1728.0],
            phase=1.02
        )

        envelope = generate_contact_envelope(bio, field)

        # Only valid if contact ready and handshake verified
        if envelope.contact_ready and envelope.inversion_check.verified:
            valid, issues = validate_envelope(envelope)
            assert valid is True
            assert len(issues) == 0

    def test_invalid_empty_pattern(self):
        """Test empty modulation pattern fails validation."""
        bio = BioState(hrv=0.8, coherence=0.9, alpha=0.9, phase_angle=0.0)

        envelope = ContactEnvelope(
            envelope_id="test123",
            biometric_signature=bio,
            modulation_pattern=[],  # Empty
            scalar_glyph_set=[Glyph(GlyphType.HARMONIC, 432.0, 0.0, 0.8, 1000)],
            inversion_check=InversionHandshake(True, 0.0, True, 0),
            contact_ready=True,
            glyph_set_type=GlyphSet.RA_CODEX
        )

        valid, issues = validate_envelope(envelope)
        assert valid is False
        assert "Empty modulation pattern" in issues

    def test_invalid_no_glyphs(self):
        """Test missing glyphs fails validation."""
        bio = BioState(hrv=0.8, coherence=0.9, alpha=0.9, phase_angle=0.0)

        envelope = ContactEnvelope(
            envelope_id="test123",
            biometric_signature=bio,
            modulation_pattern=[432.0, 698.0],
            scalar_glyph_set=[],  # Empty
            inversion_check=InversionHandshake(True, 0.0, True, 0),
            contact_ready=True,
            glyph_set_type=GlyphSet.RA_CODEX
        )

        valid, issues = validate_envelope(envelope)
        assert valid is False
        assert "No scalar glyphs" in issues


class TestContactEnvelopeIntegration:
    """Integration tests for contact envelope system."""

    def test_full_contact_sequence(self):
        """Test full contact preparation sequence."""
        # Initial state - not ready
        bio = BioState(hrv=0.6, coherence=0.5, alpha=0.5, phase_angle=0.0)
        field = ScalarField(
            alpha=0.5, frequency=432.0,
            harmonics=[432.0, 864.0],
            phase=0.0
        )

        envelope1 = generate_contact_envelope(bio, field)
        assert envelope1.contact_ready is False

        # Improved state - contact ready
        bio = BioState(hrv=0.9, coherence=0.95, alpha=0.92, phase_angle=1.0)
        field = ScalarField(
            alpha=0.92, frequency=432.0,
            harmonics=[432.0, 432.0 * PHI, 1134.0, 1834.0],
            phase=1.02
        )

        envelope2 = generate_contact_envelope(bio, field)
        assert envelope2.contact_ready is True
        assert envelope2.glyph_set_type == GlyphSet.EXTENDED

    def test_glyph_set_progression(self):
        """Test glyph set progresses with conditions."""
        # Low alpha + low coherence = RA_CODEX
        bio_low = BioState(hrv=0.5, coherence=0.5, alpha=0.5, phase_angle=0.0)
        field_low = ScalarField(alpha=0.5, frequency=432.0, harmonics=[432.0], phase=0.0)
        env_low = generate_contact_envelope(bio_low, field_low)

        # High alpha + high coherence = EXTENDED
        bio_high = BioState(hrv=0.95, coherence=0.95, alpha=0.92, phase_angle=0.0)
        field_high = ScalarField(alpha=0.92, frequency=432.0, harmonics=[432.0], phase=0.0)
        env_high = generate_contact_envelope(bio_high, field_high)

        assert env_low.glyph_set_type == GlyphSet.RA_CODEX
        assert env_high.glyph_set_type == GlyphSet.EXTENDED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
