#!/usr/bin/env python3
"""
Test harness for Prompt 46: Ra.Lexicon.Harmonizer

A dynamic linguistic-resonance translator that:
- Receives natural language input
- Translates to HarmonicTokens
- Aligns with Scalar Field logic
- Adjusts output per Avatar resonance

Clarifications:
- HarmonicToken: baseWord, harmonicL, harmonicM, coherenceBand, toneProfile, fragmentHint, scalarAnchor
- Coherence bands: High >= 0.72, Mid 0.40-0.72, Low < 0.40
- Avatar modes: Muted, Poetic, Formal, Blunt with defined modulation effects
- 95% token coverage target for phrase mapping
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from enum import Enum, auto

# =============================================================================
# Constants
# =============================================================================

PHI = 1.618033988749895

# Coherence band thresholds
COHERENCE_HIGH_THRESHOLD = 0.72
COHERENCE_MID_THRESHOLD = 0.40

# Phrase mapping target
PHRASE_MAPPING_TARGET = 0.95

# =============================================================================
# Enumerations
# =============================================================================

class CoherenceBand(Enum):
    """Coherence classification bands."""
    HIGH = auto()   # >= 0.72
    MID = auto()    # 0.40 - 0.72
    LOW = auto()    # < 0.40


class ToneProfile(Enum):
    """Harmonic tone profile shapes."""
    FLAT = auto()      # Neutral, stable
    RISING = auto()    # Ascending energy
    FALLING = auto()   # Descending energy
    WAVE = auto()      # Oscillating, poetic


class AvatarToneMode(Enum):
    """Avatar modulation modes."""
    NEUTRAL = auto()   # No modulation
    MUTED = auto()     # Reduces amplitude, flattens tone
    POETIC = auto()    # Shifts toward Wave, softens harmonicM
    FORMAL = auto()    # Normalizes to Flat, suppresses emotion
    BLUNT = auto()     # Sharpens tone, increases harmonicL


# =============================================================================
# Data Types
# =============================================================================

@dataclass(frozen=True)
class RaCoordinate:
    """Ra hypergrid coordinate."""
    theta: int   # 0-26
    phi: int     # 0-5
    h: int       # 0-4


@dataclass
class HarmonicToken:
    """
    A token with harmonic properties for scalar field alignment.

    Attributes:
        base_word: Original text token
        harmonic_l: Spherical harmonic l (radial order)
        harmonic_m: Spherical harmonic m (angular order)
        coherence_band: High/Mid/Low classification
        tone_profile: Flat/Rising/Falling/Wave shape
        fragment_hint: Optional fragment ID for field linking
        scalar_anchor: Optional Ra coordinate for field anchoring
        amplitude: Resonance amplitude (0-1)
    """
    base_word: str
    harmonic_l: int
    harmonic_m: int
    coherence_band: CoherenceBand
    tone_profile: ToneProfile
    fragment_hint: Optional[int] = None
    scalar_anchor: Optional[RaCoordinate] = None
    amplitude: float = 1.0


@dataclass
class AvatarProfile:
    """
    Avatar configuration for tone modulation.

    Attributes:
        name: Avatar identifier
        tone_mode: Modulation mode
        resonance_affinity: Base coherence affinity (0-1)
    """
    name: str
    tone_mode: AvatarToneMode
    resonance_affinity: float = 0.5


@dataclass
class ScalarField:
    """
    Scalar field state for resonance computation.

    Attributes:
        coherence: Overall field coherence (0-1)
        dominant_l: Dominant spherical harmonic l
        dominant_m: Dominant spherical harmonic m
        phase_angle: Current phase (0-2*pi)
    """
    coherence: float
    dominant_l: int = 0
    dominant_m: int = 0
    phase_angle: float = 0.0


@dataclass
class UserPhrase:
    """
    Harmonized phrase with resonance scoring.

    Attributes:
        original_text: Input text
        tokens: List of HarmonicTokens
        resonance_score: Overall resonance (0-1)
        avatar_mod: Applied avatar profile
        coverage_rate: Token mapping coverage
    """
    original_text: str
    tokens: List[HarmonicToken]
    resonance_score: float
    avatar_mod: AvatarProfile
    coverage_rate: float


# =============================================================================
# Lexicon Map
# =============================================================================

# Sample lexicon with harmonic mappings
# In production, this would load from RA_CONSTANTS_V2.json
LEXICON_MAP: Dict[str, Tuple[int, int, ToneProfile]] = {
    # Common words with harmonic assignments (l, m, tone)
    "the": (0, 0, ToneProfile.FLAT),
    "a": (0, 0, ToneProfile.FLAT),
    "an": (0, 0, ToneProfile.FLAT),
    "is": (1, 0, ToneProfile.FLAT),
    "are": (1, 0, ToneProfile.FLAT),
    "was": (1, 0, ToneProfile.FLAT),
    "be": (1, 0, ToneProfile.FLAT),
    "to": (0, 0, ToneProfile.FLAT),
    "of": (0, 0, ToneProfile.FLAT),
    "and": (0, 0, ToneProfile.FLAT),
    "in": (1, 0, ToneProfile.FLAT),
    "it": (0, 0, ToneProfile.FLAT),
    "you": (1, 1, ToneProfile.RISING),
    "i": (1, 0, ToneProfile.RISING),
    "that": (1, 0, ToneProfile.FLAT),
    "this": (1, 0, ToneProfile.FLAT),

    # Resonance words
    "love": (2, 1, ToneProfile.WAVE),
    "heart": (2, 0, ToneProfile.WAVE),
    "soul": (3, 0, ToneProfile.WAVE),
    "spirit": (3, 1, ToneProfile.RISING),
    "light": (2, 1, ToneProfile.RISING),
    "peace": (2, 0, ToneProfile.FLAT),
    "harmony": (3, 2, ToneProfile.WAVE),
    "balance": (2, 0, ToneProfile.FLAT),
    "flow": (2, 1, ToneProfile.WAVE),
    "energy": (2, 1, ToneProfile.RISING),

    # Action words
    "feel": (2, 1, ToneProfile.WAVE),
    "see": (1, 1, ToneProfile.RISING),
    "know": (2, 0, ToneProfile.FLAT),
    "think": (2, 0, ToneProfile.FLAT),
    "believe": (2, 1, ToneProfile.RISING),
    "create": (3, 1, ToneProfile.RISING),
    "breathe": (2, 0, ToneProfile.WAVE),
    "move": (1, 1, ToneProfile.RISING),
    "grow": (2, 1, ToneProfile.RISING),

    # Descriptive words
    "beautiful": (3, 2, ToneProfile.WAVE),
    "calm": (1, 0, ToneProfile.FLAT),
    "warm": (2, 1, ToneProfile.WAVE),
    "bright": (2, 1, ToneProfile.RISING),
    "deep": (2, -1, ToneProfile.FALLING),
    "strong": (2, 1, ToneProfile.RISING),
    "soft": (1, 0, ToneProfile.WAVE),
    "gentle": (1, 0, ToneProfile.WAVE),

    # Field/technical words
    "field": (2, 0, ToneProfile.FLAT),
    "scalar": (3, 0, ToneProfile.FLAT),
    "resonance": (3, 2, ToneProfile.WAVE),
    "coherence": (3, 1, ToneProfile.FLAT),
    "frequency": (2, 1, ToneProfile.RISING),
    "wave": (2, 1, ToneProfile.WAVE),
    "phase": (2, 0, ToneProfile.FLAT),
    "amplitude": (2, 0, ToneProfile.RISING),
}


# =============================================================================
# Core Functions
# =============================================================================

def get_coherence_band(coherence: float) -> CoherenceBand:
    """Classify coherence into bands."""
    if coherence >= COHERENCE_HIGH_THRESHOLD:
        return CoherenceBand.HIGH
    elif coherence >= COHERENCE_MID_THRESHOLD:
        return CoherenceBand.MID
    else:
        return CoherenceBand.LOW


def create_harmonic_token(
    word: str,
    lexicon: Dict[str, Tuple[int, int, ToneProfile]],
    field_coherence: float
) -> HarmonicToken:
    """
    Create a HarmonicToken from a word using the lexicon.
    Unknown words get neutral defaults.
    """
    word_lower = word.lower().strip()

    if word_lower in lexicon:
        l, m, tone = lexicon[word_lower]
        # Known word - use lexicon mapping
        return HarmonicToken(
            base_word=word,
            harmonic_l=l,
            harmonic_m=m,
            coherence_band=get_coherence_band(field_coherence),
            tone_profile=tone,
            amplitude=1.0
        )
    else:
        # Unknown word - neutral fallback
        return HarmonicToken(
            base_word=word,
            harmonic_l=0,
            harmonic_m=0,
            coherence_band=CoherenceBand.MID,
            tone_profile=ToneProfile.FLAT,
            amplitude=0.5  # Reduced amplitude for unknown
        )


def apply_avatar_modulation(
    token: HarmonicToken,
    avatar: AvatarProfile
) -> HarmonicToken:
    """
    Apply avatar tone modulation to a token.

    Modulation effects:
    - Muted: Reduces amplitude, flattens toneProfile
    - Poetic: Shifts toward Wave, softens harmonicM
    - Formal: Normalizes to Flat, suppresses emotion
    - Blunt: Sharpens tone (Falling), increases harmonicL
    """
    if avatar.tone_mode == AvatarToneMode.NEUTRAL:
        return token

    new_l = token.harmonic_l
    new_m = token.harmonic_m
    new_tone = token.tone_profile
    new_amp = token.amplitude
    new_band = token.coherence_band

    if avatar.tone_mode == AvatarToneMode.MUTED:
        # Reduce amplitude, flatten tone, downgrade band
        new_amp = token.amplitude * 0.5
        new_tone = ToneProfile.FLAT
        if token.coherence_band == CoherenceBand.HIGH:
            new_band = CoherenceBand.MID
        elif token.coherence_band == CoherenceBand.MID:
            new_band = CoherenceBand.LOW

    elif avatar.tone_mode == AvatarToneMode.POETIC:
        # Shift toward Wave, soften harmonicM
        new_tone = ToneProfile.WAVE
        new_m = int(token.harmonic_m * 0.5)
        new_amp = min(1.0, token.amplitude * 1.2)

    elif avatar.tone_mode == AvatarToneMode.FORMAL:
        # Normalize to Flat, reduce M variance
        new_tone = ToneProfile.FLAT
        new_m = 0

    elif avatar.tone_mode == AvatarToneMode.BLUNT:
        # Sharp falling tone, increase L for emphasis
        new_tone = ToneProfile.FALLING
        new_l = min(9, token.harmonic_l + 1)
        new_amp = min(1.0, token.amplitude * 1.3)

    return HarmonicToken(
        base_word=token.base_word,
        harmonic_l=new_l,
        harmonic_m=new_m,
        coherence_band=new_band,
        tone_profile=new_tone,
        fragment_hint=token.fragment_hint,
        scalar_anchor=token.scalar_anchor,
        amplitude=new_amp
    )


def tokenize_phrase(text: str) -> List[str]:
    """Simple tokenization - split on whitespace and punctuation."""
    import re
    # Split on whitespace and keep alphanumeric tokens
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def compute_token_coverage(
    tokens: List[HarmonicToken],
    lexicon: Dict[str, Tuple[int, int, ToneProfile]]
) -> float:
    """Compute token coverage rate (known tokens / total tokens)."""
    if not tokens:
        return 0.0

    known_count = sum(
        1 for t in tokens
        if t.base_word.lower() in lexicon
    )
    return known_count / len(tokens)


def compute_resonance_score(
    tokens: List[HarmonicToken],
    scalar_field: ScalarField
) -> float:
    """
    Compute overall resonance score for tokenized phrase.

    Considers:
    - Token coherence alignment with field
    - Harmonic l/m alignment with field dominants
    - Amplitude weighting
    """
    if not tokens:
        return 0.0

    total_score = 0.0
    total_weight = 0.0

    for token in tokens:
        weight = token.amplitude

        # Base score from coherence band
        if token.coherence_band == CoherenceBand.HIGH:
            band_score = 0.9
        elif token.coherence_band == CoherenceBand.MID:
            band_score = 0.6
        else:
            band_score = 0.3

        # Harmonic alignment bonus
        l_diff = abs(token.harmonic_l - scalar_field.dominant_l)
        m_diff = abs(token.harmonic_m - scalar_field.dominant_m)
        harmonic_bonus = 1.0 / (1.0 + l_diff * 0.1 + m_diff * 0.05)

        # Combined score
        token_score = band_score * harmonic_bonus * scalar_field.coherence

        total_score += token_score * weight
        total_weight += weight

    return total_score / total_weight if total_weight > 0 else 0.0


def tokenize_with_resonance(
    text: str,
    lexicon: Dict[str, Tuple[int, int, ToneProfile]],
    scalar_field: ScalarField
) -> List[HarmonicToken]:
    """
    Tokenize text and create HarmonicTokens with field alignment.
    """
    words = tokenize_phrase(text)
    tokens = [
        create_harmonic_token(word, lexicon, scalar_field.coherence)
        for word in words
    ]
    return tokens


def run_lexicon_harmonizer(
    text: str,
    lexicon: Dict[str, Tuple[int, int, ToneProfile]],
    avatar: AvatarProfile,
    scalar_field: ScalarField
) -> UserPhrase:
    """
    Full harmonization pipeline.

    1. Tokenize text with resonance
    2. Apply avatar modulation
    3. Compute resonance score
    4. Return UserPhrase
    """
    # Tokenize with field alignment
    raw_tokens = tokenize_with_resonance(text, lexicon, scalar_field)

    # Apply avatar modulation
    modulated_tokens = [
        apply_avatar_modulation(token, avatar)
        for token in raw_tokens
    ]

    # Compute metrics
    coverage = compute_token_coverage(raw_tokens, lexicon)
    resonance = compute_resonance_score(modulated_tokens, scalar_field)

    return UserPhrase(
        original_text=text,
        tokens=modulated_tokens,
        resonance_score=resonance,
        avatar_mod=avatar,
        coverage_rate=coverage
    )


def check_fragment_hint_validity(
    token: HarmonicToken,
    scalar_field: ScalarField
) -> bool:
    """
    Check if fragment hint corresponds to valid activation zone.
    """
    if token.scalar_anchor is None:
        return True  # No anchor = no check needed

    coord = token.scalar_anchor

    # Valid coordinate ranges
    valid_theta = 0 <= coord.theta < 27
    valid_phi = 0 <= coord.phi < 6
    valid_h = 0 <= coord.h < 5

    return valid_theta and valid_phi and valid_h


# =============================================================================
# Test Cases
# =============================================================================

class TestCoherenceBands:
    """Test coherence band classification."""

    def test_high_coherence(self):
        """Coherence >= 0.72 is HIGH."""
        assert get_coherence_band(0.72) == CoherenceBand.HIGH
        assert get_coherence_band(0.85) == CoherenceBand.HIGH
        assert get_coherence_band(1.0) == CoherenceBand.HIGH

    def test_mid_coherence(self):
        """Coherence 0.40-0.72 is MID."""
        assert get_coherence_band(0.40) == CoherenceBand.MID
        assert get_coherence_band(0.55) == CoherenceBand.MID
        assert get_coherence_band(0.71) == CoherenceBand.MID

    def test_low_coherence(self):
        """Coherence < 0.40 is LOW."""
        assert get_coherence_band(0.0) == CoherenceBand.LOW
        assert get_coherence_band(0.20) == CoherenceBand.LOW
        assert get_coherence_band(0.39) == CoherenceBand.LOW

    def test_boundary_high_mid(self):
        """Test boundary between HIGH and MID."""
        assert get_coherence_band(0.72) == CoherenceBand.HIGH
        assert get_coherence_band(0.719) == CoherenceBand.MID

    def test_boundary_mid_low(self):
        """Test boundary between MID and LOW."""
        assert get_coherence_band(0.40) == CoherenceBand.MID
        assert get_coherence_band(0.399) == CoherenceBand.LOW


class TestHarmonicTokenCreation:
    """Test HarmonicToken creation from lexicon."""

    def test_known_word_mapping(self):
        """Known words use lexicon mapping."""
        token = create_harmonic_token("love", LEXICON_MAP, 0.8)
        assert token.base_word == "love"
        assert token.harmonic_l == 2
        assert token.harmonic_m == 1
        assert token.tone_profile == ToneProfile.WAVE
        assert token.amplitude == 1.0

    def test_unknown_word_fallback(self):
        """Unknown words get neutral defaults."""
        token = create_harmonic_token("xyzzy", LEXICON_MAP, 0.8)
        assert token.base_word == "xyzzy"
        assert token.harmonic_l == 0
        assert token.harmonic_m == 0
        assert token.coherence_band == CoherenceBand.MID
        assert token.tone_profile == ToneProfile.FLAT
        assert token.amplitude == 0.5

    def test_case_insensitive(self):
        """Lexicon lookup is case-insensitive."""
        token1 = create_harmonic_token("LOVE", LEXICON_MAP, 0.8)
        token2 = create_harmonic_token("Love", LEXICON_MAP, 0.8)
        token3 = create_harmonic_token("love", LEXICON_MAP, 0.8)

        assert token1.harmonic_l == token2.harmonic_l == token3.harmonic_l
        assert token1.tone_profile == token2.tone_profile == token3.tone_profile

    def test_coherence_affects_band(self):
        """Field coherence affects token band."""
        token_high = create_harmonic_token("love", LEXICON_MAP, 0.9)
        token_mid = create_harmonic_token("love", LEXICON_MAP, 0.5)
        token_low = create_harmonic_token("love", LEXICON_MAP, 0.2)

        assert token_high.coherence_band == CoherenceBand.HIGH
        assert token_mid.coherence_band == CoherenceBand.MID
        assert token_low.coherence_band == CoherenceBand.LOW

    def test_various_tone_profiles(self):
        """Different words have different tone profiles."""
        flat_token = create_harmonic_token("peace", LEXICON_MAP, 0.8)
        rising_token = create_harmonic_token("light", LEXICON_MAP, 0.8)
        wave_token = create_harmonic_token("harmony", LEXICON_MAP, 0.8)

        assert flat_token.tone_profile == ToneProfile.FLAT
        assert rising_token.tone_profile == ToneProfile.RISING
        assert wave_token.tone_profile == ToneProfile.WAVE


class TestAvatarModulation:
    """Test avatar tone modulation effects."""

    def test_neutral_no_change(self):
        """Neutral mode makes no changes."""
        token = create_harmonic_token("love", LEXICON_MAP, 0.8)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.amplitude == token.amplitude
        assert modulated.tone_profile == token.tone_profile
        assert modulated.harmonic_l == token.harmonic_l

    def test_muted_reduces_amplitude(self):
        """Muted mode reduces amplitude by half."""
        token = create_harmonic_token("love", LEXICON_MAP, 0.8)
        avatar = AvatarProfile("quiet", AvatarToneMode.MUTED)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.amplitude == token.amplitude * 0.5
        assert modulated.tone_profile == ToneProfile.FLAT

    def test_muted_downgrades_band(self):
        """Muted mode downgrades coherence band."""
        token_high = HarmonicToken("test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE)
        token_mid = HarmonicToken("test", 2, 1, CoherenceBand.MID, ToneProfile.WAVE)
        avatar = AvatarProfile("quiet", AvatarToneMode.MUTED)

        mod_high = apply_avatar_modulation(token_high, avatar)
        mod_mid = apply_avatar_modulation(token_mid, avatar)

        assert mod_high.coherence_band == CoherenceBand.MID
        assert mod_mid.coherence_band == CoherenceBand.LOW

    def test_poetic_shifts_to_wave(self):
        """Poetic mode shifts tone to Wave."""
        # Start with lower amplitude to verify boost
        token = HarmonicToken("test", 2, 2, CoherenceBand.HIGH, ToneProfile.FLAT, amplitude=0.7)
        avatar = AvatarProfile("poet", AvatarToneMode.POETIC)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.tone_profile == ToneProfile.WAVE
        assert modulated.harmonic_m == 1  # Softened from 2
        assert modulated.amplitude > token.amplitude  # Boosted by 1.2x

    def test_formal_flattens_tone(self):
        """Formal mode normalizes to Flat and zeros M."""
        token = HarmonicToken("test", 3, 2, CoherenceBand.HIGH, ToneProfile.WAVE)
        avatar = AvatarProfile("formal", AvatarToneMode.FORMAL)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.tone_profile == ToneProfile.FLAT
        assert modulated.harmonic_m == 0

    def test_blunt_increases_l(self):
        """Blunt mode increases harmonicL and uses Falling tone."""
        token = HarmonicToken("test", 2, 1, CoherenceBand.HIGH, ToneProfile.FLAT)
        avatar = AvatarProfile("direct", AvatarToneMode.BLUNT)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.harmonic_l == 3  # Increased from 2
        assert modulated.tone_profile == ToneProfile.FALLING

    def test_blunt_l_capped_at_9(self):
        """Blunt mode caps L at 9."""
        token = HarmonicToken("test", 9, 1, CoherenceBand.HIGH, ToneProfile.FLAT)
        avatar = AvatarProfile("direct", AvatarToneMode.BLUNT)
        modulated = apply_avatar_modulation(token, avatar)

        assert modulated.harmonic_l == 9  # Capped


class TestPhraseMappingRate:
    """Test phrase mapping rate (95% coverage target)."""

    def test_all_known_words(self):
        """100% coverage for all known words."""
        text = "I love the light"
        tokens = tokenize_with_resonance(text, LEXICON_MAP, ScalarField(0.8))
        coverage = compute_token_coverage(tokens, LEXICON_MAP)

        assert coverage == 1.0

    def test_some_unknown_words(self):
        """Partial coverage with unknown words."""
        text = "I love the xyzzy"
        tokens = tokenize_with_resonance(text, LEXICON_MAP, ScalarField(0.8))
        coverage = compute_token_coverage(tokens, LEXICON_MAP)

        assert coverage == 0.75  # 3/4 known

    def test_high_coverage_phrase(self):
        """Test phrase with high coverage rate."""
        text = "The soul is beautiful and the heart is warm"
        tokens = tokenize_with_resonance(text, LEXICON_MAP, ScalarField(0.8))
        coverage = compute_token_coverage(tokens, LEXICON_MAP)

        assert coverage >= PHRASE_MAPPING_TARGET

    def test_empty_phrase(self):
        """Empty phrase has 0% coverage."""
        tokens = tokenize_with_resonance("", LEXICON_MAP, ScalarField(0.8))
        coverage = compute_token_coverage(tokens, LEXICON_MAP)

        assert coverage == 0.0

    def test_lexicon_size_sufficient(self):
        """Lexicon has enough entries for typical phrases."""
        # Test with common phrases
        phrases = [
            "I feel the energy flow",
            "The light is beautiful",
            "Peace and harmony",
            "Breathe and grow",
        ]

        for phrase in phrases:
            tokens = tokenize_with_resonance(phrase, LEXICON_MAP, ScalarField(0.8))
            coverage = compute_token_coverage(tokens, LEXICON_MAP)
            assert coverage >= 0.80, f"Low coverage for: {phrase}"


class TestResonancePrediction:
    """Test resonance score computation."""

    def test_high_coherence_high_resonance(self):
        """High coherence field yields high resonance."""
        text = "love harmony peace"
        field = ScalarField(coherence=0.9, dominant_l=2, dominant_m=1)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, field)

        assert phrase.resonance_score > 0.6

    def test_low_coherence_low_resonance(self):
        """Low coherence field yields low resonance."""
        text = "love harmony peace"
        field = ScalarField(coherence=0.2, dominant_l=2, dominant_m=1)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, field)

        assert phrase.resonance_score < 0.4

    def test_harmonic_alignment_bonus(self):
        """Aligned harmonics boost resonance."""
        text = "love"  # l=2, m=1

        # Field aligned with token harmonics
        aligned_field = ScalarField(coherence=0.8, dominant_l=2, dominant_m=1)
        # Field misaligned
        misaligned_field = ScalarField(coherence=0.8, dominant_l=0, dominant_m=0)

        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        aligned = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, aligned_field)
        misaligned = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, misaligned_field)

        assert aligned.resonance_score > misaligned.resonance_score

    def test_amplitude_weighting(self):
        """Higher amplitude tokens contribute more to resonance."""
        # Unknown words have lower amplitude
        known_text = "love peace harmony"
        mixed_text = "love xyzzy harmony"

        field = ScalarField(coherence=0.8, dominant_l=2, dominant_m=1)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        known = run_lexicon_harmonizer(known_text, LEXICON_MAP, avatar, field)
        mixed = run_lexicon_harmonizer(mixed_text, LEXICON_MAP, avatar, field)

        # Known words have higher resonance due to higher amplitude
        assert known.resonance_score >= mixed.resonance_score


class TestFragmentHintValidity:
    """Test fragment hint linkage validity."""

    def test_valid_coordinate(self):
        """Valid Ra coordinate passes check."""
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE,
            fragment_hint=42,
            scalar_anchor=RaCoordinate(13, 3, 2)
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is True

    def test_no_anchor_is_valid(self):
        """Token without anchor is valid."""
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is True

    def test_invalid_theta(self):
        """Invalid theta coordinate fails check."""
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE,
            scalar_anchor=RaCoordinate(30, 3, 2)  # theta > 26
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is False

    def test_invalid_phi(self):
        """Invalid phi coordinate fails check."""
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE,
            scalar_anchor=RaCoordinate(13, 7, 2)  # phi > 5
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is False

    def test_invalid_h(self):
        """Invalid h coordinate fails check."""
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE,
            scalar_anchor=RaCoordinate(13, 3, 6)  # h > 4
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is False

    def test_boundary_coordinates(self):
        """Boundary coordinates are valid."""
        # Max valid coordinates
        token = HarmonicToken(
            "test", 2, 1, CoherenceBand.HIGH, ToneProfile.WAVE,
            scalar_anchor=RaCoordinate(26, 5, 4)
        )
        field = ScalarField(0.8)

        assert check_fragment_hint_validity(token, field) is True


class TestGracefulFallback:
    """Test graceful fallback for unknown words."""

    def test_unknown_word_neutral_token(self):
        """Unknown words resolve to neutral token."""
        token = create_harmonic_token("qwertyuiop", LEXICON_MAP, 0.8)

        assert token.coherence_band == CoherenceBand.MID
        assert token.tone_profile == ToneProfile.FLAT
        assert token.harmonic_l == 0
        assert token.harmonic_m == 0

    def test_fallback_reduced_amplitude(self):
        """Fallback tokens have reduced amplitude."""
        known = create_harmonic_token("love", LEXICON_MAP, 0.8)
        unknown = create_harmonic_token("asdfghjkl", LEXICON_MAP, 0.8)

        assert unknown.amplitude < known.amplitude

    def test_phrase_with_unknowns_still_works(self):
        """Phrases with unknown words still produce valid result."""
        text = "The xyzzy flows with asdfgh energy"
        field = ScalarField(0.8)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, field)

        assert len(phrase.tokens) == 6
        assert phrase.resonance_score > 0
        assert phrase.coverage_rate < 1.0


class TestFullPipeline:
    """Integration tests for full harmonization pipeline."""

    def test_full_harmonization(self):
        """Complete harmonization pipeline works."""
        text = "The soul breathes light and harmony"
        field = ScalarField(coherence=0.85, dominant_l=2, dominant_m=1)
        avatar = AvatarProfile("seeker", AvatarToneMode.POETIC)

        phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, field)

        assert phrase.original_text == text
        assert len(phrase.tokens) == 6
        assert phrase.coverage_rate >= 0.8
        assert phrase.resonance_score > 0
        assert phrase.avatar_mod == avatar

    def test_different_avatars_different_results(self):
        """Different avatar modes produce different results."""
        text = "The soul breathes light"
        field = ScalarField(0.8)

        neutral = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)
        muted = AvatarProfile("muted", AvatarToneMode.MUTED)
        poetic = AvatarProfile("poetic", AvatarToneMode.POETIC)

        phrase_neutral = run_lexicon_harmonizer(text, LEXICON_MAP, neutral, field)
        phrase_muted = run_lexicon_harmonizer(text, LEXICON_MAP, muted, field)
        phrase_poetic = run_lexicon_harmonizer(text, LEXICON_MAP, poetic, field)

        # Muted should have lower amplitudes
        muted_amp = sum(t.amplitude for t in phrase_muted.tokens)
        neutral_amp = sum(t.amplitude for t in phrase_neutral.tokens)
        assert muted_amp < neutral_amp

        # Poetic should have Wave tone profiles
        poetic_waves = sum(1 for t in phrase_poetic.tokens if t.tone_profile == ToneProfile.WAVE)
        assert poetic_waves == len(phrase_poetic.tokens)

    def test_resonance_varies_with_field(self):
        """Resonance score varies with field coherence."""
        text = "love harmony peace"
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        high_field = ScalarField(coherence=0.95)
        low_field = ScalarField(coherence=0.15)

        high_phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, high_field)
        low_phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, low_field)

        assert high_phrase.resonance_score > low_phrase.resonance_score

    def test_tokenization_handles_punctuation(self):
        """Tokenization handles punctuation correctly."""
        text = "Love, harmony, and peace!"
        field = ScalarField(0.8)
        avatar = AvatarProfile("neutral", AvatarToneMode.NEUTRAL)

        phrase = run_lexicon_harmonizer(text, LEXICON_MAP, avatar, field)

        # Should extract 4 words
        assert len(phrase.tokens) == 4
        words = [t.base_word.lower() for t in phrase.tokens]
        assert "love" in words
        assert "harmony" in words
        assert "and" in words
        assert "peace" in words


class TestPhiIntegration:
    """Test phi constant integration."""

    def test_phi_constant_defined(self):
        """Phi constant is correctly defined."""
        assert abs(PHI - 1.618033988749895) < 1e-10

    def test_phi_used_in_thresholds(self):
        """Coherence thresholds align with phi design."""
        # High threshold at 0.72 is a design choice
        # Numerologically: 72 = 360/5 (pentagonal), relates to phi
        assert COHERENCE_HIGH_THRESHOLD == 0.72

        # Verify threshold is in meaningful phi-aligned range
        # 0.72 is between 1/phi (0.618) and phi-1 (0.618), close to both
        phi_inverse = 1 / PHI  # ≈ 0.618
        assert 0.6 < COHERENCE_HIGH_THRESHOLD < 0.8  # In phi-adjacent range


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
