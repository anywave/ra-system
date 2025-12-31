"""
Prompt 39: Ra-Frequency Mapping Table

Comprehensive reference table linking Ra scalar field harmonics
to Rife, Tesla, Keely, and Chakra/Neural frequency systems.

Codex References:
- Ra.Constants.Harmonics: Core Ra field definitions
- RIFE_LAKHOVSKY_SPECIFICATIONS.md: Rife frequencies
- KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md: Keely ratios
"""

import pytest
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895

# Solfeggio frequencies (canonical)
SOLFEGGIO = {
    "UT": 396,   # Liberating guilt and fear
    "RE": 417,   # Undoing situations and facilitating change
    "MI": 528,   # Transformation and miracles (DNA repair)
    "FA": 639,   # Connecting/relationships
    "SOL": 741,  # Awakening intuition
    "LA": 852,   # Returning to spiritual order
}

# Brainwave bands
BRAINWAVE_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# Chakra frequency associations
CHAKRA_FREQUENCIES = {
    "root": 396,
    "sacral": 417,
    "solar_plexus": 528,
    "heart": 639,
    "throat": 741,
    "third_eye": 852,
    "crown": 963,
}

# Keely 3:6:9 base multiplier
KEELY_BASE = 111


# ============================================================================
# Types
# ============================================================================

class NeuralBand(Enum):
    """Brainwave frequency bands."""
    DELTA = auto()
    THETA = auto()
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


class ChakraCenter(Enum):
    """Chakra energy centers."""
    ROOT = auto()
    SACRAL = auto()
    SOLAR_PLEXUS = auto()
    HEART = auto()
    THROAT = auto()
    THIRD_EYE = auto()
    CROWN = auto()


class MatchType(Enum):
    """Type of frequency match."""
    EXACT = auto()
    NEAREST = auto()
    SYMBOLIC = auto()
    RANGE = auto()


@dataclass
class FrequencyMapping:
    """Single frequency mapping entry."""
    ra_label: str
    hz_values: List[float]
    rife_equiv: List[float]
    tesla_range: Tuple[float, float]
    keely_ratio: str
    chakra_state: str
    neural_band: str
    match_type: MatchType
    notes: str


@dataclass
class FrequencyMappingTable:
    """Complete frequency mapping table."""
    mappings: List[FrequencyMapping]
    sources_used: Dict[str, str]


# ============================================================================
# Ra Harmonic Definitions
# ============================================================================

# Canonical Ra harmonics (Solfeggio + Keely + Fragment resonance)
RA_HARMONICS = {
    "Ra.Tonic.Root": {
        "hz": [396],
        "description": "Root grounding, guilt liberation",
    },
    "Ra.Tonic.Sacral": {
        "hz": [417],
        "description": "Change facilitation, creative flow",
    },
    "Ra.Tonic.Heart": {
        "hz": [528],
        "description": "DNA repair, transformation, miracles",
    },
    "Ra.Tonic.Connection": {
        "hz": [639],
        "description": "Relationship harmony, heart coherence",
    },
    "Ra.Tonic.Intuition": {
        "hz": [741],
        "description": "Awakening intuition, expression",
    },
    "Ra.Tonic.Spirit": {
        "hz": [852],
        "description": "Spiritual order, third eye activation",
    },
    "Ra.Tonic.Crown": {
        "hz": [963],
        "description": "Crown activation, cosmic unity",
    },
    "Ra.Keely.Base": {
        "hz": [111],
        "description": "Keely base unit, cellular regeneration",
    },
    "Ra.Keely.Triple": {
        "hz": [333],
        "description": "Keely 3x, ascended masters frequency",
    },
    "Ra.Keely.Hex": {
        "hz": [666],
        "description": "Keely 6x, carbon/matter frequency",
    },
    "Ra.Keely.Nines": {
        "hz": [999],
        "description": "Keely 9x, completion/transcendence",
    },
    "Ra.Tesla.369.Low": {
        "hz": [3.0, 6.0, 9.0],
        "description": "Tesla 3-6-9 low range, brain entrainment",
    },
    "Ra.Tesla.369.Mid": {
        "hz": [30.0, 60.0, 90.0],
        "description": "Tesla 3-6-9 mid range, gamma activation",
    },
    "Ra.Phi.Base": {
        "hz": [PHI * 100, PHI * 200, PHI * 300],
        "description": "Phi-scaled harmonics, golden ratio resonance",
    },
    "Ra.Schumann": {
        "hz": [7.83],
        "description": "Earth resonance, grounding frequency",
    },
}


# ============================================================================
# Rife Frequency Database
# ============================================================================

RIFE_FREQUENCIES = {
    "general_healing": [728, 787, 880, 5000, 10000],
    "immune_boost": [465, 660, 690, 727.5, 787],
    "pain_relief": [304, 320, 10000],
    "relaxation": [10, 40, 7.83],
    "dna_repair": [528],
    "cellular_regen": [111, 222, 333, 444],
}


# ============================================================================
# Tesla Frequency Ranges
# ============================================================================

TESLA_RANGES = {
    "low_impulse": (1.0, 10.0),
    "mid_impulse": (10.0, 100.0),
    "high_impulse": (100.0, 1000.0),
    "scalar_base": (3.0, 9.0),
}


# ============================================================================
# Keely Ratios
# ============================================================================

def keely_ratio_to_hz(ratio: str) -> List[float]:
    """
    Convert Keely ratio to Hz values.

    3:6:9 -> [333, 666, 999]
    """
    parts = ratio.split(":")
    return [int(p) * KEELY_BASE for p in parts]


KEELY_RATIOS = {
    "1:1:1": "Unity, fundamental",
    "2:4:8": "Octave progression",
    "3:6:9": "Tesla/Keely divine pattern",
    "1:3:9": "Ninths progression",
    "1:2:3": "Harmonic series base",
}


# ============================================================================
# Mapping Functions
# ============================================================================

def get_neural_band(hz: float) -> str:
    """Get brainwave band for frequency."""
    for band, (low, high) in BRAINWAVE_BANDS.items():
        if low <= hz < high:
            return band.capitalize()
    if hz >= 100:
        return "Super-Gamma"
    return "Sub-Delta"


def get_chakra_match(hz: float, tolerance: float = 20.0) -> Optional[str]:
    """Find matching chakra for frequency."""
    for chakra, freq in CHAKRA_FREQUENCIES.items():
        if abs(hz - freq) <= tolerance:
            return chakra.replace("_", " ").title()
    return None


def find_rife_match(hz: float, tolerance: float = 10.0) -> List[float]:
    """Find Rife frequencies near target Hz."""
    matches = []
    for category, freqs in RIFE_FREQUENCIES.items():
        for freq in freqs:
            if abs(hz - freq) <= tolerance:
                matches.append(freq)
    return sorted(set(matches))


def find_tesla_range(hz: float) -> Tuple[float, float]:
    """Find Tesla range containing frequency."""
    for name, (low, high) in TESLA_RANGES.items():
        if low <= hz <= high:
            return (low, high)
    # Default to nearest range
    if hz < 10:
        return TESLA_RANGES["low_impulse"]
    elif hz < 100:
        return TESLA_RANGES["mid_impulse"]
    else:
        return TESLA_RANGES["high_impulse"]


def find_keely_ratio(hz: float) -> str:
    """Find Keely ratio that produces this frequency."""
    if hz % KEELY_BASE == 0:
        multiplier = int(hz / KEELY_BASE)
        if multiplier in [3, 6, 9]:
            return "3:6:9"
        elif multiplier in [1, 2, 4, 8]:
            return "2:4:8"
        else:
            return f"{multiplier}:1"
    return "symbolic"


# ============================================================================
# Table Generation
# ============================================================================

def build_mapping_entry(ra_label: str, info: Dict) -> FrequencyMapping:
    """Build a single mapping entry from Ra harmonic info."""
    hz_values = info["hz"]
    primary_hz = hz_values[0] if hz_values else 0

    # Find matches
    rife_match = find_rife_match(primary_hz)
    tesla_range = find_tesla_range(primary_hz)
    keely_ratio = find_keely_ratio(primary_hz)
    chakra = get_chakra_match(primary_hz)
    neural = get_neural_band(primary_hz)

    # Determine match type
    if rife_match and abs(rife_match[0] - primary_hz) < 1:
        match_type = MatchType.EXACT
    elif rife_match:
        match_type = MatchType.NEAREST
    else:
        match_type = MatchType.SYMBOLIC

    return FrequencyMapping(
        ra_label=ra_label,
        hz_values=hz_values,
        rife_equiv=rife_match if rife_match else [],
        tesla_range=tesla_range,
        keely_ratio=keely_ratio,
        chakra_state=chakra or "None",
        neural_band=neural,
        match_type=match_type,
        notes=info["description"]
    )


def generate_frequency_table() -> FrequencyMappingTable:
    """Generate complete frequency mapping table."""
    mappings = []
    for ra_label, info in RA_HARMONICS.items():
        entry = build_mapping_entry(ra_label, info)
        mappings.append(entry)

    return FrequencyMappingTable(
        mappings=mappings,
        sources_used={
            "RaConstants": "Ra.Constants.Harmonics v1.0",
            "Solfeggio": "Standard Solfeggio Scale",
            "KeelyPhysics": "KEELY_SYMPATHETIC_VIBRATORY_PHYSICS.md",
            "Brainwaves": "Standard EEG bands",
        }
    )


def table_to_json(table: FrequencyMappingTable) -> Dict[str, Any]:
    """Convert table to JSON-serializable dict."""
    return {
        "mappings": [
            {
                "ra_label": m.ra_label,
                "hz_values": m.hz_values,
                "rife_equiv": m.rife_equiv,
                "tesla_range": list(m.tesla_range),
                "keely_ratio": m.keely_ratio,
                "chakra_state": m.chakra_state,
                "neural_band": m.neural_band,
                "match_type": m.match_type.name,
                "notes": m.notes,
            }
            for m in table.mappings
        ],
        "sources_used": table.sources_used,
    }


# ============================================================================
# Test Suite
# ============================================================================

class TestSolfeggioFrequencies:
    """Test Solfeggio frequency mappings."""

    def test_solfeggio_coverage(self):
        """All Solfeggio frequencies are mapped."""
        table = generate_frequency_table()
        mapped_hz = set()
        for m in table.mappings:
            mapped_hz.update(m.hz_values)

        for name, freq in SOLFEGGIO.items():
            assert freq in mapped_hz, f"Missing Solfeggio {name}: {freq}Hz"

    def test_528_dna_repair(self):
        """528Hz mapped correctly as DNA repair."""
        table = generate_frequency_table()
        entry = next((m for m in table.mappings if 528 in m.hz_values), None)
        assert entry is not None
        assert "DNA" in entry.notes or "transformation" in entry.notes.lower()


class TestKeelyRatios:
    """Test Keely ratio mappings."""

    def test_369_pattern(self):
        """3:6:9 pattern produces correct frequencies."""
        hz = keely_ratio_to_hz("3:6:9")
        assert hz == [333, 666, 999]

    def test_keely_base_unit(self):
        """111Hz is Keely base unit."""
        table = generate_frequency_table()
        entry = next((m for m in table.mappings if 111 in m.hz_values), None)
        assert entry is not None
        assert "Keely" in entry.ra_label


class TestNeuralBands:
    """Test brainwave band classification."""

    def test_delta_band(self):
        """Delta band is 0.5-4Hz."""
        assert get_neural_band(2.0) == "Delta"

    def test_theta_band(self):
        """Theta band is 4-8Hz."""
        assert get_neural_band(6.0) == "Theta"

    def test_alpha_band(self):
        """Alpha band is 8-13Hz."""
        assert get_neural_band(10.0) == "Alpha"

    def test_gamma_band(self):
        """Gamma band is 30-100Hz."""
        assert get_neural_band(40.0) == "Gamma"

    def test_schumann_theta(self):
        """7.83Hz (Schumann) is in Theta band."""
        assert get_neural_band(7.83) == "Theta"


class TestChakraMapping:
    """Test chakra frequency associations."""

    def test_heart_chakra_639(self):
        """Heart chakra at 639Hz."""
        chakra = get_chakra_match(639)
        assert chakra is not None
        assert "Heart" in chakra

    def test_crown_chakra_963(self):
        """Crown chakra at 963Hz."""
        chakra = get_chakra_match(963)
        assert chakra is not None
        assert "Crown" in chakra

    def test_tolerance_matching(self):
        """Nearby frequencies match with tolerance."""
        chakra = get_chakra_match(525, tolerance=10)  # Near 528
        assert chakra is not None


class TestRifeMatching:
    """Test Rife frequency matching."""

    def test_exact_match_528(self):
        """528Hz has exact Rife match."""
        matches = find_rife_match(528)
        assert 528 in matches

    def test_near_match(self):
        """Near frequencies find matches."""
        matches = find_rife_match(730, tolerance=10)
        assert len(matches) > 0


class TestTeslaRanges:
    """Test Tesla range classification."""

    def test_low_range(self):
        """Low frequencies map to low impulse range."""
        range_ = find_tesla_range(5.0)
        assert range_ == (1.0, 10.0)

    def test_mid_range(self):
        """Mid frequencies map to mid impulse range."""
        range_ = find_tesla_range(50.0)
        assert range_ == (10.0, 100.0)


class TestTableGeneration:
    """Test complete table generation."""

    def test_table_not_empty(self):
        """Table has mappings."""
        table = generate_frequency_table()
        assert len(table.mappings) > 0

    def test_json_serializable(self):
        """Table converts to valid JSON."""
        table = generate_frequency_table()
        json_data = table_to_json(table)
        json_str = json.dumps(json_data)
        parsed = json.loads(json_str)
        assert "mappings" in parsed
        assert len(parsed["mappings"]) == len(table.mappings)

    def test_sources_documented(self):
        """Sources are documented."""
        table = generate_frequency_table()
        assert len(table.sources_used) > 0


class TestMatchTypes:
    """Test frequency match type classification."""

    def test_exact_match_identified(self):
        """Exact matches are identified."""
        table = generate_frequency_table()
        exact_matches = [m for m in table.mappings if m.match_type == MatchType.EXACT]
        # 528 should be exact match
        assert any(528 in m.hz_values for m in exact_matches)

    def test_symbolic_matches(self):
        """Symbolic matches are identified for unique Ra frequencies."""
        table = generate_frequency_table()
        symbolic = [m for m in table.mappings if m.match_type == MatchType.SYMBOLIC]
        # PHI-scaled harmonics should be symbolic
        phi_entries = [m for m in symbolic if "Phi" in m.ra_label]
        assert len(phi_entries) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
