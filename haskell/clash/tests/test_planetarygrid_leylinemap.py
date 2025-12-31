"""
Test Suite for Ra.PlanetaryGrid.LeylineMap (Prompt 73)

Generates harmonic overlay of Earth's surface based on leyline crossings,
scalar resonance points, and sacred geometry nodes. Used for georesonant
integration and avatar-field amplification via Earth's energy network.

Architect Clarifications:
- Radionic rate translation: phi-ratio mapping
- Temporal modulation: solar alignments (noon peak), Schumann phase
- Geometry overlays: Becker-Hagens grid (62 base nodes, expanded to 108+)
"""

import pytest
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Tuple

# Constants
PHI = 1.618033988749895
SCHUMANN_FUNDAMENTAL = 7.83
EARTH_RADIUS_KM = 6371.0
MIN_LEYLINE_NODES = 108  # Success criteria requirement

# Becker-Hagens UVG 120 grid base (62 primary nodes)
# Selected key nodes with coordinates
BECKER_HAGENS_BASE_NODES = [
    (0.0, 31.72),       # Node 1 - Egypt/Giza area
    (0.0, -31.72),      # Node 2 - Southern vertex
    (36.0, 52.62),      # Node 3 - European vertex
    (72.0, 31.72),      # Node 4 - India/Central Asia
    (108.0, 52.62),     # Node 5 - Siberia
    (144.0, 31.72),     # Node 6 - Pacific
    (180.0, 52.62),     # Node 7 - Bering
    (-36.0, 52.62),     # Node 8 - Atlantic
    (-72.0, 31.72),     # Node 9 - Americas
    (-108.0, 52.62),    # Node 10 - Pacific
    (-144.0, 31.72),    # Node 11 - Pacific
    (31.72, 0.0),       # Equatorial nodes
    (68.28, 0.0),
    (104.28, 0.0),
    (140.28, 0.0),
    (176.28, 0.0),
    (-31.72, 0.0),
    (-68.28, 0.0),
    (-104.28, 0.0),
    (-140.28, 0.0),
    (-176.28, 0.0),
]

# Known pyramid and sacred site locations
SACRED_SITES = [
    (29.9792, 31.1342, "Giza"),        # Great Pyramid
    (13.1631, -72.5450, "Machu Picchu"),
    (27.1751, 78.0421, "Taj Mahal"),
    (51.1789, -1.8262, "Stonehenge"),
    (47.6205, -122.3493, "Seattle (Tesla experiments)"),
    (55.7520, 37.6175, "Moscow (Golod pyramids)"),
    (-33.8688, 151.2093, "Sydney"),
    (35.6762, 139.6503, "Tokyo"),
    (19.4326, -99.1332, "Mexico City (Teotihuacan)"),
    (-22.9068, -43.1729, "Rio (Christ Redeemer)"),
]


class PylonType(Enum):
    """Structural form types for leyline nodes."""
    NONE = auto()
    PYRAMID = auto()
    OBELISK = auto()
    CIRCLE = auto()
    MOUND = auto()
    TOWER = auto()
    NATURAL = auto()


class PhiPhase(Enum):
    """Phi-based phase alignment."""
    ASCENDING = auto()
    DESCENDING = auto()
    PEAK = auto()
    TROUGH = auto()
    NEUTRAL = auto()


class TemporalMode(Enum):
    """Temporal modulation mode."""
    STATIC = auto()
    SOLAR = auto()
    LUNAR = auto()
    SCHUMANN = auto()


@dataclass
class GeoCoordinate:
    """Geographic coordinate."""
    latitude: float   # Degrees
    longitude: float  # Degrees

    def distance_to(self, other: 'GeoCoordinate') -> float:
        """Haversine distance in km."""
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_KM * c


@dataclass
class LeylineNode:
    """Single leyline node on planetary grid."""
    geo_coordinate: GeoCoordinate
    harmonic_rate: float              # Hz
    structural_form: Optional[PylonType]
    emergence_sync: Optional[PhiPhase]
    node_strength: float = 1.0        # 0-1
    is_intersection: bool = False     # Leyline crossing


@dataclass
class PyramidSite:
    """Known pyramid or radionic site."""
    coordinate: GeoCoordinate
    name: str
    base_rate: float                  # Radionic rate
    structure_type: PylonType


@dataclass
class RadionicRate:
    """Radionic rate from dowsing/reference."""
    rate_value: float
    target_type: str
    source: str


@dataclass
class PlanetaryMapData:
    """Input data for leyline grid generation."""
    pyramid_sites: List[PyramidSite]
    radionic_rates: List[RadionicRate]
    base_grid_nodes: List[Tuple[float, float]]
    temporal_mode: TemporalMode


@dataclass
class LeylineGrid:
    """Complete leyline grid."""
    nodes: List[LeylineNode]
    total_harmonic_power: float
    coverage_percentage: float


def phi_ratio_map(radionic_rate: float) -> float:
    """
    Map radionic rate to harmonic frequency using phi-ratio.
    Formula: harmonic = base_schumann * (rate / 1000) * phi^n
    where n is determined by rate magnitude.
    """
    if radionic_rate <= 0:
        return SCHUMANN_FUNDAMENTAL

    # Determine phi power based on rate magnitude
    magnitude = math.log10(max(1, radionic_rate))
    phi_power = int(magnitude)

    # Apply phi scaling
    phi_factor = PHI ** phi_power

    # Map to harmonic
    base_harmonic = SCHUMANN_FUNDAMENTAL * (radionic_rate / 1000.0)
    return base_harmonic * phi_factor


def compute_solar_modulation(hour_of_day: float) -> float:
    """
    Compute solar modulation factor.
    Peak at local noon (hour 12), trough at midnight.
    """
    # Sinusoidal modulation
    angle = (hour_of_day / 24.0) * 2 * math.pi
    # Shift so peak is at noon (hour 12)
    shifted = angle - math.pi / 2
    modulation = 0.5 + 0.5 * math.sin(shifted)
    return modulation


def compute_schumann_phase(time_offset: float) -> float:
    """
    Compute Schumann resonance phase modulation.
    7.83 Hz fundamental.
    """
    phase = math.sin(2 * math.pi * SCHUMANN_FUNDAMENTAL * time_offset)
    return 0.5 + 0.5 * phase


def apply_temporal_modulation(
    base_rate: float,
    mode: TemporalMode,
    hour_of_day: float = 12.0,
    time_offset: float = 0.0
) -> float:
    """Apply temporal modulation to harmonic rate."""
    if mode == TemporalMode.STATIC:
        return base_rate

    elif mode == TemporalMode.SOLAR:
        mod = compute_solar_modulation(hour_of_day)
        return base_rate * (0.8 + 0.4 * mod)  # ±20% modulation

    elif mode == TemporalMode.SCHUMANN:
        mod = compute_schumann_phase(time_offset)
        return base_rate * (0.9 + 0.2 * mod)  # ±10% modulation

    elif mode == TemporalMode.LUNAR:
        # Simplified lunar (28-day cycle)
        lunar_phase = math.sin(2 * math.pi * time_offset / 28.0)
        return base_rate * (0.95 + 0.1 * lunar_phase)

    return base_rate


def compute_phi_phase(latitude: float, longitude: float) -> PhiPhase:
    """Compute phi phase alignment for coordinate."""
    # Use coordinate modulo to determine phase
    lat_mod = abs(latitude) % (180 / PHI)
    lon_mod = abs(longitude) % (360 / PHI)

    combined = (lat_mod + lon_mod) / 2.0
    normalized = combined / (180 / PHI)

    if normalized < 0.2:
        return PhiPhase.TROUGH
    elif normalized < 0.4:
        return PhiPhase.ASCENDING
    elif normalized < 0.6:
        return PhiPhase.PEAK
    elif normalized < 0.8:
        return PhiPhase.DESCENDING
    else:
        return PhiPhase.NEUTRAL


def find_nearest_structure(
    coord: GeoCoordinate,
    sites: List[PyramidSite],
    max_distance_km: float = 500.0
) -> Optional[PylonType]:
    """Find nearest structural form to coordinate."""
    nearest = None
    min_dist = float('inf')

    for site in sites:
        dist = coord.distance_to(site.coordinate)
        if dist < min_dist and dist <= max_distance_km:
            min_dist = dist
            nearest = site.structure_type

    return nearest


def check_leyline_intersection(
    coord: GeoCoordinate,
    all_nodes: List[LeylineNode],
    threshold_km: float = 100.0
) -> bool:
    """Check if coordinate is near multiple leyline paths."""
    nearby_count = 0
    for node in all_nodes:
        dist = coord.distance_to(node.geo_coordinate)
        if dist < threshold_km:
            nearby_count += 1

    # Intersection if 3+ nodes nearby
    return nearby_count >= 3


def generate_grid_nodes(
    base_nodes: List[Tuple[float, float]],
    target_count: int = MIN_LEYLINE_NODES
) -> List[Tuple[float, float]]:
    """
    Expand base grid nodes to target count.
    Uses interpolation and sacred geometry subdivision.
    """
    nodes = list(base_nodes)

    # Add midpoints between existing nodes until target reached
    while len(nodes) < target_count:
        new_nodes = []
        for i in range(len(nodes)):
            for j in range(i + 1, min(i + 4, len(nodes))):
                # Midpoint
                mid_lon = (nodes[i][0] + nodes[j][0]) / 2
                mid_lat = (nodes[i][1] + nodes[j][1]) / 2

                # Check not too close to existing
                too_close = any(
                    abs(mid_lon - n[0]) < 5 and abs(mid_lat - n[1]) < 5
                    for n in nodes
                )
                if not too_close:
                    new_nodes.append((mid_lon, mid_lat))

                if len(nodes) + len(new_nodes) >= target_count:
                    break
            if len(nodes) + len(new_nodes) >= target_count:
                break

        if not new_nodes:
            break
        nodes.extend(new_nodes[:target_count - len(nodes)])

    return nodes


def create_leyline_node(
    lon: float,
    lat: float,
    radionic_rates: List[RadionicRate],
    sites: List[PyramidSite],
    temporal_mode: TemporalMode,
    existing_nodes: List[LeylineNode]
) -> LeylineNode:
    """Create a single leyline node."""
    coord = GeoCoordinate(latitude=lat, longitude=lon)

    # Compute harmonic rate from nearest radionic rate
    base_rate = SCHUMANN_FUNDAMENTAL
    if radionic_rates:
        # Use average of rates
        avg_rate = sum(r.rate_value for r in radionic_rates) / len(radionic_rates)
        base_rate = phi_ratio_map(avg_rate)

    # Apply temporal modulation
    harmonic_rate = apply_temporal_modulation(base_rate, temporal_mode)

    # Find structural form
    structure = find_nearest_structure(coord, sites)

    # Compute phi phase
    phi_phase = compute_phi_phase(lat, lon)

    # Check intersection
    is_intersection = check_leyline_intersection(coord, existing_nodes)

    # Compute node strength (higher at intersections and near sites)
    strength = 0.5
    if structure:
        strength += 0.3
    if is_intersection:
        strength += 0.2
    strength = min(1.0, strength)

    return LeylineNode(
        geo_coordinate=coord,
        harmonic_rate=harmonic_rate,
        structural_form=structure,
        emergence_sync=phi_phase,
        node_strength=strength,
        is_intersection=is_intersection
    )


def generate_leyline_grid(map_data: PlanetaryMapData) -> LeylineGrid:
    """
    Generate complete leyline grid from planetary map data.
    Main function contract implementation.
    """
    # Expand grid nodes to 108+
    expanded_nodes = generate_grid_nodes(
        map_data.base_grid_nodes,
        MIN_LEYLINE_NODES
    )

    # Add sacred site locations
    for site in map_data.pyramid_sites:
        coord = (site.coordinate.longitude, site.coordinate.latitude)
        if coord not in expanded_nodes:
            expanded_nodes.append(coord)

    leyline_nodes = []
    total_power = 0.0

    for lon, lat in expanded_nodes:
        node = create_leyline_node(
            lon, lat,
            map_data.radionic_rates,
            map_data.pyramid_sites,
            map_data.temporal_mode,
            leyline_nodes
        )
        leyline_nodes.append(node)
        total_power += node.harmonic_rate * node.node_strength

    # Compute coverage (percentage of Earth surface within range of nodes)
    # Simplified: assume each node covers ~1000km radius
    node_coverage = len(leyline_nodes) * (1000 / EARTH_RADIUS_KM) * 100
    coverage = min(100.0, node_coverage)

    return LeylineGrid(
        nodes=leyline_nodes,
        total_harmonic_power=total_power,
        coverage_percentage=coverage
    )


def validate_leyline_grid(grid: LeylineGrid) -> bool:
    """Validate generated leyline grid."""
    # Must have minimum nodes
    if len(grid.nodes) < MIN_LEYLINE_NODES:
        return False

    # All nodes must have valid coordinates
    for node in grid.nodes:
        if not (-90 <= node.geo_coordinate.latitude <= 90):
            return False
        if not (-180 <= node.geo_coordinate.longitude <= 180):
            return False
        if node.harmonic_rate <= 0:
            return False

    return True


def find_resonant_nodes(
    grid: LeylineGrid,
    target_frequency: float,
    tolerance: float = 0.5
) -> List[LeylineNode]:
    """Find nodes resonating near target frequency."""
    return [
        node for node in grid.nodes
        if abs(node.harmonic_rate - target_frequency) < tolerance
    ]


# ============== TESTS ==============

class TestPhiRatioMapping:
    """Tests for phi-ratio radionic rate mapping."""

    def test_zero_rate_returns_schumann(self):
        """Zero radionic rate should return Schumann fundamental."""
        result = phi_ratio_map(0)
        assert result == pytest.approx(SCHUMANN_FUNDAMENTAL)

    def test_positive_rate_applies_phi(self):
        """Positive rate should apply phi scaling."""
        result = phi_ratio_map(1000)
        # At 1000, phi^3 factor applied
        expected = SCHUMANN_FUNDAMENTAL * 1.0 * (PHI ** 3)
        assert result == pytest.approx(expected, rel=0.1)

    def test_higher_rate_more_phi_power(self):
        """Higher rates should get more phi power."""
        low_result = phi_ratio_map(100)
        high_result = phi_ratio_map(10000)
        assert high_result > low_result

    def test_phi_factor_increases_with_magnitude(self):
        """Phi factor should increase with rate magnitude."""
        r1 = phi_ratio_map(10)
        r2 = phi_ratio_map(100)
        r3 = phi_ratio_map(1000)

        # Each magnitude increase should multiply by phi
        assert r2 / r1 > 1.5
        assert r3 / r2 > 1.5


class TestTemporalModulation:
    """Tests for temporal modulation of harmonic rates."""

    def test_static_mode_no_change(self):
        """Static mode should not modify rate."""
        base = 10.0
        result = apply_temporal_modulation(base, TemporalMode.STATIC)
        assert result == pytest.approx(base)

    def test_solar_peak_at_noon(self):
        """Solar mode should peak at noon."""
        base = 10.0
        noon = apply_temporal_modulation(base, TemporalMode.SOLAR, hour_of_day=12.0)
        midnight = apply_temporal_modulation(base, TemporalMode.SOLAR, hour_of_day=0.0)
        assert noon > midnight

    def test_solar_within_bounds(self):
        """Solar modulation should stay within ±20%."""
        base = 10.0
        for hour in range(24):
            result = apply_temporal_modulation(base, TemporalMode.SOLAR, hour_of_day=float(hour))
            assert 8.0 <= result <= 12.01  # Small tolerance for floating point

    def test_schumann_oscillates(self):
        """Schumann mode should oscillate."""
        base = 10.0
        results = [
            apply_temporal_modulation(base, TemporalMode.SCHUMANN, time_offset=t/10.0)
            for t in range(10)
        ]
        # Should have variation
        assert max(results) != min(results)


class TestPhiPhaseComputation:
    """Tests for phi phase alignment computation."""

    def test_returns_valid_phase(self):
        """Should return valid PhiPhase enum."""
        phase = compute_phi_phase(45.0, 90.0)
        assert isinstance(phase, PhiPhase)

    def test_different_coords_different_phases(self):
        """Different coordinates may have different phases."""
        phases = [
            compute_phi_phase(0, 0),
            compute_phi_phase(30, 60),
            compute_phi_phase(60, 120),
            compute_phi_phase(90, 180),
        ]
        # Should have at least some variation
        assert len(set(phases)) >= 2

    def test_symmetric_coords(self):
        """Symmetric coordinates should give consistent phases."""
        phase1 = compute_phi_phase(45, 90)
        phase2 = compute_phi_phase(-45, -90)
        # Absolute values used, so similar
        assert phase1 == phase2


class TestGridNodeExpansion:
    """Tests for grid node expansion."""

    def test_expands_to_target_count(self):
        """Should expand to target node count."""
        # Use more spread out base nodes to allow for expansion
        base = [(0, 0), (30, 0), (60, 0), (0, 30), (30, 30), (60, 30)]
        expanded = generate_grid_nodes(base, target_count=15)
        assert len(expanded) >= 15

    def test_preserves_base_nodes(self):
        """Should preserve original base nodes."""
        base = [(0, 0), (10, 0)]
        expanded = generate_grid_nodes(base, target_count=10)
        assert (0, 0) in expanded
        assert (10, 0) in expanded

    def test_reaches_108_nodes(self):
        """Should reach minimum 108 nodes."""
        expanded = generate_grid_nodes(BECKER_HAGENS_BASE_NODES, MIN_LEYLINE_NODES)
        assert len(expanded) >= MIN_LEYLINE_NODES


class TestLeylineNodeCreation:
    """Tests for individual leyline node creation."""

    def test_creates_valid_node(self):
        """Should create valid leyline node."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]
        rates = [RadionicRate(1000, "general", "test")]

        node = create_leyline_node(
            31.0, 30.0, rates, sites,
            TemporalMode.STATIC, []
        )

        assert node.geo_coordinate.latitude == 30.0
        assert node.geo_coordinate.longitude == 31.0
        assert node.harmonic_rate > 0

    def test_finds_nearby_structure(self):
        """Should find nearby structural form."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]

        node = create_leyline_node(
            31.0, 30.0, [], sites,  # Very close to Giza
            TemporalMode.STATIC, []
        )

        assert node.structural_form == PylonType.PYRAMID

    def test_no_structure_when_far(self):
        """Should have no structure when far from sites."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]

        node = create_leyline_node(
            -100.0, -50.0, [], sites,  # Far from Giza
            TemporalMode.STATIC, []
        )

        assert node.structural_form is None

    def test_strength_higher_with_structure(self):
        """Node strength should be higher with nearby structure."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]

        node_near = create_leyline_node(31.0, 30.0, [], sites, TemporalMode.STATIC, [])
        node_far = create_leyline_node(-100.0, -50.0, [], sites, TemporalMode.STATIC, [])

        assert node_near.node_strength > node_far.node_strength


class TestLeylineGridGeneration:
    """Tests for complete leyline grid generation."""

    def test_generates_minimum_108_nodes(self):
        """Should generate at least 108 nodes."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]
        rates = [RadionicRate(1000, "general", "test")]

        map_data = PlanetaryMapData(
            pyramid_sites=sites,
            radionic_rates=rates,
            base_grid_nodes=BECKER_HAGENS_BASE_NODES,
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)

        assert len(grid.nodes) >= MIN_LEYLINE_NODES

    def test_includes_sacred_sites(self):
        """Grid should include nodes near sacred sites."""
        giza_coord = GeoCoordinate(30, 31)
        sites = [
            PyramidSite(giza_coord, "Giza", 1000, PylonType.PYRAMID)
        ]

        map_data = PlanetaryMapData(
            pyramid_sites=sites,
            radionic_rates=[],
            base_grid_nodes=[(0, 0)],
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)

        # Should have node near Giza
        giza_nearby = any(
            n.geo_coordinate.distance_to(giza_coord) < 100
            for n in grid.nodes
        )
        assert giza_nearby

    def test_computes_total_harmonic_power(self):
        """Should compute total harmonic power."""
        sites = []
        rates = [RadionicRate(1000, "general", "test")]

        map_data = PlanetaryMapData(
            pyramid_sites=sites,
            radionic_rates=rates,
            base_grid_nodes=BECKER_HAGENS_BASE_NODES,
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)

        assert grid.total_harmonic_power > 0

    def test_valid_grid_passes_validation(self):
        """Generated grid should pass validation."""
        sites = [
            PyramidSite(GeoCoordinate(30, 31), "Giza", 1000, PylonType.PYRAMID)
        ]

        map_data = PlanetaryMapData(
            pyramid_sites=sites,
            radionic_rates=[],
            base_grid_nodes=BECKER_HAGENS_BASE_NODES,
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)

        assert validate_leyline_grid(grid)


class TestResonantNodeFinding:
    """Tests for finding resonant nodes."""

    def test_finds_schumann_resonant_nodes(self):
        """Should find nodes at Schumann frequency."""
        sites = []
        # No radionic rates = Schumann default
        map_data = PlanetaryMapData(
            pyramid_sites=sites,
            radionic_rates=[],
            base_grid_nodes=BECKER_HAGENS_BASE_NODES[:10],
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)
        resonant = find_resonant_nodes(grid, SCHUMANN_FUNDAMENTAL, tolerance=1.0)

        assert len(resonant) > 0

    def test_empty_for_mismatched_frequency(self):
        """Should return empty for non-existent frequency."""
        map_data = PlanetaryMapData(
            pyramid_sites=[],
            radionic_rates=[RadionicRate(1000, "test", "test")],
            base_grid_nodes=[(0, 0), (10, 10)],
            temporal_mode=TemporalMode.STATIC
        )

        grid = generate_leyline_grid(map_data)
        # Look for impossible frequency
        resonant = find_resonant_nodes(grid, 999999.0, tolerance=0.1)

        assert len(resonant) == 0


class TestGridValidation:
    """Tests for grid validation."""

    def test_valid_grid_passes(self):
        """Valid grid should pass validation."""
        # Generate enough nodes (at least 108)
        nodes = [
            LeylineNode(
                GeoCoordinate(lat, lon),
                SCHUMANN_FUNDAMENTAL,
                None, PhiPhase.NEUTRAL, 0.5, False
            )
            for lat, lon in [(i, j) for i in range(-80, 90, 15) for j in range(-170, 180, 30)]
        ]
        grid = LeylineGrid(nodes=nodes, total_harmonic_power=100, coverage_percentage=50)

        assert len(grid.nodes) >= MIN_LEYLINE_NODES
        assert validate_leyline_grid(grid)

    def test_too_few_nodes_fails(self):
        """Grid with < 108 nodes should fail."""
        nodes = [
            LeylineNode(GeoCoordinate(0, 0), 7.83, None, PhiPhase.NEUTRAL, 0.5, False)
        ]
        grid = LeylineGrid(nodes=nodes, total_harmonic_power=7.83, coverage_percentage=1)

        assert not validate_leyline_grid(grid)

    def test_invalid_coordinates_fail(self):
        """Invalid coordinates should fail validation."""
        nodes = [
            LeylineNode(
                GeoCoordinate(100, 0),  # Invalid latitude
                7.83, None, PhiPhase.NEUTRAL, 0.5, False
            )
        ] * 108
        grid = LeylineGrid(nodes=nodes, total_harmonic_power=100, coverage_percentage=50)

        assert not validate_leyline_grid(grid)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
