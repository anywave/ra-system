"""
Test Suite for Ra.Visualizer.Glyphs (Prompt 61)
Harmonic Signature Renderer

Tests symbolic glyph rendering for scalar field harmonics,
avatar resonance signatures, and fragment emergence anchors.
"""

import pytest
import math
import hashlib
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895

# Torsion color palette (RGB tuples)
TORSION_COLORS = {
    'Normal': (0, 0, 255),      # Blue
    'Inverted': (255, 0, 0),    # Red
    'Null': (128, 128, 128),    # Grey
}

# =============================================================================
# ENUMS
# =============================================================================

class TorsionPhase(Enum):
    Normal = auto()
    Inverted = auto()
    Null = auto()


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class OmegaFormat:
    """Spherical harmonic encoding (l, m)."""
    l: int  # 0-7
    m: int  # -l to +l

    def __post_init__(self):
        assert 0 <= self.l <= 7, f"l must be in [0,7], got {self.l}"
        assert -self.l <= self.m <= self.l, f"m must be in [-{self.l},{self.l}], got {self.m}"


@dataclass
class GlowLevel:
    """Coherence glow intensity (0-255)."""
    intensity: int  # 0-255

    def __post_init__(self):
        self.intensity = max(0, min(255, self.intensity))


@dataclass
class HarmonicGlyph:
    """Glyph representing scalar field harmonics."""
    glyph_id: str
    phi_phase: int                          # phi^n phase level
    harmonic_root: OmegaFormat              # spherical harmonic key
    torsion_state: TorsionPhase
    coherence_glow: Optional[GlowLevel]     # optional intensity overlay
    associated_with: Optional[str]          # fragment ID link


@dataclass
class GlyphImage:
    """Rendered glyph output."""
    svg_data: bytes
    legend: List[str]
    timestamp: float  # Unix timestamp


@dataclass
class ScalarLayer:
    """Input scalar layer for glyph rendering."""
    depth: float
    amplitude: float
    phase: float
    omega: OmegaFormat
    layer_index: int
    fragment_id: Optional[str] = None
    torsion: TorsionPhase = TorsionPhase.Normal
    coherence: float = 0.5


@dataclass
class FragmentMetadata:
    """Fragment metadata for anchor glyphs."""
    fragment_id: str
    emergence_x: float
    emergence_y: float
    emergence_z: float
    coherence: float
    harmonic_root: OmegaFormat


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def generate_glyph_id(phi_phase: int, omega: OmegaFormat, fragment_id: Optional[str] = None) -> str:
    """Generate glyph ID from fragment ID or tuple hash."""
    if fragment_id is not None:
        return fragment_id

    # Hash from (phi_phase, l, m)
    data = f"{phi_phase}:{omega.l}:{omega.m}"
    hash_bytes = hashlib.sha256(data.encode()).hexdigest()[:16]
    return f"glyph_{hash_bytes}"


def blend_color(base_rgb: Tuple[int, int, int], white_intensity: int) -> Tuple[int, int, int]:
    """Blend base hue with white based on coherence glow intensity."""
    factor = white_intensity / 255.0
    r = int(base_rgb[0] + (255 - base_rgb[0]) * factor)
    g = int(base_rgb[1] + (255 - base_rgb[1]) * factor)
    b = int(base_rgb[2] + (255 - base_rgb[2]) * factor)
    return (r, g, b)


def get_torsion_base_color(torsion: TorsionPhase) -> Tuple[int, int, int]:
    """Get base color for torsion state."""
    return TORSION_COLORS[torsion.name]


def compute_spiral_arms(m: int) -> Tuple[int, str]:
    """Compute number of spiral arms and rotation direction from m.

    Returns: (num_arms, direction) where direction is 'cw' or 'ccw' or 'none'
    """
    if m == 0:
        return (0, 'none')
    elif m > 0:
        return (abs(m), 'cw')  # clockwise
    else:
        return (abs(m), 'ccw')  # counter-clockwise


def render_harmonic_glyph(layer: ScalarLayer) -> HarmonicGlyph:
    """Extract glyph metadata from a ScalarLayer."""
    # Compute phi^n phase level from depth
    phi_phase = int(math.log(layer.depth / 0.618 + 1) / math.log(PHI)) if layer.depth > 0 else 0

    # Compute coherence glow from layer coherence
    glow = GlowLevel(int(layer.coherence * 255)) if layer.coherence > 0 else None

    # Generate glyph ID
    glyph_id = generate_glyph_id(phi_phase, layer.omega, layer.fragment_id)

    return HarmonicGlyph(
        glyph_id=glyph_id,
        phi_phase=phi_phase,
        harmonic_root=layer.omega,
        torsion_state=layer.torsion,
        coherence_glow=glow,
        associated_with=layer.fragment_id
    )


def generate_svg_circle(cx: float, cy: float, r: float, stroke: str, fill: str = "none") -> str:
    """Generate SVG circle element."""
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{stroke}" fill="{fill}" />'


def generate_svg_spiral_path(cx: float, cy: float, arms: int, direction: str, scale: float) -> str:
    """Generate SVG path for spiral arms."""
    if arms == 0:
        return ""

    paths = []
    angle_step = 2 * math.pi / arms
    rotation_sign = 1 if direction == 'cw' else -1

    for i in range(arms):
        start_angle = i * angle_step
        # Create a simple spiral arc
        r1 = scale * 0.2
        r2 = scale * 0.8
        x1 = cx + r1 * math.cos(start_angle)
        y1 = cy + r1 * math.sin(start_angle)
        x2 = cx + r2 * math.cos(start_angle + rotation_sign * math.pi / 4)
        y2 = cy + r2 * math.sin(start_angle + rotation_sign * math.pi / 4)
        paths.append(f'<path d="M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}" stroke="currentColor" fill="none" />')

    return "\n".join(paths)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB tuple to hex color string."""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def generate_glyph_svg(glyph: HarmonicGlyph, size: int = 100) -> GlyphImage:
    """Render glyph to SVG image."""
    import time

    cx, cy = size / 2, size / 2

    # Get base color from torsion
    base_color = get_torsion_base_color(glyph.torsion_state)

    # Apply coherence glow if present
    if glyph.coherence_glow:
        final_color = blend_color(base_color, glyph.coherence_glow.intensity)
    else:
        final_color = base_color

    color_hex = rgb_to_hex(final_color)

    # Start SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">',
        f'<g fill="none" stroke="{color_hex}">'
    ]

    # Generate radial rings for l (harmonic depth)
    l = glyph.harmonic_root.l
    for ring_idx in range(l + 1):
        ring_radius = (ring_idx + 1) * (size * 0.4 / (l + 1))
        svg_parts.append(generate_svg_circle(cx, cy, ring_radius, color_hex))

    # Generate spiral arms for m
    arms, direction = compute_spiral_arms(glyph.harmonic_root.m)
    if arms > 0:
        spiral_svg = generate_svg_spiral_path(cx, cy, arms, direction, size * 0.4)
        svg_parts.append(spiral_svg)

    # Add center glow circle if coherence present
    if glyph.coherence_glow and glyph.coherence_glow.intensity > 50:
        glow_radius = 5 + (glyph.coherence_glow.intensity / 255) * 10
        svg_parts.append(generate_svg_circle(cx, cy, glow_radius, "white", "white"))

    svg_parts.append('</g>')
    svg_parts.append('</svg>')

    svg_data = "\n".join(svg_parts).encode('utf-8')

    legend = [
        f"Phase: phi^{glyph.phi_phase}",
        f"Harmonic: l={glyph.harmonic_root.l}, m={glyph.harmonic_root.m}",
        f"Torsion: {glyph.torsion_state.name}",
    ]
    if glyph.coherence_glow:
        legend.append(f"Glow: {glyph.coherence_glow.intensity}/255")

    return GlyphImage(
        svg_data=svg_data,
        legend=legend,
        timestamp=time.time()
    )


def fragment_anchor_glyph(metadata: FragmentMetadata) -> GlyphImage:
    """Generate anchor glyph from fragment metadata."""
    # Build a glyph from fragment data
    glyph = HarmonicGlyph(
        glyph_id=metadata.fragment_id,
        phi_phase=0,  # Anchors are at base phase
        harmonic_root=metadata.harmonic_root,
        torsion_state=TorsionPhase.Normal,
        coherence_glow=GlowLevel(int(metadata.coherence * 255)),
        associated_with=metadata.fragment_id
    )

    image = generate_glyph_svg(glyph, size=100)

    # Add spatial coordinates to legend
    image.legend.extend([
        f"Anchor: ({metadata.emergence_x:.2f}, {metadata.emergence_y:.2f}, {metadata.emergence_z:.2f})",
        f"Fragment: {metadata.fragment_id}"
    ])

    return image


# =============================================================================
# TEST: TORSION COLOR MAPPING
# =============================================================================

class TestTorsionColors:
    """Test torsion state to color mapping."""

    def test_normal_torsion_is_blue(self):
        """Normal torsion maps to blue palette."""
        color = get_torsion_base_color(TorsionPhase.Normal)
        assert color == (0, 0, 255)

    def test_inverted_torsion_is_red(self):
        """Inverted torsion maps to red palette."""
        color = get_torsion_base_color(TorsionPhase.Inverted)
        assert color == (255, 0, 0)

    def test_null_torsion_is_grey(self):
        """Null torsion maps to grey palette."""
        color = get_torsion_base_color(TorsionPhase.Null)
        assert color == (128, 128, 128)


# =============================================================================
# TEST: COHERENCE GLOW BLENDING
# =============================================================================

class TestCoherenceGlow:
    """Test coherence glow intensity blending."""

    def test_zero_glow_preserves_base(self):
        """Zero glow intensity preserves base color."""
        base = (0, 0, 255)  # Blue
        result = blend_color(base, 0)
        assert result == base

    def test_max_glow_is_white(self):
        """Maximum glow intensity produces white."""
        base = (0, 0, 255)  # Blue
        result = blend_color(base, 255)
        assert result == (255, 255, 255)

    def test_half_glow_blends_correctly(self):
        """Half glow intensity blends 50% toward white."""
        base = (0, 0, 255)  # Blue
        result = blend_color(base, 128)
        # Expected: ~50% blend toward white
        assert 120 <= result[0] <= 135  # R
        assert 120 <= result[1] <= 135  # G
        assert result[2] == 255          # B stays max

    def test_glow_level_clamps_to_valid_range(self):
        """GlowLevel clamps intensity to 0-255."""
        glow_high = GlowLevel(300)
        assert glow_high.intensity == 255

        glow_low = GlowLevel(-50)
        assert glow_low.intensity == 0


# =============================================================================
# TEST: GLYPH ID GENERATION
# =============================================================================

class TestGlyphIdGeneration:
    """Test glyph ID generation rules."""

    def test_fragment_id_takes_precedence(self):
        """When fragment ID is present, use it as glyph ID."""
        omega = OmegaFormat(l=3, m=1)
        glyph_id = generate_glyph_id(5, omega, fragment_id="frag_123")
        assert glyph_id == "frag_123"

    def test_auto_generated_id_from_hash(self):
        """Without fragment ID, generate from tuple hash."""
        omega = OmegaFormat(l=3, m=1)
        glyph_id = generate_glyph_id(5, omega, fragment_id=None)
        assert glyph_id.startswith("glyph_")
        assert len(glyph_id) == 22  # "glyph_" + 16 hex chars

    def test_deterministic_hash_generation(self):
        """Same inputs produce same glyph ID."""
        omega = OmegaFormat(l=4, m=-2)
        id1 = generate_glyph_id(3, omega)
        id2 = generate_glyph_id(3, omega)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different inputs produce different glyph IDs."""
        omega1 = OmegaFormat(l=3, m=1)
        omega2 = OmegaFormat(l=3, m=2)

        id1 = generate_glyph_id(5, omega1)
        id2 = generate_glyph_id(5, omega2)
        assert id1 != id2


# =============================================================================
# TEST: SPIRAL ARM ENCODING
# =============================================================================

class TestSpiralArmEncoding:
    """Test spiral arm encoding from m value."""

    def test_zero_m_no_arms(self):
        """m=0 produces no arms."""
        arms, direction = compute_spiral_arms(0)
        assert arms == 0
        assert direction == 'none'

    def test_positive_m_clockwise(self):
        """Positive m produces clockwise arms."""
        for m in [1, 2, 3, 4, 5]:
            arms, direction = compute_spiral_arms(m)
            assert arms == m
            assert direction == 'cw'

    def test_negative_m_counter_clockwise(self):
        """Negative m produces counter-clockwise arms."""
        for m in [-1, -2, -3, -4, -5]:
            arms, direction = compute_spiral_arms(m)
            assert arms == abs(m)
            assert direction == 'ccw'

    def test_arm_count_equals_magnitude(self):
        """Number of arms equals |m|."""
        assert compute_spiral_arms(3)[0] == 3
        assert compute_spiral_arms(-5)[0] == 5
        assert compute_spiral_arms(7)[0] == 7


# =============================================================================
# TEST: OMEGA FORMAT VALIDATION
# =============================================================================

class TestOmegaFormat:
    """Test OmegaFormat spherical harmonic validation."""

    def test_valid_l_range(self):
        """l must be in [0, 7]."""
        for l in range(8):
            omega = OmegaFormat(l=l, m=0)
            assert omega.l == l

    def test_invalid_l_raises(self):
        """Invalid l raises assertion error."""
        with pytest.raises(AssertionError):
            OmegaFormat(l=8, m=0)

        with pytest.raises(AssertionError):
            OmegaFormat(l=-1, m=0)

    def test_m_must_be_within_l_bounds(self):
        """m must be in [-l, l]."""
        omega = OmegaFormat(l=3, m=3)
        assert omega.m == 3

        omega = OmegaFormat(l=3, m=-3)
        assert omega.m == -3

    def test_m_out_of_bounds_raises(self):
        """m outside [-l, l] raises assertion error."""
        with pytest.raises(AssertionError):
            OmegaFormat(l=2, m=3)  # |m| > l

        with pytest.raises(AssertionError):
            OmegaFormat(l=2, m=-3)


# =============================================================================
# TEST: RENDER HARMONIC GLYPH
# =============================================================================

class TestRenderHarmonicGlyph:
    """Test rendering HarmonicGlyph from ScalarLayer."""

    def test_extracts_harmonic_root(self):
        """Extracts omega from scalar layer."""
        layer = ScalarLayer(
            depth=1.0,
            amplitude=0.8,
            phase=0.5,
            omega=OmegaFormat(l=4, m=-2),
            layer_index=2
        )
        glyph = render_harmonic_glyph(layer)
        assert glyph.harmonic_root.l == 4
        assert glyph.harmonic_root.m == -2

    def test_extracts_torsion_state(self):
        """Extracts torsion state from layer."""
        layer = ScalarLayer(
            depth=1.0,
            amplitude=0.8,
            phase=0.5,
            omega=OmegaFormat(l=2, m=1),
            layer_index=0,
            torsion=TorsionPhase.Inverted
        )
        glyph = render_harmonic_glyph(layer)
        assert glyph.torsion_state == TorsionPhase.Inverted

    def test_computes_coherence_glow(self):
        """Computes coherence glow from layer coherence."""
        layer = ScalarLayer(
            depth=1.0,
            amplitude=0.8,
            phase=0.5,
            omega=OmegaFormat(l=2, m=0),
            layer_index=0,
            coherence=0.75
        )
        glyph = render_harmonic_glyph(layer)
        assert glyph.coherence_glow is not None
        assert glyph.coherence_glow.intensity == int(0.75 * 255)

    def test_uses_fragment_id_when_present(self):
        """Uses fragment ID for glyph ID when present."""
        layer = ScalarLayer(
            depth=1.0,
            amplitude=0.8,
            phase=0.5,
            omega=OmegaFormat(l=2, m=0),
            layer_index=0,
            fragment_id="frag_xyz"
        )
        glyph = render_harmonic_glyph(layer)
        assert glyph.glyph_id == "frag_xyz"
        assert glyph.associated_with == "frag_xyz"


# =============================================================================
# TEST: GENERATE SVG
# =============================================================================

class TestGenerateSVG:
    """Test SVG generation from glyph."""

    def test_svg_contains_required_elements(self):
        """SVG contains circle, path, and g elements."""
        glyph = HarmonicGlyph(
            glyph_id="test_glyph",
            phi_phase=3,
            harmonic_root=OmegaFormat(l=3, m=2),
            torsion_state=TorsionPhase.Normal,
            coherence_glow=GlowLevel(200),
            associated_with=None
        )

        image = generate_glyph_svg(glyph)
        svg_str = image.svg_data.decode('utf-8')

        assert '<svg' in svg_str
        assert '<circle' in svg_str
        assert '<g' in svg_str

    def test_svg_has_rings_for_l(self):
        """SVG contains rings for l value."""
        glyph = HarmonicGlyph(
            glyph_id="test_glyph",
            phi_phase=0,
            harmonic_root=OmegaFormat(l=4, m=0),
            torsion_state=TorsionPhase.Normal,
            coherence_glow=None,
            associated_with=None
        )

        image = generate_glyph_svg(glyph)
        svg_str = image.svg_data.decode('utf-8')

        # Count circle elements (l+1 rings)
        circle_count = svg_str.count('<circle')
        assert circle_count >= 4  # At least l rings

    def test_svg_has_spiral_paths_for_nonzero_m(self):
        """SVG contains path elements for non-zero m."""
        glyph = HarmonicGlyph(
            glyph_id="test_glyph",
            phi_phase=0,
            harmonic_root=OmegaFormat(l=3, m=3),
            torsion_state=TorsionPhase.Normal,
            coherence_glow=None,
            associated_with=None
        )

        image = generate_glyph_svg(glyph)
        svg_str = image.svg_data.decode('utf-8')

        assert '<path' in svg_str

    def test_svg_color_reflects_torsion(self):
        """SVG color reflects torsion state."""
        for torsion, expected_color in [
            (TorsionPhase.Normal, "#0000ff"),    # Blue
            (TorsionPhase.Inverted, "#ff0000"),  # Red
            (TorsionPhase.Null, "#808080"),      # Grey
        ]:
            glyph = HarmonicGlyph(
                glyph_id="test",
                phi_phase=0,
                harmonic_root=OmegaFormat(l=1, m=0),
                torsion_state=torsion,
                coherence_glow=None,
                associated_with=None
            )
            image = generate_glyph_svg(glyph)
            svg_str = image.svg_data.decode('utf-8').lower()
            assert expected_color in svg_str

    def test_legend_contains_glyph_info(self):
        """Legend contains glyph information."""
        glyph = HarmonicGlyph(
            glyph_id="test_glyph",
            phi_phase=5,
            harmonic_root=OmegaFormat(l=3, m=-1),
            torsion_state=TorsionPhase.Inverted,
            coherence_glow=GlowLevel(180),
            associated_with=None
        )

        image = generate_glyph_svg(glyph)

        assert any("phi^5" in line for line in image.legend)
        assert any("l=3" in line and "m=-1" in line for line in image.legend)
        assert any("Inverted" in line for line in image.legend)
        assert any("180" in line for line in image.legend)


# =============================================================================
# TEST: FRAGMENT ANCHOR GLYPH
# =============================================================================

class TestFragmentAnchorGlyph:
    """Test fragment anchor glyph generation."""

    def test_creates_glyph_from_fragment_metadata(self):
        """Creates valid glyph image from fragment metadata."""
        metadata = FragmentMetadata(
            fragment_id="anchor_001",
            emergence_x=1.5,
            emergence_y=-0.3,
            emergence_z=2.1,
            coherence=0.85,
            harmonic_root=OmegaFormat(l=2, m=1)
        )

        image = fragment_anchor_glyph(metadata)

        assert image.svg_data is not None
        assert len(image.svg_data) > 0

    def test_legend_includes_coordinates(self):
        """Legend includes emergence coordinates."""
        metadata = FragmentMetadata(
            fragment_id="anchor_002",
            emergence_x=1.23,
            emergence_y=4.56,
            emergence_z=7.89,
            coherence=0.9,
            harmonic_root=OmegaFormat(l=3, m=0)
        )

        image = fragment_anchor_glyph(metadata)

        coord_legend = [l for l in image.legend if "Anchor" in l]
        assert len(coord_legend) == 1
        assert "1.23" in coord_legend[0]
        assert "4.56" in coord_legend[0]
        assert "7.89" in coord_legend[0]

    def test_legend_includes_fragment_id(self):
        """Legend includes fragment ID."""
        metadata = FragmentMetadata(
            fragment_id="frag_special_123",
            emergence_x=0, emergence_y=0, emergence_z=0,
            coherence=0.5,
            harmonic_root=OmegaFormat(l=1, m=0)
        )

        image = fragment_anchor_glyph(metadata)

        assert any("frag_special_123" in line for line in image.legend)


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestGlyphIntegration:
    """Integration tests for glyph rendering pipeline."""

    def test_full_pipeline_scalar_to_svg(self):
        """Full pipeline from ScalarLayer to SVG image."""
        layer = ScalarLayer(
            depth=PHI ** 3,  # phi^3 depth
            amplitude=0.9,
            phase=0.25,
            omega=OmegaFormat(l=5, m=-3),
            layer_index=3,
            torsion=TorsionPhase.Normal,
            coherence=0.82
        )

        glyph = render_harmonic_glyph(layer)
        image = generate_glyph_svg(glyph)

        assert image.svg_data is not None
        assert image.timestamp > 0
        assert len(image.legend) >= 3

    def test_multiple_layers_unique_glyphs(self):
        """Multiple layers produce unique glyphs."""
        layers = [
            ScalarLayer(
                depth=PHI ** i,
                amplitude=0.8,
                phase=0.1 * i,
                omega=OmegaFormat(l=i, m=0),
                layer_index=i
            )
            for i in range(5)
        ]

        glyphs = [render_harmonic_glyph(layer) for layer in layers]
        glyph_ids = [g.glyph_id for g in glyphs]

        # All IDs should be unique
        assert len(set(glyph_ids)) == len(glyph_ids)

    def test_high_coherence_produces_glow(self):
        """High coherence layers produce visible glow in SVG."""
        layer = ScalarLayer(
            depth=1.0,
            amplitude=1.0,
            phase=0,
            omega=OmegaFormat(l=2, m=0),
            layer_index=0,
            coherence=0.95
        )

        glyph = render_harmonic_glyph(layer)
        image = generate_glyph_svg(glyph)
        svg_str = image.svg_data.decode('utf-8')

        # High coherence should produce white glow
        assert 'white' in svg_str.lower() or 'fff' in svg_str.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
