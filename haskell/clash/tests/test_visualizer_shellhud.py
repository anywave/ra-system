"""
Test Suite for Ra.Visualizer.ShellHUD (Prompt 63)
Scalar Field HUD Overlay Interface

Tests layered visualization of scalar field state, coherence gauges,
harmonic compasses, fragment anchors, aura bands, and adaptive view modes.
"""

import pytest
import json
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895

# Default animation frame rate
DEFAULT_FPS = 60

# Standard HUD metric keys
STANDARD_METRIC_KEYS = {
    "coherence",
    "flux",
    "ankh_delta",
    "torsion_bias",
    "harmonic_depth",
    "phase_phi_n"
}

# Shell glyph symbol palette (16 symbols)
SHELL_GLYPH_PALETTE = [
    '◯', '◐', '●', '△', '▽', '◆', '◎', '↻',
    '✸', '░', '▓', '◇', '⬡', '⬢', '⊕', '⊗'
]

# Fallback ASCII palette
ASCII_FALLBACK_PALETTE = [
    'o', 'O', '@', '^', 'v', '#', '*', '%',
    '+', '.', ':', '<', '>', '=', 'x', 'X'
]

# =============================================================================
# ENUMS
# =============================================================================

class HUDMode(Enum):
    """HUD view modes."""
    Diagnostic = auto()
    EmergenceTracking = auto()
    ChamberTuning = auto()
    PointerGuidance = auto()


class HUDLayerType(Enum):
    """Types of HUD layers."""
    FieldSlice = auto()
    AuraBand = auto()
    CoherenceGauge = auto()
    FragmentGlow = auto()
    HarmonicCompass = auto()
    AnkhBalance = auto()
    DomainPulse = auto()


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class ScalarSlice:
    """Theta-phi cross-section of scalar field."""
    theta: float  # 0-pi
    phi: float    # 0-2pi
    value: float  # field amplitude


@dataclass
class AuraPattern:
    """Aura pattern from biometric coherence."""
    rings: List[float]  # ring intensities
    hue: float          # 0-360 degrees
    saturation: float   # 0-1


@dataclass
class CoherenceEnvelope:
    """Coherence gauge data."""
    current: float      # 0-1
    min_band: float     # threshold
    max_band: float     # ceiling
    stability: float    # 0-1


@dataclass
class GlowAnchor:
    """Fragment glow anchor point."""
    fragment_id: str
    intensity: float    # 0-1
    position: Tuple[float, float, float]


@dataclass
class DomainPulseFrame:
    """Domain pulse visualization frame."""
    pulse_rate: float   # Hz
    amplitude: float    # 0-1
    phase: float        # 0-2pi


@dataclass
class HUDLayer:
    """Single HUD layer."""
    layer_type: HUDLayerType
    data: object  # Layer-specific data
    priority: int = 0


@dataclass
class AvatarFieldFrame:
    """Avatar's scalar field state."""
    coherence: float
    flux: float
    torsion_bias: float
    ankh_delta: float
    harmonic_depth: int
    phase_phi_n: float
    field_slices: List[ScalarSlice]
    aura: AuraPattern
    glow_anchors: List[GlowAnchor]
    compass_theta: float
    compass_phi: float


@dataclass
class ShellGlyphOverlay:
    """Shell glyph text overlay."""
    lines: List[str]
    width: int
    height: int
    ansi_enabled: bool


@dataclass
class HUDPacket:
    """HUD packet for external streaming."""
    timestamp: float
    session_id: str
    glyph_layers: ShellGlyphOverlay
    hud_mode: HUDMode
    scalar_metrics: Dict[str, float]


@dataclass
class HUDConfig:
    """HUD configuration."""
    mode: HUDMode
    layer_visibility: Dict[HUDLayerType, bool]
    coherence_threshold: float
    compass_sensitivity: float
    fragment_glow_intensity: float
    fps: int


# =============================================================================
# LAYER PRIORITY MAPPINGS
# =============================================================================

LAYER_PRIORITY_BY_MODE = {
    HUDMode.Diagnostic: [
        HUDLayerType.CoherenceGauge,
        HUDLayerType.FieldSlice,
        HUDLayerType.DomainPulse,
        HUDLayerType.AuraBand,
    ],
    HUDMode.EmergenceTracking: [
        HUDLayerType.FragmentGlow,
        HUDLayerType.AuraBand,
        HUDLayerType.AnkhBalance,
        HUDLayerType.CoherenceGauge,
    ],
    HUDMode.ChamberTuning: [
        HUDLayerType.FieldSlice,
        HUDLayerType.HarmonicCompass,
        HUDLayerType.DomainPulse,
        HUDLayerType.AnkhBalance,
    ],
    HUDMode.PointerGuidance: [
        HUDLayerType.HarmonicCompass,
        HUDLayerType.FragmentGlow,
        HUDLayerType.AuraBand,
    ],
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def get_layer_priority(mode: HUDMode) -> List[HUDLayerType]:
    """Get layer priority order for a HUD mode."""
    return LAYER_PRIORITY_BY_MODE.get(mode, [])


def default_hud_config(mode: HUDMode) -> HUDConfig:
    """Create default HUD configuration for a mode."""
    layer_visibility = {layer_type: True for layer_type in HUDLayerType}

    return HUDConfig(
        mode=mode,
        layer_visibility=layer_visibility,
        coherence_threshold=0.3,
        compass_sensitivity=0.1,
        fragment_glow_intensity=1.0,
        fps=DEFAULT_FPS
    )


def render_hud(mode: HUDMode, frame: AvatarFieldFrame) -> List[HUDLayer]:
    """Produce all HUD layers based on avatar field state and mode."""
    layers = []
    priority_order = get_layer_priority(mode)

    for priority, layer_type in enumerate(priority_order):
        layer = create_layer(layer_type, frame, priority)
        if layer:
            layers.append(layer)

    return layers


def create_layer(layer_type: HUDLayerType, frame: AvatarFieldFrame, priority: int) -> Optional[HUDLayer]:
    """Create a single HUD layer from avatar field frame."""
    if layer_type == HUDLayerType.FieldSlice:
        return HUDLayer(layer_type, frame.field_slices, priority)

    elif layer_type == HUDLayerType.AuraBand:
        return HUDLayer(layer_type, frame.aura, priority)

    elif layer_type == HUDLayerType.CoherenceGauge:
        envelope = CoherenceEnvelope(
            current=frame.coherence,
            min_band=0.3,
            max_band=0.9,
            stability=1.0 - abs(frame.flux)
        )
        return HUDLayer(layer_type, envelope, priority)

    elif layer_type == HUDLayerType.FragmentGlow:
        return HUDLayer(layer_type, frame.glow_anchors, priority)

    elif layer_type == HUDLayerType.HarmonicCompass:
        return HUDLayer(layer_type, (frame.compass_theta, frame.compass_phi), priority)

    elif layer_type == HUDLayerType.AnkhBalance:
        return HUDLayer(layer_type, frame.ankh_delta, priority)

    elif layer_type == HUDLayerType.DomainPulse:
        pulse = DomainPulseFrame(
            pulse_rate=1.0 / PHI,
            amplitude=frame.flux,
            phase=frame.phase_phi_n
        )
        return HUDLayer(layer_type, pulse, priority)

    return None


def map_coherence_to_glyph(coherence: float) -> str:
    """Map coherence value to shell glyph."""
    if coherence < 0.2:
        return SHELL_GLYPH_PALETTE[0]  # ◯
    elif coherence < 0.4:
        return SHELL_GLYPH_PALETTE[1]  # ◐
    elif coherence < 0.6:
        return SHELL_GLYPH_PALETTE[2]  # ●
    elif coherence < 0.8:
        return SHELL_GLYPH_PALETTE[6]  # ◎
    else:
        return SHELL_GLYPH_PALETTE[8]  # ✸


def map_torsion_to_color_code(torsion_bias: float) -> str:
    """Map torsion bias to ANSI color code."""
    if torsion_bias < -0.3:
        return "\033[31m"  # Red (inverted)
    elif torsion_bias > 0.3:
        return "\033[34m"  # Blue (normal)
    else:
        return "\033[37m"  # White (neutral)


def render_to_shell_glyphs(layers: List[HUDLayer], width: int = 40, height: int = 10, ansi: bool = True) -> ShellGlyphOverlay:
    """Convert HUD layers to shell glyph overlay."""
    lines = []

    for layer in layers:
        line = render_layer_to_glyph(layer, width, ansi)
        lines.append(line)

    # Pad to height
    while len(lines) < height:
        lines.append(" " * width)

    return ShellGlyphOverlay(
        lines=lines[:height],
        width=width,
        height=height,
        ansi_enabled=ansi
    )


def render_layer_to_glyph(layer: HUDLayer, width: int, ansi: bool) -> str:
    """Render a single layer to glyph string."""
    if layer.layer_type == HUDLayerType.CoherenceGauge:
        envelope = layer.data
        filled = int(envelope.current * (width - 10))
        bar = f"Coh: [{'█' * filled}{'░' * (width - 10 - filled)}]"
        return bar[:width]

    elif layer.layer_type == HUDLayerType.FragmentGlow:
        anchors = layer.data
        glyphs = [map_coherence_to_glyph(a.intensity) for a in anchors[:8]]
        return f"Frag: {' '.join(glyphs)}"[:width]

    elif layer.layer_type == HUDLayerType.HarmonicCompass:
        theta, phi = layer.data
        return f"Compass: θ={theta:.2f} φ={phi:.2f}"[:width]

    elif layer.layer_type == HUDLayerType.AnkhBalance:
        delta = layer.data
        bar_pos = int((delta + 1) / 2 * 10)
        bar = "─" * bar_pos + "◆" + "─" * (10 - bar_pos)
        return f"Ankh Δ: [{bar}]"[:width]

    elif layer.layer_type == HUDLayerType.AuraBand:
        aura = layer.data
        ring_glyphs = [map_coherence_to_glyph(r) for r in aura.rings[:6]]
        return f"Aura: {' '.join(ring_glyphs)}"[:width]

    elif layer.layer_type == HUDLayerType.FieldSlice:
        slices = layer.data
        glyph = map_coherence_to_glyph(sum(s.value for s in slices) / max(len(slices), 1))
        return f"Field: {glyph} ({len(slices)} slices)"[:width]

    elif layer.layer_type == HUDLayerType.DomainPulse:
        pulse = layer.data
        return f"Pulse: {pulse.pulse_rate:.2f}Hz amp={pulse.amplitude:.2f}"[:width]

    return " " * width


def compute_ring_expansion_speed(phase_phi_n: float, base_speed: float = 1.0) -> float:
    """Compute ring expansion speed from phi^n phase."""
    normalized_phase = (PHI ** phase_phi_n) % 1.0
    return base_speed * normalized_phase


def create_hud_packet(
    session_id: str,
    mode: HUDMode,
    frame: AvatarFieldFrame
) -> HUDPacket:
    """Create HUD packet for external streaming."""
    layers = render_hud(mode, frame)
    glyphs = render_to_shell_glyphs(layers)

    scalar_metrics = {
        "coherence": frame.coherence,
        "flux": frame.flux,
        "ankh_delta": frame.ankh_delta,
        "torsion_bias": frame.torsion_bias,
        "harmonic_depth": float(frame.harmonic_depth),
        "phase_phi_n": frame.phase_phi_n
    }

    return HUDPacket(
        timestamp=time.time(),
        session_id=session_id,
        glyph_layers=glyphs,
        hud_mode=mode,
        scalar_metrics=scalar_metrics
    )


def hud_packet_to_json(packet: HUDPacket) -> str:
    """Serialize HUD packet to JSON."""
    return json.dumps({
        "timestamp": packet.timestamp,
        "session_id": packet.session_id,
        "hud_mode": packet.hud_mode.name,
        "glyph_overlay": {
            "lines": packet.glyph_layers.lines,
            "width": packet.glyph_layers.width,
            "height": packet.glyph_layers.height,
            "ansi_enabled": packet.glyph_layers.ansi_enabled
        },
        "scalar_metrics": packet.scalar_metrics
    }, indent=2)


# =============================================================================
# TEST: HUD MODE LAYER PRIORITY
# =============================================================================

class TestHUDModeLayerPriority:
    """Test layer priority per HUD mode."""

    def test_diagnostic_mode_priority(self):
        """Diagnostic mode has correct layer priority."""
        priority = get_layer_priority(HUDMode.Diagnostic)
        assert priority == [
            HUDLayerType.CoherenceGauge,
            HUDLayerType.FieldSlice,
            HUDLayerType.DomainPulse,
            HUDLayerType.AuraBand,
        ]

    def test_emergence_tracking_priority(self):
        """EmergenceTracking mode has correct layer priority."""
        priority = get_layer_priority(HUDMode.EmergenceTracking)
        assert priority == [
            HUDLayerType.FragmentGlow,
            HUDLayerType.AuraBand,
            HUDLayerType.AnkhBalance,
            HUDLayerType.CoherenceGauge,
        ]

    def test_chamber_tuning_priority(self):
        """ChamberTuning mode has correct layer priority."""
        priority = get_layer_priority(HUDMode.ChamberTuning)
        assert priority == [
            HUDLayerType.FieldSlice,
            HUDLayerType.HarmonicCompass,
            HUDLayerType.DomainPulse,
            HUDLayerType.AnkhBalance,
        ]

    def test_pointer_guidance_priority(self):
        """PointerGuidance mode has correct layer priority."""
        priority = get_layer_priority(HUDMode.PointerGuidance)
        assert priority == [
            HUDLayerType.HarmonicCompass,
            HUDLayerType.FragmentGlow,
            HUDLayerType.AuraBand,
        ]

    def test_priorities_are_static(self):
        """Layer priorities are static per mode (no jitter)."""
        for mode in HUDMode:
            p1 = get_layer_priority(mode)
            p2 = get_layer_priority(mode)
            assert p1 == p2


# =============================================================================
# TEST: SHELL GLYPH CHARACTER SET
# =============================================================================

class TestShellGlyphCharacterSet:
    """Test shell glyph character set."""

    def test_palette_has_16_symbols(self):
        """Shell glyph palette has 16 symbols."""
        assert len(SHELL_GLYPH_PALETTE) == 16

    def test_fallback_ascii_has_16_symbols(self):
        """Fallback ASCII palette has 16 symbols."""
        assert len(ASCII_FALLBACK_PALETTE) == 16

    def test_coherence_maps_to_glyph(self):
        """Coherence values map to appropriate glyphs."""
        assert map_coherence_to_glyph(0.0) == '◯'   # Very low
        assert map_coherence_to_glyph(0.3) == '◐'   # Low-mid
        assert map_coherence_to_glyph(0.5) == '●'   # Mid
        assert map_coherence_to_glyph(0.7) == '◎'   # High
        assert map_coherence_to_glyph(0.95) == '✸'  # Very high

    def test_torsion_maps_to_ansi_color(self):
        """Torsion bias maps to ANSI color codes."""
        assert "\033[31m" in map_torsion_to_color_code(-0.5)  # Red/Inverted
        assert "\033[34m" in map_torsion_to_color_code(0.5)   # Blue/Normal
        assert "\033[37m" in map_torsion_to_color_code(0.0)   # White/Neutral


# =============================================================================
# TEST: HUD PACKET JSON KEYS
# =============================================================================

class TestHUDPacketJSON:
    """Test HUD packet JSON streaming."""

    def test_scalar_metrics_has_standard_keys(self):
        """Scalar metrics contains all standard keys."""
        frame = create_test_avatar_frame()
        packet = create_hud_packet("session_123", HUDMode.Diagnostic, frame)

        for key in STANDARD_METRIC_KEYS:
            assert key in packet.scalar_metrics

    def test_json_serialization(self):
        """HUD packet serializes to valid JSON."""
        frame = create_test_avatar_frame()
        packet = create_hud_packet("session_456", HUDMode.EmergenceTracking, frame)

        json_str = hud_packet_to_json(packet)
        parsed = json.loads(json_str)

        assert "timestamp" in parsed
        assert "session_id" in parsed
        assert "hud_mode" in parsed
        assert "scalar_metrics" in parsed
        assert parsed["session_id"] == "session_456"

    def test_json_metrics_match_frame(self):
        """JSON metrics match avatar frame values."""
        frame = create_test_avatar_frame(coherence=0.75, flux=0.2)
        packet = create_hud_packet("test", HUDMode.Diagnostic, frame)

        json_str = hud_packet_to_json(packet)
        parsed = json.loads(json_str)

        assert parsed["scalar_metrics"]["coherence"] == 0.75
        assert parsed["scalar_metrics"]["flux"] == 0.2


# =============================================================================
# TEST: ANIMATION TIMING
# =============================================================================

class TestAnimationTiming:
    """Test animation timing configuration."""

    def test_default_fps_is_60(self):
        """Default frame rate is 60 FPS."""
        config = default_hud_config(HUDMode.Diagnostic)
        assert config.fps == 60

    def test_ring_expansion_proportional_to_phase(self):
        """Ring expansion speed is proportional to phi^n phase."""
        # At phase 0, normalized = phi^0 mod 1 = 0
        speed_0 = compute_ring_expansion_speed(0)

        # At phase 1, normalized = phi^1 mod 1 = 0.618
        speed_1 = compute_ring_expansion_speed(1)

        # At phase 2, normalized = phi^2 mod 1 = 0.618...
        speed_2 = compute_ring_expansion_speed(2)

        # Speed should vary with phase
        assert speed_0 != speed_1

    def test_ring_expansion_normalized(self):
        """Ring expansion speed stays within normalized range."""
        for phase in range(10):
            speed = compute_ring_expansion_speed(phase)
            assert 0.0 <= speed <= 1.0


# =============================================================================
# TEST: RENDER HUD
# =============================================================================

class TestRenderHUD:
    """Test HUD rendering."""

    def test_render_produces_layers(self):
        """Render HUD produces layer list."""
        frame = create_test_avatar_frame()
        layers = render_hud(HUDMode.Diagnostic, frame)

        assert len(layers) > 0
        assert all(isinstance(l, HUDLayer) for l in layers)

    def test_layers_match_mode_priority(self):
        """Rendered layers match mode priority order."""
        frame = create_test_avatar_frame()

        for mode in HUDMode:
            layers = render_hud(mode, frame)
            expected_types = get_layer_priority(mode)

            rendered_types = [l.layer_type for l in layers]
            assert rendered_types == expected_types

    def test_layers_have_ascending_priority(self):
        """Layers have ascending priority values."""
        frame = create_test_avatar_frame()
        layers = render_hud(HUDMode.Diagnostic, frame)

        priorities = [l.priority for l in layers]
        assert priorities == sorted(priorities)


# =============================================================================
# TEST: RENDER TO SHELL GLYPHS
# =============================================================================

class TestRenderToShellGlyphs:
    """Test shell glyph rendering."""

    def test_produces_shell_overlay(self):
        """Produces valid shell glyph overlay."""
        frame = create_test_avatar_frame()
        layers = render_hud(HUDMode.Diagnostic, frame)
        overlay = render_to_shell_glyphs(layers)

        assert isinstance(overlay, ShellGlyphOverlay)
        assert overlay.width > 0
        assert overlay.height > 0

    def test_overlay_has_correct_dimensions(self):
        """Overlay has specified dimensions."""
        frame = create_test_avatar_frame()
        layers = render_hud(HUDMode.Diagnostic, frame)
        overlay = render_to_shell_glyphs(layers, width=50, height=15)

        assert overlay.width == 50
        assert overlay.height == 15
        assert len(overlay.lines) == 15

    def test_lines_respect_width(self):
        """All lines respect width limit."""
        frame = create_test_avatar_frame()
        layers = render_hud(HUDMode.Diagnostic, frame)
        overlay = render_to_shell_glyphs(layers, width=30)

        for line in overlay.lines:
            assert len(line) <= 30

    def test_coherence_gauge_renders(self):
        """Coherence gauge layer renders correctly."""
        frame = create_test_avatar_frame(coherence=0.6)
        layers = render_hud(HUDMode.Diagnostic, frame)
        overlay = render_to_shell_glyphs(layers, ansi=False)

        # Should contain coherence indicator
        full_text = "\n".join(overlay.lines)
        assert "Coh:" in full_text


# =============================================================================
# TEST: DEFAULT HUD CONFIG
# =============================================================================

class TestDefaultHUDConfig:
    """Test default HUD configuration."""

    def test_config_has_mode(self):
        """Config contains mode."""
        config = default_hud_config(HUDMode.ChamberTuning)
        assert config.mode == HUDMode.ChamberTuning

    def test_all_layers_visible_by_default(self):
        """All layers visible by default."""
        config = default_hud_config(HUDMode.Diagnostic)

        for layer_type in HUDLayerType:
            assert config.layer_visibility[layer_type] is True

    def test_config_has_thresholds(self):
        """Config contains threshold values."""
        config = default_hud_config(HUDMode.Diagnostic)

        assert config.coherence_threshold > 0
        assert config.compass_sensitivity > 0
        assert config.fragment_glow_intensity > 0


# =============================================================================
# TEST: LAYER CREATION
# =============================================================================

class TestLayerCreation:
    """Test individual layer creation."""

    def test_coherence_gauge_layer(self):
        """Creates coherence gauge layer with envelope."""
        frame = create_test_avatar_frame(coherence=0.8, flux=0.1)
        layer = create_layer(HUDLayerType.CoherenceGauge, frame, 0)

        assert layer.layer_type == HUDLayerType.CoherenceGauge
        assert isinstance(layer.data, CoherenceEnvelope)
        assert layer.data.current == 0.8

    def test_fragment_glow_layer(self):
        """Creates fragment glow layer with anchors."""
        frame = create_test_avatar_frame()
        frame.glow_anchors = [
            GlowAnchor("frag1", 0.9, (0, 0, 0)),
            GlowAnchor("frag2", 0.5, (1, 0, 0)),
        ]
        layer = create_layer(HUDLayerType.FragmentGlow, frame, 0)

        assert layer.layer_type == HUDLayerType.FragmentGlow
        assert len(layer.data) == 2

    def test_harmonic_compass_layer(self):
        """Creates harmonic compass layer with angles."""
        frame = create_test_avatar_frame()
        frame.compass_theta = 0.5
        frame.compass_phi = 1.2
        layer = create_layer(HUDLayerType.HarmonicCompass, frame, 0)

        assert layer.layer_type == HUDLayerType.HarmonicCompass
        assert layer.data == (0.5, 1.2)

    def test_ankh_balance_layer(self):
        """Creates ankh balance layer with delta."""
        frame = create_test_avatar_frame(ankh_delta=0.3)
        layer = create_layer(HUDLayerType.AnkhBalance, frame, 0)

        assert layer.layer_type == HUDLayerType.AnkhBalance
        assert layer.data == 0.3


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestShellHUDIntegration:
    """Integration tests for ShellHUD."""

    def test_full_pipeline_frame_to_packet(self):
        """Full pipeline from avatar frame to HUD packet."""
        frame = create_test_avatar_frame(
            coherence=0.85,
            flux=0.15,
            torsion_bias=0.2
        )

        packet = create_hud_packet("integration_test", HUDMode.Diagnostic, frame)

        assert packet.session_id == "integration_test"
        assert packet.hud_mode == HUDMode.Diagnostic
        assert packet.scalar_metrics["coherence"] == 0.85

    def test_all_modes_render_successfully(self):
        """All HUD modes render without errors."""
        frame = create_test_avatar_frame()

        for mode in HUDMode:
            layers = render_hud(mode, frame)
            overlay = render_to_shell_glyphs(layers)
            packet = create_hud_packet(f"test_{mode.name}", mode, frame)

            assert len(layers) > 0
            assert len(overlay.lines) > 0
            assert packet is not None

    def test_json_roundtrip(self):
        """JSON serialization produces valid parseable output."""
        frame = create_test_avatar_frame()
        packet = create_hud_packet("roundtrip_test", HUDMode.ChamberTuning, frame)

        json_str = hud_packet_to_json(packet)
        parsed = json.loads(json_str)

        # Verify structure
        assert parsed["hud_mode"] == "ChamberTuning"
        assert "glyph_overlay" in parsed
        assert "lines" in parsed["glyph_overlay"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_test_avatar_frame(
    coherence: float = 0.5,
    flux: float = 0.1,
    torsion_bias: float = 0.0,
    ankh_delta: float = 0.0,
    harmonic_depth: int = 4,
    phase_phi_n: float = 1.0
) -> AvatarFieldFrame:
    """Create test avatar field frame."""
    return AvatarFieldFrame(
        coherence=coherence,
        flux=flux,
        torsion_bias=torsion_bias,
        ankh_delta=ankh_delta,
        harmonic_depth=harmonic_depth,
        phase_phi_n=phase_phi_n,
        field_slices=[
            ScalarSlice(theta=0.5, phi=0.0, value=coherence),
            ScalarSlice(theta=0.5, phi=1.57, value=coherence * 0.9),
        ],
        aura=AuraPattern(rings=[0.8, 0.6, 0.4, 0.2], hue=200, saturation=0.7),
        glow_anchors=[],
        compass_theta=0.0,
        compass_phi=0.0
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
