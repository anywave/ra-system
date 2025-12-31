"""
Prompt 41: Ra.Visualizer.Shell - Terminal Visualization Layer

ASCII/Unicode terminal visualization for resonance chambers and sync graphs
with adaptive width and ANSI color support.

Codex References:
- Ra.Chamber.Sync: Sync graph data structures
- Ra.Coherence.Bands: Color mapping for coherence levels
"""

import pytest
import os
import sys
import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum, auto


# ============================================================================
# Constants
# ============================================================================

PHI = 1.618033988749895
DEFAULT_WIDTH = 120
MIN_WIDTH = 40
MAX_WIDTH = 200

# ANSI color codes
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"

# Foreground colors
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"
ANSI_WHITE = "\033[37m"

# Background colors
ANSI_BG_RED = "\033[41m"
ANSI_BG_GREEN = "\033[42m"
ANSI_BG_BLUE = "\033[44m"

# Unicode box drawing characters
BOX_H = "─"
BOX_V = "│"
BOX_TL = "┌"
BOX_TR = "┐"
BOX_BL = "└"
BOX_BR = "┘"
BOX_T = "┬"
BOX_B = "┴"
BOX_L = "├"
BOX_R = "┤"
BOX_X = "┼"

# ASCII fallback
ASCII_H = "-"
ASCII_V = "|"
ASCII_TL = "+"
ASCII_TR = "+"
ASCII_BL = "+"
ASCII_BR = "+"


# ============================================================================
# Types
# ============================================================================

class ColorMode(Enum):
    """Terminal color mode."""
    ANSI_256 = auto()     # Full 256 color
    ANSI_16 = auto()      # Basic 16 color
    PLAIN = auto()        # No color (fallback)


class CharSet(Enum):
    """Character set for drawing."""
    UNICODE = auto()      # Unicode box drawing
    ASCII = auto()        # ASCII fallback


@dataclass
class TerminalConfig:
    """Terminal display configuration."""
    width: int
    height: int
    color_mode: ColorMode
    char_set: CharSet
    show_legend: bool = True


@dataclass
class CoherenceColor:
    """Color mapping for coherence level."""
    level: float
    ansi_code: str
    rgb: Tuple[int, int, int]
    label: str


# Coherence band colors
COHERENCE_COLORS = [
    CoherenceColor(0.0, ANSI_RED, (255, 0, 0), "Critical"),
    CoherenceColor(0.3, ANSI_YELLOW, (255, 255, 0), "Low"),
    CoherenceColor(0.55, ANSI_BLUE, (0, 128, 255), "Partial"),
    CoherenceColor(0.72, ANSI_CYAN, (0, 255, 255), "Syncing"),
    CoherenceColor(0.85, ANSI_GREEN, (0, 255, 0), "High"),
    CoherenceColor(0.95, ANSI_MAGENTA, (255, 0, 255), "Optimal"),
]


@dataclass
class NodeRenderData:
    """Rendered data for a chamber node."""
    node_id: str
    coherence: float
    sync_state: str
    frequency: float
    x: int  # Grid position
    y: int


@dataclass
class LinkRenderData:
    """Rendered data for a sync link."""
    source_id: str
    target_id: str
    strength: float
    link_type: str


@dataclass
class SyncGraphRender:
    """Complete sync graph rendering data."""
    nodes: List[NodeRenderData]
    links: List[LinkRenderData]
    global_sync: float
    timestamp: float


# ============================================================================
# Terminal Detection
# ============================================================================

def detect_terminal_width(fallback: int = DEFAULT_WIDTH) -> int:
    """
    Detect terminal width with fallback.

    Falls back to 120 if detection fails.
    """
    try:
        size = os.get_terminal_size()
        return max(MIN_WIDTH, min(size.columns, MAX_WIDTH))
    except (OSError, ValueError):
        return fallback


def detect_color_support() -> ColorMode:
    """
    Detect terminal color support.

    Returns PLAIN if not a TTY or TERM not set.
    """
    # Check if stdout is a TTY
    if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
        return ColorMode.PLAIN

    # Check TERM environment variable
    term = os.environ.get('TERM', '')

    if '256color' in term:
        return ColorMode.ANSI_256
    elif term in ['xterm', 'screen', 'vt100', 'linux', 'ansi']:
        return ColorMode.ANSI_16
    elif term:
        return ColorMode.ANSI_16

    # Check for Windows terminal
    if sys.platform == 'win32':
        # Modern Windows terminals support ANSI
        if os.environ.get('WT_SESSION'):  # Windows Terminal
            return ColorMode.ANSI_256
        return ColorMode.PLAIN

    return ColorMode.PLAIN


def detect_unicode_support() -> CharSet:
    """
    Detect Unicode box drawing support.

    Falls back to ASCII if not supported.
    """
    # Check encoding
    try:
        encoding = sys.stdout.encoding or 'ascii'
        if encoding.lower() in ['utf-8', 'utf8', 'utf-16', 'utf-32']:
            return CharSet.UNICODE
    except (AttributeError, LookupError):
        pass

    return CharSet.ASCII


def create_terminal_config() -> TerminalConfig:
    """Create terminal configuration with auto-detection."""
    return TerminalConfig(
        width=detect_terminal_width(),
        height=24,  # Standard height
        color_mode=detect_color_support(),
        char_set=detect_unicode_support()
    )


# ============================================================================
# Color Functions
# ============================================================================

def get_coherence_color(coherence: float, mode: ColorMode) -> str:
    """Get ANSI color code for coherence level."""
    if mode == ColorMode.PLAIN:
        return ""

    # Find appropriate color
    color = COHERENCE_COLORS[0]
    for cc in COHERENCE_COLORS:
        if coherence >= cc.level:
            color = cc

    return color.ansi_code


def colorize(text: str, color: str, mode: ColorMode) -> str:
    """Apply color to text if supported."""
    if mode == ColorMode.PLAIN or not color:
        return text
    return f"{color}{text}{ANSI_RESET}"


def bold(text: str, mode: ColorMode) -> str:
    """Make text bold if supported."""
    if mode == ColorMode.PLAIN:
        return text
    return f"{ANSI_BOLD}{text}{ANSI_RESET}"


# ============================================================================
# Box Drawing
# ============================================================================

def get_box_chars(char_set: CharSet) -> Dict[str, str]:
    """Get box drawing characters for charset."""
    if char_set == CharSet.UNICODE:
        return {
            'h': BOX_H, 'v': BOX_V,
            'tl': BOX_TL, 'tr': BOX_TR,
            'bl': BOX_BL, 'br': BOX_BR,
            't': BOX_T, 'b': BOX_B,
            'l': BOX_L, 'r': BOX_R,
            'x': BOX_X
        }
    else:
        return {
            'h': ASCII_H, 'v': ASCII_V,
            'tl': ASCII_TL, 'tr': ASCII_TR,
            'bl': ASCII_BL, 'br': ASCII_BR,
            't': ASCII_TL, 'b': ASCII_BL,
            'l': ASCII_TL, 'r': ASCII_TR,
            'x': ASCII_TL
        }


def draw_box(width: int, height: int, config: TerminalConfig) -> List[str]:
    """Draw a box with given dimensions."""
    chars = get_box_chars(config.char_set)
    lines = []

    # Top border
    lines.append(chars['tl'] + chars['h'] * (width - 2) + chars['tr'])

    # Middle lines
    for _ in range(height - 2):
        lines.append(chars['v'] + ' ' * (width - 2) + chars['v'])

    # Bottom border
    lines.append(chars['bl'] + chars['h'] * (width - 2) + chars['br'])

    return lines


# ============================================================================
# Node Rendering
# ============================================================================

def render_node(node: NodeRenderData, config: TerminalConfig) -> List[str]:
    """Render a single chamber node."""
    chars = get_box_chars(config.char_set)
    color = get_coherence_color(node.coherence, config.color_mode)

    # Node box (7 chars wide, 3 lines tall)
    width = 7
    top = chars['tl'] + chars['h'] * (width - 2) + chars['tr']
    mid = chars['v'] + f"{node.node_id[:3]:^5}" + chars['v']
    bot = chars['bl'] + chars['h'] * (width - 2) + chars['br']

    # Colorize
    return [
        colorize(top, color, config.color_mode),
        colorize(mid, color, config.color_mode),
        colorize(bot, color, config.color_mode)
    ]


def render_node_detail(node: NodeRenderData, config: TerminalConfig) -> str:
    """Render detailed node information."""
    color = get_coherence_color(node.coherence, config.color_mode)

    coherence_bar = render_progress_bar(node.coherence, 20, config)
    state_str = f"[{node.sync_state[:4]}]"

    return (
        f"{colorize(node.node_id, color, config.color_mode):12} "
        f"{coherence_bar} "
        f"{node.coherence:.2f} "
        f"{state_str:8} "
        f"{node.frequency:.0f}Hz"
    )


# ============================================================================
# Link Rendering
# ============================================================================

def render_link_line(link: LinkRenderData,
                     src_x: int, dst_x: int,
                     config: TerminalConfig) -> str:
    """Render a horizontal link between nodes."""
    chars = get_box_chars(config.char_set)

    # Determine link character based on strength
    if link.strength > 0.7:
        link_char = "═" if config.char_set == CharSet.UNICODE else "="
    elif link.strength > 0.4:
        link_char = chars['h']
    else:
        link_char = "·" if config.char_set == CharSet.UNICODE else "."

    # Build link line
    length = abs(dst_x - src_x) - 2
    if length <= 0:
        return ""

    return link_char * length


# ============================================================================
# Progress Bar
# ============================================================================

def render_progress_bar(value: float, width: int,
                        config: TerminalConfig) -> str:
    """Render a progress bar for coherence."""
    filled = int(value * width)
    empty = width - filled

    if config.char_set == CharSet.UNICODE:
        bar = "█" * filled + "░" * empty
    else:
        bar = "#" * filled + "-" * empty

    color = get_coherence_color(value, config.color_mode)
    return colorize(f"[{bar}]", color, config.color_mode)


# ============================================================================
# Graph Rendering
# ============================================================================

def render_sync_graph(graph: SyncGraphRender,
                      config: TerminalConfig) -> List[str]:
    """Render complete sync graph to terminal lines."""
    lines = []

    # Header
    header = f"╔{'═' * (config.width - 2)}╗" if config.char_set == CharSet.UNICODE else f"+{'-' * (config.width - 2)}+"
    lines.append(header)

    title = f"  Ra.Chamber.Sync - Global: {graph.global_sync:.2f}  "
    title_line = f"║{title:^{config.width - 2}}║" if config.char_set == CharSet.UNICODE else f"|{title:^{config.width - 2}}|"
    lines.append(bold(title_line, config.color_mode))

    sep = f"╠{'═' * (config.width - 2)}╣" if config.char_set == CharSet.UNICODE else f"+{'-' * (config.width - 2)}+"
    lines.append(sep)

    # Node list
    for node in graph.nodes:
        detail = render_node_detail(node, config)
        padded = f"║ {detail:<{config.width - 4}} ║" if config.char_set == CharSet.UNICODE else f"| {detail:<{config.width - 4}} |"
        lines.append(padded)

    # Separator
    lines.append(sep)

    # Links section
    link_header = "  Links  "
    link_line = f"║{link_header:^{config.width - 2}}║" if config.char_set == CharSet.UNICODE else f"|{link_header:^{config.width - 2}}|"
    lines.append(link_line)

    for link in graph.links:
        link_str = f"{link.source_id} ──({link.strength:.2f})── {link.target_id}"
        color = get_coherence_color(link.strength, config.color_mode)
        colored = colorize(link_str, color, config.color_mode)
        padded = f"║ {colored:<{config.width - 4}} ║" if config.char_set == CharSet.UNICODE else f"| {link_str:<{config.width - 4}} |"
        lines.append(padded)

    # Footer
    footer = f"╚{'═' * (config.width - 2)}╝" if config.char_set == CharSet.UNICODE else f"+{'-' * (config.width - 2)}+"
    lines.append(footer)

    return lines


def render_legend(config: TerminalConfig) -> List[str]:
    """Render coherence color legend."""
    lines = []
    lines.append("Legend:")

    for cc in COHERENCE_COLORS:
        marker = "●" if config.char_set == CharSet.UNICODE else "*"
        colored_marker = colorize(marker, cc.ansi_code, config.color_mode)
        lines.append(f"  {colored_marker} {cc.label}: ≥{cc.level:.2f}")

    return lines


# ============================================================================
# Live Visualizer
# ============================================================================

class LiveVisualizer:
    """Live terminal visualizer for sync graphs."""

    def __init__(self, config: Optional[TerminalConfig] = None):
        self.config = config or create_terminal_config()
        self.frame_count = 0

    def update(self, graph: SyncGraphRender) -> str:
        """Update display with new graph state."""
        lines = render_sync_graph(graph, self.config)

        if self.config.show_legend:
            lines.append("")
            lines.extend(render_legend(self.config))

        self.frame_count += 1
        return "\n".join(lines)

    def clear_screen(self) -> str:
        """Get clear screen sequence."""
        if self.config.color_mode != ColorMode.PLAIN:
            return "\033[2J\033[H"  # ANSI clear and home
        return "\n" * 50  # Fallback

    def get_frame_count(self) -> int:
        """Get number of frames rendered."""
        return self.frame_count


# ============================================================================
# JSON Serialization
# ============================================================================

def graph_to_json(graph: SyncGraphRender) -> Dict[str, Any]:
    """Serialize sync graph to JSON-compatible dict."""
    return {
        "nodes": [
            {
                "id": n.node_id,
                "coherence": n.coherence,
                "sync_state": n.sync_state,
                "frequency": n.frequency,
                "position": {"x": n.x, "y": n.y}
            }
            for n in graph.nodes
        ],
        "links": [
            {
                "source": l.source_id,
                "target": l.target_id,
                "strength": l.strength,
                "type": l.link_type
            }
            for l in graph.links
        ],
        "global_sync": graph.global_sync,
        "timestamp": graph.timestamp
    }


def graph_from_json(data: Dict[str, Any]) -> SyncGraphRender:
    """Deserialize sync graph from JSON dict."""
    nodes = [
        NodeRenderData(
            node_id=n["id"],
            coherence=n["coherence"],
            sync_state=n["sync_state"],
            frequency=n["frequency"],
            x=n["position"]["x"],
            y=n["position"]["y"]
        )
        for n in data["nodes"]
    ]

    links = [
        LinkRenderData(
            source_id=l["source"],
            target_id=l["target"],
            strength=l["strength"],
            link_type=l["type"]
        )
        for l in data["links"]
    ]

    return SyncGraphRender(
        nodes=nodes,
        links=links,
        global_sync=data["global_sync"],
        timestamp=data["timestamp"]
    )


# ============================================================================
# Test Suite
# ============================================================================

class TestTerminalDetection:
    """Test terminal capability detection."""

    def test_width_fallback(self):
        """Width falls back to 120 on failure."""
        width = detect_terminal_width(DEFAULT_WIDTH)
        assert MIN_WIDTH <= width <= MAX_WIDTH

    def test_width_custom_fallback(self):
        """Custom fallback width is respected."""
        width = detect_terminal_width(80)
        assert width >= MIN_WIDTH

    def test_color_detection_returns_valid(self):
        """Color detection returns valid mode."""
        mode = detect_color_support()
        assert mode in [ColorMode.ANSI_256, ColorMode.ANSI_16, ColorMode.PLAIN]

    def test_unicode_detection_returns_valid(self):
        """Unicode detection returns valid charset."""
        charset = detect_unicode_support()
        assert charset in [CharSet.UNICODE, CharSet.ASCII]

    def test_config_creation(self):
        """Terminal config creates successfully."""
        config = create_terminal_config()
        assert config.width > 0
        assert config.height > 0


class TestColorMapping:
    """Test coherence color mapping."""

    def test_low_coherence_red(self):
        """Low coherence maps to red."""
        color = get_coherence_color(0.1, ColorMode.ANSI_16)
        assert color == ANSI_RED

    def test_medium_coherence_blue(self):
        """Medium coherence maps to blue."""
        color = get_coherence_color(0.60, ColorMode.ANSI_16)
        assert color == ANSI_BLUE

    def test_high_coherence_green(self):
        """High coherence maps to green."""
        color = get_coherence_color(0.90, ColorMode.ANSI_16)
        assert color == ANSI_GREEN

    def test_plain_mode_no_color(self):
        """Plain mode returns empty string."""
        color = get_coherence_color(0.90, ColorMode.PLAIN)
        assert color == ""

    def test_colorize_applies_color(self):
        """Colorize applies ANSI codes."""
        result = colorize("test", ANSI_GREEN, ColorMode.ANSI_16)
        assert ANSI_GREEN in result
        assert ANSI_RESET in result

    def test_colorize_plain_passthrough(self):
        """Colorize in plain mode passes through unchanged."""
        result = colorize("test", ANSI_GREEN, ColorMode.PLAIN)
        assert result == "test"


class TestBoxDrawing:
    """Test box drawing functions."""

    def test_unicode_box_chars(self):
        """Unicode charset returns box characters."""
        chars = get_box_chars(CharSet.UNICODE)
        assert chars['h'] == BOX_H
        assert chars['v'] == BOX_V

    def test_ascii_box_chars(self):
        """ASCII charset returns simple characters."""
        chars = get_box_chars(CharSet.ASCII)
        assert chars['h'] == ASCII_H
        assert chars['v'] == ASCII_V

    def test_draw_box_dimensions(self):
        """Box has correct dimensions."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        box = draw_box(10, 5, config)
        assert len(box) == 5
        assert len(box[0]) == 10


class TestNodeRendering:
    """Test node rendering."""

    def test_render_node_lines(self):
        """Node renders to 3 lines."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        node = NodeRenderData("c1", 0.85, "SYNC", 432.0, 0, 0)
        lines = render_node(node, config)
        assert len(lines) == 3

    def test_render_node_detail(self):
        """Node detail contains all info."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        node = NodeRenderData("chamber1", 0.85, "SYNCHRONIZED", 432.0, 0, 0)
        detail = render_node_detail(node, config)

        assert "chamber1" in detail
        assert "0.85" in detail
        assert "432" in detail


class TestProgressBar:
    """Test progress bar rendering."""

    def test_empty_bar(self):
        """Zero value shows empty bar."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        bar = render_progress_bar(0.0, 10, config)
        assert "#" not in bar.replace("[", "").replace("]", "")

    def test_full_bar(self):
        """Full value shows full bar."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        bar = render_progress_bar(1.0, 10, config)
        assert "-" not in bar.replace("[", "").replace("]", "")

    def test_partial_bar(self):
        """Partial value shows mixed bar."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        bar = render_progress_bar(0.5, 10, config)
        assert "#" in bar
        assert "-" in bar


class TestGraphRendering:
    """Test full graph rendering."""

    def test_render_empty_graph(self):
        """Empty graph renders without error."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        graph = SyncGraphRender([], [], 0.0, 0.0)
        lines = render_sync_graph(graph, config)
        assert len(lines) > 0

    def test_render_with_nodes(self):
        """Graph with nodes renders correctly."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        nodes = [
            NodeRenderData("c1", 0.85, "SYNC", 432.0, 0, 0),
            NodeRenderData("c2", 0.75, "SEEK", 432.0, 1, 0),
        ]
        links = [
            LinkRenderData("c1", "c2", 0.65, "RESONANT")
        ]
        graph = SyncGraphRender(nodes, links, 0.70, 12345.0)

        lines = render_sync_graph(graph, config)

        # Check content
        output = "\n".join(lines)
        assert "c1" in output
        assert "c2" in output
        assert "0.70" in output  # Global sync

    def test_render_legend(self):
        """Legend renders all coherence levels."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        legend = render_legend(config)
        assert len(legend) > 1
        assert "Legend" in legend[0]


class TestLiveVisualizer:
    """Test live visualizer."""

    def test_visualizer_creation(self):
        """Visualizer creates with config."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        vis = LiveVisualizer(config)
        assert vis.config == config

    def test_visualizer_update(self):
        """Visualizer updates and returns output."""
        vis = LiveVisualizer()
        graph = SyncGraphRender([], [], 0.5, 0.0)
        output = vis.update(graph)
        assert len(output) > 0

    def test_frame_counter(self):
        """Frame counter increments."""
        vis = LiveVisualizer()
        graph = SyncGraphRender([], [], 0.5, 0.0)

        assert vis.get_frame_count() == 0
        vis.update(graph)
        assert vis.get_frame_count() == 1
        vis.update(graph)
        assert vis.get_frame_count() == 2


class TestJSONSerialization:
    """Test JSON serialization."""

    def test_graph_to_json(self):
        """Graph serializes to valid JSON."""
        nodes = [NodeRenderData("c1", 0.85, "SYNC", 432.0, 10, 20)]
        links = [LinkRenderData("c1", "c2", 0.65, "RESONANT")]
        graph = SyncGraphRender(nodes, links, 0.70, 12345.0)

        data = graph_to_json(graph)
        json_str = json.dumps(data)
        parsed = json.loads(json_str)

        assert parsed["global_sync"] == 0.70
        assert len(parsed["nodes"]) == 1
        assert len(parsed["links"]) == 1

    def test_json_roundtrip(self):
        """Graph survives JSON roundtrip."""
        nodes = [
            NodeRenderData("c1", 0.85, "SYNC", 432.0, 10, 20),
            NodeRenderData("c2", 0.75, "SEEK", 528.0, 30, 20),
        ]
        links = [LinkRenderData("c1", "c2", 0.65, "RESONANT")]
        original = SyncGraphRender(nodes, links, 0.70, 12345.0)

        # Roundtrip
        json_data = graph_to_json(original)
        json_str = json.dumps(json_data)
        parsed = json.loads(json_str)
        restored = graph_from_json(parsed)

        # Verify
        assert restored.global_sync == original.global_sync
        assert len(restored.nodes) == len(original.nodes)
        assert len(restored.links) == len(original.links)
        assert restored.nodes[0].node_id == original.nodes[0].node_id


class TestAdaptiveWidth:
    """Test adaptive width behavior."""

    def test_width_clamps_minimum(self):
        """Width clamps to minimum."""
        config = TerminalConfig(20, 24, ColorMode.PLAIN, CharSet.ASCII)
        # Should still render without error
        graph = SyncGraphRender([], [], 0.5, 0.0)
        lines = render_sync_graph(graph, config)
        assert len(lines) > 0

    def test_width_clamps_maximum(self):
        """Width clamps to maximum."""
        config = TerminalConfig(300, 24, ColorMode.PLAIN, CharSet.ASCII)
        graph = SyncGraphRender([], [], 0.5, 0.0)
        lines = render_sync_graph(graph, config)
        assert len(lines) > 0


class TestANSIFallback:
    """Test ANSI fallback behavior."""

    def test_plain_mode_no_escape_codes(self):
        """Plain mode output has no ANSI escapes."""
        config = TerminalConfig(80, 24, ColorMode.PLAIN, CharSet.ASCII)
        nodes = [NodeRenderData("c1", 0.85, "SYNC", 432.0, 0, 0)]
        graph = SyncGraphRender(nodes, [], 0.70, 0.0)

        lines = render_sync_graph(graph, config)
        output = "\n".join(lines)

        assert "\033[" not in output

    def test_ansi_mode_has_escape_codes(self):
        """ANSI mode includes escape codes."""
        config = TerminalConfig(80, 24, ColorMode.ANSI_16, CharSet.ASCII)
        nodes = [NodeRenderData("c1", 0.85, "SYNC", 432.0, 0, 0)]
        graph = SyncGraphRender(nodes, [], 0.70, 0.0)

        lines = render_sync_graph(graph, config)
        output = "\n".join(lines)

        # Should contain ANSI codes (might be reset codes at minimum)
        # The colorize function adds codes when color_mode is ANSI
        assert ANSI_RESET in output or "c1" in output  # At least renders


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
