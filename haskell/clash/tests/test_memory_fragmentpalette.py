"""
Test Suite for Ra.Memory.FragmentPalette (Prompt 67)
Scalar Fragment Memory Palettes

Models memory fragments as scalar pigment nodes in a phase-continuous
coherence palette with recall strength based on alpha depth.
"""

import pytest
import math
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque

# =============================================================================
# CONSTANTS
# =============================================================================

PHI = 1.618033988749895

# Volatility calculation window
VOLATILITY_WINDOW_SIZE = 8

# Inversion artifact strings
INVERSION_ARTIFACTS = [
    "dream-bleed",
    "shadow-echo",
    "consent-drift",
    "phase-slip",
    "memory-fade"
]

# Alpha depth sigmoid parameters
SIGMOID_STEEPNESS = 6.0
SIGMOID_MIDPOINT = 0.5


# =============================================================================
# DATA TYPES
# =============================================================================

@dataclass
class PhaseID:
    """Phase identifier for Ra.Identity sync."""
    namespace: str
    phase_index: int

    def __eq__(self, other):
        if not isinstance(other, PhaseID):
            return False
        return self.namespace == other.namespace and self.phase_index == other.phase_index

    def __hash__(self):
        return hash((self.namespace, self.phase_index))


@dataclass
class FragmentID:
    """Fragment identifier."""
    fragment_id: str


@dataclass
class ScalarFieldPoint:
    """Point in scalar field."""
    x: float
    y: float
    z: float
    alpha: float


@dataclass
class ScalarField:
    """Scalar field representation."""
    points: List[ScalarFieldPoint]


@dataclass
class FragmentNode:
    """Memory fragment as scalar pigment node."""
    fragment_id: FragmentID
    alpha_depth: float          # 0-1 recall saturation
    phase_lock: Optional[PhaseID]
    inversion_artifact: Optional[str]
    raw_alpha: float            # Original alpha before transform


@dataclass
class FragmentPalette:
    """Collection of fragment nodes."""
    nodes: List[FragmentNode]


@dataclass
class AlphaTrace:
    """Time-series trace of alpha values for volatility."""
    history: deque  # Rolling window of alpha values


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def sigmoid_transform(alpha: float) -> float:
    """Transform alpha using sigmoid curve for recall saturation.

    Maps alpha [0,1] to depth [0,1] with S-curve emphasis.
    """
    # Sigmoid: 1 / (1 + exp(-k*(x - midpoint)))
    exponent = -SIGMOID_STEEPNESS * (alpha - SIGMOID_MIDPOINT)
    return 1.0 / (1.0 + math.exp(exponent))


def power_curve_transform(alpha: float, power: float = 2.0) -> float:
    """Transform alpha using power curve.

    Higher alpha values get emphasized more.
    """
    return alpha ** power


def alpha_to_depth(alpha: float, use_sigmoid: bool = True) -> float:
    """Convert raw alpha to recall depth using non-linear transform."""
    if use_sigmoid:
        return sigmoid_transform(alpha)
    else:
        return power_curve_transform(alpha)


def select_inversion_artifact(alpha: float, phase_offset: float = 0.0) -> Optional[str]:
    """Select inversion artifact based on alpha and phase.

    Low alpha or high phase offset triggers artifacts.
    """
    if alpha < 0.3:
        # Low alpha triggers memory-related artifacts
        if alpha < 0.15:
            return "memory-fade"
        else:
            return "dream-bleed"
    elif phase_offset > 0.5:
        # High phase offset triggers phase-related artifacts
        if phase_offset > 0.75:
            return "consent-drift"
        else:
            return "phase-slip"
    elif alpha < 0.5 and phase_offset > 0.3:
        return "shadow-echo"

    return None


def create_fragment_node(
    fragment_id: FragmentID,
    alpha: float,
    phase_lock: Optional[PhaseID] = None,
    phase_offset: float = 0.0
) -> FragmentNode:
    """Create fragment node from alpha value."""
    alpha_depth = alpha_to_depth(alpha)
    artifact = select_inversion_artifact(alpha, phase_offset)

    return FragmentNode(
        fragment_id=fragment_id,
        alpha_depth=alpha_depth,
        phase_lock=phase_lock,
        inversion_artifact=artifact,
        raw_alpha=alpha
    )


def generate_palette_from_scalar_field(
    field: ScalarField,
    phase_lock: Optional[PhaseID] = None
) -> FragmentPalette:
    """Map scalar field to fragment palette.

    Each field point becomes a fragment node with
    coherence depth determining saturation.
    """
    nodes = []

    for i, point in enumerate(field.points):
        frag_id = FragmentID(f"frag_{i:04d}")
        node = create_fragment_node(
            fragment_id=frag_id,
            alpha=point.alpha,
            phase_lock=phase_lock,
            phase_offset=0.0
        )
        nodes.append(node)

    return FragmentPalette(nodes=nodes)


def create_alpha_trace() -> AlphaTrace:
    """Create empty alpha trace for volatility tracking."""
    return AlphaTrace(history=deque(maxlen=VOLATILITY_WINDOW_SIZE))


def update_alpha_trace(trace: AlphaTrace, alpha: float) -> None:
    """Add alpha value to trace history."""
    trace.history.append(alpha)


def calculate_volatility(trace: AlphaTrace) -> float:
    """Calculate mnemonic volatility from alpha trace.

    volatility = stddev(α) / mean(α) over 8-tick window
    """
    if len(trace.history) < 2:
        return 0.0

    values = list(trace.history)
    mean_alpha = sum(values) / len(values)

    if mean_alpha < 1e-10:
        return 0.0

    variance = sum((v - mean_alpha) ** 2 for v in values) / len(values)
    stddev = math.sqrt(variance)

    return stddev / mean_alpha


def calculate_node_volatility(node: FragmentNode, trace: AlphaTrace) -> float:
    """Calculate volatility for a specific fragment node."""
    # Update trace with node's current alpha
    update_alpha_trace(trace, node.raw_alpha)
    return calculate_volatility(trace)


def check_phase_lock_sync(node: FragmentNode, identity_phase: PhaseID) -> bool:
    """Check if node is synchronized with Ra.Identity phase."""
    if node.phase_lock is None:
        return False
    return node.phase_lock == identity_phase


# =============================================================================
# TEST: ALPHA DEPTH MAPPING
# =============================================================================

class TestAlphaDepthMapping:
    """Test alpha to depth transformation."""

    def test_sigmoid_transform_midpoint(self):
        """Sigmoid at midpoint returns ~0.5."""
        depth = sigmoid_transform(0.5)
        assert abs(depth - 0.5) < 0.01

    def test_sigmoid_low_alpha_low_depth(self):
        """Low alpha produces low depth via sigmoid."""
        depth = sigmoid_transform(0.1)
        assert depth < 0.2

    def test_sigmoid_high_alpha_high_depth(self):
        """High alpha produces high depth via sigmoid."""
        depth = sigmoid_transform(0.9)
        assert depth > 0.8

    def test_sigmoid_is_monotonic(self):
        """Sigmoid transform is monotonically increasing."""
        prev_depth = 0.0
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            depth = sigmoid_transform(alpha)
            assert depth > prev_depth
            prev_depth = depth

    def test_power_curve_transform(self):
        """Power curve emphasizes high alpha."""
        depth_low = power_curve_transform(0.3)
        depth_high = power_curve_transform(0.9)

        # Power curve squares input
        assert abs(depth_low - 0.09) < 0.01
        assert abs(depth_high - 0.81) < 0.01

    def test_alpha_to_depth_uses_sigmoid_by_default(self):
        """alpha_to_depth uses sigmoid by default."""
        depth = alpha_to_depth(0.5)
        assert abs(depth - 0.5) < 0.01


# =============================================================================
# TEST: VOLATILITY CALCULATION
# =============================================================================

class TestVolatilityCalculation:
    """Test volatility (mnemonic drift) calculation."""

    def test_empty_trace_zero_volatility(self):
        """Empty trace has zero volatility."""
        trace = create_alpha_trace()
        vol = calculate_volatility(trace)
        assert vol == 0.0

    def test_single_value_zero_volatility(self):
        """Single value has zero volatility."""
        trace = create_alpha_trace()
        update_alpha_trace(trace, 0.5)
        vol = calculate_volatility(trace)
        assert vol == 0.0

    def test_constant_values_zero_volatility(self):
        """Constant alpha values have zero volatility."""
        trace = create_alpha_trace()
        for _ in range(8):
            update_alpha_trace(trace, 0.7)
        vol = calculate_volatility(trace)
        assert vol == 0.0

    def test_varying_values_nonzero_volatility(self):
        """Varying alpha values have non-zero volatility."""
        trace = create_alpha_trace()
        for alpha in [0.5, 0.6, 0.5, 0.7, 0.5, 0.8, 0.5, 0.4]:
            update_alpha_trace(trace, alpha)
        vol = calculate_volatility(trace)
        assert vol > 0.1

    def test_rolling_window_size(self):
        """Rolling window maintains 8 values."""
        trace = create_alpha_trace()
        for i in range(12):
            update_alpha_trace(trace, 0.5 + i * 0.01)
        assert len(trace.history) == VOLATILITY_WINDOW_SIZE


# =============================================================================
# TEST: INVERSION ARTIFACTS
# =============================================================================

class TestInversionArtifacts:
    """Test inversion artifact selection."""

    def test_very_low_alpha_memory_fade(self):
        """Very low alpha triggers memory-fade."""
        artifact = select_inversion_artifact(0.10)
        assert artifact == "memory-fade"

    def test_low_alpha_dream_bleed(self):
        """Low alpha triggers dream-bleed."""
        artifact = select_inversion_artifact(0.25)
        assert artifact == "dream-bleed"

    def test_high_phase_offset_consent_drift(self):
        """High phase offset triggers consent-drift."""
        artifact = select_inversion_artifact(0.6, phase_offset=0.8)
        assert artifact == "consent-drift"

    def test_medium_phase_offset_phase_slip(self):
        """Medium phase offset triggers phase-slip."""
        artifact = select_inversion_artifact(0.6, phase_offset=0.6)
        assert artifact == "phase-slip"

    def test_mid_alpha_mid_offset_shadow_echo(self):
        """Mid alpha with mid offset triggers shadow-echo."""
        artifact = select_inversion_artifact(0.4, phase_offset=0.4)
        assert artifact == "shadow-echo"

    def test_healthy_state_no_artifact(self):
        """Healthy alpha/offset produces no artifact."""
        artifact = select_inversion_artifact(0.8, phase_offset=0.1)
        assert artifact is None

    def test_all_artifacts_reachable(self):
        """All defined artifacts can be triggered."""
        triggered = set()
        test_cases = [
            (0.10, 0.0),  # memory-fade
            (0.25, 0.0),  # dream-bleed
            (0.6, 0.8),   # consent-drift
            (0.6, 0.6),   # phase-slip
            (0.4, 0.4),   # shadow-echo
        ]
        for alpha, offset in test_cases:
            artifact = select_inversion_artifact(alpha, offset)
            if artifact:
                triggered.add(artifact)

        assert len(triggered) == 5


# =============================================================================
# TEST: PHASE LOCK SYNC
# =============================================================================

class TestPhaseLockSync:
    """Test phase lock synchronization with Ra.Identity."""

    def test_matching_phase_syncs(self):
        """Matching phase IDs are synchronized."""
        phase = PhaseID(namespace="Ra.Identity", phase_index=3)
        node = FragmentNode(
            fragment_id=FragmentID("test"),
            alpha_depth=0.7,
            phase_lock=phase,
            inversion_artifact=None,
            raw_alpha=0.7
        )
        assert check_phase_lock_sync(node, phase) is True

    def test_different_namespace_not_synced(self):
        """Different namespace not synchronized."""
        node_phase = PhaseID(namespace="Ra.Identity", phase_index=3)
        identity_phase = PhaseID(namespace="Other", phase_index=3)
        node = FragmentNode(
            fragment_id=FragmentID("test"),
            alpha_depth=0.7,
            phase_lock=node_phase,
            inversion_artifact=None,
            raw_alpha=0.7
        )
        assert check_phase_lock_sync(node, identity_phase) is False

    def test_different_index_not_synced(self):
        """Different phase index not synchronized."""
        node_phase = PhaseID(namespace="Ra.Identity", phase_index=3)
        identity_phase = PhaseID(namespace="Ra.Identity", phase_index=5)
        node = FragmentNode(
            fragment_id=FragmentID("test"),
            alpha_depth=0.7,
            phase_lock=node_phase,
            inversion_artifact=None,
            raw_alpha=0.7
        )
        assert check_phase_lock_sync(node, identity_phase) is False

    def test_no_phase_lock_not_synced(self):
        """Node without phase lock not synchronized."""
        identity_phase = PhaseID(namespace="Ra.Identity", phase_index=3)
        node = FragmentNode(
            fragment_id=FragmentID("test"),
            alpha_depth=0.7,
            phase_lock=None,
            inversion_artifact=None,
            raw_alpha=0.7
        )
        assert check_phase_lock_sync(node, identity_phase) is False


# =============================================================================
# TEST: FRAGMENT NODE CREATION
# =============================================================================

class TestFragmentNodeCreation:
    """Test fragment node creation."""

    def test_creates_node_with_transformed_depth(self):
        """Node alpha_depth is transformed from raw alpha."""
        node = create_fragment_node(FragmentID("test"), alpha=0.8)

        # Depth should be transformed (not equal to raw)
        assert node.raw_alpha == 0.8
        assert node.alpha_depth != 0.8  # Sigmoid transform
        assert node.alpha_depth > 0.8   # Sigmoid > identity at high alpha

    def test_node_includes_phase_lock(self):
        """Node includes provided phase lock."""
        phase = PhaseID("Ra.Identity", 5)
        node = create_fragment_node(FragmentID("test"), alpha=0.7, phase_lock=phase)

        assert node.phase_lock == phase

    def test_low_alpha_gets_artifact(self):
        """Low alpha node gets inversion artifact."""
        node = create_fragment_node(FragmentID("test"), alpha=0.15)

        assert node.inversion_artifact is not None


# =============================================================================
# TEST: PALETTE GENERATION
# =============================================================================

class TestPaletteGeneration:
    """Test fragment palette generation from scalar field."""

    def test_generates_palette_from_field(self):
        """Generates palette with node per field point."""
        field = ScalarField(points=[
            ScalarFieldPoint(0, 0, 0, 0.5),
            ScalarFieldPoint(1, 0, 0, 0.7),
            ScalarFieldPoint(0, 1, 0, 0.9),
        ])

        palette = generate_palette_from_scalar_field(field)

        assert len(palette.nodes) == 3

    def test_palette_spans_full_field(self):
        """Palette includes all field points."""
        points = [ScalarFieldPoint(i, 0, 0, 0.1 * i) for i in range(10)]
        field = ScalarField(points=points)

        palette = generate_palette_from_scalar_field(field)

        assert len(palette.nodes) == 10

    def test_nodes_have_unique_ids(self):
        """Palette nodes have unique fragment IDs."""
        field = ScalarField(points=[
            ScalarFieldPoint(i, 0, 0, 0.5) for i in range(5)
        ])

        palette = generate_palette_from_scalar_field(field)

        ids = [n.fragment_id.fragment_id for n in palette.nodes]
        assert len(set(ids)) == 5

    def test_palette_with_phase_lock(self):
        """Palette applies phase lock to all nodes."""
        field = ScalarField(points=[
            ScalarFieldPoint(0, 0, 0, 0.6),
            ScalarFieldPoint(1, 0, 0, 0.7),
        ])
        phase = PhaseID("Ra.Identity", 7)

        palette = generate_palette_from_scalar_field(field, phase_lock=phase)

        for node in palette.nodes:
            assert node.phase_lock == phase


# =============================================================================
# TEST: INTEGRATION
# =============================================================================

class TestFragmentPaletteIntegration:
    """Integration tests for fragment palette system."""

    def test_full_pipeline(self):
        """Full pipeline from field to palette with volatility."""
        # Create field with varying alpha
        points = [
            ScalarFieldPoint(i, 0, 0, 0.3 + 0.1 * math.sin(i))
            for i in range(8)
        ]
        field = ScalarField(points=points)

        # Generate palette
        palette = generate_palette_from_scalar_field(field)

        # Track volatility for a node
        trace = create_alpha_trace()
        for node in palette.nodes:
            vol = calculate_node_volatility(node, trace)

        # Final volatility should be non-zero due to varying alpha
        final_vol = calculate_volatility(trace)
        assert final_vol > 0

    def test_degradation_based_on_alpha(self):
        """Low alpha nodes have lower depth (degraded)."""
        field = ScalarField(points=[
            ScalarFieldPoint(0, 0, 0, 0.2),  # Low alpha
            ScalarFieldPoint(1, 0, 0, 0.8),  # High alpha
        ])

        palette = generate_palette_from_scalar_field(field)

        low_node = palette.nodes[0]
        high_node = palette.nodes[1]

        assert low_node.alpha_depth < high_node.alpha_depth

    def test_saturation_based_on_alpha(self):
        """High alpha nodes have higher depth (saturated)."""
        field = ScalarField(points=[
            ScalarFieldPoint(0, 0, 0, 0.95),
        ])

        palette = generate_palette_from_scalar_field(field)

        node = palette.nodes[0]
        assert node.alpha_depth > 0.9  # Highly saturated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
