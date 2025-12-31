"""
Prompt 19: Ra.DomainExtensions Test Harness

Safety module for undefined scalar operations in the Ra system.
Handles division by zero, zero coherence states, and domain boundaries
with graceful degradation and meaningful fallback behaviors.

Codex References:
- Ra.Emergence: emergenceScore modulation
- Ra.Coherence: CoherenceAbyss handling
- Ra.Shadow: ShadowSurface liminal layer
- P18: Attractor type integration
"""

import pytest
import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union
from enum import Enum, auto


# ============================================================================
# Constants (Codex-aligned)
# ============================================================================

EPSILON = 1e-12  # Fixed epsilon for division safety
PHI = 1.618033988749895  # Golden ratio
COHERENCE_ABYSS_THRESHOLD = 0.0  # Exactly zero
SHADOW_SURFACE_DEPTH = 0.42  # Liminal echo layer floor


# ============================================================================
# Types
# ============================================================================

class DomainBoundary(Enum):
    """Domain boundary conditions."""
    SAFE = auto()           # Within valid domain
    ZERO_DIVISION = auto()  # Division by zero attempted
    COHERENCE_ABYSS = auto()  # Coherence exactly 0.0
    SHADOW_SURFACE = auto()   # In liminal echo layer
    NEGATIVE_EMERGENCE = auto()  # Emergence score went negative
    OVERFLOW = auto()        # Value exceeded representable range


class RecoveryStrategy(Enum):
    """Recovery strategies for domain violations."""
    EPSILON_SUBSTITUTE = auto()  # Replace zero with epsilon
    POTENTIAL_LIFT = auto()      # Apply multiplicative lift
    SHADOW_ECHO = auto()         # Return shadow surface echo
    CLAMP = auto()               # Clamp to valid range
    GRACEFUL_NAN = auto()        # Return safe NaN representation


@dataclass
class DomainResult:
    """Result from a domain-safe operation."""
    value: float
    boundary_hit: Optional[DomainBoundary] = None
    recovery_applied: Optional[RecoveryStrategy] = None
    original_value: Optional[float] = None
    potential_lift: float = 0.0


@dataclass
class Attractor:
    """Attractor type (isomorphic with P18)."""
    id: str
    enticement: float  # 0.0-1.0
    flavor: str  # "emotional", "sensory", "archetypal", "unknown"
    potential_lift: float = 0.0  # Multiplicative boost


@dataclass
class EmergenceCondition:
    """Current emergence state."""
    coherence: float
    flux: float
    shadow_threshold: float = 0.72
    emergence_score: float = 0.0


# Type alias for domain hooks
DomainHook = Callable[[float, float], DomainResult]


# ============================================================================
# Default Attractors (Shinigami Apple in ROM)
# ============================================================================

SHINIGAMI_APPLE = Attractor(
    id="shinigami_apple",
    enticement=0.999,
    flavor="archetypal",
    potential_lift=0.42  # Death note reference
)

DEFAULT_ATTRACTORS = {
    "shinigami_apple": SHINIGAMI_APPLE,
    "void_crystal": Attractor(
        id="void_crystal",
        enticement=0.01,
        flavor="unknown",
        potential_lift=-0.15
    ),
    "echo_mirror": Attractor(
        id="echo_mirror",
        enticement=0.5,
        flavor="sensory",
        potential_lift=0.0
    ),
}


# ============================================================================
# Core Domain Extension Functions
# ============================================================================

def safe_divide(numerator: float, denominator: float,
                epsilon: float = EPSILON) -> DomainResult:
    """
    Safe division with epsilon substitution for zero denominators.

    Args:
        numerator: The dividend
        denominator: The divisor
        epsilon: Minimum denominator value (default 1e-12)

    Returns:
        DomainResult with value and any boundary conditions hit
    """
    if abs(denominator) < epsilon:
        # Zero division detected
        safe_denom = epsilon if denominator >= 0 else -epsilon
        result = numerator / safe_denom
        return DomainResult(
            value=result,
            boundary_hit=DomainBoundary.ZERO_DIVISION,
            recovery_applied=RecoveryStrategy.EPSILON_SUBSTITUTE,
            original_value=numerator / denominator if denominator != 0 else float('inf')
        )

    return DomainResult(value=numerator / denominator)


def handle_zero_coherence(condition: EmergenceCondition,
                          attractor: Optional[Attractor] = None) -> DomainResult:
    """
    Handle zero coherence state (CoherenceAbyss).

    When coherence is exactly 0.0, fragment emergence enters the abyss state.
    If an attractor is present, apply potential lift to recover.

    Args:
        condition: Current emergence condition
        attractor: Optional attractor for recovery

    Returns:
        DomainResult with recovery information
    """
    if condition.coherence == COHERENCE_ABYSS_THRESHOLD:
        # Coherence abyss - exactly zero
        if attractor and attractor.potential_lift > 0:
            # Apply potential lift: multiplicative recovery
            lifted_score = condition.emergence_score * (1 + attractor.potential_lift)
            return DomainResult(
                value=lifted_score,
                boundary_hit=DomainBoundary.COHERENCE_ABYSS,
                recovery_applied=RecoveryStrategy.POTENTIAL_LIFT,
                original_value=condition.emergence_score,
                potential_lift=attractor.potential_lift
            )
        else:
            # No attractor or negative lift - return shadow surface echo
            return DomainResult(
                value=SHADOW_SURFACE_DEPTH,
                boundary_hit=DomainBoundary.COHERENCE_ABYSS,
                recovery_applied=RecoveryStrategy.SHADOW_ECHO,
                original_value=condition.emergence_score
            )

    # Normal coherence
    return DomainResult(value=condition.emergence_score)


def reframe_division_by_zero(operation: str, numerator: float,
                             denominator: float) -> DomainResult:
    """
    Reframe division by zero with semantic meaning.

    Different operations have different semantic meanings when
    dividing by zero - this function provides context-aware handling.

    Args:
        operation: Name of the operation (e.g., "flux_ratio", "coherence_factor")
        numerator: The dividend
        denominator: The divisor

    Returns:
        DomainResult with semantic recovery
    """
    if abs(denominator) < EPSILON:
        # Context-aware handling
        if operation == "flux_ratio":
            # Flux ratio with zero denominator = infinite flux potential
            return DomainResult(
                value=1.0,  # Maximum normalized flux
                boundary_hit=DomainBoundary.ZERO_DIVISION,
                recovery_applied=RecoveryStrategy.CLAMP,
                original_value=float('inf')
            )
        elif operation == "coherence_factor":
            # Coherence factor with zero = abyss state
            return DomainResult(
                value=0.0,
                boundary_hit=DomainBoundary.COHERENCE_ABYSS,
                recovery_applied=RecoveryStrategy.SHADOW_ECHO,
                original_value=0.0
            )
        elif operation == "emergence_scale":
            # Emergence scale with zero = return phi as divine proportion
            return DomainResult(
                value=PHI,
                boundary_hit=DomainBoundary.ZERO_DIVISION,
                recovery_applied=RecoveryStrategy.EPSILON_SUBSTITUTE,
                original_value=float('inf')
            )
        else:
            # Default: use safe_divide
            return safe_divide(numerator, denominator)

    return DomainResult(value=numerator / denominator)


def apply_potential_lift(emergence_score: float,
                         potential_lift: float) -> DomainResult:
    """
    Apply potential lift multiplicatively to emergence score.

    Formula: emergenceScore *= (1 + potentialLift)

    Args:
        emergence_score: Current emergence score
        potential_lift: Multiplicative factor

    Returns:
        DomainResult with lifted score
    """
    lifted = emergence_score * (1 + potential_lift)

    # Check for negative emergence
    if lifted < 0:
        return DomainResult(
            value=0.0,
            boundary_hit=DomainBoundary.NEGATIVE_EMERGENCE,
            recovery_applied=RecoveryStrategy.CLAMP,
            original_value=lifted,
            potential_lift=potential_lift
        )

    # Check for overflow (beyond reasonable emergence)
    if lifted > 1e6:
        return DomainResult(
            value=1e6,
            boundary_hit=DomainBoundary.OVERFLOW,
            recovery_applied=RecoveryStrategy.CLAMP,
            original_value=lifted,
            potential_lift=potential_lift
        )

    return DomainResult(
        value=lifted,
        potential_lift=potential_lift
    )


def shadow_surface_echo(condition: EmergenceCondition) -> DomainResult:
    """
    Return shadow surface echo value for liminal states.

    The shadow surface is a liminal echo layer at depth 0.42,
    representing the threshold between emergence and dissolution.

    Args:
        condition: Current emergence condition

    Returns:
        DomainResult with shadow echo value
    """
    if condition.coherence < condition.shadow_threshold:
        # In shadow region
        depth = max(SHADOW_SURFACE_DEPTH,
                   condition.coherence * condition.shadow_threshold)
        return DomainResult(
            value=depth,
            boundary_hit=DomainBoundary.SHADOW_SURFACE,
            recovery_applied=RecoveryStrategy.SHADOW_ECHO,
            original_value=condition.emergence_score
        )

    return DomainResult(value=condition.emergence_score)


def create_domain_hook(operation: str) -> DomainHook:
    """
    Create a domain hook for a specific operation type.

    Args:
        operation: Name of the operation

    Returns:
        DomainHook function
    """
    def hook(numerator: float, denominator: float) -> DomainResult:
        return reframe_division_by_zero(operation, numerator, denominator)
    return hook


# ============================================================================
# Test Suite
# ============================================================================

class TestSafeDivide:
    """Test safe division with epsilon substitution."""

    def test_normal_division(self):
        """Normal division returns expected result."""
        result = safe_divide(10.0, 2.0)
        assert result.value == 5.0
        assert result.boundary_hit is None
        assert result.recovery_applied is None

    def test_zero_denominator(self):
        """Zero denominator triggers epsilon substitution."""
        result = safe_divide(10.0, 0.0)
        assert result.value == 10.0 / EPSILON
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION
        assert result.recovery_applied == RecoveryStrategy.EPSILON_SUBSTITUTE

    def test_near_zero_denominator(self):
        """Near-zero denominator triggers epsilon substitution."""
        result = safe_divide(10.0, 1e-15)
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION
        assert result.recovery_applied == RecoveryStrategy.EPSILON_SUBSTITUTE

    def test_negative_near_zero(self):
        """Negative near-zero preserves sign."""
        result = safe_divide(10.0, -1e-15)
        assert result.value < 0
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION

    def test_custom_epsilon(self):
        """Custom epsilon is respected."""
        result = safe_divide(10.0, 1e-6, epsilon=1e-3)
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION


class TestHandleZeroCoherence:
    """Test zero coherence (CoherenceAbyss) handling."""

    def test_normal_coherence(self):
        """Normal coherence passes through."""
        condition = EmergenceCondition(coherence=0.5, flux=0.3, emergence_score=0.7)
        result = handle_zero_coherence(condition)
        assert result.value == 0.7
        assert result.boundary_hit is None

    def test_zero_coherence_no_attractor(self):
        """Zero coherence without attractor returns shadow echo."""
        condition = EmergenceCondition(coherence=0.0, flux=0.3, emergence_score=0.7)
        result = handle_zero_coherence(condition)
        assert result.value == SHADOW_SURFACE_DEPTH
        assert result.boundary_hit == DomainBoundary.COHERENCE_ABYSS
        assert result.recovery_applied == RecoveryStrategy.SHADOW_ECHO

    def test_zero_coherence_with_attractor(self):
        """Zero coherence with attractor applies potential lift."""
        condition = EmergenceCondition(coherence=0.0, flux=0.3, emergence_score=0.5)
        result = handle_zero_coherence(condition, SHINIGAMI_APPLE)
        # emergence_score * (1 + potential_lift) = 0.5 * 1.42 = 0.71
        expected = 0.5 * (1 + 0.42)
        assert abs(result.value - expected) < 1e-6
        assert result.boundary_hit == DomainBoundary.COHERENCE_ABYSS
        assert result.recovery_applied == RecoveryStrategy.POTENTIAL_LIFT

    def test_zero_coherence_negative_lift(self):
        """Zero coherence with negative lift returns shadow echo."""
        condition = EmergenceCondition(coherence=0.0, flux=0.3, emergence_score=0.7)
        void = DEFAULT_ATTRACTORS["void_crystal"]
        result = handle_zero_coherence(condition, void)
        assert result.value == SHADOW_SURFACE_DEPTH
        assert result.recovery_applied == RecoveryStrategy.SHADOW_ECHO

    def test_near_zero_coherence_not_abyss(self):
        """Near-zero (but not exactly zero) is not abyss."""
        condition = EmergenceCondition(coherence=1e-15, flux=0.3, emergence_score=0.7)
        result = handle_zero_coherence(condition)
        assert result.value == 0.7
        assert result.boundary_hit is None


class TestReframeDivisionByZero:
    """Test semantic reframing of division by zero."""

    def test_flux_ratio_zero(self):
        """Flux ratio with zero returns 1.0 (max flux)."""
        result = reframe_division_by_zero("flux_ratio", 5.0, 0.0)
        assert result.value == 1.0
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION
        assert result.recovery_applied == RecoveryStrategy.CLAMP

    def test_coherence_factor_zero(self):
        """Coherence factor with zero returns 0.0 (abyss)."""
        result = reframe_division_by_zero("coherence_factor", 5.0, 0.0)
        assert result.value == 0.0
        assert result.boundary_hit == DomainBoundary.COHERENCE_ABYSS

    def test_emergence_scale_zero(self):
        """Emergence scale with zero returns phi."""
        result = reframe_division_by_zero("emergence_scale", 5.0, 0.0)
        assert abs(result.value - PHI) < 1e-6
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION

    def test_unknown_operation_zero(self):
        """Unknown operation uses safe_divide."""
        result = reframe_division_by_zero("unknown_op", 10.0, 0.0)
        assert result.value == 10.0 / EPSILON
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION

    def test_normal_division(self):
        """Normal division passes through."""
        result = reframe_division_by_zero("flux_ratio", 10.0, 2.0)
        assert result.value == 5.0
        assert result.boundary_hit is None


class TestApplyPotentialLift:
    """Test multiplicative potential lift application."""

    def test_positive_lift(self):
        """Positive lift increases emergence score."""
        result = apply_potential_lift(0.5, 0.2)
        assert abs(result.value - 0.6) < 1e-6  # 0.5 * 1.2
        assert result.boundary_hit is None

    def test_negative_lift_safe(self):
        """Negative lift that stays positive."""
        result = apply_potential_lift(0.5, -0.2)
        assert abs(result.value - 0.4) < 1e-6  # 0.5 * 0.8
        assert result.boundary_hit is None

    def test_negative_lift_clamp(self):
        """Negative lift that would go negative is clamped."""
        result = apply_potential_lift(0.5, -1.5)  # 0.5 * -0.5 = -0.25
        assert result.value == 0.0
        assert result.boundary_hit == DomainBoundary.NEGATIVE_EMERGENCE
        assert result.recovery_applied == RecoveryStrategy.CLAMP

    def test_overflow_clamp(self):
        """Extreme lift is clamped to prevent overflow."""
        result = apply_potential_lift(1000.0, 1e6)
        assert result.value == 1e6
        assert result.boundary_hit == DomainBoundary.OVERFLOW
        assert result.recovery_applied == RecoveryStrategy.CLAMP


class TestShadowSurfaceEcho:
    """Test shadow surface echo for liminal states."""

    def test_above_threshold(self):
        """Above shadow threshold passes through."""
        condition = EmergenceCondition(coherence=0.8, flux=0.3,
                                       shadow_threshold=0.72, emergence_score=0.9)
        result = shadow_surface_echo(condition)
        assert result.value == 0.9
        assert result.boundary_hit is None

    def test_below_threshold(self):
        """Below shadow threshold returns echo."""
        condition = EmergenceCondition(coherence=0.5, flux=0.3,
                                       shadow_threshold=0.72, emergence_score=0.9)
        result = shadow_surface_echo(condition)
        # depth = max(0.42, 0.5 * 0.72) = max(0.42, 0.36) = 0.42
        assert result.value == SHADOW_SURFACE_DEPTH
        assert result.boundary_hit == DomainBoundary.SHADOW_SURFACE
        assert result.recovery_applied == RecoveryStrategy.SHADOW_ECHO

    def test_very_low_coherence(self):
        """Very low coherence still returns floor."""
        condition = EmergenceCondition(coherence=0.1, flux=0.3,
                                       shadow_threshold=0.72, emergence_score=0.9)
        result = shadow_surface_echo(condition)
        assert result.value == SHADOW_SURFACE_DEPTH


class TestDomainHook:
    """Test domain hook creation."""

    def test_create_flux_hook(self):
        """Create hook for flux_ratio operation."""
        hook = create_domain_hook("flux_ratio")
        result = hook(5.0, 0.0)
        assert result.value == 1.0
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION

    def test_create_coherence_hook(self):
        """Create hook for coherence_factor operation."""
        hook = create_domain_hook("coherence_factor")
        result = hook(5.0, 0.0)
        assert result.value == 0.0
        assert result.boundary_hit == DomainBoundary.COHERENCE_ABYSS


class TestShinigamiApple:
    """Test the shinigami_apple ROM attractor."""

    def test_apple_exists(self):
        """Shinigami apple is in default attractors."""
        assert "shinigami_apple" in DEFAULT_ATTRACTORS

    def test_apple_high_enticement(self):
        """Shinigami apple has very high enticement."""
        assert SHINIGAMI_APPLE.enticement == 0.999

    def test_apple_potential_lift(self):
        """Shinigami apple has 0.42 potential lift."""
        assert SHINIGAMI_APPLE.potential_lift == 0.42

    def test_apple_recovery(self):
        """Shinigami apple recovers from abyss state."""
        condition = EmergenceCondition(coherence=0.0, flux=0.3, emergence_score=1.0)
        result = handle_zero_coherence(condition, SHINIGAMI_APPLE)
        # 1.0 * 1.42 = 1.42
        assert abs(result.value - 1.42) < 1e-6


class TestIntegrationScenarios:
    """Integration tests for domain extension scenarios."""

    def test_full_abyss_recovery_chain(self):
        """Full chain: abyss -> potential lift -> emergence."""
        # Start in abyss
        condition = EmergenceCondition(coherence=0.0, flux=0.5, emergence_score=0.3)

        # Apply shinigami apple
        result = handle_zero_coherence(condition, SHINIGAMI_APPLE)
        assert result.boundary_hit == DomainBoundary.COHERENCE_ABYSS

        # Verify multiplicative lift
        expected_lift = 0.3 * (1 + 0.42)
        assert abs(result.value - expected_lift) < 1e-6

    def test_safe_divide_in_emergence_calc(self):
        """Safe divide used in emergence calculation."""
        flux = 0.8
        coherence = 0.0  # Would cause division by zero

        # Use safe divide for flux/coherence ratio
        result = safe_divide(flux, coherence)
        assert result.boundary_hit == DomainBoundary.ZERO_DIVISION
        assert result.value == flux / EPSILON  # Very large but defined

    def test_domain_hook_pipeline(self):
        """Domain hooks can be chained in a pipeline."""
        hooks = [
            create_domain_hook("flux_ratio"),
            create_domain_hook("coherence_factor"),
            create_domain_hook("emergence_scale"),
        ]

        # Test each hook with zero denominator
        results = [hook(1.0, 0.0) for hook in hooks]

        assert results[0].value == 1.0  # flux_ratio -> max flux
        assert results[1].value == 0.0  # coherence_factor -> abyss
        assert abs(results[2].value - PHI) < 1e-6  # emergence_scale -> phi


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
