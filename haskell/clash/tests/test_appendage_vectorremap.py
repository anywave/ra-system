"""
Test Suite for Ra.Appendage.VectorRemap (Prompt 71)

Non-physical limb vector remapping linked to scalar field intent pathways.
Models intent → vector → gesture → effect for post-physical avatar interfaces.

Architect Clarifications:
- IntentSignature: emotionTag :: EmotionEnum, targetCoord :: ChamberVector, phaseAngle :: Double
- Scalar routing: layered (gradient descent + shell topology routing)
- MappedEffects: symbolic strings with optional method bindings
"""

import pytest
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Callable, Any

# Constants
PHI = 1.618033988749895
PI = math.pi


class EmotionEnum(Enum):
    """Emotion tags for intent signatures."""
    NEUTRAL = auto()
    FOCUSED = auto()
    PROTECTIVE = auto()
    EXPANSIVE = auto()
    RECEPTIVE = auto()
    PROJECTIVE = auto()
    HEALING = auto()
    SHIELDING = auto()
    REVEALING = auto()


class EffectType(Enum):
    """Types of mapped effects."""
    SYMBOLIC = auto()      # String-based symbolic effect
    BOUND = auto()         # Method binding to interface
    COMPOSITE = auto()     # Multiple effects combined


class HarmonicShellIndex(Enum):
    """Harmonic shell indices (Ra.DomainExtensions)."""
    SHELL_1 = 1   # Inner shell
    SHELL_2 = 2
    SHELL_3 = 3
    SHELL_4 = 4
    SHELL_5 = 5
    SHELL_6 = 6
    SHELL_7 = 7   # Outer shell


# Symbolic effect mappings
SYMBOLIC_EFFECTS = {
    "push": "Ra.Chamber.ForceProject",
    "pull": "Ra.Chamber.ForceAttract",
    "shield": "Ra.Chamber.BarrierCreate",
    "reveal": "Ra.Visualizer.FieldShow",
    "hide": "Ra.Visualizer.FieldMask",
    "heal": "Ra.BioField.HealingPulse",
    "charge": "Ra.Energy.ChargeAccumulate",
    "discharge": "Ra.Energy.ChargeRelease",
    "connect": "Ra.Network.LinkEstablish",
    "sever": "Ra.Network.LinkBreak",
}


@dataclass
class RaCoordinate:
    """3D coordinate in Ra field."""
    x: float
    y: float
    z: float

    def distance_to(self, other: 'RaCoordinate') -> float:
        return math.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2
        )

    def __add__(self, other: 'RaCoordinate') -> 'RaCoordinate':
        return RaCoordinate(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: 'RaCoordinate') -> 'RaCoordinate':
        return RaCoordinate(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, factor: float) -> 'RaCoordinate':
        return RaCoordinate(self.x * factor, self.y * factor, self.z * factor)

    def magnitude(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> 'RaCoordinate':
        mag = self.magnitude()
        if mag < 0.0001:
            return RaCoordinate(0, 0, 0)
        return self.scale(1.0 / mag)


@dataclass
class ChamberVector:
    """Target vector in chamber space."""
    origin: RaCoordinate
    direction: RaCoordinate
    magnitude: float


@dataclass
class GradientVector:
    """Field gradient at a point."""
    dx: float
    dy: float
    dz: float

    def to_coordinate(self) -> RaCoordinate:
        return RaCoordinate(self.dx, self.dy, self.dz)


@dataclass
class ScalarFieldPoint:
    """Point in scalar field."""
    coordinate: RaCoordinate
    alpha: float
    gradient: GradientVector


@dataclass
class ScalarField:
    """Scalar field for routing."""
    points: List[ScalarFieldPoint]
    mean_alpha: float
    shell_boundaries: Dict[HarmonicShellIndex, float]  # Shell -> radius


@dataclass
class IntentSignature:
    """Intent signature for limb mapping."""
    emotion_tag: EmotionEnum
    target_coord: ChamberVector
    phase_angle: float  # Radians


@dataclass
class IntentField:
    """Field of intent signatures."""
    primary_intent: IntentSignature
    secondary_intents: List[IntentSignature]
    intensity: float  # 0-1
    coherence: float  # 0-1


@dataclass
class AvatarEffect:
    """Effect mapped from intent."""
    effect_type: EffectType
    symbolic_name: str
    method_binding: Optional[str]  # Interface method path
    parameters: Dict[str, Any]


@dataclass
class Avatar:
    """Avatar with position and state."""
    avatar_id: uuid.UUID
    position: RaCoordinate
    orientation: float  # Radians
    active_shell: HarmonicShellIndex
    coherence: float


@dataclass
class VectorLimb:
    """Non-physical vector limb."""
    limb_id: uuid.UUID
    intent_phase: IntentSignature
    scalar_route: List[RaCoordinate]
    active_shell: HarmonicShellIndex
    mapped_effect: AvatarEffect


@dataclass
class VectorRemapProfile:
    """Collection of vector limbs for an avatar."""
    limbs: List[VectorLimb]
    avatar_id: uuid.UUID
    total_intensity: float


def emotion_to_effect_mapping(emotion: EmotionEnum) -> str:
    """Map emotion to default symbolic effect."""
    mappings = {
        EmotionEnum.NEUTRAL: "reveal",
        EmotionEnum.FOCUSED: "push",
        EmotionEnum.PROTECTIVE: "shield",
        EmotionEnum.EXPANSIVE: "charge",
        EmotionEnum.RECEPTIVE: "pull",
        EmotionEnum.PROJECTIVE: "push",
        EmotionEnum.HEALING: "heal",
        EmotionEnum.SHIELDING: "shield",
        EmotionEnum.REVEALING: "reveal",
    }
    return mappings.get(emotion, "reveal")


def compute_gradient_at_point(
    field: ScalarField,
    point: RaCoordinate
) -> GradientVector:
    """Compute field gradient at a point using nearby samples."""
    if not field.points:
        return GradientVector(0, 0, 0)

    # Find closest points
    sorted_points = sorted(field.points,
                          key=lambda p: point.distance_to(p.coordinate))
    closest = sorted_points[:min(4, len(sorted_points))]

    if not closest:
        return GradientVector(0, 0, 0)

    # Average gradient weighted by distance
    total_weight = 0.0
    weighted_gradient = GradientVector(0, 0, 0)

    for fp in closest:
        dist = max(0.001, point.distance_to(fp.coordinate))
        weight = 1.0 / dist
        total_weight += weight
        weighted_gradient.dx += fp.gradient.dx * weight
        weighted_gradient.dy += fp.gradient.dy * weight
        weighted_gradient.dz += fp.gradient.dz * weight

    if total_weight > 0:
        weighted_gradient.dx /= total_weight
        weighted_gradient.dy /= total_weight
        weighted_gradient.dz /= total_weight

    return weighted_gradient


def gradient_descent_step(
    current: RaCoordinate,
    field: ScalarField,
    step_size: float = 0.1
) -> RaCoordinate:
    """Take one gradient descent step toward higher alpha."""
    gradient = compute_gradient_at_point(field, current)

    # Move in gradient direction (toward higher alpha)
    return RaCoordinate(
        current.x + gradient.dx * step_size,
        current.y + gradient.dy * step_size,
        current.z + gradient.dz * step_size
    )


def find_shell_boundary_crossing(
    start: RaCoordinate,
    end: RaCoordinate,
    shell_boundaries: Dict[HarmonicShellIndex, float],
    center: RaCoordinate = None
) -> Optional[HarmonicShellIndex]:
    """Find if path crosses a shell boundary."""
    if center is None:
        center = RaCoordinate(0, 0, 0)

    start_dist = start.distance_to(center)
    end_dist = end.distance_to(center)

    for shell, radius in shell_boundaries.items():
        # Check if we cross this shell
        if (start_dist < radius <= end_dist) or (end_dist < radius <= start_dist):
            return shell

    return None


def shell_topology_route(
    start: RaCoordinate,
    target: RaCoordinate,
    shell_boundaries: Dict[HarmonicShellIndex, float],
    current_shell: HarmonicShellIndex
) -> List[RaCoordinate]:
    """
    Route through shell topology (harmonic tube network).
    Respects shell boundaries and finds efficient path.
    """
    route = [start]
    center = RaCoordinate(0, 0, 0)

    # Direct path check
    crossing = find_shell_boundary_crossing(start, target, shell_boundaries, center)

    if crossing is None:
        # No shell crossing, direct path
        route.append(target)
        return route

    # Need to route through shell
    # Find intermediate point at shell boundary
    direction = (target - start).normalized()
    current_radius = start.distance_to(center)
    target_radius = target.distance_to(center)

    # Route through each shell we need to cross
    for shell in sorted(shell_boundaries.keys(), key=lambda s: shell_boundaries[s]):
        shell_radius = shell_boundaries[shell]

        if current_radius < shell_radius < target_radius:
            # Need to cross outward
            # Find point on shell boundary
            boundary_point = direction.scale(shell_radius)
            route.append(boundary_point)
        elif target_radius < shell_radius < current_radius:
            # Need to cross inward
            boundary_point = direction.scale(shell_radius)
            route.append(boundary_point)

    route.append(target)
    return route


def compute_scalar_route(
    start: RaCoordinate,
    target: ChamberVector,
    field: ScalarField,
    avatar_shell: HarmonicShellIndex,
    max_steps: int = 10
) -> List[RaCoordinate]:
    """
    Compute scalar route using layered approach:
    1. Gradient descent through α field
    2. Shell topology routing
    """
    route = [start]
    current = start
    target_point = target.origin

    # Phase 1: Gradient descent toward target
    for _ in range(max_steps // 2):
        # Blend gradient descent with target direction
        gradient = compute_gradient_at_point(field, current)
        target_dir = (target_point - current).normalized()

        # Weighted blend (gradient influences but target dominates)
        blend_x = gradient.dx * 0.3 + target_dir.x * 0.7
        blend_y = gradient.dy * 0.3 + target_dir.y * 0.7
        blend_z = gradient.dz * 0.3 + target_dir.z * 0.7

        step_size = 0.1 * target.magnitude
        next_point = RaCoordinate(
            current.x + blend_x * step_size,
            current.y + blend_y * step_size,
            current.z + blend_z * step_size
        )

        route.append(next_point)
        current = next_point

        # Check if close enough to target
        if current.distance_to(target_point) < 0.1:
            break

    # Phase 2: Shell topology routing for remaining distance
    if current.distance_to(target_point) > 0.1:
        shell_route = shell_topology_route(
            current, target_point,
            field.shell_boundaries,
            avatar_shell
        )
        route.extend(shell_route[1:])  # Skip first (already current)

    return route


def select_shell_for_intent(intent: IntentSignature, avatar: Avatar) -> HarmonicShellIndex:
    """Select appropriate harmonic shell based on intent."""
    # Protective/shielding intents use outer shells
    if intent.emotion_tag in [EmotionEnum.PROTECTIVE, EmotionEnum.SHIELDING]:
        return HarmonicShellIndex.SHELL_6

    # Healing uses inner shells
    if intent.emotion_tag == EmotionEnum.HEALING:
        return HarmonicShellIndex.SHELL_2

    # Projective uses outer shells
    if intent.emotion_tag in [EmotionEnum.PROJECTIVE, EmotionEnum.EXPANSIVE]:
        return HarmonicShellIndex.SHELL_5

    # Receptive uses middle shells
    if intent.emotion_tag == EmotionEnum.RECEPTIVE:
        return HarmonicShellIndex.SHELL_3

    # Default to avatar's active shell
    return avatar.active_shell


def create_avatar_effect(
    emotion: EmotionEnum,
    target: ChamberVector,
    phase_angle: float
) -> AvatarEffect:
    """Create avatar effect from intent components."""
    symbolic_name = emotion_to_effect_mapping(emotion)
    method_binding = SYMBOLIC_EFFECTS.get(symbolic_name)

    parameters = {
        "target_x": target.origin.x,
        "target_y": target.origin.y,
        "target_z": target.origin.z,
        "direction_x": target.direction.x,
        "direction_y": target.direction.y,
        "direction_z": target.direction.z,
        "magnitude": target.magnitude,
        "phase": phase_angle,
    }

    effect_type = EffectType.BOUND if method_binding else EffectType.SYMBOLIC

    return AvatarEffect(
        effect_type=effect_type,
        symbolic_name=symbolic_name,
        method_binding=method_binding,
        parameters=parameters
    )


def create_vector_limb(
    intent: IntentSignature,
    field: ScalarField,
    avatar: Avatar
) -> VectorLimb:
    """Create a single vector limb from intent."""
    # Select shell
    shell = select_shell_for_intent(intent, avatar)

    # Compute route
    route = compute_scalar_route(
        avatar.position,
        intent.target_coord,
        field,
        shell
    )

    # Create effect
    effect = create_avatar_effect(
        intent.emotion_tag,
        intent.target_coord,
        intent.phase_angle
    )

    return VectorLimb(
        limb_id=uuid.uuid4(),
        intent_phase=intent,
        scalar_route=route,
        active_shell=shell,
        mapped_effect=effect
    )


def remap_intent_to_limb(
    intent_field: IntentField,
    scalar_field: ScalarField,
    avatar: Avatar
) -> VectorRemapProfile:
    """
    Remap intent field to vector limb profile.
    Main function contract implementation.
    """
    limbs = []

    # Create limb for primary intent
    primary_limb = create_vector_limb(
        intent_field.primary_intent,
        scalar_field,
        avatar
    )
    limbs.append(primary_limb)

    # Create limbs for secondary intents (weighted by intensity)
    for secondary in intent_field.secondary_intents:
        if intent_field.intensity > 0.3:  # Only if sufficient intensity
            limb = create_vector_limb(secondary, scalar_field, avatar)
            limbs.append(limb)

    # Calculate total intensity
    total_intensity = intent_field.intensity * intent_field.coherence * len(limbs)

    return VectorRemapProfile(
        limbs=limbs,
        avatar_id=avatar.avatar_id,
        total_intensity=total_intensity
    )


def validate_limb_coherence(limb: VectorLimb, field: ScalarField) -> bool:
    """Validate that limb route maintains coherence with field."""
    if not limb.scalar_route:
        return False

    # Check route doesn't have discontinuities
    for i in range(len(limb.scalar_route) - 1):
        dist = limb.scalar_route[i].distance_to(limb.scalar_route[i + 1])
        if dist > 1.0:  # Max step size
            return False

    return True


def compute_effect_chain(profile: VectorRemapProfile) -> List[str]:
    """Extract effect chain from profile (for simulation/render)."""
    return [limb.mapped_effect.symbolic_name for limb in profile.limbs]


# ============== TESTS ==============

class TestIntentSignature:
    """Tests for intent signature structure."""

    def test_intent_has_emotion_tag(self):
        """Intent should have emotion tag."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.FOCUSED,
            target_coord=ChamberVector(
                RaCoordinate(1, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0.0
        )
        assert intent.emotion_tag == EmotionEnum.FOCUSED

    def test_intent_has_target_coord(self):
        """Intent should have target coordinate."""
        target = ChamberVector(
            RaCoordinate(1, 2, 3),
            RaCoordinate(0, 1, 0),
            2.5
        )
        intent = IntentSignature(
            emotion_tag=EmotionEnum.PROJECTIVE,
            target_coord=target,
            phase_angle=PI / 4
        )
        assert intent.target_coord.origin.x == 1
        assert intent.target_coord.origin.y == 2
        assert intent.target_coord.origin.z == 3

    def test_intent_has_phase_angle(self):
        """Intent should have phase angle in radians."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.HEALING,
            target_coord=ChamberVector(
                RaCoordinate(0, 0, 0),
                RaCoordinate(0, 0, 1),
                1.0
            ),
            phase_angle=PI / 2
        )
        assert intent.phase_angle == pytest.approx(PI / 2)


class TestEmotionEffectMapping:
    """Tests for emotion to effect mapping."""

    def test_focused_maps_to_push(self):
        """Focused emotion should map to push effect."""
        effect = emotion_to_effect_mapping(EmotionEnum.FOCUSED)
        assert effect == "push"

    def test_protective_maps_to_shield(self):
        """Protective emotion should map to shield effect."""
        effect = emotion_to_effect_mapping(EmotionEnum.PROTECTIVE)
        assert effect == "shield"

    def test_healing_maps_to_heal(self):
        """Healing emotion should map to heal effect."""
        effect = emotion_to_effect_mapping(EmotionEnum.HEALING)
        assert effect == "heal"

    def test_receptive_maps_to_pull(self):
        """Receptive emotion should map to pull effect."""
        effect = emotion_to_effect_mapping(EmotionEnum.RECEPTIVE)
        assert effect == "pull"

    def test_revealing_maps_to_reveal(self):
        """Revealing emotion should map to reveal effect."""
        effect = emotion_to_effect_mapping(EmotionEnum.REVEALING)
        assert effect == "reveal"

    def test_all_emotions_have_mappings(self):
        """All emotions should have effect mappings."""
        for emotion in EmotionEnum:
            effect = emotion_to_effect_mapping(emotion)
            assert effect is not None
            assert effect in SYMBOLIC_EFFECTS


class TestScalarRouting:
    """Tests for scalar route computation."""

    def test_gradient_descent_moves_toward_gradient(self):
        """Gradient descent should move in gradient direction."""
        point = ScalarFieldPoint(
            RaCoordinate(0, 0, 0),
            alpha=0.5,
            gradient=GradientVector(1, 0, 0)
        )
        field = ScalarField(
            points=[point],
            mean_alpha=0.5,
            shell_boundaries={}
        )

        start = RaCoordinate(0, 0, 0)
        next_pos = gradient_descent_step(start, field, step_size=0.1)

        # Should have moved in positive x direction
        assert next_pos.x > start.x

    def test_route_starts_at_avatar_position(self):
        """Route should start at avatar position."""
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(1, 2, 3),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        target = ChamberVector(
            RaCoordinate(5, 5, 5),
            RaCoordinate(1, 0, 0),
            1.0
        )

        route = compute_scalar_route(
            avatar.position, target, field,
            avatar.active_shell
        )

        assert route[0].x == pytest.approx(1)
        assert route[0].y == pytest.approx(2)
        assert route[0].z == pytest.approx(3)

    def test_route_ends_near_target(self):
        """Route should end near target."""
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        target = ChamberVector(
            RaCoordinate(1, 0, 0),
            RaCoordinate(1, 0, 0),
            1.0
        )

        route = compute_scalar_route(
            avatar.position, target, field,
            avatar.active_shell, max_steps=20
        )

        # End should be close to target
        end = route[-1]
        dist = end.distance_to(target.origin)
        assert dist < 0.5  # Within reasonable distance

    def test_route_has_multiple_waypoints(self):
        """Route should have multiple waypoints."""
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        target = ChamberVector(
            RaCoordinate(5, 5, 5),
            RaCoordinate(1, 0, 0),
            1.0
        )

        route = compute_scalar_route(
            avatar.position, target, field,
            avatar.active_shell
        )

        assert len(route) > 2


class TestShellTopologyRouting:
    """Tests for shell topology routing."""

    def test_no_crossing_direct_path(self):
        """No shell crossing should give direct path."""
        route = shell_topology_route(
            RaCoordinate(1, 0, 0),
            RaCoordinate(2, 0, 0),
            {HarmonicShellIndex.SHELL_3: 5.0},  # Shell at 5, both points inside
            HarmonicShellIndex.SHELL_1
        )

        assert len(route) == 2  # Start and end only

    def test_shell_crossing_adds_waypoint(self):
        """Crossing shell should add waypoint at boundary."""
        route = shell_topology_route(
            RaCoordinate(1, 0, 0),      # Inside shell 3
            RaCoordinate(10, 0, 0),     # Outside shell 3
            {HarmonicShellIndex.SHELL_3: 5.0},
            HarmonicShellIndex.SHELL_1
        )

        # Should have waypoint at shell boundary
        assert len(route) >= 2

    def test_finds_shell_boundary_crossing(self):
        """Should detect shell boundary crossing."""
        crossing = find_shell_boundary_crossing(
            RaCoordinate(1, 0, 0),
            RaCoordinate(10, 0, 0),
            {HarmonicShellIndex.SHELL_3: 5.0},
            RaCoordinate(0, 0, 0)
        )

        assert crossing == HarmonicShellIndex.SHELL_3


class TestShellSelection:
    """Tests for shell selection based on intent."""

    def test_protective_uses_outer_shell(self):
        """Protective intent should use outer shell."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.PROTECTIVE,
            target_coord=ChamberVector(
                RaCoordinate(1, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0
        )
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        shell = select_shell_for_intent(intent, avatar)
        assert shell == HarmonicShellIndex.SHELL_6

    def test_healing_uses_inner_shell(self):
        """Healing intent should use inner shell."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.HEALING,
            target_coord=ChamberVector(
                RaCoordinate(1, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0
        )
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        shell = select_shell_for_intent(intent, avatar)
        assert shell == HarmonicShellIndex.SHELL_2

    def test_neutral_uses_avatar_shell(self):
        """Neutral intent should use avatar's active shell."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.NEUTRAL,
            target_coord=ChamberVector(
                RaCoordinate(1, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0
        )
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_4,
            coherence=0.8
        )

        shell = select_shell_for_intent(intent, avatar)
        assert shell == HarmonicShellIndex.SHELL_4


class TestAvatarEffectCreation:
    """Tests for avatar effect creation."""

    def test_effect_has_symbolic_name(self):
        """Effect should have symbolic name."""
        target = ChamberVector(
            RaCoordinate(1, 0, 0),
            RaCoordinate(1, 0, 0),
            1.0
        )
        effect = create_avatar_effect(EmotionEnum.FOCUSED, target, 0.0)

        assert effect.symbolic_name == "push"

    def test_effect_has_method_binding(self):
        """Effect should have method binding for known effects."""
        target = ChamberVector(
            RaCoordinate(1, 0, 0),
            RaCoordinate(1, 0, 0),
            1.0
        )
        effect = create_avatar_effect(EmotionEnum.SHIELDING, target, 0.0)

        assert effect.method_binding == "Ra.Chamber.BarrierCreate"

    def test_effect_carries_parameters(self):
        """Effect should carry target parameters."""
        target = ChamberVector(
            RaCoordinate(1, 2, 3),
            RaCoordinate(0, 1, 0),
            2.5
        )
        effect = create_avatar_effect(EmotionEnum.PROJECTIVE, target, PI / 4)

        assert effect.parameters["target_x"] == 1
        assert effect.parameters["target_y"] == 2
        assert effect.parameters["target_z"] == 3
        assert effect.parameters["magnitude"] == 2.5
        assert effect.parameters["phase"] == pytest.approx(PI / 4)

    def test_bound_effect_type_for_known_effects(self):
        """Known effects should have BOUND type."""
        target = ChamberVector(
            RaCoordinate(1, 0, 0),
            RaCoordinate(1, 0, 0),
            1.0
        )
        effect = create_avatar_effect(EmotionEnum.HEALING, target, 0.0)

        assert effect.effect_type == EffectType.BOUND


class TestVectorLimbCreation:
    """Tests for vector limb creation."""

    def test_limb_has_unique_id(self):
        """Limb should have unique ID."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.FOCUSED,
            target_coord=ChamberVector(
                RaCoordinate(5, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        limb = create_vector_limb(intent, field, avatar)

        assert limb.limb_id is not None

    def test_limb_carries_intent(self):
        """Limb should carry original intent."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.PROTECTIVE,
            target_coord=ChamberVector(
                RaCoordinate(5, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=PI / 3
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        limb = create_vector_limb(intent, field, avatar)

        assert limb.intent_phase.emotion_tag == EmotionEnum.PROTECTIVE
        assert limb.intent_phase.phase_angle == pytest.approx(PI / 3)

    def test_limb_has_scalar_route(self):
        """Limb should have computed scalar route."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.FOCUSED,
            target_coord=ChamberVector(
                RaCoordinate(5, 0, 0),
                RaCoordinate(1, 0, 0),
                1.0
            ),
            phase_angle=0
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        limb = create_vector_limb(intent, field, avatar)

        assert len(limb.scalar_route) > 0

    def test_limb_has_mapped_effect(self):
        """Limb should have mapped effect."""
        intent = IntentSignature(
            emotion_tag=EmotionEnum.HEALING,
            target_coord=ChamberVector(
                RaCoordinate(2, 0, 0),
                RaCoordinate(0, 1, 0),
                1.0
            ),
            phase_angle=0
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        limb = create_vector_limb(intent, field, avatar)

        assert limb.mapped_effect.symbolic_name == "heal"


class TestRemapIntentToLimb:
    """Tests for main remap function."""

    def test_creates_profile_for_avatar(self):
        """Should create profile linked to avatar."""
        intent_field = IntentField(
            primary_intent=IntentSignature(
                emotion_tag=EmotionEnum.FOCUSED,
                target_coord=ChamberVector(
                    RaCoordinate(5, 0, 0),
                    RaCoordinate(1, 0, 0),
                    1.0
                ),
                phase_angle=0
            ),
            secondary_intents=[],
            intensity=0.8,
            coherence=0.7
        )
        scalar_field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        profile = remap_intent_to_limb(intent_field, scalar_field, avatar)

        assert profile.avatar_id == avatar.avatar_id

    def test_creates_limb_for_primary_intent(self):
        """Should create limb for primary intent."""
        intent_field = IntentField(
            primary_intent=IntentSignature(
                emotion_tag=EmotionEnum.SHIELDING,
                target_coord=ChamberVector(
                    RaCoordinate(3, 0, 0),
                    RaCoordinate(1, 0, 0),
                    1.0
                ),
                phase_angle=0
            ),
            secondary_intents=[],
            intensity=0.8,
            coherence=0.7
        )
        scalar_field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        profile = remap_intent_to_limb(intent_field, scalar_field, avatar)

        assert len(profile.limbs) >= 1
        assert profile.limbs[0].mapped_effect.symbolic_name == "shield"

    def test_creates_limbs_for_secondary_intents(self):
        """Should create limbs for secondary intents if intensity sufficient."""
        intent_field = IntentField(
            primary_intent=IntentSignature(
                emotion_tag=EmotionEnum.FOCUSED,
                target_coord=ChamberVector(
                    RaCoordinate(5, 0, 0),
                    RaCoordinate(1, 0, 0),
                    1.0
                ),
                phase_angle=0
            ),
            secondary_intents=[
                IntentSignature(
                    emotion_tag=EmotionEnum.PROTECTIVE,
                    target_coord=ChamberVector(
                        RaCoordinate(0, 5, 0),
                        RaCoordinate(0, 1, 0),
                        1.0
                    ),
                    phase_angle=PI / 2
                )
            ],
            intensity=0.8,  # Above 0.3 threshold
            coherence=0.7
        )
        scalar_field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        profile = remap_intent_to_limb(intent_field, scalar_field, avatar)

        assert len(profile.limbs) == 2

    def test_skips_secondary_on_low_intensity(self):
        """Should skip secondary intents if intensity too low."""
        intent_field = IntentField(
            primary_intent=IntentSignature(
                emotion_tag=EmotionEnum.FOCUSED,
                target_coord=ChamberVector(
                    RaCoordinate(5, 0, 0),
                    RaCoordinate(1, 0, 0),
                    1.0
                ),
                phase_angle=0
            ),
            secondary_intents=[
                IntentSignature(
                    emotion_tag=EmotionEnum.PROTECTIVE,
                    target_coord=ChamberVector(
                        RaCoordinate(0, 5, 0),
                        RaCoordinate(0, 1, 0),
                        1.0
                    ),
                    phase_angle=0
                )
            ],
            intensity=0.2,  # Below 0.3 threshold
            coherence=0.7
        )
        scalar_field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        profile = remap_intent_to_limb(intent_field, scalar_field, avatar)

        assert len(profile.limbs) == 1  # Only primary

    def test_computes_total_intensity(self):
        """Should compute total intensity from components."""
        intent_field = IntentField(
            primary_intent=IntentSignature(
                emotion_tag=EmotionEnum.FOCUSED,
                target_coord=ChamberVector(
                    RaCoordinate(5, 0, 0),
                    RaCoordinate(1, 0, 0),
                    1.0
                ),
                phase_angle=0
            ),
            secondary_intents=[],
            intensity=0.8,
            coherence=0.5
        )
        scalar_field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})
        avatar = Avatar(
            avatar_id=uuid.uuid4(),
            position=RaCoordinate(0, 0, 0),
            orientation=0,
            active_shell=HarmonicShellIndex.SHELL_3,
            coherence=0.8
        )

        profile = remap_intent_to_limb(intent_field, scalar_field, avatar)

        # total = intensity * coherence * limb_count = 0.8 * 0.5 * 1 = 0.4
        assert profile.total_intensity == pytest.approx(0.4)


class TestEffectChain:
    """Tests for effect chain extraction."""

    def test_extracts_symbolic_names(self):
        """Should extract symbolic effect names."""
        limb1 = VectorLimb(
            limb_id=uuid.uuid4(),
            intent_phase=IntentSignature(
                EmotionEnum.FOCUSED,
                ChamberVector(RaCoordinate(0, 0, 0), RaCoordinate(1, 0, 0), 1.0),
                0
            ),
            scalar_route=[RaCoordinate(0, 0, 0)],
            active_shell=HarmonicShellIndex.SHELL_3,
            mapped_effect=AvatarEffect(
                EffectType.BOUND, "push", "Ra.Chamber.ForceProject", {}
            )
        )
        limb2 = VectorLimb(
            limb_id=uuid.uuid4(),
            intent_phase=IntentSignature(
                EmotionEnum.PROTECTIVE,
                ChamberVector(RaCoordinate(0, 0, 0), RaCoordinate(0, 1, 0), 1.0),
                0
            ),
            scalar_route=[RaCoordinate(0, 0, 0)],
            active_shell=HarmonicShellIndex.SHELL_6,
            mapped_effect=AvatarEffect(
                EffectType.BOUND, "shield", "Ra.Chamber.BarrierCreate", {}
            )
        )

        profile = VectorRemapProfile(
            limbs=[limb1, limb2],
            avatar_id=uuid.uuid4(),
            total_intensity=1.0
        )

        chain = compute_effect_chain(profile)

        assert chain == ["push", "shield"]


class TestLimbValidation:
    """Tests for limb coherence validation."""

    def test_valid_limb_passes(self):
        """Valid limb should pass validation."""
        limb = VectorLimb(
            limb_id=uuid.uuid4(),
            intent_phase=IntentSignature(
                EmotionEnum.FOCUSED,
                ChamberVector(RaCoordinate(1, 0, 0), RaCoordinate(1, 0, 0), 1.0),
                0
            ),
            scalar_route=[
                RaCoordinate(0, 0, 0),
                RaCoordinate(0.5, 0, 0),
                RaCoordinate(1, 0, 0)
            ],
            active_shell=HarmonicShellIndex.SHELL_3,
            mapped_effect=AvatarEffect(EffectType.BOUND, "push", None, {})
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})

        assert validate_limb_coherence(limb, field)

    def test_empty_route_fails(self):
        """Empty route should fail validation."""
        limb = VectorLimb(
            limb_id=uuid.uuid4(),
            intent_phase=IntentSignature(
                EmotionEnum.FOCUSED,
                ChamberVector(RaCoordinate(1, 0, 0), RaCoordinate(1, 0, 0), 1.0),
                0
            ),
            scalar_route=[],
            active_shell=HarmonicShellIndex.SHELL_3,
            mapped_effect=AvatarEffect(EffectType.BOUND, "push", None, {})
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})

        assert not validate_limb_coherence(limb, field)

    def test_discontinuous_route_fails(self):
        """Route with large discontinuity should fail."""
        limb = VectorLimb(
            limb_id=uuid.uuid4(),
            intent_phase=IntentSignature(
                EmotionEnum.FOCUSED,
                ChamberVector(RaCoordinate(1, 0, 0), RaCoordinate(1, 0, 0), 1.0),
                0
            ),
            scalar_route=[
                RaCoordinate(0, 0, 0),
                RaCoordinate(10, 0, 0),  # Large jump
            ],
            active_shell=HarmonicShellIndex.SHELL_3,
            mapped_effect=AvatarEffect(EffectType.BOUND, "push", None, {})
        )
        field = ScalarField(points=[], mean_alpha=0.5, shell_boundaries={})

        assert not validate_limb_coherence(limb, field)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
