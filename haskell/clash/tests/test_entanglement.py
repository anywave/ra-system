#!/usr/bin/env python3
"""
Prompt 15: Multi-User Harmonic Entanglement - Python Test Harness

Tests the RaBridgeSync and RaCohereNet Clash modules for:
- Multi-user scalar coupling via entanglement bridges
- Consent state management (NONE/PRIVATE/THERAPEUTIC/ENTANGLED/WITHDRAWN/EMERGENCY)
- Bridge modes (MIRROR/COMPLEMENT/ASYMMETRIC/BROADCAST)
- Harmonic compatibility (l-match AND m-diff <= 1)
- Entanglement score calculation with clamping
- Group coherence synchronization (RaCohereNet)
- Safety enforcement with graceful/emergency disentanglement

Codex References:
- Ra.Gates: Consent gating
- Ra.Emergence: Fragment emergence
- Ra.Identity: User identity validation

Integration:
- Prompt 11: Group coherence (entrain())
- Prompt 14: Shared lucid navigation (leader_follow mode)
"""

import json
import sys
import math
import random
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime
from pathlib import Path


# ============================================================================
# Constants (Codex-aligned, per Architect clarifications)
# ============================================================================

PHI_THRESHOLD = 0.03           # Phase alignment window
COHERENCE_TOLERANCE = 0.08     # Adjusted per 15B
GROUP_COHERENCE_MIN = 0.72     # Group stabilization threshold
ENTANGLEMENT_SCORE_MIN = 0.6   # Minimum for stable bridge
SHADOW_COHERENCE_MIN = 0.75    # Required for shadow fragment sharing
MAX_BRIDGES_PER_USER = 5       # Concurrent bridge limit
SAFETY_LATENCY_MS = 250        # Max enforcement latency


# ============================================================================
# Enums
# ============================================================================

class ConsentState(Enum):
    """Unified consent states (Prompt 12 + 15)."""
    NONE = "NONE"
    PRIVATE = "PRIVATE"
    THERAPEUTIC = "THERAPEUTIC"
    ENTANGLED = "ENTANGLED"
    WITHDRAWN = "WITHDRAWN"
    EMERGENCY = "EMERGENCY"


class BridgeMode(Enum):
    """Entanglement bridge modes."""
    MIRROR = "MIRROR"           # Identical experience
    COMPLEMENT = "COMPLEMENT"   # Dual perspectives
    ASYMMETRIC = "ASYMMETRIC"   # Leader/follower
    BROADCAST = "BROADCAST"     # One-to-many


class BridgeState(Enum):
    """Bridge FSM states."""
    IDLE = "IDLE"
    ENTANGLED = "ENTANGLED"
    SUSPENDED = "SUSPENDED"
    TERMINATED = "TERMINATED"


class EmergenceType(Enum):
    """Fragment emergence types."""
    SHARED = "shared"           # MIRROR mode
    DUAL_REFLECT = "dual-reflect"  # COMPLEMENT mode
    RELAY_ACCESS = "relay-access"  # ASYMMETRIC mode
    MULTICAST_SYNC = "multicast-sync"  # BROADCAST mode


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ScalarUser:
    """User identity snapshot for entanglement."""
    user_id: str
    coherence: float = 0.5          # 0.0-1.0
    harmonic: Tuple[int, int] = (0, 0)  # (l, m) indices
    consent_state: ConsentState = ConsentState.NONE
    phi_phase: float = 1.618        # phi^n representation
    override_veto: bool = False     # Emergency block

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "coherence": round(self.coherence, 3),
            "harmonic": self.harmonic,
            "consent_state": self.consent_state.value,
            "phi_phase": round(self.phi_phase, 4),
            "override_veto": self.override_veto
        }


@dataclass
class ScalarBridge:
    """Entanglement bridge between two users."""
    fragment_id: str
    users: List[str]
    created: str = ""
    mode: BridgeMode = BridgeMode.MIRROR
    active: bool = True
    stable: bool = False
    coherence_delta: float = 0.0
    phase_delta: float = 0.0
    harmonic_compatible: bool = False
    entanglement_score: float = 0.0

    def __post_init__(self):
        if not self.created:
            self.created = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> dict:
        return {
            "fragment_id": self.fragment_id,
            "users": self.users,
            "created": self.created,
            "mode": self.mode.value,
            "active": self.active,
            "stable": self.stable,
            "coherence_delta": round(self.coherence_delta, 3),
            "phase_delta": round(self.phase_delta, 4),
            "harmonic_compatible": self.harmonic_compatible,
            "entanglement_score": round(self.entanglement_score, 3)
        }


@dataclass
class GroupBridge:
    """Multi-user group bridge (RaCohereNet)."""
    group_id: str
    users: List[str]
    leader_id: str
    average_coherence: float = 0.0
    phi_stability: bool = False
    fragment_sync_mode: BridgeMode = BridgeMode.BROADCAST
    entanglement_matrix: List[List[int]] = field(default_factory=list)
    safe_to_broadcast: bool = False

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "users": self.users,
            "leader_id": self.leader_id,
            "average_coherence": round(self.average_coherence, 3),
            "phi_stability": self.phi_stability,
            "fragment_sync_mode": self.fragment_sync_mode.value,
            "entanglement_matrix": self.entanglement_matrix,
            "safe_to_broadcast": self.safe_to_broadcast
        }


@dataclass
class EmergenceResult:
    """Result of triggered emergence."""
    status: str
    fragment_id: str = ""
    mode: BridgeMode = BridgeMode.MIRROR
    flux_boost: float = 0.0
    emergence_type: EmergenceType = EmergenceType.SHARED
    users: List[str] = field(default_factory=list)
    reason: str = ""


# ============================================================================
# RaEntanglement Engine
# ============================================================================

class RaEntanglement:
    """
    Multi-User Scalar Coupling Engine (Prompt 15).

    Implements:
    - Pairwise entanglement with consent validation
    - Bridge modes (MIRROR/COMPLEMENT/ASYMMETRIC/BROADCAST)
    - Safety enforcement with graceful/emergency disentanglement
    - Group coherence synchronization
    """

    @staticmethod
    def check_harmonic_compatibility(user_a: ScalarUser, user_b: ScalarUser) -> bool:
        """
        Check harmonic compatibility between users.
        Per architect: l-match AND m-diff <= 1 (stricter than OR).
        """
        l_match = user_a.harmonic[0] == user_b.harmonic[0]
        m_near = abs(user_a.harmonic[1] - user_b.harmonic[1]) <= 1
        return l_match and m_near

    @staticmethod
    def calculate_entanglement_score(user_a: ScalarUser, user_b: ScalarUser) -> float:
        """
        Calculate entanglement score with clamping.
        Formula: max(0, 1.0 - (coherence_delta + phase_delta / 2))
        """
        coherence_delta = abs(user_a.coherence - user_b.coherence)
        phase_delta = abs(user_a.phi_phase - user_b.phi_phase)
        raw_score = 1.0 - (coherence_delta + phase_delta / 2.0)
        return max(0.0, raw_score)

    @staticmethod
    def is_entanglement_possible(user_a: ScalarUser, user_b: ScalarUser) -> Tuple[bool, str]:
        """
        Check if entanglement is possible between two users.
        Returns (possible, reason).
        """
        # Consent check
        if user_a.consent_state != ConsentState.ENTANGLED:
            return False, f"User {user_a.user_id} consent state is {user_a.consent_state.value}"
        if user_b.consent_state != ConsentState.ENTANGLED:
            return False, f"User {user_b.user_id} consent state is {user_b.consent_state.value}"

        # Veto check
        if user_a.override_veto:
            return False, f"User {user_a.user_id} has override veto active"
        if user_b.override_veto:
            return False, f"User {user_b.user_id} has override veto active"

        # Phase alignment
        phase_delta = abs(user_a.phi_phase - user_b.phi_phase)
        if phase_delta > PHI_THRESHOLD:
            return False, f"Phase delta {phase_delta:.4f} exceeds threshold {PHI_THRESHOLD}"

        # Coherence tolerance
        coherence_delta = abs(user_a.coherence - user_b.coherence)
        if coherence_delta > COHERENCE_TOLERANCE:
            return False, f"Coherence delta {coherence_delta:.3f} exceeds tolerance {COHERENCE_TOLERANCE}"

        # Harmonic compatibility
        if not RaEntanglement.check_harmonic_compatibility(user_a, user_b):
            return False, "Harmonic incompatibility (l-mismatch or m-diff > 1)"

        # Entanglement score
        score = RaEntanglement.calculate_entanglement_score(user_a, user_b)
        if score < ENTANGLEMENT_SCORE_MIN:
            return False, f"Entanglement score {score:.3f} below minimum {ENTANGLEMENT_SCORE_MIN}"

        return True, "OK"

    @staticmethod
    def create_bridge(fragment_id: str, user_a: ScalarUser, user_b: ScalarUser,
                      mode: BridgeMode = BridgeMode.MIRROR) -> Dict[str, Any]:
        """Create entanglement bridge between two users."""
        possible, reason = RaEntanglement.is_entanglement_possible(user_a, user_b)

        if not possible:
            return {"status": "DENIED", "reason": reason}

        bridge = ScalarBridge(
            fragment_id=fragment_id,
            users=[user_a.user_id, user_b.user_id],
            mode=mode,
            coherence_delta=abs(user_a.coherence - user_b.coherence),
            phase_delta=abs(user_a.phi_phase - user_b.phi_phase),
            harmonic_compatible=True,
            entanglement_score=RaEntanglement.calculate_entanglement_score(user_a, user_b)
        )
        bridge.stable = bridge.entanglement_score >= ENTANGLEMENT_SCORE_MIN

        return {
            "status": "ENTANGLED",
            "bridge": bridge.to_dict(),
            "entanglement_score": bridge.entanglement_score
        }

    @staticmethod
    def enforce_safety(bridge: Dict, user_map: Dict[str, ScalarUser]) -> Tuple[bool, str]:
        """
        Enforce safety on active bridge.
        Must execute within 250ms (SAFETY_LATENCY_MS).
        """
        if not bridge.get("active", False):
            return False, "Bridge not active"

        if not bridge.get("stable", False):
            return False, "Bridge not stable"

        for uid in bridge.get("users", []):
            user = user_map.get(uid)
            if user is None:
                return False, f"User {uid} not found"
            if user.override_veto:
                return False, f"User {uid} has veto active"
            if user.consent_state != ConsentState.ENTANGLED:
                return False, f"User {uid} consent changed to {user.consent_state.value}"

        return True, "Safe"

    @staticmethod
    def trigger_emergence(bridge: Dict, user_map: Dict[str, ScalarUser]) -> EmergenceResult:
        """Trigger fragment emergence through entanglement bridge."""
        safe, reason = RaEntanglement.enforce_safety(bridge, user_map)

        if not safe:
            return EmergenceResult(status="BLOCKED", reason=reason)

        users = bridge.get("users", [])
        if len(users) < 2:
            return EmergenceResult(status="BLOCKED", reason="Insufficient users")

        u1 = user_map[users[0]]
        u2 = user_map[users[1]]
        score = bridge.get("entanglement_score", 0)
        flux_amplifier = (u1.coherence + u2.coherence) / 2 * score

        mode = BridgeMode(bridge.get("mode", "MIRROR"))
        emergence_type = {
            BridgeMode.MIRROR: EmergenceType.SHARED,
            BridgeMode.COMPLEMENT: EmergenceType.DUAL_REFLECT,
            BridgeMode.ASYMMETRIC: EmergenceType.RELAY_ACCESS,
            BridgeMode.BROADCAST: EmergenceType.MULTICAST_SYNC
        }.get(mode, EmergenceType.SHARED)

        return EmergenceResult(
            status="EMERGENCE_TRIGGERED",
            fragment_id=bridge.get("fragment_id", ""),
            mode=mode,
            flux_boost=round(flux_amplifier, 3),
            emergence_type=emergence_type,
            users=users
        )

    @staticmethod
    def can_share_shadow(user_a: ScalarUser, user_b: ScalarUser) -> bool:
        """Check if shadow fragment can be shared (requires dual THERAPEUTIC + high coherence)."""
        # Both must have at least THERAPEUTIC consent
        valid_consent = {ConsentState.THERAPEUTIC, ConsentState.ENTANGLED}
        if user_a.consent_state not in valid_consent:
            return False
        if user_b.consent_state not in valid_consent:
            return False

        # Combined coherence must exceed threshold
        avg_coherence = (user_a.coherence + user_b.coherence) / 2
        return avg_coherence >= SHADOW_COHERENCE_MIN

    @staticmethod
    def graceful_disentangle(bridge: Dict) -> Dict:
        """Gracefully suspend bridge, notify users."""
        bridge["active"] = False
        bridge["stable"] = False
        return {
            "status": "SUSPENDED",
            "bridge": bridge,
            "notification": "Bridge gracefully suspended"
        }

    @staticmethod
    def emergency_sever(bridge: Dict) -> Dict:
        """Emergency bridge severance with fragment withdrawal."""
        bridge["active"] = False
        bridge["stable"] = False
        return {
            "status": "TERMINATED",
            "bridge": bridge,
            "notification": "Emergency severance - fragments withdrawn"
        }


# ============================================================================
# RaCohereNet - Group Synchronization Engine
# ============================================================================

class RaCohereNet:
    """
    Multi-user group coherence protocol (Prompt 15D).

    Implements:
    - N-user group stabilization (N <= 8)
    - Pairwise entanglement matrix
    - Leader-based synchronization
    - Broadcast fragment sharing
    """

    @staticmethod
    def calculate_group_coherence(users: List[ScalarUser]) -> float:
        """Calculate average coherence across eligible users."""
        eligible = [u for u in users
                   if not u.override_veto and u.consent_state == ConsentState.ENTANGLED]
        if not eligible:
            return 0.0
        return sum(u.coherence for u in eligible) / len(eligible)

    @staticmethod
    def calculate_phi_stability(users: List[ScalarUser]) -> Tuple[bool, float]:
        """Check phi phase stability across group (deviation <= 0.04)."""
        eligible = [u for u in users
                   if not u.override_veto and u.consent_state == ConsentState.ENTANGLED]
        if len(eligible) < 2:
            return False, 0.0

        phases = [u.phi_phase for u in eligible]
        max_dev = max(phases) - min(phases)
        return max_dev <= 0.04, max_dev

    @staticmethod
    def build_entanglement_matrix(users: List[ScalarUser]) -> List[List[int]]:
        """Build N x N matrix of pairwise viable bridges."""
        n = len(users)
        matrix = [[0] * n for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i][j] = 1  # Self
                else:
                    possible, _ = RaEntanglement.is_entanglement_possible(users[i], users[j])
                    matrix[i][j] = 1 if possible else 0

        return matrix

    @staticmethod
    def select_leader(users: List[ScalarUser]) -> Optional[ScalarUser]:
        """Select group leader (highest coherence eligible user)."""
        eligible = [u for u in users
                   if not u.override_veto and u.consent_state == ConsentState.ENTANGLED]
        if not eligible:
            return None
        return max(eligible, key=lambda u: u.coherence)

    @staticmethod
    def group_stabilized(users: List[ScalarUser]) -> Dict[str, Any]:
        """Check if group is stabilized for broadcast."""
        avg_coherence = RaCohereNet.calculate_group_coherence(users)
        phi_stable, phi_dev = RaCohereNet.calculate_phi_stability(users)

        if avg_coherence < GROUP_COHERENCE_MIN:
            return {
                "status": "UNSTABLE",
                "group_coherence": round(avg_coherence, 3),
                "reason": f"Coherence {avg_coherence:.3f} below minimum {GROUP_COHERENCE_MIN}"
            }

        if not phi_stable:
            return {
                "status": "UNSTABLE",
                "group_coherence": round(avg_coherence, 3),
                "phi_deviation": round(phi_dev, 4),
                "reason": f"Phi deviation {phi_dev:.4f} exceeds 0.04"
            }

        return {
            "status": "STABILIZED",
            "group_coherence": round(avg_coherence, 3),
            "phi_deviation": round(phi_dev, 4)
        }

    @staticmethod
    def create_group_bridge(group_id: str, users: List[ScalarUser]) -> Dict[str, Any]:
        """Create group bridge with full synchronization data."""
        stability = RaCohereNet.group_stabilized(users)

        if stability["status"] != "STABILIZED":
            return {
                "status": "DENIED",
                "reason": stability.get("reason", "Group not stabilized")
            }

        leader = RaCohereNet.select_leader(users)
        if leader is None:
            return {"status": "DENIED", "reason": "No eligible leader"}

        matrix = RaCohereNet.build_entanglement_matrix(users)
        avg_coherence = RaCohereNet.calculate_group_coherence(users)
        phi_stable, _ = RaCohereNet.calculate_phi_stability(users)

        # Check all consents
        all_consented = all(
            u.consent_state == ConsentState.ENTANGLED and not u.override_veto
            for u in users
        )

        group = GroupBridge(
            group_id=group_id,
            users=[u.user_id for u in users],
            leader_id=leader.user_id,
            average_coherence=avg_coherence,
            phi_stability=phi_stable,
            fragment_sync_mode=BridgeMode.BROADCAST,
            entanglement_matrix=matrix,
            safe_to_broadcast=all_consented and avg_coherence >= GROUP_COHERENCE_MIN
        )

        return {
            "status": "GROUP_ENTANGLED",
            "group_bridge": group.to_dict()
        }

    @staticmethod
    def fallback_to_pairwise(group: GroupBridge, user_map: Dict[str, ScalarUser]) -> List[Dict]:
        """Fallback to pairwise bridges when group coherence drops."""
        bridges = []
        users = [user_map[uid] for uid in group.users if uid in user_map]

        for i in range(len(users)):
            for j in range(i + 1, len(users)):
                result = RaEntanglement.create_bridge(
                    f"pair-{users[i].user_id}-{users[j].user_id}",
                    users[i], users[j],
                    BridgeMode.MIRROR
                )
                if result["status"] == "ENTANGLED":
                    bridges.append(result["bridge"])

        return bridges


# ============================================================================
# Test Scenarios
# ============================================================================

def test_harmonic_compatibility():
    """Test: Harmonic compatibility (l-match AND m-diff <= 1)."""
    # Compatible: l matches, m differs by 1
    u1 = ScalarUser("a", harmonic=(3, 5))
    u2 = ScalarUser("b", harmonic=(3, 6))
    assert RaEntanglement.check_harmonic_compatibility(u1, u2)

    # Incompatible: l differs
    u3 = ScalarUser("c", harmonic=(4, 5))
    assert not RaEntanglement.check_harmonic_compatibility(u1, u3)

    # Incompatible: m differs by > 1
    u4 = ScalarUser("d", harmonic=(3, 8))
    assert not RaEntanglement.check_harmonic_compatibility(u1, u4)

    print("  [PASS] harmonic_compatibility")


def test_entanglement_score():
    """Test: Entanglement score calculation with clamping."""
    u1 = ScalarUser("a", coherence=0.8, phi_phase=1.618)
    u2 = ScalarUser("b", coherence=0.78, phi_phase=1.620)

    score = RaEntanglement.calculate_entanglement_score(u1, u2)
    assert score > 0.9  # Should be high for close values
    assert score <= 1.0

    # Test clamping for extreme values
    u3 = ScalarUser("c", coherence=0.1, phi_phase=0.5)
    u4 = ScalarUser("d", coherence=0.9, phi_phase=2.5)
    score2 = RaEntanglement.calculate_entanglement_score(u3, u4)
    assert score2 >= 0.0  # Should be clamped to 0

    print("  [PASS] entanglement_score")


def test_bridge_creation():
    """Test: Bridge creation with consent validation."""
    u1 = ScalarUser("user_a", coherence=0.81, harmonic=(3, 5),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.618)
    u2 = ScalarUser("user_b", coherence=0.79, harmonic=(3, 6),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.620)

    result = RaEntanglement.create_bridge("frag-001", u1, u2, BridgeMode.MIRROR)
    assert result["status"] == "ENTANGLED"
    assert result["bridge"]["stable"]

    # Test with wrong consent
    u3 = ScalarUser("user_c", coherence=0.80, harmonic=(3, 5),
                    consent_state=ConsentState.PRIVATE, phi_phase=1.618)
    result2 = RaEntanglement.create_bridge("frag-002", u1, u3)
    assert result2["status"] == "DENIED"

    print("  [PASS] bridge_creation")


def test_safety_enforcement():
    """Test: Safety enforcement with veto and consent changes."""
    u1 = ScalarUser("user_a", coherence=0.81, harmonic=(3, 5),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.618)
    u2 = ScalarUser("user_b", coherence=0.79, harmonic=(3, 6),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.620)

    result = RaEntanglement.create_bridge("frag-001", u1, u2)
    bridge = result["bridge"]
    user_map = {"user_a": u1, "user_b": u2}

    # Should be safe initially
    safe, reason = RaEntanglement.enforce_safety(bridge, user_map)
    assert safe

    # Activate veto
    u1.override_veto = True
    safe2, reason2 = RaEntanglement.enforce_safety(bridge, user_map)
    assert not safe2
    assert "veto" in reason2.lower()

    print("  [PASS] safety_enforcement")


def test_emergence_trigger():
    """Test: Fragment emergence through bridge."""
    u1 = ScalarUser("user_a", coherence=0.85, harmonic=(3, 5),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.618)
    u2 = ScalarUser("user_b", coherence=0.82, harmonic=(3, 6),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.620)

    result = RaEntanglement.create_bridge("frag-001", u1, u2, BridgeMode.COMPLEMENT)
    bridge = result["bridge"]
    user_map = {"user_a": u1, "user_b": u2}

    emergence = RaEntanglement.trigger_emergence(bridge, user_map)
    assert emergence.status == "EMERGENCE_TRIGGERED"
    assert emergence.emergence_type == EmergenceType.DUAL_REFLECT
    assert emergence.flux_boost > 0

    print("  [PASS] emergence_trigger")


def test_shadow_sharing():
    """Test: Shadow fragment sharing requires dual consent + high coherence."""
    # Valid for shadow sharing
    u1 = ScalarUser("user_a", coherence=0.85, consent_state=ConsentState.THERAPEUTIC)
    u2 = ScalarUser("user_b", coherence=0.80, consent_state=ConsentState.ENTANGLED)
    assert RaEntanglement.can_share_shadow(u1, u2)

    # Invalid: low coherence
    u3 = ScalarUser("user_c", coherence=0.60, consent_state=ConsentState.THERAPEUTIC)
    u4 = ScalarUser("user_d", coherence=0.65, consent_state=ConsentState.THERAPEUTIC)
    assert not RaEntanglement.can_share_shadow(u3, u4)

    # Invalid: wrong consent
    u5 = ScalarUser("user_e", coherence=0.90, consent_state=ConsentState.PRIVATE)
    assert not RaEntanglement.can_share_shadow(u1, u5)

    print("  [PASS] shadow_sharing")


def test_group_stabilization():
    """Test: RaCohereNet group stabilization."""
    users = [
        ScalarUser("u1", coherence=0.80, harmonic=(3, 5),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.618),
        ScalarUser("u2", coherence=0.78, harmonic=(3, 6),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.620),
        ScalarUser("u3", coherence=0.75, harmonic=(3, 5),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.615),
    ]

    result = RaCohereNet.group_stabilized(users)
    assert result["status"] == "STABILIZED"
    assert result["group_coherence"] >= GROUP_COHERENCE_MIN

    # Unstable group (low coherence)
    users[0].coherence = 0.40
    users[1].coherence = 0.45
    users[2].coherence = 0.50
    result2 = RaCohereNet.group_stabilized(users)
    assert result2["status"] == "UNSTABLE"

    print("  [PASS] group_stabilization")


def test_entanglement_matrix():
    """Test: Pairwise entanglement matrix generation."""
    users = [
        ScalarUser("u1", coherence=0.80, harmonic=(3, 5),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.618),
        ScalarUser("u2", coherence=0.78, harmonic=(3, 6),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.620),
        ScalarUser("u3", coherence=0.75, harmonic=(4, 5),  # Different l!
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.615),
    ]

    matrix = RaCohereNet.build_entanglement_matrix(users)
    assert len(matrix) == 3
    assert len(matrix[0]) == 3
    assert matrix[0][0] == 1  # Self
    assert matrix[0][1] == 1  # u1-u2 compatible
    assert matrix[0][2] == 0  # u1-u3 incompatible (different l)

    print("  [PASS] entanglement_matrix")


def test_group_bridge_creation():
    """Test: Full group bridge creation."""
    users = [
        ScalarUser("u1", coherence=0.80, harmonic=(3, 5),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.618),
        ScalarUser("u2", coherence=0.78, harmonic=(3, 6),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.620),
        ScalarUser("u3", coherence=0.82, harmonic=(3, 5),
                   consent_state=ConsentState.ENTANGLED, phi_phase=1.615),
    ]

    result = RaCohereNet.create_group_bridge("session_alpha", users)
    assert result["status"] == "GROUP_ENTANGLED"
    assert result["group_bridge"]["leader_id"] == "u3"  # Highest coherence
    assert result["group_bridge"]["safe_to_broadcast"]

    print("  [PASS] group_bridge_creation")


def test_graceful_disentangle():
    """Test: Graceful bridge suspension."""
    u1 = ScalarUser("user_a", coherence=0.81, harmonic=(3, 5),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.618)
    u2 = ScalarUser("user_b", coherence=0.79, harmonic=(3, 6),
                    consent_state=ConsentState.ENTANGLED, phi_phase=1.620)

    result = RaEntanglement.create_bridge("frag-001", u1, u2)
    bridge = result["bridge"]

    suspend_result = RaEntanglement.graceful_disentangle(bridge)
    assert suspend_result["status"] == "SUSPENDED"
    assert not suspend_result["bridge"]["active"]

    print("  [PASS] graceful_disentangle")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" PROMPT 15: MULTI-USER HARMONIC ENTANGLEMENT - TEST SUITE".center(70))
    print("=" * 70)
    print()

    tests = [
        ("Harmonic Compatibility", test_harmonic_compatibility),
        ("Entanglement Score", test_entanglement_score),
        ("Bridge Creation", test_bridge_creation),
        ("Safety Enforcement", test_safety_enforcement),
        ("Emergence Trigger", test_emergence_trigger),
        ("Shadow Sharing", test_shadow_sharing),
        ("Group Stabilization", test_group_stabilization),
        ("Entanglement Matrix", test_entanglement_matrix),
        ("Group Bridge Creation", test_group_bridge_creation),
        ("Graceful Disentangle", test_graceful_disentangle),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        print(f"\nTest: {name}")
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed += 1

    print("\n" + "-" * 70)
    print(f" Results: {passed} passed, {failed} failed".center(70))
    print("-" * 70)

    # Demo
    if "--demo" in sys.argv or len(sys.argv) == 1:
        print("\n\n")
        print("=" * 70)
        print(" ENTANGLEMENT DEMO".center(70))
        print("=" * 70)

        print("\n[1] Creating two entangled users...")
        u1 = ScalarUser("Alice", coherence=0.85, harmonic=(3, 5),
                        consent_state=ConsentState.ENTANGLED, phi_phase=1.618)
        u2 = ScalarUser("Bob", coherence=0.82, harmonic=(3, 6),
                        consent_state=ConsentState.ENTANGLED, phi_phase=1.620)
        print(f"    Alice: coherence={u1.coherence}, harmonic={u1.harmonic}")
        print(f"    Bob: coherence={u2.coherence}, harmonic={u2.harmonic}")

        print("\n[2] Creating MIRROR bridge...")
        result = RaEntanglement.create_bridge("frag-demo", u1, u2, BridgeMode.MIRROR)
        print(f"    Status: {result['status']}")
        print(f"    Score: {result.get('entanglement_score', 0):.3f}")

        print("\n[3] Triggering emergence...")
        bridge = result["bridge"]
        user_map = {"Alice": u1, "Bob": u2}
        emergence = RaEntanglement.trigger_emergence(bridge, user_map)
        print(f"    Status: {emergence.status}")
        print(f"    Type: {emergence.emergence_type.value}")
        print(f"    Flux boost: {emergence.flux_boost:.3f}")

        print("\n[4] Creating 3-user group bridge...")
        u3 = ScalarUser("Carol", coherence=0.80, harmonic=(3, 5),
                        consent_state=ConsentState.ENTANGLED, phi_phase=1.615)
        group_result = RaCohereNet.create_group_bridge("session_demo", [u1, u2, u3])
        print(f"    Status: {group_result['status']}")
        if group_result["status"] == "GROUP_ENTANGLED":
            gb = group_result["group_bridge"]
            print(f"    Leader: {gb['leader_id']}")
            print(f"    Safe to broadcast: {gb['safe_to_broadcast']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
