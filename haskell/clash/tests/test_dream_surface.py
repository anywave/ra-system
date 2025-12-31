"""
Test Suite for Ra.Dream.Surface (P75)
Dream-memory buffer system managing fragments from sleep/liminal states.

Tests temporal fragility, reverse-inversion logic, reentry detection,
and non-linear coherence behavior.
"""

import pytest
import math
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

# Constants
PHI = 1.618033988749895
REENTRY_WINDOW_HOURS = 72
FRAGILITY_DECAY_PER_CYCLE = 0.2
ALPHA_REENTRY_THRESHOLD = 0.7
INVERSION_ALPHA_THRESHOLD = 0.5


class DreamState(Enum):
    """Dream state phases."""
    LIMINAL = auto()
    REM = auto()
    DEEP = auto()
    WAKING = auto()
    LUCID = auto()


class InversionTrigger(Enum):
    """What triggers inversion flag."""
    ALPHA_FLIP = auto()        # ∇α polarity reversal
    VALENCE_SHIFT = auto()     # Emotional valence change
    SYMBOLIC_FLIP = auto()     # Symbolic contradiction


@dataclass
class DreamTime:
    """Dream-relative timestamp."""
    real_timestamp: datetime
    dream_ticks: int          # Ticks within dream
    cycle_index: int          # Which sleep cycle
    state: DreamState


@dataclass
class FragmentID:
    """Fragment identifier."""
    namespace: int
    index: int
    uuid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


@dataclass
class DreamFragment:
    """Dream-originated fragment."""
    fragment_id: FragmentID
    dream_timestamp: DreamTime
    inversion_flag: bool
    fragility_score: float    # 0.0-1.0, decay risk
    reentry_detected: bool
    valence: float            # -1.0 to 1.0 emotional tone
    symbolic_hash: int        # Hash of symbolic content


@dataclass
class WakingFragment:
    """Fragment encountered in waking pipeline."""
    fragment_id: FragmentID
    timestamp: datetime
    alpha: float
    symbolic_hash: int


class DreamSurfaceBuffer:
    """Dream-memory buffer system."""

    def __init__(self):
        self.fragments: List[DreamFragment] = []
        self.reentry_log: List[Tuple[FragmentID, datetime]] = []

    def insert_fragment(self, fragment: DreamFragment) -> None:
        """Insert a dream fragment into the buffer."""
        self.fragments.append(fragment)

    def get_fragment(self, fragment_id: FragmentID) -> Optional[DreamFragment]:
        """Retrieve fragment by ID."""
        for f in self.fragments:
            if (f.fragment_id.namespace == fragment_id.namespace and
                f.fragment_id.index == fragment_id.index):
                return f
        return None

    def detect_reentry(self, waking_fragments: List[WakingFragment],
                       current_time: datetime) -> List[DreamFragment]:
        """
        Detect fragments that originated in dreams and reappeared in waking.
        Uses 72-hour window and alpha > 0.7 coherence threshold.
        """
        reentries = []

        for waking in waking_fragments:
            for dream in self.fragments:
                if dream.reentry_detected:
                    continue  # Already detected

                # Check symbolic hash match
                if dream.symbolic_hash != waking.symbolic_hash:
                    continue

                # Check time window (72 hours)
                time_delta = current_time - dream.dream_timestamp.real_timestamp
                if time_delta > timedelta(hours=REENTRY_WINDOW_HOURS):
                    continue

                # Check coherence threshold
                if waking.alpha < ALPHA_REENTRY_THRESHOLD:
                    continue

                # Reentry detected!
                dream.reentry_detected = True
                self.reentry_log.append((dream.fragment_id, current_time))
                reentries.append(dream)

        return reentries

    def decay_fragility(self, cycles_elapsed: int) -> None:
        """Decay fragility scores based on wake/dream cycles."""
        for fragment in self.fragments:
            decay = cycles_elapsed * FRAGILITY_DECAY_PER_CYCLE
            fragment.fragility_score = max(0.0, fragment.fragility_score - decay)

    def get_active_fragments(self, min_fragility: float = 0.1) -> List[DreamFragment]:
        """Get fragments that haven't fully decayed."""
        return [f for f in self.fragments if f.fragility_score >= min_fragility]

    def prune_decayed(self, threshold: float = 0.05) -> int:
        """Remove fragments below fragility threshold."""
        before = len(self.fragments)
        self.fragments = [f for f in self.fragments if f.fragility_score >= threshold]
        return before - len(self.fragments)


def check_inversion_trigger(
    prev_alpha: float,
    curr_alpha: float,
    prev_valence: float,
    curr_valence: float,
    prev_symbolic: int,
    curr_symbolic: int,
    trigger_type: InversionTrigger
) -> bool:
    """Check if inversion should be triggered."""
    if trigger_type == InversionTrigger.ALPHA_FLIP:
        # ∇α polarity reversal (sign change in gradient)
        gradient_prev = curr_alpha - prev_alpha
        return (prev_alpha > INVERSION_ALPHA_THRESHOLD and
                curr_alpha < INVERSION_ALPHA_THRESHOLD)

    elif trigger_type == InversionTrigger.VALENCE_SHIFT:
        # Emotional valence sign change
        return (prev_valence > 0 and curr_valence < 0) or \
               (prev_valence < 0 and curr_valence > 0)

    elif trigger_type == InversionTrigger.SYMBOLIC_FLIP:
        # Symbolic hash contradiction (XOR pattern)
        return prev_symbolic != curr_symbolic

    return False


def compute_fragility_from_state(state: DreamState, alpha: float) -> float:
    """Compute initial fragility score from dream state and coherence."""
    base_fragility = {
        DreamState.LIMINAL: 0.9,   # Very fragile
        DreamState.REM: 0.7,
        DreamState.DEEP: 0.5,
        DreamState.WAKING: 0.3,
        DreamState.LUCID: 0.4,
    }

    base = base_fragility.get(state, 0.5)
    # Higher alpha = more stable = lower fragility
    alpha_factor = 1.0 - (alpha * 0.5)
    return min(1.0, base * alpha_factor)


def create_dream_fragment(
    namespace: int,
    index: int,
    timestamp: datetime,
    dream_ticks: int,
    cycle_index: int,
    state: DreamState,
    alpha: float,
    valence: float,
    symbolic_hash: int,
    check_inversion: bool = False,
    prev_alpha: float = 0.0
) -> DreamFragment:
    """Create a new dream fragment."""
    fragment_id = FragmentID(namespace, index)
    dream_time = DreamTime(timestamp, dream_ticks, cycle_index, state)

    fragility = compute_fragility_from_state(state, alpha)

    # Check for inversion (alpha flip by default)
    inversion = False
    if check_inversion:
        inversion = check_inversion_trigger(
            prev_alpha, alpha, 0.0, valence, 0, symbolic_hash,
            InversionTrigger.ALPHA_FLIP
        )

    return DreamFragment(
        fragment_id=fragment_id,
        dream_timestamp=dream_time,
        inversion_flag=inversion,
        fragility_score=fragility,
        reentry_detected=False,
        valence=valence,
        symbolic_hash=symbolic_hash
    )


# ============== TEST CLASSES ==============

class TestDreamFragment:
    """Tests for DreamFragment creation and properties."""

    def test_create_basic_fragment(self):
        """Test basic fragment creation."""
        frag = create_dream_fragment(
            namespace=1, index=100,
            timestamp=datetime.now(),
            dream_ticks=42,
            cycle_index=2,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=12345
        )

        assert frag.fragment_id.namespace == 1
        assert frag.fragment_id.index == 100
        assert frag.dream_timestamp.dream_ticks == 42
        assert frag.dream_timestamp.cycle_index == 2
        assert not frag.reentry_detected

    def test_fragility_score_range(self):
        """Test fragility score is in valid range."""
        for state in DreamState:
            for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
                fragility = compute_fragility_from_state(state, alpha)
                assert 0.0 <= fragility <= 1.0

    def test_liminal_most_fragile(self):
        """Test liminal state has highest fragility."""
        liminal = compute_fragility_from_state(DreamState.LIMINAL, 0.5)
        rem = compute_fragility_from_state(DreamState.REM, 0.5)
        deep = compute_fragility_from_state(DreamState.DEEP, 0.5)

        assert liminal > rem > deep

    def test_high_alpha_reduces_fragility(self):
        """Test higher alpha reduces fragility."""
        low_alpha = compute_fragility_from_state(DreamState.REM, 0.2)
        high_alpha = compute_fragility_from_state(DreamState.REM, 0.9)

        assert high_alpha < low_alpha

    def test_inversion_flag_alpha_flip(self):
        """Test inversion triggered by alpha flip."""
        frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=10,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.3,  # Below threshold
            valence=0.5,
            symbolic_hash=999,
            check_inversion=True,
            prev_alpha=0.8  # Was above threshold
        )

        assert frag.inversion_flag is True

    def test_no_inversion_when_alpha_stable(self):
        """Test no inversion when alpha stays above threshold."""
        frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=10,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.7,
            valence=0.5,
            symbolic_hash=999,
            check_inversion=True,
            prev_alpha=0.8
        )

        assert frag.inversion_flag is False


class TestInversionTriggers:
    """Tests for inversion trigger detection."""

    def test_alpha_flip_inversion(self):
        """Test alpha flip triggers inversion."""
        result = check_inversion_trigger(
            prev_alpha=0.8, curr_alpha=0.3,
            prev_valence=0.5, curr_valence=0.5,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.ALPHA_FLIP
        )
        assert result is True

    def test_alpha_flip_no_inversion_both_high(self):
        """Test no inversion when both alphas high."""
        result = check_inversion_trigger(
            prev_alpha=0.8, curr_alpha=0.7,
            prev_valence=0.5, curr_valence=0.5,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.ALPHA_FLIP
        )
        assert result is False

    def test_valence_shift_positive_to_negative(self):
        """Test valence shift from positive to negative."""
        result = check_inversion_trigger(
            prev_alpha=0.5, curr_alpha=0.5,
            prev_valence=0.7, curr_valence=-0.3,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.VALENCE_SHIFT
        )
        assert result is True

    def test_valence_shift_negative_to_positive(self):
        """Test valence shift from negative to positive."""
        result = check_inversion_trigger(
            prev_alpha=0.5, curr_alpha=0.5,
            prev_valence=-0.5, curr_valence=0.5,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.VALENCE_SHIFT
        )
        assert result is True

    def test_no_valence_shift_same_sign(self):
        """Test no inversion when valence stays same sign."""
        result = check_inversion_trigger(
            prev_alpha=0.5, curr_alpha=0.5,
            prev_valence=0.3, curr_valence=0.8,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.VALENCE_SHIFT
        )
        assert result is False

    def test_symbolic_flip(self):
        """Test symbolic hash difference triggers flip."""
        result = check_inversion_trigger(
            prev_alpha=0.5, curr_alpha=0.5,
            prev_valence=0.5, curr_valence=0.5,
            prev_symbolic=100, curr_symbolic=200,
            trigger_type=InversionTrigger.SYMBOLIC_FLIP
        )
        assert result is True

    def test_no_symbolic_flip_same_hash(self):
        """Test no flip when symbolic hash same."""
        result = check_inversion_trigger(
            prev_alpha=0.5, curr_alpha=0.5,
            prev_valence=0.5, curr_valence=0.5,
            prev_symbolic=100, curr_symbolic=100,
            trigger_type=InversionTrigger.SYMBOLIC_FLIP
        )
        assert result is False


class TestDreamSurfaceBuffer:
    """Tests for DreamSurfaceBuffer operations."""

    def test_insert_and_retrieve(self):
        """Test fragment insertion and retrieval."""
        buffer = DreamSurfaceBuffer()

        frag = create_dream_fragment(
            namespace=1, index=42,
            timestamp=datetime.now(),
            dream_ticks=100,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=12345
        )

        buffer.insert_fragment(frag)

        retrieved = buffer.get_fragment(FragmentID(1, 42))
        assert retrieved is not None
        assert retrieved.symbolic_hash == 12345

    def test_get_nonexistent_fragment(self):
        """Test retrieving non-existent fragment returns None."""
        buffer = DreamSurfaceBuffer()
        result = buffer.get_fragment(FragmentID(99, 99))
        assert result is None

    def test_multiple_fragments(self):
        """Test buffer with multiple fragments."""
        buffer = DreamSurfaceBuffer()

        for i in range(5):
            frag = create_dream_fragment(
                namespace=1, index=i,
                timestamp=datetime.now(),
                dream_ticks=i * 10,
                cycle_index=1,
                state=DreamState.REM,
                alpha=0.5 + i * 0.1,
                valence=0.0,
                symbolic_hash=1000 + i
            )
            buffer.insert_fragment(frag)

        assert len(buffer.fragments) == 5
        assert buffer.get_fragment(FragmentID(1, 3)).symbolic_hash == 1003


class TestReentryDetection:
    """Tests for reentry detection logic."""

    def test_detect_reentry_within_window(self):
        """Test reentry detected within 72-hour window."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        # Dream fragment from 24 hours ago
        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=24),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=9999
        )
        buffer.insert_fragment(dream_frag)

        # Waking fragment with matching hash
        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.8,
            symbolic_hash=9999
        )

        reentries = buffer.detect_reentry([waking], now)

        assert len(reentries) == 1
        assert reentries[0].fragment_id.index == 1
        assert reentries[0].reentry_detected is True

    def test_no_reentry_outside_window(self):
        """Test no reentry when outside 72-hour window."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        # Dream fragment from 100 hours ago (outside window)
        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=100),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=9999
        )
        buffer.insert_fragment(dream_frag)

        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.8,
            symbolic_hash=9999
        )

        reentries = buffer.detect_reentry([waking], now)
        assert len(reentries) == 0

    def test_no_reentry_low_alpha(self):
        """Test no reentry when alpha below threshold."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=12),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=9999
        )
        buffer.insert_fragment(dream_frag)

        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.5,  # Below 0.7 threshold
            symbolic_hash=9999
        )

        reentries = buffer.detect_reentry([waking], now)
        assert len(reentries) == 0

    def test_no_reentry_different_hash(self):
        """Test no reentry when symbolic hash differs."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=12),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=9999
        )
        buffer.insert_fragment(dream_frag)

        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.9,
            symbolic_hash=1111  # Different hash
        )

        reentries = buffer.detect_reentry([waking], now)
        assert len(reentries) == 0

    def test_reentry_only_detected_once(self):
        """Test fragment only marked as reentry once."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=12),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.5,
            symbolic_hash=9999
        )
        buffer.insert_fragment(dream_frag)

        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.9,
            symbolic_hash=9999
        )

        # First detection
        reentries1 = buffer.detect_reentry([waking], now)
        assert len(reentries1) == 1

        # Second call - should not detect again
        reentries2 = buffer.detect_reentry([waking], now + timedelta(hours=1))
        assert len(reentries2) == 0

    def test_multiple_reentries(self):
        """Test detecting multiple reentries."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        for i in range(3):
            dream_frag = create_dream_fragment(
                namespace=1, index=i,
                timestamp=now - timedelta(hours=12 + i),
                dream_ticks=50,
                cycle_index=1,
                state=DreamState.REM,
                alpha=0.6,
                valence=0.5,
                symbolic_hash=1000 + i
            )
            buffer.insert_fragment(dream_frag)

        waking_frags = [
            WakingFragment(FragmentID(2, 0), now, 0.8, 1000),
            WakingFragment(FragmentID(2, 2), now, 0.8, 1002),
        ]

        reentries = buffer.detect_reentry(waking_frags, now)
        assert len(reentries) == 2


class TestFragilityDecay:
    """Tests for fragility score decay."""

    def test_single_cycle_decay(self):
        """Test decay after one sleep cycle."""
        buffer = DreamSurfaceBuffer()

        frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.REM,
            alpha=0.5,
            valence=0.0,
            symbolic_hash=100
        )
        initial_fragility = frag.fragility_score
        buffer.insert_fragment(frag)

        buffer.decay_fragility(1)

        assert frag.fragility_score == pytest.approx(
            initial_fragility - FRAGILITY_DECAY_PER_CYCLE, abs=0.001
        )

    def test_multiple_cycle_decay(self):
        """Test decay after multiple cycles."""
        buffer = DreamSurfaceBuffer()

        frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.LIMINAL,
            alpha=0.3,
            valence=0.0,
            symbolic_hash=100
        )
        initial_fragility = frag.fragility_score
        buffer.insert_fragment(frag)

        buffer.decay_fragility(3)

        expected = initial_fragility - (3 * FRAGILITY_DECAY_PER_CYCLE)
        assert frag.fragility_score == pytest.approx(max(0.0, expected), abs=0.001)

    def test_decay_floors_at_zero(self):
        """Test fragility doesn't go negative."""
        buffer = DreamSurfaceBuffer()

        frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.WAKING,  # Low fragility
            alpha=0.9,
            valence=0.0,
            symbolic_hash=100
        )
        buffer.insert_fragment(frag)

        buffer.decay_fragility(10)  # Many cycles

        assert frag.fragility_score >= 0.0

    def test_get_active_fragments(self):
        """Test filtering active fragments by fragility."""
        buffer = DreamSurfaceBuffer()

        # High fragility fragment (LIMINAL with low alpha = very fragile ~0.81)
        high_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=datetime.now(),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.LIMINAL,
            alpha=0.2,
            valence=0.0,
            symbolic_hash=100
        )
        buffer.insert_fragment(high_frag)

        # Low fragility fragment (WAKING with high alpha = ~0.165)
        low_frag = create_dream_fragment(
            namespace=1, index=2,
            timestamp=datetime.now(),
            dream_ticks=50,
            cycle_index=1,
            state=DreamState.WAKING,
            alpha=0.9,
            valence=0.0,
            symbolic_hash=200
        )
        buffer.insert_fragment(low_frag)

        # Light decay (2 cycles = 0.4 reduction)
        buffer.decay_fragility(2)

        # Get fragments with fragility > 0.1
        active = buffer.get_active_fragments(min_fragility=0.1)

        # High fragility started ~0.81, after 2 cycles = ~0.41, should be above 0.1
        assert any(f.fragment_id.index == 1 for f in active)

    def test_prune_decayed_fragments(self):
        """Test removing fully decayed fragments."""
        buffer = DreamSurfaceBuffer()

        for i in range(5):
            frag = create_dream_fragment(
                namespace=1, index=i,
                timestamp=datetime.now(),
                dream_ticks=50,
                cycle_index=1,
                state=DreamState.WAKING if i < 3 else DreamState.LIMINAL,
                alpha=0.9 if i < 3 else 0.2,
                valence=0.0,
                symbolic_hash=100 + i
            )
            buffer.insert_fragment(frag)

        # Heavy decay
        buffer.decay_fragility(5)

        pruned = buffer.prune_decayed(threshold=0.1)

        # Some fragments should have been pruned
        assert pruned > 0
        assert len(buffer.fragments) < 5


class TestDreamStateIntegration:
    """Integration tests for dream state transitions."""

    def test_full_dream_cycle_workflow(self):
        """Test full workflow: dream → decay → reentry."""
        buffer = DreamSurfaceBuffer()
        now = datetime.now()

        # Night 1: Dream fragments created
        dream_frag = create_dream_fragment(
            namespace=1, index=1,
            timestamp=now - timedelta(hours=8),
            dream_ticks=100,
            cycle_index=2,
            state=DreamState.REM,
            alpha=0.6,
            valence=0.7,
            symbolic_hash=42424
        )
        buffer.insert_fragment(dream_frag)

        # Morning: One sleep cycle decay
        buffer.decay_fragility(1)

        # Day: Waking fragment matches
        waking = WakingFragment(
            fragment_id=FragmentID(2, 1),
            timestamp=now,
            alpha=0.85,
            symbolic_hash=42424
        )

        reentries = buffer.detect_reentry([waking], now)

        assert len(reentries) == 1
        assert reentries[0].reentry_detected is True
        assert reentries[0].fragility_score > 0  # Still has some fragility

    def test_lucid_dream_special_handling(self):
        """Test lucid dreams have moderate fragility."""
        lucid_fragility = compute_fragility_from_state(DreamState.LUCID, 0.7)
        rem_fragility = compute_fragility_from_state(DreamState.REM, 0.7)

        # Lucid dreams should be more stable than regular REM
        assert lucid_fragility < rem_fragility


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
