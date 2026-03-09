"""
Tests — run with:  pytest tests.py -v
"""

import pytest
import numpy as np
from unittest.mock import patch

from nlp_layer import extract_signals
from state_classifier import classify_state, encode_context_id, N_CONTEXTS
from bandit import UCBBandit, N_ACTIONS
from config_applier import apply_action
from schemas import CurrentConfig, Turn


# ── Helpers ───────────────────────────────────────────────────────

def _turns(*texts):
    return [Turn(role="user", text=t) for t in texts]


def _zero_tables():
    return np.zeros((N_CONTEXTS, N_ACTIONS)), np.zeros((N_CONTEXTS, N_ACTIONS))


# ── NLP Layer ─────────────────────────────────────────────────────

class TestNLPLayer:
    def test_sentiment_negative(self):
        turns = _turns("I hate this, nothing works!")
        s, _, _, _ = extract_signals(turns)
        assert s < -0.2

    def test_sentiment_positive(self):
        turns = _turns("This is wonderful, thank you!")
        s, _, _, _ = extract_signals(turns)
        assert s > 0.2

    def test_repetition_high(self):
        msg = "I need help with my password reset"
        turns = _turns(msg, msg)
        _, r, _, _ = extract_signals(turns)
        assert r > 0.9

    def test_repetition_low(self):
        turns = _turns("Hello there", "The weather is nice today")
        _, r, _, _ = extract_signals(turns)
        assert r < 0.4

    def test_single_turn_no_repetition(self):
        turns = _turns("Only one message")
        _, r, _, _ = extract_signals(turns)
        assert r == 0.0

    def test_empty_turns(self):
        s, r, c, sad = extract_signals([])
        assert s == 0.0 and r == 0.0 and c == 0.0 and sad == 0.0

    def test_confusion_keyword_detected(self):
        turns = _turns("I don't understand what you are saying at all")
        _, _, c, _ = extract_signals(turns)
        assert c > 0.5

    def test_sadness_keyword_detected(self):
        turns = _turns("I feel so lonely, my daughter hasn't called in weeks")
        _, _, _, sad = extract_signals(turns)
        assert sad > 0.5


# ── State Classifier ──────────────────────────────────────────────

class TestStateClassifier:
    def test_frustrated(self):
        affect, _, signals, _rule = classify_state(-0.5, 0.8)
        assert affect == "frustrated"
        assert "sentiment" in signals and "repetition" in signals

    def test_confused_repetition_only(self):
        affect, _, signals, _rule = classify_state(0.1, 0.8)
        assert affect == "confused"
        assert "repetition" in signals

    def test_negative_sentiment_only_maps_to_sad(self):
        # Negative sentiment with no confusion keywords and no repetition
        # correctly maps to sad, not confused. The sad path fires when:
        #   negative=True, confused_kw=False, repetitive=False
        # This test was previously named test_confused_sentiment_only but that
        # expectation predates the sad state being added to the classifier.
        affect, _, signals, _rule = classify_state(-0.5, 0.3)
        assert affect == "sad"
        assert "sentiment" in signals

    def test_calm(self):
        affect, _, signals, _rule = classify_state(0.3, 0.2)
        assert affect == "calm"
        assert signals == []

    def test_sad_via_keywords(self):
        affect, _, signals, _rule = classify_state(0.0, 0.0, sadness=0.8)
        assert affect == "sad"
        assert "sadness_keywords" in signals

    def test_sad_not_confused_when_only_sadness(self):
        # Lonely statement must not be classified as confused
        affect, _, _, _rule = classify_state(-0.3, 0.0, confusion=0.0, sadness=0.9)
        assert affect == "sad"

    def test_disengaged_short_neutral_message(self):
        affect, _, signals, _rule = classify_state(0.0, 0.0, last_user_text="ok")
        assert affect == "disengaged"
        assert "short_message" in signals

    def test_disengaged_not_triggered_without_text(self):
        # Backward-compatible: no last_user_text → never disengaged
        affect, _, _, _rule = classify_state(0.0, 0.0)
        assert affect == "calm"   # not disengaged — no text provided

    # FIX: updated range to 0-44 (was 0-35 — left over from 36-context design)
    def test_context_id_range(self):
        for affect in ["frustrated", "confused", "sad", "calm", "disengaged"]:
            for clarity in [1, 2, 3]:
                for pace in ["slow", "normal", "fast"]:
                    ctx = encode_context_id(affect, clarity, pace)
                    assert 0 <= ctx <= 44, f"context_id {ctx} out of range for {affect}/{clarity}/{pace}"

    # FIX: updated unique count to 45 (was 36 — left over from 36-context design)
    def test_context_id_unique(self):
        ids = set()
        for a in ["frustrated", "confused", "sad", "calm", "disengaged"]:
            for c in [1, 2, 3]:
                for p in ["slow", "normal", "fast"]:
                    ids.add(encode_context_id(a, c, p))
        assert len(ids) == 45

    def test_sad_context_id_distinct_from_others(self):
        # Ensures the sad affect produces IDs in the 18-26 range (affect_idx=2 × 9)
        for clarity in [1, 2, 3]:
            for pace in ["slow", "normal", "fast"]:
                ctx = encode_context_id("sad", clarity, pace)
                assert 18 <= ctx <= 26, f"sad context_id {ctx} not in expected range 18-26"

    # ── Escalation smoother ───────────────────────────────────────

    def test_escalation_R3_all_calm_downgrades_frustrated(self):
        # R3: all-calm history → frustrated must become confused
        window = ["calm", "calm", "calm", "calm"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "confused"
        assert rule == "R3_all_calm_history"

    def test_escalation_R1_streak_1_downgrades_frustrated(self):
        # R1: only 1 non-calm entry at end → streak too short, downgrade
        # This is the turn-8 boundary bug scenario: 3 calm + 1 confused + frustrated_raw
        window = ["calm", "calm", "calm", "confused"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "confused"
        assert rule == "R1_insufficient_streak"

    def test_escalation_R1_streak_0_last_calm_downgrades(self):
        # R1: last entry is calm → streak = 0 → downgrade (R2 is now redundant/absorbed)
        window = ["confused", "confused", "calm"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "confused"
        assert rule == "R1_insufficient_streak"

    def test_escalation_R1_interrupted_streak_downgrades(self):
        # R1: non-calm then calm resets streak to 0 at the end
        # ["calm","confused","calm","calm","confused"] → trailing streak = 1
        window = ["calm", "confused", "calm", "calm", "confused"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "confused"
        assert rule == "R1_insufficient_streak"

    def test_escalation_no_rule_streak_2_allows_frustrated(self):
        # Streak of 2 consecutive non-calm turns → frustrated stands
        # This is the fix for the turn-8 boundary: 3 calm + 2 confused = OK
        window = ["calm", "calm", "calm", "confused", "confused"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "frustrated"
        assert rule is None

    def test_escalation_no_rule_streak_3_allows_frustrated(self):
        # Streak of 3 → definitely frustrated
        window = ["calm", "confused", "confused", "frustrated"]
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=window)
        assert affect == "frustrated"
        assert rule is None

    def test_escalation_full_signals_bypasses_R1_streak_1(self):
        # all_signals_fired bypasses R1 even with streak of 1
        # (negative + repetition + confusion keywords all firing = strong evidence)
        window = ["calm", "calm", "calm", "confused"]
        affect, _, _, rule = classify_state(-0.5, 0.8, confusion=0.9, affect_window=window)
        assert affect == "frustrated"
        assert rule is None

    def test_escalation_full_signals_still_blocked_by_R3(self):
        # R3 is never bypassed — all-calm history blocks frustrated even with
        # all three signals firing
        window = ["calm", "calm", "calm", "calm"]
        affect, _, _, rule = classify_state(-0.5, 0.8, confusion=0.9, affect_window=window)
        assert affect == "confused"
        assert rule == "R3_all_calm_history"

    def test_escalation_no_window_no_rule(self):
        # Without a window, escalation smoother is disabled — raw affect stands
        affect, _, _, rule = classify_state(-0.5, 0.8, affect_window=None)
        assert affect == "frustrated"
        assert rule is None


# ── Bandit ────────────────────────────────────────────────────────

class TestBandit:
    def test_cold_start_frustrated_uses_rule_default(self):
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        ctx = encode_context_id("frustrated", 2, "normal")
        action, _ = b.select_action(ctx)
        assert action == 5  # DECREASE_CLARITY_AND_PACE

    def test_cold_start_calm_uses_rule_default(self):
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        ctx = encode_context_id("calm", 2, "normal")
        action, _ = b.select_action(ctx)
        assert action == 0  # DO_NOTHING

    def test_calm_always_do_nothing_regardless_of_q(self):
        # Even if Q table is manipulated to prefer action 5 for a calm context,
        # the guard must still return DO_NOTHING.
        N, Q = _zero_tables()
        ctx = encode_context_id("calm", 2, "normal")
        Q[ctx][5] = 99.0
        N[ctx][5] = 100.0
        b = UCBBandit(N, Q)
        action, _ = b.select_action(ctx)
        assert action == 0

    def test_sad_cold_start_uses_enable_patience(self):
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        ctx = encode_context_id("sad", 2, "normal")
        action, _ = b.select_action(ctx)
        assert action == 4  # ENABLE_PATIENCE

    def test_sad_only_allows_permitted_actions(self):
        # Even with high Q values for forbidden actions, sad must stay within
        # SAD_ALLOWED_ACTIONS = {0, 4}
        N, Q = _zero_tables()
        ctx = encode_context_id("sad", 2, "normal")
        # Give forbidden action 5 a very high Q value
        Q[ctx][5] = 99.0
        N[ctx][5] = 100.0
        N[ctx][0] = 1.0
        N[ctx][4] = 1.0
        b = UCBBandit(N, Q)
        action, _ = b.select_action(ctx)
        assert action in (0, 4), f"sad state selected forbidden action {action}"

    def test_update_increases_N(self):
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        b.update(0, 3, 1.0)
        assert b.N[0][3] == 1.0

    def test_update_q_incremental_mean(self):
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        b.update(0, 0, 1.0)
        b.update(0, 0, 0.0)
        assert abs(b.Q[0][0] - 0.5) < 1e-9

    def test_bandit_learns_good_action(self):
        """After many positive rewards for action 3, it should be preferred."""
        N, Q = _zero_tables()
        b = UCBBandit(N, Q)
        ctx = 0  # frustrated context (affect_idx=0)
        # Give action 3 a strong positive signal
        for _ in range(20):
            b.update(ctx, 3, 1.0)
        # Give all other actions enough negative updates to suppress their
        # UCB exploration bonuses. With only 1 update each the bonus term
        # sqrt(2*log(total)/1) is large enough to dominate; 10 updates each
        # brings it below action 3's learned Q value.
        for a in range(N_ACTIONS):
            if a != 3:
                for _ in range(10):
                    b.update(ctx, a, -0.5)
        action, _ = b.select_action(ctx)
        assert action == 3


# ── Config Applier ────────────────────────────────────────────────

class TestConfigApplier:
    def _cfg(self, **kw):
        return CurrentConfig(**{"pace": "normal", "clarity_level": 2,
                                "confirmation_frequency": "low",
                                "patience_mode": False, **kw})

    def test_do_nothing(self):
        _, changes, _ = apply_action(0, self._cfg(), "calm")
        assert changes == {}

    def test_decrease_clarity(self):
        _, changes, _ = apply_action(1, self._cfg(clarity_level=2), "confused")
        assert changes.get("clarity_level") == 1

    def test_decrease_pace(self):
        _, changes, _ = apply_action(2, self._cfg(pace="normal"), "frustrated")
        assert changes.get("pace") == "slow"

    def test_increase_confirmation(self):
        _, changes, _ = apply_action(3, self._cfg(confirmation_frequency="low"), "confused")
        assert changes.get("confirmation_frequency") == "medium"

    def test_enable_patience(self):
        _, changes, _ = apply_action(4, self._cfg(patience_mode=False), "confused")
        assert changes.get("patience_mode") is True

    def test_already_at_min_clarity_no_change(self):
        _, changes, _ = apply_action(1, self._cfg(clarity_level=1), "confused")
        assert "clarity_level" not in changes

    def test_already_slow_no_pace_change(self):
        _, changes, _ = apply_action(2, self._cfg(pace="slow"), "frustrated")
        assert "pace" not in changes

    def test_decrease_clarity_and_pace(self):
        _, changes, _ = apply_action(5, self._cfg(), "frustrated")
        assert changes.get("clarity_level") == 1
        assert changes.get("pace") == "slow"

    def test_clarity_and_confirmation(self):
        _, changes, _ = apply_action(6, self._cfg(), "confused")
        assert changes.get("clarity_level") == 1
        assert changes.get("confirmation_frequency") == "medium"