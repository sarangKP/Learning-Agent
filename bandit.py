"""
Layer 3 — Contextual Bandit with UCB1

Tables:
  N[ctx][action]  — visit counts   (45 × 7)
  Q[ctx][action]  — average reward (45 × 7)

Context ID encoding (from state_classifier):
  affect_idx (0-4) × 9  +  clarity_idx (0-2) × 3  +  pace_idx (0-2)
  → 0..44  (45 unique contexts)

  Affect index mapping:
    0 = frustrated
    1 = confused
    2 = sad
    3 = calm
    4 = disengaged

Cold-start: if N[ctx] is all-zero, action selection falls back to
rule-based defaults (see RULE_BASED_DEFAULTS).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List

# FIX: import the single source of truth instead of re-defining here.
# N_CONTEXTS was previously defined in both bandit.py and state_classifier.py;
# keeping them in sync manually is error-prone.
from state_classifier import N_CONTEXTS, AFFECT_MAP

N_ACTIONS = 7

# Rule-based fallbacks indexed by affect_idx
# 0=frustrated, 1=confused, 2=sad, 3=calm, 4=disengaged
RULE_BASED_DEFAULTS = {
    0: 5,   # frustrated → DECREASE_CLARITY_AND_PACE
    1: 6,   # confused   → CLARITY_AND_CONFIRMATION
    2: 4,   # sad        → ENABLE_PATIENCE
    3: 0,   # calm       → DO_NOTHING
    4: 4,   # disengaged → ENABLE_PATIENCE
}

# FIX: actions that are permitted for the sad state.
# Sad users need empathy (patience_mode), not clarity/pace changes.
# Allowing the bandit to learn arbitrary actions for sad contexts risks
# it discovering spurious correlations (e.g. a user who calms down for
# unrelated reasons while clarity was also changed).
SAD_ALLOWED_ACTIONS = {0, 4}   # DO_NOTHING, ENABLE_PATIENCE

# affect_idx for calm and sad (used in guards below)
_CALM_IDX = AFFECT_MAP["calm"]   # 3
_SAD_IDX  = AFFECT_MAP["sad"]    # 2


class UCBBandit:
    def __init__(self, N: np.ndarray, Q: np.ndarray):
        assert N.shape == (N_CONTEXTS, N_ACTIONS)
        assert Q.shape == (N_CONTEXTS, N_ACTIONS)
        self.N = N.copy().astype(float)
        self.Q = Q.copy().astype(float)

    # ── Update ────────────────────────────────────────────────────

    def update(self, context_id: int, action_id: int, reward: float) -> None:
        """Incremental mean update."""
        self.N[context_id][action_id] += 1
        n = self.N[context_id][action_id]
        self.Q[context_id][action_id] += (1.0 / n) * (
            reward - self.Q[context_id][action_id]
        )

    # ── Select ────────────────────────────────────────────────────

    def select_action(self, context_id: int) -> Tuple[int, List[float]]:
        """
        Returns (best_action_id, ucb_scores_for_all_actions).

        Decision logic:
          1. calm context      → always DO_NOTHING (never change config on a happy user)
          2. sad context       → restrict to SAD_ALLOWED_ACTIONS (empathy only, no
                                 clarity/pace changes)
          3. cold start        → rule-based default for the affect
          4. enough data       → UCB picks the best learned action (sad: within
                                 SAD_ALLOWED_ACTIONS only)
        """
        affect_idx = context_id // 9
        n_ctx      = self.N[context_id]
        total      = int(self.N.sum())

        # ── Rule 1: calm → never touch the config ─────────────────
        if affect_idx == _CALM_IDX:
            display = [round(float(self.Q[context_id][a]), 4) for a in range(N_ACTIONS)]
            return 0, display   # DO_NOTHING

        # ── Rule 2: sad → empathy-only actions ────────────────────
        if affect_idx == _SAD_IDX:
            # Cold start for sad context
            if n_ctx.sum() == 0:
                action_id = RULE_BASED_DEFAULTS[_SAD_IDX]  # ENABLE_PATIENCE
                display   = [0.0] * N_ACTIONS
                display[action_id] = 1.0
                return action_id, display

            # UCB restricted to SAD_ALLOWED_ACTIONS
            ucb_scores = []
            for a in range(N_ACTIONS):
                if a not in SAD_ALLOWED_ACTIONS:
                    ucb_scores.append(-float("inf"))   # never selected
                elif n_ctx[a] == 0:
                    ucb_scores.append(float("inf"))    # force exploration
                else:
                    exploration = math.sqrt(2.0 * math.log(total + 1) / n_ctx[a])
                    ucb_scores.append(self.Q[context_id][a] + exploration)

            action_id = int(np.argmax(ucb_scores))
            display   = [
                s if s not in (float("inf"), -float("inf")) else (99.0 if s > 0 else -99.0)
                for s in ucb_scores
            ]
            return action_id, display

        # ── Rule 3: cold start for this context ───────────────────
        if n_ctx.sum() == 0:
            action_id = RULE_BASED_DEFAULTS.get(affect_idx, 0)
            display   = [0.0] * N_ACTIONS
            display[action_id] = 1.0
            return action_id, display

        # ── Rule 4: UCB with exploration ──────────────────────────
        ucb_scores = []
        for a in range(N_ACTIONS):
            if n_ctx[a] == 0:
                ucb_scores.append(float("inf"))
            else:
                exploration = math.sqrt(2.0 * math.log(total + 1) / n_ctx[a])
                ucb_scores.append(self.Q[context_id][a] + exploration)

        action_id = int(np.argmax(ucb_scores))
        display   = [s if s != float("inf") else 99.0 for s in ucb_scores]
        return action_id, display