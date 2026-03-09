"""
Layer 2 — State Classifier

Affect states: frustrated | confused | sad | calm | disengaged

Four input signals:
  sentiment_score  : VADER compound       (-1 to +1)
  repetition_score : TF-IDF cosine        (0 to 1)
  confusion_score  : confusion keywords   (0 to 1)
  sadness_score    : sadness keywords     (0 to 1)

Decision priority (checked top to bottom, first match wins):
  frustrated   : strong negative + repetition  → needs clarity + pace + patience
  confused     : confusion keywords or repetition → needs clarity + confirmation
  sad          : sadness keywords (no confusion) → needs patience + empathy only
  disengaged   : very short / minimal input with neutral/no signal → needs re-engagement
  calm         : none of the above

── Escalation Smoother ───────────────────────────────────────────────────────
After the raw affect is determined, apply_escalation_rules() inspects the
rolling affect_window (last N classifications sent by the caller) and may
downgrade the raw affect if the evidence is insufficient to justify a sudden
jump.

The core principle: affect should escalate gradually.
  calm → confused → frustrated    ✓  (one step at a time)
  calm → frustrated               ✗  (skip-level jump — needs very strong evidence)

Rules (checked in order, first match wins):
  R3  all_calm_history
      If every entry in the window is calm, frustrated is downgraded to
      confused regardless of signal strength. The user cannot jump from
      entirely calm history to genuine frustration in one message.

  R1  insufficient_streak
      Frustrated is only allowed to stand if the window ends with at least
      MIN_NONCALM_STREAK (2) consecutive non-calm entries. A single non-calm
      turn before frustrated is always suppressed; two or more consecutive
      non-calm turns means the escalation is genuine and earned.
      This replaces the old ratio-based check and makes R2 redundant:
      a last_affect of calm means streak=0 < 2, so R1 already catches it.

These rules are additive dampeners — they never upgrade an affect, only
downgrade it. The full multi-signal frustrated path (negative + repetition +
strong keywords) bypasses R1 so genuine sustained frustration is still caught.

N_CONTEXTS is exported here as the single source of truth.
bandit.py and storage.py import from here to avoid definition drift.
"""

from __future__ import annotations
from typing import Tuple, List, Optional

# ── Thresholds ────────────────────────────────────────────────────────────────
SENTIMENT_THRESHOLD   = -0.2
REPETITION_THRESHOLD  = 0.55
CONFUSION_THRESHOLD   = 0.5
SADNESS_THRESHOLD     = 0.5
DISENGAGED_WORD_COUNT = 3    # messages with ≤ this many words trigger disengaged check

# Escalation smoother settings
WINDOW_SIZE        = 5   # how many past affects the caller should send
MIN_NONCALM_STREAK = 2   # consecutive non-calm turns required before frustrated is allowed

# ── Encoding maps ─────────────────────────────────────────────────────────────
AFFECT_MAP  = {"frustrated": 0, "confused": 1, "sad": 2, "calm": 3, "disengaged": 4}
CLARITY_MAP = {1: 0, 2: 1, 3: 2}
PACE_MAP    = {"slow": 0, "normal": 1, "fast": 2}

# Single source of truth — imported by bandit.py and storage.py
N_CONTEXTS = 45   # 5 affects × 9 (3 clarity × 3 pace)
N_ACTIONS  = 7


def classify_state(
    sentiment: float,
    repetition: float,
    confusion: float = 0.0,
    sadness: float   = 0.0,
    last_user_text: str = "",      # optional: enables disengaged detection
    affect_window: Optional[List[str]] = None,  # rolling history from caller
) -> Tuple[str, float, List[str], Optional[str]]:
    """
    Returns (affect, confidence, signals_used, escalation_rule_applied).

    escalation_rule_applied is None when no rule fired, or a short string
    describing which rule downgraded the raw affect — useful for the
    diagnostics panel and tests.

    Extra args default to 0.0 / "" / None for backward compatibility.
    """
    negative    = sentiment  < SENTIMENT_THRESHOLD
    repetitive  = repetition > REPETITION_THRESHOLD
    confused_kw = confusion  > CONFUSION_THRESHOLD
    sad_kw      = sadness    > SADNESS_THRESHOLD

    signals_used: List[str] = []
    if negative:    signals_used.append("sentiment")
    if repetitive:  signals_used.append("repetition")
    if confused_kw: signals_used.append("confusion_keywords")
    if sad_kw:      signals_used.append("sadness_keywords")

    # ── 1. frustrated ────────────────────────────────────────────
    # Triggered by EITHER:
    #   a) strong negative signal + repetition (classic frustration arc)
    #   b) very high confusion keyword score alone (e.g. "making me very upset",
    #      "you never remember", "already told you" compound to >0.8)
    strong_neg   = negative or confusion > 0.65
    high_conf_kw = confusion > 0.8
    if (strong_neg and repetitive) or (high_conf_kw and (negative or repetitive or confusion > 0.9)):
        affect = "frustrated"
        conf = _blend(
            abs(sentiment - SENTIMENT_THRESHOLD) / 1.0 if negative else confusion,
            max(
                (repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD),
                (confusion  - 0.65) / 0.35,
            ),
        )

    # ── 2. confused ──────────────────────────────────────────────
    elif confused_kw or (repetitive and not sad_kw):
        affect = "confused"
        scores = []
        if confused_kw:
            scores.append((confusion - CONFUSION_THRESHOLD) / (1.0 - CONFUSION_THRESHOLD))
        if repetitive:
            scores.append((repetition - REPETITION_THRESHOLD) / (1.0 - REPETITION_THRESHOLD))
        if negative and not sad_kw:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 3. sad ───────────────────────────────────────────────────
    elif sad_kw or (negative and not confused_kw and not repetitive):
        affect = "sad"
        scores = []
        if sad_kw:
            scores.append((sadness - SADNESS_THRESHOLD) / (1.0 - SADNESS_THRESHOLD))
        if negative:
            scores.append(abs(sentiment - SENTIMENT_THRESHOLD) / 1.0)
        conf = _clamp(max(scores)) if scores else 0.5

    # ── 4. disengaged ────────────────────────────────────────────
    elif (
        last_user_text
        and len(last_user_text.split()) <= DISENGAGED_WORD_COUNT
        and not negative
        and not confused_kw
        and not sad_kw
        and not repetitive
    ):
        affect = "disengaged"
        signals_used.append("short_message")
        conf = 0.6

    # ── 5. calm ──────────────────────────────────────────────────
    else:
        affect = "calm"
        conf = _clamp(
            0.5
            + (sentiment  - SENTIMENT_THRESHOLD)  / 2.0
            + (REPETITION_THRESHOLD - repetition) / 2.0
            + (CONFUSION_THRESHOLD  - confusion)  / 2.0
            + (SADNESS_THRESHOLD    - sadness)    / 2.0
        )
        signals_used = []

    # ── Escalation smoother ──────────────────────────────────────
    # Apply after raw classification. Only fires when affect_window is
    # provided and the raw affect is "frustrated" (the highest-impact state).
    affect, conf, escalation_rule = apply_escalation_rules(
        raw_affect=affect,
        raw_conf=conf,
        affect_window=affect_window,
        all_signals_fired=(negative and repetitive and confused_kw),
    )

    if escalation_rule:
        signals_used.append(f"escalation:{escalation_rule}")

    return affect, round(conf, 4), signals_used, escalation_rule


def apply_escalation_rules(
    raw_affect: str,
    raw_conf: float,
    affect_window: Optional[List[str]],
    all_signals_fired: bool,
) -> Tuple[str, float, Optional[str]]:
    """
    Inspect the rolling affect window and downgrade raw_affect if the
    evidence doesn't justify a sudden jump.

    Returns (final_affect, final_conf, rule_name_or_None).

    Only acts on "frustrated" — the highest-impact affect that triggers the
    most aggressive config changes. Calm/confused/sad/disengaged are left
    unchanged because their downstream actions are gentler.

    all_signals_fired: True when sentiment + repetition + confusion keywords
    all crossed their thresholds simultaneously. This bypasses R1 so genuine
    multi-signal frustration can still escalate even from a short streak.
    """
    # Nothing to check if no window or affect is not frustrated
    if not affect_window or raw_affect != "frustrated":
        return raw_affect, raw_conf, None

    window = affect_window[-WINDOW_SIZE:]   # use at most last WINDOW_SIZE entries

    non_calm_count = sum(1 for a in window if a != "calm")

    # ── R3: history is entirely calm → frustrated not allowed ────
    # Strictest rule — checked first.
    # Even all_signals_fired doesn't override this: if every prior reading
    # was calm, one message is never reliable evidence of frustration.
    if non_calm_count == 0:
        return "confused", raw_conf * 0.8, "R3_all_calm_history"

    # ── R1: insufficient trailing streak ─────────────────────────
    # Count how many consecutive non-calm entries END the window.
    # ["calm","calm","calm","confused","confused"] → streak = 2  ✓ frustrated ok
    # ["calm","calm","calm","calm","confused"]     → streak = 1  ✗ downgrade
    # ["calm","confused","calm","calm","confused"] → streak = 1  ✗ downgrade
    #   (the calm in position -2 resets the streak)
    #
    # all_signals_fired bypasses this: negative + repetition + confusion
    # keywords all firing together is strong enough to allow frustrated
    # even with a streak of 1.
    streak = _trailing_noncalm_streak(window)
    if streak < MIN_NONCALM_STREAK and not all_signals_fired:
        return "confused", raw_conf * 0.75, "R1_insufficient_streak"

    # All rules passed — frustrated stands
    return raw_affect, raw_conf, None


def _trailing_noncalm_streak(window: List[str]) -> int:
    """
    Count how many consecutive non-calm entries END the window.
    e.g. ["calm","confused","confused"] → 2
         ["calm","confused","calm"]     → 0  (calm resets the streak)
    """
    streak = 0
    for affect in reversed(window):
        if affect == "calm":
            break
        streak += 1
    return streak


def encode_context_id(affect: str, clarity_level: int, pace: str) -> int:
    """Encodes a (affect, clarity, pace) triple into a 0-44 integer."""
    a = AFFECT_MAP.get(affect, 3)   # default to calm
    c = CLARITY_MAP.get(clarity_level, 1)
    p = PACE_MAP.get(pace, 1)
    return a * 9 + c * 3 + p


# ── Helpers ───────────────────────────────────────────────────────────────────

def _blend(a: float, b: float) -> float:
    return _clamp((a + b) / 2.0)

def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))