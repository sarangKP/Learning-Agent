"""
Pydantic v2 schemas — shared contract between caller and microservice.

Changes from v1:
  - ConversationWindow.turns: capped at MAX_TURNS (50) to bound processing
    time and prevent unbounded TF-IDF input.
  - AnalyseRequest.affect_window: validated to contain only known affect
    strings ("frustrated" | "confused" | "sad" | "calm" | "disengaged"),
    capped at WINDOW_SIZE (5). Unknown values raise a 422 at the API boundary
    before reaching state_classifier.
"""

from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

# ── Constants (mirrored from state_classifier to avoid circular import) ───────
_VALID_AFFECTS = {"frustrated", "confused", "sad", "calm", "disengaged"}
MAX_TURNS      = 50    # hard ceiling on conversation_window length
MAX_WINDOW     = 5     # matches state_classifier.WINDOW_SIZE


# ── Sub-models ────────────────────────────────────────────────────────────────

class Turn(BaseModel):
    role: str                          # "user" | "agent"
    text: str
    timestamp: Optional[str] = None


class ConversationWindow(BaseModel):
    # FIX: bound the turns list so a caller can't send thousands of turns and
    # cause unbounded TF-IDF processing. 50 turns covers any realistic
    # sliding window while rejecting obviously malformed requests early.
    turns: List[Turn] = Field(..., max_length=MAX_TURNS)


class CurrentConfig(BaseModel):
    pace: str = "normal"               # slow | normal | fast
    clarity_level: int = 2             # 1 | 2 | 3
    confirmation_frequency: str = "low"  # low | medium | high
    patience_mode: bool = False


# ── Request ───────────────────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    timestamp: Optional[str] = None
    conversation_window: ConversationWindow
    current_config: CurrentConfig = Field(default_factory=CurrentConfig)
    previous_affect: Optional[str] = None
    previous_action_id: Optional[int] = None
    previous_context_id: Optional[int] = None
    # FIX: validate affect_window entries at the API boundary.
    # Unknown affect strings (typos, stale data from an old caller version)
    # now raise a 422 Unprocessable Entity before reaching state_classifier,
    # making the error visible at the right layer instead of being silently
    # swallowed deep in escalation logic.
    # Also capped at MAX_WINDOW to match what state_classifier actually uses.
    affect_window: Optional[List[str]] = Field(default=None, max_length=MAX_WINDOW)
    interaction_count: int = 0
    user_profile_hint: Optional[str] = None

    @field_validator("affect_window", mode="before")
    @classmethod
    def validate_affect_window(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        invalid = [entry for entry in v if entry not in _VALID_AFFECTS]
        if invalid:
            raise ValueError(
                f"affect_window contains unknown affect value(s): {invalid}. "
                f"Valid values are: {sorted(_VALID_AFFECTS)}"
            )
        return v

    @field_validator("previous_affect", mode="before")
    @classmethod
    def validate_previous_affect(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in _VALID_AFFECTS:
            raise ValueError(
                f"previous_affect '{v}' is not a valid affect. "
                f"Valid values are: {sorted(_VALID_AFFECTS)}"
            )
        return v


# ── Response sub-models ───────────────────────────────────────────────────────

class InferredState(BaseModel):
    affect: str
    confidence: float
    context_id: int
    signals_used: List[str]
    escalation_rule_applied: Optional[str] = None


class ConfigDelta(BaseModel):
    apply: bool
    changes: Dict[str, Any]
    reason: str


class BanditContext(BaseModel):
    context_id: int
    action_id: int


class Diagnostics(BaseModel):
    sentiment_score: float
    repetition_score: float
    confusion_score: float = 0.0
    sadness_score: float = 0.0
    ucb_scores: List[float]
    reward_applied: Optional[float]
    total_tries: int


# ── Response ──────────────────────────────────────────────────────────────────

class AnalyseResponse(BaseModel):
    schema_version: str
    session_id: str
    processing_time_ms: int
    inferred_state: InferredState
    config_delta: ConfigDelta
    bandit_context: BanditContext
    diagnostics: Diagnostics