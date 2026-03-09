"""
Pydantic v2 schemas — shared contract between caller and microservice.
"""

from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


# ── Sub-models ────────────────────────────────────────────────────

class Turn(BaseModel):
    role: str                          # "user" | "agent"
    text: str
    timestamp: Optional[str] = None


class ConversationWindow(BaseModel):
    turns: List[Turn]


class CurrentConfig(BaseModel):
    pace: str = "normal"               # slow | normal | fast
    clarity_level: int = 2             # 1 | 2 | 3
    confirmation_frequency: str = "low"  # low | medium | high
    patience_mode: bool = False


# ── Request ───────────────────────────────────────────────────────

class AnalyseRequest(BaseModel):
    schema_version: str = "1.0"
    session_id: str
    timestamp: Optional[str] = None
    conversation_window: ConversationWindow
    current_config: CurrentConfig = Field(default_factory=CurrentConfig)
    previous_affect: Optional[str] = None       # affect from last call
    previous_action_id: Optional[int] = None    # action taken last call
    # FIX: store the context_id from the previous call so reward is attributed
    # to the exact context that was active when the action was taken, rather
    # than re-encoding with the *current* config (which may have changed).
    previous_context_id: Optional[int] = None
    # Rolling window of the last N affect classifications from the caller.
    # Used by state_classifier to enforce escalation rules — e.g. a single
    # "frustrated" signal after 4 calm turns is downgraded to "confused".
    affect_window: Optional[List[str]] = None
    interaction_count: int = 0
    user_profile_hint: Optional[str] = None     # e.g. "elderly"


# ── Response sub-models ───────────────────────────────────────────

class InferredState(BaseModel):
    affect: str
    confidence: float
    context_id: int
    signals_used: List[str]
    # Set when the escalation smoother overrode the raw classifier output.
    # e.g. 'calm*4->frustrated downgraded to confused' — useful for debugging.
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
    sadness_score: float = 0.0          # FIX: was computed but silently dropped from response
    ucb_scores: List[float]
    reward_applied: Optional[float]
    total_tries: int


# ── Response ──────────────────────────────────────────────────────

class AnalyseResponse(BaseModel):
    schema_version: str
    session_id: str
    processing_time_ms: int
    inferred_state: InferredState
    config_delta: ConfigDelta
    bandit_context: BanditContext
    diagnostics: Diagnostics