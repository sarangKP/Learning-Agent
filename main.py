"""
Learning Agent Microservice — LinUCB Edition
FastAPI app — stateless, uses tables_locked() for atomic matrix updates.
"""

import logging
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

from schemas import AnalyseRequest, AnalyseResponse, InferredState, ConfigDelta, BanditContext, Diagnostics
from nlp_layer import extract_signals
from state_classifier import classify_state, encode_context_id, encode_context_features
from bandit import LinUCBBandit
from config_applier import apply_action
from storage import tables_locked

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="ELARA Learning Service (LinUCB)", version="1.1.0")

# CORS Configuration
_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    t0 = time.time()

    # ── Layer 1: NLP Signal Extraction ───────────────────────────
    turns = req.conversation_window.turns
    sentiment, repetition, confusion, sadness = extract_signals(turns)

    # ── Layer 2: State Classification ────────────────────────────
    last_user_text = next((t.text for t in reversed(turns) if t.role == "user"), "")
    affect, confidence, signals_used, escalation_rule = classify_state(
        sentiment, repetition, confusion, sadness,
        last_user_text=last_user_text, affect_window=req.affect_window,
    )

    # Feature encoding for the current state (7D vector)
    cfg = req.current_config
    curr_features = encode_context_features(affect, cfg.clarity_level, cfg.pace)
    # context_id is kept for legacy response schema support
    context_id = encode_context_id(affect, cfg.clarity_level, cfg.pace)

# ── Layer 3: LinUCB Bandit (Atomic Update & Select) ──────────
    reward_applied = None
    action_id: int
    ucb_scores: list

    with tables_locked() as (A, b):
        bandit = LinUCBBandit(A, b, alpha=0.8, gamma=0.95)

        # FIX: Explicit checks allow Pylance to verify 'previous_affect' is not None
        if (req.previous_affect is not None and 
            req.previous_action_id is not None and 
            req.previous_config is not None):
            
            # Now Pylance knows these are not None
            prev_features = encode_context_features(
                req.previous_affect, 
                req.previous_config.clarity_level, 
                req.previous_config.pace
            )
            reward = _compute_reward(req.previous_affect, affect)
            bandit.update(prev_features, req.previous_action_id, reward)
            reward_applied = reward

        # Decision Step
        action_id, ucb_scores = bandit.select_action(curr_features)
        A[:] = bandit.A
        b[:] = bandit.b

    # ── Layer 4: Config Adaptation ───────────────────────────────
    new_cfg, changes, reason = apply_action(action_id, cfg, affect)
    elapsed_ms = round((time.time() - t0) * 1000)

    return AnalyseResponse(
        schema_version="1.1",
        session_id=req.session_id,
        processing_time_ms=elapsed_ms,
        inferred_state=InferredState(
            affect=affect, confidence=confidence, context_id=context_id,
            signals_used=signals_used, escalation_rule_applied=escalation_rule,
        ),
        config_delta=ConfigDelta(apply=len(changes) > 0, changes=changes, reason=reason),
        bandit_context=BanditContext(context_id=context_id, action_id=action_id),
        diagnostics=Diagnostics(
            sentiment_score=round(sentiment, 4),
            repetition_score=round(repetition, 4),
            confusion_score=round(confusion, 4),
            sadness_score=round(sadness, 4),
            ucb_scores=[round(s, 4) for s in ucb_scores],
            reward_applied=reward_applied,
            total_tries=0  # Placeholder: cumulative counts are deprecated in LinUCB
        ),
    )

# ── Reward Logic ─────────────────────────────────────────────────────────────

_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("frustrated", "calm"):       +1.0,
    ("frustrated", "confused"):   +0.3,
    ("frustrated", "frustrated"): -0.5,
    ("confused",   "calm"):       +1.0,
    ("confused",   "confused"):   -0.3,
    ("confused",   "frustrated"): -1.0,
    ("sad",        "calm"):       +1.0,
    ("sad",        "sad"):        -0.2,
    ("calm",       "calm"):        0.0,
    ("calm",       "confused"):   -0.5,
}

def _compute_reward(prev: str, curr: str) -> float:
    """Calculates reward based on affect transition."""
    if curr == "disengaged":
        return -1.0
    return _REWARD_TABLE.get((prev, curr), 0.0)