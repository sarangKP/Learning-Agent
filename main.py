"""
Learning Agent Microservice
FastAPI app — stateless, loads bandit tables per request.
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalyseRequest, AnalyseResponse, InferredState, ConfigDelta, BanditContext, Diagnostics
from nlp_layer import extract_signals
from state_classifier import classify_state, encode_context_id
from bandit import UCBBandit
from config_applier import apply_action
from storage import load_tables, save_tables

app = FastAPI(title="Learning Agent Microservice", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(req: AnalyseRequest):
    t0 = time.time()

    # ── Layer 1: NLP signals ──────────────────────────────────────
    turns = req.conversation_window.turns
    sentiment_score, repetition_score, confusion_score, sadness_score = extract_signals(turns)

    # ── Layer 2: State classification ────────────────────────────
    # Pass affect_window so the escalation smoother can prevent
    # single-turn skip-level jumps (e.g. calm → frustrated in one message).
    last_user_text = next(
        (t.text for t in reversed(turns) if t.role == "user"), ""
    )
    affect, confidence, signals_used, escalation_rule = classify_state(
        sentiment_score,
        repetition_score,
        confusion_score,
        sadness_score,
        last_user_text=last_user_text,
        affect_window=req.affect_window,
    )

    cfg = req.current_config
    context_id = encode_context_id(affect, cfg.clarity_level, cfg.pace)

    # ── Layer 3: UCB Bandit ───────────────────────────────────────
    N, Q = load_tables()
    bandit = UCBBandit(N, Q)

    # Compute reward from previous interaction and update tables.
    # FIX: use req.previous_context_id (the context active when the action was
    # taken) instead of re-encoding with the current config, which may have
    # changed since then and would attribute the reward to the wrong context.
    reward_applied = None
    if (
        req.previous_affect is not None
        and req.previous_action_id is not None
        and req.previous_context_id is not None
    ):
        reward = _compute_reward(req.previous_affect, affect)
        bandit.update(req.previous_context_id, req.previous_action_id, reward)
        reward_applied = reward

    # Fallback for callers that don't yet send previous_context_id
    # (keeps backward-compatibility with older ollama_agent versions)
    elif req.previous_affect is not None and req.previous_action_id is not None:
        prev_ctx = encode_context_id(req.previous_affect, cfg.clarity_level, cfg.pace)
        reward = _compute_reward(req.previous_affect, affect)
        bandit.update(prev_ctx, req.previous_action_id, reward)
        reward_applied = reward

    # Pick best action
    action_id, ucb_scores = bandit.select_action(context_id)
    save_tables(bandit.N, bandit.Q)

    # ── Apply action to config ────────────────────────────────────
    new_cfg, changes, reason = apply_action(action_id, cfg, affect)
    apply_flag = len(changes) > 0

    elapsed_ms = round((time.time() - t0) * 1000)

    return AnalyseResponse(
        schema_version="1.0",
        session_id=req.session_id,
        processing_time_ms=elapsed_ms,
        inferred_state=InferredState(
            affect=affect,
            confidence=confidence,
            context_id=context_id,
            signals_used=signals_used,
            escalation_rule_applied=escalation_rule,
        ),
        config_delta=ConfigDelta(
            apply=apply_flag,
            changes=changes,
            reason=reason,
        ),
        bandit_context=BanditContext(
            context_id=context_id,
            action_id=action_id,
        ),
        diagnostics=Diagnostics(
            sentiment_score=round(sentiment_score, 4),
            repetition_score=round(repetition_score, 4),
            confusion_score=round(confusion_score, 4),
            sadness_score=round(sadness_score, 4),   # FIX: now included in response
            ucb_scores=[round(s, 4) for s in ucb_scores],
            reward_applied=reward_applied,
            total_tries=int(bandit.N.sum()),
        ),
    )


def _compute_reward(prev_affect: str, curr_affect: str) -> float:
    table = {
        ("frustrated", "calm"):       1.0,
        ("frustrated", "sad"):        0.3,   # de-escalated but still hurting
        ("frustrated", "confused"):   0.5,
        ("frustrated", "frustrated"): -0.5,
        ("confused",   "calm"):       1.0,
        ("confused",   "sad"):        0.2,   # moved away from confusion
        ("confused",   "confused"):  -0.3,
        ("confused",   "frustrated"): -1.0,
        ("sad",        "calm"):       1.0,   # empathy worked
        ("sad",        "sad"):       -0.2,   # still sad — try harder
        ("sad",        "confused"):  -0.5,   # made things worse
        ("sad",        "frustrated"): -1.0,
        ("calm",       "calm"):       0.0,
        ("calm",       "sad"):       -0.3,   # missed emotional cue
        ("calm",       "confused"):  -0.5,
    }
    if curr_affect == "disengaged":
        return -1.0
    return table.get((prev_affect, curr_affect), 0.0)