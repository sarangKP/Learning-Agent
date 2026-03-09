"""
Learning Agent Microservice
FastAPI app — stateless, uses tables_locked() for atomic bandit updates.
"""

import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import AnalyseRequest, AnalyseResponse, InferredState, ConfigDelta, BanditContext, Diagnostics
from nlp_layer import extract_signals
from state_classifier import classify_state, encode_context_id
from bandit import UCBBandit
from config_applier import apply_action
from storage import tables_locked

log = logging.getLogger(__name__)

app = FastAPI(title="Learning Agent Microservice", version="1.0.0")

# FIX: restrict CORS to localhost origins only.
# The original allow_origins=["*"] is fine for a closed local demo but would
# allow any web page to call this service if it were ever port-forwarded or
# deployed. Lock it down to the only callers that should exist in this system.
# Add any additional internal origins here if the topology changes.
_ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",   # reserve for any future frontend
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

    # ── Layer 1: NLP signals ──────────────────────────────────────
    turns = req.conversation_window.turns
    sentiment_score, repetition_score, confusion_score, sadness_score = extract_signals(turns)

    # ── Layer 2: State classification ────────────────────────────
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

    # ── Layer 3: UCB Bandit (atomic load → update → save) ─────────
    # FIX: replaced the old load_tables() / save_tables() pair with
    # tables_locked(), which holds both a threading.Lock and an fcntl
    # advisory lock for the entire read-modify-write cycle.  This prevents
    # two concurrent requests from loading the same stale tables and having
    # the second save silently overwrite the first reward update.
    reward_applied = None
    action_id: int
    ucb_scores: list

    with tables_locked() as (N, Q):
        bandit = UCBBandit(N, Q)

        if (
            req.previous_affect is not None
            and req.previous_action_id is not None
            and req.previous_context_id is not None
        ):
            reward = _compute_reward(req.previous_affect, affect)
            bandit.update(req.previous_context_id, req.previous_action_id, reward)
            reward_applied = reward

        elif req.previous_affect is not None and req.previous_action_id is not None:
            # FIX: mark this path explicitly as a known-incorrect fallback.
            # This branch exists only to keep backward-compatibility with
            # ollama_agent versions that predate the previous_context_id field.
            # It re-encodes the previous context using the *current* config,
            # which may have changed — so the reward could be attributed to
            # the wrong context row in the bandit table.
            # DEPRECATION: remove once all callers send previous_context_id.
            log.warning(
                "[analyse] previous_context_id missing — falling back to "
                "re-encoded context (reward attribution may be incorrect). "
                "Upgrade ollama_agent.py to send previous_context_id."
            )
            prev_ctx = encode_context_id(req.previous_affect, cfg.clarity_level, cfg.pace)
            reward = _compute_reward(req.previous_affect, affect)
            bandit.update(prev_ctx, req.previous_action_id, reward)
            reward_applied = reward

        action_id, ucb_scores = bandit.select_action(context_id)

        # Write updated tables back into the yielded arrays so storage saves them
        N[:] = bandit.N
        Q[:] = bandit.Q

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
            sadness_score=round(sadness_score, 4),
            ucb_scores=[round(s, 4) for s in ucb_scores],
            reward_applied=reward_applied,
            total_tries=int(bandit.N.sum()),
        ),
    )


# ── Reward table ──────────────────────────────────────────────────────────────

_REWARD_TABLE: dict[tuple[str, str], float] = {
    ("frustrated", "calm"):       +1.0,
    ("frustrated", "sad"):        +0.3,
    ("frustrated", "confused"):   +0.5,
    ("frustrated", "frustrated"): -0.5,
    ("confused",   "calm"):       +1.0,
    ("confused",   "sad"):        +0.2,
    ("confused",   "confused"):   -0.3,
    ("confused",   "frustrated"): -1.0,
    ("sad",        "calm"):       +1.0,
    ("sad",        "sad"):        -0.2,
    ("sad",        "confused"):   -0.5,
    ("sad",        "frustrated"): -1.0,
    ("calm",       "calm"):        0.0,
    ("calm",       "sad"):        -0.3,
    ("calm",       "confused"):   -0.5,
}

def _compute_reward(prev: str, curr: str) -> float:
    if curr == "disengaged":
        return -1.0
    return _REWARD_TABLE.get((prev, curr), 0.0)