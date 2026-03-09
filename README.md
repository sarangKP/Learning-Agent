# Learning Agent Microservice

A stateless FastAPI microservice that monitors conversations between the Conversation Agent
and an elderly user, detects emotional affect (frustrated / confused / sad / calm / disengaged),
and recommends live config adjustments to the Conversation Agent using
NLP + a Contextual Bandit (Discounted UCB1, γ=0.99).

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull an Ollama model (first time only)
ollama pull llama3.2:3b        # recommended
# or: ollama pull tinyllama    # lighter, ~600MB

# 4. Start everything (3 terminals)

# Terminal 1 — Learning Agent service
uvicorn main:app --reload --port 8000

# Terminal 2 — Ollama (if not already running)
ollama serve

# Terminal 3 — ELARA Conversation Agent (you chat as the user)
python ollama_agent.py --model llama3.2:3b

# 5. Run tests
pytest tests.py -v
```

---

## Project Structure

```
learning_agent/
├── main.py             ← FastAPI app + reward computation
├── schemas.py          ← Pydantic v2 request/response models (input validation)
├── nlp_layer.py        ← Layer 1: VADER sentiment + TF-IDF repetition + keyword detectors
├── state_classifier.py ← Layer 2: Rule-based affect classifier + escalation smoother + context encoder
├── bandit.py           ← Layer 3: Discounted UCB1 contextual bandit (γ=0.99)
├── config_applier.py   ← Maps action_id → config delta (step-clamped)
├── storage.py          ← File-based N/Q table persistence (swap for Redis in production)
├── ollama_agent.py     ← ELARA Conversation Agent powered by Ollama LLM
├── tests.py            ← pytest test suite (47 tests)
└── tables/             ← Auto-created; stores bandit_N.npy + bandit_Q.npy
```

---

## ELARA Conversation Agent

`ollama_agent.py` is the primary entry point. It runs a real LLM (via Ollama)
as ELARA, an empathetic elderly companion robot. The Learning Agent analyses
the conversation **every turn** and updates ELARA's behaviour config live —
those changes are injected directly into the LLM system prompt so ELARA's
tone, pace, and language change between replies.

```bash
# Interactive — you act as the elderly user
python ollama_agent.py

# Choose model
python ollama_agent.py --model llama3.2:3b

# Automated demo (scripted confusion/frustration arc, no keyboard needed)
python ollama_agent.py --auto

# Pure Ollama chat, no learning-agent adaptation
python ollama_agent.py --no-service
```

**In-session commands:**
```
/config    — show the live config right now
/prompt    — show the exact system prompt ELARA is using
/history   — show all learning-agent calls and what changed
/quit      — exit with session summary
```

**Recommended models (pull before use):**
```bash
ollama pull llama3.2:3b     # best overall for conversation following
ollama pull gemma2:2b       # excellent instruction following
ollama pull phi3:mini       # fast, good quality
ollama pull qwen2.5:3b      # good multilingual support
ollama pull tinyllama       # lightest, ~600MB (may ignore prompt instructions)
```

---

## How Config Affects ELARA

The Learning Agent updates 4 config parameters. Each is translated into
concrete instructions injected into the LLM system prompt:

| Parameter | Values | Effect on ELARA |
|---|---|---|
| `pace` | slow / normal / fast | Controls reply length and token budget (slow=280, normal=160, fast=70 tokens) |
| `clarity_level` | 1 / 2 / 3 | 1 = simplest words, one idea per sentence; 3 = more detail allowed |
| `confirmation_frequency` | low / medium / high | high = ELARA always repeats back what it understood before answering |
| `patience_mode` | true / false | true = every reply opens with a warm empathetic acknowledgement |

**Default config** (session start):
```json
{
  "pace": "normal",
  "clarity_level": 2,
  "confirmation_frequency": "low",
  "patience_mode": false
}
```

---

## Architecture

```
User message
     │
     ▼
ollama_agent.py  (ELARAAgent)
     │  every turn
     ├──────────────────────────► POST /analyse
     │                                 │
     │          ┌──────────────────────┤
     │          │   Learning Agent     │
     │          │                      │
     │          │  Layer 1: NLP        │
     │          │    VADER sentiment   │
     │          │    TF-IDF repetition │
     │          │    confusion keywords│
     │          │    sadness keywords  │
     │          │                      │
     │          │  Layer 2: Classifier │
     │          │    affect + ctx_id   │
     │          │    escalation smoother│
     │          │                      │
     │          │  Layer 3: UCB Bandit │
     │          │    update Q/N tables │
     │          │    select action     │
     └──────────┤                      │
                └──────────────────────┤
                                       │
     ◄─────────────────────────── config_delta
     │
     │  apply changes to self.config
     │  rebuild system prompt
     ▼
Ollama LLM  (ELARA replies with updated behaviour)
```

---

## State Classification

Five affect states. Decision priority (first match wins):

```
(negative AND repetitive) OR (very high confusion keywords)  →  frustrated
confusion keywords OR repetition (without sadness)           →  confused
sadness keywords OR (negative, no confusion, no repetition)  →  sad
≤3 word message, neutral, no signals                         →  disengaged
none of the above                                            →  calm
```

### Escalation Smoother

After raw classification, a rolling window of the last 5 affect states is
checked before high-impact affects are allowed to stand. This prevents
single-turn misclassifications from triggering aggressive config changes.

**Frustrated rules** (checked in order):

| Rule | Condition | Action |
|---|---|---|
| R3 `all_calm_history` | Every window entry is calm | Downgrade frustrated → confused (always, even with strong signals) |
| R1 `insufficient_streak` | Trailing non-calm streak < 2 | Downgrade frustrated → confused (bypassed if all 3 signals fire simultaneously) |

**Disengaged rule:**

| Rule | Condition | Action |
|---|---|---|
| R4 `calm_history_not_disengaged` | All window entries calm AND raw affect is disengaged | Downgrade disengaged → calm |

R4 prevents brief acknowledgements like "Yes." or "Okay." from being
misclassified as disengagement after a calm conversation. A genuinely
disengaged user will show a *pattern* of short messages across multiple turns,
not a single short message after calm history.

Rules are **additive dampeners only** — they never upgrade an affect.

---

## Input Validation

Validation is applied at two layers to catch malformed requests early:

**Layer 1 — API boundary (`schemas.py`):**
- `conversation_window.turns` is capped at 50 entries (returns 422 if exceeded)
- `affect_window` entries must be one of: `frustrated`, `confused`, `sad`, `calm`, `disengaged` (returns 422 on unknown values)
- `previous_affect` is similarly validated
- `affect_window` is capped at 5 entries (matches `WINDOW_SIZE`)

**Layer 2 — Classifier (`state_classifier.py`):**
- Any unknown affect strings that slip through are stripped with a warning log before the escalation smoother uses the window

---

## Bandit — Discounted UCB1

The bandit learns which config action works best for each context (affect × clarity × pace combination).
Discounting (γ=0.99) prevents the tables from becoming stale as the user's
needs shift over weeks or months.

```
N[ctx][action] = γ × N[ctx][action] + 1
Q[ctx][action] = γ × Q[ctx][action] + (1 − γ) × reward
```

UCB score per action (balances exploitation vs exploration):
```
score = Q[ctx][action] + sqrt(2 × log(total) / N[ctx][action])
          ↑ exploitation              ↑ exploration bonus
```

**Tuning γ per deployment:**
```
Single user, stable needs        → 0.999  (forgets very slowly)
Single user, changing needs      → 0.99   (default)
Small care facility (~10 users)  → 0.95   (adapts faster)
```

**Rule-based cold-start defaults:**
```
frustrated → 5 (DECREASE_CLARITY_AND_PACE)
confused   → 6 (CLARITY_AND_CONFIRMATION)
sad        → 4 (ENABLE_PATIENCE)
calm       → 0 (DO_NOTHING)
disengaged → 4 (ENABLE_PATIENCE)
```

**Context space:** 45 contexts (5 affects × 3 clarity levels × 3 pace values) × 7 actions

---

## Reward Table

The bandit learns from how affect transitions between consecutive calls:

| Transition | Reward |
|---|---|
| frustrated → calm | +1.0 |
| frustrated → sad | +0.3 |
| frustrated → confused | +0.5 |
| frustrated → frustrated | −0.5 |
| confused → calm | +1.0 |
| confused → sad | +0.2 |
| confused → confused | −0.3 |
| confused → frustrated | −1.0 |
| sad → calm | +1.0 |
| sad → sad | −0.2 |
| sad → confused | −0.5 |
| sad → frustrated | −1.0 |
| calm → calm | 0.0 |
| calm → sad | −0.3 |
| calm → confused | −0.5 |
| any → disengaged | −1.0 |

---

## API Reference

### GET /health

```json
{ "status": "ok" }
```

### POST /analyse

**Request:**
```json
{
  "schema_version": "1.0",
  "session_id": "elara-1234567890",
  "conversation_window": {
    "turns": [
      { "role": "user",  "text": "I don't understand. You're confusing me." },
      { "role": "agent", "text": "I'm sorry, let me explain more simply." }
    ]
  },
  "current_config": {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": false
  },
  "affect_window": ["calm", "calm", "calm", "confused"],
  "previous_affect": "confused",
  "previous_action_id": 6,
  "previous_context_id": 10,
  "interaction_count": 4,
  "user_profile_hint": "elderly"
}
```

> `affect_window`, `previous_affect`, `previous_action_id`, and `previous_context_id`
> are all `null` on the first turn of a session.

> `affect_window` must contain only valid affect strings. Invalid values return HTTP 422.

**Response:**
```json
{
  "schema_version": "1.0",
  "session_id": "elara-1234567890",
  "processing_time_ms": 4,
  "inferred_state": {
    "affect": "confused",
    "confidence": 0.72,
    "context_id": 10,
    "signals_used": ["confusion_keywords"],
    "escalation_rule_applied": null
  },
  "config_delta": {
    "apply": true,
    "changes": { "clarity_level": 1 },
    "reason": "affect_confused_detected"
  },
  "bandit_context": {
    "context_id": 10,
    "action_id": 6
  },
  "diagnostics": {
    "sentiment_score": -0.1,
    "repetition_score": 0.12,
    "confusion_score": 0.8,
    "sadness_score": 0.0,
    "ucb_scores": [1.2, 0.8, 0.9, 0.7, 1.1, 0.6, 99.0],
    "reward_applied": -0.3,
    "total_tries": 12
  }
}
```

**Action IDs:**

| ID | Name | Effect |
|---|---|---|
| 0 | DO_NOTHING | No change (calm: gradual recovery toward defaults) |
| 1 | DECREASE_CLARITY | clarity_level − 1 step |
| 2 | DECREASE_PACE | pace − 1 step |
| 3 | INCREASE_CONFIRMATION | confirmation_frequency + 1 step |
| 4 | ENABLE_PATIENCE | patience_mode → true |
| 5 | DECREASE_CLARITY_AND_PACE | clarity_level − 1 and pace − 1 |
| 6 | CLARITY_AND_CONFIRMATION | clarity_level − 1 and confirmation_frequency + 1 |

---

## Caller Contract

The service is **stateless** — the caller (ollama_agent.py) owns all session
state and echoes it back on every request.

**Required fields to echo back each turn:**
- `affect_window` — append the previous `inferred_state.affect` to your local list (keep last 5)
- `previous_affect` — the `inferred_state.affect` from the last response
- `previous_action_id` — the `bandit_context.action_id` from the last response
- `previous_context_id` — the `bandit_context.context_id` from the last response

This allows the bandit to correctly attribute rewards to the context and action
that were actually active when the response was generated.

---

## Known Limitations

- **Single-user concurrency:** The file-based table persistence (`tables/bandit_N.npy`, `tables/bandit_Q.npy`) is not concurrency-safe across multiple simultaneous sessions. For multi-user deployment, replace the storage backend with Redis (stub already present in `storage.py`).
- **Text-only affect signals:** The NLP pipeline operates on text alone. Multimodal signals (speech rate, tone of voice) would improve affect classification accuracy, particularly for the `disengaged` and `sad` states.
- **Context space sparsity:** With 45 contexts and a single user, many context cells will rarely be visited. The UCB exploration bonus ensures these are sampled eventually, but convergence is slow for infrequently-visited contexts.

---

## Development

```bash
# Run full test suite
pytest tests.py -v

# Run a specific test class
pytest tests.py::TestStateClassifier -v

# Check service is up
curl http://localhost:8000/health
```

**Test suite covers:** NLP signals, state classification, escalation rules (R1/R3/R4), context encoding, bandit cold-start and update logic, config applier transitions, and full request/response integration.