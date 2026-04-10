# ELARA Learning Agent Microservice

A stateless FastAPI microservice that monitors conversations between ELARA and an elderly user,
detects emotional affect (frustrated / confused / sad / calm / disengaged), and recommends live
config adjustments using NLP + a **Contextual Bandit (Discounted LinUCB, γ=0.95)**.

> **v1.1 — LinUCB Edition.**  
> The bandit has been upgraded from tabular Discounted UCB1 to a **Linear Upper Confidence Bound
> (LinUCB)** model. Instead of a fixed 45-context lookup table, LinUCB learns a linear reward
> model over a 7-dimensional feature vector, giving it the ability to generalise across unseen
> state combinations from the very first session.

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
# or: ollama pull tinyllama    # lighter, ~600 MB

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
├── main.py             ← FastAPI app + reward computation (LinUCB edition)
├── schemas.py          ← Pydantic v2 request/response models; now includes previous_config
├── nlp_layer.py        ← Layer 1: VADER sentiment + TF-IDF repetition + keyword detectors
├── state_classifier.py ← Layer 2: Rule-based affect classifier + escalation smoother
│                           + context encoder + feature vector encoder (encode_context_features)
├── bandit.py           ← Layer 3: Discounted LinUCB bandit (γ=0.95)
├── config_applier.py   ← Maps action_id → config delta (step-clamped)
├── storage.py          ← File-based A/b matrix persistence (swap for Redis in production)
├── ollama_agent.py     ← ELARA Conversation Agent powered by Ollama LLM
├── tests.py            ← pytest test suite
└── tables/             ← Auto-created; stores bandit_A.npy + bandit_b.npy
```

---

## ELARA Conversation Agent

`ollama_agent.py` is the primary entry point. It runs a real LLM (via Ollama) as ELARA, an
empathetic elderly companion robot. The Learning Agent analyses the conversation **every turn**
and updates ELARA's behaviour config live — those changes are injected directly into the LLM
system prompt so ELARA's tone, pace, and language adapt between replies.

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
ollama pull tinyllama       # lightest, ~600 MB (may ignore prompt instructions)
```

---

## How Config Affects ELARA

The Learning Agent updates 4 config parameters. Each is translated into concrete instructions
injected into the LLM system prompt:

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
     │          │    feature vector    │
     │          │    (7D for LinUCB)   │
     │          │                      │
     │          │  Layer 3: LinUCB     │
     │          │    update A/b mats   │
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

After raw classification, a rolling window of the last 5 affect states is checked before
high-impact affects are allowed to stand. This prevents single-turn misclassifications from
triggering aggressive config changes.

**Frustrated rules** (checked in order):

| Rule | Condition | Action |
|---|---|---|
| R3 `all_calm_history` | Every window entry is calm | Downgrade frustrated → confused (always, even with strong signals) |
| R1 `insufficient_streak` | Trailing non-calm streak < 2 | Downgrade frustrated → confused (bypassed if all 3 signals fire simultaneously) |

**Disengaged rule:**

| Rule | Condition | Action |
|---|---|---|
| R4 `calm_history_not_disengaged` | All window entries calm AND raw affect is disengaged | Downgrade disengaged → calm |

R4 prevents brief acknowledgements like "Yes." or "Okay." from being misclassified as
disengagement after a calm conversation. A genuinely disengaged user will show a *pattern* of
short messages across multiple turns, not a single short message after calm history.

Rules are **additive dampeners only** — they never upgrade an affect.

---

## Bandit — Discounted LinUCB

### Why LinUCB?

The previous UCB1 bandit used a 45 × 7 lookup table (45 contexts × 7 actions). Every new
context cell started cold and had to be visited independently before it could learn anything
useful. LinUCB replaces this with a **linear reward model**: it learns a weight vector `θ` per
action that maps a feature vector to an expected reward. This means it can generalise — a
pattern learned in a "confused + slow pace" context immediately informs estimates in related
contexts like "confused + normal pace".

### Feature Vector (7 dimensions)

Each state is encoded as a 7-dimensional vector before being passed to the bandit:

```
[  One-Hot Affect (5D)  |  clarity_level (1D)  |  pace_value (1D)  ]

Affect one-hot: frustrated=0, confused=1, sad=2, calm=3, disengaged=4
clarity_level:  passed as-is (1, 2, or 3)
pace_value:     slow=0.0, normal=1.0, fast=2.0
```

### LinUCB Update & Selection

One pair of matrices is maintained **per action** (7 action pairs total):

```
A[a]  — (7×7) covariance matrix, initialised to I (identity)
b[a]  — (7×1) reward-weighted feature vector, initialised to 0
```

**Update** (after observing reward r for action a with feature x):

```
A[a] = γ · A[a] + (1 − γ) · I + x · xᵀ     ← discount + new observation
b[a] = γ · b[a] + r · x                      ← decay old, add new
```

The `(1 − γ) · I` term keeps `A[a]` invertible even after heavy discounting.

**Selection** (Thompson / UCB style — pick highest score):

```
θ[a]  = A[a]⁻¹ · b[a]                        ← ridge regression coefficients
score = θ[a]ᵀ · x  +  α · √(xᵀ · A[a]⁻¹ · x)
         ↑ exploitation           ↑ exploration bonus (α = 0.8)
```

### Tuning Parameters

| Parameter | Default | Effect |
|---|---|---|
| `alpha` | 0.8 | Exploration strength. Higher = more exploratory early on. |
| `gamma` | 0.95 | Discount factor. Lower = forgets old data faster. |

```
Single user, stable needs        → gamma=0.999, alpha=0.5
Single user, changing needs      → gamma=0.95,  alpha=0.8  (default)
Small care facility (~10 users)  → gamma=0.90,  alpha=1.0
```

### Persistent Storage

Matrices are stored as NumPy arrays in the `tables/` directory:

```
tables/bandit_A.npy   — shape (7, 7, 7): one 7×7 covariance matrix per action
tables/bandit_b.npy   — shape (7, 7):    one 7D reward vector per action
```

Both files are replaced atomically on every request using a two-layer lock (threading.Lock +
fcntl.flock) to prevent race conditions under concurrent workers.

---

## Input Validation

Validation is applied at two layers to catch malformed requests early.

**Layer 1 — API boundary (`schemas.py`):**
- `conversation_window.turns` capped at 50 entries (returns 422 if exceeded)
- `affect_window` entries must be one of: `frustrated`, `confused`, `sad`, `calm`, `disengaged` (returns 422 on unknown values)
- `previous_affect` validated the same way
- `affect_window` capped at 5 entries (matches `WINDOW_SIZE`)
- `previous_config` (new in v1.1) carries the config that was active when the previous action was taken, enabling correct LinUCB reward attribution

**Layer 2 — Classifier (`state_classifier.py`):**
- Unknown affect strings that slip through are stripped with a warning log before the escalation smoother uses the window

---

## Reward Table

The bandit learns from how affect transitions between consecutive calls:

| Transition | Reward |
|---|---|
| frustrated → calm | +1.0 |
| frustrated → confused | +0.3 |
| frustrated → frustrated | −0.5 |
| confused → calm | +1.0 |
| confused → confused | −0.3 |
| confused → frustrated | −1.0 |
| sad → calm | +1.0 |
| sad → sad | −0.2 |
| calm → calm | 0.0 |
| calm → confused | −0.5 |
| any → disengaged | −1.0 |

---

## API Reference

### GET /health

```json
{ "status": "ok" }
```

### POST /analyse

**Request (v1.1):**

```json
{
  "schema_version": "1.1",
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
  "previous_config": {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": false
  },
  "interaction_count": 4
}
```

> `affect_window`, `previous_affect`, `previous_action_id`, and `previous_config` are all
> `null` on the first turn of a session.

> `previous_config` (new in v1.1) is the config that was active when the previous action was
> chosen. The service uses it to reconstruct the exact feature vector for correct reward
> attribution. Sending `current_config` here (the old behaviour) may attribute reward to the
> wrong feature vector if the config changed between turns.

**Response (v1.1):**

```json
{
  "schema_version": "1.1",
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
    "ucb_scores": [0.42, 0.31, 0.29, 0.35, 0.40, 0.28, 1.15],
    "reward_applied": -0.3,
    "total_tries": 0
  }
}
```

> `total_tries` is always 0 in v1.1. The LinUCB model does not maintain per-action visit
> counts the way UCB1 did; this field is preserved in the response schema for backward
> compatibility with existing callers.

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

The service is **stateless** — the caller (`ollama_agent.py`) owns all session state and echoes
it back on every request.

**Required fields to echo back each turn:**

| Field | Source | Purpose |
|---|---|---|
| `affect_window` | Append last `inferred_state.affect`; keep last 5 | Escalation smoother |
| `previous_affect` | Last `inferred_state.affect` | Reward computation |
| `previous_action_id` | Last `bandit_context.action_id` | Reward attribution |
| `previous_config` | Config that was active when last action was taken | Correct LinUCB feature vector for reward update |

`previous_config` is the key addition in v1.1. Because the config may change between turns
(the service itself recommends changes), using `current_config` for reward attribution would
reconstruct the wrong feature vector. Always send the config snapshot that was live at the
time the previous action was selected.

---

## Schema Version History

| Version | Changes |
|---|---|
| 1.0 | Initial release — tabular Discounted UCB1, 45 contexts × 7 actions |
| 1.1 | Bandit upgraded to Discounted LinUCB; `previous_config` added to request; storage switched from `bandit_N.npy`/`bandit_Q.npy` to `bandit_A.npy`/`bandit_b.npy`; `previous_context_id` removed (feature vector replaces context ID for reward attribution) |

---

## Known Limitations

- **Single-user concurrency:** File-based matrix persistence is not safe across multiple simultaneous sessions. For multi-user deployment, replace the storage backend with Redis (stub present in `storage.py`).
- **Text-only affect signals:** The NLP pipeline operates on text alone. Multimodal signals (speech rate, tone of voice) would improve classification accuracy, particularly for `disengaged` and `sad` states.
- **Linear reward assumption:** LinUCB assumes rewards are a linear function of the feature vector. Non-linear relationships (e.g., a user who only responds well to patience at night) are not captured. A kernelised or neural bandit would handle this better at the cost of increased complexity.
- **`total_tries` deprecated:** This field is always 0 in v1.1 and will be removed in a future version. Callers should not depend on it.

---

## Development

```bash
# Run full test suite
pytest tests.py -v

# Run a specific test class
pytest tests.py::TestStateClassifier -v

# Check service is up
curl http://localhost:8000/health

# Reset bandit matrices (clears all learned data)
python -c "from storage import reset_tables; reset_tables()"
```

**Test suite covers:** NLP signals, state classification, escalation rules (R1/R3/R4), context encoding, feature vector encoding, LinUCB update and selection logic, config applier transitions, and full request/response integration.