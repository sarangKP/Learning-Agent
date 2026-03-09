# Learning Agent Microservice

A stateless FastAPI microservice that monitors conversations between Conversation Agent
and an elderly user, detects affect (frustrated / confused / calm), and
recommends live config adjustments to the Conversation Agent using
NLP + a Contextual Bandit (UCB1).

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
├── schemas.py          ← Pydantic v2 request/response models
├── nlp_layer.py        ← Layer 1: VADER sentiment + TF-IDF repetition
├── state_classifier.py ← Layer 2: Rule-based affect classifier + context encoder
├── bandit.py           ← Layer 3: UCB1 contextual bandit
├── config_applier.py   ← Maps action_id → config delta (step-clamped)
├── storage.py          ← File-based N/Q table persistence (swap for Redis)
├── ollama_agent.py     ← ELARA Conversation Agent powered by Ollama LLM
├── simulator.py        ← Lightweight simulator (no LLM, scripted replies)
├── tests.py            ← pytest test suite
├── requirements.txt
└── tables/             ← Auto-created; stores bandit_N.npy + bandit_Q.npy
```

---

## ELARA Conversation Agent

`ollama_agent.py` is the primary entry point. It runs a real LLM (via Ollama)
as ELARA, an empathetic elderly companion robot. The Learning Agent analyses
the conversation every 2 turns and updates ELARA's behaviour config live —
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
     │  every 2 turns
     ├──────────────────────────► POST /analyse
     │                                 │
     │          ┌──────────────────────┤
     │          │   Learning Agent     │
     │          │                      │
     │          │  Layer 1: NLP        │
     │          │    VADER sentiment   │
     │          │    TF-IDF repetition │
     │          │                      │
     │          │  Layer 2: Classifier │
     │          │    affect + ctx_id   │
     │          │                      │
     │          │  Layer 3: UCB Bandit │
     │          │    update Q/N tables │
     │          │    select action     │
     │          └──────────────────────┤
     │                                 │
     ◄─────────────────────────── config_delta
     │
     │  apply changes to self.config
     │  rebuild system prompt
     ▼
Ollama LLM  (ELARA replies with updated behaviour)
```

---

## State Classification

```
sentiment < -0.2  AND  repetition > 0.65  →  frustrated
repetition > 0.65 only                    →  confused
sentiment < -0.2  only                    →  confused
default                                   →  calm
```

Context ID encoding (0–35):
```
affect_idx (0–3) × 9  +  clarity_idx (0–2) × 3  +  pace_idx (0–2)
```

---

## Actions

| ID | Name | Effect |
|---|---|---|
| 0 | DO_NOTHING | No change (always used when affect = calm) |
| 1 | DECREASE_CLARITY | clarity_level − 1 |
| 2 | DECREASE_PACE | pace one step slower |
| 3 | INCREASE_CONFIRMATION | confirmation_frequency + 1 |
| 4 | ENABLE_PATIENCE | patience_mode = true |
| 5 | DECREASE_CLARITY_AND_PACE | clarity − 1, pace slower |
| 6 | CLARITY_AND_CONFIRMATION | clarity − 1, confirmation + 1 |

**Bandit rules:**
- `calm` context → always returns action 0 (DO_NOTHING), never changes config on a happy user
- Cold start (no visits for context) → rule-based default for the affect
- Otherwise → UCB1 selects the best learned action

---

## Reward Table

The bandit learns from how affect changes between calls:

| Transition | Reward |
|---|---|
| frustrated → calm | +1.0 |
| frustrated → confused | +0.5 |
| frustrated → frustrated | −0.5 |
| confused → calm | +1.0 |
| confused → confused | −0.3 |
| confused → frustrated | −1.0 |
| any → disengaged | −1.0 |
| calm → calm | 0.0 |

---

## API Reference

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
  "previous_affect": "confused",
  "previous_action_id": 6,
  "interaction_count": 4,
  "user_profile_hint": "elderly"
}
```

**Response:**
```json
{
  "schema_version": "1.0",
  "session_id": "elara-1234567890",
  "processing_time_ms": 42,
  "inferred_state": {
    "affect": "frustrated",
    "confidence": 0.84,
    "context_id": 14,
    "signals_used": ["sentiment", "repetition"]
  },
  "config_delta": {
    "apply": true,
    "changes": { "clarity_level": 1, "pace": "slow" },
    "reason": "affect_frustrated_detected"
  },
  "bandit_context": { "context_id": 14, "action_id": 5 },
  "diagnostics": {
    "sentiment_score": -0.61,
    "repetition_score": 0.71,
    "ucb_scores": [0.1, 0.4, 0.6, 0.3, 0.5, 0.7, 0.2],
    "reward_applied": -0.3,
    "total_tries": 8
  }
}
```

### GET /health
```json
{ "status": "ok" }
```

---

## Storage

N and Q tables persist across sessions as numpy files:
```
tables/bandit_N.npy   ← visit counts  (36 × 7)
tables/bandit_Q.npy   ← avg rewards   (36 × 7)
```

**Reset tables** (start fresh):
```bash
rm -rf tables/
```

**Swap to Redis for production** — in `storage.py`:
```python
import redis, pickle
_r = redis.Redis()

def load_tables():
    raw_N = _r.get("bandit:N")
    raw_Q = _r.get("bandit:Q")
    if raw_N and raw_Q:
        return pickle.loads(raw_N), pickle.loads(raw_Q)
    return np.zeros((36, 7)), np.zeros((36, 7))

def save_tables(N, Q):
    _r.set("bandit:N", pickle.dumps(N))
    _r.set("bandit:Q", pickle.dumps(Q))
```

---

## Tech Stack

```
FastAPI + uvicorn       API server
Pydantic v2             Request/response validation
vaderSentiment          Sentiment analysis (Layer 1)
scikit-learn            TF-IDF + cosine similarity (Layer 1)
numpy                   Bandit N/Q tables (Layer 3)
Ollama                  Local LLM inference for ELARA
requests                HTTP client (ollama_agent → service)
pytest                  Test suite
```