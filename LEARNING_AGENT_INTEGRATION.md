# Conversation Agent - Ready for Learning Agent Integration

## Hi Learning Agent Team! 👋

This document explains what the Conversation Agent does and how to integrate with it.

---

## What We Built

A **complete Conversation Agent** that:
1. Manages conversations with an elderly user (Akbar, 76 years old)
2. Sends you complete data every turn
3. Receives your affect analysis and config recommendations
4. Applies your recommendations immediately
5. Includes cache memory for pattern inference

---

## Quick Overview

```
User speaks → Conversation Agent → Sends payload to YOU (Learning Agent)
                                          ↓
                                   You analyze affect & decide config
                                          ↓
                                   Send back response
                                          ↓
Conversation Agent applies changes → Responds to user with new behavior
```

---

## What You'll Receive (Every Turn)

### Endpoint
```
POST http://127.0.0.1:8000/analyse
```

### Payload Structure
```json
{
  "schema_version": "1.0",
  "session_id": "session-1773310509-c1b12ef2",
  "timestamp": "2026-03-12T10:15:09.559579+00:00",
  "interaction_count": 5,
  
  "conversation_window": {
    "turns": [
      {"role": "user", "text": "My knee hurts"},
      {"role": "agent", "text": "I understand"}
    ]
  },
  
  "current_config": {
    "pace": "normal",
    "clarity_level": 2,
    "confirmation_frequency": "low",
    "patience_mode": false
  },
  
  "affect_window": ["neutral", "pain"],
  "previous_affect": "pain",
  "previous_action_id": "action_123",
  "previous_context_id": "context_456",
  
  "user_profile_hint": "elderly",
  
  "cache_summary": {
    "size": 21,
    "entries": [...],
    "patterns": {
      "total_turns": 10,
      "affect_changes": 5,
      "config_updates": 2,
      "recent_topics": []
    }
  }
}
```

### Field Explanations

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | string | Unique session identifier |
| `timestamp` | ISO 8601 | When this turn happened |
| `interaction_count` | int | Turn number in this session |
| `conversation_window.turns` | array | Last 10 conversation turns |
| `current_config` | object | Current behavior settings |
| `affect_window` | array | Last 5 detected affects |
| `previous_affect` | string | Most recent affect (from your last response) |
| `previous_action_id` | string | Last action you chose (for bandit) |
| `previous_context_id` | string | Last context (for bandit) |
| `user_profile_hint` | string | User type ("elderly") |
| `cache_summary` | object | Short-cycle memory with patterns |

---

## What We Need From You

### Response Format
```json
{
  "inferred_state": {
    "affect": "pain",
    "confidence": 0.85
  },
  
  "config_delta": {
    "apply": true,
    "changes": {
      "pace": "slow",
      "patience_mode": true,
      "clarity_level": 1
    }
  },
  
  "bandit_context": {
    "action_id": "action_empathy_high",
    "context_id": "context_pain_elderly"
  }
}
```

### Field Explanations

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `inferred_state.affect` | string | Yes | Detected emotion (e.g., "pain", "angry", "calm") |
| `inferred_state.confidence` | float | No | Confidence score 0-1 |
| `config_delta.apply` | bool | Yes | Should we apply changes? |
| `config_delta.changes` | object | If apply=true | Config parameters to change |
| `bandit_context.action_id` | string | Yes | Action ID for your bandit algorithm |
| `bandit_context.context_id` | string | Yes | Context ID for your bandit algorithm |

---

## Configuration Parameters

These are the parameters you can change to adapt the agent's behavior:

### 1. `pace` (string)
- **"slow"**: 130 WPM, longer pauses - Use when user is stressed/confused
- **"normal"**: 150 WPM, standard - Default
- **"fast"**: 170 WPM, quicker - Use when user is engaged/alert

### 2. `clarity_level` (int: 1-3)
- **1**: Simplest single sentences, no complex words - Use for high confusion/pain
- **2**: Gentle clear language, short sentences - Default
- **3**: Conversational and natural - Use when user is calm/engaged

### 3. `confirmation_frequency` (string)
- **"low"**: Minimal confirmation - Default
- **"medium"**: Occasional understanding checks
- **"high"**: Frequent checks - Use when user seems confused

### 4. `patience_mode` (bool)
- **false**: Direct responses - Default
- **true**: Start with warm empathetic acknowledgment - Use when user is upset/angry

---

## Cache Memory (For Your Inference)

We maintain a cache of up to 50 recent events. You receive a summary:

```json
"cache_summary": {
  "size": 21,
  "entries": [
    {"type": "turn", "role": "user", "text": "...", "cached_at": 1773310509},
    {"type": "affect", "value": "pain", "confidence": 0.85},
    {"type": "config_update", "changes": {"pace": "slow"}},
    {"type": "bandit_decision", "action_id": "...", "context_id": "..."}
  ],
  "patterns": {
    "total_turns": 10,
    "affect_changes": 5,
    "config_updates": 2,
    "recent_topics": ["health", "coffee"]
  }
}
```

**Use this to:**
- Identify conversation patterns
- Track affect change frequency
- See how often config needs adjustment
- Understand user topics/concerns

---

## Echo Fields (For Your Bandit Algorithm)

We send back what you told us last time:
- `previous_affect`: The affect you detected last turn
- `previous_action_id`: The action you chose last turn
- `previous_context_id`: The context you identified last turn

**Why?** So you can:
1. Evaluate if your last action worked
2. Update your bandit algorithm
3. Learn which actions work in which contexts

---

## Example Scenario

### Turn 1: User is in pain

**We send you:**
```json
{
  "conversation_window": {
    "turns": [{"role": "user", "text": "My knee is killing me today"}]
  },
  "current_config": {"pace": "normal", "patience_mode": false},
  "previous_affect": null
}
```

**You respond:**
```json
{
  "inferred_state": {"affect": "pain", "confidence": 0.9},
  "config_delta": {
    "apply": true,
    "changes": {"pace": "slow", "patience_mode": true, "clarity_level": 1}
  },
  "bandit_context": {
    "action_id": "action_empathy_high",
    "context_id": "context_pain_knee"
  }
}
```

**What happens:**
1. We update config: pace→slow, patience→true, clarity→1
2. We store affect: "pain"
3. We store your action/context IDs
4. Agent responds with: "I hear you, and I understand your knee is really hurting. Let's make sure you're comfortable..."
5. Response is slower, simpler, more empathetic

### Turn 2: User calms down

**We send you:**
```json
{
  "conversation_window": {
    "turns": [
      {"role": "user", "text": "My knee is killing me today"},
      {"role": "agent", "text": "I hear you..."},
      {"role": "user", "text": "Thanks, I feel better now"}
    ]
  },
  "current_config": {"pace": "slow", "patience_mode": true},
  "previous_affect": "pain",
  "previous_action_id": "action_empathy_high",
  "previous_context_id": "context_pain_knee"
}
```

**You respond:**
```json
{
  "inferred_state": {"affect": "calm", "confidence": 0.8},
  "config_delta": {
    "apply": true,
    "changes": {"pace": "normal", "patience_mode": false}
  },
  "bandit_context": {
    "action_id": "action_maintain",
    "context_id": "context_calm_recovery"
  }
}
```

**What happens:**
1. We update config back to normal
2. You can see your "action_empathy_high" worked (pain→calm)
3. Your bandit algorithm learns this action works for pain context

---

## Testing Integration

### Step 1: Start Your Learning Agent
```bash
cd Learning-Agent
uvicorn main:app --port 8000
```

### Step 2: We Start Conversation Agent
```bash
python conversation_agent.py
```

### Step 3: Test Conversation
```
[Akbar] My knee hurts badly
[Detected affect: pain]
[Config updated: {'pace': 'slow', 'patience_mode': True}]
[Agent] I understand your knee is hurting...
```

### Step 4: Verify
- Check your logs for incoming payloads
- Check our logs for your responses
- Verify config changes apply

---

## Health Check

We call this on startup:
```
GET http://127.0.0.1:8000/health
```

Should return 200 OK if you're ready.

---

## Error Handling

**If you're not available:**
- We continue working without adaptive learning
- We use default config
- We log "Learning Agent unavailable"
- No crash, graceful degradation

**If you return an error:**
- We keep current config
- We continue conversation
- We retry next turn

---

## Files You Might Want

### 1. `conversation_agent.py`
Main implementation - shows exactly how we call you

### 2. `test_conversation_agent.py`
Our test suite - shows payload structure

### 3. `test_export.json`
Sample session export - shows data format

### 4. `CONVERSATION_AGENT_DOCS.md`
Complete API documentation

### 5. `INTEGRATION_GUIDE.md`
Step-by-step integration guide

---

## User Profile (Akbar)

The elderly user we're caring for:
- **Name**: Akbar, 76 years old
- **Personality**: Gets angry fast, addicted to coffee, only wears brown shirts
- **Health**: Knee pain, neck pain, asthma
- **Family**: 2 daughters (both doctors)
- **Triggers**: Being rushed, delayed coffee, being told what to do
- **Calming factors**: Talking about daughters, memories, having coffee

This context helps you understand why certain affects appear.

---

## What We Track

### Statistics
- Total interactions
- Config changes
- Learning Agent calls (successful/failed)
- Tokens generated

### Session Export
When session ends, we export:
- Complete conversation history
- All affect states
- All config changes
- Cache contents
- Your action/context IDs

You can use this for offline analysis and model training.

---

## Questions?

### Q: What if I don't have a recommendation?
**A:** Set `config_delta.apply = false`, we'll keep current config

### Q: What affects should I return?
**A:** Common ones: "neutral", "pain", "angry", "calm", "happy", "confused", "frustrated"

### Q: How often will you call me?
**A:** Every single turn (every user message)

### Q: Can I see the cache entries?
**A:** Yes! They're in `cache_summary.entries` in every payload

### Q: What if my response is slow?
**A:** We have a 5-second timeout, then continue without your response

---

## Ready to Test?

1. Start your Learning Agent on port 8000
2. Let us know when ready
3. We'll start the Conversation Agent
4. Watch the integration work!

---

## Contact

If you need clarification on any payload fields or response format, just ask!

**Our Status:** ✅ Ready and tested (9/9 tests passing)

**Waiting for:** Your Learning Agent on port 8000

**Then:** Full integration testing together!
