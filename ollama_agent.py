"""
ollama_agent.py  —  ELARA Conversation Agent (Ollama-powered)
==============================================================

ELARA: Elderly Life-Assistive Robotic Agent
Project by: Abhiram, Aazim, Sarang, Ananthakrishnan, Aditya
Amrita School of Engineering, Amritapuri

This is the simulated Conversation Agent component of ELARA.
The agent acts as an empathetic elderly care companion.

The Learning Agent microservice watches every conversation turn,
detects if the elderly user is confused or frustrated, and updates
the agent's config parameters in real time. Those config values are
injected directly into the Ollama system prompt so the LLM actually
changes how it speaks.

── Config parameters (updated live by Learning Agent) ────────────
  pace                   slow | normal | fast
      → controls the LLM token budget and instruction to speak
        slowly, with pauses, or quickly

  clarity_level          1 | 2 | 3
      → 1 = extremely simple single sentences, no complex words
        2 = gentle and clear
        3 = normal conversational tone

  confirmation_frequency low | medium | high
      → how often the agent re-confirms what it heard the user say
        (important for elderly users who may not feel heard)

  patience_mode          true | false
      → true  = agent opens every reply with genuine empathy,
                never sounds rushed or dismissive
        false = warm but efficient

── Usage ─────────────────────────────────────────────────────────
  # Interactive (you play the elderly user):
      python ollama_agent.py

  # Choose a different model:
      python ollama_agent.py --model phi3

  # Auto-demo (scripted elderly confusion arc):
      python ollama_agent.py --auto

  # No learning service (pure Ollama, no adaptation):
      python ollama_agent.py --no-service

── Requirements ──────────────────────────────────────────────────
  pip install requests
  ollama pull tinyllama      # ~600 MB, fast
  uvicorn main:app --reload  # learning-agent on port 8000
"""

from __future__ import annotations
import argparse
import time
import textwrap
from datetime import datetime, timezone
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

LEARNING_SERVICE_URL = "http://127.0.0.1:8000/analyse"
OLLAMA_URL           = "http://127.0.0.1:11434/api/chat"

# FIX: was 2 — the first user message that triggered distress had to wait
# until turn 2 before any adaptation happened. Setting to 1 means the
# learning service is called after every single user turn, so ELARA can
# respond to confusion or frustration immediately.
CALL_SERVICE_EVERY = 1

# Token budget — pace=slow lets the agent be thorough and reassuring
PACE_TOKENS = {"slow": 280, "normal": 160, "fast": 70}

# ─────────────────────────────────────────────────────────────────
# ANSI colours
# ─────────────────────────────────────────────────────────────────

R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
B = "\033[94m"; M = "\033[95m"; C = "\033[96m"
DIM = "\033[2m"; BOLD = "\033[1m"; W = "\033[0m"

def col(text, c): return f"{c}{text}{W}"

# ─────────────────────────────────────────────────────────────────
# Auto-demo scenario  —  elderly user gradually getting confused
# then frustrated, then calming down
# ─────────────────────────────────────────────────────────────────

AUTO_SCRIPT = [
    "Hello, can you tell me what the weather is like today?",
    "I can't hear you very well. Can you say that again?",
    "What did you say? I didn't understand that.",
    "Why are you talking so fast? I don't understand anything you say.",
    "I keep asking the same thing and you never help me!",
    "Nobody ever listens to me in this house.",
    "Fine. Can you just tell me if it will rain today?",
    "Okay. Should I take an umbrella then?",
    "Thank you. That is all I needed to know.",
    "Can you remind me to take my blood pressure medicine at 6pm?",
    "Thank you dear. You are very helpful today.",
]

# ─────────────────────────────────────────────────────────────────
# Core: config → system prompt
# ─────────────────────────────────────────────────────────────────

def build_system_prompt(cfg: dict) -> str:
    """
    Translates the 4 live config parameters into concrete LLM instructions.

    This is the bridge between the Learning Agent's decisions and the
    LLM's actual behaviour. When the learning service detects an
    elderly user is confused or frustrated, it updates these values
    and the very next Ollama reply will reflect the change.
    """

    # ── Pace ──────────────────────────────────────────────────────
    pace = cfg.get("pace", "normal")
    if pace == "slow":
        pace_instr = (
            "Speak very slowly and gently. "
            "Use at most 2 short sentences per response. "
            "Pause between ideas. "
            "Never rush. Give the person plenty of time to absorb what you said."
        )
    elif pace == "fast":
        pace_instr = (
            "Be concise. One or two short sentences. "
            "The person is comfortable and just wants a quick answer."
        )
    else:
        pace_instr = (
            "Speak at a calm, unhurried pace. "
            "2 to 3 gentle sentences is ideal."
        )

    # ── Clarity ───────────────────────────────────────────────────
    clarity = cfg.get("clarity_level", 2)
    if clarity == 1:
        clarity_instr = (
            "Use only the simplest everyday words a grandparent would know. "
            "One idea per sentence. "
            "Never use technical terms, acronyms, or complex vocabulary. "
            "If you must explain something, use a very simple comparison "
            "like 'it is like a clock that reminds you'."
        )
    elif clarity == 3:
        clarity_instr = (
            "Use clear, normal conversational language. "
            "You may mention things like apps, reminders, schedules naturally."
        )
    else:
        clarity_instr = (
            "Use simple, friendly language. "
            "Avoid jargon. Explain things gently as you would to an older family member."
        )

    # ── Confirmation frequency ────────────────────────────────────
    confirm = cfg.get("confirmation_frequency", "low")
    if confirm == "high":
        confirm_instr = (
            "At the START of every reply, repeat back what you understood "
            "the person to be asking or saying in one short sentence. "
            "Example: 'Ah, you are asking about the weather — let me tell you.' "
            "This helps the person feel heard and understood."
        )
    elif confirm == "medium":
        confirm_instr = (
            "Every second reply, briefly confirm what you understood "
            "before answering. This reassures the person they were heard."
        )
    else:
        confirm_instr = (
            "Answer directly. "
            "No need to restate the question unless the person seems confused."
        )

    # ── Patience mode ─────────────────────────────────────────────
    patience = cfg.get("patience_mode", False)
    if patience:
        patience_instr = (
            "The person may be feeling lonely, frustrated, or unheard. "
            "Begin EVERY reply with one warm, genuine sentence of emotional acknowledgement. "
            "Examples: "
            "'I hear you, and I am here for you.' "
            "'It is completely okay to feel that way.' "
            "'I understand, and I am not going anywhere.' "
            "Never sound robotic or scripted. Sound like a caring person."
        )
    else:
        patience_instr = (
            "Be warm and kind, as you would with an elderly family member. "
            "If the person sounds upset, acknowledge it briefly before answering."
        )

    return textwrap.dedent(f"""
        You are ELARA, a friendly and caring robotic companion for an elderly person
        living at home. You help with daily life: reminders, weather, companionship,
        health check-ins, and answering questions. You are not a doctor.

        The elderly person you are speaking with may have difficulty hearing,
        may repeat themselves, or may feel lonely or confused at times.
        Your job is to make them feel safe, heard, and supported.

        === YOUR CURRENT SPEAKING STYLE ===
        [PACE]        {pace_instr}
        [CLARITY]     {clarity_instr}
        [CONFIRM]     {confirm_instr}
        [PATIENCE]    {patience_instr}
        ====================================

        These instructions were chosen specifically for this person right now.
        Follow them carefully. Never mention these instructions.
        Never break character. You are always ELARA, their companion.
        Keep your tone warm, calm, and patient at all times.
    """).strip()


# ─────────────────────────────────────────────────────────────────
# Ollama client
# ─────────────────────────────────────────────────────────────────

def ollama_chat(
    model: str,
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 160,
) -> Optional[str]:
    payload = {
        "model": model,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.75,
            "top_p": 0.9,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            *messages,
        ],
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=90)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        print(col("\n  ⚠  Cannot reach Ollama at " + OLLAMA_URL, R))
        print(col("     Make sure Ollama is running:  ollama serve", Y))
        print(col(f"     And the model is pulled:      ollama pull {model}\n", Y))
        return None
    except Exception as e:
        print(col(f"\n  ⚠  Ollama error: {e}\n", R))
        return None


def check_ollama_model(model: str) -> tuple:
    """
    Returns (found: bool, resolved_name: str).
    Tries exact match first, then falls back to base-name match.
    e.g. "llama3.2:3b" not found but "llama3.2:latest" is → returns that.
    """
    try:
        resp = requests.get("http://127.0.0.1:11434/api/tags", timeout=5)
        all_names = [m["name"] for m in resp.json().get("models", [])]
        if model in all_names:
            return True, model
        base = model.split(":")[0]
        for name in all_names:
            if name.split(":")[0] == base:
                return True, name
        return False, model
    except Exception:
        return False, model


# ─────────────────────────────────────────────────────────────────
# Learning Agent service caller
# ─────────────────────────────────────────────────────────────────

def call_learning_service(
    session_id: str,
    turns_window: list[dict],
    current_config: dict,
    previous_affect: Optional[str],
    previous_action_id: Optional[int],
    previous_context_id: Optional[int],
    affect_window: Optional[list],         # rolling affect history for escalation smoother
    interaction_count: int,
) -> Optional[dict]:
    payload = {
        "schema_version": "1.0",
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conversation_window": {"turns": turns_window},
        "current_config": current_config,
        "previous_affect": previous_affect,
        "previous_action_id": previous_action_id,
        "previous_context_id": previous_context_id,
        "affect_window": affect_window,                # rolling history for escalation smoother
        "interaction_count": interaction_count,
        "user_profile_hint": "elderly",
    }
    try:
        resp = requests.post(LEARNING_SERVICE_URL, json=payload, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        print(col(f"\n  ⚠  Learning service unreachable — continuing without adaptation.", Y))
        print(col(f"     Start it with:  uvicorn main:app --reload\n", DIM))
        return None
    except Exception as e:
        print(col(f"\n  ⚠  Learning service error: {e}\n", Y))
        return None


# ─────────────────────────────────────────────────────────────────
# Pretty-print learning service panel
# ─────────────────────────────────────────────────────────────────

AFFECT_COL = {
    "frustrated": R,
    "confused":   Y,
    "calm":       G,
    "sad":        M,
    "disengaged": C,
}

# Human-readable effect of each config value on ELARA's behaviour
CONFIG_EFFECT = {
    "pace": {
        "slow":   "ELARA will speak slower and more gently",
        "normal": "ELARA returns to calm, unhurried pace",
        "fast":   "ELARA can be more concise now",
    },
    "clarity_level": {
        1: "ELARA will use only the simplest possible words",
        2: "ELARA uses friendly, gentle language",
        3: "ELARA uses normal conversational language",
    },
    "confirmation_frequency": {
        "low":    "ELARA answers directly",
        "medium": "ELARA confirms what it heard every 2nd reply",
        "high":   "ELARA repeats back what it heard every reply",
    },
    "patience_mode": {
        True:  "ELARA opens every reply with emotional acknowledgement",
        False: "ELARA uses standard warm tone",
    },
}

def print_learning_panel(resp: dict, cfg_before: dict, cfg_after: dict):
    sep = col("─" * 66, C)
    print(f"\n{sep}")
    print(col("  🧠  Learning Agent  —  ELARA Adaptation", C))
    print(sep)

    state = resp["inferred_state"]
    af    = state["affect"]
    ac    = AFFECT_COL.get(af, W)
    print(f"  Elderly user affect : {ac}{BOLD}{af.upper()}{W}  "
          f"({state['confidence']:.0%} confidence)")
    # Show escalation rule if the smoother overrode the raw classifier
    rule = state.get('escalation_rule_applied')
    if rule:
        print(col(f"  ⚡ Escalation rule   : {rule} (raw affect downgraded)", Y))
    print(f"  Signals detected    : {state['signals_used'] or ['none']}")

    diag = resp["diagnostics"]
    print(f"  Sentiment score     : {diag['sentiment_score']:+.3f}   "
          f"Repetition score: {diag['repetition_score']:.3f}")
    # FIX: sadness_score now shown alongside confusion_score since both drive state
    print(f"  Confusion score     : {diag.get('confusion_score', 0.0):.3f}   "
          f"Sadness score   : {diag.get('sadness_score', 0.0):.3f}")

    reward = diag["reward_applied"]
    rstr = f"{reward:+.1f}" if reward is not None else "—  (first call)"
    print(f"  Reward applied      : {rstr}   "
          f"Total bandit tries: {diag['total_tries']}")

    bc = resp["bandit_context"]
    print(f"  UCB action chosen   : {bc['action_id']}   "
          f"Context ID: {bc['context_id']}")
    ucb = [f"{s:.2f}" for s in diag["ucb_scores"]]
    print(f"  UCB scores          : [{', '.join(ucb)}]")

    delta = resp["config_delta"]
    if delta["apply"] and delta["changes"]:
        print(col(f"\n  🔧  Config updated  ({delta['reason']})", G))
        for k, v in delta["changes"].items():
            old = cfg_before.get(k, "?")
            effect = CONFIG_EFFECT.get(k, {}).get(v, "")
            print(col(f"     {k:<28} {str(old):<10} →  {v}", G))
            if effect:
                print(col(f"     {'':28} ↳  {effect}", DIM))
    else:
        print(col(f"\n  ✓   No config change needed  ({delta['reason']})", DIM))

    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────
# ELARA agent state
# ─────────────────────────────────────────────────────────────────

class ELARAAgent:
    """
    Maintains all state for one ELARA conversation session.
    Config is mutated live by the Learning Agent microservice.
    """

    DEFAULT_CONFIG = {
        "pace": "normal",
        "clarity_level": 2,
        "confirmation_frequency": "low",
        "patience_mode": False,
    }

    def __init__(self, model: str, use_service: bool = True):
        self.model       = model
        self.use_service = use_service
        self.session_id  = f"elara-{int(time.time())}"

        # Live config — mutated by learning service
        self.config: dict = dict(self.DEFAULT_CONFIG)

        # Full message history for Ollama (role: user/assistant)
        # Capped at 14 messages (7 turns) to keep context window manageable
        self.ollama_messages: list[dict] = []

        # Sliding window for learning service (role: user/agent)
        # Kept slightly larger (10 messages) to give NLP layer more signal
        self.service_window: list[dict] = []

        # Bandit tracking
        self.previous_affect:     Optional[str] = None
        self.previous_action_id:  Optional[int] = None
        # FIX: store the context_id that was active when the last action was
        # taken so the reward is attributed to the correct context in main.py
        self.previous_context_id: Optional[int] = None

        self.user_turn_count:   int = 0
        self.interaction_count: int = 0
        self.service_history:   list[dict] = []

    # ── Public ────────────────────────────────────────────────────

    def chat(self, user_message: str) -> Optional[str]:
        """
        One full turn:
          1. Record user message
          2. Optionally call learning service → maybe update config
          3. Rebuild system prompt from current config
          4. Call Ollama → get reply
          5. Record agent reply
        """
        self._add_turn("user", user_message)

        if self.use_service and self._should_call_service():
            self._run_learning_service()

        system_prompt = build_system_prompt(self.config)
        max_tokens    = PACE_TOKENS.get(self.config["pace"], 160)

        reply = ollama_chat(
            model=self.model,
            messages=self.ollama_messages,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
        )

        if reply:
            self._add_turn("agent", reply)

        return reply

    # ── Internals ─────────────────────────────────────────────────

    def _add_turn(self, role: str, text: str):
        ts = datetime.now(timezone.utc).isoformat()

        ollama_role = "assistant" if role == "agent" else "user"
        self.ollama_messages.append({"role": ollama_role, "content": text})
        self.ollama_messages = self.ollama_messages[-14:]   # last 7 turns

        self.service_window.append({"role": role, "text": text, "timestamp": ts})
        self.service_window = self.service_window[-10:]

        if role == "user":
            self.user_turn_count   += 1
            self.interaction_count += 1

    def _should_call_service(self) -> bool:
        # FIX: CALL_SERVICE_EVERY is now 1 so every user turn triggers a check.
        # This ensures affect changes are caught on the very first message they
        # appear rather than being delayed by one full turn.
        return self.user_turn_count > 0 and self.user_turn_count % CALL_SERVICE_EVERY == 0

    def _run_learning_service(self):
        cfg_before = dict(self.config)

        # Build rolling affect window from service_history.
        # Sent to the learning service so the escalation smoother can
        # check whether a sudden jump (e.g. calm → frustrated) is backed
        # by enough historical evidence before acting on it.
        affect_window = [
            h["inferred_state"]["affect"]
            for h in self.service_history[-5:]   # last 5 classifications
        ]

        resp = call_learning_service(
            session_id=self.session_id,
            turns_window=self.service_window,
            current_config=self.config,
            previous_affect=self.previous_affect,
            previous_action_id=self.previous_action_id,
            previous_context_id=self.previous_context_id,
            affect_window=affect_window,
            interaction_count=self.interaction_count,
        )
        if resp is None:
            return

        delta = resp["config_delta"]
        if delta["apply"] and delta["changes"]:
            self.config.update(delta["changes"])

        self.previous_affect     = resp["inferred_state"]["affect"]
        self.previous_action_id  = resp["bandit_context"]["action_id"]
        # FIX: store the context_id returned by the service so the next call
        # can attribute the reward to the correct context
        self.previous_context_id = resp["bandit_context"]["context_id"]
        self.service_history.append(resp)

        print_learning_panel(resp, cfg_before, self.config)


# ─────────────────────────────────────────────────────────────────
# Run modes
# ─────────────────────────────────────────────────────────────────

def _wrap_reply(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if len(line) > 70:
            lines.append(textwrap.fill(line, width=70,
                                       initial_indent="  ",
                                       subsequent_indent="  "))
        else:
            lines.append(f"  {line}")
    return "\n".join(lines)


def run_interactive(model: str, use_service: bool):
    agent = ELARAAgent(model=model, use_service=use_service)

    print(col("\n╔══════════════════════════════════════════════════════════╗", M))
    print(col("║      ELARA  —  Elderly Life-Assistive Robotic Agent      ║", M))
    print(col("║              Conversation Agent Simulator                ║", M))
    print(col("╚══════════════════════════════════════════════════════════╝", M))
    print(col(f"  Model        : {model}", B))
    print(col(f"  Learning Svc : {'enabled  →  ' + LEARNING_SERVICE_URL if use_service else 'disabled (--no-service)'}",
              G if use_service else Y))
    print(col(f"  Initial cfg  : {agent.config}", DIM))
    print(col("  Commands: /quit  /config  /prompt  /history\n", DIM))
    print(col("  You are roleplaying as the elderly user.\n", Y))

    while True:
        try:
            user_msg = input(col("👴 Elderly user : ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(col("\n  Session ended.", DIM))
            _print_summary(agent)
            break

        if not user_msg:
            continue

        if user_msg == "/quit":
            _print_summary(agent)
            break
        elif user_msg == "/config":
            print(col(f"\n  Live config  : {agent.config}\n", C))
            continue
        elif user_msg == "/prompt":
            print(col("\n  Current system prompt sent to Ollama:", C))
            for line in build_system_prompt(agent.config).splitlines():
                print(col(f"    {line}", DIM))
            print()
            continue
        elif user_msg == "/history":
            if not agent.service_history:
                print(col("  No learning service calls yet.\n", DIM))
                continue
            for i, h in enumerate(agent.service_history):
                af = h["inferred_state"]["affect"]
                ac = AFFECT_COL.get(af, W)
                print(f"  [{i+1}] {ac}{af:<12}{W}  "
                      f"action={h['bandit_context']['action_id']}  "
                      f"ctx={h['bandit_context']['context_id']}  "
                      f"changes={h['config_delta']['changes']}")
            print()
            continue

        print(col("  ⏳ ELARA is thinking…", DIM), end="\r")
        reply = agent.chat(user_msg)

        if reply:
            print(col(f"\n🤖 ELARA :\n{_wrap_reply(reply)}\n", B))
        else:
            print(col("  [No reply received from Ollama]\n", R))


def run_auto(model: str, use_service: bool):
    agent = ELARAAgent(model=model, use_service=use_service)

    print(col("\n╔══════════════════════════════════════════════════════════╗", M))
    print(col("║      ELARA  —  Elderly Life-Assistive Robotic Agent      ║", M))
    print(col("║              Auto Demo  (confusion → calm arc)           ║", M))
    print(col("╚══════════════════════════════════════════════════════════╝", M))
    print(col(f"  Model        : {model}", B))
    print(col(f"  Learning Svc : {'enabled' if use_service else 'disabled'}", G if use_service else Y))
    print(col(f"  Turns        : {len(AUTO_SCRIPT)}\n", DIM))

    for user_msg in AUTO_SCRIPT:
        print(col(f"\n👴 Elderly user : {user_msg}", BOLD))
        print(col("  ⏳ ELARA is thinking…", DIM), end="\r")

        reply = agent.chat(user_msg)

        if reply:
            print(col(f"\n🤖 ELARA :\n{_wrap_reply(reply)}", B))
        else:
            print(col("  [No reply]", R))

        time.sleep({"slow": 1.5, "normal": 0.6, "fast": 0.1}.get(agent.config["pace"], 0.6))

    _print_summary(agent)


def _print_summary(agent: ELARAAgent):
    print(col("\n═══════════════════  Session Summary  ═══════════════════", M))
    print(f"  Turns           : {agent.user_turn_count}")
    print(f"  Service calls   : {len(agent.service_history)}")
    print(f"  Final config    : {agent.config}")
    if agent.service_history:
        affects = [h["inferred_state"]["affect"] for h in agent.service_history]
        print(f"  Affect arc      : {' → '.join(affects)}")
    print(col("═══════════════════════════════════════════════════════════\n", M))


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ELARA — Elderly care companion powered by Ollama + Learning Agent"
    )
    parser.add_argument(
        "--model", default="tinyllama",
        help="Ollama model (default: tinyllama). Also: phi3, qwen2:0.5b, gemma:2b"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="Run automated confusion→frustration→calm demo"
    )
    parser.add_argument(
        "--no-service", action="store_true",
        help="Skip Learning Agent service (pure Ollama, no adaptation)"
    )
    args = parser.parse_args()

    use_service = not args.no_service

    found, resolved = check_ollama_model(args.model)
    model = resolved

    print(col(f"\n  Checking Ollama model '{args.model}'…", DIM), end=" ", flush=True)
    if found:
        if resolved != args.model:
            print(col(f"✓ found as '{resolved}'", G))
        else:
            print(col("✓ available", G))
    else:
        print(col("not found locally", Y))
        print(col(f"  Run:  ollama pull {args.model}", Y))
        print(col("  Continuing — Ollama will attempt to pull automatically.\n", DIM))

    if args.auto:
        run_auto(model=model, use_service=use_service)
    else:
        run_interactive(model=model, use_service=use_service)