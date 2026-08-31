"""Mock LLM for smoke-testing without API keys.

Provides a simple heuristic classifier that matches the TS project's intent
classification, so the beat-based state machine can be tested offline.
"""
from __future__ import annotations

import json
import re

from .config import Config


class MockLLM:
    def __init__(self, config: Config):
        self.config = config

    def chat(self, messages, temperature=0.7, max_tokens=1500) -> str:
        system = ""
        user_parts = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            elif m["role"] == "user":
                user_parts.append(m["content"])

        combined = "\n".join(user_parts)
        last_user = user_parts[-1] if user_parts else ""

        if "classify" in system.lower() or "intent classifier" in system.lower():
            return self._classify(combined)
        return self._respond(last_user)

    def chat_json(self, messages, temperature=0.3) -> dict:
        raw = self.chat(messages, temperature=temperature, max_tokens=500)
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {"intent": "UNCLEAR", "confidence": 0.5}

    def models_available(self) -> bool:
        return False

    def _classify(self, text: str) -> str:
        low = text.lower()
        result = {
            "intent": "UNCLEAR",
            "decision_maker": False,
            "confidence": 0.5,
            "dnc": False,
            "busy": False,
            "transfer": False,
            "reason": "mock classifier",
        }

        if any(w in low for w in ["take me off", "stop calling", "don't call", "remove"]):
            result["intent"] = "DNC"
            result["dnc"] = True
            result["confidence"] = 0.95
            return json.dumps(result)

        if any(w in low for w in ["busy", "can't talk", "in a meeting", "call back"]):
            result["intent"] = "BUSY"
            result["busy"] = True
            result["confidence"] = 0.9
            return json.dumps(result)

        if any(w in low for w in ["who is", "who's", "what is this", "what's this about"]):
            result["intent"] = "QUESTION"
            result["confidence"] = 0.9
            return json.dumps(result)

        if any(w in low for w in [
            "yes", "yeah", "yep", "sure", "i do", "that's me", "i handle",
            "i make", "correct", "right", "absolutely", "definitely",
        ]):
            result["intent"] = "POSITIVE"
            result["decision_maker"] = True
            result["transfer"] = True
            result["confidence"] = 0.9
            result["reason"] = "caller confirmed decision-maker status"
            return json.dumps(result)

        if any(w in low for w in [
            "my wife", "my husband", "my son", "my daughter",
            "someone else", "not interested", "no thanks",
        ]):
            result["intent"] = "NEGATIVE"
            result["decision_maker"] = False
            result["confidence"] = 0.85
            result["reason"] = "caller indicated not decision-maker"
            return json.dumps(result)

        return json.dumps(result)

    def _respond(self, text: str) -> str:
        # Only consider the instruction portion of the prompt ("Your task for this line: ..."),
        # so conversation history doesn't confuse keyword matching.
        task = text
        idx = text.lower().find("your task for this line")
        if idx != -1:
            task = text[idx + len("your task for this line"):]
        low = task.lower()
        if "greet them professionally" in low:
            return "Hello, this is Lina calling from Senior Resource Center. How are you doing today?"
        if "how young are you" in low or "how old are you" in low:
            return "I understand. We offer fixed-income coverage for final expenses. How young are you?"
        if "licensed agent" in low:
            return "Great — you're eligible for a lower monthly premium. Let me connect you with a licensed agent who can assist you further."
        if "lower monthly premium" in low:
            return "That's great — you're eligible for a lower monthly premium. One moment, let me get you connected."
        if "under 55" in low or "don't qualify" in low or "45 to 80" in low or "outside the 45-80" in low:
            return "I appreciate you asking, but this option is for ages 45 to 80. Thank you for your time."
        if "didn't catch that" in low or "didn't catch their age" in low:
            return "I'm sorry, I didn't catch that. How old are you?"
        if "calmly and respectfully clarify" in low or "let me be a bit clearer" in low:
            return "Let me be a bit clearer. Could you tell me your age?"
        if "offer to follow up later" in low or "acknowledge they're busy" in low:
            return "I understand. I'll let you go for now. Thank you for your time."
        if "acknowledge their decision" in low or "thank them for their time" in low:
            return "I understand completely. Thank you for your time today."
        if "already covered" in low:
            return "That's good to hear. Thank you for your time today."
        if "your name and that you're calling from" in low:
            return "This is Lina calling from Senior Resource Center. We offer fixed-income coverage for final expenses. How young are you?"
        if "it's about fixed-income coverage" in low:
            return "It's about fixed-income coverage for final expenses. How young are you?"
        if "confirm they're still on the line" in low:
            return "Hello? Are you still there?"
        return "I understand. Let me see how I can help."
