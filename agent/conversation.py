"""Beat-based state machine for the fronter dialogue.

Matches the TS architecture: GREETING → HOOK → CLASSIFY → TRANSFER/CLOSE.
Each beat generates one line via the LLM, with static fallbacks for reliability.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .llm import LLM
from .prompt import build_persona, BEAT_INSTRUCTIONS, next_line_prompt
from .qualifier import classify_utterance, ClassificationResult, should_transfer, should_hangup, should_clarify, extract_age

FALLBACKS = {
    "greet": "Hi, this is {name} from {campaign} — how are you today?",
    "react_and_hook": "Good to hear. We have fixed-income coverage for final expenses — how young are you?",
    "ask_age": "And how old are you, if you don't mind me asking?",
    "age_eligible": "That's great — you're eligible for a lower monthly premium. One moment, let me get you connected.",
    "age_not_eligible": "I appreciate you asking, but this option is for ages 45 to 80. Thank you for your time.",
    "clarify_age": "I'm sorry, I didn't catch that. How old are you?",
    "clarify": "I just mean, do you make your own financial decisions?",
    "transfer_handoff": "Perfect, one moment.",
    "busy_close": "No problem. Have a great day.",
    "not_interested_close": "I understand. Have a great day.",
    "already_covered_close": "Understood. Have a great day.",
    "who_is_this": "This is {name} with {campaign}. We have fixed-income coverage for final expenses — how young are you?",
    "what_is_this_about": "It's regarding fixed-income coverage for final expenses. How young are you?",
    "hello_checkin": "Hello? Just checking you're still there.",
}


@dataclass
class BeatResult:
    ai_speech: str | None
    end_call: bool
    disposition: str = ""
    classification: dict = field(default_factory=dict)


@dataclass
class ConversationState:
    state: str = "GREETING"
    clarify_attempts: int = 0
    age_clarify_attempts: int = 0
    last_ai_line: str = ""
    transcript: list[dict] = field(default_factory=list)
    disposition: str = ""
    dnc: bool = False
    decision_maker: bool = False
    age: int | None = None
    transfer_status: str = ""


class Conversation:
    def __init__(self, config: Config, llm: LLM):
        self.config = config
        self.llm = llm
        self.persona = build_persona(config)
        self.ctx = ConversationState()
        self.ended = False

    def _fallback(self, beat: str) -> str:
        return FALLBACKS.get(beat, "").format(
            name=self.config.agent_name,
            campaign=self.config.business_name,
        )

    def _recent_turns(self) -> str:
        turns = self.ctx.transcript[-6:]
        return "\n".join(
            f"{'You' if t['role'] == 'ai' else 'Customer'}: {t['text']}"
            for t in turns
        )

    def _generate_line(self, beat: str, customer_utterance: str | None) -> str:
        prompt = next_line_prompt(self.config, beat, customer_utterance, self._recent_turns())
        messages = [
            {"role": "system", "content": self.persona},
            {"role": "user", "content": prompt},
        ]
        try:
            line = self.llm.chat(messages, temperature=0.6, max_tokens=120).strip().strip('"')
            if not line:
                return self._fallback(beat)
            return line
        except Exception:
            return self._fallback(beat)

    def start(self) -> str:
        line = self._generate_line("greet", None)
        self.ctx.transcript.append({"role": "ai", "text": line})
        self.ctx.last_ai_line = line
        self.ctx.state = "WAIT_FOR_RESPONSE"
        return line

    def hear(self, customer_text: str) -> BeatResult:
        if self.ended:
            return BeatResult(ai_speech=None, end_call=True)

        self.ctx.transcript.append({"role": "customer", "text": customer_text})

        # After greeting + hook, the caller should have answered "how young are you?".
        # Move to age detection.
        if self.ctx.state in ("GREETING", "WAIT_FOR_RESPONSE"):
            self.ctx.state = "AGE_DETECTION"
            line = self._generate_line("react_and_hook", customer_text)
            self.ctx.transcript.append({"role": "ai", "text": line})
            self.ctx.last_ai_line = line
            return BeatResult(ai_speech=line, end_call=False)

        if self.ctx.state == "AGE_DETECTION":
            return self._handle_age(customer_text)

        if self.ctx.state in ("HOOK", "DECISION_MAKER_DETECTION", "CLARIFY"):
            return self._handle_classify(customer_text)

        return BeatResult(ai_speech=None, end_call=True)

    def _handle_age(self, customer_text: str) -> BeatResult:
        age_res = extract_age(customer_text)

        # Handle DNC / busy / questions even during age collection
        result = classify_utterance(self.llm, customer_text, self.ctx.last_ai_line)
        if result.dnc and result.confidence >= 0.75:
            self.ctx.dnc = True
            self._finalize("AI-DNC")
            return BeatResult(ai_speech=None, end_call=True, disposition="AI-DNC")
        if result.intent == "BUSY":
            self._finalize("AI-BUSY")
            line = self._generate_line("busy_close", customer_text)
            return BeatResult(ai_speech=line, end_call=True, disposition="AI-BUSY")

        if age_res.detected:
            self.ctx.age = age_res.age
            if age_res.eligible:
                # 55+ → directly eligible, no decision-maker step, go to transfer
                self._finalize("AI-TRANSFER")
                line = self._generate_line("transfer_handoff", customer_text)
                return BeatResult(
                    ai_speech=line, end_call=True, disposition="AI-TRANSFER",
                    classification={"age": age_res.age, "eligible": True},
                )
            else:
                # Under 55 → not eligible
                self._finalize("AI-NOT-ELIGIBLE")
                line = self._generate_line("age_not_eligible", customer_text)
                return BeatResult(ai_speech=line, end_call=True, disposition="AI-NOT-ELIGIBLE")

        # No clear age yet — clarify once, then give up
        if self.ctx.age_clarify_attempts >= 1:
            self._finalize("AI-UNCLEAR")
            return BeatResult(ai_speech=None, end_call=True, disposition="AI-UNCLEAR")
        self.ctx.age_clarify_attempts += 1
        line = self._generate_line("clarify_age", customer_text)
        self.ctx.transcript.append({"role": "ai", "text": line})
        self.ctx.last_ai_line = line
        return BeatResult(ai_speech=line, end_call=False)

    def _handle_classify(self, customer_text: str) -> BeatResult:
        result = classify_utterance(
            self.llm,
            customer_text,
            self.ctx.last_ai_line,
        )

        if result.dnc and result.confidence >= 0.75:
            self.ctx.dnc = True
            self._finalize("AI-DNC")
            return BeatResult(ai_speech=None, end_call=True, disposition="AI-DNC")

        if result.intent == "BUSY":
            self._finalize("AI-BUSY")
            line = self._generate_line("busy_close", customer_text)
            return BeatResult(ai_speech=line, end_call=True, disposition="AI-BUSY")

        if result.intent == "QUESTION":
            beat = "who_is_this" if "who" in customer_text.lower() else "what_is_this_about"
            line = self._generate_line(beat, customer_text)
            self.ctx.transcript.append({"role": "ai", "text": line})
            self.ctx.last_ai_line = line
            return BeatResult(ai_speech=line, end_call=False)

        if should_transfer(result):
            self.ctx.decision_maker = True
            self.ctx.transfer_status = "pending"
            self._finalize("AI-TRANSFER")
            line = self._generate_line("transfer_handoff", customer_text)
            return BeatResult(
                ai_speech=line, end_call=True, disposition="AI-TRANSFER",
                classification={"intent": result.intent, "confidence": result.confidence},
            )

        if should_hangup(result):
            self.ctx.decision_maker = False
            self._finalize("AI-NOT-DECISION-MAKER")
            line = self._generate_line("not_interested_close", customer_text)
            return BeatResult(ai_speech=line, end_call=True, disposition="AI-NOT-DECISION-MAKER")

        if should_clarify(result):
            if self.ctx.clarify_attempts >= 1:
                self._finalize("AI-UNCLEAR")
                return BeatResult(ai_speech=None, end_call=True, disposition="AI-UNCLEAR")
            self.ctx.clarify_attempts += 1
            self.ctx.state = "CLARIFY"
            line = self._generate_line("clarify", customer_text)
            self.ctx.transcript.append({"role": "ai", "text": line})
            self.ctx.last_ai_line = line
            return BeatResult(ai_speech=line, end_call=False)

        self._finalize("AI-UNCLEAR")
        return BeatResult(ai_speech=None, end_call=True, disposition="AI-UNCLEAR")

    def _finalize(self, disposition: str):
        self.ctx.disposition = disposition
        self.ended = True
