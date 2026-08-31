"""Intent classifier for the beat-based state machine.

Classifies customer utterances into POSITIVE/NEGATIVE/UNCLEAR/DNC/BUSY/QUESTION
with confidence scores. The state machine decides what to do with the classification.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import Config
from .llm import LLM
from .prompt import CLASSIFIER_PROMPT


MIN_ELIGIBLE_AGE = 45
MAX_ELIGIBLE_AGE = 80


@dataclass
class AgeResult:
    """Structured result of parsing a customer's stated age."""

    age: int | None
    detected: bool
    eligible: bool
    confirmed: bool = False


def extract_age(text: str) -> AgeResult:
    """Extract a customer's age from their utterance.

    Handles common phrasings like "I'm 67", "67 years old", "67 years of age",
    "about 70", and a bare affirmance like "yes". Returns None if no clear age
    is found.
    """
    if not text:
        return AgeResult(age=None, detected=False, eligible=False)

    low = text.lower().strip()

    # A bare "yes" / "yeah" / "sure" counts as confirmation -> eligible
    if re.fullmatch(r"(yes|yeah|yep|sure|ok|okay|correct|right|uh huh|mhm)", low):
        return AgeResult(age=None, detected=True, eligible=True, confirmed=True)

    patterns = [
        r"(\d{2,3})\s*(?:years?\s*)?(?:old|of\s*age)\b",
        r"\b(?:i['’]m|i\s+am)\s+(\d{2,3})",
        r"\b(?:a[gə]e[d]?)\s+(?:of\s+)?(\d{2,3})",
        r"\b(\d{2,3})\s*(?:years?\s*)(?:old|of\s*age)",
    ]

    for pat in patterns:
        m = re.search(pat, low)
        if m:
            try:
                age = int(m.group(1))
            except (ValueError, IndexError):
                continue
            if 18 <= age <= 120:
                return _age_result(age)

    m = re.search(r"\b(\d{2,3})\b", low)
    if m:
        try:
            age = int(m.group(1))
        except ValueError:
            age = None
        if age and 18 <= age <= 120:
            return _age_result(age)

    num_words = {
        "forty": 40, "forty one": 41, "forty two": 42, "forty three": 43,
        "forty four": 44, "forty five": 45, "forty six": 46, "forty seven": 47,
        "forty eight": 48, "forty nine": 49, "fifty": 50, "fifty one": 51,
        "fifty two": 52, "fifty three": 53,
        "fifty four": 54, "fifty five": 55, "fifty six": 56, "fifty seven": 57,
        "fifty eight": 58, "fifty nine": 59, "sixty": 60, "sixty one": 61,
        "sixty two": 62, "sixty three": 63, "sixty four": 64, "sixty five": 65,
        "sixty six": 66, "sixty seven": 67, "sixty eight": 68, "sixty nine": 69,
        "seventy": 70, "seventy one": 71, "seventy two": 72, "seventy three": 73,
        "seventy four": 74, "seventy five": 75, "seventy six": 76, "seventy seven": 77,
        "seventy eight": 78, "seventy nine": 79, "eighty": 80, "eighty one": 81,
        "eighty two": 82, "eighty three": 83, "eighty four": 84, "eighty five": 85,
        "eighty six": 86, "eighty seven": 87, "eighty eight": 88, "eighty nine": 89,
        "ninety": 90, "hundred": 100,
    }
    # Longest phrases first so "sixty seven" wins over "sixty"
    for word, age in sorted(num_words.items(), key=lambda kv: -len(kv[0])):
        if re.search(rf"\b{word}\b", low):
            return _age_result(age)

    return AgeResult(age=None, detected=False, eligible=False)


def _age_result(age: int) -> AgeResult:
    return AgeResult(
        age=age,
        detected=True,
        eligible=MIN_ELIGIBLE_AGE <= age <= MAX_ELIGIBLE_AGE,
    )


@dataclass
class ClassificationResult:
    intent: str
    decision_maker: bool
    confidence: float
    dnc: bool
    busy: bool
    transfer: bool
    reason: str

    @classmethod
    def from_dict(cls, d: dict) -> "ClassificationResult":
        return cls(
            intent=d.get("intent", "UNCLEAR"),
            decision_maker=d.get("decision_maker", False),
            confidence=float(d.get("confidence", 0)),
            dnc=d.get("dnc", False),
            busy=d.get("busy", False),
            transfer=d.get("transfer", False),
            reason=d.get("reason", ""),
        )


def classify_utterance(llm: LLM, customer_text: str, prior_context: str | None = None) -> ClassificationResult:
    parts = []
    if prior_context:
        parts.append(f'Prior AI question: "{prior_context}"')
    parts.append(f'Customer said: "{customer_text}"')
    user_msg = "\n".join(parts)

    messages = [
        {"role": "system", "content": CLASSIFIER_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = llm.chat_json(messages, temperature=0.1)
        if not raw:
            return _fail_safe("empty response")
        return ClassificationResult.from_dict(raw)
    except Exception as e:
        return _fail_safe(str(e))


def _fail_safe(reason: str) -> ClassificationResult:
    return ClassificationResult(
        intent="UNCLEAR",
        decision_maker=False,
        confidence=0.0,
        dnc=False,
        busy=False,
        transfer=False,
        reason=f"classifier_error: {reason}",
    )


POSITIVE_THRESHOLD = 0.85
NEGATIVE_THRESHOLD = 0.85
DNC_THRESHOLD = 0.75
MAX_CLARIFY_ATTEMPTS = 1


def should_transfer(result: ClassificationResult) -> bool:
    return result.intent == "POSITIVE" and result.confidence >= POSITIVE_THRESHOLD


def should_hangup(result: ClassificationResult) -> bool:
    if result.dnc and result.confidence >= DNC_THRESHOLD:
        return True
    if result.intent == "NEGATIVE" and result.confidence >= NEGATIVE_THRESHOLD:
        return True
    if result.intent == "BUSY":
        return True
    return False


def should_clarify(result: ClassificationResult) -> bool:
    return result.intent in ("UNCLEAR",) and result.confidence < POSITIVE_THRESHOLD
