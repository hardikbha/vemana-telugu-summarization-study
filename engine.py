"""Baseline output checks. Passing these does not establish factual correctness."""
import re


def terminology_check(source, summary):
    lower = summary.lower()
    if "పింఛను" in source and ("pension" not in lower or "plant" in lower):
        raise ValueError("Summary did not preserve the pension term from the note.")
    if "విద్యుత్ తీగలు" in source and "తెగిపడ్డాయి" in source:
        if not any(word in lower for word in ["fallen", "fell", "broken", "downed"]) or any(word in lower for word in ["erected", "installed"]):
            raise ValueError("Summary did not preserve the broken/fallen electrical-wire event.")


def validate_summary(text, max_words=120):
    if not text or len(text.split()) > max_words:
        raise ValueError("The model did not produce a short English summary.")
    letters = [char for char in text if char.isalpha()]
    latin = sum("a" <= char.lower() <= "z" for char in letters)
    if not letters or latin / len(letters) < 0.8:
        raise ValueError("The model output is not predominantly English.")
    if re.search(r"\b(user (has )?requested|telugu (text|note)|specific constraints|as an ai|english summary:)\b", text, flags=re.I):
        raise ValueError("The model described the task instead of summarizing the note.")
    words = text.lower().split()
    phrases = [tuple(words[i:i+7]) for i in range(max(0, len(words)-6))]
    if phrases and max(phrases.count(phrase) for phrase in set(phrases)) >= 3:
        raise ValueError("The model output contains excessive repetition.")
