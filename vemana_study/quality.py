"""Mechanical diagnostics only. No diagnostic establishes source fidelity."""

from collections import Counter

from engine import terminology_check
from .prompts import EXAMPLES


def check_output(source, output, language, *, ended, hit_limit, examples=0):
    if language not in {"en", "te"}:
        raise ValueError("language must be en or te")
    errors = []
    letters = [char for char in output if char.isalpha()]
    if language == "en":
        matching = sum("a" <= char.lower() <= "z" for char in letters)
    else:
        matching = sum("\u0c00" <= char <= "\u0c7f" for char in letters)
    fraction = matching / len(letters) if letters else 0.0
    if not output.strip():
        errors.append("empty_output")
    if fraction < 0.8:
        errors.append("target_script_mismatch")
    words = output.casefold().split()
    if len(words) > 120:
        errors.append("over_120_words")
    phrases = Counter(tuple(words[i:i + 7]) for i in range(len(words) - 6))
    if phrases and max(phrases.values()) >= 3:
        errors.append("excessive_repetition")
    if hit_limit and not ended:
        errors.append("output_budget_exhausted")
    if language == "en":
        try:
            terminology_check(source, output)
        except ValueError as exc:
            errors.append(str(exc))
    copied = []
    for index, example in enumerate(EXAMPLES[:examples], 1):
        example_words = example[language].casefold().split()
        if any(tuple(example_words[i:i + 7]) in phrases for i in range(len(example_words) - 6)):
            copied.append(index)
    return {
        "status": "rejected_basic_checks" if errors else "needs_human_review",
        "errors": errors,
        "target_script_fraction": round(fraction, 4),
        "example_overlap_ids": copied,
        "requires_human_review": True,
        "factual_accuracy": None,
    }
