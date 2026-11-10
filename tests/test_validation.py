import importlib.util
from pathlib import Path

import pytest

from engine import terminology_check, validate_summary


def runner():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_vemana.py"
    spec = importlib.util.spec_from_file_location("benchmark_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["direct", "fewshot"])
def test_prompt_contains_the_complete_unchanged_source(mode):
    source = "మా గ్రామంలో రెండు రోజులుగా తాగునీరు రావడం లేదు."
    module = runner()
    assert source in module.prompt(source, mode)
    assert module.prompt(source, mode).endswith("English summary:")


def test_custom_completion_prompt_preserves_source():
    module = runner()
    module.CUSTOM_PROMPT = "పాఠ్యం:\n{source}\nసారాంశం:"
    assert module.prompt("తెలుగు మూలం", "fewshot") == "పాఠ్యం:\nతెలుగు మూలం\nసారాంశం:"


def test_context_budget_does_not_silently_shorten_the_source():
    module = runner()
    assert module.output_budget(500, 192) == 192
    assert module.output_budget(1000, 192) == 24
    assert module.output_budget(1100, 192) == 0


def test_language_and_repetition_rejections_are_not_accuracy_checks():
    with pytest.raises(ValueError, match="English"):
        validate_summary("నీటి సరఫరా నిలిచిపోయింది.")
    with pytest.raises(ValueError, match="repetition"):
        validate_summary("The resident requests the officials to remove the garbage. " * 4)
    # A fluent but unsupported sentence can pass: demonstrates the validator's limit.
    validate_summary("The school has been damaged by a severe flood.")


def test_known_terminology_checks():
    with pytest.raises(ValueError, match="pension"):
        terminology_check("పింఛను అందడం లేదు.", "The plant has not grown.")
    with pytest.raises(ValueError, match="fallen"):
        terminology_check("విద్యుత్ తీగలు తెగిపడ్డాయి.", "Power lines were installed.")
    terminology_check("విద్యుత్ తీగలు తెగిపడ్డాయి.", "Electrical wires have fallen down.")
