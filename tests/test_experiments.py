import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vemana_study.common import fresh_directory
from vemana_study.inference import Summarizer
from vemana_study.prompts import EXAMPLES, build_prompt
from vemana_study.quality import check_output
from vemana_study.training import encode_pair, read_pairs, validate_splits


@pytest.mark.parametrize("language", ["te", "en"])
@pytest.mark.parametrize("examples", [0, 1, 3])
def test_prompt_preserves_whole_source_and_uses_requested_language(language, examples):
    source = "  మొదటి వాక్యం.\nరెండవ వాక్యం.  "
    prompt = build_prompt(source, language, examples)
    assert source in prompt
    assert prompt.endswith("సారాంశం:\n" if language == "te" else "English summary:\n")
    for example in EXAMPLES[:examples]:
        assert example[language] in prompt


def test_language_checks_and_hallucination_limit_are_explicit():
    source = "మా గ్రామంలో నీటి సరఫరా లేదు."
    te = check_output(source, "నీటి సరఫరాను పునరుద్ధరించాలని కోరుతున్నారు.", "te", ended=True, hit_limit=False)
    assert te["status"] == "needs_human_review"
    wrong_language = check_output(source, "The village requests water.", "te", ended=True, hit_limit=False)
    assert "target_script_mismatch" in wrong_language["errors"]
    fabricated = check_output(source, "The water supply was restored yesterday.", "en", ended=True, hit_limit=False)
    assert fabricated["status"] == "needs_human_review"
    assert fabricated["requires_human_review"] and fabricated["factual_accuracy"] is None


def test_copying_and_multiple_failures_are_recorded_without_rewriting_output():
    output = EXAMPLES[0]["en"] * 4
    result = check_output("వేరే ఫిర్యాదు", output, "en", ended=False, hit_limit=True, examples=1)
    assert result["example_overlap_ids"] == [1]
    assert "excessive_repetition" in result["errors"]
    assert "output_budget_exhausted" in result["errors"]


def test_output_directory_cannot_overwrite_runs_or_baseline(tmp_path):
    run = fresh_directory(tmp_path / "new")
    with pytest.raises(FileExistsError):
        fresh_directory(run)
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside results"):
        fresh_directory(root / "results" / "new-experiment")


def pair(id, source="మూలం", group="case-a"):
    return {"id": id, "group_id": group, "source": source, "target": "A summary.",
            "source_language": "te", "target_language": "en", "split": "train", "provenance": "synthetic test"}


def test_training_requires_explicit_provenance_and_split(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = pair("one")
    path.write_text(json.dumps(row))
    assert len(read_pairs(path, "train", "en")) == 1
    with pytest.raises(ValueError, match="expected split"):
        read_pairs(path, "validation", "en")
    del row["provenance"]
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="nonempty strings"):
        read_pairs(path, "train", "en")


def test_split_checks_reject_normalized_duplicates_group_overlap_and_benchmark():
    train = [pair("one", "మూలం ఒకటి")]
    with pytest.raises(ValueError, match="Duplicate source"):
        validate_splits(train, [pair("two", "  మూలం\nఒకటి  ", "other")])
    with pytest.raises(ValueError, match="groups overlap"):
        validate_splits(train, [pair("two", "వేరే మూలం")])
    with pytest.raises(ValueError, match="Duplicate id"):
        validate_splits(train, [pair("one", "వేరే మూలం", "other")])
    with pytest.raises(ValueError, match="Benchmark"):
        validate_splits(train, [], forbidden_sources=["మూలం ఒకటి"])
    validate_splits(train, [pair("two", "వేరే మూలం", "other")])


class CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text, add_special_tokens=True):
        return ([1] if add_special_tokens else []) + [ord(char) + 3 for char in text]


def test_training_masks_prompt_but_learns_entire_target_and_eos():
    tokenizer = CharacterTokenizer()
    row = pair("one")
    encoded = encode_pair(tokenizer, row, "en")
    prompt_ids = tokenizer.encode(build_prompt(row["source"], "en"))
    target_ids = tokenizer.encode(row["target"], add_special_tokens=False) + [2]
    assert encoded["input_ids"] == prompt_ids + target_ids
    assert encoded["labels"] == [-100] * len(prompt_ids) + target_ids
    assert encoded["attention_mask"] == [1] * len(encoded["input_ids"])
    with pytest.raises(ValueError, match="no truncation"):
        encode_pair(tokenizer, pair("long", "అ" * 1024), "en")


def test_overflow_is_returned_before_any_model_generation():
    class Tokens:
        input_ids = SimpleNamespace(shape=(1, 900))

        def to(self, device):
            return self

    summarizer = Summarizer.__new__(Summarizer)
    summarizer.device = "cpu"
    summarizer.metadata = {}
    summarizer.tokenizer = lambda *args, **kwargs: Tokens()
    # No model or torch attributes: touching either before rejection would fail.
    result = summarizer.summarize("మూలం", max_new_tokens=192)
    assert result["status"] == "context_budget_failure"
    assert result["source"] == "మూలం" and not result["source_truncated"]


def test_checkpoint_language_cannot_silently_change():
    summarizer = Summarizer.__new__(Summarizer)
    summarizer.metadata = {"summarization_training.json": {"target_language": "te", "prompt_version": "complaint-summary-v1"}}
    with pytest.raises(ValueError, match="trained target language"):
        summarizer.summarize("మూలం", target_language="en")


def test_interrupted_cli_keeps_completed_rows_and_marks_run(tmp_path, monkeypatch):
    import sys
    from vemana_study import cli

    notes = tmp_path / "notes"
    notes.mkdir()
    for name in ("a.txt", "b.txt"):
        (notes / name).write_text("మూలం", encoding="utf-8")

    class InterruptedSummarizer:
        metadata = {}

        def __init__(self, *args):
            self.calls = 0

        def summarize(self, source, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return {"source": source, "output": "A summary.", "status": "needs_human_review", "generated_tokens": 3}

    output = tmp_path / "run"
    monkeypatch.setattr(cli, "Summarizer", InterruptedSummarizer)
    monkeypatch.setattr(sys, "argv", ["vemana-summarize", "--input-dir", str(notes), "--output-dir", str(output)])
    with pytest.raises(KeyboardInterrupt):
        cli.main()
    assert json.loads((output / "run.json").read_text())["status"] == "interrupted"
    rows = (output / "results.jsonl").read_text().splitlines()
    assert len(rows) == 1 and json.loads(rows[0])["id"] == "a.txt"
