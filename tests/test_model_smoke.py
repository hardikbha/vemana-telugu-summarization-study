"""Offline architecture/optimizer/save/reload checks; no pretrained weights required."""

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from vemana_study.inference import Summarizer
from vemana_study.cli import main as summarize_main
from vemana_study.training import collate, encode_pair, main as train_main


def test_real_gemma_training_checkpoint_reload_and_generation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3,
                                   "A": 4, "summary": 5, ".": 6}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", bos_token="<s>", eos_token="</s>", unk_token="<unk>",
    )
    config = json.loads((Path(__file__).resolve().parents[1] / "model_metadata/Vemana-Base/config.json").read_text())
    config.update(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=8, global_head_dim=8,
                  per_layer_config={"0": {"head_dim": 8}, "1": {"head_dim": 8}},
                  layer_types=["full_attention"] * 2, vocab_size_per_layer_input=32)
    model = transformers.AutoModelForCausalLM.from_config(transformers.Gemma4TextConfig(**config))
    original = next(model.parameters()).detach().clone()
    base = tmp_path / "base"
    model.save_pretrained(base)
    tokenizer.save_pretrained(base)
    train = [{"id": "train-one", "group_id": "train-one", "source": "మొదటి ఫిర్యాదు", "target": "A summary.",
              "source_language": "te", "target_language": "en", "split": "train", "provenance": "synthetic test"}]
    validation = [{"id": "val-one", "group_id": "val-one", "source": "మరొక ఫిర్యాదు", "target": "A summary.",
                   "source_language": "te", "target_language": "en", "split": "validation", "provenance": "synthetic test"}]
    train_path, val_path = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    train_path.write_text(json.dumps(train[0]))
    val_path.write_text(json.dumps(validation[0]))
    trained = tmp_path / "trained"
    monkeypatch.setattr(sys, "argv", ["vemana-train", "--model-path", str(base),
                        "--train-file", str(train_path), "--validation-file", str(val_path),
                        "--target-language", "en", "--output-dir", str(trained),
                        "--epochs", "1", "--threads", "1", "--gradient-checkpointing"])
    train_main()
    history = json.loads((trained / "history.json").read_text())
    assert torch.isfinite(torch.tensor(history[0]["validation_target_token_loss"]))
    assert json.loads((trained / "training_run.json").read_text())["status"] == "completed"
    assert (trained / "model" / "model.safetensors").is_file()
    loaded = Summarizer(trained / "model", threads=1)
    assert not torch.equal(original, next(loaded.model.parameters()).detach())
    result = loaded.summarize("మరొక ఫిర్యాదు", max_new_tokens=8)
    assert result["generated_tokens"] <= 8
    assert result["requires_human_review"] and result["factual_accuracy"] is None
    assert result["status"] in {"needs_human_review", "rejected_basic_checks"}
    second = loaded.summarize("మరొక ఫిర్యాదు", max_new_tokens=8, decoding="sample", seed=17)
    repeat = loaded.summarize("మరొక ఫిర్యాదు", max_new_tokens=8, decoding="sample", seed=17)
    assert second["output"] == repeat["output"]
    encoded = encode_pair(tokenizer, train[0], "en")
    shorter = encode_pair(tokenizer, dict(train[0], source="మూలం"), "en")
    padded = collate([encoded, shorter], tokenizer.pad_token_id)
    assert torch.all(padded["labels"][padded["attention_mask"] == 0] == -100)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "short.txt").write_text("మరొక ఫిర్యాదు", encoding="utf-8")
    (notes / "long.txt").write_text("మూలం " * 1100, encoding="utf-8")
    predictions = tmp_path / "predictions"
    monkeypatch.setattr(sys, "argv", ["vemana-summarize", "--model-path", str(trained / "model"),
                        "--input-dir", str(notes), "--output-dir", str(predictions),
                        "--max-new-tokens", "8", "--threads", "1"])
    summarize_main()
    run = json.loads((predictions / "run.json").read_text())
    records = [json.loads(line) for line in (predictions / "results.jsonl").read_text().splitlines()]
    assert run["status"] == "completed" and run["notes_attempted"] == 2
    assert run["status_counts"]["context_budget_failure"] == 1
    assert records[0]["id"] == "long.txt" and not records[0]["source_truncated"]
    assert "long.txt" in (predictions / "human_review.csv").read_text()
    with pytest.raises(FileExistsError):
        summarize_main()
