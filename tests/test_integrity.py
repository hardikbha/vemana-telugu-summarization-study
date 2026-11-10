import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_all_synthetic_source_files_and_labels_match_manifest():
    dataset = json.loads((ROOT / "benchmark_100/dataset.json").read_text())
    assert dataset["synthetic"] and dataset["templated"]
    assert len(dataset["notes"]) == 100
    total = 0
    for row in dataset["notes"]:
        data = (ROOT / "benchmark_100/input_notes" / row["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["sha256"]
        count = len(data.decode().split())
        assert count == row["words"] and 300 <= count <= 320
        total += count
    assert total == dataset["word_count_total"] == 30721
    assert sum(dataset["source_category_counts"].values()) == 100
    assert sum(row["urgent"] for row in dataset["notes"]) == 1


def test_recorded_outputs_cover_all_sources_without_truncation():
    result = json.loads((ROOT / "results/vemana_baseline/results.json").read_text())
    dataset = json.loads((ROOT / "benchmark_100/dataset.json").read_text())
    rows = {row["id"]: row for row in result["long_note_runs"]}
    assert set(rows) == {row["id"] for row in dataset["notes"]}
    for item in dataset["notes"]:
        row = rows[item["id"]]
        source = (ROOT / "benchmark_100/input_notes" / item["file"]).read_text().strip()
        assert row["source"] == source
        assert row["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
        assert not row["source_truncated"]
        assert row["input_tokens"] + row["max_new_tokens"] <= 1024
        separate = json.loads((ROOT / "results/vemana_baseline" / (row["id"] + ".json")).read_text())
        assert separate == row
    assert result["network_attempts"] == []


def test_reported_counts_are_derived_from_raw_results():
    folder = ROOT / "results/vemana_baseline"
    result = json.loads((folder / "results.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    rows = result["long_note_runs"]
    assert sum(row["status"] == "accepted_basic_checks" for row in rows) == result["accepted_basic_checks"] == analysis["accepted_basic_checks"]
    assert sum("garbage has not been collected" in row["output"].lower() for row in rows) == analysis["copied_garbage_example_phrase"]
    assert result["notes_attempted"] == 100
    assert result["base_model"] and not result["instruction_tuned"]
