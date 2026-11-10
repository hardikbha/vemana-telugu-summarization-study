"""Recompute GPU experiment diagnostics and verify source/prompt integrity."""
from collections import Counter
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, default=ROOT / "results/gpu-en-v1")
    args = parser.parse_args()
    folder = args.run_directory.resolve()
    rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
    run = json.loads((folder / "run.json").read_text())
    operator = json.loads((folder / "operator_run.json").read_text())
    sources = sorted((ROOT / "benchmark_100/input_notes").glob("*.txt"))
    if run["status"] != "completed" or operator["returncode"] != 0:
        raise ValueError("Experiment did not complete successfully")
    if len(rows) != len(sources) or len(rows) != 100:
        raise ValueError("Expected all 100 input/output records")
    for row, source_file in zip(rows, sources):
        source = source_file.read_text(encoding="utf-8")
        if row["id"] != source_file.name or row["source"] != source:
            raise ValueError(f"Source mismatch: {source_file.name}")
        for field in ("source", "prompt"):
            actual = hashlib.sha256(row[field].encode()).hexdigest()
            if row[f"{field}_sha256"] != actual:
                raise ValueError(f"{field} hash mismatch: {source_file.name}")
        if row["source_truncated"] or row["input_tokens"] + row["max_new_tokens"] > 1024:
            raise ValueError(f"Source truncation or context overflow: {source_file.name}")
    with (folder / "human_review.csv").open(encoding="utf-8", newline="") as handle:
        review_rows = list(csv.DictReader(handle))
    if [row["id"] for row in review_rows] != [row["id"] for row in rows]:
        raise ValueError("Review CSV does not cover the same inputs in the same order")
    statuses = dict(Counter(row["status"] for row in rows))
    if statuses != run["status_counts"] or run["notes_attempted"] != len(rows):
        raise ValueError("CLI summary does not match raw records")
    result = {
        "source_commit": operator["commit"],
        "notes_attempted": len(rows), "status_counts": statuses,
        "errors": dict(Counter(error for row in rows for error in row["errors"])),
        "notes_with_example_overlap": sum(bool(row["example_overlap_ids"]) for row in rows),
        "example_overlap_counts": dict(Counter(str(index) for row in rows for index in row["example_overlap_ids"])),
        "context_budget_failures": statuses.get("context_budget_failure", 0),
        "source_truncations": sum(row["source_truncated"] for row in rows),
        "prompt_tokens_min": min(row["input_tokens"] for row in rows),
        "prompt_tokens_max": max(row["input_tokens"] for row in rows),
        "generated_tokens_total": sum(row["generated_tokens"] for row in rows),
        "stopped_at_eos_count": sum(row["stopped_at_eos"] for row in rows),
        "generation_seconds_total": round(sum(row["seconds"] for row in rows), 3),
        "generation_seconds_mean": round(statistics.mean(row["seconds"] for row in rows), 5),
        "wall_seconds": operator["wall_seconds"],
        "observed_peak_gpu_process_memory_mib": operator["observed_peak_gpu_process_memory_mib"],
        "observed_peak_process_tree_rss_bytes": operator["observed_peak_process_tree_rss_bytes"],
        "resource_sample_count": len(operator["resource_samples"]),
        "gpu_query_error_count": sum("gpu_query_error" in sample for sample in operator["resource_samples"]),
        "integrity_checks": {"all_source_text_and_hashes_match": True,
                             "all_prompt_hashes_match": True,
                             "all_context_budgets_respected": True,
                             "review_csv_input_ids_match": True,
                             "cli_summary_matches_records": True},
        "factual_accuracy": None,
        "interpretation": "Mechanical checks and example overlap do not establish factual accuracy.",
    }
    (folder / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
