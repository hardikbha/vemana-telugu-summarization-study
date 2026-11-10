"""Run a language/prompt/decoding experiment and export a human review worksheet."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from .common import fresh_directory, versions, write_json
from .inference import Summarizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/Vemana-Base"))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-file", type=Path)
    inputs.add_argument("--input-dir", type=Path, help="Read .txt files in filename order")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    parser.add_argument("--target-language", choices=["en", "te"], default="en")
    parser.add_argument("--examples", type=int, choices=range(4), default=0)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--decoding", choices=["greedy", "sample"], default="greedy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    files = [args.input_file] if args.input_file else sorted(args.input_dir.glob("*.txt"))
    if not files:
        parser.error("No .txt inputs found")
    # Validate input before loading the model or creating an output directory.
    sources = [(file.name, file.read_text(encoding="utf-8")) for file in files]
    if any(not source.strip() for _, source in sources):
        parser.error("Every source must contain text")
    target = fresh_directory(args.output_dir)
    run = {"status": "running", "versions": versions(), "device": args.device,
           "dtype": "float32", "threads": args.threads,
           "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path) and v is not None},
           "inputs": [name for name, _ in sources], "factual_accuracy": None}
    write_json(target / "run.json", run)
    try:
        summarizer = Summarizer(args.model_path, args.device, args.threads)
        run["model"] = summarizer.metadata
        counts = Counter()
        fields = ["id", "source", "output", "status", "errors", "example_overlap_ids",
                  "issue_preserved", "action_preserved", "negation_preserved",
                  "numbers_supported", "unsupported_claims", "critical_omissions", "reviewer_notes"]
        with (target / "results.jsonl").open("w", encoding="utf-8") as results, \
                (target / "human_review.csv").open("w", encoding="utf-8", newline="") as review:
            writer = csv.DictWriter(review, fieldnames=fields)
            writer.writeheader()
            for name, source in sources:
                record = summarizer.summarize(
                    source, target_language=args.target_language, examples=args.examples,
                    max_new_tokens=args.max_new_tokens, decoding=args.decoding,
                    seed=args.seed, temperature=args.temperature, top_p=args.top_p,
                )
                record["id"] = name
                results.write(json.dumps(record, ensure_ascii=False) + "\n")
                results.flush()
                writer.writerow({key: record[key] for key in fields if key in record})
                review.flush()
                counts[record["status"]] += 1
                print(f"{name}: {record['status']} ({record['generated_tokens']} tokens)", flush=True)
        run.update(status="completed", notes_attempted=len(sources), status_counts=dict(counts))
    except KeyboardInterrupt:
        run.update(status="interrupted")
        raise
    except Exception as exc:
        run.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(target / "run.json", run)
    print(f"Review source/output pairs in {target / 'human_review.csv'}")


if __name__ == "__main__":
    main()
