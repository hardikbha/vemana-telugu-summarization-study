"""Full supervised fine-tuning with target-only loss and separate validation data."""

import argparse
import json
import math
from pathlib import Path
import shutil
import unicodedata

from .common import digest, fresh_directory, model_metadata, versions, write_json
from .inference import CONTEXT_LIMIT
from .prompts import EXAMPLES, PROMPT_VERSION, build_prompt


def normalized_source(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def read_pairs(path, split, language):
    rows = []
    required = ("id", "group_id", "source", "target", "provenance")
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key].strip() for key in required):
            raise ValueError(f"{path}:{line_number}: nonempty strings required for {required}")
        if row.get("split") != split or row.get("source_language") != "te" or row.get("target_language") != language:
            raise ValueError(f"{path}:{line_number}: expected split={split}, source_language=te, target_language={language}")
        rows.append(row)
    if not rows:
        raise ValueError(f"No pairs in {path}")
    return rows


def validate_splits(train, validation, *, forbidden_sources=()):
    """Reject explicit leakage; near-duplicate family grouping remains a human task."""
    forbidden = {normalized_source(source) for source in forbidden_sources}
    seen_ids, seen_sources = set(), set()
    for row in train + validation:
        source = normalized_source(row["source"])
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate id: {row['id']}")
        if source in seen_sources:
            raise ValueError(f"Duplicate source across or within splits: {row['id']}")
        if source in forbidden:
            raise ValueError(f"Benchmark, pilot or prompt demonstration cannot be used for training/validation: {row['id']}")
        seen_ids.add(row["id"])
        seen_sources.add(source)
    overlap = {r["group_id"] for r in train} & {r["group_id"] for r in validation}
    if overlap:
        raise ValueError(f"Case/near-duplicate groups overlap across splits: {sorted(overlap)}")


def encode_pair(tokenizer, row, language):
    # Explicit token boundary: inference sees exactly these same prompt tokens.
    # No target BOS; append one EOS, and mask ALL prompt tokens from the loss.
    prompt_ids = tokenizer.encode(build_prompt(row["source"], language), add_special_tokens=True)
    target_ids = tokenizer.encode(row["target"], add_special_tokens=False)
    if not target_ids or tokenizer.eos_token_id is None:
        raise ValueError(f"{row['id']}: target tokens and tokenizer EOS are required")
    if target_ids[-1] != tokenizer.eos_token_id:
        target_ids.append(tokenizer.eos_token_id)
    if len(prompt_ids) + len(target_ids) > CONTEXT_LIMIT:
        raise ValueError(f"{row['id']}: prompt plus target exceeds {CONTEXT_LIMIT} tokens; no truncation performed")
    return {"input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids}


def collate(rows, pad_token_id):
    import torch

    length = max(len(row["input_ids"]) for row in rows)
    result = {}
    for key, padding in (("input_ids", pad_token_id), ("attention_mask", 0), ("labels", -100)):
        result[key] = torch.tensor([row[key] + [padding] * (length - len(row[key])) for row in rows], dtype=torch.long)
    return result


def fit(model, tokenizer, train, validation, output, *, language, epochs=3,
        batch_size=1, learning_rate=5e-5, seed=42, device="cpu", metadata=None):
    """Train a loaded causal LM and save the epoch with lowest target-token validation loss."""
    import torch
    from torch.utils.data import DataLoader
    from functools import partial

    torch.manual_seed(seed)
    encoded_train = [encode_pair(tokenizer, row, language) for row in train]
    encoded_validation = [encode_pair(tokenizer, row, language) for row in validation]
    batch = partial(collate, pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(encoded_train, batch_size=batch_size, shuffle=True, collate_fn=batch,
                              generator=torch.Generator().manual_seed(seed))
    validation_loader = DataLoader(encoded_validation, batch_size=batch_size, collate_fn=batch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    best = float("inf")
    history = []
    output = Path(output)
    for epoch in range(1, epochs + 1):
        model.train()
        training_loss, training_tokens = 0.0, 0
        for values in train_loader:
            values = {key: value.to(device) for key, value in values.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model(**values, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            count = int((values["labels"][:, 1:] != -100).sum())
            training_loss += loss.item() * count
            training_tokens += count
        model.eval()
        validation_loss, validation_tokens = 0.0, 0
        with torch.inference_mode():
            for values in validation_loader:
                values = {key: value.to(device) for key, value in values.items()}
                loss = model(**values, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite validation loss")
                count = int((values["labels"][:, 1:] != -100).sum())
                validation_loss += loss.item() * count
                validation_tokens += count
        metrics = {"epoch": epoch, "train_target_token_loss": training_loss / training_tokens,
                   "validation_target_token_loss": validation_loss / validation_tokens}
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        write_json(output / "history.json", history)
        if metrics["validation_target_token_loss"] < best:
            best = metrics["validation_target_token_loss"]
            checkpoint = output / "model"
            model.save_pretrained(checkpoint, safe_serialization=True)
            tokenizer.save_pretrained(checkpoint)
            write_json(checkpoint / "summarization_training.json", {
                **(metadata or {}), "prompt_version": PROMPT_VERSION, "target_language": language,
                "objective": "causal_lm_target_only", "prompt_labels": -100,
                "trained_context_limit": CONTEXT_LIMIT, "selected_epoch": epoch,
                "validation_target_token_loss": best, "factual_accuracy": None,
            })
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/Vemana-Base"))
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--validation-file", type=Path, required=True)
    parser.add_argument("--target-language", choices=["en", "te"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--check-data-only", action="store_true", help="Validate splits and token budgets without loading weights")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads) < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("epochs, batch-size, threads and learning-rate must be positive and finite")
    if not args.model_path.is_dir():
        parser.error("Local model directory is missing; run scripts/download_model.py first")
    train = read_pairs(args.train_file, "train", args.target_language)
    validation = read_pairs(args.validation_file, "validation", args.target_language)
    root = Path(__file__).resolve().parents[1]
    forbidden = [example["source"] for example in EXAMPLES]
    for folder in (root / "benchmark_100/input_notes", root / "input_notes"):
        forbidden.extend(path.read_text(encoding="utf-8") for path in folder.glob("*.txt"))
    validate_splits(train, validation, forbidden_sources=forbidden)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=False)
    for row in train + validation:
        encode_pair(tokenizer, row, args.target_language)
    if args.check_data_only:
        print(f"Validated {len(train)} train / {len(validation)} validation pairs and their token budgets. No training performed.")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    target = fresh_directory(args.output_dir)
    metadata = {
        "status": "running", "base_model": model_metadata(args.model_path), "versions": versions(),
        "train_file_sha256": digest(args.train_file.read_text(encoding="utf-8")),
        "validation_file_sha256": digest(args.validation_file.read_text(encoding="utf-8")),
        "train_count": len(train), "validation_count": len(validation),
        "settings": {k: v for k, v in vars(args).items() if not isinstance(v, Path)},
        "optimizer": "AdamW", "weight_decay": 0.01, "max_grad_norm": 1.0,
        "dtype": "float32", "training_type": "full_fine_tuning",
    }
    write_json(target / "training_run.json", metadata)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, local_files_only=True, trust_remote_code=False,
            dtype=torch.float32, attn_implementation="sdpa",
        ).to(args.device)
        if tokenizer.pad_token_id is None:
            pad = model.config.pad_token_id
            tokenizer.pad_token_id = pad if pad is not None else tokenizer.eos_token_id
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Checkpoint metadata describes the selected epoch, not an in-progress run.
        checkpoint_metadata = {k: v for k, v in metadata.items() if k != "status"}
        fit(model, tokenizer, train, validation, target, language=args.target_language,
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            seed=args.seed, device=args.device, metadata=checkpoint_metadata)
        for pattern in ("LICENSE*", "NOTICE*"):
            for notice in args.model_path.glob(pattern):
                if notice.is_file():
                    shutil.copy2(notice, target / "model" / notice.name)
        metadata["status"] = "completed"
    except KeyboardInterrupt:
        metadata.update(status="interrupted")
        raise
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(target / "training_run.json", metadata)
    print(f"Selected checkpoint: {target / 'model'}. Validation loss is not summarization accuracy.")


if __name__ == "__main__":
    main()
