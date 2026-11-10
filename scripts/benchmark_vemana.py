"""Actual offline Vemana-Base completion tests; no chat template or fine-tuning.

Uses three short pilot notes and the same 100 long Telugu notes. The model card's
1,024-token trained length is respected, including output. Raw outputs are never
rewritten or translated by a second model.
"""
import argparse
import hashlib
import json
import os
import resource
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CUSTOM_PROMPT = None


def prompt(text, mode):
    if CUSTOM_PROMPT is not None:
        return CUSTOM_PROMPT.replace("{source}", text)
    if mode == "direct":
        return ("Summarize the following Telugu complaint in English in 1 to 3 short sentences. "
                "Preserve the problem and requested action. Do not invent facts.\n\nTelugu note:\n" + text + "\n\nEnglish summary:")
    return ("Telugu: మా వీధిలో వారం రోజులుగా చెత్తను సేకరించడం లేదు. దయచేసి శుభ్రం చేయండి.\n"
            "English summary: Garbage has not been collected in the street for one week. The resident requests cleaning.\n\n"
            "Telugu: " + text + "\nEnglish summary:")


def output_budget(input_tokens, requested_tokens, context_limit=1024):
    """Reserve output space without ever truncating source text."""
    return max(0, min(requested_tokens, context_limit-input_tokens))


def main():
    global CUSTOM_PROMPT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-pilots", action="store_true", help="Reuse identical completed pilot outputs from this model's prior setup run")
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Vemana-Base")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/latest")
    parser.add_argument("--limit", type=int, default=100, help="Evaluate the first N long notes, without truncating any note")
    parser.add_argument("--input-file", type=Path, help="Evaluate one UTF-8 Telugu note instead of the 100-note dataset")
    parser.add_argument("--prompt-mode", choices=["direct", "fewshot"], default="fewshot")
    parser.add_argument("--prompt-file", type=Path, help="Optional plain completion template containing {source}")
    parser.add_argument("--skip-pilots", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    if args.prompt_file:
        CUSTOM_PROMPT = args.prompt_file.read_text(encoding="utf-8")
        if "{source}" not in CUSTOM_PROMPT:
            parser.error("The prompt template must contain {source}; source text is never silently omitted.")
    attempts = []
    original = socket.socket.connect
    def deny(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            attempts.append(str(address))
            raise RuntimeError("Network access disabled during model evaluation.")
        return original(sock, address)
    socket.socket.connect = deny
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from engine import validate_summary, terminology_check
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("No working CUDA device available.")
    torch.set_num_threads(args.threads)
    target = args.output_dir.resolve()
    target.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path.resolve()
    manifest = json.loads((model_path / "model_manifest.json").read_text())
    if target.is_relative_to((ROOT / "results").resolve()):
        parser.error("Choose an output directory outside results/ so the recorded baseline remains immutable.")
    if args.reuse_pilots and not args.skip_pilots:
        previous = target / "results.json"
        if not previous.exists() or json.loads(previous.read_text()).get("revision") != manifest["revision"]:
            parser.error("Pilot reuse requires previous run metadata for this exact model revision.")
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=False,
                                               device_map=args.device, dtype=torch.float32,
                                               attn_implementation="sdpa").eval()
    load_seconds = time.perf_counter()-started
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(model.config.pad_token_id)
    history = []
    def run(text, mode, id):
        formatted = prompt(text, mode)
        tokens = tokenizer(formatted, return_tensors="pt").to(args.device)
        n_input = tokens.input_ids.shape[-1]
        output_limit = output_budget(n_input, args.max_new_tokens)
        record = {"id": id, "mode": mode, "source": text,
                  "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "prompt_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
                  "input_tokens": n_input, "max_new_tokens": output_limit,
                  "trained_context_limit": 1024, "source_truncated": False}
        if output_limit < 64:
            record.update({"status": "context_budget_failure", "output": "", "seconds": 0})
        else:
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**tokens, max_new_tokens=output_limit, do_sample=False,
                                           use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                           eos_token_id=tokenizer.eos_token_id)
            if args.device == "cuda":
                torch.cuda.synchronize()
            generated = generated[0, n_input:]
            output = tokenizer.decode(generated, skip_special_tokens=True).strip()
            ended = bool(len(generated) and int(generated[-1]) == tokenizer.eos_token_id)
            errors = []
            if not ended and len(generated) >= output_limit:
                errors.append("Output reached the token budget without ending.")
            try:
                validate_summary(output)
                terminology_check(text, output)
            except ValueError as exc:
                errors.append(str(exc))
            record.update({"output": output, "generated_tokens": len(generated),
                           "seconds": round(time.perf_counter()-start, 3),
                           "status": "accepted_basic_checks" if not errors else "rejected",
                           "errors": errors, "output_words": len(output.split()),
                           "stopped_at_eos": ended})
        history.append(record)
        (target / (id + ".json")).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key:record.get(key) for key in ["id", "status", "input_tokens", "generated_tokens", "seconds", "errors"]}), flush=True)
        return record
    # Base-model completion sanity check, separate from summarization quality.
    if args.skip_pilots:
        sanity = "Skipped by explicit request."
    elif args.reuse_pilots:
        sanity = (target / "telugu_continuation.txt").read_text().strip()
    else:
        torch.manual_seed(42)
        seed = tokenizer("తెలుగు భాష", return_tensors="pt").to(args.device)
        with torch.inference_mode():
            continuation = model.generate(**seed, max_new_tokens=80, do_sample=True, temperature=0.8,
                                          top_p=0.95, pad_token_id=tokenizer.pad_token_id)
        sanity = tokenizer.decode(continuation[0], skip_special_tokens=True)
        (target / "telugu_continuation.txt").write_text(sanity + "\n")
    pilot_files = [] if args.skip_pilots else ["01_water_supply.txt", "02_pension_delay.txt", "03_electricity_danger.txt"]
    for index, filename in enumerate(pilot_files, 1):
        source = (ROOT / "input_notes" / filename).read_text().strip()
        for mode in ["direct", "fewshot"]:
            id = f"pilot_{index}_{mode}"
            if args.reuse_pilots:
                record = json.loads((target / (id + ".json")).read_text())
                assert record["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
                assert record["prompt_sha256"] == hashlib.sha256(prompt(source, mode).encode()).hexdigest()
                history.append(record)
            else:
                run(source, mode, id)
    dataset = json.loads((ROOT / "benchmark_100/dataset.json").read_text())
    dataset["notes"] = dataset["notes"][:args.limit]
    if args.input_file:
        dataset["notes"] = [{"id": "N_custom", "file": args.input_file.name}]
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    tokenizer.padding_side = "left"
    for offset in range(0, len(dataset["notes"]), args.batch_size):
        items = dataset["notes"][offset:offset+args.batch_size]
        sources = [((args.input_file.resolve() if args.input_file else ROOT / "benchmark_100/input_notes" / item["file"]).read_text(encoding="utf-8").strip()) for item in items]
        formatted = [prompt(source, args.prompt_mode) for source in sources]
        counts = [len(tokenizer.encode(value)) for value in formatted]
        # Pad for a shared model forward pass without changing any source text.
        # Every sequence, including its padding and output, stays within 1,024.
        limit = output_budget(max(counts), args.max_new_tokens)
        if limit < 64:
            for item, source in zip(items, sources):
                run(source, args.prompt_mode, item["id"])
            continue
        tokens = tokenizer(formatted, padding=True, return_tensors="pt").to(args.device)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(**tokens, max_new_tokens=limit, do_sample=False, use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        if args.device == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter()-start
        for index, (item, source, value, n_input) in enumerate(zip(items, sources, formatted, counts)):
            output_tokens = generated[index, tokens.input_ids.shape[-1]:].tolist()
            ended = tokenizer.eos_token_id in output_tokens
            if ended:
                output_tokens = output_tokens[:output_tokens.index(tokenizer.eos_token_id)+1]
            output = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            errors = []
            if not ended and len(output_tokens) >= limit:
                errors.append("Output reached the token budget without ending.")
            try:
                validate_summary(output)
                terminology_check(source, output)
            except ValueError as exc:
                errors.append(str(exc))
            record = {"id":item["id"],"mode":args.prompt_mode,"source":source,"output":output,
                      "source_sha256":hashlib.sha256(source.encode()).hexdigest(),
                      "prompt_sha256":hashlib.sha256(value.encode()).hexdigest(),
                      "input_tokens":n_input,"max_new_tokens":limit,"trained_context_limit":1024,
                      "source_truncated":False,"generated_tokens":len(output_tokens),
                      "batch_size":len(items),"batch_wall_seconds":round(seconds,3),
                      "seconds":round(seconds/len(items),3),"seconds_note":"Amortized batch wall time, not individual latency.",
                      "status":"accepted_basic_checks" if not errors else "rejected","errors":errors,
                      "output_words":len(output.split()),"stopped_at_eos":ended}
            history.append(record)
            (target/(item["id"]+".json")).write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
        print(json.dumps({"batch_completed":offset+len(items),"batch_size":len(items),"batch_seconds":round(seconds,3)}),flush=True)
    full = [row for row in history if row["id"].startswith("N")]
    summary = {"model": manifest["repo_id"], "revision": manifest["revision"], "parameters": model.num_parameters(),
               "license": "Apache-2.0", "base_model": True, "instruction_tuned": False,
               "device": args.device, "dtype": "float32", "threads": args.threads, "long_note_batch_size": args.batch_size,
               "pilot_outputs_reused": args.reuse_pilots,
               "pilots_skipped": args.skip_pilots, "prompt_mode": args.prompt_mode,
               "custom_prompt_template": CUSTOM_PROMPT,
               "tokenizer_has_chat_template": bool(tokenizer.chat_template),
               "load_seconds": round(load_seconds, 3), "elapsed_seconds": round(time.perf_counter()-started, 3),
               "peak_process_memory_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2, 3),
               "notes_attempted": len(full), "accepted_basic_checks": sum(r["status"] == "accepted_basic_checks" for r in full),
               "context_budget_failures": sum(r["status"] == "context_budget_failure" for r in full),
               "rejected_outputs": sum(r["status"] == "rejected" for r in full),
               "english_requirement": True, "trained_context_limit": 1024,
               "network_attempts": attempts, "pilot_runs": [row for row in history if row['id'].startswith('pilot_')], "long_note_runs": full,
               "telugu_continuation": sanity,
               "method": f"Plain completion; {len([row for row in history if row['id'].startswith('pilot_')])} pilot records; "
                         f"{len(full)} evaluated input(s), {'custom template' if CUSTOM_PROMPT is not None else args.prompt_mode} prompt, "
                         "no chat template or silent source truncation. No training performed.",
               "limits": "Synthetic, partly templated sources; basic acceptance is not semantic accuracy. Prompting differs from Qwen because this is a base model. Different output budget and architecture prevent a controlled benchmark ranking."}
    assert not attempts
    (target / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in ["pilot_runs", "long_note_runs", "telugu_continuation"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
