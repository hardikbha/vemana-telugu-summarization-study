"""Local reusable inference. No downloads, chat template, or source truncation."""

import math
import time
from pathlib import Path

from .common import digest, model_metadata
from .prompts import PROMPT_VERSION, build_prompt
from .quality import check_output

CONTEXT_LIMIT = 1024


class Summarizer:
    def __init__(self, model_path, device="cpu", threads=8):
        if device not in {"cpu", "cuda"} or threads < 1:
            raise ValueError("Use cpu or cuda and a positive thread count")
        path = Path(model_path).resolve()
        if not path.is_dir():
            raise ValueError(f"Local model directory does not exist: {path}. Run scripts/download_model.py first.")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is not available")
        torch.set_num_threads(threads)
        self.torch, self.device = torch, device
        self.metadata = model_metadata(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, dtype=torch.float32,
            attn_implementation="sdpa",
        ).to(device).eval()
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define an EOS token")
        if self.tokenizer.pad_token_id is None:
            pad = self.model.config.pad_token_id
            self.tokenizer.pad_token_id = pad if pad is not None else self.tokenizer.eos_token_id

    def summarize(self, source, *, target_language="en", examples=0,
                  max_new_tokens=192, decoding="greedy", seed=42,
                  temperature=0.7, top_p=0.9):
        if not 1 <= max_new_tokens < CONTEXT_LIMIT:
            raise ValueError("max_new_tokens must be between 1 and 1023")
        if decoding not in {"greedy", "sample"}:
            raise ValueError("decoding must be greedy or sample")
        if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1:
            raise ValueError("temperature must be positive and finite; top_p must be in (0, 1]")
        training = self.metadata.get("summarization_training.json")
        if training and (training["target_language"] != target_language or examples != 0
                         or training["prompt_version"] != PROMPT_VERSION):
            raise ValueError("Use the trained target language, prompt version and --examples 0 for this checkpoint")
        prompt = build_prompt(source, target_language, examples)
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=False).to(self.device)
        count = tokens.input_ids.shape[-1]
        settings = {"do_sample": decoding == "sample", "num_beams": 1,
                    "repetition_penalty": 1.0, "no_repeat_ngram_size": 0}
        if decoding == "sample":
            settings.update(temperature=temperature, top_p=top_p, top_k=0)
        record = {
            "source": source, "source_sha256": digest(source), "prompt": prompt,
            "prompt_sha256": digest(prompt), "prompt_version": PROMPT_VERSION,
            "target_language": target_language, "examples": examples,
            "input_tokens": count, "max_new_tokens": max_new_tokens,
            "trained_context_limit": CONTEXT_LIMIT, "source_truncated": False,
            "decoding": settings, "seed": seed, "requires_human_review": True,
            "factual_accuracy": None,
        }
        # Preserve the requested output budget, making comparisons consistent.
        if count + max_new_tokens > CONTEXT_LIMIT:
            return dict(record, status="context_budget_failure", output="", generated_tokens=0,
                        errors=["Prompt plus requested output exceeds 1024 tokens; shorten the input or remove examples."],
                        seconds=0.0)
        torch = self.torch
        start = time.perf_counter()
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(seed)
            generated = self.model.generate(
                **tokens, max_new_tokens=max_new_tokens, **settings,
                use_cache=True, pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        if self.device == "cuda":
            torch.cuda.synchronize()
        ids = generated[0, count:].tolist()
        ended = self.tokenizer.eos_token_id in ids
        if ended:
            ids = ids[:ids.index(self.tokenizer.eos_token_id) + 1]
        output = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        record.update(output=output, generated_tokens=len(ids), stopped_at_eos=ended,
                      seconds=round(time.perf_counter() - start, 3))
        record.update(check_output(source, output, target_language, ended=ended,
                                   hit_limit=len(ids) >= max_new_tokens, examples=examples))
        return record
