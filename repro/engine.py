"""Local 0.8B Telugu-to-English summary engine with CPU/CUDA selection."""
import hashlib
import json
import re
from pathlib import Path

VERSION = "telugu-to-english-v4-scale"
SYSTEM = (
    "Summarize the Telugu note in English. Output only the facts in 1 to 3 short English sentences. "
    "Preserve negation, durations, events and requests. Do not invent or change facts. "
    "Do not describe the user's request or your task."
)
GLOSSARY = {
    "తాగునీరు": "drinking water", "నీటి సరఫరా": "water supply", "పింఛను": "pension payment",
    "వృద్ధాప్య": "old-age", "విద్యుత్ తీగలు": "electrical wires", "తెగిపడ్డాయి": "have broken and fallen down",
    "పాఠశాల": "school", "గేటు": "gate", "పిల్లలకు ప్రమాదం": "danger to children",
    "అందడం లేదు": "has not been received", "రావడం లేదు": "is not being supplied",
    "పునరుద్ధరించండి": "please restore", "విడుదల చేయండి": "please release the payment",
    "మరమ్మతులు": "repairs", "వెంటనే": "immediately", "మూడు నెలలుగా": "for three months",
    "రెండు రోజులుగా": "for two days",
}


class SummaryError(ValueError):
    def __init__(self, message, attempts):
        super().__init__(message)
        self.attempts = attempts


def terminology_check(source, summary):
    lower = summary.lower()
    if "పింఛను" in source and ("pension" not in lower or "plant" in lower):
        raise ValueError("Summary did not preserve the pension term from the note.")
    if "విద్యుత్ తీగలు" in source and "తెగిపడ్డాయి" in source:
        if not any(word in lower for word in ["fallen", "fell", "broken", "downed"]) or any(word in lower for word in ["erected", "installed"]):
            raise ValueError("Summary did not preserve the broken/fallen electrical-wire event.")


def validate_summary(text, max_words=120):
    if not text or len(text.split()) > max_words:
        raise ValueError("The model did not produce a short English summary.")
    letters = [char for char in text if char.isalpha()]
    latin = sum("a" <= char.lower() <= "z" for char in letters)
    if not letters or latin / len(letters) < 0.8:
        raise ValueError("The model output is not predominantly English.")
    if re.search(r"\b(user (has )?requested|telugu (text|note)|specific constraints|as an ai|english summary:)\b", text, flags=re.I):
        raise ValueError("The model described the task instead of summarizing the note.")
    words = text.lower().split()
    phrases = [tuple(words[i:i+7]) for i in range(max(0, len(words)-6))]
    if phrases and max(phrases.count(phrase) for phrase in set(phrases)) >= 3:
        raise ValueError("The model output contains excessive repetition.")


def partition(text, budget, token_count):
    while text:
        if token_count(text) <= budget:
            yield text
            return
        low, high = 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if token_count(text[:middle]) <= budget:
                low = middle
            else:
                high = middle - 1
        boundary = max(text.rfind("\n", 0, low), text.rfind(" ", 0, low))
        end = boundary + 1 if boundary > low // 2 else low
        while end > 1 and token_count(text[:end]) > budget:
            end -= 1
        yield text[:end]
        text = text[end:]


class Summarizer:
    def __init__(self, model_path, threads=4, dtype="bf16", device="cpu", chunk_tokens=1500):
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but no working CUDA runtime/GPU is available.")
        if not 128 <= chunk_tokens <= 2800:
            raise ValueError("Chunk budget must be between 128 and 2800 tokens.")
        self.device, self.chunk_tokens = device, chunk_tokens
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        path = Path(model_path).resolve()
        if not (path / "model_manifest.json").exists():
            raise ValueError("Local verified model missing. This application does not download weights.")
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForImageTextToText.from_pretrained(
            path, local_files_only=True, trust_remote_code=False, device_map=device,
            dtype=torch.bfloat16 if dtype == "bf16" else torch.float32, attn_implementation="sdpa").eval()
        self.identity = hashlib.sha256((path / "model_manifest.json").read_bytes()).hexdigest()
        self.calls = 0
        self.context = 4096
        self.dtype, self.threads = dtype, threads

    def count(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, text, aggregate=False):
        task = ("Combine the following English summaries into one short English overview. Every record was selected. "
                "Group recurring issues without merging separate incidents or transferring durations. "
                "Preserve potentially serious issues and the main themes across the summaries. "
                "Return an English briefing of no more than 180 words; do not calculate or invent counts.\n\n" if aggregate else
                "Telugu note:\n")
        prompt = task + text
        if not aggregate:
            hints = {key: value for key, value in GLOSSARY.items() if key in text}
            if hints:
                prompt += "\nVocabulary for this note (use these meanings): " + json.dumps(hints, ensure_ascii=False)
            prompt += "\nEnglish summary:"
        errors, responses = [], []
        for attempt in range(2):
            if attempt:
                prompt += "\nReturn only a brief English summary. Do not reproduce the source note."
            system = ("Write an English briefing using only the supplied summary facts. Keep recurring themes and exceptional dangers. "
                      "Do not invent counts, merge separate incidents, or claim unresolved issues are resolved. "
                      "Use at most 180 words. Output the briefing only." if aggregate else SYSTEM)
            messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            inputs = self.tokenizer(formatted, return_tensors="pt").to(self.device)
            max_output = 768 if aggregate else 384
            if inputs.input_ids.shape[-1] + max_output > self.context:
                raise ValueError("Input exceeds budget; no note text was silently truncated.")
            self.calls += 1
            with self.torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=max_output, do_sample=False,
                                             pad_token_id=self.tokenizer.eos_token_id, use_cache=True)
            new = output[0, inputs.input_ids.shape[-1]:]
            decoded = self.tokenizer.decode(new, skip_special_tokens=True).strip()
            decoded = re.sub(r"<think>.*?</think>", "", decoded, flags=re.S).strip()
            responses.append(decoded)
            if len(new) >= max_output:
                errors.append("Model output reached its token limit.")
                continue
            try:
                validate_summary(decoded, max_words=220 if aggregate else 120)
                if not aggregate:
                    terminology_check(text, decoded)
                return {"summary": decoded, "attempts": responses, "retry_errors": errors}
            except ValueError as exc:
                errors.append(str(exc))
        raise SummaryError("Model could not produce a complete English summary: " + "; ".join(errors), responses)

    def reduce(self, texts):
        current, audit = list(texts), []
        for level in range(12):
            batches, batch = [], []
            for text in current:
                candidate = batch + [text]
                if batch and self.count("\n".join(candidate)) > 1500:
                    batches.append(batch)
                    batch = []
                batch.append(text)
            if batch:
                batches.append(batch)
            reduced = []
            for values in batches:
                result = self.generate("\n\n".join(values), aggregate=True)
                audit.append({"level": level, "input_count": len(values), **result})
                reduced.append(result["summary"])
            if len(reduced) == 1:
                return reduced[0], audit
            if self.count("\n".join(reduced)) >= self.count("\n".join(current)):
                raise ValueError("The summary hierarchy did not reduce input size.")
            current = reduced
        raise ValueError("Summary hierarchy exceeded 12 levels.")

    def summarize(self, text):
        chunks = list(partition(text, self.chunk_tokens, self.count))
        results = [self.generate(chunk) for chunk in chunks]
        if len(results) == 1:
            return {"summary": results[0]["summary"], "chunks": results, "reduction": []}
        summary, audit = self.reduce([result["summary"] for result in results])
        return {"summary": summary, "chunks": results, "reduction": audit}
