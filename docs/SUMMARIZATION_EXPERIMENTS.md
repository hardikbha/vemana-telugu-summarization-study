# Test and adapt Vemana for summarization

These changes provide a way to run controlled experiments and train a summarization
objective. They do not turn Vemana-Base into a reliable summarizer by themselves.
The original `scripts/benchmark_vemana.py`, validators and recorded baseline are
preserved so old results remain interpretable.

## 1. Install and verify

From the repository root, use Python 3.11+ on Linux or macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,test]'
python -m pytest -q
python scripts/download_model.py
python scripts/verify_model.py
```

On a Linux CPU machine, install the CPU PyTorch build before the editable install:
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
For CUDA, install a PyTorch build compatible with your GPU first. The new inference
and training commands support `--device cpu` and `--device cuda`; macOS uses CPU.
Models are loaded from local files with `trust_remote_code=False`. These commands
do not install a process-wide socket override or provide host network isolation.

The offline tests include a tiny, randomly initialized Gemma 4 model when model
dependencies are installed: one training epoch, checkpoint save/reload, and seeded
generation. This tests the implementation, not pretrained summarization quality.

## 2. Compare the unchanged base model under new prompts

Begin with the three short pilot files. They are development cases, not a final test set.
Each output directory must be new; use new names for subsequent runs.

```bash
vemana-summarize --input-dir input_notes --target-language en --examples 0 \
  --output-dir outputs/en-zero-v1
vemana-summarize --input-dir input_notes --target-language en --examples 3 \
  --output-dir outputs/en-three-v1
vemana-summarize --input-dir input_notes --target-language te --examples 0 \
  --output-dir outputs/te-zero-v1
vemana-summarize --input-dir input_notes --target-language te --examples 3 \
  --output-dir outputs/te-three-v1
```

The demonstrations cover a bus service, library fans and service-centre opening
hours. They are hand-written synthetic illustrations and must be reviewed before
being treated as good target summaries. They are not held-out test cases. `--examples
1` and `--examples 2` use the first one or two demonstrations in that fixed order;
the sequence is logged through the exact prompt. All default runs use greedy decoding.

To try bounded sampling, hold the input, prompt and output budget fixed:

```bash
vemana-summarize --input-dir input_notes --target-language en --examples 3 \
  --decoding sample --temperature 0.7 --top-p 0.9 --seed 42 \
  --output-dir outputs/en-three-sample42-v1
```

Repeat with other seeds, retaining failures. Sampling is an ablation, not a known
fix. Seed repeatability is only expected within a compatible device/software setup.
Repetition blocking is not enabled by default: hiding repetition does not establish
source grounding.

For your own note:

```bash
vemana-summarize --input-file path/to/note.txt --target-language en \
  --output-dir outputs/my-note-v1
```

Each run saves:

- `results.jsonl`: full sources, prompts, raw decoded outputs, hashes, token counts,
  actual generation settings, failures, and example-overlap diagnostics.
- `human_review.csv`: every input/output pair with blank rubric columns for review.
- `run.json`: model metadata, package versions, settings and mechanical status counts.

Passing the checks gives `needs_human_review`, never a factual-accuracy verdict.
`rejected_basic_checks` covers language, repetition, length, completion and selected
English terminology checks. Example overlap is a diagnostic, not proof of copying:
valid summaries can share phrases. Telugu checks measure script, not linguistic or
semantic correctness. A hallucinated but fluent sentence may pass either language's
checks. All cases need source-level review; rejected cases stay in the denominator.

For the worksheet, use `1` (preserved/supported), `0` (failed), or `NA` for the
issue/action/negation/numbers columns. Enter counts for unsupported claims and
critical omissions; leave unreviewed cells blank, not zero. Record reasons in
`reviewer_notes` and apply the same rubric to every compared model. Reviewers must
read Telugu sources directly rather than judging the generated English alone.

The full prompt plus the requested output allowance must fit within 1,024 tokens.
An oversized request is recorded as `context_budget_failure`; neither source nor
requested allowance is silently shortened. Remove demonstrations, explicitly lower
`--max-new-tokens`, or design and separately evaluate a long-document workflow.

## 3. Use it from Python

```python
from vemana_study import Summarizer

summarizer = Summarizer("models/Vemana-Base", device="cpu")
result = summarizer.summarize(
    "మా గ్రామంలో రెండు రోజులుగా తాగునీరు రావడం లేదు. నీటి సరఫరాను పునరుద్ధరించండి.",
    target_language="en",
    examples=0,
)
print(result["output"])
print(result["status"])  # This is not a factual-accuracy judgment.
```

The model stays loaded across calls. The current implementation is sequential and
FP32; it does not provide a web server, concurrency management or batch-throughput
claims. Training uses the exact zero-example prompt from this interface.

## 4. Prepare supervised data

If the base model remains ungrounded, train on reviewed source/summary pairs.
Choose English or Telugu as the target for a run. Keep private data under
`data/private/` (ignored by Git). Each JSONL row needs:

```json
{"id":"case-001","group_id":"case-family-001","source_language":"te","target_language":"en","source":"తెలుగు ఫిర్యాదు","target":"Human-reviewed English summary.","provenance":"Dataset version and review/provenance reference","split":"train"}
```

Put validation rows in a different file with `split: "validation"`. Keep a third,
untouched test set for final source-level scoring. Assign every case and its
near-duplicates/paraphrases to the same `group_id` before splitting. IDs and normalized
sources must be unique; train and validation group IDs must not overlap. The loader
also rejects exact normalized source matches to this checkout's benchmark, pilots
and prompt demonstrations. These checks cannot discover every semantic near-duplicate:
the grouping and independent test split need human review.

`examples/sft_smoke/` contains just two training rows and one validation row for
testing the commands. It is **not sufficient training data or quality evidence**.
The older `docs/training_pair.example.jsonl` is intentionally rejected: its split is
`illustration_only` and its sources overlap the pilots.

Validate schema, split separation and actual token budgets without loading weights:

```bash
vemana-train --train-file examples/sft_smoke/train.jsonl \
  --validation-file examples/sft_smoke/validation.jsonl --target-language en \
  --output-dir outputs/sft-smoke --check-data-only
```

Data checking needs the downloaded tokenizer, but creates no output directory.
For Telugu targets, supply reviewed Telugu target strings and `target_language: "te"`
in the data, and pass `--target-language te`.

## 5. Fine-tune and reload

After preparing your own reviewed data:

```bash
vemana-train --train-file data/private/train.jsonl \
  --validation-file data/private/validation.jsonl --target-language en \
  --output-dir outputs/sft-en-v1 --device cuda --gradient-checkpointing \
  --epochs 3 --batch-size 1 --learning-rate 5e-5 --seed 42
```

For a plumbing-only trial, use the smoke files above, `--epochs 1`, and a fresh
output directory. `--device cpu` also works but full-model training can be slow and
requires substantially more memory than inference. No GPU-size or quality claim is
made here. These hyperparameters are starting values, not optimized recommendations.

This is full-model FP32 supervised fine-tuning, not LoRA. AdamW updates all parameters;
gradient clipping is 1.0 and weight decay is 0.01. Prompt token labels and batch
padding labels are `-100`; only target tokens and EOS contribute to the causal-LM
loss. The full training sequence must fit within 1,024 tokens; oversized examples
raise an error rather than being truncated. Validation loss is averaged over target
tokens, and the epoch with the lowest validation loss is saved to `model/`.

The output contains data-file hashes, settings, base-model metadata, package versions,
epoch losses, and the selected model and tokenizer. Upstream LICENSE/NOTICE files
are copied when present. Runs require fresh directories. There is no optimizer-state
resume support; an interrupted run must be restarted in a new directory. An epoch
checkpoint saved before interruption may exist, but check `training_run.json` for
run completion before comparing results.

Load the selected checkpoint with the same language and zero-example format:

```bash
vemana-summarize --model-path outputs/sft-en-v1/model \
  --input-dir path/to/held-out-test-notes --target-language en --examples 0 \
  --output-dir outputs/sft-en-test-v1
```

Compare that run with the unchanged base model on the same test notes and prompt.
Use [the evaluation rubric](EVALUATION_PROTOCOL.md) for source support, negation,
numbers, action preservation and critical omissions. Training/validation loss and
mechanical check counts cannot replace this evaluation. Keep a simple baseline and
a stronger model comparison under the same rubric before deciding on deployment.

## 6. Verify and push the code

```bash
python -m pytest -q
git diff --check
git status --short
git add pyproject.toml README.md CONTRIBUTING.md .gitignore .github/workflows/tests.yml \
  vemana_study tests/test_experiments.py tests/test_model_smoke.py \
  examples/sft_smoke docs/SUMMARIZATION_EXPERIMENTS.md docs/SUPERVISOR_REVIEW.md
git commit -m "Add language-aware summarization experiments and supervised training"
git push -u origin feat/summarization-experiments
```

Use your actual branch name if different. Generated outputs, downloaded models,
private training data and virtual environments are excluded by `.gitignore`.
No improvement in summarization quality should be claimed until measured on the
independent test set.
