# Vemana-Base: Telugu to English, three-example GPU experiment

**The evaluation ran successfully, but the summarization task failed.** All
100 generated outputs were rejected by the existing mechanical checks. The
model frequently continued or copied the demonstrations instead of summarizing
the source complaint. This checkpoint and prompt configuration are unsuitable
for producing the requested one-page English helpline summary as tested.

The run had already completed when the user asked to stop if it was failing.
No inference process remained, and no further experiment or training was started.
The earlier queued duplicate, PBS job `46307.mgmt01`, was cancelled before the
direct SSH launch. Other jobs and processes were left running.

## Reproducible setup

| Item | Recorded value |
| --- | --- |
| Source repository | `hardiksharma6555/vemana-telugu-summarization-study` |
| Source branch / commit | `feat/summarization-experiments` / `82bedc160dc80d573249ed8e8d015f6a3b59de2a` |
| Model | `udgamlabs/Vemana-Base`, 278,214,144 parameters; base checkpoint, no fine-tuning |
| Model revision | `93cefe2594c5157e4aa8eb9908639c253652fed9` |
| Corpus | 100 synthetic, partly templated Telugu complaints, 302–317 words each; 30,721 words total |
| Target | English, one short summary per note |
| Prompt | `complaint-summary-v1`; three demonstrations |
| Decoding | Greedy, seed 42, maximum 192 new tokens, no source truncation |
| Context | 739–762 prompt tokens; prompt + requested output at most 954 / 1,024 tokens |
| Precision | FP32 |
| Host / device | Direct SSH to `gpu01`, physical GPU 5, NVIDIA H100 80GB HBM3 |
| Environment | Python 3.11.14; PyTorch 2.10.0+cu128; Transformers 5.22.2; Accelerate 1.15.0 |
| Tests before inference | 25 passed |
| Model download and verification | Repository scripts completed; pinned weights SHA-256 verified |

The requested command was executed unchanged:

```bash
.venv/bin/vemana-summarize --device cuda \
  --input-dir benchmark_100/input_notes \
  --target-language en --examples 3 \
  --output-dir outputs/gpu-en-v1
```

The model loader used local files only, with `HF_HUB_OFFLINE=1` and
`HF_HUB_DISABLE_TELEMETRY=1`. `TOKENIZERS_PARALLELISM=false` and
`RAYON_NUM_THREADS=1` were set before launch. The cluster run itself was not an
infrastructure-level air-gap test, and outbound sockets were not instrumented
for this new CLI. The model's CPU thread setting remained eight.

## Results

| Diagnostic | Count out of 100 |
| --- | ---: |
| Inputs attempted / output records preserved | 100 |
| Passed basic checks | **0** |
| Rejected by basic checks | **100** |
| Exhausted the 192-token output budget without EOS | 100 |
| Excessive repetition flagged | 69 |
| English target-script mismatch flagged | 54 |
| Seven-word demonstration overlap detected | 85 |
| Pension term missing | 15 of 15 pension notes |
| Fallen electrical-wire event missing | 1 of 1 urgent-wire note |
| Context-budget failures / source truncations | 0 / 0 |

Diagnostic categories overlap. Seven-word overlap detects copying of example
output text; it is not a complete measure of prompt copying. Neither pass/fail
counts nor keyword checks measure factual accuracy. `factual_accuracy` is left
null, and the human-review fields in the original CSV remain blank.

Two directly inspected failures illustrate the problem:

- `note_001_water.txt` reports no drinking-water supply and requests restoration.
  The generation starts: "There are two library fans are not working, while the
  books are undamaged." It then repeats the undamaged-books statement.
- `note_100_urgent_wires.txt` reports fallen electrical wires near a school gate,
  danger to children, and requests immediate repairs. The generation starts:
  "Two library fans are not working, while the books are undamaged." It omits
  the electrical hazard and repeats the books statement.

These observations support stopping this direct-use configuration. The model
card describes Vemana-Base as a Telugu continuation model rather than an
instruction-following model. Three demonstrations did not make this run a
reliable Telugu-to-English summarizer. This is not proof that supervised
adaptation cannot work; such adaptation has not been evaluated here. A separate
training set and held-out evaluation would be needed before claiming improvement.

## Timing and memory

- Operator-measured elapsed time: **227.658 seconds**, including process startup,
  model loading, generation, and monitoring overhead.
- Sum of per-note generation timings: **203.544 seconds**; mean **2.035 seconds**
  per note, including the slower first generation.
- Observed peak GPU memory for this inference process: **1,902 MiB (1.86 GiB)**.
- Observed peak process-tree resident host RAM: **1,356,820,480 bytes (1.26 GiB)**.

GPU and RAM values were sampled about once per second; shorter spikes may have
been missed. GPU values include this process's CUDA allocations and context,
and exclude other users' GPU processes. They are not PyTorch allocator peaks,
total device utilization, or EC2 measurements. See the 195 timestamped samples
in [operator_run.json](operator_run.json).

## Review and reproduce

- [human_review.csv](human_review.csv): all source/output pairs and blank review fields.
- [results.jsonl](results.jsonl): exact sources, prompts, generations, hashes,
  settings, token counts, timings, and diagnostics for all 100 notes.
- [run.json](run.json): CLI completion metadata, versions, and model manifest.
- [analysis.json](analysis.json): independently recomputed counts and integrity checks.
- [operator_run.json](operator_run.json): hardware, command, timing, sampled resource use.
- [operator_diagnostics.json](operator_diagnostics.json): original operator summary.
- [tests.log](tests.log): pre-inference test result.
- [SHA256SUMS.json](SHA256SUMS.json): hashes of the files in this results directory.

Recompute the analysis using Python's standard library, without a GPU or weights:

```bash
python repro/analyze_gpu_en_v1.py
```

For another inference run, check out the source commit, stage the CUDA dependencies
and model using the repository scripts, select an available GPU, then execute the
command above with a new output directory. The original operator wrapper is
archived as [operator_wrapper.py](operator_wrapper.py) for provenance; it assumes
placement in `operator_logs/` at the repository root, a detached checkout at the
recorded source commit, and an existing log directory. It is not a general-purpose
runner from its archive location.

These files were copied from `outputs/gpu-en-v1/` because the repository excludes
runtime outputs from version control. The original CLI result files are unchanged.
Model weights, Python environments, credentials, and unrelated scheduler logs are
not part of this published result set.
