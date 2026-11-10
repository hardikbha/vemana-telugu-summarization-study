# Vemana: Telugu-to-English summarization study

This repository makes an actual direct-use experiment with
[`udgamlabs/Vemana-Base`](https://huggingface.co/udgamlabs/Vemana-Base) reproducible
and easy to review with a supervisor. It includes the code, prompts, 100 synthetic
Telugu source notes, raw generations, checks, timings, memory measurements, and reports.

**Current finding:** the tested base checkpoint did not function as a reliable
Telugu-to-English summarizer out of the box. This is a starting point for research,
not a production summarization service or a claim that adaptation cannot work.

## Latest GPU experiment: three examples

The requested run of commit `82bedc1` completed on an H100 GPU on 10 November
2026, using all 100 Telugu notes and three prompt examples. **All 100 outputs
failed basic checks**; 85 contained a seven-word overlap with a demonstration.
The urgent fallen-wire complaint was missed. This is not a factual-accuracy score.

See [the GPU results and review](results/gpu-en-v1/README.md),
[all source/output pairs](results/gpu-en-v1/human_review.csv), and
[raw generations](results/gpu-en-v1/results.jsonl). The run took 227.658 seconds;
sampled peak GPU process memory was 1,902 MiB (1.86 GiB), with FP32 weights.
These are measurements for Vemana-Base, not Qwen3.5-0.8B or an EC2 benchmark.

A subsequent [three-note Telugu-output pilot](results/gpu-te-pilot-v1/README.md)
also failed all three cases. The outputs used Telugu script but copied prompt
instructions and examples; the urgent electrical hazard was again omitted.
This small pilot does not establish a full-corpus Telugu accuracy score.

## New experiments and supervised training

The `vemana_study` package adds a reusable `Summarizer`, English/Telugu output,
zero to three varied demonstrations, seeded decoding, a human-review worksheet,
and a full supervised fine-tuning command with target-only loss. These are
experimental tools, **not evidence that the base model now summarizes reliably**.
The historical runner and recorded baseline remain unchanged.

See [the runnable experiment and training guide](docs/SUMMARIZATION_EXPERIMENTS.md)
for installation, comparison runs, the training-data schema, and checkpoint use.

## Start here for review

- [One-page evaluation report](results/vemana_baseline/Vemana_Base_Evaluation.pdf)
- [Editable Word report](results/vemana_baseline/Vemana_Base_Evaluation.docx)
- [Results, examples, and methodological limits](docs/RESULTS.md)
- [Questions and next experiments for a supervisor](docs/SUPERVISOR_REVIEW.md)
- [All raw outputs as CSV](results/vemana_baseline/all_outputs.csv)
- [Full recorded run, sources and hashes](results/vemana_baseline/results.json)

## What was tested

| Item | Recorded setup/result |
| --- | --- |
| Task | Telugu input → short English summary |
| Checkpoint | Vemana-Base, 278,214,144 parameters, base model |
| Revision | `93cefe2594c5157e4aa8eb9908639c253652fed9` |
| Pilots | Three short notes, direct and one-example completion prompts |
| Longer test | 100 synthetic notes, 302–317 words each; 30,721 words total |
| Context | Prompt plus output within the model card's 1,024-token trained length |
| Decoding | Greedy; maximum 192 generated tokens; no chat template |
| Hardware mode | CPU FP32; eight threads; batches of eight for long notes |
| Basic checks passed | **0/100** long-note outputs; this is **not a factual-accuracy score** |
| Example-copy diagnostic | 100/100 included the garbage-example opening phrase |
| Peak resident process RAM | 2.188 GiB in the recorded batched CPU run |
| Run wall time | 183.269 s; includes model loading, excludes reused pilot time |

The model card says this checkpoint continues Telugu text rather than following
instructions. Direct prompting produced unrelated or repetitive content. One-example
prompting caused example copying. All source text fit without truncation. Neither
model fine-tuning nor systematic prompt/decoding optimization has been performed.

Synthetic notes are partly templated and are **not actual citizen records**. Basic
English/completion checks and keyword diagnostics do not prove semantic correctness.
See [the full evaluation protocol](docs/EVALUATION_PROTOCOL.md).

## Install for CPU experiments

Use Python 3.11+ on Linux. The recorded run used CPython 3.11.14 and the versions
in [environment.json](results/vemana_baseline/environment.json).

```bash
git clone https://github.com/hardikbha/vemana-telugu-summarization-study.git
cd vemana-telugu-summarization-study
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.15.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[model,reports,test]'
python scripts/download_model.py
python scripts/verify_model.py
```

Weights are approximately **1.12 GB** on disk and are not committed to Git.
The downloader pins the tested revision and writes SHA-256 checksums. It does not
execute model-supplied Python code. For a GPU, install a compatible CUDA PyTorch
build during staging instead of the CPU build. The separate
[GPU experiment](results/gpu-en-v1/README.md) records CUDA dependencies and behavior;
the historical CPU run does not establish GPU performance.

## Run a small experiment first

```bash
python scripts/benchmark_vemana.py --limit 3 --skip-pilots --output-dir outputs/small
python scripts/benchmark_vemana.py --input-file input_notes/02_pension_delay.txt \
  --skip-pilots --prompt-mode direct --output-dir outputs/my-note
```

Outputs include the untouched model generations, source text/hashes, prompt hashes,
output checks, input/output token counts and timings. `accepted_basic_checks` means
only those technical checks passed. Read the generation against its source.
Successful script completion means the evaluation ran, not that the model passed.
Use Linux or WSL; the runner uses Linux process-memory accounting.

To rerun the complete experiment, including pilots:

```bash
python scripts/benchmark_vemana.py --batch-size 8 --output-dir outputs/full-run
```

For the exact historical scripts, including the original shared validation code,
see [repro/](repro/README.md). The historical baseline used the same six short pilot
outputs from an earlier setup run, after matching their source/prompt hashes; those
pilots are explicitly marked as reused in the recorded metadata. New runs default
to fresh pilots. Results can vary across hardware/library builds.

## Try a different prompt

Create a UTF-8 text file containing the literal `{source}` placeholder:

```text
పాఠ్యం:
{source}

సారాంశం:
```

Then run:

```bash
python scripts/benchmark_vemana.py --input-file input_notes/01_water_supply.txt \
  --prompt-file my_prompt.txt --skip-pilots --output-dir outputs/prompt-trial
```

The custom prompt may request Telugu, but the **existing acceptance checks still
expect English**. Preserve raw outputs and update the evaluation protocol explicitly
when changing the target language. Do not interpret an English-check rejection as a
Telugu-quality score. The script rejects a template without `{source}`.

`--max-new-tokens`, `--batch-size`, `--threads`, `--prompt-mode`, `--model-path`, and
`--output-dir` are configurable. The 1,024-token trained-length budget is enforced.
An input that cannot leave adequate output room is recorded as a context failure,
not silently shortened. The default long-note corpus remains unchanged by a run.

## Inspect results without downloading any model

The committed reports and raw results are immediately readable. To rebuild the
recorded **100-note baseline report**:

```bash
python -m pip install -e '.[reports]'
python scripts/report_vemana.py --output-dir outputs/baseline-report
```

This report builder intentionally describes the recorded baseline; its qualitative
observations are not an automated verdict on new custom experiments.

## Offline use

Download weights and stage dependencies on a connected machine, then transfer the
repository, model folder and dependency wheels into the isolated environment.
See [OFFLINE.md](docs/OFFLINE.md). During evaluation the loader uses local files only,
disables Hugging Face lookups/telemetry, and rejects outbound internet sockets.
These are application-level controls; deployment network isolation remains a host
or infrastructure responsibility. No Bedrock service or hosted model API is used.

## Tests

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

The default tests do not download weights or run inference. They check dataset/result
integrity, context budgeting, prompt preservation, and validation behavior. GitHub
Actions runs these checks on pushes and pull requests. A local inference smoke run
is recorded separately; unit tests do not establish model quality.

## What is included and what remains

Included: runnable completion tests, pinned model download/verification, actual
data and outputs, a corpus generator, report generation, tests and review guidance.
Model weights, virtual environments, credentials and real helpline data are excluded.

Not provided: a curated bilingual training set, a quality-validated fine-tuned model,
independent human evaluation, production authentication/UI, or AWS deployment.
[SUPERVISOR_REVIEW.md](docs/SUPERVISOR_REVIEW.md) describes concrete ways to investigate
these gaps before making production claims.

## Licenses

Our code and synthetic fixtures use the [MIT license](LICENSE). Vemana's weights and
tokenizer are published under Apache 2.0; preserve its
[LICENSE](model_metadata/Vemana-Base/LICENSE) and
[NOTICE](model_metadata/Vemana-Base/NOTICE) when required by their terms. The upstream
model card and notices are retained as third-party material, not relicensed as ours.
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
