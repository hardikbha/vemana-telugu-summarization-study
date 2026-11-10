# Observed results and interpretation

The committed result is a direct-use test of `udgamlabs/Vemana-Base` at revision
`93cefe2594c5157e4aa8eb9908639c253652fed9`. It is a 278M-parameter Telugu base
completion model. It was not trained, instruction-tuned or modified for this trial.

## What happened

- All 100 long-note outputs failed one or more basic checks: completion, English
  script/length, repetition, or a few domain-terminology checks.
- No note failed because of the trained context budget. Full source notes were
  included; input plus reserved output stayed within 1,024 tokens.
- Every long-note output included the opening garbage-collection sentence from
  the single example, even when the source concerned another topic.
- A source-topic word appeared in 10/100 outputs, but all ten are weak matches to
  the sanitation topic already present in the example. That is not evidence of
  faithful source understanding or a 10% accuracy score.

## Concrete short-pilot examples

| Source meaning | Prompt style | Observed generation |
| --- | --- | --- |
| Drinking water has not arrived for two days; restore supply | Direct | An unrelated story about a woman arrested in a crime case, followed by repetition |
| Same water complaint | One example | Copied garbage-collection content and invented a publication date |
| Old-age pension has not arrived for three months; release payment | One example | Copied the garbage sentence and mixed in a pension request |

These descriptions refer to the raw pilot records, not corrected model summaries.
The model also generated Telugu continuation text from `తెలుగు భాష`; that sanity
check establishes that loading/generation worked, not that its facts were correct.

## Why this is not a general model-quality verdict

The model card explicitly describes a base model, not an assistant. We tested a
cross-language summarization task with limited direct and one-example prompting,
fixed greedy decoding and a 192-token output cap. We did not search many prompts,
train adapters, collect human judgments, or evaluate every possible Telugu task.
The data are 100 partly templated synthetic notes, not 100 independent real cases.

“0/100 passed” describes the technical acceptance checks in this experiment. It is
not a standardized summarization accuracy measurement. It is sufficient evidence
to avoid treating this checkpoint as a ready-made summarizer in the tested setup.

## Resources and timing

The long-note run used CPU FP32, eight threads, batches of eight. Its recorded
wall time was 183.269 seconds, including model load and excluding reused short-pilot
time. Peak resident process memory was 2.188 GiB. These are local measurements,
not AWS latency, VRAM requirements, or production capacity guarantees.

## Earlier Qwen context

[The Qwen reference JSON](../results/qwen_reference/comparison_results.json) records
an earlier 0.8B trial on these synthetic sources: 58/100 CPU and 66/100 GPU outputs
passed its basic checks. The CPU final overview invented details; GPU aggregation
failed. Those checks are not factual-accuracy scores either.

Qwen and Vemana used different architectures, tokenizers, precision, prompt formats,
worker/batch configurations and output budgets. Do not rank them by these numbers
as though this were a controlled head-to-head benchmark.

## Next evidence needed

Build a held-out set with reference summaries and native-speaker review. Score
source support, negation, numbers, omitted issues, unrelated content, and rare-case
retention. Keep latency and RAM metrics separate from those quality assessments.
Compare instruction prompting, a native-Telugu objective and supervised adaptation
under a written protocol before selecting a production model.
