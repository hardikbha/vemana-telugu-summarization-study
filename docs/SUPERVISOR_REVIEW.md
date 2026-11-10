# Supervisor review guide

This is an open starting point for investigation. None of the suggestions below
has been established as an effective fix by this repository's recorded results.

The new [experiment and training guide](SUMMARIZATION_EXPERIMENTS.md) implements
language/prompt/decoding trials and target-only supervised training. The historical
baseline below is unchanged; implementation tests do not establish improved summary quality.

## First files to inspect

1. [One-page report](../results/vemana_baseline/Vemana_Base_Evaluation.pdf).
2. [Prompt construction and inference](../scripts/benchmark_vemana.py).
3. [Basic checks](../engine.py): note how limited they are.
4. [Water pilot source and output](../results/vemana_baseline/pilot_1_fewshot.json).
5. [Dataset manifest](../benchmark_100/dataset.json) and [raw run](../results/vemana_baseline/results.json).

## Questions worth discussing

- Is Telugu-to-English generation the right objective for this small, Telugu-focused
  base checkpoint, or should native Telugu summarization be studied first?
- Does a completion-style format with several task examples improve source grounding,
  or simply increase example copying? How much of the 1,024-token context remains?
- How should unsupported claims, negation reversals, missing requested actions and
  transfer of facts between citizens be measured independently of fluency?
- What paired Telugu-source/English-summary data can be collected with proper consent,
  licensing and a clean train/validation/test separation?
- Can supervised fine-tuning teach the objective at 278M parameters, and does it
  generalize beyond templates? What is the fair stronger-model baseline?
- How should an aggregate page preserve a rare dangerous case and represent rejected
  or unreadable inputs? A single page cannot retain every individual fact.

## Suggested sequence of experiments

| Experiment | Change | Evidence required |
| --- | --- | --- |
| Completion-format ablation | Direct vs 1–3 varied examples; do not tune on the final test set | Source-supported summaries, example-copy rate, context usage |
| Target-language ablation | Telugu summaries vs English summaries | Separate language-aware checks and native-speaker scoring |
| Decoding ablation | Greedy vs a documented bounded sampling configuration | Repetition, omissions, unsupported claims and stability across seeds |
| Supervised adaptation | Train on reviewed source/summary pairs; fixed held-out set | Improvements over unchanged base and simple baselines, with error examples |
| Stronger-model comparison | Same data, scoring rubric and comparable generation budget | Quality and latency/cost reported separately |
| Batch aggregation | Include all successful notes, retain rejection coverage | Rare-case retention, no cross-case fact transfer, source traceability |

Do not label outputs “better” solely because they sound fluent, end at EOS or
pass a language regex. Do not remove troublesome cases from the held-out test.

## Training-data proposal (not a training implementation)

`docs/training_pair.example.jsonl` shows a possible plain source/target format. The
examples are hand-written synthetic illustrations, not a sufficient training set.
They overlap the short-pilot scenarios; if used for training, do not treat those
pilots or closely related templates as independent held-out evidence.
Collect enough varied, reviewed examples to cover the actual domain and split by
case/near-duplicate family before generating augmented examples. Keep synthetic
training notes distinct from the benchmark to avoid leakage.

For a base-model supervised objective, the source/instruction portion should be
masked from target loss and only the intended answer should be optimized. Respect
the context budget for the full training sequence. Adapter/full-fine-tuning choices,
data size, hyperparameters, and GPU requirements need a separate plan and benchmark;
no training results or required GPU sizing are claimed here.

## Collaboration

Open an issue or pull request with the hypothesis, exact revision, prompt, data split,
seed/decoding settings, measured outputs and failure cases. Keep baseline results
immutable and save new experiments under a separate output directory. Suggested
reporting format: [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).
