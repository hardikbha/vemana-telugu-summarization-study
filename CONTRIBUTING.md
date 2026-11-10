# Contributing experiments

Start with [the supervisor review guide](docs/SUPERVISOR_REVIEW.md). Issues and pull
requests should name a concrete hypothesis and distinguish observed results from
proposed changes.

1. Keep `results/vemana_baseline/` and the benchmark fixture immutable for comparison.
2. Save new outputs under a distinct `outputs/` directory, with model revision,
   prompts, source hashes and decoding settings. Include failed attempts.
3. Run `python -m pytest -q` before submitting code.
4. Report semantic source fidelity separately from fluency, script/completion checks,
   runtime, and RAM. Do not call basic-check counts an accuracy score.
5. Do not use the held-out benchmark as training data and then report its score as
   independent evidence. Group near-duplicate cases before splitting.
6. Do not commit weights, API keys, environments or real citizen data. Attach only
   approved, de-identified review material to public discussions.

The experimental `vemana-train` command is described in
[SUMMARIZATION_EXPERIMENTS.md](docs/SUMMARIZATION_EXPERIMENTS.md). Training contributions
should include data provenance, objective/loss masking, reproducible hyperparameters,
train/test separation, checkpoint license, and source-level evaluation—not only a
training loss. Do not publish private source/target pairs or model outputs without
reviewing them; local training and experiment directories are ignored by Git.
