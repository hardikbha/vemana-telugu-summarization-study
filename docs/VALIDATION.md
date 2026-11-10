# Repository preparation checks

The following checks were completed locally before the initial GitHub push:

- **9 offline tests passed.** No model weights or network access are needed for
  these tests. They cover source hashes, 100-note coverage, token budgets, raw
  result counts, prompts and basic-check behavior.
- All downloaded Vemana files matched the recorded SHA-256 manifest for revision
  `93cefe2594c5157e4aa8eb9908639c253652fed9`.
- The portable runner completed a fresh one-note inference smoke run using an
  explicit local `--model-path`, `--input-file`, `--skip-pilots`, and an ignored
  output directory. That model output was **rejected**, consistent with the observed
  quality limitations; script execution success is not summary-quality success.
- The portable report builder regenerated the recorded one-page PDF, editable
  Word report and report archive without needing downloaded model weights.
- Source compilation and command-line help checks passed.
- The publication set was checked for machine-specific paths, credential patterns,
  weight files and large environment artifacts. Only synthetic inputs and model
  outputs from those synthetic inputs are included.

GitHub Actions independently reruns the offline test suite after pushes. These
checks do not validate a fine-tuned model, full model quality, AWS deployment,
or an operating-system-enforced physical air gap.
