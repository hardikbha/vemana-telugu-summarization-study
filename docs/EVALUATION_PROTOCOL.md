# Evaluation protocol and provenance

## Recorded baseline

- Model: Vemana-Base, revision `93cefe2594c5157e4aa8eb9908639c253652fed9`.
- Weights: publisher's FP32 safetensors, loaded locally with `AutoModelForCausalLM`.
- Tokenizer: publisher's own tokenizer; no chat template; BOS added by tokenizer.
- Trained context: 1,024 tokens total; not the larger positional capacity in config.
- Short pilots: three notes × direct/one-example prompts, greedy generation.
- Long trial: 100 synthetic notes; one-example prompt; batches of eight; CPU FP32;
  eight CPU threads; maximum 192 new tokens.
- Left padding configured from the model's pad token before batch generation.
- Long-note source and prompt hashes are saved; no source truncation was used.
- No inference API, training, automatic translation service or human correction of
  the raw model outputs was used. Outbound sockets were blocked during inference.

## Provenance caveats

The initial serial development run was interrupted while preparing a faster batched
run. The batch setup also exposed a missing tokenizer pad-token declaration. This
was configured from the model config; a setup error was not scored as a model error.
The final run reused the completed six short-pilot records after source/prompt hash
checks. Long-note results were regenerated in the completed batched run. The recorded
wall time excludes the reused pilot time. New user-facing runs default to fresh pilots.

The original scripts and shared helper are in `repro/`; the scripts under `scripts/`
add configurable paths, a subset limit and custom prompt templates. Those additions
were not used to obtain the historical baseline. Model files and their publisher
license/notice metadata are pinned in `model_metadata/`; actual weights are downloaded
separately. The original shared helper also contains unrelated Qwen functionality;
only its two output checks were used for Vemana, and those are isolated in `engine.py`.

## What the checks mean

1. **Completion:** reject an output that exhausts its allowed token budget without EOS.
2. **Language/length:** require predominantly English letters and a bounded length.
3. **Repetition/meta-output:** reject excessive repeated phrases or task descriptions.
4. **Known terminology:** reject a few specific pension/fallen-wire mistranslations.
5. **Diagnostics:** separately count example-phrase copying and source-topic keywords.

These checks are not a semantic-accuracy metric. They can reject an otherwise useful
partial answer or miss an invented English claim. The example-copy diagnostic is
also weak for a sanitation source; its presence is not automatically a semantic
error in every such case. Inspect source/target pairs with a human rubric.

## Rubric for the next experiment

For each held-out case, score independently:

- Main issue preserved; requested action preserved.
- Negation/status preserved (not received vs received; unresolved vs resolved).
- Numbers, dates, duration, location and names supported by the source.
- No invented cause, injury, payment, investigation result or completion claim.
- Critical omissions and unrelated content.
- English/Telugu fluency separately from source fidelity.

For an aggregate page, also score coverage of all selected inputs, explicit failed
input counts, rare serious-case retention, and prevention of fact transfer between
separate cases. Report denominator choices and preserve source references.

Track model revision, tokenizer, prompt text/hash, source hashes, train/test split,
precision, device, batch size, software versions, stopping rules, timing boundaries,
RAM/VRAM definitions and all rejected attempts. Do not infer quality from memory
size or use a changed prompt/data split as though it were the same baseline.
