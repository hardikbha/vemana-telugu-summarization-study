# Vemana-Base evaluation

Requested model: `udgamlabs/Vemana-Base`, pinned at revision
`93cefe2594c5157e4aa8eb9908639c253652fed9`.

The publisher releases the model and tokenizer under Apache 2.0. The downloaded
LICENSE and NOTICE are in `models/Vemana-Base/` and included with the result ZIP.
This 278M-parameter model was trained from scratch as a Telugu **base completion
model**, not an instruction/chat or translation model. The card recommends keeping
prompt plus output within its 1,024-token trained length. That limit is respected
in this trial, even though the configuration permits larger positional indices.

The current application task is Telugu input -> English summaries. Six short-note
pilot runs tested direct instruction-shaped completion prompts and a one-example
Telugu/English continuation prompt. The long-note trial uses that one-example
prompt on the same 100 synthetic Telugu notes (302–317 words each) from benchmark_100.
No chat template is applied because none is supplied. No source text is truncated.

Execution is CPU FP32 with eight threads and long-note batches of eight. Outputs
are retained exactly as generated, including unrelated stories, copied example
content, repetitions and incomplete outputs. Basic English, length, completion
and terminology checks are separate from semantic accuracy.

The batch setup initially needed explicit pad-token configuration; that setup
error is not counted as a model failure. Completed short-pilot results were reused
only after checking that their source and prompt hashes match. The final elapsed
time is for the completed run; reused pilot wall time is not included. The serial
development logs remain under validation/vemana-serial-pilot/.

Files:

- `results.json`: complete evaluation, prompts, setup, raw outputs and timings.
- `N001.json` ... `N100.json`: all long-note outputs with source text and hashes.
- `pilot_*.json`: short pilot outputs, with both prompting methods.
- `telugu_continuation.txt`: native Telugu continuation sanity check.
- `analysis.json`: diagnostic counts and conclusions after completion.
- `all_outputs.csv`: spreadsheet-friendly results after completion.
- `Vemana_Base_Evaluation.pdf` / `.docx`: review reports after completion.

No inference API was used. No model fine-tuning was performed and no application
default was replaced. This is a direct-use test, not evidence that adaptation could
never improve the model or a controlled model-quality ranking against Qwen.

Primary source: https://huggingface.co/udgamlabs/Vemana-Base
