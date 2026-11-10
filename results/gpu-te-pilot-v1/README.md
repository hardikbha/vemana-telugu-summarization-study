# Telugu-output pilot compared with English

**Telugu script is correct, but summarization is still unreliable in this pilot.**
Vemana-Base was tested on three of the same Telugu notes used in the English
GPU experiment: drinking water, pension delay, and fallen electrical wires near
a school. All three Telugu outputs copied instructions or demonstrations instead
of producing a faithful summary. The school electrical hazard was omitted.

This was a purposive, three-note check, not a full 100-note Telugu evaluation or
a factual-accuracy benchmark. No larger run or fine-tuning was started.

| Output language | Same three cases | Observation |
| --- | --- | --- |
| English, from the recorded 100-note run | 3/3 rejected by basic checks | Unrelated library-fan text and repetition or prompt continuation |
| Telugu, new pilot | 3/3 rejected by basic checks | Telugu instructions/examples copied; all outputs exhaust the budget |

All Telugu outputs have a target-script fraction of 1.0. This says only that
their letters use Telugu script; copied prompt text also passes this check.
All three overlap with all three demonstrations, generate exactly 192 tokens,
and fail to produce EOS. Two repeat the summary instruction, followed by the
bus, library-fan, and farmer-service-centre examples. The urgent case starts
with a generic sentence from the source, then copies the examples and repeats.

For example, the water and urgent-wire outputs both include this unrelated text:

> గ్రంథాలయంలో రెండు ఫ్యాన్లు పనిచేయడం లేదు; పుస్తకాలు దెబ్బతినలేదు.

Its meaning is that two library fans are not working and the books are undamaged.
This does not describe either complaint.

## Setup and limitations

- Model: `udgamlabs/Vemana-Base`, revision
  `93cefe2594c5157e4aa8eb9908639c253652fed9`; no fine-tuning.
- Code checkout: `82718e0465f2e754410f899cc3721f79da8c0dce`;
  inference package and `engine.py` unchanged since `82bedc1`.
- Physical GPU 5 on `gpu01`: NVIDIA H100 80GB HBM3; FP32; eight CPU threads.
- Same three-example count, greedy decoding, seed 42, and 192-token output
  allowance as English. The target language changes the instruction, labels,
  and demonstration outputs to Telugu using the existing prompt builder.
- Prompt lengths: 625, 634, and 645 tokens. Every source and full output budget
  fit within 1,024 tokens. There was no truncation or context-budget failure.
- Each source text and hash matches the prior English run exactly.
- Local-only model loading and Hugging Face offline mode; no infrastructure
  air-gap claim or newly instrumented peak-memory measurement for this pilot.
- Mechanical checks are not factual accuracy. In particular, Telugu's mechanical
  checker does not run the English terminology checks; both outputs still failed.

Switching the output language alone did not solve the tested failure. Vemana-Base
is a text-continuation checkpoint; these results do not establish whether a
properly trained summarization checkpoint would succeed. Supervised adaptation
would require a separate curated dataset and held-out validation.

## Files and reproduction

- [paired_comparison.csv](paired_comparison.csv): source, English output, Telugu
  output, and diagnostics side by side.
- [results.jsonl](results.jsonl): untouched Telugu source/prompt/output records.
- [human_review.csv](human_review.csv): original CLI review worksheet.
- [run.json](run.json): completion metadata and exact model manifest.
- [comparison.json](comparison.json): paired counts, source integrity, and limits.
- [English GPU results](../gpu-en-v1/README.md): the separate 100-note run.

With the model and dependencies staged, create a directory containing copies of
`note_001_water.txt`, `note_021_pension.txt`, and `note_100_urgent_wires.txt` from
`benchmark_100/input_notes/`. The executed command was:

```bash
.venv/bin/vemana-summarize --device cuda \
  --input-dir operator_logs/te-pilot-inputs \
  --target-language te --examples 3 \
  --output-dir outputs/gpu-te-pilot-v1
```

Use a new output directory for a rerun. The runtime selected the physical device
by `CUDA_VISIBLE_DEVICES`, with `HF_HUB_OFFLINE=1`,
`HF_HUB_DISABLE_TELEMETRY=1`, `TOKENIZERS_PARALLELISM=false`, and
`RAYON_NUM_THREADS=1`. All notes are synthetic and were already public.
