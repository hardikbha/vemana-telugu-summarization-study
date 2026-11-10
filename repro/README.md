# Historical experiment source

These are verbatim copies of the scripts used for the committed baseline:

- `benchmark_vemana.py`: original runner.
- `report_vemana.py`: original report builder.
- `engine.py`: original shared helper; Vemana imported only `validate_summary` and
  `terminology_check`. Its unrelated Qwen class was not instantiated by Vemana.
- `create_100_notes.py`: synthetic fixture generator.

The working repository puts the same three short notes in `input_notes/` and the
same 100-note fixture in `benchmark_100/`. After downloading the pinned checkpoint,
the original runner can be used from the repository root:

```bash
python repro/benchmark_vemana.py --batch-size 8
```

It writes an ignored `vemana_evaluation/` output directory. The recorded final run
reused six already completed pilot records after matching source/prompt hashes;
running without `--reuse-pilots` recomputes those pilots. The pinned code's reuse
checks do not independently authenticate the checkpoint; use a separate output
directory and do not reuse pilots across a changed checkpoint or prompt.

For normal new experiments use `scripts/benchmark_vemana.py`, which adds explicit
paths, corpus subsets, single-file inputs and optional custom completion templates.
The baseline reports are already committed under `results/vemana_baseline/`; new
experiments should not overwrite those historical outputs.
