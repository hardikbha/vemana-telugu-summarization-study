# Synthetic benchmark fixture

100 notes; 302–317 whitespace-separated Telugu words per file; 30,721 words total.
These are partly templated, synthetic complaints created for the experiment, not
real helpline records or independently collected natural-language cases.

The manifest stores IDs, source categories, key facts, word counts, file SHA-256
hashes and one rare urgent-wire case. Labels are used for evaluation diagnostics,
not passed as extra model features. File text contains application numbers/place
labels only as synthetic scenario details.

`reference_overview.md` is a separately written source-derived overview, explicitly
not a model generation or a replacement for a failed generation.

Regenerate with `python scripts/create_100_notes.py`. The integrity tests compare
the committed source hashes, counts and recorded model inputs. Do not edit this
fixture silently when comparing a new run with the baseline.
