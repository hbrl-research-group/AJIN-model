# data/

This folder is intentionally empty in the public release.

The manuscript's labeled dataset (`Final_Study_Dataset_N560.xlsx`) and
its precomputed structural embeddings (`master_embeddings.json`) are
unpublished research data and are **not distributed in this
repository**. What *is* public is the trained model weights in
`../models/*.pkl` — `python ajin.py predict` works immediately with
those for any precursor/substrate/reactant already covered by the
manuscript's dataset, with no files needed here at all.

You only need files in this folder if you want to: predict on a
molecule outside that set, retrain (`train --all9` / `train --best`),
or reproduce the full 9-combination benchmark from scratch. In every
case, both files below matter: `ajin.py` builds every feature — for
training AND prediction alike — from `master_embeddings.json` only, so
training and inference always draw from the exact same feature space.

## `Final_Study_Dataset_N560.xlsx`

One row per (precursor, substrate, reactant, temperature) experimental
condition. Required columns:

| Column | Type | Notes |
|---|---|---|
| `Precursor` | str | Matches a key in `master_embeddings.json` (case-insensitive) |
| `Substrate` | str | Same |
| `Reactant` | str | Same; blank defaults to H2O |
| `Temp` | number | Process temperature, °C |
| `Result(Ads)` | 0/1 | Adsorption outcome (label) |
| `Result(Dep)` | 0/1 | Deposition outcome (label) |
| `Role` | str, optional | `Inhibitor` for Small-Molecule-Inhibitor rows, blank otherwise |

## `master_embeddings.json`

A flat JSON object: `{ "<molecule name>": [<403 floats>], ... }`, one
entry per unique precursor/substrate/reactant name you want to train or
predict on. The 403-length vector layout used by `ajin.py`:

- `[211:275]` — 64-dim CHGNet structural embedding ("CHG")
- `[275:403]` — 128-dim ALIGNN structural embedding ("ALI")
- `[0:211]` — unused by this script (reserved / other feature blocks
  from the broader research pipeline)

There is no live embedding step in this script (no CHGNet/ALIGNN
inference at runtime) — every molecule you want to use, in training or
prediction, must already have an entry here. This is a deliberate
choice: an earlier version of this repo tried computing embeddings live
from `.cif` files for molecules missing from this file, but that
computation does not occupy the same feature space as whatever produced
these precomputed vectors, and it silently produced wrong predictions.

Requesting access: contact the corresponding author (see the main
manuscript) regarding data availability for this unpublished work.
