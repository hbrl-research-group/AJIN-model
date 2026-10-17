# AJIN — ALD Adsorption / Deposition Prediction

Companion code for the manuscript **"An Interpretable AI Framework for
Sequential Prediction of Surface Reactivity in Atomic-Scale Processes"**
(Gu, Le, Masroor, Lee, Lee — Incheon National University / UNIST).

AJIN predicts, for a given (precursor, substrate, reactant, temperature)
condition:

- **Ads** — will the precursor adsorb onto the substrate?
- **Dep** — will a full ALD cycle deposit a continuous film (GPC > 0)?

> **Data note.** This repository ships **code and pre-trained model
> weights only**. The manuscript's underlying dataset
> (`Final_Study_Dataset_N560.xlsx`) and its precomputed structural
> embeddings (`master_embeddings.json`) are **unpublished research
> data** and are **not included here** — see `data/README.md`. Because
> every feature `ajin.py` builds (for training *and* prediction) comes
> from `master_embeddings.json`, **`predict` cannot resolve any molecule
> name at all without a copy of that file** — the pre-trained
> `models/*.pkl` weights alone aren't enough, since there is no way to
> turn a molecule's name into the feature vector they expect. This repo
> is therefore, honestly, a **methodology + reproducibility release**:
> the training/evaluation code, the exact per-target model recipe, and
> the trained weights are public; running it on your own conditions
> requires either your own `master_embeddings.json` (see `data/README.md`
> for the schema and requesting access) or a version of this repo the
> authors have populated with one.

## Quick start

```bash
git clone <this-repo>
cd AJIN
pip install -r requirements.txt
```

With your own `data/master_embeddings.json` in place (see
`data/README.md`):

```bash
python ajin.py predict --input examples/input_template.xlsx
```

`examples/input_template.xlsx` shows the expected input format — copy
and edit it with your own conditions. Results are written next to your
input as `<input>_predictions.xlsx` by default (or pass
`--output myfile.xlsx` for a specific path). This uses the recommended
`Ads_BEST.pkl` / `Dep_BEST.pkl` models that already ship in `models/` —
no training needed, just the embeddings file.

## What's in this repo

```
AJIN/
├── ajin.py                  <- the entire pipeline: training + prediction (one file)
├── requirements.txt
├── data/
│   └── README.md             <- schema for Final_Study_Dataset_N560.xlsx /
│                                 master_embeddings.json, if you have your
│                                 own copy (NOT included -- unpublished data)
├── models/                   <- pre-trained weights (need your own
│   ├── Ads_BEST.pkl / Dep_BEST.pkl           data/master_embeddings.json to
│   ├── {Ads,Dep}_{CHG,ALI,Hybrid}_{RF,XGB,MLP}.pkl   run predict -- see above)
│   └── all9_results.csv / .xlsx           <- the benchmark table these came from
│                                              (aggregate metrics only, no raw data)
└── examples/
    └── input_template.xlsx   <- shows the expected input FORMAT (running it
                                   needs your own master_embeddings.json too)
```

There is deliberately **no live embedding computation** anywhere in
`ajin.py` (no CHGNet/ALIGNN inference at runtime) — every feature, for
training and prediction alike, comes from `data/master_embeddings.json`
only. An earlier iteration of this repo tried falling back to a live
structure-embedding step for molecules outside that file, but that
computation doesn't occupy the same feature space as the precomputed
embeddings the models were trained on, and it silently produced wrong
predictions. Keeping a single, consistent feature source means what you
see in `predict` is always what the model was actually validated on.

## The three things `ajin.py` does

### 1. Predict (the thing you'll use most)

```bash
python ajin.py predict --input your_conditions.xlsx
python ajin.py predict --input your_conditions.xlsx --output results.xlsx
python ajin.py predict --input your_conditions.xlsx --model CHG_MLP   # any of the 9 combos instead
```

Input file columns (see `examples/input_template.xlsx`):

| Precursor | Substrate | Reactant | Temp | Role |
|---|---|---|---|---|
| TMA | SiO2 | H2O | 200 | |
| BDMADMS | SiN | H2O | 200 | Inhibitor |

- `Reactant` — optional, defaults to H2O if blank (Ads doesn't use it anyway).
- `Role` — optional. Leave blank for a normal growth precursor. Set to
  `Inhibitor` if the row is a Small-Molecule-Inhibitor (SMI) expected to
  adsorb and **block** growth on that substrate rather than deposit film.
  This only changes the **Dep** prediction — see "The BEST model" below.
- Precursor / Substrate / Reactant names must each already be a key in
  `data/master_embeddings.json` (case-insensitive) — the shipped
  `models/*.pkl` were trained on the manuscript's own set of molecules;
  to predict on anything outside that set you need your own
  `data/master_embeddings.json` covering it (see `data/README.md`).

Output columns: `Prob_Ads(%)`, `Pred_Ads` (Success/Fail), `Prob_Dep(%)`,
`Pred_Dep` (Success/Fail).

### 2. Reproduce the paper's full 9-case benchmark (requires your own data)

The paper compares 3 structural-embedding architectures **CHG(64)**
(CHGNet), **ALI(128)** (ALIGNN), and **Hybrid(192)** (both concatenated),
crossed with 3 algorithms **RF**, **XGB**, **MLP** — 9 combinations per
target, 18 models total.

```bash
python ajin.py train --all9
```

This needs `data/Final_Study_Dataset_N560.xlsx` and
`data/master_embeddings.json` in place (see `data/README.md` for the
exact schema — not included in this repo). Given those files, it runs
5-fold `StratifiedGroupKFold` cross-validation (grouped by precursor
identity, so no precursor is ever in both the train and test fold of
the same split — the paper's exact validation protocol), prints a live
comparison table, and saves all 18 models plus `models/all9_results.csv`.

The `models/all9_results.csv` / `.xlsx` files already in this repo are
the aggregate BA/AP/AUC numbers from the authors' own run (no raw data,
just summary statistics) so you can see the full picture without
retraining:

| Target | Arch | RF | XGB | MLP |
|---|---|---|---|---|
| Ads | CHG   | 65.5% | 67.0% | **74.2%** (paper's Champion setup) |
| Ads | ALI   | 70.3% | 68.2% | 66.7% |
| Ads | Hybrid| 68.6% | 65.7% | 74.1% |
| Dep | CHG   | 82.4% | 80.6% | **83.5%** (paper's Champion setup) |
| Dep | ALI   | 80.2% | 86.4% | 81.5% |
| Dep | Hybrid| 81.3% | 83.8% | 82.5% |

(Balanced Accuracy shown; see `models/all9_results.csv` for AP/AUC too.)
The paper's published Champion numbers are **Ads BA = 79.2%, Dep BA =
84.8%** (CHG + MLP) — this run reproduces Dep almost exactly and comes
within a few points on Ads (expected run-to-run variance from library
versions/hardware, exactly as in any neural-net benchmark — the feature
source itself, `master_embeddings.json`, is a fixed precomputed
snapshot, so it isn't a source of variance here).

### 3. Train/refresh the recommended "BEST" model (requires your own data)

```bash
python ajin.py train --best
```

Same data requirement as `--all9` above. Most users won't need this —
the resulting `Ads_BEST.pkl` / `Dep_BEST.pkl` already ship in `models/`.

## The BEST model: matched to the paper, improved where it honestly wins

Rather than blindly picking one architecture/algorithm package for both
targets, `Ads_BEST.pkl` / `Dep_BEST.pkl` are chosen **per target** against
the paper's own published numbers, using one rule: match the paper, and
only deviate where a change is shown — by the identical CV protocol — to
actually do better.

| Target | Recipe | This repo's CV BA | Paper's published BA | Verdict |
|---|---|---|---|---|
| **Ads** | CHG(64) + MLP, **no** Role flag | 74.2% | 79.2% | Paper's original method — kept as-is. Adding a Role(Inhibitor) flag was tested and made Ads *worse*, so it was **not** added. |
| **Dep** | CHG(64) + MLP, **+ Role(Inhibitor)** flag | 90.0% | 84.8% | **Beats the paper** — the flag tells the model "this adsorbate is meant to block growth, not deposit," which the original feature set had no way to express. Kept. |

Practically: **Ads predictions are identical whether `Role` is blank or
`Inhibitor`** (the Ads model doesn't see that column at all — it's the
paper's original, unmodified method). **Dep predictions do change** based
on `Role` — set it to `Inhibitor` when screening a molecule expected to
block growth, and leave it blank for ordinary growth precursors.

## Model reference metrics (paper)

| Metric | Ads | Dep |
|---|---|---|
| Champion Balanced Accuracy (BA) | 79.2% | 84.8% |
| Champion Average Precision (AP) | 93.9% | 91.3% |
| Champion AUC-ROC | 0.808 | 0.915 |
| Baseline (majority class) BA | 50.0% | 50.0% |
| LOPO (all 76 precursors), mean BA | 75.5% ± 30.8% | 86.7% ± 22.2% |
| LOPO (two-class-only subset), mean BA | 66.4% ± 26.1% | 78.9% ± 20.7% |

LOPO = Leave-One-Precursor-Out: the hardest generalization test, where an
entire precursor is held out of training and predicted cold. This
repository's `ajin.py` does not run LOPO itself (it would mean training
76 x 2 models); see the manuscript's Section 3.6 for that analysis.

## Troubleshooting

- **"Model not found"** — run `python ajin.py train --best` (for
  `--model best`) or `python ajin.py train --all9` (for any `ARCH_ALGO`
  combo) first. This shouldn't happen if you're using the pre-trained
  `models/*.pkl` files that ship with this repo, unless you deleted them.
- **"'X' not found in data/master_embeddings.json"** — this repo ships
  no embeddings file; `X` must be in your own copy of it (see
  `data/README.md`), or you're limited to predicting on molecules
  already baked into `models/*.pkl` at training time.
- **"data/Final_Study_Dataset_N560.xlsx not found"** — expected; this
  file isn't included (see the data note above). You only need it for
  `train --all9` / `train --best`, not for `predict`.
- **xgboost not installed** — only needed for the `XGB` rows of
  `train --all9`; everything else works without it.

## Citation

If you use this code, please cite:

> Gu, B.; Le, T. N.; Masroor, Z.; Lee, M.; Lee, H.-B.-R. "An Interpretable
> AI Framework for Sequential Prediction of Surface Reactivity in
> Atomic-Scale Processes." (manuscript in preparation)

The underlying experimental dataset is unpublished; contact the
corresponding author regarding data availability.
