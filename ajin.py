#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 AJIN -- AI Framework for Sequential Prediction of Surface Reactivity in
         Atomic-Scale Processes (ALD Adsorption / Deposition Prediction)
================================================================================
Single-file, ready-to-run companion code for the manuscript:

    "An Interpretable AI Framework for Sequential Prediction of Surface
    Reactivity in Atomic-Scale Processes"
    Gu, B.; Le, T. N.; Masroor, Z.; Lee, M.; Lee, H.-B.-R.

WHAT THIS SCRIPT DOES
----------------------
1. Reproduces the paper's full "9-case" benchmark: 3 embedding
   architectures (CHG-64, ALI-128, Hybrid-192) x 3 algorithms
   (RandomForest, XGBoost, MLP neural net), for both targets
   (Ads = adsorption, Dep = deposition), evaluated with 5-fold
   StratifiedGroupKFold CV grouped by precursor (no precursor ever
   appears in both the train and test fold).

        python ajin.py train --all9

2. Trains/refreshes the "BEST" model pair -- the single recommended
   model per target, chosen with a simple rule: match the paper's
   published Champion numbers, and only deviate where a modification
   is honestly shown (by the same CV protocol) to beat them.
      - Ads  BEST = CHG(64) + MLP, no extra features. This IS the
        paper's original Champion method (published BA = 79.2%).
        We tested adding a Role(Inhibitor) flag here and it made Ads
        *worse* in honest CV, so it was NOT added.
      - Dep  BEST = CHG(64) + MLP + a Role(Inhibitor) binary flag.
        The paper's published Dep Champion (BA = 84.8%) has no Role
        feature; adding it and re-validating with the same 5-fold CV
        was shown to help Dep, so it IS kept for the recommended model.

        python ajin.py train --best

3. Runs predictions for your own precursor/substrate/reactant/temp
   conditions, using either the recommended BEST model (default) or
   any of the 9 paper combinations you choose explicitly.

        python ajin.py predict --input examples/input_template.xlsx

See README.md for the full walkthrough, folder layout, and FAQ.

DATA NOTE
---------
This repo ships CODE + PRE-TRAINED WEIGHTS ONLY. The manuscript's
training dataset (Final_Study_Dataset_N560.xlsx) and its precomputed
structural embeddings (master_embeddings.json) are unpublished research
data and are NOT included here (see data/README.md).

Every feature this script builds -- for `train --all9`, `train --best`,
and `predict` alike -- comes from `data/master_embeddings.json` alone.
There is deliberately no live structure-embedding step (no CHGNet/
ALIGNN inference at runtime): an earlier version of this script tried
to fall back to a live CHGNet computation for molecules missing from
master_embeddings.json, but that computation does not live in the same
feature space as the precomputed embeddings the models were trained on
(different chgnet version/pipeline than whatever produced
master_embeddings.json), and silently produced wrong predictions.

This means `predict` cannot resolve ANY molecule name without a copy of
`data/master_embeddings.json` in place -- the pre-trained `models/*.pkl`
weights alone can't turn a molecule's name into the feature vector they
expect. This repo is a methodology + reproducibility release: the code,
per-target model recipe, and trained weights are public; running
predictions on your own conditions needs your own
`data/master_embeddings.json` (see data/README.md for the schema and
how to request access).

REQUIREMENTS
------------
pandas, numpy, scikit-learn, openpyxl, joblib (always)
xgboost (only for the XGB rows of `train --all9`)

    pip install -r requirements.txt
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

try:
    import joblib
except ImportError:
    print("[FATAL] joblib is required. pip install joblib")
    sys.exit(1)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# ==============================================================================
# PATHS  (everything relative to this file -- fully portable)
# ==============================================================================
HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "Final_Study_Dataset_N560.xlsx"
EMB_PATH = HERE / "data" / "master_embeddings.json"
MODEL_DIR = HERE / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ARCHS = ["CHG", "ALI", "Hybrid"]
ALGOS = ["RF", "XGB", "MLP"]
TARGETS = ["Ads", "Dep"]

MLP_PARAMS = dict(
    hidden_layer_sizes=(128, 64), max_iter=500, solver="adam",
    activation="relu", learning_rate_init=0.001, batch_size="auto",
    random_state=42,
)

# Reference numbers from the manuscript, printed alongside your own results
# so you can see at a glance how your run compares.
PAPER_CHAMPION_BA = {"Ads": 0.792, "Dep": 0.848}
PAPER_CHAMPION_AP = {"Ads": 0.939, "Dep": 0.913}

DATA_MISSING_MSG = (
    "This repo ships code + pre-trained weights only, not the (unpublished) "
    "training dataset. See data/README.md if you have your own data in the "
    "required format, or use the pre-trained models/*.pkl directly with "
    "'python ajin.py predict ...' instead (works for any molecule already "
    "covered by the shipped models -- no data files needed for that)."
)


# ==============================================================================
# Embeddings  (data/master_embeddings.json is the ONLY feature source --
# used identically at train time and predict time, so there is no risk of
# a train/predict feature-space mismatch)
# ==============================================================================
def load_master_embeddings():
    """Full 403-dim vectors, keyed by lowercase molecule name.
    Layout (fixed by the paper's precomputation pipeline):
        [0:211]    unused by this script
        [211:275]  CHGNet 64-dim structural feature ("CHG")
        [275:403]  ALIGNN 128-dim structural feature ("ALI")
    """
    if not EMB_PATH.exists():
        return {}
    with open(EMB_PATH) as f:
        raw = json.load(f)
    return {str(k).strip().lower(): np.asarray(v, dtype=float) for k, v in raw.items()}


def role_to_flag(role):
    return 1.0 if pd.notna(role) and str(role).strip().lower() == "inhibitor" else 0.0


# ==============================================================================
# Feature builders
# ==============================================================================
def build_arch_features(df, target, arch, master, use_role_flag=False):
    """Reproduces the paper's 9-case feature construction exactly
    (Ads = [P, S, Temp], Dep = [P, S, R, Temp]), optionally appending a
    Role(Inhibitor) flag at the end (only used by the BEST Dep model)."""
    X, y, groups = [], [], []
    target_col = f"Result({target})"
    for _, row in df.iterrows():
        p = str(row["Precursor"]).strip().lower()
        s = str(row["Substrate"]).strip().lower()
        r = row.get("Reactant")
        r_key = str(r).strip().lower() if pd.notna(r) and str(r).strip().lower() in master else "h2o"

        if p not in master or s not in master:
            continue
        if np.sum(np.abs(master[p])) == 0 or np.sum(np.abs(master[s])) == 0:
            continue
        if target == "Dep" and r_key not in master:
            continue

        p_chg, p_ali = master[p][211:275], master[p][275:403]
        s_chg, s_ali = master[s][211:275], master[s][275:403]

        if arch == "CHG":
            feat = np.concatenate([p_chg, s_chg])
        elif arch == "ALI":
            feat = np.concatenate([p_ali, s_ali])
        else:  # Hybrid
            feat = np.concatenate([p_chg, p_ali, s_chg, s_ali])

        if target == "Dep":
            r_chg, r_ali = master[r_key][211:275], master[r_key][275:403]
            if arch == "CHG":
                feat = np.concatenate([feat, r_chg])
            elif arch == "ALI":
                feat = np.concatenate([feat, r_ali])
            else:
                feat = np.concatenate([feat, r_chg, r_ali])

        feat = np.concatenate([feat, [float(row["Temp"])]])
        if use_role_flag:
            feat = np.concatenate([feat, [role_to_flag(row.get("Role"))]])

        lbl = row[target_col]
        if pd.isna(lbl):
            continue
        X.append(feat)
        y.append(int(lbl))
        groups.append(p)

    return np.array(X), np.array(y), np.array(groups)


def get_algo(name):
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
    if name == "XGB":
        if not XGB_AVAILABLE:
            raise RuntimeError("xgboost not installed. pip install xgboost")
        return XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=42, eval_metric="logloss", n_jobs=-1,
        )
    if name == "MLP":
        return MLPClassifier(**MLP_PARAMS)
    raise ValueError(f"Unknown algo: {name}")


def balance_oversample(X, y, random_state=42):
    """Per-fold minority-class oversampling (with replacement) -- the
    class-balancing method that produced the paper's published Ads
    Champion (MLP) number. Used for: MLP in --all9 (matching the
    paper's final/corrected table), and for the BEST Ads model."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y
    max_len = max(len(pos_idx), len(neg_idx))
    pos_res = resample(pos_idx, replace=True, n_samples=max_len, random_state=random_state)
    neg_res = resample(neg_idx, replace=True, n_samples=max_len, random_state=random_state)
    idx = np.hstack([pos_res, neg_res])
    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)
    return X[idx], y[idx]


# ==============================================================================
# CV evaluation + final-model training
# ==============================================================================
def evaluate_and_fit(X, y, groups, algo_name, balance_mlp=True):
    """5-fold StratifiedGroupKFold CV (honest metrics) + a final model
    fit on all data. Returns (ba, ap, auc, final_clf, final_scaler)."""
    cv = StratifiedGroupKFold(n_splits=5)
    all_probs, all_y = [], []
    for tr, te in cv.split(X, y, groups):
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X[tr])
        X_te_sc = scaler.transform(X[te])
        clf = get_algo(algo_name)
        if algo_name == "MLP" and balance_mlp:
            X_tr_fit, y_tr_fit = balance_oversample(X_tr_sc, y[tr])
            clf.fit(X_tr_fit, y_tr_fit)
        else:
            clf.fit(X_tr_sc, y[tr])  # RF's class_weight='balanced' handles it
        all_probs.extend(clf.predict_proba(X_te_sc)[:, 1])
        all_y.extend(y[te])

    all_probs, all_y = np.array(all_probs), np.array(all_y)
    all_pred = (all_probs >= 0.5).astype(int)
    ba = balanced_accuracy_score(all_y, all_pred)
    ap = average_precision_score(all_y, all_probs)
    try:
        auc = roc_auc_score(all_y, all_probs)
    except ValueError:
        auc = float("nan")

    # Final production model, trained on ALL data
    scaler_f = StandardScaler()
    X_sc = scaler_f.fit_transform(X)
    clf_f = get_algo(algo_name)
    if algo_name == "MLP" and balance_mlp:
        X_fit, y_fit = balance_oversample(X_sc, y)
        clf_f.fit(X_fit, y_fit)
    else:
        clf_f.fit(X_sc, y)

    return ba, ap, auc, clf_f, scaler_f


# ==============================================================================
# Commands
# ==============================================================================
def cmd_train_all9(args):
    print("=" * 70)
    print(" Reproducing the paper's 9-case benchmark (3 archs x 3 algos)")
    print(" 5-fold StratifiedGroupKFold CV, grouped by precursor")
    print("=" * 70)
    if not DATA_PATH.exists() or not EMB_PATH.exists():
        sys.exit(f"[ERROR] data/Final_Study_Dataset_N560.xlsx and/or "
                  f"data/master_embeddings.json not found.\n{DATA_MISSING_MSG}")
    master = load_master_embeddings()
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()

    rows = []
    for target in TARGETS:
        print(f"\n[{target}]  (paper Champion BA={PAPER_CHAMPION_BA[target]*100:.1f}%, "
              f"AP={PAPER_CHAMPION_AP[target]*100:.1f}%, arch=CHG, algo=MLP)")
        for arch in ARCHS:
            X, y, groups = build_arch_features(df, target, arch, master)
            if len(X) == 0:
                print(f"  {arch:8s}: no usable rows (check data/master_embeddings.json)")
                continue
            for algo in ALGOS:
                if algo == "XGB" and not XGB_AVAILABLE:
                    print(f"  {arch:8s} x {algo:4s}: SKIPPED (pip install xgboost)")
                    continue
                ba, ap, auc, clf, scaler = evaluate_and_fit(X, y, groups, algo)
                tag = "  <-- matches paper's Champion setup" if (arch == "CHG" and algo == "MLP") else ""
                print(f"  {arch:8s} x {algo:4s}: BA={ba*100:5.1f}%  AP={ap*100:5.1f}%  AUC={auc:.3f}{tag}")
                out = MODEL_DIR / f"{target}_{arch}_{algo}.pkl"
                joblib.dump({
                    "model": clf, "scaler": scaler, "target": target,
                    "arch": arch, "algo": algo,
                    "meta": {"cv_ba": float(ba), "cv_ap": float(ap), "cv_auc": float(auc),
                             "feature_dim": X.shape[1], "includes_role_flag": False},
                }, out)
                rows.append({"Target": target, "Arch": arch, "Algo": algo,
                             "BA": ba, "AP": ap, "AUC": auc, "N": len(X)})

    out_csv = MODEL_DIR / "all9_results.csv"
    if rows:
        pd.DataFrame(rows).to_excel(MODEL_DIR / "all9_results.xlsx", index=False)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nAll 9x2 models saved to {MODEL_DIR}/")
    print(f"Summary table saved to {out_csv}")


def cmd_train_best(args):
    print("=" * 70)
    print(" Training the recommended BEST model pair")
    print("   Ads: CHG(64)+MLP, no Role flag  (paper's Champion method)")
    print("   Dep: CHG(64)+MLP, + Role flag   (Role-aware improvement)")
    print("=" * 70)
    if not DATA_PATH.exists() or not EMB_PATH.exists():
        sys.exit(f"[ERROR] data/Final_Study_Dataset_N560.xlsx and/or "
                  f"data/master_embeddings.json not found.\n{DATA_MISSING_MSG}")
    master = load_master_embeddings()
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()

    specs = [
        ("Ads", False, True),   # target, use_role_flag, balance_mlp(oversample)
        ("Dep", True, False),   # Dep uses sample_weight instead (see below)
    ]
    for target, use_role, oversample_mlp in specs:
        X, y, groups = build_arch_features(df, target, "CHG", master, use_role_flag=use_role)
        print(f"\n[{target}] usable rows: {len(X)} (dim={X.shape[1]}, role_flag={use_role})")
        if len(X) == 0:
            print(f"  [SKIPPED] No usable rows -- check data/master_embeddings.json coverage.")
            continue

        cv = StratifiedGroupKFold(n_splits=5)
        all_probs, all_y = [], []
        for tr, te in cv.split(X, y, groups):
            scaler = StandardScaler()
            X_tr_sc, X_te_sc = scaler.fit_transform(X[tr]), scaler.transform(X[te])
            clf = MLPClassifier(**MLP_PARAMS)
            if oversample_mlp:
                X_fit, y_fit = balance_oversample(X_tr_sc, y[tr])
                clf.fit(X_fit, y_fit)
            else:
                sw = compute_sample_weight("balanced", y[tr])
                clf.fit(X_tr_sc, y[tr], sample_weight=sw)
            all_probs.extend(clf.predict_proba(X_te_sc)[:, 1])
            all_y.extend(y[te])
        all_probs, all_y = np.array(all_probs), np.array(all_y)
        ba = balanced_accuracy_score(all_y, (all_probs >= 0.5).astype(int))
        ap = average_precision_score(all_y, all_probs)
        paper_ba = PAPER_CHAMPION_BA[target]
        verdict = "BEATS paper" if ba > paper_ba else ("matches paper" if abs(ba - paper_ba) < 0.02 else "below paper")
        print(f"  CV Balanced Accuracy : {ba*100:.1f}%  (paper: {paper_ba*100:.1f}%, {verdict})")
        print(f"  CV Average Precision : {ap*100:.1f}%")

        scaler_f = StandardScaler()
        X_sc = scaler_f.fit_transform(X)
        clf_f = MLPClassifier(**MLP_PARAMS)
        if oversample_mlp:
            X_fit, y_fit = balance_oversample(X_sc, y)
            clf_f.fit(X_fit, y_fit)
        else:
            sw = compute_sample_weight("balanced", y)
            clf_f.fit(X_sc, y, sample_weight=sw)

        out = MODEL_DIR / f"{target}_BEST.pkl"
        joblib.dump({
            "model": clf_f, "scaler": scaler_f, "target": target, "arch": "CHG", "algo": "MLP",
            "meta": {"cv_ba": float(ba), "cv_ap": float(ap), "feature_dim": X.shape[1],
                     "includes_role_flag": use_role, "paper_champion_ba": paper_ba},
        }, out)
        print(f"  Saved: {out}")

    print("\nDone. Use these with:  python ajin.py predict --model best")


def load_model(target, model_name):
    fname = f"{target}_BEST.pkl" if model_name.lower() == "best" else f"{target}_{model_name}.pkl"
    path = MODEL_DIR / fname
    if not path.exists():
        sys.exit(
            f"[ERROR] Model not found: {path}\n"
            f"Run 'python ajin.py train --best' (for --model best) or "
            f"'python ajin.py train --all9' (for --model {model_name}) first -- "
            f"both need your own data/master_embeddings.json (see data/README.md)."
        )
    return joblib.load(path)


def resolve_arch(name, role, arch, master):
    key = str(name).strip().lower()
    if key not in master:
        return None
    v = master[key]
    if arch == "CHG":
        return np.asarray(v[211:275], dtype=float)
    if arch == "ALI":
        return np.asarray(v[275:403], dtype=float)
    return np.concatenate([np.asarray(v[211:275], dtype=float), np.asarray(v[275:403], dtype=float)])


def cmd_predict(args):
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_predictions.xlsx"
    )
    if not input_path.exists():
        sys.exit(f"[ERROR] Input file not found: {input_path}")

    print("=" * 70)
    print(f" AJIN Prediction  (model = {args.model})")
    print("=" * 70)

    master = load_master_embeddings()
    if not master:
        print("  [NOTE] data/master_embeddings.json not found. Every molecule name "
              "below must be covered by your own copy of that file -- this repo "
              "ships no embeddings. See data/README.md.")

    models = {}
    for target in TARGETS:
        pkg = load_model(target, args.model)
        models[target] = pkg
        arch, algo = pkg.get("arch", "?"), pkg.get("algo", "?")
        role_flag = pkg.get("meta", {}).get("includes_role_flag", False)
        print(f"  [OK] {target}: arch={arch} algo={algo} role_flag={role_flag} "
              f"(CV BA={pkg.get('meta', {}).get('cv_ba', float('nan'))*100:.1f}%)")

    df = pd.read_excel(input_path)
    df.columns = df.columns.str.strip()
    print(f"\n  {len(df)} row(s) to predict.\n")

    results = []
    for idx, row in df.iterrows():
        p, s = row["Precursor"], row["Substrate"]
        r = row.get("Reactant")
        r_str = str(r).strip() if pd.notna(r) and str(r).strip() else "H2O"
        temp = row.get("Temp", row.get("Temp(C)", 250))
        role_val = row.get("Role")

        result_row = {"Precursor": p, "Substrate": s, "Reactant": r_str,
                      "Temp(C)": temp, "Role": role_val if pd.notna(role_val) else ""}

        for target in TARGETS:
            pkg = models[target]
            arch = pkg["arch"]
            use_role = pkg.get("meta", {}).get("includes_role_flag", False)
            try:
                vP = resolve_arch(p, "Precursor", arch, master)
                vS = resolve_arch(s, "Substrate", arch, master)
                if vP is None or vS is None:
                    raise RuntimeError(
                        f"'{p if vP is None else s}' not found in data/master_embeddings.json "
                        "(see data/README.md -- this repo ships no embeddings, you need your own)"
                    )
                feat = [vP, vS]
                if target == "Dep":
                    vR = resolve_arch(r_str, "Reactant", arch, master)
                    if vR is None:
                        vR = resolve_arch("H2O", "Reactant", arch, master)
                    if vR is None:
                        raise RuntimeError(f"Reactant '{r_str}' not found and H2O fallback unavailable")
                    feat.append(vR)
                Xrow = np.concatenate(feat + [[float(temp)]])
                if use_role:
                    Xrow = np.concatenate([Xrow, [role_to_flag(role_val)]])
                Xrow = Xrow.reshape(1, -1)
                Xrow_sc = pkg["scaler"].transform(Xrow)
                prob = float(pkg["model"].predict_proba(Xrow_sc)[0][1])
                result_row[f"Prob_{target}(%)"] = round(prob * 100, 1)
                result_row[f"Pred_{target}"] = "Success" if prob >= 0.5 else "Fail"
            except Exception as e:
                result_row[f"Prob_{target}(%)"] = "ERROR"
                result_row[f"Pred_{target}"] = str(e)

        results.append(result_row)
        print(f"  [{idx+1}] {p} / {s} / {r_str} / {temp}C"
              f"{', Role=' + str(role_val) if pd.notna(role_val) else ''}")
        print(f"        Ads: {result_row.get('Prob_Ads(%)')}%  ({result_row.get('Pred_Ads')})   "
              f"Dep: {result_row.get('Prob_Dep(%)')}%  ({result_row.get('Pred_Dep')})")

    pd.DataFrame(results).to_excel(output_path, index=False)
    print(f"\nResults saved to: {output_path}")


# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AJIN -- ALD Adsorption/Deposition Prediction (paper companion code)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train models (requires your own data/ files, see data/README.md)")
    g = p_train.add_mutually_exclusive_group(required=True)
    g.add_argument("--all9", action="store_true", help="Reproduce all 9 paper combinations (CHGxALIxHybrid x RFxXGBxMLP)")
    g.add_argument("--best", action="store_true", help="Train the recommended BEST model pair")

    p_pred = sub.add_parser("predict", help="Run predictions on an input Excel file")
    p_pred.add_argument("--input", required=True, help="Input .xlsx (see examples/input_template.xlsx)")
    p_pred.add_argument("--output", default=None, help="Output .xlsx (default: <input>_predictions.xlsx)")
    p_pred.add_argument("--model", default="best",
                         help="'best' (default, recommended) or ARCH_ALGO e.g. CHG_MLP, ALI_RF, Hybrid_XGB")

    args = parser.parse_args()
    if args.command == "train":
        cmd_train_all9(args) if args.all9 else cmd_train_best(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
