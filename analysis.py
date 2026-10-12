"""Final pipeline: leak-free cleaning (drop cholesterol + per-fold winsorization)
then the same three-method benchmark (heuristic rule vs gradient boosting vs
random forest).

This is a cleaned variant of heart-failure/main/scripts/unified-benchmark.py. It
keeps the *same methods* and the *same figures*, but adds a data-cleaning step
that drops cholesterol and winsorizes the continuous numeric columns -- WITHOUT
leaking test information:

  1. loads the Heart Failure dataset (heart.csv next to this file),
  2. CLEANS the data:
       - drops the Cholesterol column (and any cholesterol-derived fields),
       - treats RestingBP = 0 (impossible) as missing and median-imputes it,
       - winsorizes the continuous numeric columns to their 1.5*IQR fences
         (Age, RestingBP, MaxHR, Oldpeak).
     For the learned models BOTH the imputation median and the fences are fit on
     the TRAINING fold only and then applied to that fold's train + test rows, so
     no test statistic leaks into training. EDA figures use cohort-level medians
     and fences for a descriptive view only.
  3. renders dataset-overview figures: class balance, disease rate by category,
     numeric distributions by outcome, and the correlation matrix,
  4. benchmarks three analytic methods under IDENTICAL stratified 5-fold CV:
       - Heuristic baseline  hand-specified clinical rule (no learning, score >= 2)
       - Gradient boosting   boosted trees (sklearn defaults)
       - Random forest       bagged trees (sklearn defaults)
  5. scores them on a full metric panel (accuracy, precision, recall, specificity,
     F1, ROC-AUC, MCC) plus pooled out-of-fold confusion matrices and ROC curves,
  6. explains what drives the predictions: per-feature importances averaged over
     folds, next to the rule's own marker statistics.

Figures -> final_for_sure/figures/*.png

Usage
-----
    heart-failure/.venv/bin/python final_for_sure/analysis.py
    heart-failure/.venv/bin/python final_for_sure/analysis.py --folds 10 --out final_for_sure/results.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


# Self-contained: dataset cache lives next to this script (heart.csv).
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR / "heart.csv"
FIG_DIR = SCRIPT_DIR / "figures"

LABEL_COL = "HeartDisease"
K_FOLDS = 5
RANDOM_STATE = 42
RULE_CUTOFF = 2
DECISION_THRESHOLD = 0.5

# Cleaning: drop cholesterol, mark sentinel zeros as missing, then impute +
# winsorize the continuous numerics (all fitted per training fold).
DROP_COLUMNS = ["Cholesterol"]
NUMERIC_COLS = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
MODEL_NUMERIC_COLS = [c for c in NUMERIC_COLS if c not in DROP_COLUMNS]
# Sentinel codes that mean "missing" (physiologically impossible readings).
MISSING_SENTINELS = {"RestingBP": 0}
# Columns median-imputed (median fit on the training fold only).
IMPUTE_COLUMNS = ["RestingBP"]
# Continuous columns winsorized to 1.5*IQR fences. FastingBS is binary (0/1) so
# IQR clamping is meaningless for it.
CLAMP_COLUMNS = ["Age", "RestingBP", "MaxHR", "Oldpeak"]

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
# derived columns (mirrors features.py, minus the cholesterol-derived fields)
ENGINEERED_COLS = ["MaxHR_pct", "ST_Score", "ExAngina_ST", "ClinicalRiskScore"]

# Plot grouping: FastingBS is binary, so display it with the categorical features
# and keep it out of the continuous panel. (Model inputs are unchanged.)
CATEGORICAL_PLOT_COLS = CATEGORICAL_COLS + ["FastingBS"]
NUMERIC_PLOT_COLS = [c for c in MODEL_NUMERIC_COLS if c not in CATEGORICAL_PLOT_COLS]

RULE_METHOD_NAME = "Heuristic baseline"

METRIC_ORDER = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "mcc"]

METHOD_COLORS = {
    RULE_METHOD_NAME: "#55A868",
    "Gradient boosting": "#4C72B0",
    "Random forest": "#C44E52",
}


# ---------------------------------------------------------------------------
# Load + clean
# ---------------------------------------------------------------------------
def load_heart_failure_dataset() -> pd.DataFrame:
    """Return the raw dataset as a DataFrame (918 rows x 12 columns)."""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"expected dataset at {CSV_PATH}")
    return pd.read_csv(CSV_PATH)


def drop_cholesterol(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove cholesterol (and any cholesterol-derived) columns."""
    dropped = [c for c in DROP_COLUMNS if c in raw.columns]
    return raw.drop(columns=dropped), dropped


def cohort_fences(
    df: pd.DataFrame, columns: list[str]
) -> dict[str, tuple[float, float, int, int]]:
    """1.5*IQR fences over a full column, with clamp counts (descriptive/report)."""
    report: dict[str, tuple[float, float, int, int]] = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        below = int((df[col] < lo).sum())
        above = int((df[col] > hi).sum())
        report[col] = (float(lo), float(hi), below, above)
    return report


def clamp_with(df: pd.DataFrame, fences: dict[str, tuple[float, float, int, int]]) -> pd.DataFrame:
    """Clip columns to the supplied fences (returns a copy)."""
    out = df.copy()
    for col, (lo, hi, _, _) in fences.items():
        out[col] = out[col].clip(lo, hi)
    return out


def mark_missing(
    df: pd.DataFrame, sentinels: dict[str, int]
) -> tuple[pd.DataFrame, int]:
    """Turn sentinel codes (e.g. RestingBP = 0) into NaN. Returns (frame, n marked)."""
    out = df.copy()
    n = 0
    for col, val in sentinels.items():
        mask = out[col] == val
        n += int(mask.sum())
        out.loc[mask, col] = np.nan
    return out, n


def impute_train_medians(
    X_train: pd.DataFrame, X_test: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Median-impute columns using TRAINING-fold medians only (leak-free)."""
    Xtr, Xte = X_train.copy(), X_test.copy()
    medians: dict[str, float] = {}
    for col in columns:
        med = float(X_train[col].median())
        medians[col] = med
        Xtr[col] = Xtr[col].fillna(med)
        Xte[col] = Xte[col].fillna(med)
    return Xtr, Xte, medians


def descriptive_clean(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float], dict]:
    """Cohort-level median impute + winsorization, for EDA/figures only."""
    desc = base.copy()
    medians: dict[str, float] = {}
    for col in IMPUTE_COLUMNS:
        med = float(base[col].median())
        medians[col] = med
        desc[col] = desc[col].fillna(med)
    fences = cohort_fences(desc, CLAMP_COLUMNS)
    return clamp_with(desc, fences), medians, fences


def encode_features(base: pd.DataFrame) -> pd.DataFrame:
    """Model inputs: numeric columns plus one-hot categoricals (un-winsorized)."""
    onehot = pd.get_dummies(base[CATEGORICAL_COLS], drop_first=True, dtype=int)
    X = pd.concat(
        [base[MODEL_NUMERIC_COLS].reset_index(drop=True), onehot.reset_index(drop=True)],
        axis=1,
    )
    return X


# ---------------------------------------------------------------------------
# The heuristic baseline (no learning)
# ---------------------------------------------------------------------------
RULE_MARKERS = {
    "Asymptomatic chest pain": lambda df: df["ChestPainType"] == "ASY",
    "Exercise angina": lambda df: df["ExerciseAngina"] == "Y",
    "Fasting blood sugar > 120": lambda df: df["FastingBS"] == 1,
    "Flat/down-sloping ST": lambda df: df["ST_Slope"].isin(["Flat", "Down"]),
}


def rule_score(base: pd.DataFrame) -> np.ndarray:
    """ClinicalRiskScore: +1 per marker, 0-4. Uses only categorical columns."""
    score = np.zeros(len(base), dtype=int)
    for predicate in RULE_MARKERS.values():
        score += predicate(base).to_numpy().astype(int)
    return score


def engineer_features(base: pd.DataFrame) -> pd.DataFrame:
    """Add the project's derived columns for EDA correlations (EDA only)."""
    fe = base.copy()
    fe["MaxHR_pct"] = fe["MaxHR"] / (220 - fe["Age"]) * 100
    slope_code = {"Up": 0, "Flat": 1, "Down": 2}
    fe["ST_Score"] = fe["Oldpeak"] * (1 + fe["ST_Slope"].map(slope_code))
    fe["ExAngina_ST"] = (fe["ExerciseAngina"] == "Y").astype(int) * fe["ST_Score"]
    fe["ClinicalRiskScore"] = rule_score(base)
    return fe


# ---------------------------------------------------------------------------
# Leak-free per-fold winsorization
# ---------------------------------------------------------------------------
def train_fences(X_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Fit 1.5*IQR fences on the TRAINING fold only."""
    fences: dict[str, tuple[float, float]] = {}
    for col in CLAMP_COLUMNS:
        q1 = X_train[col].quantile(0.25)
        q3 = X_train[col].quantile(0.75)
        iqr = q3 - q1
        fences[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    return fences


def apply_train_fences(X: pd.DataFrame, fences: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Apply training-fold fences to a (train or test) slice."""
    out = X.copy()
    for col, (lo, hi) in fences.items():
        out[col] = out[col].clip(lo, hi)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metric_bundle(y_true: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    def safe(num, den):
        return float(num) / float(den) if den else 0.0

    return {
        "accuracy": safe(tp + tn, tp + tn + fp + fn),
        "precision": safe(tp, tp + fp),
        "recall": safe(tp, tp + fn),
        "specificity": safe(tn, tn + fp),
        "f1": safe(2 * tp, 2 * tp + fp + fn),
        "auc": float(roc_auc_score(y_true, score)) if len(set(y_true)) > 1 else 0.0,
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Cross-validation driver (shared by all three methods)
# ---------------------------------------------------------------------------
def run_cv(predict_fn, y: np.ndarray, folds: list, threshold: float):
    """Run k folds, return per-fold metric table + pooled OOF scores."""
    rows = []
    oof = np.empty(len(y), dtype=float)
    for fold, (tr, te) in enumerate(folds, start=1):
        score = predict_fn(tr, te)
        oof[te] = score
        m = metric_bundle(y[te], score, threshold)
        m["fold"] = fold
        m["n_test"] = len(te)
        m["n_pos"] = int(y[te].sum())
        rows.append(m)
    cols = ["fold", "n_test", "n_pos"] + METRIC_ORDER + ["tp", "fp", "tn", "fn"]
    return pd.DataFrame(rows)[cols], oof


def make_folds(y: np.ndarray, k: int, seed: int) -> list:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------
def _banner(k: int) -> None:
    print("=" * 82)
    print("  Final benchmark (drop cholesterol + leak-free winsorize) -- rule/GB/RF")
    print(f"  identical folds: stratified {k}-fold cross-validation")
    print("=" * 82)


def report_cleaning(
    dropped: list[str],
    n_missing: int,
    medians: dict[str, float],
    fences: dict,
    n: int,
) -> None:
    print("\n[0] Cleaning (leak-free)")
    print(f"  dropped columns      : {', '.join(dropped) if dropped else '(none)'}")
    print(f"  rows                 : {n}  (unchanged)")
    sentinels = ", ".join(f"{c}={v}" for c, v in MISSING_SENTINELS.items())
    print(f"  sentinel -> missing  : {sentinels}  ({n_missing} cell marked NaN)")
    print(f"  median imputation    : {', '.join(IMPUTE_COLUMNS)}"
          f"  (median fit per TRAINING fold -> train + test)")
    print(f"  winsorized columns   : {', '.join(CLAMP_COLUMNS)}  (1.5*IQR, per training fold)")
    print("  descriptive (EDA)    : cohort-level median + fences used for figures only")
    print("  cohort medians (EDA) : " + ", ".join(f"{c}={m:.1f}" for c, m in medians.items()))
    print(f"  {'column':<12s} {'lower':>8s} {'upper':>8s} {'low':>6s} {'high':>6s}  <- whole-cohort reference")
    for col, (lo, hi, below, above) in fences.items():
        print(f"  {col:<12s} {lo:>8.1f} {hi:>8.1f} {below:>6d} {above:>6d}")


def report_data(desc: pd.DataFrame, X: pd.DataFrame, y: np.ndarray) -> None:
    print("\n[1] Dataset overview")
    print(f"  patients : {len(desc)}")
    print(f"  cols     : {desc.shape[1]}  -> {', '.join(desc.columns)}")
    pos = int(y.sum())
    print(f"  labels   : disease={pos} ({pos / len(y) * 100:.1f}%)"
          f"  healthy={len(y) - pos} ({(len(y) - pos) / len(y) * 100:.1f}%)")
    print(f"  model X  : {X.shape[1]} numeric features "
          f"({len(MODEL_NUMERIC_COLS)} numeric + one-hot categoricals)")

    print("\n  numeric summary (mean by outcome; descriptive cleaned frame)")
    print(f"  {'feature':<12s} {'healthy mean':>13s} {'disease mean':>13s} {'delta':>8s}")
    for col in MODEL_NUMERIC_COLS:
        h = desc[desc[LABEL_COL] == 0][col].mean()
        d = desc[desc[LABEL_COL] == 1][col].mean()
        print(f"  {col:<12s} {h:>13.1f} {d:>13.1f} {d - h:>+8.1f}")

    print("\n  disease rate by category")
    for col in CATEGORICAL_COLS:
        rates = desc.groupby(col, observed=True)[LABEL_COL].mean() * 100
        spread = rates.max() - rates.min()
        levels = "  ".join(f"{k}={v:.0f}%" for k, v in rates.sort_values().items())
        print(f"  {col:<15s} spread {spread:>5.1f}pp   {levels}")


def report_method(name: str, folds: pd.DataFrame, k: int) -> None:
    print(f"\n[2] {name}")
    print(f"  {'fold':>4s} " + " ".join(f"{m:>11s}" for m in METRIC_ORDER))
    print("  " + "-" * (5 + 12 * len(METRIC_ORDER)))
    for _, r in folds.iterrows():
        print(f"  {int(r['fold']):>4d} " + " ".join(f"{r[m]:>11.3f}" for m in METRIC_ORDER))
    print("  " + "-" * (5 + 12 * len(METRIC_ORDER)))
    print(f"  {'mean':>4s} " + " ".join(f"{folds[m].mean():>11.3f}" for m in METRIC_ORDER))
    print(f"  {'sd':>4s} " + " ".join(f"{folds[m].std(ddof=1):>11.3f}" for m in METRIC_ORDER))


def report_pooled(name: str, y: np.ndarray, oof: np.ndarray, threshold: float) -> dict:
    m = metric_bundle(y, oof, threshold)
    print(f"\n[3] {name} -- pooled out-of-fold ({len(y)} patients)")
    print(f"  confusion: TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print("  " + "  ".join(f"{k}={m[k]:.3f}" for k in METRIC_ORDER))
    return m


def report_ranking(summaries: dict[str, pd.DataFrame]) -> list[str]:
    print("\n[4] Ranking (mean over folds)")
    print(f"  {'method':<26s} {'MCC':>8s} {'AUC':>8s} {'F1':>8s} {'acc':>8s} {'recall':>8s}")
    print("  " + "-" * 68)
    order = sorted(summaries, key=lambda n: summaries[n]["mcc"].mean(), reverse=True)
    for rank, name in enumerate(order, 1):
        d = summaries[name]
        print(f"  {rank}. {name:<22s} {d['mcc'].mean():>8.3f} {d['auc'].mean():>8.3f} "
              f"{d['f1'].mean():>8.3f} {d['accuracy'].mean():>8.3f} "
              f"{d['recall'].mean():>8.3f}")
    return order


def report_importance(gb_imp: pd.Series, rf_imp: pd.Series, desc: pd.DataFrame, top: int = 12) -> None:
    print(f"\n[5] What matters -- top {top} features by model importance")
    print(f"  {'feature':<24s} {'grad boosting':>14s} {'random forest':>14s}")
    print("  " + "-" * 54)
    for feat in gb_imp.head(top).index:
        print(f"  {feat:<24s} {gb_imp[feat]:>14.3f} {rf_imp.get(feat, 0.0):>14.3f}")

    print(f"\n  {RULE_METHOD_NAME} markers -- disease rate among carriers")
    overall = desc[LABEL_COL].mean() * 100
    print(f"  {'marker':<28s} {'carry':>7s} {'P(disease|yes)':>15s} {'swing':>8s}")
    for label, predicate in RULE_MARKERS.items():
        carries = predicate(desc)
        p_yes = desc[LABEL_COL][carries].mean() * 100
        p_no = desc[LABEL_COL][~carries].mean() * 100
        print(f"  {label:<28s} {carries.mean() * 100:>6.1f}% {p_yes:>14.1f}% "
              f"{p_yes - p_no:>+7.1f}pp")
    print(f"  (cohort average disease rate = {overall:.1f}%)")


def report_insights(order: list[str], summaries: dict[str, pd.DataFrame]) -> None:
    best = order[0]
    print("\n[6] Insights")
    print(f"  * best method by MCC: {best} "
          f"(MCC {summaries[best]['mcc'].mean():.3f}, AUC {summaries[best]['auc'].mean():.3f})")
    print("  * the rule uses only 4 columns yet tracks the ensembles closely ->")
    print("    most of the signal is carried by a handful of clinical flags")
    print("  * tree ensembles can use interactions and graded features")
    print("    (Oldpeak, MaxHR) that the rule does not use")
    print("  * cleaning: cholesterol dropped; RestingBP=0 median-imputed;"
          " numerics winsorized -- all per training fold")
    print("  * per-fold SD describes variation across these folds;")
    print("    it does not establish statistical superiority or clinical readiness")


# ---------------------------------------------------------------------------
# Figures -- dataset overview
# ---------------------------------------------------------------------------
def _save(fig, name: str) -> None:
    path = FIG_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    print(f"  saved figure -> {path}")


def _setup_style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")
    sns.set_palette("colorblind")
    return plt, sns


def fig_class_balance(desc: pd.DataFrame) -> None:
    plt, _ = _setup_style()
    counts = desc[LABEL_COL].value_counts().reindex([0, 1])
    colors = ["#4C72B0", "#C44E52"]
    labels = ["No disease", "Heart disease"]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", width=0.55)
    for bar, n in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, n + 8,
                f"n = {n}\n({n / len(desc) * 100:.1f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("patients")
    ax.set_title("Target distribution")
    fig.tight_layout()
    _save(fig, "final_eda1_class_balance.png")


def fig_category_rates(desc: pd.DataFrame) -> None:
    plt, _ = _setup_style()
    overall = desc[LABEL_COL].mean() * 100
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    for ax, col in zip(axes.ravel(), CATEGORICAL_PLOT_COLS):
        agg = (desc.groupby(col, observed=True)[LABEL_COL]
               .agg(["mean", "count"]).sort_values("mean"))
        agg["mean"] *= 100
        labels = [f"{i} (n={int(n)})" for i, n in zip(agg.index, agg["count"])]
        ax.barh(labels, agg["mean"], color="#C44E52", alpha=0.85, height=0.62)
        ax.axvline(overall, color="#444444", linestyle="--", linewidth=1)
        for y, v in enumerate(agg["mean"]):
            ax.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=8.5)
        ax.set_xlim(0, 105)
        ax.set_title(col, fontsize=11)
    fig.suptitle("Heart disease rate by category", fontsize=13)
    fig.tight_layout()
    _save(fig, "final_eda2_category_rates.png")


def fig_numeric_by_outcome(desc: pd.DataFrame) -> None:
    plt, sns = _setup_style()
    plot_df = desc.assign(Outcome=desc[LABEL_COL].map({0: "No disease", 1: "Heart disease"}))
    order = ["No disease", "Heart disease"]
    palette = {"No disease": "#4C72B0", "Heart disease": "#C44E52"}

    cols = NUMERIC_PLOT_COLS  # Age, RestingBP, MaxHR, Oldpeak (FastingBS is categorical)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, col in zip(axes.ravel(), cols):
        sns.violinplot(data=plot_df, x="Outcome", y=col, hue="Outcome",
                       order=order, hue_order=order, palette=palette,
                       alpha=0.35, inner=None, cut=0, legend=False, ax=ax)
        sns.boxplot(data=plot_df, x="Outcome", y=col, hue="Outcome",
                    order=order, hue_order=order, palette=palette,
                    width=0.16, showfliers=False, legend=False, ax=ax)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
    fig.suptitle("Continuous features vs Heart disease", fontsize=13)
    fig.tight_layout()
    _save(fig, "final_eda3_numeric_by_outcome.png")


def fig_correlation_matrix(desc: pd.DataFrame) -> None:
    plt, sns = _setup_style()
    fe = engineer_features(desc)
    cols = MODEL_NUMERIC_COLS + ENGINEERED_COLS + [LABEL_COL]
    corr = fe[cols].corr()

    fig, ax = plt.subplots(figsize=(10.5, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.75, "label": "Pearson correlation"},
                annot_kws={"fontsize": 7.5}, ax=ax)
    ax.set_title("Correlation matrix", fontsize=12)
    ax.tick_params(axis="x", labelsize=8.5, rotation=90)
    ax.tick_params(axis="y", labelsize=8.5)
    fig.tight_layout()
    _save(fig, "final_eda4_correlation_matrix.png")


# ---------------------------------------------------------------------------
# Figures -- benchmark
# ---------------------------------------------------------------------------
def fig_metric_bars(summaries: dict[str, pd.DataFrame], k: int) -> None:
    plt, _ = _setup_style()
    metrics = ["accuracy", "precision", "recall", "specificity", "f1", "auc", "mcc"]
    x = np.arange(len(metrics))
    width = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, (name, d) in enumerate(summaries.items()):
        means = [d[m].mean() for m in metrics]
        sds = [d[m].std(ddof=1) for m in metrics]
        ax.bar(x + (i - 1) * width, means, width, yerr=sds, capsize=3,
               label=name, color=METHOD_COLORS[name], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title(f"Metrics by method ({k}-fold CV)")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "final_bench_metric_bars.png")


def fig_boxplots(summaries: dict[str, pd.DataFrame], k: int) -> None:
    plt, _ = _setup_style()
    metrics = ["auc", "f1", "mcc"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))

    for ax, metric in zip(axes, metrics):
        names = list(summaries)
        data = [summaries[n][metric].to_numpy() for n in names]
        for pos, (name, vals) in enumerate(zip(names, data)):
            ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True,
                       medianprops=dict(color="white", linewidth=1.6),
                       boxprops=dict(facecolor=METHOD_COLORS[name], alpha=0.9),
                       whiskerprops=dict(color="#555555"),
                       capprops=dict(color="#555555"),
                       flierprops=dict(marker="o", markersize=4, alpha=0.6))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
        ax.set_title(f"{metric.upper()} per fold (k={k})", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0.4, 1.0)
    fig.suptitle("Per-fold stability", fontsize=12)
    fig.tight_layout()
    _save(fig, "final_bench_boxplots.png")


def fig_roc(oof: dict[str, np.ndarray], y: np.ndarray) -> None:
    plt, _ = _setup_style()
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.plot([0, 1], [0, 1], ":", color="#888888", linewidth=1.2,
            label="random guess (AUC 0.500)")
    for name, score in oof.items():
        fpr, tpr, _ = roc_curve(y, score)
        auc = roc_auc_score(y, score)
        style = "-o" if name == RULE_METHOD_NAME else "-"
        ax.plot(fpr, tpr, style, color=METHOD_COLORS[name], linewidth=2.1,
                markersize=5, label=f"{name} (AUC {auc:.3f})")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (out-of-fold)")
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "final_bench_roc.png")


def fig_confusion(pooled: dict[str, dict]) -> None:
    plt, _ = _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (name, m) in zip(axes, pooled.items()):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        fontsize=15, color="#111111" if cm[i, j] < cm.max() else "white")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred healthy", "pred disease"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["true healthy", "true disease"])
        ax.set_title(f"{name}\nMCC {m['mcc']:.3f}  acc {m['accuracy']:.3f}", fontsize=10)
    fig.suptitle("Confusion matrices (out-of-fold)", fontsize=12)
    fig.tight_layout()
    _save(fig, "final_bench_confusion.png")


def fig_importance(gb_imp: pd.Series, rf_imp: pd.Series, desc: pd.DataFrame) -> None:
    plt, _ = _setup_style()
    top = 12
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))

    for ax, (title, imp, color) in zip(
        axes[:2],
        [("Gradient boosting", gb_imp, "#4C72B0"),
         ("Random forest", rf_imp, "#C44E52")],
    ):
        head = imp.head(top)[::-1]
        ax.barh(head.index, head.values, color=color, alpha=0.9, height=0.68)
        ax.set_title(f"{title} -- importance", fontsize=11)
        ax.set_xlabel("mean importance over folds")
        ax.tick_params(axis="y", labelsize=8.5)

    ax = axes[2]
    swings, labels = [], []
    for label, predicate in RULE_MARKERS.items():
        carries = predicate(desc)
        swings.append((desc[LABEL_COL][carries].mean() - desc[LABEL_COL][~carries].mean()) * 100)
        labels.append(label)
    order = np.argsort(swings)
    ax.barh([labels[i] for i in order], [swings[i] for i in order],
            color="#55A868", alpha=0.9, height=0.6)
    for y, i in enumerate(order):
        ax.text(swings[i] + 1, y, f"{swings[i]:.0f}pp", va="center", fontsize=8.5)
    ax.set_xlabel("P(disease | marker) - P(disease | no marker)")
    ax.set_title(f"{RULE_METHOD_NAME} markers -- swing", fontsize=11)
    ax.set_xlim(0, 75)
    ax.tick_params(axis="y", labelsize=8.5)

    fig.suptitle("Feature importance vs rule markers", fontsize=13)
    fig.tight_layout()
    _save(fig, "final_bench_importance.png")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folds", type=int, default=K_FOLDS, help="number of CV folds")
    p.add_argument("--seed", type=int, default=RANDOM_STATE, help="CV random state")
    p.add_argument("--cutoff", type=int, default=RULE_CUTOFF, help="rule score cutoff")
    p.add_argument("--out", type=Path, default=SCRIPT_DIR / "results.csv",
                   help="CSV of per-fold summaries")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    raw = load_heart_failure_dataset()
    if LABEL_COL not in raw.columns:
        raise KeyError(f"dataset must contain a '{LABEL_COL}' label column")

    base, dropped = drop_cholesterol(raw)
    base, n_missing = mark_missing(base, MISSING_SENTINELS)
    desc, medians, fences = descriptive_clean(base)  # EDA/figures only

    y = base[LABEL_COL].to_numpy()
    X = encode_features(base)                        # NaN kept; imputed per fold
    folds = make_folds(y, args.folds, args.seed)
    score = rule_score(base)

    _banner(args.folds)
    report_cleaning(dropped, n_missing, medians, fences, len(base))
    report_data(desc, X, y)

    print("\n  rendering dataset-overview figures")
    fig_class_balance(desc)
    fig_category_rates(desc)
    fig_numeric_by_outcome(desc)
    fig_correlation_matrix(desc)

    # ---- three methods on identical folds --------------------------------
    summaries: dict[str, pd.DataFrame] = {}
    oofs: dict[str, np.ndarray] = {}
    thresholds: dict[str, float] = {}

    # 1. heuristic baseline (no fitting, no numeric columns -> already leak-free)
    rule_folds, rule_oof = run_cv(lambda tr, te: score[te], y, folds, args.cutoff)
    summaries[RULE_METHOD_NAME] = rule_folds
    oofs[RULE_METHOD_NAME] = rule_oof
    thresholds[RULE_METHOD_NAME] = args.cutoff
    report_method(RULE_METHOD_NAME, rule_folds, args.folds)

    # 2 and 3. learned tree ensembles with per-fold winsorization
    gb_imp, rf_imp = [], []
    for name, factory, store in (
        ("Gradient boosting", lambda: GradientBoostingClassifier(random_state=args.seed), gb_imp),
        ("Random forest", lambda: RandomForestClassifier(random_state=args.seed), rf_imp),
    ):
        def predict(tr, te, factory=factory, store=store):
            Xtr, Xte, _ = impute_train_medians(X.iloc[tr], X.iloc[te], IMPUTE_COLUMNS)
            fences = train_fences(Xtr)                    # fences fit on TRAIN only
            Xtr = apply_train_fences(Xtr, fences)
            Xte = apply_train_fences(Xte, fences)         # same fences -> no leak
            model = factory()
            model.fit(Xtr, y[tr])
            store.append(model.feature_importances_)
            return model.predict_proba(Xte)[:, 1]

        tbl, oof = run_cv(predict, y, folds, DECISION_THRESHOLD)
        summaries[name] = tbl
        oofs[name] = oof
        thresholds[name] = DECISION_THRESHOLD
        report_method(name, tbl, args.folds)

    print(f"\n[2b] Shared design: {args.folds} stratified folds, same split for all "
          f"methods, seed={args.seed}, rule cutoff={args.cutoff}")
    print("     learned models: median-impute + winsorize fitted per training fold")

    pooled = {
        name: report_pooled(name, y, oofs[name], thresholds[name]) for name in summaries
    }
    order = report_ranking(summaries)

    gb_series = pd.Series(np.mean(gb_imp, axis=0), index=X.columns).sort_values(ascending=False)
    rf_series = pd.Series(np.mean(rf_imp, axis=0), index=X.columns).sort_values(ascending=False)
    report_importance(gb_series, rf_series, desc)

    print("\n  rendering benchmark + insight figures")
    fig_metric_bars(summaries, args.folds)
    fig_boxplots(summaries, args.folds)
    fig_roc(oofs, y)
    fig_confusion(pooled)
    fig_importance(gb_series, rf_series, desc)

    report_insights(order, summaries)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [t.assign(method=name) for name, t in summaries.items()], ignore_index=True
    )
    combined.to_csv(out, index=False)
    print(f"\nsaved per-fold results -> {out}")
    print("\n" + "=" * 82)


if __name__ == "__main__":
    main()
