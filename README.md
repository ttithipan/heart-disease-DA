# Heart Disease Prediction — Final Comparative Analytics

Predictive analytics on **918 clinical records**: a hand-written clinical rule
compared against two learned tree ensembles (gradient boosting and random forest)
under **identical, leak-free, stratified 5-fold cross-validation**.

Research question: *can a simple, explainable rule compete with learned models?*

---

## Dataset

| | |
|---|---|
| Source | Fedesoriano (2021), *Heart Failure Prediction*, Kaggle — CC BY 4.0 |
| Size | 918 rows × 12 columns (11 features + binary target) |
| Target | `HeartDisease` — 508 positive (55.3%) / 410 negative (44.7%) |
| License | CC BY 4.0 |

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

---

## Pipeline (leak-free cleaning)

1. **Drop cholesterol.** `Cholesterol` (and any cholesterol-derived fields) is
   removed entirely — its impossible `0` sentinels (172 rows, 18.7%) make it an
   unreliable predictor.
2. **Mark impossible values as missing.** The one physiologically impossible
   `RestingBP = 0` reading is converted to missing and **median-imputed**
   (cohort median = 130 mmHg).
3. **Winsorize continuous numerics.** `Age`, `RestingBP`, `MaxHR`, `Oldpeak` are
   clamped to their **1.5×IQR fences**.
4. **Leak-free by construction.** For the learned models, **both** the
   imputation median **and** the IQR fences are fit on **each training fold
   only**, then applied to that fold's training *and* test rows — no test
   statistic ever influences training. (EDA figures use cohort-level
   medians/fences for a descriptive view only; they are never used for scoring.
   `FastingBS` is binary, so IQR clamping is meaningless for it.)
5. **Encoding.** 5 numeric features + one-hot categoricals (`drop_first`) = **14
   encoded model inputs**.

---

## Methods

| Method | Type | Inputs | Fitting |
|---|---|---|---|
| **Heuristic baseline** | hand-specified rule (`ClinicalRiskScore`) | 4 raw columns: chest pain, exercise angina, fasting blood sugar, ST slope | none |
| **Gradient boosting** | boosted shallow trees (sklearn defaults) | 14 encoded inputs | refit per fold |
| **Random forest** | bagged deep trees (sklearn defaults) | 14 encoded inputs | refit per fold |

The rule adds **+1 per marker** (score 0–4); `score ≥ 2` is the "at risk"
verdict. All three methods share the exact same `StratifiedKFold(5, shuffle,
seed=42)` folds. Primary metric: **MCC** (uses all four confusion-matrix cells).

---

## Results (mean over 5 folds)

| Method | MCC | AUC | F1 | Accuracy | Recall | Specificity |
|---|---|---|---|---|---|---|
| **Random forest** | **0.714** | 0.924 | 0.874 | 0.858 | 0.886 | 0.824 |
| Gradient boosting | 0.706 | 0.918 | 0.870 | 0.854 | 0.884 | 0.817 |
| Heuristic baseline | 0.697 | 0.904 | 0.865 | 0.850 | 0.868 | 0.827 |

### Pooled out-of-fold (all 918 patients)

| Method | TP | FP | TN | FN | MCC |
|---|---|---|---|---|---|
| Random forest | 450 | 72 | 338 | 58 | 0.713 |
| Gradient boosting | 449 | 75 | 335 | 59 | 0.704 |
| Heuristic baseline | 441 | 71 | 339 | 67 | 0.696 |

Random forest has the highest mean MCC and catches the most disease cases
(450 vs 441 for the rule); the 4-marker rule stays within ~0.02 MCC of the full
ensembles. These are **descriptive, dataset-specific** comparisons — five folds
alone do not establish statistical or clinical superiority.

Top learned features: `ST_Slope_Up` (dominant), `Oldpeak`, `MaxHR`.

---

## Figures

Generated into `figures/`:

| File | Shows |
|---|---|
| `final_eda1_class_balance.png` | target distribution |
| `final_eda2_category_rates.png` | disease rate by category |
| `final_eda3_numeric_by_outcome.png` | numeric distributions by outcome |
| `final_eda4_correlation_matrix.png` | feature/target correlations |
| `final_bench_metric_bars.png` | metric comparison (mean ± SD) |
| `final_bench_boxplots.png` | per-fold stability |
| `final_bench_roc.png` | pooled out-of-fold ROC curves |
| `final_bench_confusion.png` | pooled confusion matrices |
| `final_bench_importance.png` | learned importances vs rule markers |

---

## Reproduce

```bash
# from the repository root
python analysis.py                       # writes figures/ + results.csv
python analysis.py --folds 10 --out results.csv
```

Requires: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

---

## Notes & limitations

- **Leak-free imputation + winsorization** is the key methodological choice
  here: both the missing-value median and the IQR fences are fit on training
  folds only. (Earlier drafts that processed the whole cohort, or merely clamped
  the impossible zero, shifted results slightly — RF MCC 0.723 → **0.714**, GB
  0.706 unchanged.)
- Imputing/winsorizing barely changes the tree models because they split on
  thresholds and are relatively robust to extreme values.
- The `ClinicalRiskScore` rule is a project-defined heuristic, **not** a
  validated clinical score. Results are internal cross-validation only and are
  not evidence of clinical readiness.

*Dataset: Fedesoriano (2021), Heart Failure Prediction, Kaggle — CC BY 4.0.*
