# Bank Marketing — End-to-End Machine Learning Project

## Problem Statement

This project builds a complete, reproducible Machine Learning pipeline to predict whether a bank client will **subscribe to a term deposit** (`yes`/`no`) based on demographic, financial, and campaign contact data.

This is a **binary classification** problem with real-world class imbalance (~88% No, ~12% Yes), making it a meaningful challenge requiring thoughtful metric selection and handling of the imbalanced target.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI Bank Marketing Dataset |
| File | `bank.csv` |
| Rows | 45,211 |
| Columns | 17 (16 features + 1 target) |
| Target | `deposit` → `yes` or `no` |
| Separator | Comma (`,`) |

### Feature Summary

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numerical | Client age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Numerical | Average yearly balance (EUR) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numerical | Last contact day of month |
| `month` | Categorical | Last contact month |
| `duration` | Numerical | Last contact duration (seconds) |
| `campaign` | Numerical | Number of contacts this campaign |
| `pdays` | Numerical | Days since last contact (-1 = never) |
| `previous` | Numerical | Number of previous contacts |
| `poutcome` | Categorical | Previous campaign outcome |
| `deposit` | **Target** | Term deposit subscribed? |

---

## Data Cleaning

All cleaning decisions were documented with rationale:

1. **No literal NaN values** — the dataset uses `'unknown'` as a missing indicator in categorical columns (`job`, `education`, `contact`, `poutcome`). These were **kept as a separate category** because missingness itself is informative (e.g., `poutcome='unknown'` = no prior campaign).

2. **Duplicates** — checked and removed if present (0 found in original data).

3. **`pdays = -1` sentinel** — means the client was never contacted before. Retained and addressed in Feature Engineering by creating a binary flag and replacing -1 with 999.

4. **Outliers** — detected via IQR × 1.5 method. **Retained** because:
   - Tree-based models are inherently robust to outliers.
   - Extreme values (e.g., very long call durations) carry genuine predictive signal.
   - Feature scaling (StandardScaler) handles numeric range for distance-based models.

5. **Data types** — all types were already correct (`int64` for numeric, `object` for categorical). No changes needed.

---

## EDA Insights

Key findings from exploratory analysis:

1. **Class Imbalance** — ~88% No / ~12% Yes. ROC-AUC and F1-Macro were chosen as primary metrics.
2. **`duration` is the strongest single predictor** — longer calls correlate strongly with subscription. Note: duration is only known post-call, so treat as a benchmark feature.
3. **`balance`** — subscribing clients tend to have higher average balances.
4. **Seasonal patterns** — March, September, October, and December show the highest subscription rates.
5. **Contact type** — Cellular contact significantly outperforms telephone and unknown.
6. **Previous outcome** — `poutcome='success'` dramatically increases conversion probability.
7. **Job type** — Students and retired clients subscribe at higher rates.
8. **`pdays=-1`** — the majority of clients were never previously contacted, making this an important binary flag.

Plots are saved to the `plots/` directory (10 visualizations total).

---

## Feature Engineering

| Feature | Description | Rationale |
|---------|-------------|-----------|
| `was_contacted_before` | Binary: 1 if `pdays != -1` | Captures prior contact history cleanly |
| `pdays_clean` | `pdays` with -1 replaced by 999 | Removes sentinel, enables numeric use |
| `contact_ratio` | `campaign / (previous + 1)` | Measures current vs. historical contact intensity |
| `age_group` | Binned age (<30, 30-40, 40-50, 50-60, 60+) | Captures non-linear age effects |
| `balance_group` | Binned balance (negative/low/medium/high) | Segments financial tier |
| `deposit_enc` | Target: yes→1, no→0 | Binary encoding for classification |

**Dropped**: `pdays` (replaced by derived features to avoid redundancy)

**Encoding strategy**: `OneHotEncoder(handle_unknown='ignore')` for all categoricals within a `ColumnTransformer` pipeline — preventing data leakage by fitting only on training data.

---

## Models Tested

Six baseline models were evaluated using 5-Fold Stratified Cross-Validation:

| Model | Notes |
|-------|-------|
| Logistic Regression | Linear baseline, `class_weight='balanced'` |
| Decision Tree | Non-linear, interpretable, `class_weight='balanced'` |
| Random Forest | Ensemble of trees, robust to imbalance |
| Gradient Boosting | Boosting ensemble, strong on tabular data |
| AdaBoost | Adaptive boosting, SAMME algorithm |
| K-Nearest Neighbors | Distance-based, included for diversity |

The top 2 models by CV AUC were selected for hyperparameter tuning.

---

## Evaluation Metrics

| Metric | Rationale |
|--------|-----------|
| **ROC-AUC** | Primary metric — threshold-independent, handles imbalance well |
| **F1 Macro** | Secondary — balances precision & recall across both classes |
| **Accuracy** | Reported but de-emphasized (misleading with imbalance) |
| **Precision** | Minimizes false positives (wasted campaign calls) |
| **Recall** | Minimizes missed subscribers |

---

## Hyperparameter Tuning

- **Method**: `RandomizedSearchCV` (for RF, GB) and `GridSearchCV` (for LR, DT, KNN)
- **Cross-validation**: 5-Fold Stratified CV on the training set
- **Scoring**: `roc_auc`
- **Iterations**: n_iter=30 for RandomizedSearchCV
- **Overfitting prevention**: Parameters were tuned exclusively on the training set; the test set remained **completely unseen** until final evaluation.

---

## Final Model

The final model was selected based on the **highest ROC-AUC on the held-out test set**.

See `model_comparison_results.csv` for the full comparison table.

The winning pipeline is saved as `final_model_pipeline.joblib` and contains:
```
ColumnTransformer (preprocessing)
  ├── Numerical: SimpleImputer(median) → StandardScaler
  └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder
└── Classifier (best model)
```

---

## Final Results

See terminal output and `model_comparison_results.csv` for exact numbers after running the pipeline.

---

## How to Run the Project

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

### Run the Pipeline

```bash
cd bank-marketing
python bank_ml_pipeline.py
```

This will:
1. Load and inspect `bank.csv`
2. Clean the data
3. Run EDA and save 10 plots to `plots/`
4. Engineer features
5. Split data (80/20 stratified)
6. Train 6 baseline models with 5-Fold CV
7. Tune top 2 models
8. Evaluate all models on the test set
9. Save the best pipeline to `final_model_pipeline.joblib`
10. Save comparison results to `model_comparison_results.csv`

### Use the Saved Model

```python
import joblib
import pandas as pd

# Load
pipeline = joblib.load('final_model_pipeline.joblib')

# Predict on new raw data (same columns as bank.csv, minus 'deposit')
new_data = pd.DataFrame([{
    'age': 35, 'job': 'management', 'marital': 'married',
    'education': 'tertiary', 'default': 'no', 'balance': 1500,
    'housing': 'yes', 'loan': 'no', 'contact': 'cellular',
    'day': 15, 'month': 'may', 'duration': 300, 'campaign': 1,
    'pdays': -1, 'previous': 0, 'poutcome': 'unknown',
    # Derived features (also needed):
    'was_contacted_before': 0, 'pdays_clean': 999,
    'contact_ratio': 1.0, 'age_group': '30_40', 'balance_group': 'medium'
}])

predictions   = pipeline.predict(new_data)       # [0] or [1]
probabilities = pipeline.predict_proba(new_data)[:, 1]  # P(yes)
```

---

## Project Structure

```
bank-marketing/
├── bank.csv                          # Raw dataset
├── bank_ml_pipeline.py               # Complete ML pipeline (all 12 tasks)
├── final_model_pipeline.joblib       # Saved best model + preprocessing
├── model_comparison_results.csv      # All models comparison table
├── README.md                         # This file
└── plots/
    ├── 01_target_distribution.png
    ├── 02_numerical_distributions.png
    ├── 03_categorical_analysis.png
    ├── 04_correlation_heatmap.png
    ├── 05_boxplots_by_target.png
    ├── 06_monthly_contact_trends.png
    ├── 07_baseline_model_comparison.png
    ├── 08_confusion_matrix_roc.png
    ├── 09_feature_importance.png
    └── 10_final_model_comparison.png
```
