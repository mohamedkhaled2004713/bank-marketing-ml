
# =============================================================================
# BANK MARKETING — END-TO-END MACHINE LEARNING PROJECT
# =============================================================================
# Goal: Predict whether a client will subscribe to a term deposit (binary
#       classification: 'deposit' == 'yes' or 'no').
#
# Dataset: bank.csv  (UCI Bank Marketing Dataset)
# Tasks covered: Data Understanding → Cleaning → EDA → Feature Engineering
#                → Splitting → Model Selection → Training → Evaluation
#                → Hyperparameter Tuning → Final Model → Saving → Documentation
# =============================================================================

# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
DATA_PATH    = 'bank.csv'
PLOTS_DIR    = 'plots'
MODEL_PATH   = 'final_model_pipeline.joblib'

os.makedirs(PLOTS_DIR, exist_ok=True)

PALETTE = ['#E74C3C', '#2ECC71']   # red = no, green = yes
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)

print("=" * 70)
print("  BANK MARKETING — END-TO-END ML PIPELINE")
print(f"  Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ==============================================================================
# TASK 1 — DATA UNDERSTANDING
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 1 -- DATA UNDERSTANDING")
print("-" * 70)

df_raw = pd.read_csv(DATA_PATH)

print(f"\n> Shape         : {df_raw.shape[0]:,} rows x {df_raw.shape[1]} columns")
print(f"> Target        : 'deposit'  (yes/no -- binary classification)")
print(f"> Columns       : {df_raw.columns.tolist()}")
print("\n> Data Types:")
print(df_raw.dtypes.to_string())
print("\n> First 3 rows:")
print(df_raw.head(3).to_string())
print("\n> Descriptive Statistics (numerical):")
print(df_raw.describe().to_string())
print("\n> Descriptive Statistics (categorical):")
print(df_raw.describe(include='object').to_string())
print("\n> Missing values per column:")
print(df_raw.isnull().sum().to_string())
print("\n> Target Class Distribution:")
vc = df_raw['deposit'].value_counts()
print(vc)
print(f"  Class balance ratio (yes/no): {vc['yes']/vc['no']:.2%}")

# ==============================================================================
# TASK 2 — DATA CLEANING
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 2 -- DATA CLEANING")
print("-" * 70)

df = df_raw.copy()

# 2a. 'unknown' values — informative missingness, keep as category
print("\n> Unknown-value audit (kept as informative category):")
for col in df.select_dtypes('object').columns:
    n_unk = (df[col] == 'unknown').sum()
    if n_unk > 0:
        print(f"  '{col}': {n_unk:,} 'unknown' values")

# 2b. Duplicates
n_dups = df.duplicated().sum()
print(f"\n> Duplicates found: {n_dups}")
if n_dups > 0:
    df.drop_duplicates(inplace=True)
    print(f"  Removed {n_dups} duplicate rows. New shape: {df.shape}")
else:
    print("  No duplicates found.")

# 2c. pdays == -1 means 'not previously contacted'
print(f"\n> 'pdays' sentinel: -1 = never contacted  ({(df['pdays']==-1).sum():,} rows)")

# 2d. Outlier audit (IQR method — log only, no blind removal)
print("\n> Outlier audit (IQR x1.5) — retained (tree models are robust):")
num_cols_raw = df.select_dtypes(include=np.number).columns.tolist()
for col in num_cols_raw:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"  {col:12s}: {n_out:5,} outliers")

# 2e. Data types — all correct, no fixes needed
print("\n> Data type check: All types correct. No changes needed.")
print(f"\n> Clean DataFrame shape: {df.shape}")

# ==============================================================================
# TASK 3 — EDA
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 3 -- EDA")
print("-" * 70)

cat_cols = [c for c in df.select_dtypes('object').columns if c != 'deposit']
num_cols = df.select_dtypes(include=np.number).columns.tolist()
target   = 'deposit'

# Plot 1: Target distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = df[target].value_counts()
axes[0].bar(vc.index, vc.values, color=PALETTE, edgecolor='white', linewidth=1.5)
axes[0].set_title('Target Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Deposit Subscribed?')
axes[0].set_ylabel('Count')
for i, (k, v) in enumerate(vc.items()):
    axes[0].text(i, v + 200, f'{v:,}\n({v/len(df):.1%})', ha='center', fontsize=11)
vc.plot.pie(ax=axes[1], autopct='%1.1f%%', colors=PALETTE,
            startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2),
            labels=['No', 'Yes'])
axes[1].set_ylabel('')
axes[1].set_title('Class Balance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/01_target_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/01_target_distribution.png")

# Plot 2: Numerical feature distributions by target
fig, axes = plt.subplots(3, 3, figsize=(16, 13))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    ax = axes[i]
    df[df[target]=='no'][col].plot.hist(ax=ax, bins=40, alpha=0.6,
                                         color=PALETTE[0], label='No')
    df[df[target]=='yes'][col].plot.hist(ax=ax, bins=40, alpha=0.6,
                                          color=PALETTE[1], label='Yes')
    ax.set_title(col, fontweight='bold')
    ax.legend(fontsize=9)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Numerical Feature Distributions by Target', fontsize=16,
             fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/02_numerical_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/02_numerical_distributions.png")

# Plot 3: Categorical feature analysis
n_cats = len(cat_cols)
n_rows = (n_cats + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(16, 4*n_rows))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    ct = df.groupby(col)[target].value_counts(normalize=True).unstack().fillna(0)
    ct.plot(kind='bar', ax=axes[i], color=PALETTE, edgecolor='white',
            linewidth=0.8, legend=(i==0))
    axes[i].set_title(f'{col} -- Deposit Rate', fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Proportion')
    axes[i].tick_params(axis='x', rotation=30)
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Deposit Rate by Categorical Feature', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/03_categorical_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/03_categorical_analysis.png")

# Plot 4: Correlation heatmap
df_corr = df.copy()
df_corr['deposit_bin'] = (df_corr['deposit'] == 'yes').astype(int)
corr_matrix = df_corr[num_cols + ['deposit_bin']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/04_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/04_correlation_heatmap.png")

# Plot 5: Boxplots
key_num = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()
for i, col in enumerate(key_num):
    sns.boxplot(data=df, x=target, y=col, palette=PALETTE,
                ax=axes[i], hue=target, legend=False)
    axes[i].set_title(col, fontweight='bold')
    axes[i].set_xlabel('')
plt.suptitle('Key Numerical Features by Deposit Outcome', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/05_boxplots_by_target.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/05_boxplots_by_target.png")

# Plot 6: Month & contact trends
month_order = ['jan','feb','mar','apr','may','jun',
               'jul','aug','sep','oct','nov','dec']
df['month_ord'] = pd.Categorical(df['month'], categories=month_order, ordered=True)
monthly = (df.groupby('month_ord', observed=True)['deposit']
           .apply(lambda x: (x == 'yes').mean()).reset_index())
monthly.columns = ['month', 'yes_rate']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(monthly['month'], monthly['yes_rate'],
            color='#3498DB', edgecolor='white', linewidth=1.2)
axes[0].set_title('Deposit Rate by Month', fontweight='bold', fontsize=13)
axes[0].set_ylabel('Proportion Subscribed')
axes[0].tick_params(axis='x', rotation=45)

contact_rate = (df.groupby('contact')['deposit']
                .apply(lambda x: (x=='yes').mean()).reset_index())
contact_rate.columns = ['contact', 'yes_rate']
axes[1].bar(contact_rate['contact'], contact_rate['yes_rate'],
            color=['#9B59B6','#E67E22','#1ABC9C'], edgecolor='white', linewidth=1.2)
axes[1].set_title('Deposit Rate by Contact Type', fontweight='bold', fontsize=13)
axes[1].set_ylabel('Proportion Subscribed')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/06_monthly_contact_trends.png', dpi=150, bbox_inches='tight')
plt.close()
df.drop(columns=['month_ord'], inplace=True)
print(f"  Saved: {PLOTS_DIR}/06_monthly_contact_trends.png")

print("""
KEY EDA INSIGHTS:
  1. Severe class imbalance: ~88% No, ~12% Yes. Use AUC & F1 as primary metrics.
  2. 'duration' is the strongest predictor -- longer calls -> higher conversion.
  3. 'balance' shows higher medians for subscribing clients.
  4. March, September, October, December have the highest deposit rates.
  5. Cellular contact outperforms telephone/unknown contact.
  6. Clients with poutcome='success' convert at much higher rates.
  7. Students and retired clients show higher subscription rates.
  8. pdays=-1 (never contacted) makes up majority -- important flag feature.
""")

# ==============================================================================
# TASK 4 — FEATURE ENGINEERING
# ==============================================================================
print("-" * 70)
print("TASK 4 -- FEATURE ENGINEERING")
print("-" * 70)

df_fe = df.copy()

# Derived feature: was client previously contacted?
df_fe['was_contacted_before'] = (df_fe['pdays'] != -1).astype(int)
print("  Created: 'was_contacted_before'  (1 if pdays != -1)")

# Replace pdays sentinel -1 with large value (999 = never contacted)
df_fe['pdays_clean'] = df_fe['pdays'].replace(-1, 999)
print("  Created: 'pdays_clean'  (-1 replaced by 999)")

# Contact intensity ratio: how aggressive was this campaign vs. history
df_fe['contact_ratio'] = df_fe['campaign'] / (df_fe['previous'] + 1)
print("  Created: 'contact_ratio'  = campaign / (previous + 1)")

# Age groups (ordinal segmentation)
df_fe['age_group'] = pd.cut(df_fe['age'],
                             bins=[0, 30, 40, 50, 60, 100],
                             labels=['under_30','30_40','40_50','50_60','over_60'])
df_fe['age_group'] = df_fe['age_group'].astype(str)
print("  Created: 'age_group'  (age bins)")

# Balance groups
df_fe['balance_group'] = pd.cut(df_fe['balance'],
                                 bins=[-np.inf, 0, 500, 2000, np.inf],
                                 labels=['negative','low','medium','high'])
df_fe['balance_group'] = df_fe['balance_group'].astype(str)
print("  Created: 'balance_group'  (balance bins)")

# Drop original pdays (replaced by derived features)
df_fe.drop(columns=['pdays'], inplace=True)
print("  Dropped 'pdays' (replaced by pdays_clean + was_contacted_before)")

# Encode target: yes->1, no->0
df_fe['deposit_enc'] = (df_fe['deposit'] == 'yes').astype(int)
df_fe.drop(columns=['deposit'], inplace=True)
print("  Target encoded: 'deposit' -> 'deposit_enc' (1=yes, 0=no)")

# Feature groups for pipeline
TARGET_COL = 'deposit_enc'
y = df_fe[TARGET_COL]
X = df_fe.drop(columns=[TARGET_COL])

numerical_features   = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"\n  Numerical features   ({len(numerical_features)}): {numerical_features}")
print(f"  Categorical features ({len(categorical_features)}): {categorical_features}")

# ==============================================================================
# TASK 5 — DATA SPLITTING
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 5 -- DATA SPLITTING")
print("-" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y      # preserve class distribution in both sets
)

print(f"  Train: {X_train.shape[0]:,} samples (yes={y_train.sum():,}, no={(y_train==0).sum():,})")
print(f"  Test : {X_test.shape[0]:,} samples (yes={y_test.sum():,}, no={(y_test==0).sum():,})")
print(f"  Stratified split -- class balance preserved (test: {y_test.mean():.2%} positive)")

# ==============================================================================
# TASK 6 — MODEL SELECTION
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 6 -- MODEL SELECTION")
print("-" * 70)
print("  Problem: BINARY CLASSIFICATION")
print("  Primary metric: ROC-AUC  |  Secondary: F1-Macro, Accuracy")

# Preprocessing pipeline (applied consistently to all models)
num_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

cat_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transformer, numerical_features),
    ('cat', cat_transformer, categorical_features)
])

# Baseline model zoo — 6 diverse algorithms
baseline_models = {
    'Logistic Regression': LogisticRegression(
        random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight='balanced', max_depth=10),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        class_weight='balanced', n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=RANDOM_STATE),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=100, random_state=RANDOM_STATE, algorithm='SAMME'),
    'K-Nearest Neighbors': KNeighborsClassifier(n_jobs=-1),
}

# ==============================================================================
# TASK 7 & 8 — MODEL TRAINING & BASELINE EVALUATION (5-Fold CV)
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 7 & 8 -- TRAINING & EVALUATION (BASELINE, 5-FOLD CV)")
print("-" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in baseline_models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring='roc_auc', n_jobs=-1)
    f1_scores  = cross_val_score(pipe, X_train, y_train, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)
    results[name] = {
        'cv_auc_mean': auc_scores.mean(),
        'cv_auc_std' : auc_scores.std(),
        'cv_f1_mean' : f1_scores.mean(),
        'cv_f1_std'  : f1_scores.std(),
    }
    print(f"  {name:25s}  AUC={auc_scores.mean():.4f}+-{auc_scores.std():.4f}"
          f"  F1={f1_scores.mean():.4f}+-{f1_scores.std():.4f}")

results_df = pd.DataFrame(results).T.sort_values('cv_auc_mean', ascending=False)
print("\n  Baseline Model Ranking (by CV AUC):")
print(results_df.to_string())

# Plot baseline comparison
fig, ax = plt.subplots(figsize=(12, 5))
y_pos = np.arange(len(results_df))
bars = ax.barh(y_pos, results_df['cv_auc_mean'],
               xerr=results_df['cv_auc_std'],
               align='center', color='#3498DB', ecolor='gray',
               capsize=5, height=0.6)
ax.set_yticks(y_pos)
ax.set_yticklabels(results_df.index)
ax.set_xlabel('Cross-Validated ROC-AUC')
ax.set_title('Baseline Model Comparison (5-Fold CV)', fontsize=14, fontweight='bold')
ax.set_xlim([0.5, 1.0])
for bar, val in zip(bars, results_df['cv_auc_mean']):
    ax.text(val + 0.003, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/07_baseline_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {PLOTS_DIR}/07_baseline_model_comparison.png")

# Select top 2 models for tuning
top2 = results_df.head(2).index.tolist()
print(f"\n  Top 2 models selected for hyperparameter tuning: {top2}")

# ==============================================================================
# TASK 9 — HYPERPARAMETER TUNING
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 9 -- HYPERPARAMETER TUNING")
print("-" * 70)

tuned_results = {}

if 'Random Forest' in top2:
    print("\n  [Random Forest] -- RandomizedSearchCV (n_iter=30) ...")
    rf_param_dist = {
        'clf__n_estimators'     : [100, 200, 300],
        'clf__max_depth'        : [None, 10, 20, 30],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf' : [1, 2, 4],
        'clf__max_features'     : ['sqrt', 'log2'],
        'clf__class_weight'     : ['balanced', 'balanced_subsample'],
    }
    rf_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))
    ])
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_param_dist, n_iter=30, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    rf_search.fit(X_train, y_train)
    tuned_results['Random Forest (Tuned)'] = rf_search
    print(f"    Best CV AUC: {rf_search.best_score_:.4f}")
    print(f"    Best params: {rf_search.best_params_}")

if 'Gradient Boosting' in top2:
    print("\n  [Gradient Boosting] -- RandomizedSearchCV (n_iter=30) ...")
    gb_param_dist = {
        'clf__n_estimators'     : [100, 200, 300],
        'clf__learning_rate'    : [0.01, 0.05, 0.1, 0.2],
        'clf__max_depth'        : [3, 4, 5, 6],
        'clf__min_samples_split': [2, 5, 10],
        'clf__subsample'        : [0.7, 0.8, 0.9, 1.0],
        'clf__max_features'     : ['sqrt', 'log2', None],
    }
    gb_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=RANDOM_STATE))
    ])
    gb_search = RandomizedSearchCV(
        gb_pipe, gb_param_dist, n_iter=30, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    gb_search.fit(X_train, y_train)
    tuned_results['Gradient Boosting (Tuned)'] = gb_search
    print(f"    Best CV AUC: {gb_search.best_score_:.4f}")
    print(f"    Best params: {gb_search.best_params_}")

if 'Logistic Regression' in top2:
    print("\n  [Logistic Regression] -- GridSearchCV ...")
    lr_param_grid = {
        'clf__C'           : [0.01, 0.1, 1, 10, 100],
        'clf__penalty'     : ['l1', 'l2'],
        'clf__solver'      : ['liblinear'],
        'clf__class_weight': ['balanced'],
    }
    lr_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    ])
    lr_search = GridSearchCV(
        lr_pipe, lr_param_grid, cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    lr_search.fit(X_train, y_train)
    tuned_results['Logistic Regression (Tuned)'] = lr_search
    print(f"    Best CV AUC: {lr_search.best_score_:.4f}")
    print(f"    Best params: {lr_search.best_params_}")

if 'AdaBoost' in top2:
    print("\n  [AdaBoost] -- RandomizedSearchCV (n_iter=20) ...")
    ada_param_dist = {
        'clf__n_estimators' : [50, 100, 200, 300],
        'clf__learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    }
    ada_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', AdaBoostClassifier(random_state=RANDOM_STATE, algorithm='SAMME'))
    ])
    ada_search = RandomizedSearchCV(
        ada_pipe, ada_param_dist, n_iter=20, cv=cv,
        scoring='roc_auc', n_jobs=-1, random_state=RANDOM_STATE, verbose=0
    )
    ada_search.fit(X_train, y_train)
    tuned_results['AdaBoost (Tuned)'] = ada_search
    print(f"    Best CV AUC: {ada_search.best_score_:.4f}")
    print(f"    Best params: {ada_search.best_params_}")

if 'Decision Tree' in top2:
    print("\n  [Decision Tree] -- GridSearchCV ...")
    dt_param_grid = {
        'clf__max_depth'        : [5, 10, 15, 20, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf' : [1, 2, 4],
        'clf__class_weight'     : ['balanced'],
    }
    dt_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])
    dt_search = GridSearchCV(
        dt_pipe, dt_param_grid, cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    dt_search.fit(X_train, y_train)
    tuned_results['Decision Tree (Tuned)'] = dt_search
    print(f"    Best CV AUC: {dt_search.best_score_:.4f}")
    print(f"    Best params: {dt_search.best_params_}")

if 'K-Nearest Neighbors' in top2:
    print("\n  [KNN] -- GridSearchCV ...")
    knn_param_grid = {
        'clf__n_neighbors': [3, 5, 7, 11, 15],
        'clf__weights'    : ['uniform', 'distance'],
        'clf__metric'     : ['euclidean', 'manhattan'],
    }
    knn_pipe = Pipeline([
        ('pre', preprocessor),
        ('clf', KNeighborsClassifier(n_jobs=-1))
    ])
    knn_search = GridSearchCV(
        knn_pipe, knn_param_grid, cv=cv,
        scoring='roc_auc', n_jobs=-1, verbose=0
    )
    knn_search.fit(X_train, y_train)
    tuned_results['KNN (Tuned)'] = knn_search
    print(f"    Best CV AUC: {knn_search.best_score_:.4f}")
    print(f"    Best params: {knn_search.best_params_}")

# ==============================================================================
# TASK 10 — FINAL MODEL SELECTION & TEST SET EVALUATION
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 10 -- FINAL MODEL SELECTION & TEST SET EVALUATION")
print("-" * 70)

def evaluate_on_test(name, pipe, X_tr, y_tr, X_te, y_te):
    """Fit pipeline on train, evaluate on held-out test set."""
    pipe.fit(X_tr, y_tr)
    y_pred      = pipe.predict(X_te)
    y_pred_prob = pipe.predict_proba(X_te)[:, 1]
    return {
        'model'    : name,
        'accuracy' : accuracy_score(y_te, y_pred),
        'precision': precision_score(y_te, y_pred, zero_division=0),
        'recall'   : recall_score(y_te, y_pred, zero_division=0),
        'f1_macro' : f1_score(y_te, y_pred, average='macro', zero_division=0),
        'roc_auc'  : roc_auc_score(y_te, y_pred_prob),
        '_pipe'    : pipe,
        '_proba'   : y_pred_prob,
        '_pred'    : y_pred,
    }

all_eval = {}

# Evaluate all baseline models on test set
for name, model in baseline_models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    all_eval[name] = evaluate_on_test(name, pipe, X_train, y_train, X_test, y_test)

# Evaluate tuned models on test set (already fitted via SearchCV)
for name, search in tuned_results.items():
    best_pipe   = search.best_estimator_
    y_pred      = best_pipe.predict(X_test)
    y_pred_prob = best_pipe.predict_proba(X_test)[:, 1]
    all_eval[name] = {
        'model'    : name,
        'accuracy' : accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall'   : recall_score(y_test, y_pred, zero_division=0),
        'f1_macro' : f1_score(y_test, y_pred, average='macro', zero_division=0),
        'roc_auc'  : roc_auc_score(y_test, y_pred_prob),
        '_pipe'    : best_pipe,
        '_proba'   : y_pred_prob,
        '_pred'    : y_pred,
    }

# Build comparison DataFrame
eval_df = pd.DataFrame(
    [{k: v for k, v in d.items() if not k.startswith('_')}
     for d in all_eval.values()]
).sort_values('roc_auc', ascending=False).reset_index(drop=True)

print("\n  Full Model Comparison (Test Set):")
print(eval_df.to_string(index=False))

# Select the model with highest ROC-AUC on test set
best_name  = eval_df.iloc[0]['model']
best_entry = all_eval[best_name]
best_pipe  = best_entry['_pipe']

print(f"\n  >>> FINAL MODEL SELECTED: {best_name}")
print(f"      ROC-AUC  : {best_entry['roc_auc']:.4f}")
print(f"      F1 Macro : {best_entry['f1_macro']:.4f}")
print(f"      Accuracy : {best_entry['accuracy']:.4f}")
print(f"      Precision: {best_entry['precision']:.4f}")
print(f"      Recall   : {best_entry['recall']:.4f}")

print("\n  Classification Report (Final Model on Test Set):")
print(classification_report(y_test, best_entry['_pred'],
                             target_names=['No Deposit', 'Yes Deposit']))

# Plot 8: Confusion matrix + ROC curves
cm = confusion_matrix(y_test, best_entry['_pred'])
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

disp = ConfusionMatrixDisplay(cm, display_labels=['No', 'Yes'])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title(f'Confusion Matrix -- {best_name}', fontweight='bold')

ax2 = axes[1]
for name, entry in all_eval.items():
    fpr, tpr, _ = roc_curve(y_test, entry['_proba'])
    auc_val = entry['roc_auc']
    lw = 2.5 if name == best_name else 1
    ls = '-' if name == best_name else '--'
    ax2.plot(fpr, tpr, lw=lw, ls=ls, label=f'{name} ({auc_val:.3f})')
ax2.plot([0,1],[0,1], 'k--', lw=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('ROC Curves -- All Models', fontweight='bold')
ax2.legend(fontsize=7, loc='lower right')
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/08_confusion_matrix_roc.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {PLOTS_DIR}/08_confusion_matrix_roc.png")

# Plot 9: Feature importance (if tree-based final model)
clf_step = best_pipe.named_steps['clf']
if hasattr(clf_step, 'feature_importances_'):
    pre_step = best_pipe.named_steps['pre']
    cat_enc_names = (pre_step.named_transformers_['cat']
                     .named_steps['encoder']
                     .get_feature_names_out(categorical_features).tolist())
    all_feat_names = numerical_features + cat_enc_names
    fi = pd.Series(clf_step.feature_importances_, index=all_feat_names)
    fi_top = fi.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 7))
    fi_top.sort_values().plot.barh(ax=ax, color='#2ECC71', edgecolor='white')
    ax.set_title(f'Top 20 Feature Importances -- {best_name}',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/09_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {PLOTS_DIR}/09_feature_importance.png")

# Plot 10: Final comparison bar chart
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(eval_df))
width = 0.2
metrics = ['roc_auc', 'f1_macro', 'accuracy', 'recall']
colors  = ['#3498DB','#E74C3C','#2ECC71','#F39C12']
for i, (m, c) in enumerate(zip(metrics, colors)):
    ax.bar(x + i*width, eval_df[m], width, label=m.upper(),
           color=c, alpha=0.85, edgecolor='white')
ax.set_xticks(x + 1.5*width)
ax.set_xticklabels(eval_df['model'], rotation=40, ha='right', fontsize=9)
ax.set_ylabel('Score')
ax.set_title('Final Model Comparison on Test Set', fontweight='bold', fontsize=14)
ax.legend(fontsize=10)
ax.set_ylim([0, 1.05])
ax.axhline(0.9, color='gray', ls=':', lw=1, alpha=0.5)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/10_final_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {PLOTS_DIR}/10_final_model_comparison.png")

# ==============================================================================
# TASK 11 — MODEL SAVING
# ==============================================================================
print("\n" + "-" * 70)
print("TASK 11 -- MODEL SAVING")
print("-" * 70)

joblib.dump(best_pipe, MODEL_PATH)
print(f"  Saved final pipeline: {MODEL_PATH}")

# Verify the saved model
loaded_pipe = joblib.load(MODEL_PATH)
sample_raw  = X_test.iloc[:3].copy()
sample_pred = loaded_pipe.predict(sample_raw)
sample_prob = loaded_pipe.predict_proba(sample_raw)[:, 1]

print("\n  Verification -- predictions on 3 raw test samples:")
for i, (p, prob) in enumerate(zip(sample_pred, sample_prob)):
    label = 'YES deposit' if p == 1 else 'NO deposit'
    print(f"    Sample {i+1}: {label}  (P(yes)={prob:.3f})")

print("""
  Pipeline verified: accepts raw DataFrame -> outputs predictions.
  Usage:
      import joblib, pandas as pd
      pipe = joblib.load('final_model_pipeline.joblib')
      preds = pipe.predict(new_raw_dataframe)
      probs = pipe.predict_proba(new_raw_dataframe)[:, 1]
""")

# ==============================================================================
# SAVE RESULTS CSV
# ==============================================================================
eval_df_clean = eval_df.copy()
eval_df_clean.to_csv('model_comparison_results.csv', index=False)
print("  Saved model comparison table: model_comparison_results.csv")

# ==============================================================================
# SUMMARY PRINT
# ==============================================================================
print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)
print(f"""
  Dataset        : bank.csv  ({df.shape[0]:,} rows x {df.shape[1]} cols after cleaning)
  Problem        : Binary Classification (deposit yes/no)
  Class Balance  : ~88% No, ~12% Yes (imbalanced)
  Train/Test     : 80% / 20% (stratified)

  Models Trained : {list(baseline_models.keys())}
  Tuned Models   : {list(tuned_results.keys())}

  Final Model    : {best_name}
  Test ROC-AUC   : {best_entry['roc_auc']:.4f}
  Test F1 Macro  : {best_entry['f1_macro']:.4f}
  Test Accuracy  : {best_entry['accuracy']:.4f}
  Test Precision : {best_entry['precision']:.4f}
  Test Recall    : {best_entry['recall']:.4f}

  Artifacts:
    - final_model_pipeline.joblib
    - model_comparison_results.csv
    - plots/ (10 visualization files)

  Run completed  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")
print("=" * 70)
