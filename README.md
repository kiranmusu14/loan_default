# Loan Default Prediction

End-to-end binary classification pipeline that predicts whether a borrower will default on a loan using LendingClub data (2.26M loans, 145 features).

## Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.7401 |
| Average Precision | 0.4476 |
| Best F1 | ~0.52 at threshold 0.518 |

## Project Structure

```
├── notebooks/
│   ├── loan_default_model.ipynb           # Source notebook (run this)
│   ├── loan_default_model_executed.ipynb  # Notebook with full outputs
│   └── exploration.ipynb                  # Initial data exploration
├── plots/
│   ├── confusion_matrix.png               # Confusion matrix
│   ├── feature_importance.png             # Top 30 feature importances
│   ├── roc_pr_curves.png                  # ROC and Precision-Recall curves
│   ├── score_distribution.png             # Risk score distribution
│   └── threshold_analysis.png             # Precision/Recall/F1 vs threshold
├── data/
│   └── LCDataDictionary.xlsx              # Column definitions
├── requirements.txt                        # Python dependencies
└── README.md
```

> `data/loan.csv` (1.1 GB) is not included. Download from [Kaggle — LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club).

## Pipeline Overview

```
Raw CSV (2.26M rows, 145 cols)
    ↓ Filter to closed loans only (remove 'Current')
    ↓ Define binary target (Charged Off / Default / Late → 1)
    ↓ Blacklist leaky post-loan columns (total_pymnt, recoveries, hardship_*, etc.)
    ↓ Feature engineering (parse strings, derive ratios, encode categoricals)
    ↓ Drop columns with >50% missing values
    ↓ Stratified 68/12/20 train/val/test split
    ↓ LightGBM with early stopping on validation AUC
    ↓ Threshold optimization for best F1
```

## Feature Engineering

| Transformation | Example |
|---|---|
| String → numeric | `"36 months"` → `36`, `"15.02%"` → `15.02` |
| Employment length | `"10+ years"` → `10`, `"< 1 year"` → `0` |
| Credit age | `earliest_cr_line` → months since Jan 2020 |
| Ratio features | `loan_to_income`, `installment_to_inc` |
| Grade ordinal | `A→7, B→6, ..., G→1` |
| Categoricals | Label encoded (`home_ownership`, `purpose`, `addr_state`, etc.) |

## Top Predictive Features

1. `int_rate` — interest rate (lender's built-in risk signal)
2. `sub_grade` — fine-grained risk tier (A1–G5)
3. `dti` — debt-to-income ratio
4. `annual_inc` — borrower repayment capacity
5. `loan_to_income` — engineered affordability ratio
6. `revol_util` — credit utilization
7. `cr_line_age_mths` — length of credit history
8. `bc_util` — bankcard utilization
9. `installment_to_inc` — monthly burden relative to income
10. `mo_sin_old_rev_tl_op` — age of oldest revolving account

## Model

**LightGBM** (`LGBMClassifier`) with:
- `n_estimators=1000`, early stopping at 50 rounds (best iteration: 988)
- `learning_rate=0.05`, `num_leaves=63`
- `class_weight='balanced'` — handles 22% default rate imbalance
- L1/L2 regularization (`reg_alpha=0.1`, `reg_lambda=0.1`)
- Feature/bagging fraction 0.8 — reduces overfitting

## Setup

```bash
pip install -r requirements.txt
```

Then open `notebooks/loan_default_model.ipynb` and run all cells.
