# Adult Income Classification

A machine-learning classification project using the UCI Adult Income dataset to predict whether an individual earns more than $50,000 per year.

## Project Overview

This notebook builds and evaluates a leak-resistant classification pipeline for an imbalanced tabular dataset. It includes feature engineering, categorical encoding, stratified cross-validation, histogram-based gradient boosting, randomized hyperparameter search, Optuna optimization, and held-out test evaluation.

## Methods

- **Dataset:** UCI Adult Income, loaded through `sklearn.datasets.fetch_openml`
- **Target:** Annual income `>50K` versus `<=50K`
- **Feature engineering:**
  - Removed `fnlwgt`
  - Retained ordinal `education-num` and removed duplicate `education`
  - Combined capital gains and losses into `capital_net`
  - Added log-scaled `capital_net_log`
- **Preprocessing:** `OrdinalEncoder` inside a `ColumnTransformer` and `Pipeline`
- **Model:** `HistGradientBoostingClassifier`
- **Class imbalance:** Balanced class weights and stratified folds
- **Optimization:** `RandomizedSearchCV` and Optuna
- **Metrics:** F1 score, precision, recall, and balanced accuracy

## Results

| Evaluation | Result |
|---|---:|
| Baseline cross-validation F1 | 0.7123 |
| Randomized-search best F1 | 0.7120 |
| Optuna best F1 | 0.7122 |
| Test balanced accuracy | 0.8448 |
| Test macro precision | 0.78 |
| Test macro recall | 0.84 |
| Test macro F1 | 0.80 |

The final model achieved high recall for the `>50K` class, identifying most positive cases, with a tradeoff in precision.

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, SciPy, Optuna, and Jupyter Notebook.

## Running the Notebook

Install the main dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn scipy optuna tqdm requests
```

Then open:

```text
adult_income_classification.ipynb
```

The dataset is downloaded automatically through OpenML when the notebook runs.
