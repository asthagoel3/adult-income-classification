# Adult Income Classification

A portfolio-style machine learning project that predicts whether an individual's annual income exceeds $50,000 using the UCI Adult Income dataset.

## Project Highlights

- End-to-end tabular classification workflow
- Feature engineering for education and capital income
- Leak-resistant scikit-learn preprocessing pipeline
- Histogram Gradient Boosting classifier
- Stratified cross-validation for an imbalanced target
- RandomizedSearchCV and Optuna hyperparameter optimization
- Evaluation with F1, precision, recall, and balanced accuracy

## Results

| Evaluation | Score |
|---|---:|
| Baseline cross-validation F1 | 0.7123 |
| Randomized-search best F1 | 0.7120 |
| Optuna best F1 | 0.7122 |
| Test balanced accuracy | 0.8448 |
| Test macro precision | 0.78 |
| Test macro recall | 0.84 |
| Test macro F1 | 0.80 |

The final model emphasizes recall for the higher-income class, identifying most positive cases while accepting some false positives.

## Repository Contents

- `adult_income_classification.ipynb` — complete analysis and modeling workflow
- `README.md` — project overview and results
- `.gitignore` — standard Python and Jupyter exclusions

## Tools

Python, pandas, NumPy, Matplotlib, scikit-learn, SciPy, Optuna, and Jupyter Notebook.

## Run Locally

```bash
pip install numpy pandas matplotlib scikit-learn scipy optuna tqdm requests
jupyter notebook adult_income_classification.ipynb
```

The notebook downloads the dataset automatically through OpenML.
