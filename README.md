# Weak Learners vs Boosted Models in Classification

## Overview
This project evaluates the performance of **weak learners** (KNN, SVM, Naïve Bayes, Decision Trees) and **boosted models** (XGBoost, CATBoost, LightGBM) on multiple datasets. It also explores the effect of **dataset augmentation** on model performance.

## Features
- **Weak Learners Evaluation**: KNN, SVM, Naïve Bayes, Decision Trees (CART, ID3)
- **Boosted Models**: XGBoost, CatBoost, LightGBM
- **Dataset Augmentation**: Expanding dataset size by duplicating records
- **Performance Metrics**: Accuracy, Cross-Validation, and Execution Time

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries:
```bash
pip install numpy pandas scikit-learn xgboost catboost lightgbm
```

## Datasets
This project uses three datasets from **Scikit-learn**:
- **Iris**
- **Digits**
- **Breast Cancer**

## Usage
### Running the Script
Execute the script using:
```bash
python weak_learners_boosted_models_3.767.py
```

### Steps Performed
1. Loads **three datasets** (Iris, Digits, Breast Cancer).
2. Splits datasets into **training (70%) and testing (30%)**.
3. Evaluates **weak learners** using:
   - Test-train split accuracy.
   - 10-fold cross-validation.
4. Augments datasets (expands training data **3x**).
5. Evaluates **boosted models (XGBoost, CatBoost, LightGBM)**.
6. Measures **execution time** for each boosted model.

## Output
- Accuracy scores for each **weak learner**.
- Comparison of **cross-validation results**.
- **Execution time and accuracy** for boosted models.
- Performance impact of **dataset augmentation**.

## Findings
- **Boosted models outperformed weak learners** in all datasets.
- **XGBoost and CatBoost achieved the highest accuracy**.
- **Dataset augmentation slightly improved model performance**.
- **Decision Trees showed high variance**, benefitting from boosting.

## License
This project is open-source and available for modification and use.

