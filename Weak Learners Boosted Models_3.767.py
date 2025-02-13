from sklearn.datasets import load_iris, load_digits, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
import time

# Load datasets
def load_datasets():
    datasets = {
        "Iris": load_iris(),
        "Digits": load_digits(),
        "Breast Cancer": load_breast_cancer()
    }
    return datasets

# Train and evaluate weak learners
def evaluate_weak_learners(X_train, y_train, X_test, y_test):
    models = {
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "SVM": SVC(),
        "Naive Bayes": GaussianNB(),
        "Decision Tree (CART)": DecisionTreeClassifier(criterion='gini'),
        "Decision Tree (ID3)": DecisionTreeClassifier(criterion='entropy')
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cross_val = np.mean(cross_val_score(model, X_train, y_train, cv=10))
        results[name] = {"Test Accuracy": round(accuracy, 2), "Cross-Validation": round(cross_val, 2)}
    
    return results

# Augment dataset (duplicate records to increase size)
def augment_data(X, y, multiplier):
    X_augmented = np.tile(X, (multiplier, 1))
    y_augmented = np.tile(y, multiplier)
    return X_augmented, y_augmented

# Train and evaluate boosted models
def evaluate_boosted_models(X_train, y_train, X_test, y_test):
    models = {
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "CatBoost": CatBoostClassifier(verbose=0),
        "LightGBM": LGBMClassifier()
    }
    
    results = {}
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        elapsed_time = time.time() - start_time
        results[name] = {"Test Accuracy": round(accuracy, 2), "Training Time (s)": round(elapsed_time, 2)}
    
    return results

# Main script execution
if __name__ == "__main__":
    datasets = load_datasets()
    
    for dataset_name, dataset in datasets.items():
        print(f"\nEvaluating models on {dataset_name} dataset:")
        X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.3, random_state=1)
        weak_learner_results = evaluate_weak_learners(X_train, y_train, X_test, y_test)
        print("Weak Learner Results:", weak_learner_results)
        
        # Augment data and evaluate boosted models
        X_aug, y_aug = augment_data(X_train, y_train, multiplier=3)
        boosted_results = evaluate_boosted_models(X_aug, y_aug, X_test, y_test)
        print("Boosted Model Results:", boosted_results)
