"""
TRAIN.PY - Smart Farming Recommendation System
Trains 4 ML models with optimized hyperparameters for maximum accuracy.

Models:
    - Decision Tree  (GridSearchCV tuned)
    - Random Forest  (high n_estimators, tuned params)
    - K-Nearest Neighbors (GridSearchCV tuned)
    - Support Vector Machine (stronger C/gamma settings)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score)
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_and_explore_data(csv_path):
    """Load dataset and perform basic EDA."""
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(csv_path)
    print(f"\n  Shape        : {df.shape[0]} samples x {df.shape[1]} features")
    print(f"  Unique crops : {df['label'].nunique()}")
    print(f"  Missing vals : {df.isnull().sum().sum()}")
    print(f"\n  Crop distribution:\n{df['label'].value_counts()}")
    return df


# ============================================================================
# STEP 2: PREPROCESS DATA
# ============================================================================

def preprocess_data(df):
    """Preprocess: scale features, 80/20 stratified split."""
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING  (80% train / 20% test, stratified)")
    print("=" * 70)

    X = df.drop('label', axis=1)
    y = df['label']

    feature_names = X.columns.tolist()
    print(f"\n  Features: {feature_names}")

    # Fill any missing values
    X = X.fillna(X.mean())

    # Scale features - critical for KNN and SVM
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 80/20 split (more training data = better generalisation)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"  Training set : {X_train.shape[0]} samples")
    print(f"  Testing set  : {X_test.shape[0]} samples")

    return X_train, X_test, y_train, y_test, scaler, feature_names


# ============================================================================
# HELPER: COMPUTE & PRINT METRICS
# ============================================================================

def _compute_metrics(y_test, y_pred, name):
    metrics = {
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall':    recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1':        f1_score(y_test, y_pred, average='weighted', zero_division=0),
    }
    print(f"\n  {name} Test Results:")
    print(f"    Accuracy  : {metrics['accuracy']*100:.2f}%")
    print(f"    Precision : {metrics['precision']*100:.2f}%")
    print(f"    Recall    : {metrics['recall']*100:.2f}%")
    print(f"    F1-Score  : {metrics['f1']*100:.2f}%")
    return metrics


# ============================================================================
# STEP 3: DECISION TREE  (GridSearchCV tuned)
# ============================================================================

def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train Decision Tree with GridSearchCV hyperparameter tuning."""
    print("\n" + "=" * 70)
    print("STEP 3: DECISION TREE  (GridSearchCV)")
    print("=" * 70)

    param_grid = {
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }

    base = DecisionTreeClassifier(random_state=42)
    grid = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    dt_model = grid.best_estimator_
    print(f"\n  Best params  : {grid.best_params_}")
    print(f"  CV accuracy  : {grid.best_score_*100:.2f}%")

    y_pred = dt_model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, "Decision Tree")

    cv_scores = cross_val_score(dt_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  5-Fold CV    : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    return dt_model, metrics


# ============================================================================
# STEP 4: RANDOM FOREST  (500 trees, tuned)
# ============================================================================

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train a highly tuned Random Forest Classifier."""
    print("\n" + "=" * 70)
    print("STEP 4: RANDOM FOREST  (500 trees, tuned)")
    print("=" * 70)

    rf_model = RandomForestClassifier(
        n_estimators=500,       # More trees = more stable predictions
        max_depth=None,         # Let trees grow fully
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',    # sqrt(n_features) per split - reduces correlation
        bootstrap=True,
        oob_score=True,         # Out-of-bag accuracy estimate
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)
    print(f"\n  OOB accuracy : {rf_model.oob_score_*100:.2f}%")

    y_pred = rf_model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, "Random Forest")

    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  5-Fold CV    : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    return rf_model, metrics


# ============================================================================
# STEP 5: K-NEAREST NEIGHBORS  (GridSearchCV tuned)
# ============================================================================

def train_knn(X_train, X_test, y_train, y_test):
    """Train KNN with GridSearchCV to find the best k and distance metric."""
    print("\n" + "=" * 70)
    print("STEP 5: K-NEAREST NEIGHBORS  (GridSearchCV)")
    print("=" * 70)

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    base = KNeighborsClassifier(n_jobs=-1)
    grid = GridSearchCV(base, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    knn_model = grid.best_estimator_
    print(f"\n  Best params  : {grid.best_params_}")
    print(f"  CV accuracy  : {grid.best_score_*100:.2f}%")

    y_pred = knn_model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, "KNN")

    cv_scores = cross_val_score(knn_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  5-Fold CV    : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    return knn_model, metrics


# ============================================================================
# STEP 6: SUPPORT VECTOR MACHINE  (stronger C and gamma)
# ============================================================================

def train_svm(X_train, X_test, y_train, y_test):
    """Train SVM with stronger regularisation for higher accuracy."""
    print("\n" + "=" * 70)
    print("STEP 6: SUPPORT VECTOR MACHINE  (C=10, RBF kernel)")
    print("=" * 70)

    svm_model = SVC(
        kernel='rbf',
        C=10,              # Higher C = less regularisation = tighter fit
        gamma='scale',     # 1 / (n_features * X.var()) - good automatic default
        probability=True,  # Enables confidence scores in the web app
        random_state=42,
        cache_size=500
    )

    svm_model.fit(X_train, y_train)
    print("\n  SVM training complete")

    y_pred = svm_model.predict(X_test)
    metrics = _compute_metrics(y_test, y_pred, "SVM")

    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"  5-Fold CV    : {cv_scores.mean()*100:.2f}% +/- {cv_scores.std()*100:.2f}%")

    return svm_model, metrics


# ============================================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(dt_model, rf_model, feature_names):
    """Print feature importances from tree-based models."""
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE IMPORTANCE")
    print("=" * 70)

    for name, model in [("Decision Tree", dt_model), ("Random Forest", rf_model)]:
        imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n  {name}:\n{imp.to_string(index=False)}")

    print("\n  (KNN and SVM do not have explicit feature importances)")


# ============================================================================
# STEP 8: COMPARE MODELS
# ============================================================================

def compare_models(dt_metrics, rf_metrics, knn_metrics, svm_metrics):
    """Rank all 4 models by accuracy."""
    print("\n" + "=" * 70)
    print("STEP 8: MODEL COMPARISON")
    print("=" * 70)

    rows = [
        {'Model': 'Decision Tree', **dt_metrics},
        {'Model': 'Random Forest', **rf_metrics},
        {'Model': 'KNN',           **knn_metrics},
        {'Model': 'SVM',           **svm_metrics},
    ]
    df = pd.DataFrame(rows).sort_values('accuracy', ascending=False)

    print(f"\n  {'Model':<18} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 60)
    for _, row in df.iterrows():
        print(f"  {row['Model']:<18} {row['accuracy']*100:>9.2f}%"
              f" {row['precision']*100:>9.2f}%"
              f" {row['recall']*100:>9.2f}%"
              f" {row['f1']*100:>9.2f}%")

    best = df.iloc[0]
    print(f"\n  Best Model: {best['Model']}  ({best['accuracy']*100:.2f}% accuracy)")
    return df


# ============================================================================
# STEP 9: SAVE MODELS
# ============================================================================

def save_models(dt_model, rf_model, knn_model, svm_model, scaler, feature_names):
    """Save all models, scaler, and feature names to disk."""
    print("\n" + "=" * 70)
    print("STEP 9: SAVING MODELS")
    print("=" * 70)

    os.makedirs('models', exist_ok=True)

    saves = {
        'models/decision_tree_model.pkl': dt_model,
        'models/random_forest_model.pkl': rf_model,
        'models/knn_model.pkl':           knn_model,
        'models/svm_model.pkl':           svm_model,
        'models/scaler.pkl':              scaler,
        'models/feature_names.pkl':       feature_names,
    }

    for path, obj in saves.items():
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"  Saved: {path}")

    print("\n  All models saved successfully!")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  SMART FARMING SYSTEM - OPTIMIZED MODEL TRAINING")
    print("=" * 70 + "\n")

    try:
        df = load_and_explore_data('crop_data.csv')

        X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)

        dt_model,  dt_metrics  = train_decision_tree(X_train, X_test, y_train, y_test)
        rf_model,  rf_metrics  = train_random_forest(X_train, X_test, y_train, y_test)
        knn_model, knn_metrics = train_knn(X_train, X_test, y_train, y_test)
        svm_model, svm_metrics = train_svm(X_train, X_test, y_train, y_test)

        analyze_feature_importance(dt_model, rf_model, feature_names)
        compare_models(dt_metrics, rf_metrics, knn_metrics, svm_metrics)
        save_models(dt_model, rf_model, knn_model, svm_model, scaler, feature_names)

        print("\n" + "=" * 70)
        print("  Training complete! Run: python app.py  ->  http://localhost:5000")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n  Error: {e}")
        raise
