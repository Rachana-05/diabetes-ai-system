# models/ml_models.py
# Diabetes Prediction - Machine Learning Models

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

RANDOM_SEED = 42
SAVED_MODELS = "saved_models"

os.makedirs(SAVED_MODELS, exist_ok=True)

# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

def compute_metrics(
    y_true,
    y_pred,
    y_train,
    y_pred_train,
    probs,
    name
):

    train_accuracy = accuracy_score(
        y_train,
        y_pred_train
    )

    test_accuracy = accuracy_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    confidence = float(
        np.mean(probs)
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    ).tolist()

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True
    )

    metrics = {

        "model":
            name,

        "train_accuracy":
            round(train_accuracy, 4),

        "test_accuracy":
            round(test_accuracy, 4),

        "f1_score":
            round(f1, 4),

        "precision":
            round(precision, 4),

        "recall":
            round(recall, 4),

        "confidence":
            round(confidence, 4),

        "confusion_matrix":
            cm,

        "classification_report":
            report

    }

    print(
        f"[{name}] "
        f"train={train_accuracy:.4f} "
        f"test={test_accuracy:.4f} "
        f"f1={f1:.4f} "
        f"confidence={confidence:.4f}"
    )

    return metrics

# ─────────────────────────────────────────────
# GENERIC TRAINER
# ─────────────────────────────────────────────

def train_model(
    model,
    name,
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print(f"\n--- {name} ---")

    t0 = time.time()

    # ─────────────────────────────────────
    # TRAIN
    # ─────────────────────────────────────

    model.fit(X_tr, y_tr)

    # ─────────────────────────────────────
    # PREDICTIONS
    # ─────────────────────────────────────

    y_pred_tr = model.predict(X_tr)

    y_pred_te = model.predict(X_te)

    # ─────────────────────────────────────
    # PROBABILITIES
    # ─────────────────────────────────────

    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(
            X_te
        )[:, 1]

    else:

        probs = np.zeros(
            len(y_pred_te)
        )

    # ─────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────

    metrics = compute_metrics(

        y_te,
        y_pred_te,

        y_tr,
        y_pred_tr,

        probs,

        name

    )

    metrics["train_time_s"] = round(
        time.time() - t0,
        2
    )

    # ─────────────────────────────────────
    # SAVE MODEL
    # ─────────────────────────────────────

    path = os.path.join(
        SAVED_MODELS,
        f"{name}.pkl"
    )

    joblib.dump(
        model,
        path
    )

    # ─────────────────────────────────────
    # RETURN
    # ─────────────────────────────────────

    return {

        "trained_model":
            model,

        "metrics":
            metrics,

        "model_path":
            path,

        "predictions":
            y_pred_te.tolist()

    }

# ─────────────────────────────────────────────
# LOGISTIC REGRESSION
# ─────────────────────────────────────────────

def run_logistic_regression(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    model = LogisticRegression(

        max_iter=1000,

        random_state=RANDOM_SEED

    )

    return train_model(

        model,
        "LogisticRegression",

        X_tr,
        y_tr,

        X_te,
        y_te

    )

# ─────────────────────────────────────────────
# RANDOM FOREST
# ─────────────────────────────────────────────

def run_random_forest(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    model = RandomForestClassifier(

        n_estimators=200,

        max_depth=10,

        min_samples_split=5,

        min_samples_leaf=2,

        random_state=RANDOM_SEED,

        n_jobs=-1

    )

    return train_model(

        model,
        "RandomForest",

        X_tr,
        y_tr,

        X_te,
        y_te

    )

# ─────────────────────────────────────────────
# SVM
# ─────────────────────────────────────────────

def run_svm(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    model = SVC(

        kernel="rbf",

        C=10,

        gamma="scale",

        probability=True,

        random_state=RANDOM_SEED

    )

    return train_model(

        model,
        "SVM",

        X_tr,
        y_tr,

        X_te,
        y_te

    )

# ─────────────────────────────────────────────
# KNN
# ─────────────────────────────────────────────

def run_knn(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    return train_model(

        model,
        "KNN",

        X_tr,
        y_tr,

        X_te,
        y_te

    )

# ─────────────────────────────────────────────
# DECISION TREE
# ─────────────────────────────────────────────

def run_decision_tree(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    model = DecisionTreeClassifier(

        max_depth=5,

        min_samples_split=5,

        random_state=RANDOM_SEED

    )

    return train_model(

        model,
        "DecisionTree",

        X_tr,
        y_tr,

        X_te,
        y_te

    )

# ─────────────────────────────────────────────
# RUN ALL ML MODELS
# ─────────────────────────────────────────────

def run_all_ml_models(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n===================================")
    print("RUNNING ALL ML MODELS")
    print("===================================")

    results = []

    model_functions = [

        run_logistic_regression,

        run_random_forest,

        run_svm,

        run_knn,

        run_decision_tree

    ]

    for func in model_functions:

        result = func(

            X_tr,
            y_tr,

            X_te,
            y_te

        )

        results.append(result)

    print("\n===================================")
    print("ALL ML MODELS COMPLETED")
    print("===================================")

    return results