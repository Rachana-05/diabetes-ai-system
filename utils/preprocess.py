# utils/preprocess.py
# Diabetes Dataset Preprocessing
import kagglehub
import os

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset():

    import kagglehub
    import os

    print("\nDownloading Dataset from Kaggle...")

    # Download dataset
    path = kagglehub.dataset_download(
        "uciml/pima-indians-diabetes-database"
    )

    # CSV file path
    csv_path = os.path.join(
        path,
        "diabetes.csv"
    )

    print("\nLoading Dataset...")

    # Load CSV
    df = pd.read_csv(csv_path)

    # Save locally
    os.makedirs("data", exist_ok=True)

    df.to_csv(
        "data/diabetes.csv",
        index=False
    )

    print(f"Dataset Shape: {df.shape}")

    return df

    import kagglehub
    import os

    print("\nDownloading Dataset from Kaggle...")

    # Download dataset
    path = kagglehub.dataset_download(
        "uciml/pima-indians-diabetes-database"
    )

    # CSV file path
    csv_path = os.path.join(
        path,
        "diabetes.csv"
    )

    print("\nLoading Dataset...")

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"Dataset Shape: {df.shape}")

    return df




# ─────────────────────────────────────────────
# HANDLE MISSING / INVALID VALUES
# ─────────────────────────────────────────────

def clean_dataset(df):

    print("\nCleaning Dataset...")

    # Columns where 0 is invalid
    invalid_zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in invalid_zero_columns:

        median_value = df[col].median()

        df[col] = df[col].replace(
            0,
            median_value
        )

    return df


# ─────────────────────────────────────────────
# FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────

def split_features_target(df):

    print("\nSplitting Features and Target...")

    X = df.drop(
        "Outcome",
        axis=1
    )

    y = df["Outcome"]

    return X, y


# ─────────────────────────────────────────────
# FEATURE SCALING
# ─────────────────────────────────────────────

def scale_features(X):

    print("\nScaling Features...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ─────────────────────────────────────────────
# TRAIN TEST SPLIT
# ─────────────────────────────────────────────

def create_train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
):

    print("\nCreating Train-Test Split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape : {X_test.shape}")

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ─────────────────────────────────────────────
# COMPLETE PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess_data(path="data/diabetes.csv"):

    # Step 1: Load dataset
    df = load_dataset()

    # Step 2: Clean dataset
    df = clean_dataset(df)

    # Step 3: Split features and target
    X, y = split_features_target(df)

    # Step 4: Scale features
    X_scaled, scaler = scale_features(X)

    # Step 5: Train-test split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = create_train_test_split(
        X_scaled,
        y
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        df
    )


# ─────────────────────────────────────────────
# LIVE USER INPUT PREPROCESSING
# ─────────────────────────────────────────────

def preprocess_user_input(
    input_data,
    scaler
):

    """
    input_data format:
    [
        Pregnancies,
        Glucose,
        BloodPressure,
        SkinThickness,
        Insulin,
        BMI,
        DiabetesPedigreeFunction,
        Age
    ]
    """

    input_array = np.array(
        input_data
    ).reshape(1, -1)

    input_scaled = scaler.transform(
        input_array
    )

    return input_scaled