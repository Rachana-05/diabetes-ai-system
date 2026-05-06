# models/dl_models.py
# Diabetes Prediction - Deep Learning Models

import os
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    DataLoader,
    TensorDataset
)

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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EPOCHS = 10
BATCH_SIZE = 16
LEARNING_RATE = 0.001

SAVED_MODELS = "saved_models"

os.makedirs(SAVED_MODELS, exist_ok=True)

# ─────────────────────────────────────────────
# METRICS HELPER
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

        "model": name,

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
# TRAINING FUNCTION
# ─────────────────────────────────────────────

def train_dl_model(
    model,
    X_tr,
    y_tr,
    X_te,
    y_te,
    name,
    seq_model=False
):

    model = model.to(DEVICE)

    # ─────────────────────────────────────
    # SEQUENCE RESHAPE
    # ─────────────────────────────────────

    if seq_model:

        X_tr = X_tr.reshape(
            X_tr.shape[0],
            X_tr.shape[1],
            1
        )

        X_te = X_te.reshape(
            X_te.shape[0],
            X_te.shape[1],
            1
        )

    # ─────────────────────────────────────
    # TENSORS
    # ─────────────────────────────────────

    Xt = torch.tensor(
        X_tr,
        dtype=torch.float32
    )

    yt = torch.tensor(
        y_tr,
        dtype=torch.long
    )

    Xe = torch.tensor(
        X_te,
        dtype=torch.float32
    )

    loader = DataLoader(

        TensorDataset(Xt, yt),

        batch_size=BATCH_SIZE,

        shuffle=True

    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    t0 = time.time()

    # ─────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────

    for epoch in range(EPOCHS):

        model.train()

        for xb, yb in loader:

            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(xb)

            loss = criterion(
                outputs,
                yb
            )

            loss.backward()

            optimizer.step()

    # ─────────────────────────────────────
    # EVALUATION
    # ─────────────────────────────────────

    model.eval()

    with torch.no_grad():

        train_outputs = model(
            Xt.to(DEVICE)
        )

        test_outputs = model(
            Xe.to(DEVICE)
        )

        y_pred_tr = torch.argmax(
            train_outputs,
            dim=1
        ).cpu().numpy()

        y_pred_te = torch.argmax(
            test_outputs,
            dim=1
        ).cpu().numpy()

        probs = torch.softmax(
            test_outputs,
            dim=1
        )[:, 1].cpu().numpy()

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
        f"{name}.pt"
    )

    torch.save(
        model.state_dict(),
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
            y_pred_te.tolist(),

        "seq_model":
            seq_model

    }

# ─────────────────────────────────────────────
# ANN
# ─────────────────────────────────────────────

class ANNModel(nn.Module):

    def __init__(self, input_dim=8):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(input_dim, 64),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 2)

        )

    def forward(self, x):

        return self.network(x)

def run_ann(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n--- ANN ---")

    model = ANNModel()

    return train_dl_model(

        model,

        X_tr,
        y_tr,

        X_te,
        y_te,

        "ANN"

    )

# ─────────────────────────────────────────────
# RNN
# ─────────────────────────────────────────────

class RNNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.rnn = nn.RNN(

            input_size=1,

            hidden_size=64,

            batch_first=True

        )

        self.fc = nn.Linear(
            64,
            2
        )

    def forward(self, x):

        out, _ = self.rnn(x)

        out = out[:, -1, :]

        return self.fc(out)

def run_rnn(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n--- RNN ---")

    model = RNNModel()

    return train_dl_model(

        model,

        X_tr,
        y_tr,

        X_te,
        y_te,

        "RNN",

        seq_model=True

    )

# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=64,

            batch_first=True

        )

        self.fc = nn.Linear(
            64,
            2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)

def run_lstm(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n--- LSTM ---")

    model = LSTMModel()

    return train_dl_model(

        model,

        X_tr,
        y_tr,

        X_te,
        y_te,

        "LSTM",

        seq_model=True

    )

# ─────────────────────────────────────────────
# BiLSTM
# ─────────────────────────────────────────────

class BiLSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(

            input_size=1,

            hidden_size=64,

            batch_first=True,

            bidirectional=True

        )

        self.fc = nn.Linear(
            128,
            2
        )

    def forward(self, x):

        out, _ = self.lstm(x)

        out = out[:, -1, :]

        return self.fc(out)

def run_bilstm(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n--- BiLSTM ---")

    model = BiLSTMModel()

    return train_dl_model(

        model,

        X_tr,
        y_tr,

        X_te,
        y_te,

        "BiLSTM",

        seq_model=True

    )

# ─────────────────────────────────────────────
# TCN
# ─────────────────────────────────────────────

class TCNModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(

            in_channels=1,

            out_channels=32,

            kernel_size=2,

            padding=1

        )

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(

            in_channels=32,

            out_channels=64,

            kernel_size=2,

            padding=1

        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            64,
            2
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.squeeze(-1)

        return self.fc(x)

def run_tcn(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n--- TCN ---")

    model = TCNModel()

    return train_dl_model(

        model,

        X_tr,
        y_tr,

        X_te,
        y_te,

        "TCN",

        seq_model=True

    )

# ─────────────────────────────────────────────
# RUN ALL DL MODELS
# ─────────────────────────────────────────────

def run_all_dl_models(
    X_tr,
    y_tr,
    X_te,
    y_te
):

    print("\n===================================")
    print("TRAINING ALL DL MODELS")
    print("===================================")

    results = []

    models = [

        run_ann,

        run_rnn,

        run_lstm,

        run_bilstm,

        run_tcn

    ]

    for model_func in models:

        result = model_func(

            X_tr,
            y_tr,

            X_te,
            y_te

        )

        results.append(result)

    print("\n===================================")
    print("ALL DL MODELS COMPLETED")
    print("===================================")

    return results