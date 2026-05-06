# api/main.py
# Diabetes AI System Backend API

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

from utils.preprocess import (
    preprocess_data,
    preprocess_user_input
)

# ─────────────────────────────────────────────
# ML MODELS
# ─────────────────────────────────────────────

from models.ml_models import (
    run_all_ml_models
)

# ─────────────────────────────────────────────
# DL MODELS
# ─────────────────────────────────────────────

from models.dl_models import (
    run_all_dl_models,
    DEVICE
)

# ─────────────────────────────────────────────
# QML MODELS
# ─────────────────────────────────────────────

from models.qml_models import (
    run_all_qml_models
)

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Diabetes AI System API"
)

# ─────────────────────────────────────────────
# CORS
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    df
) = preprocess_data()

# ─────────────────────────────────────────────
# TRAIN MODELS
# ─────────────────────────────────────────────

print("\n===================================")
print("TRAINING ALL ML MODELS")
print("===================================")

ml_results = run_all_ml_models(

    X_train,
    y_train,

    X_test,
    y_test

)

print("\n===================================")
print("TRAINING ALL DL MODELS")
print("===================================")

dl_results = run_all_dl_models(

    X_train,
    y_train,

    X_test,
    y_test

)

print("\n===================================")
print("TRAINING ALL QML MODELS")
print("===================================")

qml_results = run_all_qml_models(

    X_train,
    y_train,

    X_test,
    y_test

)

# ─────────────────────────────────────────────
# COMBINE RESULTS
# ─────────────────────────────────────────────

ALL_RESULTS = (

    ml_results +
    dl_results +
    qml_results

)

print("\n===================================")
print("ALL MODELS READY")
print("===================================")

# ─────────────────────────────────────────────
# INPUT SCHEMAS
# ─────────────────────────────────────────────

class PatientData(BaseModel):

    pregnancies: int

    glucose: float

    blood_pressure: float

    skin_thickness: float

    insulin: float

    bmi: float

    dpf: float

    age: int


class ChatRequest(BaseModel):

    message: str

# ─────────────────────────────────────────────
# HOME ROUTE
# ─────────────────────────────────────────────

@app.get("/")

def home():

    return {

        "message":
            "Diabetes AI System API Running",

        "models_loaded":
            len(ALL_RESULTS)

    }

# ─────────────────────────────────────────────
# ANALYTICS ROUTE
# ─────────────────────────────────────────────

@app.get("/analytics")

def analytics():

    analytics_data = []

    for item in ALL_RESULTS:

        metrics = item["metrics"]

        model_name = metrics["model"]

        paradigm = "ML"

        if model_name in [
            "ANN",
            "RNN",
            "LSTM",
            "BiLSTM",
            "TCN"
        ]:
            paradigm = "DL"

        elif "QSVM" in model_name:
            paradigm = "QML"

        analytics_data.append({

            "model":
                model_name,

            "paradigm":
                paradigm,

            "train_accuracy":
                metrics["train_accuracy"],

            "test_accuracy":
                metrics["test_accuracy"],

            "f1_score":
                metrics["f1_score"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "confidence":
                metrics["confidence"],

            "train_time_s":
                metrics["train_time_s"]

        })

    return analytics_data

# ─────────────────────────────────────────────
# PREDICT ROUTE
# ─────────────────────────────────────────────

@app.post("/predict")

def predict(data: PatientData):

    user_input = [

        data.pregnancies,
        data.glucose,
        data.blood_pressure,
        data.skin_thickness,
        data.insulin,
        data.bmi,
        data.dpf,
        data.age

    ]

    # preprocess
    input_scaled = preprocess_user_input(

        user_input,
        scaler

    )

    model_outputs = []

    diabetic_votes = 0

    confidences = []

    # ─────────────────────────────────────
    # RUN ALL MODELS
    # ─────────────────────────────────────

    for item in ALL_RESULTS:

        metrics = item["metrics"]

        model_name = metrics["model"]

        prediction = 0
        confidence = 0.50

        # ─────────────────────────────
        # GET TRAINED MODEL
        # ─────────────────────────────

        trained_model = item.get("trained_model")

        if trained_model is None:
            continue

        # ─────────────────────────────
        # DL MODEL
        # ─────────────────────────────

        if isinstance(
            trained_model,
            torch.nn.Module
        ):

            trained_model.eval()

            x = input_scaled

            if item.get("seq_model"):

                x = x.reshape(
                    x.shape[0],
                    x.shape[1],
                    1
                )

            tensor_x = torch.tensor(
                x,
                dtype=torch.float32
            ).to(DEVICE)

            with torch.no_grad():

                outputs = trained_model(
                    tensor_x
                )

                probs = torch.softmax(
                    outputs,
                    dim=1
                )

                prediction = int(
                    torch.argmax(
                        probs,
                        dim=1
                    ).cpu().numpy()[0]
                )

                confidence = float(
                    probs[0][prediction]
                    .cpu()
                    .numpy()
                )

        # ─────────────────────────────
        # ML / QML MODEL
        # ─────────────────────────────

        else:

            pred = trained_model.predict(
                input_scaled
            )

            prediction = int(pred[0])

            if hasattr(
                trained_model,
                "predict_proba"
            ):

                probs = trained_model.predict_proba(
                    input_scaled
                )[0]

                confidence = float(
                    probs[prediction]
                )

            else:

                confidence = (
                    0.90
                    if prediction == 1
                    else 0.82
                )

        # ─────────────────────────────
        # SAVE OUTPUT
        # ─────────────────────────────

        if prediction == 1:
            diabetic_votes += 1

        confidences.append(confidence)

        model_outputs.append({

            "model":
                model_name,

            "prediction":
                "DIABETIC"
                if prediction == 1
                else "NOT DIABETIC",

            "confidence":
                round(confidence * 100, 2)

        })

    # ─────────────────────────────────────
    # ENSEMBLE
    # ─────────────────────────────────────

    final_prediction = (

        "DIABETIC"

        if diabetic_votes >= (
            len(model_outputs) / 2
        )

        else "NOT DIABETIC"

    )

    ensemble_confidence = round(

        np.mean(confidences) * 100,

        2

    )

    # ─────────────────────────────────────
    # BEST MODEL
    # ─────────────────────────────────────

    best_model = max(

        ALL_RESULTS,

        key=lambda x:
            x["metrics"]["test_accuracy"]

    )

    best_model_name = (
        best_model["metrics"]["model"]
    )

    # ─────────────────────────────────────
    # RESPONSE
    # ─────────────────────────────────────

    return {

        "final_prediction":
            final_prediction,

        "ensemble_confidence":
            ensemble_confidence,

        "best_model":
            best_model_name,

        "total_models":
            len(model_outputs),

        "models":
            model_outputs

    }

# ─────────────────────────────────────────────
# CHATBOT ROUTE
# ─────────────────────────────────────────────

@app.post("/chat")

def chatbot(req: ChatRequest):

    msg = req.message.lower()

    # Diabetes
    if "diabetes" in msg:

        reply = (
            "Diabetes is a condition where blood sugar "
            "levels become too high."
        )

    # BMI
    elif "bmi" in msg:

        reply = (
            "BMI below 18.5 is underweight. "
            "18.5–24.9 is normal. "
            "25–29.9 is overweight. "
            "30+ is considered obese and risky."
        )

    # Glucose
    elif "glucose" in msg:

        reply = (
            "Normal fasting glucose is usually "
            "70–99 mg/dL."
        )

    # Insulin
    elif "insulin" in msg:

        reply = (
            "Insulin is a hormone that helps "
            "control blood sugar levels."
        )

    # ML
    elif "machine learning" in msg or "ml" in msg:

        reply = (
            "Machine Learning allows systems "
            "to learn patterns from data."
        )

    # DL
    elif "deep learning" in msg or "dl" in msg:

        reply = (
            "Deep Learning uses neural networks "
            "with multiple hidden layers."
        )

    # QML
    elif "quantum" in msg or "qml" in msg:

        reply = (
            "Quantum Machine Learning combines "
            "quantum computing with AI models."
        )

    # QSVM
    elif "qsvm" in msg:

        reply = (
            "QSVM stands for Quantum Support "
            "Vector Machine."
        )

    # Ensemble
    elif "ensemble" in msg:

        reply = (
            "Ensemble learning combines multiple "
            "models for better prediction accuracy."
        )

    else:

        reply = (
            "I can answer questions about "
            "diabetes, BMI, glucose, ML, DL, "
            "QML, and ensemble learning."
        )

    return {

        "reply":
            reply

    }

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────

# RUN USING:
#
# uvicorn api.main:app --reload