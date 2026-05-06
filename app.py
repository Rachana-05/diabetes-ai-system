import streamlit as st
import pandas as pd
import numpy as np
from utils.preprocess import load_data
from models.ml_models import run_ml_models
from models.dl_models import run_dl_models
from models.qml_models import run_qml_models
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Diabetes AI System", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    🧠 Diabetes Prediction System (ML vs DL vs QML)
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🧾 Enter Patient Details")

preg = st.sidebar.slider("Pregnancies", 0, 20, 1)
glucose = st.sidebar.slider("Glucose", 0, 200, 120)
bp = st.sidebar.slider("Blood Pressure", 0, 140, 70)
skin = st.sidebar.slider("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.slider("Insulin", 0, 900, 80)
bmi = st.sidebar.slider("BMI", 0.0, 50.0, 25.0)
dpf = st.sidebar.slider("Diabetes Pedigree", 0.0, 2.5, 0.5)
age = st.sidebar.slider("Age", 10, 100, 30)

input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])

# -------------------- LOAD DATA --------------------
X_train, X_test, y_train, y_test = load_data()

# -------------------- TRAIN BEST MODEL (FAST) --------------------
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

prediction = rf_model.predict(input_data)[0]
prob = rf_model.predict_proba(input_data)[0][1]  # probability of diabetic

# -------------------- MAIN LAYOUT --------------------
col1, col2 = st.columns([1, 2])

# -------- LEFT: LIVE PREDICTION --------
with col1:
    st.subheader("🔍 Live Prediction")

    confidence = prob if prediction == 1 else (1 - prob)

    if prediction == 1:
        st.error(f"⚠️ Diabetic\n\nConfidence: {confidence:.2f}")
    else:
        st.success(f"✅ Not Diabetic\n\nConfidence: {confidence:.2f}")

# Progress bar (visual impact 🔥)
st.progress(int(confidence * 100))

    st.markdown("### 📊 Input Summary")
    st.write({
        "Glucose": glucose,
        "BMI": bmi,
        "Age": age
    })

# -------- RIGHT: MODEL COMPARISON --------
with col2:
    st.subheader("📊 Model Comparison")

    if st.button("🚀 Run All Models"):

        ml_results = run_ml_models(X_train, X_test, y_train, y_test)
        dl_results = run_dl_models(X_train, X_test, y_train, y_test)
        qml_results = run_qml_models(X_train, X_test, y_train, y_test)

        all_results = {**ml_results, **dl_results, **qml_results}

        df = pd.DataFrame(list(all_results.items()), columns=["Model", "Accuracy"])
        df = df.sort_values(by="Accuracy", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.bar_chart(df.set_index("Model"))

        best_model = df.iloc[0]["Model"]

        st.success(f"🏆 Best Model: {best_model}")

st.markdown("---")
st.caption("Built with Streamlit | ML + DL + QML Demo")