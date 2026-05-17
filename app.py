import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL AND SCALER
# =========================

model = joblib.load('rf_model.pkl')

scaler = joblib.load('scaler.pkl')

# =========================
# TITLE
# =========================

st.title("Diabetes Prediction using Random Forest")

st.write("Enter patient details to predict diabetes")

# =========================
# USER INPUTS
# =========================

pregnancies = st.number_input(
    "Pregnancies",
    min_value=0,
    max_value=20,
    value=1
)

glucose = st.number_input(
    "Glucose Level",
    min_value=0,
    max_value=300,
    value=120
)

blood_pressure = st.number_input(
    "Blood Pressure",
    min_value=0,
    max_value=200,
    value=70
)

skin_thickness = st.number_input(
    "Skin Thickness",
    min_value=0,
    max_value=100,
    value=20
)

insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=1000,
    value=80
)

bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=70.0,
    value=25.0
)

dpf = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=25
)

# =========================
# PREDICTION BUTTON
# =========================

if st.button("Predict"):

    input_data = np.array([
        [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
    ])

    # Scaling

    input_data = scaler.transform(input_data)

    # Prediction

    prediction = model.predict(input_data)

    # Output

    if prediction[0] == 1:
        st.error("The Person is Diabetic")
    else:
        st.success("The Person is Not Diabetic")