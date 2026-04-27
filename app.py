
# 1 = Low Risk, 0 = High Risk

import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("best_extra_trees_model.pkl")

# Load encoders (including Purpose)
encoders = {
    col: joblib.load(f"{col}_encoder.pkl")
    for col in ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
}

# App UI
st.title("Credit Risk Prediction App")
st.write("Enter the applicant details to predict credit risk.")

# Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)

sex = st.selectbox("Sex", ["male", "female"])

job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1)

housing = st.selectbox("Housing", ["own", "rent", "free"])

saving_accounts = st.selectbox(
    "Saving accounts", ["little", "moderate", "rich", "quite rich"]
)

checking_account = st.selectbox(
    "Checking account", ["little", "moderate", "rich", "quite rich"]
)

purpose = st.selectbox(
    "Purpose",
    [
        "car",
        "furniture/equipment",
        "radio/TV",
        "domestic appliances",
        "repairs",
        "education",
        "business",
        "vacation/others",
    ],
)

credit_amount = st.number_input("Credit amount", min_value=0, value=5000)

# Create input dataframe (MATCH TRAINING DATA EXACTLY)
input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Purpose": [encoders["Purpose"].transform([purpose])[0]]
})

# Ensure correct column order
input_df = input_df[model.feature_names_in_]

# Prediction
if st.button("Predict"):
    try:
        pred = model.predict(input_df)[0]
        risk = "Low Risk (1)" if pred == 1 else "High Risk (0)"
        st.success(f"Predicted Credit Risk: {risk}")
    except Exception as e:
        st.error(f"Error: {e}")


