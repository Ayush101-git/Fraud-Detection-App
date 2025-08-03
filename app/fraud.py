import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load("../models/Time_based_LightGBM_SMOTE.pkl")

model = load_model()

# ---------------- UI ---------------- #
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection Prediction App")
st.markdown("Enter transaction details below and click **Predict** to identify potential fraud.")
st.divider()

# --------- Input Form Layout -------- #
with st.form("fraud_form"):
    col1, col2 = st.columns(2)

    with col1:
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN"])
        amount = st.number_input("Amount", min_value=0.0, value=1000.00, step=100.0)
        oldbalanceOrg = st.number_input("Old Balance of Sender", min_value=0.0, value=10000.00)

    with col2:
        newbalanceOrig = st.number_input("New Balance of Sender", min_value=0.0, value=10000.00)
        oldbalanceDest = st.number_input("Old Balance of Receiver", min_value=0.0, value=0.00)
        newbalanceDest = st.number_input("New Balance of Receiver", min_value=0.0, value=0.00)

    submitted = st.form_submit_button("🔍 Predict")

# ----------- Prediction ------------- #
if submitted:
    try:
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        prediction = model.predict(input_data)[0]
        probas = model.predict_proba(input_data)[0]

        st.divider()
        if prediction == 1:
            st.error("⚠️ The transaction is predicted to be **fraudulent**.")
        else:
            st.success("✅ The transaction is predicted to be **legitimate**.")

        st.markdown(f"**Fraud Probability:** `{probas[1]:.4f}`")
        st.markdown(f"**Non-Fraud Probability:** `{probas[0]:.4f}`")

    except Exception as e:
        st.error("❌ Error while making prediction.")
        st.exception(e)
