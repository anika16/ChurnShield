# src/dashboard/app.py
import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="ChurnShield Dashboard", layout="wide")
st.title("ChurnShield — Customer Churn Prediction")

# --- Sidebar Inputs ---
st.sidebar.header("Sample customer input")
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 20000.0, float(monthly_charges * tenure))
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment_method = st.sidebar.selectbox(
    "PaymentMethod",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)
internet_service = st.sidebar.selectbox("InternetService", ["DSL", "Fiber optic", "No"])

# --- Prepare Sample Payload ---
sample = {
    "gender": gender,
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": tenure,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": internet_service,
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": contract,
    "PaperlessBilling": "Yes",
    "PaymentMethod": payment_method,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges
}

# --- Feature Engineering ---
sample['num_services'] = (
    int(sample['OnlineSecurity'] == 'Yes') +
    int(sample['OnlineBackup'] == 'Yes') +
    int(sample['DeviceProtection'] == 'Yes') +
    int(sample['TechSupport'] == 'Yes') +
    int(sample['StreamingTV'] == 'Yes') +
    int(sample['StreamingMovies'] == 'Yes')
)
sample['avg_charge_per_month'] = (
    sample['MonthlyCharges']
    if sample['tenure'] == 0
    else sample['TotalCharges'] / sample['tenure']
)

if sample['tenure'] <= 12:
    sample['tenure_bucket'] = '0-1 year'
elif sample['tenure'] <= 24:
    sample['tenure_bucket'] = '1-2 years'
elif sample['tenure'] <= 48:
    sample['tenure_bucket'] = '2-4 years'
else:
    sample['tenure_bucket'] = '4+ years'

# --- Predict Button ---

def success_card(prob):
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #d4fcd4, #b2f5b2); padding: 25px; border-radius: 15px; text-align: center; font-size: 24px; border: 2px solid #6fdc6f;">
            <span style="font-size:60px;">😀</span><br>
            <b>Customer will STAY</b><br>
            Probability: {prob:.2f}
        </div>""", unsafe_allow_html=True)

def danger_card(prob):
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, #ffd6d6, #ffb3b3); padding: 25px; border-radius: 15px; text-align: center; font-size: 24px; border: 2px solid #ff6b6b;">
            <span style="font-size:60px;">😟</span><br>
            <b>Customer will CHURN</b><br>
            Probability: {prob:.2f}
        </div>""", unsafe_allow_html=True)


if st.button("Predict churn"):
    try:
        response = requests.post("http://localhost:5000/predict", json=sample)
        if response.status_code == 200:
            result = response.json()
            pred = result.get("prediction")
            prob_churn = result.get("probability", None)

            if pred == 1:
                danger_card(prob_churn)               # Show churn probability
            else:
                success_card(1 - prob_churn)         # Show probability of staying
        else:
            st.error(f"Error from API: {response.text}")
    except Exception as e:
        st.exception(e)
