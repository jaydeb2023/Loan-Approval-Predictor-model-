import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from preprocess import preprocess_data  # Assuming the preprocess function is in preprocess.py

# Load the trained model
model = joblib.load('loan_model.pkl')

# Streamlit UI
st.title("Loan Approval Prediction")

# Input form for loan application
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", [0, 1, 2, '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=1000, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount", min_value=50, step=10)
loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=12, max_value=480, step=12)
credit_history = st.selectbox("Credit History", [0, 1])
property_area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Prepare data for prediction
if st.button("Predict"):
    # Create DataFrame for user input
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Married': [married],
        'Dependents': [dependents],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'CoapplicantIncome': [coapplicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Property_Area': [property_area]
    })

    # Preprocess the input data (similar to the training preprocessing)
    processed_data = preprocess_data(input_data)

    # Make prediction
    prediction = model.predict(processed_data)

    # Display the result
    if prediction == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Denied")
