import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(
    page_title="Fraud Detection App",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    try:
        return joblib.load('fraud_detection_pipeline.pkl')
    except FileNotFoundError:
        st.error("Error: 'fraud_detection_pipeline.pkl' not found. Please ensure the file is in the same directory.")
        return None

model = load_model()

st.title('🛡️ Fraud Detection System')
st.markdown("Enter the transaction details below to assess the risk of fraud.")
st.divider()


if model:
    with st.form("fraud_form"):
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox(
                'Transaction Type', 
                ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT', 'DEBIT']
            )
            amount = st.number_input('Amount ($)', min_value=0.0, value=1000.0, step=100.0)

        st.markdown("---")

        st.subheader("Sender Information (Origin)")
        col3, col4 = st.columns(2)
        with col3:
            oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
        with col4:
            newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)
            
        st.caption(f"Sender Balance Change: {newbalanceOrig - oldbalanceOrg:,.2f}")

        st.markdown("---")

        st.subheader("Receiver Information (Destination)")
        col5, col6 = st.columns(2)
        with col5:
            oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
        with col6:
            newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0, value=1000.0)
            
        st.caption(f"Receiver Balance Change: {newbalanceDest - oldbalanceDest:,.2f}")

        st.markdown("---")

        submitted = st.form_submit_button("Analyze Transaction", type="primary")

    if submitted:
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": oldbalanceOrg,
            "newbalanceOrig": newbalanceOrig,
            "oldbalanceDest": oldbalanceDest,
            "newbalanceDest": newbalanceDest
        }])

        with st.spinner('Analyzing transaction patterns...'):
            try:
                prediction = model.predict(input_data)[0]

                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(input_data)[0][1]
                else:
                    probability = None

                st.divider()
                st.subheader("Analysis Result")

                if prediction == 1:
                    st.error("🚨 ALERT: This transaction is flagged as **FRAUD**.")
                    if probability is not None:
                        st.write(f"**Risk Probability:** {probability:.2%}")
                        st.progress(probability)
                else:
                    st.success("✅ SAFE: This transaction appears legitimate.")
                    if probability is not None:
                        st.write(f"**Fraud Probability:** {probability:.2%}")
                        st.progress(probability)
                        
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    with st.sidebar:
        st.info("ℹ️ **About this App**")
        st.markdown("""
        This tool uses a Machine Learning model to detect fraudulent transactions based on patterns in:
        - Transaction Type
        - Amount involved
        - Balance discrepancies
        """)

else:
    st.warning("Please upload the model file to proceed.")