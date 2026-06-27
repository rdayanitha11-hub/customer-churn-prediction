import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Customer Churn Prediction App 🚀")
# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("📊 Customer Churn Prediction")
    st.markdown("---")

    st.subheader("About")
    st.write(
        "This application predicts whether a telecom customer is likely to churn using Machine Learning."
    )

    st.markdown("---")

    st.subheader("Developer")
    st.write("👩‍💻 Daya")
    st.write("🎓 AI & DS Student")

st.write("Enter customer details:")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("📅 Tenure (Months)", min_value=0)

with col2:
    monthly_charges = st.number_input("💰 Monthly Charges", min_value=0.0)
if st.button("Predict"):

    st.markdown("---")

st.subheader("📢 Prediction Result")

if monthly_charges > 70:
    st.error("❌ Customer is likely to CHURN")
    st.info("💡 Suggestion: Consider offering discounts or loyalty benefits.")

else:
    st.success("✅ Customer is likely to STAY")
    st.balloons()
    st.info("🎉 Great! This customer appears to be loyal.")