import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Customer Churn Prediction App 🚀")

st.write("Enter customer details:")

tenure = st.number_input("Tenure")
monthly_charges = st.number_input("Monthly Charges")

if st.button("Predict"):

    # dummy logic (we will replace with model soon)
    if monthly_charges > 70:
        st.error("Customer may CHURN ❌")
    else:
        st.success("Customer will STAY ✅")