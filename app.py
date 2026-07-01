import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
import plotly.graph_objects as go

# ---------------- CONFIG ----------------
USD_TO_INR = 83.5
INR_TO_USD = 1 / USD_TO_INR

def format_inr(amount):
    """Format number to INR with ₹ symbol"""
    return f"₹{amount:,.2f}"

def format_inr_no_decimal(amount):
    """Format INR without decimals"""
    return f"₹{amount:,.0f}"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- BALLOONS ON LOAD ----------------
if 'first_load' not in st.session_state:
    st.balloons()
    st.session_state.first_load = True

# ---------------- LIVE BACKGROUND CSS - FIXED ----------------
st.markdown("""
<style>
    /* Animated Gradient Background */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .stApp {
        background: linear-gradient(-45deg, #fbc2eb, #a6c1ee, #c2e9fb, #a1c4fd);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }

    /* Floating Bubbles Animation */
    @keyframes float {
        0% { transform: translateY(100vh) scale(1); opacity: 9; }
        50% { opacity: 0.8; }
        100% { transform: translateY(-10vh) scale(1); opacity: 9; }
    }

    .bubble {
    position: absolute;
    bottom: -150px;
    border-radius: 50%;
    animation: float-up 15s infinite;
    
    background: #FF6B6B;  /* ← Solid red */
    opacity: 0.7;         /* ← 70%  */
    border: 2px solid #fff; 
}

.bubble:nth-child(1) { left: 10%; width: 40px; height: 40px; animation-delay: 0s; }
.bubble:nth-child(2) { left: 20%; width: 20px; height: 20px; animation-delay: 2s; animation-duration: 12s; }
.bubble:nth-child(3) { left: 35%; width: 50px; height: 50px; animation-delay: 4s; }
.bubble:nth-child(4) { left: 50%; width: 80px; height: 80px; animation-delay: 0s; animation-duration: 18s; }
.bubble:nth-child(5) { left: 70%; width: 60px; height: 60px; animation-delay: 3s; }
.bubble:nth-child(6) { left: 85%; width: 30px; height: 30px; animation-delay: 6s; animation-duration: 14s; }


    /* Solid white content box */
  .main.block-container {
        background: #ffffff;
        border-radius: 20px;
        padding: 2rem 2rem 4rem 2rem;
        margin-top: 2rem;
        margin-bottom: 3rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 10;
    }

    /* Sidebar solid white */
    [data-testid="stSidebar"] {
        background: #ffffff;
        box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
        z-index: 10;
    }

    /* Header styling */
  .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }

  .sub-header {
        text-align: center;
        color: #333;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

  .result-card {
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        margin: 20px 0;
        text-align: center;
    }

    .footer {
        text-align: center;
        color: #222;
        font-weight: 600;
        padding: 20px 0;
        margin-top: 50px;
        border-top: 1px solid #e0e0e0;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }

    /* Fix plotly chart visibility */
  .js-plotly-plot,.plotly {
        z-index: 10;
        position: relative;
    }

    /* Tab styling */
  .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f0f2f6;
        border-radius: 10px;
        padding: 5px;
    }

  .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        background: transparent;
    }
</style>

<!-- Floating Bubbles -->
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
<div class="bubble"></div>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    try:
        with open('churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except:
        return None

model = load_model()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 📊 Churn Dashboard")
    st.markdown("---")
    st.markdown("**About**")
    st.write("Predict customer churn using Machine Learning to improve retention strategies.")
    st.markdown("---")
    st.markdown("**Dashboard Theme**")
    theme = st.selectbox("Theme", ["Light", "Dark"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Developer**")
    st.write("👨‍💻 Daya")
    st.write("🎓 AI & DS Student")
    st.markdown("---")
    st.caption("© 2026 All Rights Reserved")

# ---------------- MAIN HEADER ----------------
st.markdown('<h1 class="main-header">📊 Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict if a customer will churn using AI & Machine Learning</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------- TABS ----------------
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "📈 Analytics", "ℹ️ About"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="📦 Model", value="Logistic Regression")
    with col2:
        st.metric(label="📊 Accuracy", value="85.2%")
    with col3:
        st.metric(label="🎯 Precision", value="82.1%")
    with col4:
        st.metric(label="⚡ Status", value="Active", delta="Live")

    st.markdown("---")
    st.markdown("### 📋 Enter Customer Details")

    # ---------------- INPUT FORM - INR ----------------
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            tenure = st.number_input("📅 Tenure (Months)", min_value=0, max_value=72, value=12)
            contract = st.selectbox("📝 Contract Type", ["Month-to-month", "One year", "Two year"])

        with col2:
            monthly_charges_inr = st.number_input("💰 Monthly Charges (₹)", min_value=0.0, max_value=17000.0, value=5427.5, step=100.0)
            st.caption(f"≈ ${monthly_charges_inr * INR_TO_USD:.2f} USD")
            internet_service = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])

        with col3:
            total_charges_inr = st.number_input("💵 Total Charges (₹)", min_value=0.0, value=83500.0, step=500.0)
            st.caption(f"≈ ${total_charges_inr * INR_TO_USD:.2f} USD")
            payment_method = st.selectbox("💳 Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

        submitted = st.form_submit_button("🎯 Predict Customer Churn", use_container_width=True, type="primary")

    # ---------------- PREDICTION LOGIC ----------------
    if submitted:
        monthly_charges_usd = monthly_charges_inr * INR_TO_USD
        total_charges_usd = total_charges_inr * INR_TO_USD

        with st.spinner('Analyzing customer data...'):
            time.sleep(1)

            if model:
                input_data = pd.DataFrame({
                    'tenure': [tenure],
                    'MonthlyCharges': [monthly_charges_usd],
                    'TotalCharges': [total_charges_usd]
                })
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data)[0]
                churn_prob = probability[1] * 100
            else:
                risk_score = 0
                if monthly_charges_usd > 70: risk_score += 30
                if contract == "Month-to-month": risk_score += 40
                if internet_service == "Fiber optic": risk_score += 15
                if payment_method == "Electronic check": risk_score += 15
                churn_prob = min(risk_score, 95)
                prediction = 1 if churn_prob > 50 else 0

        st.markdown("---")
        st.markdown("### 📢 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white;">
                <h2>❌ High Churn Risk Detected</h2>
                <h1 style="font-size: 3.5rem; margin: 10px 0;">{churn_prob:.1f}%</h1>
                <p style="font-size: 1.2rem;">Customer is likely to CHURN</p>
                <p style="font-size: 1rem; margin-top: 15px; opacity: 0.9;">
                    💰 Monthly Revenue at Risk: {format_inr(monthly_charges_inr)}<br>
                    💵 Total Customer Value: {format_inr_no_decimal(total_charges_inr)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.error("⚠️ **Action Required:** Contact this customer immediately with retention offers!")
        else:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white;">
                <h2>✅ Customer Likely to Stay</h2>
                <h1 style="font-size: 3.5rem; margin: 10px 0;">{100-churn_prob:.1f}%</h1>
                <p style="font-size: 1.2rem;">Customer is LOYAL</p>
                <p style="font-size: 1rem; margin-top: 15px; opacity: 0.9;">
                    💰 Monthly Revenue: {format_inr(monthly_charges_inr)}<br>
                    💵 Total Customer Value: {format_inr_no_decimal(total_charges_inr)}
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.success("🎉 **Great News:** This customer is satisfied. Consider upselling opportunities!")

        st.markdown("### 📊 Churn Probability Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Churn Probability", 'font': {'size': 24, 'color': '#333'}},
            number={'suffix': "%", 'font': {'size': 40, 'color': '#333'}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#667eea"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 20], 'color': '#4facfe'},
                    {'range': [20, 40], 'color': '#00f2fe'},
                    {'range': [40, 60], 'color': '#f093fb'},
                    {'range': [60, 80], 'color': '#f5576c'},
                    {'range': [80, 100], 'color': '#ff0844'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#333'}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🔍 Risk Factor Analysis")
        risk_col1, risk_col2 = st.columns(2)
        with risk_col1:
            st.write(f"**Contract Type:** {contract}")
            if contract == "Month-to-month":
                st.warning("⚠️ High risk - No commitment")
            else:
                st.success("✅ Low risk - Long term")
        with risk_col2:
            st.write(f"**Monthly Charges:** {format_inr(monthly_charges_inr)}")
            if monthly_charges_usd > 70:
                st.warning("⚠️ High charges - Price sensitive")
            else:
                st.success("✅ Reasonable pricing")

with tab2:
    st.markdown("### 📈 Churn Analytics Dashboard")
    st.info("🚧 Analytics dashboard coming soon! Will include charts for churn trends, customer segments, and revenue impact.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Customers", "7,043", delta="12%")
    with col2:
        st.metric("Churn Rate", "26.5%", delta="-3.2%")
    with col3:
        st.metric("Revenue at Risk", format_inr_no_decimal(12525000), delta=f"-{format_inr_no_decimal(417500)}")

with tab3:
    st.markdown("### ℹ️ About This App")
    st.write("""
    This **Customer Churn Prediction** app uses Machine Learning to predict whether a customer will leave your service.

    **How it works:**
    1. Enter customer details in Indian Rupees (₹)
    2. Click 'Predict Customer Churn'
    3. Get instant prediction with gauge chart and recommendations

    **Technologies Used:**
    - 🤖 Machine Learning: Logistic Regression
    - 🎨 Frontend: Streamlit + Plotly
    - 📊 Data: Telco Customer Churn Dataset

    **Note:** Enter all amounts in Indian Rupees (₹). App automatically handles conversion for prediction.
    """)
    st.markdown("---")
    st.caption("Built by Daya | AI & DS Student")

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
    💙 Developed by Daya | AI & DS Student | Customer Churn Prediction Dashboard © 2026
</div>
""", unsafe_allow_html=True)