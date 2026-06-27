# Customer Churn Prediction

**Live Demo:** https://customer-churn-prediction-wbnvevdbfxkkgxyvqs4fnu.streamlit.app/

A Machine Learning web application built with Streamlit to predict customer churn for a telecommunications company. The app allows users to input customer details and receive real-time predictions on whether a customer is likely to leave.

### 📌 Project Overview
This project aims to solve a critical business problem for subscription-based companies: identifying customers at high risk of churning. By leveraging machine learning, businesses can proactively implement retention strategies and reduce revenue loss.

### 🎯 Problem Statement
Customer churn directly impacts a company's revenue and growth. The goal of this project is to build a predictive model that accurately identifies customers who are likely to discontinue their service, enabling the business to take preventive action.

### 🛠️ Technologies Used
- **Python**: Core programming language
- **Pandas & NumPy**: Data manipulation and analysis
- **Scikit-learn**: Model building and evaluation
- **Streamlit**: Web application framework and deployment
- **Joblib**: Model serialization
- **Git & GitHub**: Version control

### 📂 Dataset
**Telco Customer Churn Dataset** from Kaggle  
This dataset contains customer information including services signed up for, account information, and demographic data. The target variable is `Churn`, indicating whether the customer left within the last month.

### 🔍 Steps Performed
1. **Data Cleaning**: Handled missing values, converted data types, and processed categorical variables.
2. **Exploratory Data Analysis (EDA)**: Analyzed feature distributions and correlations with churn.
3. **Data Visualization**: Used Matplotlib and Seaborn to identify key patterns and insights.
4. **Feature Engineering**: Encoded categorical variables and scaled numerical features.
5. **Model Building**: Trained and compared multiple classification models.
6. **Model Evaluation**: Evaluated models using accuracy, precision, recall, and F1-score.

### 🤖 Machine Learning Model
The final deployed model is a **Random Forest Classifier**, chosen for its high accuracy and robustness on this dataset. The model was serialized using Joblib and integrated into the Streamlit application for real-time inference.

Key input features for the deployed app:
- **Tenure**: Number of months the customer has stayed with the company.
- **Monthly Charges**: The amount charged to the customer monthly.

### 📊 Result
The model achieved **~80% accuracy** on the test dataset. The most influential factors for churn were identified as contract type, tenure, and monthly charges.

### 🚀 Outcome
A fully functional and deployed web application that provides instant churn predictions. This tool can help business teams quickly assess customer risk without needing technical expertise.

### ⚙️ How to Run Locally
1. **Clone the repository**
   ```bash
   git clone https://github.com/rdayanitha11-hub/customer-churn-prediction.git
   cd customer-churn-prediction
