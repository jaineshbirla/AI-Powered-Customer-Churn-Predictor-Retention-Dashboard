"""
AI-Powered Customer Churn Predictor & Retention Dashboard
---------------------------------------------------------
A end-to-end Machine Learning web application that predicts 
customer churn risk and offers AI-driven retention strategies.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Page Config
st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

# Title & Description
st.title("🔮 AI-Powered Customer Churn Predictor")
st.write("Predict customer churn probability using Random Forest Classification and get instant retention insights.")

# Cache Data Generation & Model Training
@st.cache_resource
def train_model():
    np.random.seed(42)
    n_samples = 1000
    
    # Synthetic Telco Customer Data
    tenure = np.random.randint(1, 72, size=n_samples)
    monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, size=n_samples)
    tech_support = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])
    contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2]) # 0: Month-to-month, 1: One year, 2: Two year
    
    # Define Churn Logic (Ground Truth Generation)
    churn_prob = (
        0.4 * (contract_type == 0) +
        0.3 * (tenure < 12) +
        0.2 * (tech_support == 0) +
        0.1 * (monthly_charges > 80)
    )
    churn = (churn_prob > np.random.uniform(0, 1, size=n_samples)).astype(int)

    df = pd.DataFrame({
        'Tenure_Months': tenure,
        'Monthly_Charges': monthly_charges,
        'Total_Charges': total_charges,
        'Tech_Support': tech_support,
        'Contract_Type': contract_type,
        'Churn': churn
    })

    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, df, acc, X_test, y_test

model, df, accuracy, X_test, y_test = train_model()

# Sidebar - User Inputs for Live Prediction
st.sidebar.header("📋 Input Customer Profile")

tenure_input = st.sidebar.slider("Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_input = st.sidebar.number_input("Monthly Charges ($)", min_value=20.0, max_value=150.0, value=70.0)
tech_support_input = st.sidebar.selectbox("Has Tech Support?", options=["No", "Yes"])
contract_input = st.sidebar.selectbox("Contract Type", options=["Month-to-Month", "One Year", "Two Year"])

# Format inputs for model prediction
tech_support_val = 1 if tech_support_input == "Yes" else 0
contract_mapping = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
contract_val = contract_mapping[contract_input]
total_charges_val = tenure_input * monthly_input

user_data = pd.DataFrame({
    'Tenure_Months': [tenure_input],
    'Monthly_Charges': [monthly_input],
    'Total_Charges': [total_charges_val],
    'Tech_Support': [tech_support_val],
    'Contract_Type': [contract_val]
})

# Layout Tabs
tab1, tab2 = st.tabs(["🎯 Live Prediction", "📊 Model Performance & Data"])

with tab1:
    st.subheader("Customer Risk Assessment")
    
    if st.sidebar.button("Predict Churn Risk", type="primary"):
        prediction = model.predict(user_data)[0]
        prediction_prob = model.predict_proba(user_data)[0][1] * 100

        col1, col2 = st.columns(2)

        with col1:
            st.metric(label="Churn Probability", value=f"{prediction_prob:.1f}%")
            if prediction == 1:
                st.error("⚠️ **High Risk of Churn!** This customer is likely to cancel their service.")
            else:
                st.success("✅ **Low Risk of Churn!** This customer is likely to stay.")

        with col2:
            st.subheader("💡 Recommended Retention Actions")
            if prediction == 1:
                st.write("- Offer a discount for upgrading to a **One-Year or Two-Year Contract**.")
                st.write("- Provide **free complimentary Tech Support** for 3 months.")
                st.write("- Send a personalized satisfaction check-in email.")
            else:
                st.write("- Offer loyalty rewards or referral discounts.")
                st.write("- Recommend premium service add-ons.")

with tab2:
    st.subheader("Model Overview & Dataset Insights")
    st.write(f"**Random Forest Classifier Accuracy:** `{accuracy * 100:.2f}%`")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.write("#### Feature Importances")
        importances = pd.Series(model.feature_importances_, index=X_test.columns)
        fig, ax = plt.subplots()
        importances.sort_values().plot(kind='barh', ax=ax, color='#4CAF50')
        st.pyplot(fig)

    with col_b:
        st.write("#### Sample Training Data")
        st.dataframe(df.head(10))
