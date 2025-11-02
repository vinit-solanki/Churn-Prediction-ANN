import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# -------------------------------
# Load model and preprocessing tools
# -------------------------------
model = tf.keras.models.load_model('churn_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("💼 Customer Churn Prediction App")
st.write("Fill in customer details to predict whether they are likely to churn.")

# -------------------------------
# User Inputs
# -------------------------------
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=600)
age = st.number_input('🎂 Age', min_value=18, max_value=92, value=35)
tenure = st.number_input('📅 Tenure (Years)', min_value=0, max_value=10, value=5)
balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0)
num_of_products = st.number_input('🛍️ Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])
estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0)

# -------------------------------
# Data Preprocessing
# -------------------------------

# Encode gender
gender_encoded = label_encoder_gender.transform([gender])[0]

# Create base dataframe
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(
    geo_encoded, 
    columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
)

# Merge all features
input_df = pd.concat([input_data, geo_encoded_df], axis=1)

# Scale the data
input_data_scaled = scaler.transform(input_df)

# -------------------------------
# Prediction
# -------------------------------
prediction = model.predict(input_data_scaled)
prediction_prob = prediction[0][0]

# -------------------------------
# Output
# -------------------------------
st.subheader("📊 Prediction Result:")
if prediction_prob > 0.5:
    st.error(f"The customer is **likely to churn** ⚠️ (Probability: {prediction_prob:.2f})")
else:
    st.success(f"The customer is **unlikely to churn** ✅ (Probability: {prediction_prob:.2f})")

st.write("---")
st.caption("Model powered by TensorFlow & Streamlit")
