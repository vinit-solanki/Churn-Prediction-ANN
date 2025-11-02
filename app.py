import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pickle

# ======================================================
# 1️⃣ Load Model and Preprocessing Objects
# ======================================================
st.set_page_config(page_title="Customer Churn Prediction", page_icon="💼", layout="centered")

@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model('churn_model_v2.h5')

    with open('label_encoder_gender.pkl', 'rb') as f:
        label_encoder_gender = pickle.load(f)

    with open('onehot_encoder_geo.pkl', 'rb') as f:
        onehot_encoder_geo = pickle.load(f)

    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_tools()

# ⚙️ Use the optimized threshold from your training script
OPTIMAL_THRESHOLD = 0.35  # 🔧 update this to your best_threshold from model training

# ======================================================
# 2️⃣ App Header
# ======================================================
st.title("💼 Customer Churn Prediction App")
st.markdown("""
This app predicts **whether a customer is likely to churn** based on demographic and account details.  
The prediction uses an **Artificial Neural Network (ANN)** optimized for **high recall** on churners.
""")

st.divider()

# ======================================================
# 3️⃣ Collect User Inputs
# ======================================================
st.header("🧾 Customer Details")

col1, col2 = st.columns(2)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
    age = st.number_input('🎂 Age', min_value=18, max_value=92, value=35)
    credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=600)
    tenure = st.number_input('📅 Tenure (Years)', min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0)
    num_of_products = st.number_input('🛍️ Number of Products', min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
    is_active_member = st.selectbox('✅ Is Active Member?', [0, 1])
    estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=50000.0)

# ======================================================
# 4️⃣ Data Preprocessing
# ======================================================
if st.button("🔍 Predict Churn"):
    # Encode gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Create base DataFrame
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
    input_scaled = scaler.transform(input_df)

    # ======================================================
    # 5️⃣ Prediction
    # ======================================================
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction_label = int(prediction_prob > OPTIMAL_THRESHOLD)

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction_label == 1:
        st.error(f"⚠️ The customer is **likely to churn**.")
        st.write(f"**Churn Probability:** `{prediction_prob:.2f}` (Threshold = {OPTIMAL_THRESHOLD})")
    else:
        st.success(f"✅ The customer is **unlikely to churn**.")
        st.write(f"**Churn Probability:** `{prediction_prob:.2f}` (Threshold = {OPTIMAL_THRESHOLD})")

    st.progress(prediction_prob if prediction_prob <= 1 else 1.0)
    st.caption("Higher probability = higher churn risk")

st.write("---")
st.caption("Powered by TensorFlow • Streamlit • Scikit-learn")