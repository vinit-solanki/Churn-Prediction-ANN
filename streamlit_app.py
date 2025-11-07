import streamlit as st
import requests

st.set_page_config(page_title="Customer & Salary Predictor", layout="centered")

# --- Connection ---
st.sidebar.header("⚙️ Settings")
API_BASE = st.sidebar.text_input("Flask API URL", value="http://localhost:5000")

tabs = st.tabs(["💼 Customer Churn", "💰 Salary Prediction"])

# =================== CHURN TAB ===================
with tabs[0]:
    st.title("💼 Customer Churn Prediction")
    st.write("Enter customer details below and get the churn probability instantly.")

    with st.form("churn_form"):
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox('🌍 Geography', ['France', 'Spain', 'Germany'])
            gender = st.selectbox('👤 Gender', ['Male', 'Female'])
            age = st.slider('🎂 Age', 18, 100, 35)
            credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=900, value=650)
            tenure = st.slider('📅 Tenure (Years)', 0, 12, 3)
        with c2:
            balance = st.number_input('🏦 Balance', min_value=0.0, value=10000.0, step=100.0, format="%.2f")
            num_of_products = st.number_input('🛍️ Number of Products', min_value=1, max_value=10, value=1)
            has_cr_card = st.radio('💳 Has Credit Card?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.radio('✅ Is Active Member?', [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input('💰 Estimated Salary', min_value=0.0, value=60000.0, step=100.0)

        submitted = st.form_submit_button("🔍 Predict Churn")

    if submitted:
        payload = {
            "CreditScore": int(credit_score),
            "Gender": gender,
            "Age": int(age),
            "Tenure": int(tenure),
            "Balance": float(balance),
            "NumOfProducts": int(num_of_products),
            "HasCrCard": int(has_cr_card),
            "IsActiveMember": int(is_active_member),
            "EstimatedSalary": float(estimated_salary),
            "Geography": geography
        }

        try:
            r = requests.post(f"{API_BASE}/predict_churn", json=payload, timeout=10)
            data = r.json()
            if data.get("success"):
                prob = data.get("probability", 0)
                label = data.get("label", 0)
                st.metric("Predicted Churn Probability", f"{prob:.2%}")
                if label == 1:
                    st.error("⚠️ High risk of churn detected!")
                else:
                    st.success("✅ Customer likely to stay.")
            else:
                st.error(f"API error: {data.get('error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# =================== SALARY TAB ===================
with tabs[1]:
    st.title("💰 Salary Prediction")
    st.write("Predict a person's salary using experience, education, and location.")

    st.markdown("""
    ### 📋 Fill the following details:
    """)

    with st.form("salary_form"):
        c1, c2 = st.columns(2)

        with c1:
            years_exp = st.slider("🧠 Years of Experience", 0, 40, 5)
            education = st.selectbox(
                "🎓 Education Level",
                ["High School", "Bachelor’s", "Master’s", "PhD"]
            )
        with c2:
            job_level = st.selectbox(
                "🏢 Job Level",
                ["Entry", "Mid", "Senior", "Manager", "Director"]
            )
            city = st.selectbox(
                "📍 City",
                ["New York", "San Francisco", "London", "Bangalore", "Berlin"]
            )

        st.divider()

        st.markdown("### ⚙️ Additional Factors")
        c3, c4 = st.columns(2)
        with c3:
            company_size = st.slider("🏬 Company Size (employees)", 10, 5000, 500, step=50)
        with c4:
            performance_score = st.slider("⭐ Performance Rating (1–10)", 1, 10, 7)

        predict_btn = st.form_submit_button("🚀 Predict Salary")

    if predict_btn:
        # Convert categorical values to numeric placeholders (you can replace this with encoder logic)
        edu_map = {"High School": 0, "Bachelor’s": 1, "Master’s": 2, "PhD": 3}
        job_map = {"Entry": 0, "Mid": 1, "Senior": 2, "Manager": 3, "Director": 4}
        city_map = {"New York": 0, "San Francisco": 1, "London": 2, "Bangalore": 3, "Berlin": 4}

        features = [
            years_exp,
            edu_map[education],
            job_map[job_level],
            city_map[city],
            company_size,
            performance_score
        ]

        # pad to expected length (12)
        while len(features) < 12:
            features.append(0.0)
        payload = {"features": features}


        try:
            r = requests.post(f"{API_BASE}/predict_salary", json=payload, timeout=10)
            data = r.json()
            if data.get("success"):
                st.success(f"💵 Predicted Salary: **${data['predicted_salary']:,.2f}**")
            else:
                st.error(f"API error: {data.get('error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")

    st.info("""
    💡 *Tip:* Increase experience or education level to see how it affects salary predictions in real-time.
    """)

# ========== FOOTER ==========
st.markdown("---")
st.caption("Built with ❤️ using Streamlit + Flask | Predictive Analytics Dashboard")
