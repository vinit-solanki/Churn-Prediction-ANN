import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('salary_regression_model.h5')

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('onehot_encoder_job.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('onehot_encoder.pkl', 'rb') as f:
    onehot_encoder = pickle.load(f)

st.title('Salary Regression Prediction')
st.write('Enter the details to predict the salary:')
age = st.number_input('Age', min_value=18, max_value=70, value=30)
gender = st.selectbox( )
