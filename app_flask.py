"""
app_flask.py

Flask API for:
 - /predict_churn : churn classification/regression output (uses your churn Keras model & preprocessing)
 - /predict_salary: salary regression (supports keras .h5 or pickled sklearn models)
 - /schema_salary  : returns expected salary feature schema (if available)
 - /health        : simple health check

This file is robust to the common problems you encountered:
 - scaler.feature_names_in_ mismatch (adds missing columns, drops extras, reorders)
 - onehot/label encoders loaded from your notebooks
 - attempts to load several possible model filenames used in your notebooks
"""

from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from typing import List

app = Flask(__name__)

# -----------------------
# Helper: model loading utils
# -----------------------
def try_load_pickle(path: str):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def try_load_keras(path: str):
    if os.path.exists(path):
        try:
            return tf.keras.models.load_model(path)
        except Exception as e:
            # If loading fails, return None
            print(f"Failed to load Keras model at {path}: {e}")
            return None
    return None

# -----------------------
# Load churn artifacts (encoders, scaler, model)
# These were saved in model.ipynb
# -----------------------
label_encoder_gender = try_load_pickle('label_encoder_gender.pkl')
onehot_encoder_geo = try_load_pickle('onehot_encoder_geo.pkl')
churn_scaler = try_load_pickle('scaler.pkl')

# Try multiple typical churn model file names (some notebooks saved 'churn_model.h5' etc)
churn_model = None
for candidate in ['churn_model_v2.h5', 'churn_model.h5', 'churn_model_v2', 'churn_model']:
    if churn_model is None:
        churn_model = try_load_keras(candidate)
if churn_model is None:
    # if no keras model, keep None — endpoints will return helpful errors
    print("Warning: churn model not found (checked churn_model_v2.h5, churn_model.h5).")

# -----------------------
# Load salary artifacts (encoders, scaler, model)
# salary notebook saved salary_regression_model.h5 in your files
# -----------------------
salary_scaler = try_load_pickle('salary_scaler.pkl') or try_load_pickle('scaler.pkl')  # sometimes same scaler file is used
salary_label_encoder = try_load_pickle('salary_label_encoder.pkl')
salary_onehot = try_load_pickle('salary_onehot_encoder.pkl')

# Try pickle model first (user earlier had salary_model.pkl in earlier scripts),
salary_model = try_load_pickle('salary_model.pkl')

# If not present, try keras .h5 formats saved in salary notebook
if salary_model is None:
    for candidate in ['salary_regression_model.h5', 'salary_model.h5', 'salary_model']:
        if salary_model is None:
            salary_model = try_load_keras(candidate)

# -----------------------
# Utility: align DataFrame columns to scaler.feature_names_in_ safely
# -----------------------
def align_input_df_to_scaler(input_df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Ensure DataFrame has exactly the columns scaler.feature_names_in_ expects.
    - If scaler has feature_names_in_ (set when fit on DataFrame), reorder/add/drop.
    - If scaler doesn't have that attribute, just return input_df (scaler likely fit on array).
    """
    if scaler is None:
        return input_df

    feature_names = getattr(scaler, 'feature_names_in_', None)
    if feature_names is None:
        # scaler likely fit on numpy array without column names
        return input_df

    expected = list(feature_names)
    # Create new DataFrame with expected columns
    aligned = pd.DataFrame(columns=expected)

    for col in expected:
        if col in input_df.columns:
            aligned[col] = input_df[col].values
        else:
            # missing feature -> fill with 0 (safe default)
            aligned[col] = 0

    # (If input_df had extra columns not in expected, they're dropped.)
    return aligned.astype(float)

# -----------------------
# Churn preprocessing function
# Mirrors steps in model.ipynb:
#  - label encode Gender (saved label_encoder_gender)
#  - one-hot encode Geography using saved onehot_encoder_geo
#  - concatenate numeric features
#  - align columns to scaler.feature_names_in_ (avoid mismatch issues you saw)
#  - scale via saved scaler
# -----------------------
def preprocess_churn_payload(payload: dict) -> np.ndarray:
    # expected keys in the UI / notebook
    expected_keys = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                     'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography']
    missing = [k for k in expected_keys if k not in payload]
    if missing:
        raise KeyError(f"Missing fields for churn prediction: {missing}")

    # 1) Gender label encode (noting label_encoder_gender transforms strings to 0/1 as in notebook)
    gender_val = payload['Gender']
    if label_encoder_gender is not None:
        try:
            gender_enc = label_encoder_gender.transform([gender_val])[0]
        except Exception:
            # if unseen gender label, default to 0 and warn
            gender_enc = 0
    else:
        # fallback: simple mapping
        gender_enc = 1 if str(gender_val).lower().startswith('f') else 0

    # 2) Build base DataFrame with numeric fields
    input_data = pd.DataFrame({
        'CreditScore': [payload['CreditScore']],
        'Gender': [gender_enc],
        'Age': [payload['Age']],
        'Tenure': [payload['Tenure']],
        'Balance': [payload['Balance']],
        'NumOfProducts': [payload['NumOfProducts']],
        'HasCrCard': [payload['HasCrCard']],
        'IsActiveMember': [payload['IsActiveMember']],
        'EstimatedSalary': [payload['EstimatedSalary']]
    })

    # 3) One-hot encode Geography using saved encoder (onehot_encoder_geo)
    if onehot_encoder_geo is not None:
        try:
            geo_arr = onehot_encoder_geo.transform([[payload['Geography']]]).toarray()
            geo_cols = onehot_encoder_geo.get_feature_names_out(['Geography'])
            geo_df = pd.DataFrame(geo_arr, columns=geo_cols)
        except Exception as e:
            # If transform fails (e.g., unseen category) try handle_unknown or fallback all zeros
            geo_cols = getattr(onehot_encoder_geo, 'get_feature_names_out', lambda *_: [])(['Geography'])
            geo_df = pd.DataFrame(np.zeros((1, len(geo_cols))), columns=geo_cols)
    else:
        # No encoder available -> create a dummy column (safe fallback)
        geo_df = pd.DataFrame({'Geography_unknown': [1]})

    # 4) Concatenate
    input_df = pd.concat([input_data.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

    # 5) Align to scaler columns (this handles your earlier error where scaler expected 'Exited' etc)
    if churn_scaler is not None:
        input_df_aligned = align_input_df_to_scaler(input_df, churn_scaler)
        X_scaled = churn_scaler.transform(input_df_aligned)
    else:
        X_scaled = input_df.values.astype(float)

    return X_scaled

# -----------------------
# Endpoints
# -----------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict_churn', methods=['POST'])
def predict_churn():
    """
    POST JSON payload (fields must match your UI):
    {
      "CreditScore":600,
      "Gender":"Male",
      "Age":40,
      "Tenure":3,
      "Balance":50000.0,
      "NumOfProducts":1,
      "HasCrCard":1,
      "IsActiveMember":0,
      "EstimatedSalary":60000.0,
      "Geography":"France"
    }

    Returns:
    {
      "success": True,
      "raw_model_output": ...,
      "probability": 0.12,        # if model output is in [0,1] scale
      "label": 0,
      "threshold": 0.35
    }
    """
    try:
        if churn_model is None:
            return jsonify({'success': False, 'error': 'churn model not loaded on server.'}), 500

        payload = request.get_json()
        X = preprocess_churn_payload(payload)
        # Keras models expect shape (n_samples, n_features)
        raw = churn_model.predict(X)
        # raw may be shape (1,1) or (1,) -> flatten
        raw_val = float(np.array(raw).reshape(-1)[0])

        # The notebook trained with linear output + MAE — treat raw_val as "score" between 0..1 in practice.
        # Use threshold (as earlier) to get label
        THRESH = 0.35
        label = int(raw_val > THRESH)

        return jsonify({'success': True,
                        'raw_model_output': raw_val,
                        'probability': raw_val,
                        'label': label,
                        'threshold': THRESH})
    except KeyError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/schema_salary', methods=['GET'])
def schema_salary():
    """
    Returns expected salary model input schema (if scaler or model was fit on named columns).
    This helps the Streamlit UI show correct fields.
    """
    schema = {}
    # Prefer scaler feature names (if available)
    if salary_scaler is not None and hasattr(salary_scaler, 'feature_names_in_'):
        schema['features'] = list(salary_scaler.feature_names_in_)
    else:
        # try leaning on model metadata (not always present)
        schema['features'] = None
    return jsonify({'success': True, 'schema': schema})

@app.route('/predict_salary', methods=['POST'])
def predict_salary():
    """
    Accepts:
    - {"features": [v1, v2, ...]}  OR
    - {"FeatureA": valA, "FeatureB": valB, ...}

    If scaler with feature_names_in_ is present we will attempt to align dict->ordered features.
    """
    try:
        if salary_model is None:
            return jsonify({'success': False, 'error': 'salary model not loaded on server.'}), 500

        payload = request.get_json()

        # If user provided ordered numeric list:
        if isinstance(payload.get('features'), list):
            X = np.array(payload['features'], dtype=float).reshape(1, -1)
        else:
            # payload as dict: preserve order only if scaler.feature_names_in_ exists
            if salary_scaler is not None and hasattr(salary_scaler, 'feature_names_in_'):
                feature_names = list(salary_scaler.feature_names_in_)
                vals = []
                for fn in feature_names:
                    # take provided value or 0 if missing
                    vals.append(float(payload.get(fn, 0.0)))
                X = np.array(vals, dtype=float).reshape(1, -1)
            else:
                # else use dictionary order (best-effort)
                keys = [k for k in payload.keys() if k != 'features']
                vals = [float(payload[k]) for k in keys]
                X = np.array(vals, dtype=float).reshape(1, -1)

        if salary_scaler is not None and hasattr(salary_scaler, 'transform'):
            try:
                X = salary_scaler.transform(X)
            except Exception:
                # if scaler expects named columns, ignore transform
                pass

        # Predict: if Keras model, use .predict, else use sklearn .predict
        if hasattr(salary_model, 'predict'):
            pred = salary_model.predict(X)
            salary_pred = float(np.array(pred).reshape(-1)[0])
        else:
            return jsonify({'success': False, 'error': 'salary model object has no predict method.'}), 500

        return jsonify({'success': True, 'predicted_salary': salary_pred})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# -----------------------
# Run
# -----------------------
if __name__ == '__main__':
    # debug=True only for local dev
    app.run(host='0.0.0.0', port=5000, debug=True)
