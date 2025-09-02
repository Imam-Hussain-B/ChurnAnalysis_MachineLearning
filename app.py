# coding: utf-8

import os
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# ---- Paths ----
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "Dataset", "tel_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models_pipeline.pkl")

# ---- Load training dataset ----
df_1 = pd.read_csv(DATA_PATH)

# ---- Load saved models ----
with open(MODEL_PATH, "rb") as f:
    models = pickle.load(f)

# Unpack models
rf_model = models["random_forest"]
# logistic_model = models["logistic_regression"]   # if needed

@app.route("/")
def loadPage():
    return render_template("index.html", query="")

@app.route("/", methods=["POST"])
def predict():
    # ---- Collect form inputs ----
    inputs = [request.form[f"query{i}"] for i in range(1, 20)]

    # ---- Create DataFrame ----
    new_df = pd.DataFrame([inputs], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender',
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure'
    ])

    # Convert numeric columns safely
    numeric_cols = ['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'tenure']
    for col in numeric_cols:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0)

    # ---- Append for consistent preprocessing ----
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # ---- Bin tenure ----
    labels = [f"{i} - {i+11}" for i in range(1, 72, 12)]
    df_2["tenure_group"] = pd.cut(
        df_2.tenure.astype(int), range(1, 80, 12),
        right=False, labels=labels
    )

    # ---- Drop unused ----
    df_2.drop(columns=["tenure"], inplace=True)

    # ---- One-hot encode ----
    df_2_dummies = pd.get_dummies(df_2[[
        'gender', 'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'InternetService',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
        'PaymentMethod', 'tenure_group'
    ]])

    # ---- Take input row ----
    input_processed = df_2_dummies.tail(1)

    # Align with training model features
    input_processed = input_processed.reindex(
        columns=rf_model.feature_names_in_, fill_value=0
    )

    # ---- Random Forest Prediction ----
    rf_pred = rf_model.predict(input_processed)[0]
    rf_prob = rf_model.predict_proba(input_processed)[:, 1][0]

    def format_output(pred, prob):
        if pred == 1:
            return ("This customer is likely to churn!!",
                    f"Confidence: {prob * 100:.2f}%")
        else:
            return ("This customer is likely to continue!!",
                    f"Confidence: {prob * 100:.2f}%")

    o5, o6 = format_output(rf_pred, rf_prob)

    return render_template(
        "index.html",
        output5=o5, output6=o6,
        **{f"query{i}": request.form[f"query{i}"] for i in range(1, 20)}
    )

if __name__ == "__main__":
    app.run(debug=True)
