"""
==============================================================
AI Powered Fraud Detection Platform
predict.py

Author : Antony Selvamuthu

Description
------------
Loads the trained model and predicts whether a
transaction is fraudulent.

Uses
-----
✔ fraud_model.pkl
✔ preprocessor.pkl
✔ feature_columns.pkl

Compatible with:
    • Streamlit
    • FastAPI
    • LangChain
    • LangGraph
    • Gradio
==============================================================
"""

import os
import joblib
import pandas as pd


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = "/content/drive/MyDrive"

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


# ==========================================================
# Load Files
# ==========================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("fraud_model.pkl not found.")

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError("preprocessor.pkl not found.")

if not os.path.exists(FEATURE_PATH):
    raise FileNotFoundError("feature_columns.pkl not found.")


model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

feature_columns = joblib.load(FEATURE_PATH)


# ==========================================================
# Fraud Predictor
# ==========================================================

class FraudPredictor:

    def __init__(self):

        self.model = model
        self.preprocessor = preprocessor
        self.feature_columns = feature_columns

    # ======================================================
    # Preprocess
    # ======================================================

    def preprocess(self, dataframe):

        features = self.preprocessor.transform(dataframe)

        return features


    # ======================================================
    # Risk Level
    # ======================================================

    def risk_level(self, score):

        if score < 30:
            return "LOW"

        elif score < 70:
            return "MEDIUM"

        else:
            return "HIGH"


    # ======================================================
    # Prediction
    # ======================================================

    def predict(
            self,
            amount,
            transaction_hour,
            transaction_type,
            merchant_category,
            device_type,
            customer_age,
            previous_fraud
    ):

        data = pd.DataFrame({

            "Amount":[amount],

            "Transaction_Hour":[transaction_hour],

            "Transaction_Type":[transaction_type],

            "Merchant_Category":[merchant_category],

            "Device_Type":[device_type],

            "Customer_Age":[customer_age],

            "Previous_Fraud":[previous_fraud]

        })

        processed = self.preprocess(data)

        prediction = self.model.predict(processed)[0]

        probability = self.model.predict_proba(processed)[0]

        fraud_probability = probability[1]

        risk_score = round(fraud_probability * 100,2)

        if prediction == 1:
            label = "Fraudulent Transaction"
        else:
            label = "Legitimate Transaction"

        reasons = []

        if amount > 10000:
            reasons.append("High transaction amount")

        if transaction_hour < 5:
            reasons.append("Transaction during unusual hours")

        if previous_fraud == 1:
            reasons.append("Customer has previous fraud history")

        if device_type.lower() == "unknown":
            reasons.append("Unknown device detected")

        if merchant_category.lower() in [
            "electronics",
            "crypto",
            "gift cards"
        ]:
            reasons.append("High-risk merchant category")

        if len(reasons) == 0:
            reasons.append("No obvious suspicious indicators")

        result = {

            "Prediction": int(prediction),

            "Label": label,

            "Fraud Probability": round(fraud_probability,4),

            "Risk Score": risk_score,

            "Risk Level": self.risk_level(risk_score),

            "Reasons": reasons

        }

        return result


# ==========================================================
# Batch Prediction
# ==========================================================

def predict_csv(csv_path):

    predictor = FraudPredictor()

    df = pd.read_csv(csv_path)

    processed = predictor.preprocess(df)

    prediction = predictor.model.predict(processed)

    probability = predictor.model.predict_proba(processed)

    df["Prediction"] = prediction

    df["Fraud Probability"] = probability[:,1]

    df["Risk Score"] = probability[:,1] * 100

    df["Risk Level"] = df["Risk Score"].apply(
        predictor.risk_level
    )

    df["Prediction"] = df["Prediction"].replace({

        0:"Legitimate",

        1:"Fraud"

    })

    return df


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    predictor = FraudPredictor()

    result = predictor.predict(

        amount=25000,

        transaction_hour=2,

        transaction_type="Online",

        merchant_category="Electronics",

        device_type="Mobile",

        customer_age=32,

        previous_fraud=1

    )

    print("\nPrediction Result\n")

    print("="*50)

    for key,value in result.items():

        print(f"{key} : {value}")