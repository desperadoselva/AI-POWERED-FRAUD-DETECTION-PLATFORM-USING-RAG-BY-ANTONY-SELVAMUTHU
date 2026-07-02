"""
==============================================================
AI Powered Fraud Detection Platform
feature_engineering.py

Author : Antony Selvamuthu

Description
------------
Creates additional features to improve fraud detection.

Compatible with:
    ✔ train_model.py
    ✔ predict.py
    ✔ Streamlit
    ✔ FastAPI
    ✔ LangChain
==============================================================
"""

import pandas as pd
import numpy as np


class FeatureEngineer:

    def __init__(self):
        pass

    # ======================================================
    # Transaction Amount Categories
    # ======================================================

    def amount_category(self, df):

        df = df.copy()

        bins = [0, 1000, 5000, 10000, np.inf]

        labels = [
            "Low",
            "Medium",
            "High",
            "Very_High"
        ]

        df["Amount_Category"] = pd.cut(
            df["Amount"],
            bins=bins,
            labels=labels
        )

        return df

    # ======================================================
    # Night Transaction Flag
    # ======================================================

    def night_transaction(self, df):

        df = df.copy()

        df["Night_Transaction"] = (
            (df["Transaction_Hour"] >= 0) &
            (df["Transaction_Hour"] <= 5)
        ).astype(int)

        return df

    # ======================================================
    # Weekend Transaction
    # ======================================================

    def weekend_transaction(self, df):

        df = df.copy()

        if "Transaction_Day" in df.columns:

            df["Weekend"] = df["Transaction_Day"].isin(
                ["Saturday", "Sunday"]
            ).astype(int)

        return df

    # ======================================================
    # High Value Transaction
    # ======================================================

    def high_value_transaction(self, df):

        df = df.copy()

        df["High_Value"] = (
            df["Amount"] >= 10000
        ).astype(int)

        return df

    # ======================================================
    # Customer Risk Score
    # ======================================================

    def customer_risk(self, df):

        df = df.copy()

        risk = np.zeros(len(df))

        risk += (df["Previous_Fraud"] * 50)

        risk += (df["Amount"] > 10000) * 20

        risk += (
            (df["Transaction_Hour"] < 5)
        ) * 15

        df["Customer_Risk_Score"] = risk

        return df

    # ======================================================
    # Age Groups
    # ======================================================

    def age_group(self, df):

        df = df.copy()

        bins = [18, 25, 40, 60, 100]

        labels = [
            "Young",
            "Adult",
            "Senior",
            "Elder"
        ]

        df["Age_Group"] = pd.cut(
            df["Customer_Age"],
            bins=bins,
            labels=labels
        )

        return df

    # ======================================================
    # Merchant Risk
    # ======================================================

    def merchant_risk(self, df):

        df = df.copy()

        risky_merchants = [

            "Electronics",

            "Crypto",

            "Gift Cards",

            "Jewellery",

            "Luxury"

        ]

        df["Merchant_Risk"] = df[
            "Merchant_Category"
        ].isin(risky_merchants).astype(int)

        return df

    # ======================================================
    # Device Risk
    # ======================================================

    def device_risk(self, df):

        df = df.copy()

        df["Unknown_Device"] = (
            df["Device_Type"].str.lower() == "unknown"
        ).astype(int)

        return df

    # ======================================================
    # Amount per Age
    # ======================================================

    def amount_per_age(self, df):

        df = df.copy()

        df["Amount_Per_Age"] = (
            df["Amount"] /
            (df["Customer_Age"] + 1)
        )

        return df

    # ======================================================
    # Complete Pipeline
    # ======================================================

    def engineer_features(self, df):

        df = self.amount_category(df)

        df = self.night_transaction(df)

        df = self.weekend_transaction(df)

        df = self.high_value_transaction(df)

        df = self.customer_risk(df)

        df = self.age_group(df)

        df = self.merchant_risk(df)

        df = self.device_risk(df)

        df = self.amount_per_age(df)

        return df


# ==========================================================
# Standalone Function
# ==========================================================

def create_features(df):

    engineer = FeatureEngineer()

    return engineer.engineer_features(df)


# ==========================================================
# Example
# ==========================================================

if __name__ == "__main__":

    sample = pd.DataFrame({

        "Amount":[15000],

        "Transaction_Hour":[2],

        "Transaction_Type":["Online"],

        "Merchant_Category":["Electronics"],

        "Device_Type":["Mobile"],

        "Customer_Age":[30],

        "Previous_Fraud":[1]

    })

    feature_engineer = FeatureEngineer()

    output = feature_engineer.engineer_features(sample)

    print(output.head())