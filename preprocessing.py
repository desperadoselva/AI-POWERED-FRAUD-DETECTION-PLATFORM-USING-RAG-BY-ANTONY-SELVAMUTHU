"""
==============================================================
AI Powered Fraud Detection Platform
preprocessing.py

Description
-----------
Preprocesses transaction data for training and prediction.

Responsibilities:
✔ Feature Engineering
✔ Handle Missing Values
✔ Encode Categorical Columns
✔ Scale Numerical Columns
✔ Save Preprocessor
==============================================================
"""

import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


class FraudPreprocessor:

    def __init__(self):

        self.preprocessor = None

    # ----------------------------------------------------
    # Feature Engineering
    # ----------------------------------------------------

    def feature_engineering(self, df):
        df_copy = df.copy()
        # Convert categorical columns that are ints to strings for feature engineering
        # Note: This converts integer codes to their string representation (e.g., 1 -> '1').
        # If feature engineering relies on specific semantic string labels (e.g., 'Mobile'),
        # a proper mapping from integer codes to labels would be needed for logical correctness.
        if 'Device_Type' in df_copy.columns and df_copy['Device_Type'].dtype != 'object':
            df_copy['Device_Type'] = df_copy['Device_Type'].astype(str)
        if 'Merchant_Category' in df_copy.columns and df_copy['Merchant_Category'].dtype != 'object':
            df_copy['Merchant_Category'] = df_copy['Merchant_Category'].astype(str)
        if 'Transaction_Type' in df_copy.columns and df_copy['Transaction_Type'].dtype != 'object':
            df_copy['Transaction_Type'] = df_copy['Transaction_Type'].astype(str)
        return create_features(df_copy)

    # ----------------------------------------------------
    # Build Preprocessor
    # ----------------------------------------------------

    def build(self, df):

        df = self.feature_engineering(df)

        if "Fraud" in df.columns:

            X = df.drop(columns=["Fraud"])

        else:

            X = df.copy()

        numeric_features = X.select_dtypes(
            include=["int64", "float64"]
        ).columns.tolist()

        categorical_features = X.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),

                (
                    "scaler",
                    StandardScaler()
                )

            ]

        )

        categorical_pipeline = Pipeline(

            steps=[

                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),

                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )

            ]

        )

        self.preprocessor = ColumnTransformer(

            transformers=[

                (
                    "num",
                    numeric_pipeline,
                    numeric_features
                ),

                (
                    "cat",
                    categorical_pipeline,
                    categorical_features
                )

            ]

        )

        return self.preprocessor

    # ----------------------------------------------------
    # Fit & Transform
    # ----------------------------------------------------

    def fit_transform(self, df):

        if self.preprocessor is None:

            self.build(df)

        X = df.copy()

        if "Fraud" in X.columns:

            X = X.drop(columns=["Fraud"])

        X = self.feature_engineering(X)

        return self.preprocessor.fit_transform(X)

    # ----------------------------------------------------
    # Transform
    # ----------------------------------------------------

    def transform(self, df):

        X = self.feature_engineering(df)

        return self.preprocessor.transform(X)

    # ----------------------------------------------------
    # Save
    # ----------------------------------------------------

    def save(self, path):

        joblib.dump(

            self.preprocessor,

            path

        )

    # ----------------------------------------------------
    # Load
    # ----------------------------------------------------

    def load(self, path):

        self.preprocessor = joblib.load(path)

        return self.preprocessor


# ==========================================================
# Helper Functions
# ==========================================================

def build_preprocessor(df):

    processor = FraudPreprocessor()

    processor.build(df)

    return processor


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":
    # Define paths
    BASE_DIR = "/content/drive/MyDrive"
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    # Create the models directory if it doesn't exist
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(os.path.join(BASE_DIR, "transactions.csv"))

    processor = FraudPreprocessor()

    X = processor.fit_transform(df)

    processor.save(os.path.join(MODEL_DIR, "preprocessor.pkl"))

    print("=" * 60)

    print("Shape After Preprocessing")

    print(X.shape)

    print("=" * 60)

    print("Preprocessor Saved Successfully")