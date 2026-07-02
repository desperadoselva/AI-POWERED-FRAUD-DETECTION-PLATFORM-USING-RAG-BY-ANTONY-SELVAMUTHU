"""
==============================================================
AI Powered Fraud Detection Platform
dashboard/analytics.py

Description
-----------
Computes dashboard analytics and KPIs.

Compatible with:
✓ Streamlit
✓ Plotly
✓ Pandas
==============================================================
"""

import pandas as pd


class FraudAnalytics:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    # ----------------------------------------------------------
    # Total Transactions
    # ----------------------------------------------------------

    def total_transactions(self):

        return len(self.df)

    # ----------------------------------------------------------
    # Total Fraud Cases
    # ----------------------------------------------------------

    def total_fraud(self):

        return int(self.df["Fraud"].sum())

    # ----------------------------------------------------------
    # Genuine Transactions
    # ----------------------------------------------------------

    def genuine_transactions(self):

        return len(self.df) - self.total_fraud()

    # ----------------------------------------------------------
    # Fraud Percentage
    # ----------------------------------------------------------

    def fraud_percentage(self):

        if len(self.df) == 0:
            return 0

        return round(

            self.total_fraud()

            / len(self.df)

            * 100,

            2

        )

    # ----------------------------------------------------------
    # Total Transaction Amount
    # ----------------------------------------------------------

    def total_amount(self):

        return round(

            self.df["Amount"].sum(),

            2

        )

    # ----------------------------------------------------------
    # Average Transaction Amount
    # ----------------------------------------------------------

    def average_amount(self):

        return round(

            self.df["Amount"].mean(),

            2

        )

    # ----------------------------------------------------------
    # Average Fraud Amount
    # ----------------------------------------------------------

    def average_fraud_amount(self):

        fraud = self.df[

            self.df["Fraud"] == 1

        ]

        if fraud.empty:

            return 0

        return round(

            fraud["Amount"].mean(),

            2

        )

    # ----------------------------------------------------------
    # Maximum Transaction
    # ----------------------------------------------------------

    def maximum_amount(self):

        return self.df["Amount"].max()

    # ----------------------------------------------------------
    # Minimum Transaction
    # ----------------------------------------------------------

    def minimum_amount(self):

        return self.df["Amount"].min()

    # ----------------------------------------------------------
    # Fraud By Hour
    # ----------------------------------------------------------

    def fraud_by_hour(self):

        return (

            self.df

            .groupby("Transaction_Hour")["Fraud"]

            .sum()

            .reset_index()

        )

    # ----------------------------------------------------------
    # Fraud By Merchant
    # ----------------------------------------------------------

    def fraud_by_merchant(self):

        return (

            self.df

            .groupby("Merchant_Category")["Fraud"]

            .sum()

            .reset_index()

            .sort_values(

                "Fraud",

                ascending=False

            )

        )

    # ----------------------------------------------------------
    # Fraud By Device
    # ----------------------------------------------------------

    def fraud_by_device(self):

        return (

            self.df

            .groupby("Device_Type")["Fraud"]

            .sum()

            .reset_index()

        )

    # ----------------------------------------------------------
    # Fraud By Transaction Type
    # ----------------------------------------------------------

    def fraud_by_transaction(self):

        return (

            self.df

            .groupby("Transaction_Type")["Fraud"]

            .sum()

            .reset_index()

        )

    # ----------------------------------------------------------
    # Previous Fraud Distribution
    # ----------------------------------------------------------

    def previous_fraud(self):

        return (

            self.df

            .groupby("Previous_Fraud")

            .size()

            .reset_index(name="Count")

        )

    # ----------------------------------------------------------
    # Age Distribution
    # ----------------------------------------------------------

    def age_distribution(self):

        return (

            self.df

            .groupby("Customer_Age")

            .size()

            .reset_index(name="Count")

        )

    # ----------------------------------------------------------
    # Fraud Summary
    # ----------------------------------------------------------

    def summary(self):

        return {

            "Total Transactions":

                self.total_transactions(),

            "Fraud Cases":

                self.total_fraud(),

            "Genuine Transactions":

                self.genuine_transactions(),

            "Fraud Percentage":

                self.fraud_percentage(),

            "Total Amount":

                self.total_amount(),

            "Average Amount":

                self.average_amount(),

            "Average Fraud Amount":

                self.average_fraud_amount(),

            "Maximum Amount":

                self.maximum_amount(),

            "Minimum Amount":

                self.minimum_amount()

        }


# ==============================================================
# Helper Function
# ==============================================================

def load_analytics(df):

    return FraudAnalytics(df)


# ==============================================================
# Testing
# ==============================================================

if __name__ == "__main__":

    import os

    #BASE_DIR = "/content/drive/MyDrive"

    DATA = "/content/drive/MyDrive/transactions.csv"

    df = pd.read_csv(DATA)

    analytics = FraudAnalytics(df)

    print("=" * 60)

    print("Dashboard Summary")

    print("=" * 60)

    for key, value in analytics.summary().items():

        print(f"{key:<25}: {value}")