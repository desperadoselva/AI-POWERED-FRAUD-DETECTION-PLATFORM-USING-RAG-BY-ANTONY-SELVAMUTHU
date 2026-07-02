"""
==============================================================
AI Powered Fraud Detection Platform
dashboard/charts.py

Description
-----------
Plotly charts for the fraud detection dashboard.


==============================================================
"""

import plotly.express as px


class FraudCharts:

    def __init__(self, dataframe):

        self.df = dataframe.copy()

    # ==========================================================
    # Fraud vs Genuine Pie Chart
    # ==========================================================

    def fraud_distribution(self):

        counts = self.df["Fraud"].value_counts()

        fig = px.pie(

            names=["Genuine", "Fraud"],

            values=[
                counts.get(0, 0),
                counts.get(1, 0)
            ],

            title="Fraud Distribution"

        )

        return fig

    # ==========================================================
    # Fraud by Hour
    # ==========================================================

    def fraud_by_hour(self):

        data = (

            self.df

            .groupby("Transaction_Hour")["Fraud"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            data,

            x="Transaction_Hour",

            y="Fraud",

            title="Fraud Cases by Transaction Hour"

        )

        return fig

    # ==========================================================
    # Transaction Amount Distribution
    # ==========================================================

    def amount_distribution(self):

        fig = px.histogram(

            self.df,

            x="Amount",

            nbins=30,

            title="Transaction Amount Distribution"

        )

        return fig

    # ==========================================================
    # Fraud by Merchant Category
    # ==========================================================

    def merchant_chart(self):

        data = (

            self.df

            .groupby("Merchant_Category")["Fraud"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            data,

            x="Merchant_Category",

            y="Fraud",

            title="Fraud by Merchant Category"

        )

        return fig

    # ==========================================================
    # Fraud by Device Type
    # ==========================================================

    def device_chart(self):

        data = (

            self.df

            .groupby("Device_Type")["Fraud"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            data,

            x="Device_Type",

            y="Fraud",

            title="Fraud by Device Type"

        )

        return fig

    # ==========================================================
    # Fraud by Transaction Type
    # ==========================================================

    def transaction_type_chart(self):

        data = (

            self.df

            .groupby("Transaction_Type")["Fraud"]

            .sum()

            .reset_index()

        )

        fig = px.bar(

            data,

            x="Transaction_Type",

            y="Fraud",

            title="Fraud by Transaction Type"

        )

        return fig

    # ==========================================================
    # Customer Age Distribution
    # ==========================================================

    def age_distribution(self):

        fig = px.histogram(

            self.df,

            x="Customer_Age",

            nbins=20,

            title="Customer Age Distribution"

        )

        return fig

    # ==========================================================
    # Previous Fraud History
    # ==========================================================

    def previous_fraud_chart(self):

        data = (

            self.df

            .groupby("Previous_Fraud")

            .size()

            .reset_index(name="Count")

        )

        fig = px.bar(

            data,

            x="Previous_Fraud",

            y="Count",

            title="Previous Fraud History"

        )

        return fig

    # ==========================================================
    # Fraud by Amount (Box Plot)
    # ==========================================================

    def fraud_amount_box(self):

        fig = px.box(

            self.df,

            x="Fraud",

            y="Amount",

            title="Fraud vs Transaction Amount"

        )

        return fig

    # ==========================================================
    # Correlation Heatmap
    # ==========================================================

    def correlation_heatmap(self):

        numeric = self.df.select_dtypes(include="number")

        corr = numeric.corr()

        fig = px.imshow(

            corr,

            text_auto=True,

            title="Feature Correlation"

        )

        return fig


# ==========================================================
# Helper Function
# ==========================================================

def get_charts(df):

    return FraudCharts(df)


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    import os
    import pandas as pd

   # BASE_DIR = os.path.dirname(
  #      os.path.dirname(os.path.abspath(__file__))
   # )

    DATA_PATH = "/content/drive/MyDrive/transactions.csv"

    df = pd.read_csv(DATA_PATH)

    charts = FraudCharts(df)

    charts.fraud_distribution().show()