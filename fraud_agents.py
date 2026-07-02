"""
==============================================================
AI Powered Fraud Detection Platform
fraud_agent.py

Description
------------
AI Fraud Investigation Agent



Author : Antony Selvamuthu
==============================================================
"""

import os

from dotenv import load_dotenv

from langchain_groq import ChatGroq


# ==========================================================
# Load Environment Variables
# ==========================================================

load_dotenv()

GROQ_API_KEY = "groq_api_key"

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found in .env")


# ==========================================================
# Groq LLM
# ==========================================================

llm = ChatGroq(

    groq_api_key=GROQ_API_KEY,

    model_name="llama-3.3-70b-versatile",

    temperature=0.2,

    max_tokens=1024

)


# ==========================================================
# Fraud Investigation Agent
# ==========================================================

class FraudAgent:

    def __init__(self):

        self.predictor = FraudPredictor()


    # ------------------------------------------------------
    # ML Prediction
    # ------------------------------------------------------

    def predict_transaction(

        self,

        transaction

    ):

        # Unpack the transaction dictionary into keyword arguments
        return self.predictor.predict(**transaction)


    # ------------------------------------------------------
    # Retrieve Knowledge
    # ------------------------------------------------------

    def retrieve_context(

        self,

        transaction,

        prediction

    ):

        query = f"""

        Transaction Amount : {transaction['amount']}

        Transaction Hour : {transaction['transaction_hour']}

        Merchant : {transaction['merchant_category']}

        Device : {transaction['device_type']}

        Fraud Prediction : {prediction['Label']}

        Risk Score : {prediction['Risk Score']}

        Explain the banking policies,
        RBI guidelines,
        AML policy,
        fraud indicators,
        and investigation procedure.

        """

        return retrieve(query)


    # ------------------------------------------------------
    # AI Investigation
    # ------------------------------------------------------

    def investigate(

        self,

        transaction

    ):

        prediction = self.predict_transaction(

            transaction

        )

        context = self.retrieve_context(

            transaction,

            prediction

        )

        system_prompt = build_system_prompt()

        user_prompt = build_investigation_prompt(

            transaction,

            prediction,

            context

        )

        response = llm.invoke(

            [

                ("system", system_prompt),

                ("human", user_prompt)

            ]

        )

        return {

            "prediction": prediction,

            "retrieved_context": context,

            "investigation": response.content

        }


# ==========================================================
# Singleton
# ==========================================================

fraud_agent = FraudAgent()


# ==========================================================
# Helper Function
# ==========================================================

def investigate_transaction(transaction):

    return fraud_agent.investigate(transaction)


# ==========================================================
# Demo
# ==========================================================

if __name__ == "__main__":

    sample_transaction = {

        "amount": 25000,

        "transaction_hour": 2,

        "transaction_type": "Online",

        "merchant_category": "Electronics",

        "device_type": "Mobile",

        "customer_age": 28,

        "previous_fraud": 1

    }

    result = investigate_transaction(

        sample_transaction

    )

    print("=" * 60)

    print("Prediction")

    print("=" * 60)

    print(result["prediction"])

    print()

    print("=" * 60)

    print("AI Investigation")

    print("=" * 60)

    print(result["investigation"])