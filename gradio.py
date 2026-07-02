"""
==========================================================
AI Powered Fraud Detection Platform

Gradio Chatbot Interface

Author : Antony Selvamuthu
==========================================================
"""

import gradio as gr



# --------------------------------------------------------
# Fraud Investigation Function
# --------------------------------------------------------

def fraud_chatbot(
    amount,
    hour,
    transaction_type,
    merchant,
    device,
    age,
    previous_fraud
):

    transaction = {

        "Amount": float(amount),

        "Transaction_Hour": int(hour),

        "Transaction_Type": int(transaction_type),

        "Merchant_Category": int(merchant),

        "Device_Type": int(device),

        "Customer_Age": int(age),

        "Previous_Fraud": int(previous_fraud)

    }

    try:

        result = investigate_transaction(transaction)

        prediction = result["prediction"]

        report = result["investigation"]

        output = f"""
# AI Fraud Investigation Report

## ML Prediction

**Label:** {prediction['Label']}

**Risk Score:** {prediction['Risk Score']}

---

## AI Investigation

{report}
"""

        return output

    except Exception as e:

        return f"Error: {e}"


# --------------------------------------------------------
# Gradio Interface
# --------------------------------------------------------

with gr.Blocks(title="AI Fraud Detection Platform") as demo:

    gr.Markdown(
        """
# 🛡 AI Powered Fraud Detection Platform

Machine Learning + RAG + Groq
"""
    )

    with gr.Row():

        amount = gr.Number(
            label="Transaction Amount",
            value=5000
        )

        hour = gr.Slider(
            0,
            23,
            value=12,
            step=1,
            label="Transaction Hour"
        )

    with gr.Row():

        transaction_type = gr.Dropdown(

            choices=[
                (0, "Online"),
                (1, "ATM"),
                (2, "POS")
            ],

            value=0,

            label="Transaction Type"

        )

        merchant = gr.Number(
            label="Merchant Category",
            value=0
        )

    with gr.Row():

        device = gr.Number(
            label="Device Type",
            value=0
        )

        age = gr.Number(
            label="Customer Age",
            value=30
        )

        previous_fraud = gr.Number(
            label="Previous Fraud",
            value=0
        )

    investigate = gr.Button(
        "🔍 Investigate Transaction"
    )

    result = gr.Markdown()

    investigate.click(

        fn=fraud_chatbot,

        inputs=[

            amount,

            hour,

            transaction_type,

            merchant,

            device,

            age,

            previous_fraud

        ],

        outputs=result

    )

# --------------------------------------------------------
# Launch
# --------------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        share=True
    )