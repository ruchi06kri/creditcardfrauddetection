
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load trained model
@st.cache_resource
def load_model():
    model_path = (
        Path(__file__).resolve().parent
        / "fraud_detection_pipeline.pkl"
    )
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# Custom styling
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #202330;
}

[data-testid="stSidebar"] div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 600;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #e63d3d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("💳 Credit Card Fraud Detection")
st.caption("AI-powered transaction risk assessment")

# Sidebar input form
with st.sidebar:
    st.header("Transaction Details")

    step = st.number_input(
        "Transaction step (hour)",
        min_value=1,
        value=1
    )

    transaction_type = st.selectbox(
        "Transaction type",
        ["TRANSFER", "CASH_OUT", "PAYMENT",
         "CASH_IN", "DEBIT"]
    )

    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=1000.0
    )

    st.subheader("Sender balances")

    oldbalanceOrg = st.number_input(
        "Sender balance before",
        min_value=0.0,
        value=5000.0
    )

    newbalanceOrig = st.number_input(
        "Sender balance after",
        min_value=0.0,
        value=0.0
    )

    st.subheader("Receiver balances")

    oldbalanceDest = st.number_input(
        "Receiver balance before",
        min_value=0.0,
        value=0.0
    )

    newbalanceDest = st.number_input(
        "Receiver balance after",
        min_value=0.0,
        value=500.0
    )

    analyze = st.button(
        "Analyze transaction",
        use_container_width=True
    )

# Main dashboard
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Transaction type", transaction_type)

with col2:
    st.metric("Amount", f"{amount:,.2f}")

with col3:
    balance_change = oldbalanceOrg - newbalanceOrig
    st.metric(
        "Sender balance change",
        f"{balance_change:,.2f}"
    )

st.divider()

st.header("Model result")

if analyze:
    input_data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest
    }])

    # Use the same input features as training
    feature_names = getattr(
        model, "feature_names_in_", None
    )

    if feature_names is not None:
        missing = [
            col for col in feature_names
            if col not in input_data.columns
        ]
        if missing:
            st.error(f"Missing model features: {missing}")
            st.stop()

        input_data = input_data[list(feature_names)]

    try:
        prediction = int(model.predict(input_data)[0])

        if hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            if 1 in classes:
                fraud_index = classes.index(1)
                probability = float(
                    model.predict_proba(input_data)[0][fraud_index]
                )

                st.write("Estimated fraud probability")
                st.metric(
                    "Fraud probability",
                    f"{probability * 100:.2f}%"
                )

                st.progress(
                    max(0.0, min(1.0, probability))
                )

        if prediction == 1:
            st.error(
                "Higher-risk prediction: The model "
                "classifies this transaction as fraudulent."
            )
        else:
            st.success(
                "Lower-risk prediction: The model does not "
                "classify this transaction as fraudulent "
                "at its current decision threshold."
            )

    except Exception as e:
        st.error(f"Prediction error: {e}")

else:
    st.info(
        "Enter the transaction details in the sidebar "
        "and click Analyze transaction."
    )

st.warning(
    "This is an educational model, not a banking-grade "
    "fraud decision. A low score does not guarantee a "
    "transaction is safe."
)

st.divider()

st.caption(
    "Features: transaction step, type, amount, "
    "and sender/receiver balances."
)
