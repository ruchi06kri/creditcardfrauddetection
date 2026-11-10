
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# LOAD TRAINED MODEL
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fraud_detection_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as e:
    st.error(f"Unable to load trained model: {e}")
    st.stop()


# ==========================================
# CUSTOM CSS - COLORFUL INTERFACE
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0e1117;
    color: #ffffff;
}

[data-testid="stSidebar"] {
    background-color: #242633;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: white;
}

/* Red prediction button */
[data-testid="stSidebar"] div.stButton > button {
    background-color: #ff4b4b;
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: bold;
    padding: 12px;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #e63b3b;
    color: white;
}

/* Main title */
h1 {
    color: #ffffff;
    font-weight: 800;
}

/* Dashboard metrics */
[data-testid="stMetricValue"] {
    font-size: 35px;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# APP HEADER
# ==========================================

st.title("💳 Credit Card Fraud Detection")

st.caption("AI-powered transaction risk assessment")

st.write("")


# ==========================================
# SIDEBAR - USER INPUT
# ==========================================

with st.sidebar:

    st.header("Transaction Details")

    step = st.number_input(
        "Transaction step (hour)",
        min_value=1,
        value=1,
        step=1
    )

    transaction_type = st.selectbox(
        "Transaction type",
        [
            "TRANSFER",
            "CASH_OUT",
            "PAYMENT",
            "CASH_IN",
            "DEBIT"
        ]
    )

    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
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


# ==========================================
# DASHBOARD METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Transaction type",
        transaction_type
    )

with col2:
    st.metric(
        "Amount",
        f"{amount:,.2f}"
    )

with col3:
    balance_change = oldbalanceOrg - newbalanceOrig

    st.metric(
        "Sender balance change",
        f"{balance_change:,.2f}"
    )


st.divider()

st.header("Model result")


# ==========================================
# FRAUD PREDICTION
# ==========================================

if analyze:

    # Calculate additional model features.
    # These formulas must match model training.
    balanceDiffOrig = oldbalanceOrg - newbalanceOrig

    balanceDiffDest = newbalanceDest - oldbalanceDest

    # Build model input
    input_data = pd.DataFrame([{
        "step": step,
        "type": transaction_type,
        "amount": amount,
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrig,
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "balanceDiffOrig": balanceDiffOrig,
        "balanceDiffDest": balanceDiffDest
    }])

    # Match the saved model's original input columns
    feature_names = getattr(
        model,
        "feature_names_in_",
        None
    )

    if feature_names is None:
        st.error(
            "The saved model does not expose input "
            "feature names. Verify the training feature "
            "order before making predictions."
        )
        st.stop()

    missing_features = [
        feature for feature in feature_names
        if feature not in input_data.columns
    ]

    if missing_features:
        st.error(
            f"Missing model features: {missing_features}"
        )
        st.stop()

    # Select only expected features in correct order
    input_data = input_data[list(feature_names)]

    try:

        with st.spinner("Analyzing transaction..."):

            # Model prediction
            prediction = int(
                model.predict(input_data)[0]
            )

            # Fraud probability
            fraud_probability = None

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    input_data
                )[0]

                classes = list(model.classes_)

                if 1 in classes:
                    fraud_index = classes.index(1)

                    fraud_probability = float(
                        probabilities[fraud_index]
                    )

        # ==================================
        # DISPLAY PROBABILITY
        # ==================================

        if fraud_probability is not None:

            st.write("Estimated fraud probability")

            st.metric(
                label="Fraud probability",
                value=f"{fraud_probability * 100:.2f}%"
            )

        # ==================================
        # DISPLAY FRAUD RESULT
        # ==================================

        if prediction == 1:

            st.error(
                "Higher-risk prediction: "
                "The model classifies this transaction "
                "as fraudulent."
            )

        else:

            st.success(
                "Lower-risk prediction: "
                "The model does not classify this "
                "transaction as fraudulent at its "
                "current decision threshold."
            )

        # ==================================
        # PROBABILITY PROGRESS BAR
        # ==================================

        if fraud_probability is not None:

            st.progress(
                min(
                    1.0,
                    max(0.0, fraud_probability)
                )
            )

    except Exception as e:

        st.error(
            f"Prediction error: {e}"
        )

else:

    st.info(
        "Enter the transaction details in the sidebar "
        "and click Analyze transaction."
    )


# ==========================================
# EDUCATIONAL DISCLAIMER
# ==========================================

st.warning(
    "This is an educational model, not a banking-grade "
    "fraud decision. A low score does not guarantee "
    "a transaction is safe. Review false positives, "
    "false negatives, calibration, and threshold "
    "choice before any real-world use."
)

st.divider()

st.caption(
    "Features used: transaction step, transaction type, "
    "amount, sender and receiver balances, and any "
    "balance-difference features expected by the "
    "saved model."
)
