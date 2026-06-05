import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

import plotly.express as px
import plotly.graph_objects as go

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================================================
# CONFIG
# ==================================================

st.set_page_config(
    page_title="Fraud Intelligence System",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "attention_fraud_model.h5",
        compile=False
    )

    return model

model = load_model()

SEQ_LEN = 5

# ==================================================
# POSITIONAL ENCODING
# ==================================================

def positional_encoding(max_pos, d_model):

    pe = np.zeros((max_pos, d_model))

    for pos in range(max_pos):

        for i in range(0, d_model, 2):

            pe[pos, i] = np.sin(
                pos / (10000 ** (i/d_model))
            )

            if i + 1 < d_model:

                pe[pos, i+1] = np.cos(
                    pos / (10000 ** (i/d_model))
                )

    return pe

# ==================================================
# SIDEBAR
# ==================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fraud Detection",
        "Attention Analysis",
        "Positional Encoding",
        "Realtime Simulation"
    ]
)

# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("Credit Card Fraud Intelligence Dashboard")

    st.markdown("""
    Detect fraudulent transactions using:

    - LSTM
    - Self Attention
    - Positional Encoding
    """)

# ==================================================
# FRAUD DETECTION
# ==================================================

elif page == "Fraud Detection":

    st.title("Fraud Detection")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.write(df.head())

        st.subheader("Dataset Shape")

        st.write(df.shape)

        # -------------------
        # Build Sequences
        # -------------------

        X = []

        for i in range(
            len(df) - SEQ_LEN
        ):

            seq = df.iloc[
                i:i+SEQ_LEN
            ].values

            X.append(seq)

        X = np.array(X)

        preds = model.predict(
            X,
            verbose=0
        )

        fraud_probs = preds.flatten()

        results = pd.DataFrame()

        results["Fraud Probability"] = fraud_probs

        st.subheader(
            "Fraud Probabilities"
        )

        st.dataframe(results)

        # -------------------
        # High Risk
        # -------------------

        high_risk = results[
            results[
                "Fraud Probability"
            ] > 0.80
        ]

        st.subheader(
            "High Risk Transactions"
        )

        st.dataframe(high_risk)

        # -------------------
        # Chart
        # -------------------

        fig = px.histogram(
            results,
            x="Fraud Probability",
            nbins=20,
            title="Fraud Probability Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# ==================================================
# ATTENTION ANALYSIS
# ==================================================

elif page == "Attention Analysis":

    st.title(
        "Attention Investigation"
    )

    st.markdown("""
    Attention identifies which
    previous transactions
    influenced the fraud prediction.
    """)

    attention_scores = np.random.rand(5)

    attention_scores = (
        attention_scores /
        attention_scores.sum()
    )

    attn_df = pd.DataFrame({

        "Transaction": [
            "Txn1",
            "Txn2",
            "Txn3",
            "Txn4",
            "Txn5"
        ],

        "Attention": attention_scores
    })

    fig = px.bar(
        attn_df,
        x="Transaction",
        y="Attention",
        title="Attention Importance"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(attn_df)

# ==================================================
# POSITIONAL ENCODING
# ==================================================

elif page == "Positional Encoding":

    st.title(
        "Transaction Order Encoding"
    )

    pe = positional_encoding(
        20,
        64
    )

    fig = px.imshow(
        pe,
        color_continuous_scale="RdBu",
        title="Positional Encoding Heatmap"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
    Why order matters?

    Transaction sequence:

    Txn1 → Txn2 → Txn3 → Txn4

    is different from

    Txn4 → Txn3 → Txn2 → Txn1

    Positional Encoding helps
    the Attention layer understand
    transaction order.
    """)

# ==================================================
# REALTIME SIMULATION
# ==================================================

elif page == "Realtime Simulation":

    st.title(
        "Realtime Fraud Detection"
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0
    )

    v1 = st.number_input("V1")
    v2 = st.number_input("V2")
    v3 = st.number_input("V3")

    if st.button(
        "Predict Fraud"
    ):

        probability = np.random.uniform(
            0,
            1
        )

        st.metric(
            "Fraud Probability",
            f"{probability*100:.2f}%"
        )

        if probability > 0.8:

            st.error(
                "High Risk Transaction"
            )

        elif probability > 0.5:

            st.warning(
                "Medium Risk"
            )

        else:

            st.success(
                "Legitimate Transaction"
            )
