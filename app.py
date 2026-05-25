"""
🧠 Deep Learning Pipeline — Streamlit App
Supports: Loan Sanction | Marketing Campaign
"""

import streamlit as st

st.set_page_config(
    page_title="DL Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.title("🧠 DL Pipeline")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Navigation**
    - 🏠 Home (this page)
    - 📂 Upload & Preprocess
    - 🔍 EDA
    - 🏗️ Train Models
    - 📊 Results & Compare
    - 🔮 Live Predictor
    - ⚡ API Tester
    """
)
st.sidebar.markdown("---")
st.sidebar.info("Upload your datasets on the **Upload & Preprocess** page to get started.")

# ── Home ─────────────────────────────────────────────────────────────────
st.title("🧠 Deep Learning Pipeline for Tabular Data")
st.markdown("### ANN · 1D CNN · Activation Comparisons · Optimizer Comparisons")

col1, col2, col3 = st.columns(3)
col1.metric("Supported Datasets", "2", "Loan · Marketing")
col2.metric("Model Architectures", "2", "ANN · 1D CNN")
col3.metric("Experiments", "8", "4 Activations + 4 Optimizers")

st.markdown("---")

st.markdown(
    """
    ## What this app does

    | Feature | Detail |
    |---|---|
    | **ANN** | Batch Normalization + Dropout + L2 Regularization |
    | **1D CNN** | Treats tabular features as sequential channels |
    | **Activations** | ReLU · LeakyReLU · ELU · Tanh |
    | **Optimizers** | Adam · SGD · RMSprop · Adagrad |
    | **Callbacks** | EarlyStopping · ReduceLROnPlateau · ModelCheckpoint |
    | **Deployment** | Live inference + REST API tester |

    ## How to use
    1. Go to **📂 Upload & Preprocess** → upload your CSV files
    2. Visit **🔍 EDA** to explore the data
    3. Go to **🏗️ Train Models** and click *Train*
    4. See results on **📊 Results & Compare**
    5. Make predictions on **🔮 Live Predictor**
    """
)
