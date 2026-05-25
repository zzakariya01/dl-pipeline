# 🧠 Deep Learning Pipeline — Streamlit App

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

A full deep learning pipeline for tabular classification, deployed as a multi-page Streamlit app.

## Features

| Feature | Detail |
|---|---|
| **ANN** | Batch Normalization + Dropout + L2 Regularization |
| **1D CNN** | Treats tabular features as 1D sequences |
| **Activations** | ReLU · LeakyReLU · ELU · Tanh |
| **Optimizers** | Adam · SGD · RMSprop · Adagrad |
| **Callbacks** | EarlyStopping · ReduceLROnPlateau · ModelCheckpoint |
| **Datasets** | Loan Sanction · Marketing Campaign |

## App Pages

1. **📂 Upload & Preprocess** — Upload CSV files and preprocess
2. **🔍 EDA** — Exploratory data analysis with visualizations
3. **🏗️ Train Models** — Configure and train ANN / 1D CNN experiments
4. **📊 Results & Compare** — Training curves, ROC curves, leaderboard
5. **🔮 Live Predictor** — Interactive single-record prediction
6. **⚡ API Tester** — Test deployed FastAPI / Flask endpoints

## Project Structure

```
├── app.py                          # Main entry point (Home page)
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # Theme & server config
├── pages/
│   ├── 1_📂_Upload_Preprocess.py
│   ├── 2_🔍_EDA.py
│   ├── 3_🏗️_Train_Models.py
│   ├── 4_📊_Results_Compare.py
│   ├── 5_🔮_Live_Predictor.py
│   └── 6_⚡_API_Tester.py
└── utils/
    ├── __init__.py
    ├── preprocessor.py             # Dataset preprocessing
    ├── model_builder.py            # ANN & 1D CNN builders
    └── plots.py                    # Reusable Matplotlib/Seaborn plots
```

## Deploy on Streamlit Community Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done!

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
streamlit run app.py
```

## Datasets

Upload these CSV files inside the app:
- `loan_sanction_train.csv` — Loan approval (binary classification)
- `loan_sanction_test.csv` — Loan test set (optional)
- `marketing_campaign.csv` — Campaign response (binary classification)

## REST API

After training in the Colab notebook, run the FastAPI server:

```bash
pip install fastapi uvicorn
python fastapi_app.py
# Swagger UI → http://localhost:8000/docs
```
