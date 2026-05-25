"""utils/preprocessor.py — Shared preprocessing helpers."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def preprocess_loan(train_df: pd.DataFrame):
    df = train_df.copy()
    df.drop(columns=["Loan_ID"], errors="ignore", inplace=True)

    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != "Loan_Status"]
    num_cols = [c for c in df.select_dtypes(include=["int64", "float64"]).columns if c != "Loan_Status"]

    for c in cat_cols:
        df[c].fillna(df[c].mode()[0], inplace=True)
    for c in num_cols:
        df[c].fillna(df[c].median(), inplace=True)

    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c])
        encoders[c] = le

    if "Loan_Status" in df.columns:
        df["Loan_Status"] = (df["Loan_Status"].str.upper() == "Y").astype(int)

    return df, encoders


def preprocess_marketing(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df.drop(columns=["ID", "Dt_Customer", "Z_CostContact", "Z_Revenue"], errors="ignore", inplace=True)
    df["Income"].fillna(df["Income"].median(), inplace=True)

    for c in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    return df


def prepare_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(class_weights))

    return X_train, X_val, y_train, y_val, scaler, class_weight_dict
