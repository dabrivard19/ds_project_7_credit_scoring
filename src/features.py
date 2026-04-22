import numpy as np
import pandas as pd

def basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # Remplace inf par NaN, puis laisse la stratégie d'imputation au pipeline (ou fillna ici si vous préférez)
    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    # Exemples simples, à compléter selon votre kernel
    df = df.copy()
    if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(df.columns):
        df["RATIO_CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        df["RATIO_ANNUITY_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    if {"DAYS_EMPLOYED", "DAYS_BIRTH"}.issubset(df.columns):
        df["RATIO_EMPLOYED_BIRTH"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    return df

def split_X_y(df: pd.DataFrame, target_col: str = "TARGET"):
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    return X, y
