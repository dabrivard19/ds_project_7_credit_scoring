
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

DEFAULT_PAYLOAD_PATH = Path("app/default_payload.json")

import numpy as np

def add_features(df):
    df = df.copy()

    df["RATIO_CREDIT_INCOME"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].replace(0, np.nan)
    df["RATIO_ANNUITY_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].replace(0, np.nan)
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["RATIO_EMPLOYED_BIRTH"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"].replace(0, np.nan)

    return df

def load_default_payload() -> dict:
    print(f"\nChargement du payload par défaut depuis {DEFAULT_PAYLOAD_PATH}...")
    with open(DEFAULT_PAYLOAD_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_features(user_features: dict) -> dict:
    print(f"\nNormalisation des features...")
    full_payload = load_default_payload()
    print(f"\nPayload par défaut : {full_payload}")
    merged = {**full_payload, **user_features}
    print(f"\nPayload après fusion : {merged}")

    if "AGE_YEARS" in merged:
        print(f"\nConversion de AGE_YEARS en DAYS_BIRTH...")
        age_years = int(merged.pop("AGE_YEARS"))
        merged["DAYS_BIRTH"] = -age_years * 365

    merged = add_features(pd.DataFrame([merged])).iloc[0].to_dict()
    print(f"\nPayload final normalisé : {merged}")
    return merged


def prepare_dataframe(user_features: dict) -> pd.DataFrame:
    print("\nPréparation du DataFrame...")
    full_payload = normalize_features(user_features)
    print(f"\nDataFrame préparé : {full_payload}")
    return pd.DataFrame([full_payload])


def run_prediction(model, user_features: dict) -> tuple[int, float | None, list[str]]:
    print("\nExécution de la prédiction...")
    X = prepare_dataframe(user_features)
    print(f"\nDataFrame pour la prédiction :\n{X}")
    try:
        prediction = int(model.predict(X)[0])
        print(f"\nPrédiction brute du modèle : {prediction}")
    except Exception as exc:
        print(f"\nErreur lors de la prédiction : {exc}")
        raise f"Erreur lors de la prédiction : {exc}" from exc      

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(X)[0][1])

    print(f"Prédiction : {prediction}, Probabilité : {probability}")
    return prediction, probability, list(X.columns)
