
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


# def run_prediction(model, user_features: dict) -> tuple[int, float | None, list[str]]:
#     print("\nExécution de la prédiction...")
#     X = prepare_dataframe(user_features)
#     print(f"\nDataFrame pour la prédiction :\n{X}")
#     try:
#         prediction = int(model.predict(X)[0])
#         print(f"\nPrédiction brute du modèle : {prediction}")
#     except Exception as exc:
#         print(f"\nErreur lors de la prédiction : {exc}")
#         raise f"Erreur lors de la prédiction : {exc}" from exc      

#     probability = None
#     if hasattr(model, "predict_proba"):
#         print(f"\nCalcul de la probabilité: ", model.predict_proba(X))
#         probability = float(model.predict_proba(X)[0][1])

#     print(f"Prédiction : {prediction}, Probabilité : {probability}")
#     return prediction, probability, list(X.columns)

import numpy as np
import pandas as pd
import shap


def get_final_estimator(model):
    """
    Récupère le vrai modèle final si model est un Pipeline.
    Sinon retourne model directement.
    """
    if hasattr(model, "steps"):
        return model.steps[-1][1]
    return model


def get_global_importance(model, feature_names: list[str]) -> list[dict]:
    """
    Importance globale pour XGBClassifier.
    """
    try:
        final_model = get_final_estimator(model)

        if not hasattr(final_model, "feature_importances_"):
            print("Le modèle n'a pas feature_importances_", flush=True)
            return []

        importances = final_model.feature_importances_

        global_importance = [
            {
                "feature": feature,
                "importance": float(importance),
            }
            for feature, importance in zip(feature_names, importances)
        ]

        return sorted(
            global_importance,
            key=lambda x: x["importance"],
            reverse=True
        )

    except Exception as exc:
        print(f"Erreur importance globale : {exc}", flush=True)
        return []


def get_local_importance(model, X: pd.DataFrame) -> list[dict]:
    """
    Importance locale pour un XGBClassifier avec SHAP.
    """
    try:
        final_model = get_final_estimator(model)

        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)

        # Cas ancien format SHAP : liste [classe_0, classe_1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Cas nouveau format éventuel : (n_samples, n_features, n_classes)
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        client_values = shap_values[0]

        local_importance = [
            {
                "feature": feature,
                "contribution": float(value),
                "abs_contribution": float(abs(value)),
                "effect": "augmente le risque" if value > 0 else "diminue le risque",
            }
            for feature, value in zip(X.columns, client_values)
        ]

        return sorted(
            local_importance,
            key=lambda x: x["abs_contribution"],
            reverse=True
        )

    except Exception as exc:
        print(f"Erreur SHAP XGBClassifier : {exc}", flush=True)
        return []


def run_prediction(model, user_features: dict) -> dict:
    print("\nExécution de la prédiction...", flush=True)

    X = prepare_dataframe(user_features)

    print(f"\nDataFrame pour la prédiction :\n{X}", flush=True)

    feature_names = list(X.columns)

    try:
        proba = model.predict_proba(X)
        probability = float(proba[0][1])

        threshold = 0.5
        prediction = int(probability >= threshold)

        print(f"\nProbabilités : {proba}", flush=True)
        print(f"Prédiction avec seuil {threshold} : {prediction}", flush=True)

    except Exception as exc:
        print(f"\nErreur lors de la prédiction : {exc}", flush=True)
        raise RuntimeError(f"Erreur lors de la prédiction : {exc}") from exc

    global_importance = get_global_importance(model, feature_names)
    local_importance = get_local_importance(model, X)

    print(f"Importance globale : {len(global_importance)} variables", flush=True)
    print(f"Importance locale : {len(local_importance)} variables", flush=True)

    # return all infos in a dict for better extensibility
    return {
        "prediction": prediction,
        "probability": probability,
        "used_features": feature_names,
        "local_importance": local_importance,
        "global_importance": global_importance,
    }
