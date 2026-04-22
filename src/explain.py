import shap
import numpy as np
import pandas as pd

def shap_explain_tree(model, X_sample: pd.DataFrame):
    # Pour les modèles type RF/GBDT, shap.TreeExplainer marche bien
    explainer = shap.Explainer(model)
    shap_values = explainer(X_sample)
    return explainer, shap_values
