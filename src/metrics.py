import numpy as np
from sklearn.metrics import confusion_matrix

def business_cost_from_preds(y_true, y_pred) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return 10.0 * fn + 1.0 * fp

def optimal_threshold_cost(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 181)  # pas fin mais stable
    best = {"threshold": 0.5, "cost": float("inf")}
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        cost = business_cost_from_preds(y_true, y_pred)
        if cost < best["cost"]:
            best = {"threshold": float(t), "cost": float(cost)}
    return best

def business_score_neg_min_cost(estimator, X, y):
    # Score pour GridSearchCV: on maximise, donc on retourne le négatif du coût minimum
    y_proba = get_proba(estimator, X)
    best = optimal_threshold_cost(y, y_proba)
    return -best["cost"]

def business_threshold_scorer(estimator, X, y):
    # Permet de récupérer un seuil "moyen" via cv_results_
    y_proba = get_proba(estimator, X)
    best = optimal_threshold_cost(y, y_proba)
    return best["threshold"]

def get_proba(estimator, X):
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    # if hasattr(estimator, "decision_function"):
    #     scores = estimator.decision_function(X)
    #     # transformation en [0,1] (sigmoïde)
    #     return 1 / (1 + np.exp(-scores))
    raise AttributeError("Le modèle doit exposer predict_proba.") # ou decision_function.")
