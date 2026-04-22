from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .metrics import business_score_neg_min_cost, business_threshold_scorer

def build_preprocessor(X):
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor

# def make_scoring():
#     scoring = {
#         "business": make_scorer(business_score_neg_min_cost, needs_proba=False), # callable(estimator,X,y)
#         "threshold": make_scorer(business_threshold_scorer, needs_proba=False),
#         "AUC": "roc_auc",
#         "f1": "f1",
#     }
#     return scoring

def make_scoring():
    scoring = {
        "business": business_score_neg_min_cost,   # <-- direct, pas make_scorer
        "threshold": business_threshold_scorer,    # <-- direct aussi si même signature
        "AUC": "roc_auc",
        "f1": "f1",
    }
    return scoring

def make_cv(n_splits=5, random_state=42):
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

def gridsearch_dummy(X, y, cv):
    pre = build_preprocessor(X)
    pipe = Pipeline(steps=[
        ("pre", pre),
        ("model", DummyClassifier(strategy="most_frequent")),
    ])
    return pipe, {}

def gridsearch_logreg_smote(X, y, cv):
    pre = build_preprocessor(X)
    pipe = ImbPipeline(steps=[
        ("pre", pre),
        #("smote", SMOTE()),
        ("model", LogisticRegression(max_iter=2000, n_jobs=None)),
    ])
    param_grid = {
        "model__C": [0.5, 1], #[0.05, 0.1, 0.5, 1.0, 3.0],
        #"model__class_weight": [None, "balanced"],
        "model__solver": ["lbfgs"],
    }
    return pipe, param_grid

def gridsearch_rf_smote(X, y, cv):
    pre = build_preprocessor(X)
    pipe = ImbPipeline(steps=[
        ("pre", pre),
        #("smote", SMOTE()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=-1)),
    ])
    # param_grid = {
    #     "model__n_estimators": [500], #[200, 500],
    #     "model__max_depth": [8], #[None, 8, 16],
    #     "model__min_samples_split": [10], #[2, 10],
    #     "model__min_samples_leaf": [5]  #[1, 5],
    #     #"model__class_weight": [None, "balanced_subsample"],
    # }

    # param_grid = {
    #     "model__n_estimators": [300],          # fixe
    #     "model__max_depth": [8, 16],
    #     "model__min_samples_leaf": [5, 20],
    #     "model__max_features": ["sqrt", 0.3],  # important après OHE
    # }

    param_grid = {
    "model__n_estimators": [200],
    "model__max_depth": [8],
    "model__min_samples_leaf": [20],
    "model__max_features": ["sqrt"],
    }
    return pipe, param_grid

def run_gridsearch(pipe, param_grid, X, y, cv, refit_metric="business"):
    scoring = make_scoring()
    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=2,
        verbose=3,
        return_train_score=False,
        error_score="raise",
        pre_dispatch=2,  # pour limiter la mémoire utilisée
    )
    gs.fit(X, y)
    return gs

from xgboost import XGBClassifier

def gridsearch_xgb_smote(X, y, cv):
    pre = build_preprocessor(X)

    # (optionnel) utile en déséquilibre: n_neg / n_pos
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    pipe = ImbPipeline(steps=[
        ("pre", pre),
        # ("smote", SMOTE(random_state=42)),  # décommente si tu veux vraiment SMOTE ici
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_jobs=-1,
            random_state=42,
            tree_method="hist",          # CPU rapide (gpu_hist si GPU)
            scale_pos_weight=scale_pos_weight,
        )),
    ])

    # param_grid = {
    #     "model__n_estimators": [400, 800],
    #     "model__max_depth": [3, 5, 7],
    #     "model__learning_rate": [0.03, 0.1],
    #     "model__subsample": [0.8, 1.0],
    #     "model__colsample_bytree": [0.8, 1.0],
    #     "model__min_child_weight": [1, 5],
    #     "model__reg_alpha": [0.0, 0.1],
    #     "model__reg_lambda": [1.0, 2.0],
    # }
    param_grid = {
        "model__n_estimators": [500],
        "model__learning_rate": [0.05],
        "model__max_depth": [4, 6],
        "model__min_child_weight": [5],
        "model__subsample": [0.8],
        "model__colsample_bytree": [0.8],
    }
    return pipe, param_grid

# LightGBM
# Dépendance : pip install lightgbm

from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

class ToDense(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.toarray() if sparse.issparse(X) else X

def gridsearch_lgbm_smote(X, y, cv):
    pre = build_preprocessor(X)

    pipe = ImbPipeline(steps=[
        ("pre", pre),
        ("dense", ToDense()),   # ✅ rend LGBM stable avec OHE
        ("model", LGBMClassifier(
            objective="binary",
            random_state=42,
            n_jobs=2,
            verbose=2
            # petit réglage utile en binaire déséquilibré (optionnel)
            # class_weight="balanced",
        )),
    ])

    # param_grid = {
    #     "model__n_estimators": [600],
    #     "model__learning_rate": [0.03, 0.1],
    #     "model__num_leaves": [31, 63],
    #     "model__max_depth": [-1, 10],
    #     "model__min_child_samples": [30, 60],
    #     "model__subsample": [0.8, 1.0],
    #     "model__colsample_bytree": [0.8, 1.0],
    # }
    param_grid = {
    "model__n_estimators": [400],
    "model__learning_rate": [0.05],
    "model__num_leaves": [31, 63],
    "model__max_depth": [8, 10],
    "model__min_child_samples": [60, 120],
    "model__subsample": [0.8],
    "model__colsample_bytree": [0.8],
    }
    return pipe, param_grid

# def gridsearch_lgbm_smote(X, y, cv):
#     pre = build_preprocessor(X)

#     pipe = ImbPipeline(steps=[
#         ("pre", pre),
#         # ("smote", SMOTE(random_state=42)),  # décommente si tu veux vraiment SMOTE ici
#         ("model", LGBMClassifier(
#             objective="binary",
#             n_jobs=-1,
#             random_state=42,
#         )),
#     ])

#     # param_grid = {
#     #     "model__n_estimators": [400, 800],
#     #     "model__learning_rate": [0.03, 0.1],
#     #     "model__num_leaves": [31, 63, 127],
#     #     "model__max_depth": [-1, 5, 10],
#     #     "model__min_child_samples": [10, 30, 60],
#     #     "model__subsample": [0.8, 1.0],
#     #     "model__colsample_bytree": [0.8, 1.0],
#     #     "model__reg_alpha": [0.0, 0.1],
#     #     "model__reg_lambda": [0.0, 1.0],
#     # }

#     param_grid = {
#     "model__n_estimators": [600],                 # on fixe
#     "model__learning_rate": [0.03, 0.1],
#     "model__num_leaves": [31, 63],
#     "model__max_depth": [-1, 10],                 # -1 = pas de limite
#     "model__min_child_samples": [30, 60],
#     "model__subsample": [0.8, 1.0],
#     "model__colsample_bytree": [0.8, 1.0],
#     }

#     return pipe, param_grid

# Remarque importante (SMOTE + arbres boostés)
# Pour XGBoost / LightGBM, SMOTE n’est pas toujours recommandé (ça peut dégrader, selon le bruit / la frontière de décision).
# Souvent on préfère :

# XGBoost : scale_pos_weight

# LightGBM : class_weight="balanced" ou is_unbalance=True

# pipe, param_grid = gridsearch_xgb_smote(X_train, y_train, cv)
# gs = run_gridsearch(pipe, param_grid, X_train, y_train, cv)
# et ton bloc track_run(...) reste inchangé.

