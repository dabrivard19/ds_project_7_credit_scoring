import os
import json
import time
import datetime
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from .config import MLFLOW

def init_mlflow():
    """Initialise MLflow en mode local FileStore (fichiers)."""
    mlflow.set_tracking_uri(MLFLOW.tracking_uri)
    mlflow.set_experiment(MLFLOW.experiment_name)

def log_json(data: dict, artifact_name: str):
    with open(artifact_name, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    mlflow.log_artifact(artifact_name)
    os.remove(artifact_name)

def log_dataframe_csv(df, artifact_name: str):
    df.to_csv(artifact_name, index=False)
    mlflow.log_artifact(artifact_name)
    os.remove(artifact_name)

def log_roc_curve(y_true, y_proba, artifact_name="roc_curve.png"):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(artifact_name)
    plt.close()
    mlflow.log_artifact(artifact_name)
    os.remove(artifact_name)

def log_confusion_matrix_files(y_true, y_pred, prefix="confusion_matrix"):
    """Log confusion matrix as both .npy and .png (artefact lisible)."""
    cm = confusion_matrix(y_true, y_pred)

    # NPY
    npy_name = f"{prefix}.npy"
    np.save(npy_name, cm)
    mlflow.log_artifact(npy_name)
    os.remove(npy_name)

    # PNG (simple heatmap)
    png_name = f"{prefix}.png"
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(png_name)
    plt.close()
    mlflow.log_artifact(png_name)
    os.remove(png_name)

def log_run_report_md(
    run_name: str,
    params: dict,
    metrics: dict,
    threshold_info: dict,
    artifacts: list,
    filename: str = "report.md",
):
    """Crée un rapport Markdown récapitulatif et le log comme artefact.
    Note : la navigation se fait via l'UI MLflow ; ici on liste les chemins/artefacts.
    """
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = []
    lines.append(f"# Rapport MLflow — {run_name}\n")
    lines.append(f"- Généré le : **{now}**\n")
    lines.append("## Paramètres\n")
    if params:
        for k in sorted(params.keys()):
            lines.append(f"- `{k}` = `{params[k]}`")
    else:
        lines.append("- *(aucun)*")
    lines.append("\n## Métriques\n")
    if metrics:
        for k in sorted(metrics.keys()):
            lines.append(f"- **{k}** : `{metrics[k]}`")
    else:
        lines.append("- *(aucune)*")

    lines.append("\n## Seuil & coût métier\n")
    if threshold_info:
        for k in sorted(threshold_info.keys()):
            lines.append(f"- **{k}** : `{threshold_info[k]}`")
    else:
        lines.append("- *(non fourni)*")

    lines.append("\n## Artefacts enregistrés\n")
    if artifacts:
        for a in artifacts:
            lines.append(f"- `{a}`")
    else:
        lines.append("- *(aucun)*")

    content = "\n".join(lines) + "\n"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    mlflow.log_artifact(filename)
    os.remove(filename)

def track_run(
    run_name: str,
    model,
    X_train, y_train,
    X_valid, y_valid,
    params: dict,
    extra_metrics: dict,
    y_valid_proba: np.ndarray,
    y_valid_pred: np.ndarray,
    threshold_info: dict,
    fit_time: float,
    predict_time: float,
    cv_results_df=None,
):
    """Tracking standardisé :
    - params
    - metrics (AUC + score métier etc.)
    - temps fit/predict
    - artefacts: ROC + confusion matrix + json seuil + (option) cv_results.csv + report.md
    - modèle (pipeline complet)
    """
    with mlflow.start_run(run_name=run_name):
        # Params
        mlflow.log_params(params)

        # Metrics (scalars)
        metrics = dict(extra_metrics or {})
        metrics["fit_time"] = float(fit_time)
        metrics["predict_time"] = float(predict_time)
        metrics["threshold_opt"] = float(threshold_info.get("threshold", 0.5)) if threshold_info else 0.5
        metrics["business_cost_opt"] = float(threshold_info.get("cost", np.nan)) if threshold_info else float("nan")
        metrics["AUC"] = float(roc_auc_score(y_valid, y_valid_proba))

        for k, v in metrics.items():
            mlflow.log_metric(k, float(v) if v is not None else float("nan"))

        # Artefacts
        artifacts = []
        log_roc_curve(y_valid, y_valid_proba)
        artifacts.append("roc_curve.png")

        log_confusion_matrix_files(y_valid, y_valid_pred)
        artifacts.extend(["confusion_matrix.png", "confusion_matrix.npy"])

        if threshold_info is not None:
            log_json(threshold_info, "threshold_info.json")
            artifacts.append("threshold_info.json")

        if cv_results_df is not None:
            log_dataframe_csv(cv_results_df, "cv_results.csv")
            artifacts.append("cv_results.csv")

        # Rapport markdown récapitulatif (utile soutenance)
        log_run_report_md(
            run_name=run_name,
            params=params or {},
            metrics=metrics,
            threshold_info=threshold_info or {},
            artifacts=artifacts + ["model/"],
            filename="report.md",
        )
        artifacts.append("report.md")

        # Model (pipeline complet)
        #model_type = model.__class__.__module__ + "." + model.__class__.__name__
        final_estimator = model.steps[-1][1]
        model_name = final_estimator.__class__.__name__

        from mlflow.models import infer_signature
        signature = infer_signature(X_train, y_valid_pred)
        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=model_name,
                                signature=signature,          # <- input & output schema
                                input_example=X_train.head(5) # <- example payload (also helps auto-infer)
                                )
