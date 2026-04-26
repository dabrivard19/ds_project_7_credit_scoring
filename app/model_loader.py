
from functools import lru_cache
from pathlib import Path
import joblib

MODEL_PATH = Path("model/model.joblib")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        print(f"\nModèle introuvable : {MODEL_PATH}. Ajoute ton modele.joblib dans le dossier model/.")
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Ajoute ton modele.joblib dans le dossier model/."
        )
    print(f"\nOK: Chargement du modèle depuis {MODEL_PATH}...")
    try:
        model = joblib.load(MODEL_PATH)
        print(f"\nModèle chargé avec succès.")
        return model
    except Exception as exc:
        print(f"\nErreur lors du chargement du modèle : {exc}")
        raise RuntimeError(f"Erreur lors du chargement du modèle : {exc}") from exc
    # return joblib.load(MODEL_PATH)
