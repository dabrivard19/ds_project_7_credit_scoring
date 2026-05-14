
from functools import lru_cache
from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model/best_model_xgBoost.joblib"   

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

MODEL_PATH = PROJECT_DIR / "model" / "best_model_xgBoost.joblib"


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
