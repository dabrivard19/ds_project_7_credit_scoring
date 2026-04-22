from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Paths:
    project_root: Path = Path(__file__).resolve().parents[1]
    #print("dana 1: ", list((Path(__file__).resolve().parents)))
    data_raw: Path = project_root / "data" / "raw"
    data_processed: Path = project_root / "data" / "processed"
    artifacts: Path = project_root / "artifacts"
    models: Path = project_root / "models"

@dataclass(frozen=True)
class MlflowConfig:
    # FileStore local (pas de base de données)
    tracking_uri: str = "file:./mlruns"
    # Artifacts au même endroit (par défaut) ou dossier dédié
    artifact_location: str = "./mlruns"
    experiment_name: str = "home_credit"

PATHS = Paths()
MLFLOW = MlflowConfig()
