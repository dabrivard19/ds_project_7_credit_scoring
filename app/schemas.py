
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(default_factory=dict)


class PredictionResponse(BaseModel):
    prediction: int
    probability: float | None = None
    used_features: List[str]
