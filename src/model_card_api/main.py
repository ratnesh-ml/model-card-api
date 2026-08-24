from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import MODEL_CARD, FEATURE_NAMES, explain, predict

app = FastAPI(title="Model Card API", version=MODEL_CARD["version"])


class PredictionRequest(BaseModel):
    study_hours: float = Field(ge=0, le=24)
    attendance_rate: float = Field(ge=0, le=1)
    practice_sessions: float = Field(ge=0, le=30)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_CARD["name"], "version": MODEL_CARD["version"]}


@app.get("/model-card")
def model_card() -> dict[str, object]:
    return MODEL_CARD


@app.post("/predict")
def prediction(request: PredictionRequest) -> dict[str, object]:
    return predict(request.model_dump())


@app.post("/explain")
def explanation(request: PredictionRequest) -> dict[str, object]:
    return explain(request.model_dump())
