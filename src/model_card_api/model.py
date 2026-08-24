from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = ["study_hours", "attendance_rate", "practice_sessions"]
MODEL_CARD = {
    "name": "study-outcome-baseline",
    "version": "0.1.0",
    "task": "Educational demonstration of binary classification",
    "features": FEATURE_NAMES,
    "intended_use": "Learning how to expose a model with a documented API contract",
    "not_intended_for": "Real academic decisions, grading, admissions, or student intervention",
    "decision_threshold": 0.5,
    "confidence_bands": {"low": "0.40-0.60", "moderate": "0.20-0.40 or 0.60-0.80", "high": "below 0.20 or above 0.80"},
    "limitations": ["Synthetic training data", "Small feature set", "No causal interpretation"],
}


def build_model() -> Pipeline:
    X = np.array([
        [2, .55, 1], [3, .62, 2], [4, .70, 3], [5, .76, 4], [6, .85, 5],
        [1, .45, 0], [2, .50, 1], [3, .58, 1], [4, .66, 2], [5, .72, 3],
        [6, .90, 6], [7, .92, 6], [1, .35, 0], [2, .42, 0], [3, .48, 1],
    ], dtype=float)
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0])
    model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(random_state=42))])
    model.fit(X, y)
    return model


MODEL = build_model()


def confidence_band(probability: float) -> str:
    """Describe how close a probability is to the decision boundary."""
    if .40 <= probability <= .60:
        return "low"
    if .20 <= probability < .40 or .60 < probability <= .80:
        return "moderate"
    return "high"


def predict(features: dict[str, float]) -> dict[str, object]:
    values = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    probability = float(MODEL.predict_proba(values)[0, 1])
    return {
        "label": "ready" if probability >= .5 else "needs_support",
        "probability": round(probability, 4),
        "confidence_band": confidence_band(probability),
        "features": features,
    }


def explain(features: dict[str, float]) -> dict[str, object]:
    values = np.array([[features[name] for name in FEATURE_NAMES]], dtype=float)
    classifier = MODEL.named_steps["classifier"]
    scaled = MODEL.named_steps["scale"].transform(values)[0]
    contributions = scaled * classifier.coef_[0]
    probability = float(MODEL.predict_proba(values)[0, 1])
    return {
        "base_signal": float(classifier.intercept_[0]),
        "decision_threshold": .5,
        "confidence_band": confidence_band(probability),
        "feature_contributions": {name: round(float(value), 4) for name, value in zip(FEATURE_NAMES, contributions)},
    }
