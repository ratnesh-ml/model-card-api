from fastapi.testclient import TestClient

from model_card_api.main import app

client = TestClient(app)


def test_health_and_model_card_are_machine_readable():
    health = client.get("/health")
    card = client.get("/model-card")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert card.json()["version"] == "0.1.0"


def test_prediction_and_explanation_have_stable_contracts():
    payload = {"study_hours": 5, "attendance_rate": .8, "practice_sessions": 4}
    prediction = client.post("/predict", json=payload)
    explanation = client.post("/explain", json=payload)
    assert prediction.status_code == 200
    assert prediction.json()["label"] in {"ready", "needs_support"}
    assert prediction.json()["confidence_band"] in {"low", "moderate", "high"}
    assert set(explanation.json()["feature_contributions"]) == {"study_hours", "attendance_rate", "practice_sessions"}
    assert explanation.json()["decision_threshold"] == .5


def test_input_validation_rejects_impossible_values():
    response = client.post("/predict", json={"study_hours": 30, "attendance_rate": .8, "practice_sessions": 4})
    assert response.status_code == 422
