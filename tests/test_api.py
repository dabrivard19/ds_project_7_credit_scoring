
from fastapi.testclient import TestClient
import app.main as main_module
from app.main import app


class FakeModel:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.15, 0.85]]


def fake_load_model():
    return FakeModel()


def test_root():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "API OK"


def test_form_config():
    client = TestClient(app)
    response = client.get("/form-config")
    assert response.status_code == 200
    body = response.json()
    assert "features" in body
    assert len(body["features"]) == 10


def test_predict(monkeypatch):
    monkeypatch.setattr(main_module, "load_model", fake_load_model)

    client = TestClient(app)
    payload = {
        "features": {
            "CODE_GENDER": "F",
            "FLAG_OWN_CAR": "N",
            "FLAG_OWN_REALTY": "Y",
            "CNT_CHILDREN": 1,
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 500000.0,
            "AMT_ANNUITY": 25000.0,
            "NAME_INCOME_TYPE": "Working",
            "NAME_EDUCATION_TYPE": "Higher education",
            "AGE_YEARS": 35,
        }
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] == 1
    assert body["probability"] == 0.85
    assert "AMT_CREDIT" in body["used_features"]
    assert "DAYS_BIRTH" in body["used_features"]
