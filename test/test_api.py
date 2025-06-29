from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_predict():
    payload = {
        "Recency": 10.0,
        "Frequency": 5,
        "Monetary": 1000.0,
        "Transaction_Hour": 14
        # add other features here
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_probability" in data
    assert 0.0 <= data["risk_probability"] <= 1.0
