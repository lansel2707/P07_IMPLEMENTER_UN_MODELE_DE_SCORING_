import pytest
import requests
import os

BASE_URL = "http://localhost:8000"  # adapte si nécessaire

@pytest.mark.skipif(os.getenv("SKIP_MODEL_LOAD_IN_TESTS") == "1", reason="Skip heavy model load")
def test_healthcheck():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200

@pytest.mark.skipif(os.getenv("SKIP_MODEL_LOAD_IN_TESTS") == "1", reason="Skip heavy model load")
def test_prediction_endpoint():
    sample = {"feature1": 1, "feature2": 0}  # à adapter à ton modèle
    r = requests.post(f"{BASE_URL}/predict", json=sample)
    assert r.status_code == 200
    assert "prediction" in r.json()

