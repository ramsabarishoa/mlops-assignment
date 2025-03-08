import pytest
from fastapi.testclient import TestClient
from main import app
import requests
import os

client = TestClient(app)

def test_predict_endpoint():
    """Tests the /predict endpoint with sample data."""
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)

def test_health_endpoint():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.integration
def test_docker_integration():
    """Integration test: checks the API endpoint in a running container."""
    # Assumes the app is running at localhost:8000 inside the container
    # Use an environment variable to configure the URL during CI if needed
    api_url = os.getenv("API_URL", "http://localhost:8000")
    predict_url = f"{api_url}/predict"

    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    try:
        response = requests.post(predict_url, json=sample_data, timeout=10)  # Add timeout
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        assert response.status_code == 200
        assert "prediction" in response.json()
        assert isinstance(response.json()["prediction"], int)

    except requests.exceptions.RequestException as e:
        pytest.fail(f"Integration test failed: {e}")
