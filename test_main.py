from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)

def test_health_endpoint():
    """Tests the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio  # Mark the test function as asynchronous
async def test_predict_endpoint():
    """Tests the /predict endpoint with sample data."""
    sample_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = await client.post("/predict", json=sample_data)  # Await the coroutine
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_metrics_endpoint():
    """Tests the /metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]