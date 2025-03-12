from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
import json
from prometheus_client import CollectorRegistry, Gauge, generate_latest, Counter
from prometheus_client import Histogram, Summary
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Load the model
try:
    model = joblib.load("iris_model.joblib")
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise  # Exit if model loading fails, preventing the app from running

# Prometheus metrics setup
registry = CollectorRegistry()

# Request metrics - using prometheus.Counter 
total_requests_counter = Counter("iris_total_requests", "Total number of requests to the API", registry=registry)
error_count_counter = Counter("iris_error_count", "Total number of errors during predictions", registry=registry)
predict_request_latency_summary = Summary("iris_predict_request_latency", "Request latency to predict", registry=registry)

# Inference request and response logging counter
inference_request_counter = Counter("iris_inference_requests_total", "Total number of inference requests", registry=registry)

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    total_requests_counter.inc()  # Increment total requests here, even for /metrics

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        # Handle exceptions in middleware to ensure request count increments
        error_count_counter.inc()
        raise e

@app.post("/predict")
@predict_request_latency_summary.time()
async def predict(data: IrisData, request: Request, request_id: Optional[str] = None):
    """
    Predicts the Iris species based on input data.
    """
    if request_id is None:
         request_id = request.headers.get("X-Request-ID", "N/A")  # Get request ID (optional)

    inference_request_counter.inc()
    try:
        input_data = np.array([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]).reshape(1, -1)
        prediction = model.predict(input_data)[0]

        log_data = {
            "request_id": request_id,
            "input_data": data.dict(),
            "prediction": int(prediction),
        }
        logger.info(json.dumps(log_data))  # JSON logging

        return {"prediction": int(prediction)}  # Return as integer for JSON serialization
    except Exception as e:
        log_data = {
            "request_id": request_id,
            "input_data": data.dict(),
            "error": str(e),
        }
        logger.error(json.dumps(log_data))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}

@app.get("/metrics")
async def get_metrics():
    """
    Exposes Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)