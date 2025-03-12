from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import joblib
import numpy as np
import logging
import time
import json

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

# In-memory metrics storage
metrics = {
    "total_requests": 0,
    "error_count": 0,
    "total_latency": 0.0
}

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.middleware("http")  # Add a middleware to capture request timing
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)  #adding to headers
    return response

@app.post("/predict")
async def predict(data: IrisData, request: Request):
    """
    Predicts the Iris species based on input data.
    """
    request_id = request.headers.get("X-Request-ID", "N/A")  # Get request ID (optional)
    start_time = time.time()
    try:
        metrics["total_requests"] += 1
        input_data = np.array([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        end_time = time.time()
        latency = end_time - start_time
        metrics["total_latency"] += latency

        log_data = {
            "request_id": request_id,
            "input_data": data.dict(),
            "prediction": int(prediction),
            "latency": latency
        }
        logger.info(json.dumps(log_data))  # JSON logging

        return {"prediction": int(prediction)} # Return as integer for JSON serialization
    except Exception as e:
        metrics["error_count"] += 1
        end_time = time.time()
        latency = end_time - start_time

        log_data = {
            "request_id": request_id,
            "input_data": data.dict(),
            "error": str(e),
            "latency": latency
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
    Exposes basic metrics.
    """
    avg_latency = metrics["total_latency"] / metrics["total_requests"] if metrics["total_requests"] > 0 else 0
    return {
        "total_requests": metrics["total_requests"],
        "error_count": metrics["error_count"],
        "average_latency": avg_latency
    }
