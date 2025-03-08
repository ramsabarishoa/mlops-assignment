from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

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

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
async def predict(data: IrisData):
    """
    Predicts the Iris species based on input data.
    """
    try:
        input_data = np.array([data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]).reshape(1, -1)
        prediction = model.predict(input_data)[0]
        logger.info(f"Prediction: {prediction} for input data: {data}")
        return {"prediction": int(prediction)} # Return as integer for JSON serialization
    except Exception as e:
        logger.error(f"Prediction error: {e} for input data: {data}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "healthy"}