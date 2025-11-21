
from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Initialize FastAPI application
app = FastAPI(title="Crop Recommendation API")

# Load the saved model, scaler, and label encoder classes
best_model = joblib.load('best_model.joblib')
scaler = joblib.load('scaler.joblib')
le_classes = joblib.load('le_classes.joblib')

# Define the Pydantic model for input data
class CropFeatures(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# Define the /predict endpoint
@app.post("/predict")
async def predict_crop(features: CropFeatures):
    try:
        # Convert the incoming CropFeatures object into a pandas DataFrame
        input_data = features.dict()
        input_df = pd.DataFrame([input_data])

        # Ensure the column order matches the training data features
        feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        input_df = input_df[feature_names]

        # Scale the input features
        input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction_encoded = best_model.predict(input_scaled)

        # Inverse transform the prediction to get the original label
        predicted_label = le_classes[prediction_encoded[0]]

        return {
            "prediction": predicted_label
        }

    except Exception as e:
        return {
            "error": str(e),
            "message": "Error processing prediction request. Ensure input format is correct."
        }
