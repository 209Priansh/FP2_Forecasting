import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from autots import AutoTS
import numpy as np

# Absolute path to the AutoTS model file
model_path = 'autots_model.pkl'

# Load the AutoTS model
try:
    model = joblib.load(model_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {model_path}")

# FastAPI app instance
app = FastAPI()

# Define request body using Pydantic BaseModel
class InputData(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float
    rolling_mean_3: float
    rolling_std_3: float
    quarter: int
    day_of_week: int
    is_weekend: int
    ema_3: float
    autocorr_1: float
    autocorr_2: float
    period: str

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Petroleum Production Forecasting API"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to DataFrame without 'period'
        input_data = data.dict(exclude={'period'})
        input_df = pd.DataFrame([input_data])
        # Ensure input_df is correctly formatted and inspect if necessary
        print("Input DataFrame:")
        print(input_df)

        # Make prediction using the loaded AutoTS model
        forecasts = model.predict()
        forecast_values = forecasts.forecast
        prediction = forecast_values.values.flatten()[0]  # Take the first forecasted value

        return {"prediction": prediction, "period": data.period}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI server with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
