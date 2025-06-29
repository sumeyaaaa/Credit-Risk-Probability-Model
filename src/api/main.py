from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from src.api.pydantic_models import CustomerData, PredictionResponse

app = FastAPI()

# Load model from MLflow Model Registry
model_name = "best_model"
model_version = 1  # or 'latest'
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.dict()])
    
    # Add any required preprocessing here if needed
    
    risk_prob = model.predict_proba(input_df)[:, 1][0]
    return PredictionResponse(risk_probability=risk_prob)
