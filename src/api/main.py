import mlflow
import pandas as pd
from fastapi import FastAPI

from .pydantic_models import CustomerData, PredictionResponse


app = FastAPI()

# Set MLflow tracking URI to the container's path
mlflow.set_tracking_uri("file:///app/mlruns")

# Load the model from MLflow artifacts inside Docker
model_uri = (
    "file:///app/mlruns/1/models/m-b56f931bfa444e04b71e0ac2ecbe00fb/artifacts"
)
model = mlflow.sklearn.load_model(model_uri)


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    try:
        input_df = pd.DataFrame([data.dict()])
        prob_array = model.predict_proba(input_df)
        print("🔥 Prediction probabilities:", prob_array)
        prob = prob_array[0][1]
        return {"risk_probability": prob}
    except Exception as e:
        return {"detail": f"Prediction failed: {e}"}
from fastapi import FastAPI

app = FastAPI()

@app.get("/ping")
def ping():
    return {"msg": "pong"}
