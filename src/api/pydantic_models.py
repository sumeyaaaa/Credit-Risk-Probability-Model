from pydantic import BaseModel

class CustomerData(BaseModel):
    Recency: float
    Frequency: int
    Monetary: float
    Transaction_Hour: int
    # Add other features your model expects here

class PredictionResponse(BaseModel):
    risk_probability: float
