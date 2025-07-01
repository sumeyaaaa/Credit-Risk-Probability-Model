from pydantic import BaseModel

class CustomerData(BaseModel):
    Recency: int
    Frequency: int
    Monetary: int
    Transaction_Hour: int
    FraudResult: int
    Average_Transaction_Amount: float
    Transaction_Day: int
    ChannelId_ChannelId_2: bool
    ChannelId_ChannelId_3: bool
    ChannelId_ChannelId_5: bool
    ProviderId_ProviderId_2: bool
    ProviderId_ProviderId_3: bool
    ProviderId_ProviderId_4: bool
    ProviderId_ProviderId_5: bool
    ProviderId_ProviderId_6: bool
    PricingStrategy_1: bool
    PricingStrategy_2: bool
    PricingStrategy_4: bool
    ProductCategory_data_bundles: bool
    ProductCategory_financial_services: bool
    ProductCategory_movies: bool
    ProductCategory_other: bool
    ProductCategory_ticket: bool
    ProductCategory_transport: bool
    ProductCategory_tv: bool
    ProductCategory_utility_bill: bool

class PredictionResponse(BaseModel):
    risk_probability: float
