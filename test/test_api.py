import sys
import os
import warnings
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add src folder to sys.path so 'api' can be imported
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'src',
        )
    ),
)

mock_model = MagicMock()
mock_model.predict_proba.return_value = [[0.3, 0.7]]

patcher = patch('mlflow.sklearn.load_model', return_value=mock_model)
patcher.start()

from fastapi.testclient import TestClient  # noqa: E402
from api.main import app   # noqa: E402
client = TestClient(app)


def test_predict():
    sample_data = {
        "Recency": 1,
        "Frequency": 127,
        "Monetary": 489358,
        "Transaction_Hour": 8,
        "FraudResult": 0,
        "Average_Transaction_Amount": 2424.58,
        "Transaction_Day": 12,
        "ChannelId_ChannelId_2": False,
        "ChannelId_ChannelId_3": True,
        "ChannelId_ChannelId_5": False,
        "ProviderId_ProviderId_2": False,
        "ProviderId_ProviderId_3": False,
        "ProviderId_ProviderId_4": False,
        "ProviderId_ProviderId_5": False,
        "ProviderId_ProviderId_6": True,
        "PricingStrategy_1": False,
        "PricingStrategy_2": True,
        "PricingStrategy_4": False,
        "ProductCategory_data_bundles": False,
        "ProductCategory_financial_services": False,
        "ProductCategory_movies": False,
        "ProductCategory_other": False,
        "ProductCategory_ticket": False,
        "ProductCategory_transport": False,
        "ProductCategory_tv": False,
        "ProductCategory_utility_bill": False,
    }

    response = client.post("/predict", json=sample_data)
    assert response.status_code == 200
    assert "risk_probability" in response.json()
    assert abs(response.json()["risk_probability"] - 0.7) < 1e-6


patcher.stop()
