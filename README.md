
# Credit Risk Probability Model

## Executive Summary

This project delivers an end-to-end credit scoring system for Bati Bank’s Buy-Now-Pay-Later (BNPL) program in partnership with an eCommerce platform. It transforms transactional behavior data into risk signals using interpretable and auditable machine learning models in compliance with Basel II. The solution includes feature engineering, proxy target creation, model training and evaluation, deployment via FastAPI, and automated CI/CD using GitHub Actions and Docker.

---

## Table of Contents

1. [Business Problem](#1-business-problem)  
2. [Credit Scoring Business Understanding](#2-credit-scoring-business-understanding)  
3. [Data Overview](#3-data-overview)  
4. [Feature Engineering](#4-feature-engineering)  
5. [Proxy Target Variable](#5-proxy-target-variable)  
6. [Modeling and Experiment Tracking](#6-modeling-and-experiment-tracking)  
7. [Evaluation Metrics](#7-evaluation-metrics)  
8. [Model Deployment and API](#8-model-deployment-and-api)  
9. [Testing and CI/CD Pipeline](#9-testing-and-cicd-pipeline)  
10. [Installation](#10-installation)  
11. [Usage](#11-usage)  
12. [Contributing](#12-contributing)  
13. [License](#13-license)  
14. [Acknowledgments](#14-acknowledgments)

---

## 1. Business Problem

Bati Bank aims to launch a BNPL offering via an eCommerce platform. To minimize credit risk and comply with Basel II regulations, a credit scoring model must be developed to assess customer risk based on behavioral transaction data, despite the absence of traditional credit history or default labels.

---

## 2. Credit Scoring Business Understanding

### 2.1 Basel II and Interpretability

Basel II requires banks to use risk-sensitive capital measurement systems. This favors **interpretable, auditable, and validated** models. Techniques such as **Logistic Regression with Weight of Evidence (WoE)** are preferred due to their transparency and ease of communication with regulators.

### 2.2 Proxy Variables in Place of Defaults

In the absence of true default labels, proxy variables (e.g., RFM behavioral segments) are necessary for supervised learning. However, they carry risks:

- **Label Risk**: Poor proxies lead to misleading predictions.
- **Bias Risk**: Behavioral proxies may encode socio-economic or demographic bias.
- **Regulatory Risk**: Decisions based on proxies must be well-documented and justified.

### 2.3 Simple vs. Complex Models

| Feature                  | Simple Model (LogReg + WoE)     | Complex Model (e.g., XGBoost)   |
|--------------------------|----------------------------------|---------------------------------|
| Interpretability         | ✅ High                          | ❌ Low (needs SHAP/LIME)       |
| Regulatory Acceptance    | ✅ Preferred                     | ⚠ Requires explainability      |
| Accuracy                 | ⚠ Moderate                      | ✅ High                         |
| Deployment Simplicity    | ✅ Easy                          | ⚠ More involved                 |

We prioritize interpretability and use complex models only when explanations (e.g., SHAP) are provided.

---

## 3. Data Overview


### 3.1 Project Structure

Copy
credit-risk-model/
├── .github/workflows/ci.yml        # CI/CD configuration
├── data/                           # Data storage (add to .gitignore)
│   ├── raw/                        # Raw data
│   └── processed/                  # Processed data for training
├── notebooks/                      # Jupyter notebooks for analysis
│   └── task 1 and 2               # Exploratory data analysis
       └── load_EDA.ipynb               
    └── task 3               # Feauture engineering
       └── feature-engineering.ipynb               
    └── task 4               # RFMmetrics
       └── RFMmetrics.pynb               
    └── task 5               # Modeling
        └── modeling.ipynb              
├── src/                            # Source code
│   ├── __init__.py
    ├── load.py
    ├── RFMmetrics.py.py
    ├── saveFile.py.py
    ├── visualization.py
│   ├── PreProcessing.py          # Feature engineering script
│   ├── train.py                    # Model training script
│   ├── predict.py                  # Inference script
│   └── api/
│       ├── main.py                 # FastAPI application
│       └── pydantic_models.py      # Pydantic models for API
├── tests/                          # Unit tests
│   └── test_api.py
├── Dockerfile                      # Docker configuration
├── docker-compose.yml              # Docker Compose configuration
├── requirements.txt                # Project dependencies
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
└── LICENSE                       # Project license
└── register.py                      # register

### 3.1 Data Overview


| Column             | Description                                      |
|--------------------|--------------------------------------------------|
| TransactionId      | Unique transaction identifier                    |
| AccountId          | Unique customer identifier                       |
| CustomerId         | Shared ID for customer                           |
| Amount / Value     | Transaction value (debit/credit)                 |
| ChannelId          | Platform used (web, Android, iOS)                |
| ProductCategory    | Grouped product type                             |
| FraudResult        | Fraud label (1: fraud, 0: normal)                |
| ...                | Other behavioral and demographic features        |

Additional engineered features include RFM metrics and time-based aggregates.

---

## 4. Feature Engineering

We applied the following techniques:

- **Aggregate Features**:  
  - Total transaction count  
  - Total/avg/std of transaction values  
  - Transaction recency metrics

- **Extracted Features**:  
  - Hour, day, month of transaction  
  - Transaction patterns across time

- **Encoding**:  
  - One-Hot Encoding for nominal features  
  - WoE encoding for regulatory explainability

- **Handling Missing Values**:  
  - Imputation with median/most frequent  
  - Removal when necessary

- **Scaling**:  
  - StandardScaler for numerical inputs

---

## 5. Proxy Target Variable

Since no "default" column exists:

1. **RFM Metrics** were calculated per customer
2. **K-Means Clustering** (k=3) was applied on scaled RFM values
3. The **least engaged cluster** (low frequency & monetary value) was labeled as `is_high_risk = 1`
4. All other clusters were labeled `0`

This binary proxy was added back to the dataset for supervised learning.

---

## 6. Modeling and Experiment Tracking

- **Models Trained**:  
  - Logistic Regression (with WoE)  
  - Decision Tree  
  - Random Forest  
  - Gradient Boosting

- **Experiment Tracking**:  
  - Tracked all runs with `mlflow`  
  - Registered best model in the **MLflow Model Registry**

- **Training Pipeline**:  
  - `sklearn.pipeline.Pipeline` used for reproducibility  
  - GridSearchCV for hyperparameter tuning

---

## 7. Evaluation Metrics

| Metric         | Meaning                                                 |
|----------------|----------------------------------------------------------|
| Accuracy       | Overall correct predictions                             |
| Precision      | Correct positive predictions                            |
| Recall         | Ability to detect all actual positives                   |
| F1-Score       | Harmonic average of Precision and Recall                 |
| ROC AUC        | Class separation capacity of model                      |

The final model was chosen based on best **F1-score** and **ROC AUC**.

---

## 8. Model Deployment and API

The model is deployed using FastAPI with Docker support.

### 🛠️ Endpoint: `/predict`

#### Request Format
```json
{
  "Recency": 14,
  "Frequency": 5,
  "Monetary": 1200,
  "Transaction_Hour": 16
}
```

#### Response
```json
{
  "risk_probability": 0.73
}
```

Deployed using:
- **FastAPI**
- **Uvicorn**
- **Docker**
- **MLflow model loading**

---

## 9. Testing and CI/CD Pipeline

- ✅ Unit tests with `pytest` (`tests/` folder)
- ✅ Linting with `flake8`
- ✅ GitHub Actions for Continuous Integration

```yaml
# .github/workflows/ci.yml
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run Linter
        run: flake8 .
      - name: Run Tests
        run: pytest
```

---

## 10. Installation

```bash
git clone <repository-url>
cd credit-risk-model
pip install -r requirements.txt
```

---

## 11. Usage

### Run API locally
```bash
uvicorn src.api.main:app --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.

---

## 12. Contributing

Contributions are welcome!  
1. Fork the repo  
2. Create a new branch: `git checkout -b feature-branch`  
3. Make changes and commit: `git commit -m 'Add feature'`  
4. Push: `git push origin feature-branch`  
5. Open a Pull Request

---

## 13. License

This project is licensed under the **Apache License 2.0**.  
See the [LICENSE](./LICENSE) file for more details.

---

## 14. Acknowledgments

- [10 Academy](https://10academy.org) for the challenge and guidance  
- [Xente Data (Kaggle)](https://www.kaggle.com/datasets/atwine/xente-challenge) for providing the dataset  
- Basel II Accord and HKMA for regulatory guidelines  
- [Shichen.name Scorecard](https://shichen.name/scorecard/) for WoE & credit scoring tools

---
