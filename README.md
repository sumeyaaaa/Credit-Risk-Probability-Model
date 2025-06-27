# Credit-Risk-Probability-Model
This project aims to develop an end-to-end credit scoring system for Bati Bank to assess the creditworthiness of customers using a Buy-Now-Pay-Later (BNPL) service offered in partnership with an eCommerce platform. 

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord emphasizes risk-based capital adequacy and demands internal models that are transparent, well-documented, and auditable. For a credit scoring model to be Basel II-compliant, it must offer clear explanations of how inputs affect predicted outcomes. This need favors the use of interpretable methods like logistic regression with WoE transformations, which align well with regulatory expectations.

### 2. Why Proxy Variables Are Necessary (and Risky)

Due to the lack of direct "default" labels in our data, we construct a proxy variable using behavioral patterns (e.g., RFM analysis) to approximate customer credit risk. This proxy is essential for training a supervised model, but carries risks including misclassification, bias, and regulatory scrutiny. All proxy definitions must be logically justified, empirically validated, and clearly documented.

### 3. Trade-Offs Between Simple and Complex Models

Simple models (e.g., logistic regression) provide transparency and easier regulatory approval but may lack performance in capturing nonlinear patterns. Complex models (e.g., gradient boosting) often deliver better predictive accuracy but at the cost of explainability. In regulated settings like ours, we prioritize interpretability, and only consider complex models when their decisions can be made transparent using tools like SHAP or monotonic constraints.
