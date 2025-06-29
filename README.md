# Credit-Risk-Probability-Model
This project aims to develop an end-to-end credit scoring system for Bati Bank to assess the creditworthiness of customers using a Buy-Now-Pay-Later (BNPL) service offered in partnership with an eCommerce platform. 

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability
The Basel II Accord emphasizes **risk-sensitive capital requirements**, mandating that banks implement internal systems capable of accurately measuring and managing credit risk. According to both the Basel II framework and guidance in the HKMA report, this regulatory focus drives the need for credit scoring models to be:

- **Interpretable**: Regulators require transparency on how inputs influence risk estimates. Complex “black-box” models must be avoided unless accompanied by strong explanation tools.
- **Auditable**: Every modeling step—from feature selection to probability mapping—must be documented and traceable.
- **Validated**: Basel II demands model validation across multiple timeframes, customer types, and economic conditions to ensure stability and fairness.

This makes **logistic regression with Weight of Evidence (WoE)** and scorecards a popular starting point—they are explainable, stable, and easy to present to risk committees and regulators.

**Summary**

The Basel II Accord emphasizes risk-based capital adequacy and demands internal models that are transparent, well-documented, and auditable. For a credit scoring model to be Basel II-compliant, it must offer clear explanations of how inputs affect predicted outcomes. This need favors the use of interpretable methods like logistic regression with WoE transformations, which align well with regulatory expectations.

### 2. Why Proxy Variables Are Necessary (and Risky)
In the absence of labeled "defaults"—typically defined as **90+ day payment delinquencies**—organizations must rely on **proxy variables** to estimate creditworthiness. As highlighted in the *Statistica Sinica* paper, proxy outcomes such as **RFM segmentation**, **repayment behavior**, or **digital activity patterns** allow for supervised modeling where true outcomes are unavailable.

However, using proxies comes with business risks:

- **Label Risk**: If the proxy does not closely match actual default behavior, model predictions may be misleading.
- **Bias Risk**: Proxy construction can unintentionally encode demographic, behavioral, or platform biases, leading to fairness or ethical concerns.
- **Regulatory Risk**: Decisions based on weak proxies may be questioned by regulators, especially if high-risk customers are misclassified as low risk, affecting capital buffers.

Therefore, proxy variables must be **empirically justified**, **back-tested**, and clearly **disclosed** as approximations of real-world outcomes.

**Summary**

Due to the lack of direct "default" labels in our data, we construct a proxy variable using behavioral patterns (e.g., RFM analysis) to approximate customer credit risk. This proxy is essential for training a supervised model, but carries risks including misclassification, bias, and regulatory scrutiny. All proxy definitions must be logically justified, empirically validated, and clearly documented.

### 3. Trade-Offs Between Simple and Complex Models

As both documents emphasize, financial institutions must carefully choose between **simple models** and **complex models** based on their specific regulatory and business context:

| Feature                   | Simple Model (Logistic Regression + WoE) | Complex Model (Gradient Boosting, etc.)       |
|---------------------------|------------------------------------------|------------------------------------------------|
| **Interpretability**      | Highly interpretable                     | Often opaque or "black-box"                   |
| **Regulatory Acceptance** | Preferred (Basel II compliant)           | Requires justification and explainability tools |
| **Performance**           | Moderate accuracy                        | High predictive power                         |
| **Fairness & Bias Detection** | Easier to audit and correct         | Difficult without tools like SHAP/LIME        |
| **Business Alignment**    | Easier to align with credit policy       | May diverge from policy logic                 |
| **Documentation Effort**  | Lower                                    | Higher (due to complexity)                    |

**Summary**
Simple models (e.g., logistic regression) provide transparency and easier regulatory approval but may lack performance in capturing nonlinear patterns. Complex models (e.g., gradient boosting) often deliver better predictive accuracy but at the cost of explainability. In regulated settings like ours, we prioritize interpretability, and only consider complex models when their decisions can be made transparent using tools like SHAP or monotonic constraints.

**Task 2 - Exploratory Data Analysis (EDA)**
Key EDA Steps

Dataset overview, data types, and summary statistics

Visualization of numerical and categorical distributions

Outlier detection and correlation analysis

Missing value identification

**Key Insights**

Amount and Value are highly skewed with visible outliers

Many unique AccountIds indicating a broad customer base

Imbalanced categorical features (e.g., ChannelId, ProductCategory)

High correlation between Amount and Value (0.99)

Moderate correlation between Amount and FraudResult (0.57)

**Next Steps**

Apply log transformation to Amount and Value

One-hot encode low-cardinality categorical features

## Task 3 - Feature Engineering

**Aggregate Features**

Total Transaction Amount

Average Transaction Amount

Transaction Count

Standard Deviation of Amounts

**Temporal Features**

Extracted hour, day, month, year, and other calendar components from TransactionStartTime

Categorical Encoding

One-Hot Encoding for low-cardinality features

WoE Encoding for interpretable modeling

**Missing Value Handling**
Imputation with mean/median/mode as appropriate

Feature Scaling

RobustScaler for outlier-resistant normalization

## Task 4 - Proxy Target Variable Engineering

**RFM Metric Calculation**

Recency: Days since last transaction

Frequency: Number of transactions

Monetary: Total transaction amount

**K-Means Clustering**

Scaled RFM values, used K-Means to create 3 clusters

Cluster with low Recency, Frequency, and Monetary defined as "high-risk"

**Target Variable Creation**

is_high_risk column: 1 = high-risk, 0 = others

Merged back into full dataset for supervised learning

## Task 5 - Model Training and Tracking

**Setup**

Split data into training and testing sets (e.g., 80/20)

Use MLflow for experiment tracking

Use pytest for unit testing

**Model Training**

Logistic Regression (baseline)

Random Forest

Gradient Boosting (e.g., LightGBM, XGBoost)

**Hyperparameter tuning via GridSearch/RandomizedSearch with CV**

**Evaluation Metrics**

Accuracy

Precision, Recall

F1 Score

ROC-AUC

**MLflow Tracking**

Log experiments, metrics, and parameters

Register best model in MLflow Registry

**Testing**

At least 2 unit tests in tests/test_data_processing.py

Validate helper functions and transformations

## Future Work Plan

Final Model Evaluation

Tune and compare models

Select the most interpretable and performant one

Deployment Pipeline

Create REST API with FastAPI

Dockerize the app

Test locally (curl/Postman)

Set up GitHub Actions for CI/CD

Final Reporting

Slide deck and technical report

Model cards for stakeholders

Documentation for regulators