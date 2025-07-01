import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def process_data(file_path):
    # Load data
    data = pd.read_excel(file_path)

    # Remove rows with missing values
    data.dropna(inplace=True)

    # Convert TransactionStartTime to datetime
    data['TransactionStartTime'] = pd.to_datetime(
        data['TransactionStartTime']
    )

    # Define feature types
    numerical_features = ['Amount', 'Value']
    categorical_features = [
        'ProviderId', 'ProductId', 'ProductCategory',
        'ChannelId', 'PricingStrategy', 'FraudResult'
    ]

    # Define the numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', RobustScaler())
    ])

    # Define the categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ]
    )

    # Fit and transform the data (not returned in this function)
    _ = preprocessor.fit_transform(data)

    # Create aggregate features
    data['Net_Total_Transaction_Amount'] = data.groupby(
        'CustomerId'
    )['Amount'].transform('sum')

    data['Gross_Transaction_Amount'] = data.groupby(
        'CustomerId'
    )['Value'].transform('sum')

    data['Average_Transaction_Amount'] = data.groupby(
        'CustomerId'
    )['Amount'].transform('mean')

    data['Transaction_Count'] = data.groupby(
        'CustomerId'
    )['TransactionId'].transform('count')

    data['Std_Transaction_Amount'] = data.groupby(
        'CustomerId'
    )['Amount'].transform('std')

    data['Last_Transaction_Date'] = data.groupby(
        'CustomerId'
    )['TransactionStartTime'].transform('max')

    data['Recency_in_person'] = (
        data['TransactionStartTime'].max() - data['Last_Transaction_Date']
    ).dt.days

    # Extract date features
    data['Transaction_Hour'] = data['TransactionStartTime'].dt.hour
    data['Transaction_Day'] = data['TransactionStartTime'].dt.day
    data['Transaction_Month'] = data['TransactionStartTime'].dt.month
    data['Transaction_Year'] = data['TransactionStartTime'].dt.year

    return data


if __name__ == "__main__":
    processed_data = process_data(
        r'C:\Users\ABC\Desktop\10Acadamy\Week 5\Credit-Risk-'
        r'Probability-Model\data\processed'
    )

    processed_data.to_excel(
        r'C:\Users\ABC\Desktop\10Acadamy\Week 5\Credit-Risk-'
        r'Probability-Model\data\processed\processed_data.xlsx',
        index=False
    )
