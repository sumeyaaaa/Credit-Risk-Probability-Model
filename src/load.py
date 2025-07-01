import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def load_data(path):
    """Load Excel data"""
    return pd.read_excel(path)


def load_data_csv(path):
    """Load CSV data"""
    return pd.read_csv(path)


def summary_stats(df):
    """Print basic info and stats"""
    print(df.info())
    print("\n--- Describe Numerical ---\n", df.describe())
    print("\n--- Describe Categorical ---\n")
    print(df.select_dtypes(include='object').describe())


def logarithmic_numerical_distribution(
    df,
    columns=None,
):
    if columns is None:
        columns = ['Amount', 'Value', 'FraudResult']

    """
    Plot log-scale histograms for positive and negative values of specified columns.
    """
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            continue

        pos_vals = df[df[column] > 0][column]
        if not pos_vals.empty:
            plt.hist(pos_vals, bins=50, log=True)
            plt.title(f'{column} (positive values, log scale)')
            plt.xlabel(column)
            plt.ylabel('Frequency (log scale)')
            plt.show()

        neg_vals = np.abs(df[df[column] < 0][column])
        if not neg_vals.empty:
            plt.hist(neg_vals, bins=50, log=True)
            plt.title(f'{column} (negative abs values, log scale)')
            plt.xlabel(f'Absolute {column}')
            plt.ylabel('Frequency (log scale)')
            plt.show()


num_cols = ['CurrencyCode', 'CountryCode', 'PricingStrategy']


def plot_numeric_distributions(df, num_cols=num_cols):
    """Plot histograms for numeric columns"""
    if num_cols is None:
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in num_cols:
        plt.figure(figsize=(8, 4))
        plt.hist(df[col].dropna(), bins=30, edgecolor='black')
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


cat_cols = [
    'CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory',
    'ChannelId', 'PricingStrategy', 'FraudResult'
]


def plot_categorical_distributions(df, cat_cols=cat_cols, top_k=10):
    """Plot bar plots for categorical features"""
    if cat_cols is None:
        cat_cols = df.select_dtypes(include='object').columns

    for col in cat_cols:
        plt.figure(figsize=(8, 4))
        sns.countplot(data=df, x=col, order=df[col].value_counts().index)
        plt.title(f"Distribution of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def check_missing_values(df):
    """Display missing value counts and percentages"""
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percent': missing_percent
    })
    print(missing_df[missing_df['Missing Values'] > 0])


def plot_correlations(df):
    """Plot correlation heatmap for numeric features"""
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.show()


def cramers_v(confusion_matrix):
    """Calculate Cramér's V for categorical association"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    r_corr = r - ((r - 1)**2) / (n - 1)
    k_corr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))


def cramers_v_matrix(df, cat_cols):
    """Compute Cramér's V matrix for categorical features"""
    matrix = pd.DataFrame(
        np.zeros((len(cat_cols), len(cat_cols))),
        index=cat_cols,
        columns=cat_cols
    )
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                cm = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(cm)
    return matrix


def plot_cramers_v_heatmap(
    df, categorical_features, figsize=(6, 4), cmap='YlOrBr'
):
    """
    Plot Cramér's V heatmap for categorical columns.
    """
    cramers_matrix = cramers_v_matrix(df, categorical_features)
    plt.figure(figsize=figsize)
    sns.heatmap(cramers_matrix, annot=True, cmap=cmap, fmt='.2f')
    plt.title("Cramér's V Correlation Between Categorical Features")
    plt.show()


def detect_outliers(df, num_cols=None):
    """Boxplot for numeric outlier detection"""
    if num_cols is None:
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns

    for col in num_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Outlier Detection - {col}")
        plt.tight_layout()
        plt.show()


def drop_rows_with_missing(df, columns):
    """
    Drop rows with missing values in specified columns.

    Args:
        df (DataFrame): The input DataFrame.
        columns (list): Columns to check for missing values.

    Returns:
        DataFrame: Cleaned DataFrame with specified NaNs dropped.
    """
    return df.dropna(subset=columns)
