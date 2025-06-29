# src/eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(path):
    """Load excel data"""
    return pd.read_excel(path)

def summary_stats(df):
    """Print basic info and stats"""
    print(df.info())
    print("\n--- Describe Numerical ---\n", df.describe())
    print("\n--- Describe Categorical ---\n", df.select_dtypes(include='object').describe())
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def logarithmic_numerical_distribution(df, columns=['Amount', 'Value', 'FraudResult']):
    """
    Plots two log-scale histograms for each specified numeric column:
    one for positive values, one for negative (absolute) values.

    Args:
        df (DataFrame): Input DataFrame.
        columns (list of str): List of numeric columns to plot.
    """
    for column in columns:
        if column not in df.columns:
            print(f"Column '{column}' not found in DataFrame.")
            continue

        # Positive values
        pos_vals = df[df[column] > 0][column]
        if not pos_vals.empty:
            plt.hist(pos_vals, bins=50, log=True)
            plt.title(f'{column} Distribution (positive values only, log scale)')
            plt.xlabel(column)
            plt.ylabel('Frequency (log scale)')
            plt.show()

        # Negative values (absolute)
        neg_vals = np.abs(df[df[column] < 0][column])
        if not neg_vals.empty:
            plt.hist(neg_vals, bins=50, log=True)
            plt.title(f'{column} Distribution (negative values only, log scale on abs values)')
            plt.xlabel(f'Absolute {column}')
            plt.ylabel('Frequency (log scale)')
            plt.show()


num_cols = ['CurrencyCode', 'CountryCode',  'PricingStrategy' ]

def plot_numeric_distributions(df, num_cols=num_cols):
    """Plot histograms for numeric columns with axis labels"""
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

cat_cols = ['CurrencyCode', 'CountryCode', 'ProviderId', 'ProductCategory', 
            'ChannelId', 'PricingStrategy', 'FraudResult']
def plot_categorical_distributions(df, cat_cols=cat_cols, top_k=10):
    """Plot barplots for categorical features"""
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
    missing_df = pd.DataFrame({'Missing Values': missing, 'Percent': missing_percent})
    print(missing_df[missing_df['Missing Values'] > 0])

def plot_correlations(df):
    """Plot correlation heatmap for numeric features"""
    corr = df.select_dtypes(include=['int64', 'float64']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
    plt.title("Correlation Matrix")
    plt.show()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

def cramers_v(confusion_matrix):
    """Calculate Cramér's V statistic for categorical-categorical association"""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))  # Bias correction
    r_corr = r - ((r - 1)**2) / (n - 1)
    k_corr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2_corr / min((k_corr - 1), (r_corr - 1)))


def cramers_v_matrix(df, cat_cols):
    """Compute Cramér's V matrix for multiple categorical features"""
    matrix = pd.DataFrame(np.zeros((len(cat_cols), len(cat_cols))), 
                          index=cat_cols, columns=cat_cols)
    for col1 in cat_cols:
        for col2 in cat_cols:
            if col1 == col2:
                matrix.loc[col1, col2] = 1.0
            else:
                cm = pd.crosstab(df[col1], df[col2])
                matrix.loc[col1, col2] = cramers_v(cm)
    return matrix


def plot_cramers_v_heatmap(df, categorical_features, figsize=(6,4), cmap='YlOrBr'):
    """
    Computes and plots a heatmap of Cramér's V correlation matrix for given categorical features.

    Args:
        df (DataFrame): The dataset containing the features.
        categorical_features (list of str): List of categorical column names to analyze.
        figsize (tuple): Figure size for the plot (default: (6,4)).
        cmap (str): Colormap for the heatmap (default: 'YlOrBr').

    Returns:
        None. Displays the heatmap plot.
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
    Drops rows from the DataFrame that have missing values in the specified columns.

    Args:
        df (DataFrame): The input DataFrame.
        columns (list of str): List of column names to check for missing values.

    Returns:
        DataFrame: A cleaned DataFrame with rows dropped where specified columns have NaNs.
    """
    return df.dropna(subset=columns)
