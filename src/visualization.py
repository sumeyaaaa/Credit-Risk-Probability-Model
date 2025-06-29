import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature_distributions(data):
    plt.figure(figsize=(12, 8))
    sns.histplot(data['Net_Total_Transaction_Amount'], bins=30, kde=True)
    plt.title('Distribution of Net Total Transaction Amount')
    plt.xlabel('Net_Total_Transaction_Amount')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.histplot(data['Gross_Transaction_Amount'], bins=30, kde=True)
    plt.title('Distribution of Growth Transaction Amount')
    plt.xlabel('Gross_Transaction_Amount')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.histplot(data['Average_Transaction_Amount'], bins=30, kde=True)
    plt.title('Distribution of Average Transaction Amount')
    plt.xlabel('Average Transaction Amount')
    plt.ylabel('Frequency')
    plt.show()

    plt.figure(figsize=(12, 8))
    sns.countplot(x='Transaction_Hour', data=data)
    plt.title('Transaction Count by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Count of Transactions')
    plt.show()

