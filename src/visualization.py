import matplotlib.pyplot as plt
import seaborn as sns


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


def visualize_individual_metrics(rfm_df):
    sns.set(style='whitegrid')
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    sns.barplot(
        x=['Recency'],
        y=[rfm_df['Recency'].mean()],
        ax=axs[0],
        palette='Blues'
    )
    axs[0].set_title('Average Recency', fontsize=16)
    axs[0].set_ylabel('Days', fontsize=14)
    axs[0].annotate(
        f'{rfm_df["Recency"].mean():.2f}',
        xy=(0, rfm_df['Recency'].mean()),
        xytext=(0, rfm_df['Recency'].mean() + 1),
        ha='center',
        fontsize=12,
        color='black'
    )

    sns.barplot(
        x=['Frequency'],
        y=[rfm_df['Frequency'].mean()],
        ax=axs[1],
        palette='Greens'
    )
    axs[1].set_title('Average Frequency', fontsize=16)
    axs[1].set_ylabel('Number of Transactions', fontsize=14)
    axs[1].annotate(
        f'{rfm_df["Frequency"].mean():.2f}',
        xy=(0, rfm_df['Frequency'].mean()),
        xytext=(0, rfm_df['Frequency'].mean() + 5),
        ha='center',
        fontsize=12,
        color='black'
    )

    sns.barplot(
        x=['Monetary'],
        y=[rfm_df['Monetary'].mean()],
        ax=axs[2],
        palette='Reds'
    )
    axs[2].set_title('Average Monetary Value', fontsize=16)
    axs[2].set_ylabel('Total Amount', fontsize=14)
    axs[2].annotate(
        f'{rfm_df["Monetary"].mean():,.2f}',
        xy=(0, rfm_df['Monetary'].mean()),
        xytext=(0, rfm_df['Monetary'].mean() + 50000),
        ha='center',
        fontsize=12,
        color='black'
    )

    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def visualize_rfm(rfm_df):
    plt.figure(figsize=(12, 6))
    rfm_df[['Recency', 'Frequency', 'Monetary']].mean().plot(
        kind='bar',
        color=['orange', 'blue', 'green']
    )
    plt.title('Average RFM Metrics')
    plt.ylabel('Average Value')
    plt.xticks(rotation=45)
    plt.show()


def visualize_clusters(rfm_df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=rfm_df,
        x='Recency',
        y='Monetary',
        hue='Cluster',
        palette='viridis',
        s=100
    )
    plt.title('Customer Clusters Based on RFM Metrics')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')
    plt.legend(title='Cluster')
    plt.show()


def visualize_high_risk_distribution(rfm_df):
    high_risk_counts = rfm_df['is_high_risk'].value_counts()
    plt.figure(figsize=(8, 6))
    plt.pie(
        high_risk_counts,
        labels=['Low Risk', 'High Risk'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['lightblue', 'salmon']
    )
    plt.title('Distribution of High-Risk Customers')
    plt.show()
