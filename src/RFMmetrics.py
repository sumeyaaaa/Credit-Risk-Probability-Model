import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_data(filepath):
    """
    Load CSV data from the specified file path.

    Parameters:
        filepath (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Data loaded successfully from: {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"❌ File is empty: {filepath}")
        return None
    except Exception as e:
        print(f"❌ An error occurred while loading the data: {e}")
        return None


def get_snapshot_date(
    data, time_col='TransactionStartTime', last_col='Last_Transaction_Date'
):
    """
    Converts date columns to datetime and returns the snapshot date.

    Parameters:
        data (pd.DataFrame): DataFrame containing transaction data.
        time_col (str): Name of the main transaction time column.
        last_col (str): Name of the last transaction date column.

    Returns:
        pd.Timestamp: The most recent transaction timestamp.
    """
    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
    data[last_col] = pd.to_datetime(data[last_col], errors='coerce')
    snapshot_date = data[time_col].max()
    print("Snapshot Date:", snapshot_date)
    return snapshot_date


def calculate_rfm(data, snapshot_date):
    rfm_df = data.groupby('CustomerId').agg({
        'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,
        'TransactionId': 'count',
        'Value': 'sum'
    }).rename(columns={
        'TransactionStartTime': 'Recency',
        'TransactionId': 'Frequency',
        'Value': 'Monetary'
    }).reset_index()
    return rfm_df


def scale_rfm(rfm_df):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(
        rfm_df[['Recency', 'Frequency', 'Monetary']]
    )
    return rfm_scaled


def cluster_customers(rfm_scaled, n_clusters=3, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(rfm_scaled)
    return clusters


def assign_high_risk_label(rfm_df, high_risk_cluster_number):
    rfm_df['is_high_risk'] = (
        rfm_df['Cluster'] == high_risk_cluster_number
    ).astype(int)
    return rfm_df


def integrate_target_variable(main_data, rfm_df):
    merged_data = main_data.merge(
        rfm_df, on='CustomerId', how='left'
    )
    return merged_data


def main_task_4(
    data, snapshot_date, n_clusters=3, high_risk_cluster_number=2
):
    rfm_df = calculate_rfm(data, snapshot_date)
    rfm_scaled = scale_rfm(rfm_df)
    rfm_df['Cluster'] = cluster_customers(rfm_scaled, n_clusters)
    rfm_df = assign_high_risk_label(rfm_df, high_risk_cluster_number)
    final_data = integrate_target_variable(data, rfm_df)
    return final_data
