import os
import pandas as pd

# Define your output path using os.path.join
output_path = os.path.join(
    "C:/Users/ABC/Desktop/10Acadamy/Week 5",
    "Credit-Risk-Probability-Model",
    "data",
    "processed",
)


def convert_tz_aware_to_naive(df):
    """
    Convert timezone-aware datetime to timezone-unaware.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: DataFrame with timezone-aware datetimes converted to naive.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if hasattr(df[col].dt, "tz") and df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    return df


def save_dataframe_to_csv(df, filename):
    """
    Save a DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    filename (str): The name of the CSV file (without extension).
    """
    df = convert_tz_aware_to_naive(df)
    full_path = os.path.join(output_path, f"{filename}.csv")
    df.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")
