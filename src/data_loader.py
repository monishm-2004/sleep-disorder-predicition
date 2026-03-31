"""Module for loading and managing datasets"""
import pandas as pd

def load_data(url=None, filepath=None):
    """
    Load dataset from URL or local file
    
    Args:
        url: URL to CSV file
        filepath: Local path to CSV file
    
    Returns:
        DataFrame with the dataset
    """
    if url:
        return pd.read_csv(url)
    elif filepath:
        return pd.read_csv(filepath)
    else:
        raise ValueError("Either url or filepath must be provided")