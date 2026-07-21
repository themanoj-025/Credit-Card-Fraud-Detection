"""
Data Loader Module

Handles loading the credit card fraud dataset and providing basic statistics.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and inspect the credit card fraud dataset."""
    
    # Business cost assumptions (can be configured)
    DEFAULT_AVG_FRAUD_LOSS = 150.0  # Average loss per fraudulent transaction ($)
    DEFAULT_REVIEW_COST = 5.0       # Cost to manually review a flagged transaction ($)
    
    def __init__(
        self,
        data_path: str = "Dataset/Dataset/creditcard.csv",
        avg_fraud_loss: float = None,
        review_cost: float = None
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV dataset
            avg_fraud_loss: Average dollar loss per fraudulent transaction
            review_cost: Cost to manually review a flagged transaction
        """
        self.data_path = Path(data_path)
        self.avg_fraud_loss = avg_fraud_loss or self.DEFAULT_AVG_FRAUD_LOSS
        self.review_cost = review_cost or self.DEFAULT_REVIEW_COST
        self.df = None
        
    def load(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Returns:
            DataFrame with the loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at: {self.data_path}")
        
        logger.info(f"Loading dataset from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
        return self.df
    
    def get_basic_stats(self) -> dict:
        """
        Get basic statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        df = self.df
        n_fraud = df['Class'].sum()
        n_legit = len(df) - n_fraud
        fraud_rate = n_fraud / len(df) * 100
        
        stats = {
            'n_samples': len(df),
            'n_features': df.shape[1] - 1,  # Exclude target
            'n_fraud': int(n_fraud),
            'n_legitimate': int(n_legit),
            'fraud_rate_pct': round(fraud_rate, 4),
            'imbalance_ratio': round(n_legit / n_fraud, 1),
            'amount_mean': round(df['Amount'].mean(), 2),
            'amount_max': round(df['Amount'].max(), 2),
            'time_range': (int(df['Time'].min()), int(df['Time'].max())),
            'avg_fraud_loss': self.avg_fraud_loss,
            'review_cost': self.review_cost,
        }
        
        return stats
    
    def get_column_info(self) -> pd.DataFrame:
        """
        Get information about each column.
        
        Returns:
            DataFrame with column info
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        info = pd.DataFrame({
            'dtype': self.df.dtypes,
            'non_null': self.df.count(),
            'null_pct': (self.df.isnull().sum() / len(self.df) * 100).round(2),
            'n_unique': self.df.nunique(),
        })
        
        return info
    
    def get_class_distribution(self) -> dict:
        """
        Get the class distribution with business metrics.
        
        Returns:
            Dictionary with class distribution and business impact
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        n_fraud = int(self.df['Class'].sum())
        n_legit = int(len(self.df) - n_fraud)
        
        # Business impact if we detected nothing (baseline)
        baseline_loss = n_fraud * self.avg_fraud_loss
        
        return {
            'n_fraud': n_fraud,
            'n_legitimate': n_legit,
            'fraud_rate': round(n_fraud / len(self.df) * 100, 4),
            'baseline_loss': round(baseline_loss, 2),
            'avg_fraud_loss': self.avg_fraud_loss,
            'review_cost': self.review_cost,
        }
    
    def sample_transaction(self) -> dict:
        """
        Get a sample fraudulent transaction for API testing.
        
        Returns:
            Dictionary with sample transaction data
        """
        if self.df is None:
            raise ValueError("Dataset not loaded. Call load() first.")
        
        # Get a fraud sample
        fraud_sample = self.df[self.df['Class'] == 1].iloc[0]
        
        # Convert to dict, excluding the target
        transaction = fraud_sample.drop('Class').to_dict()
        
        return transaction


def load_data(
    data_path: str = "Dataset/Dataset/creditcard.csv",
) -> Tuple[pd.DataFrame, dict]:
    """
    Convenience function to load data and get stats.
    
    Args:
        data_path: Path to the CSV file
        
    Returns:
        Tuple of (DataFrame, stats_dict)
    """
    loader = DataLoader(data_path)
    df = loader.load()
    stats = loader.get_basic_stats()
    
    return df, stats


if __name__ == "__main__":
    # Quick demo
    df, stats = load_data()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print(f"\nFirst 3 rows:\n{df.head(3)}")
