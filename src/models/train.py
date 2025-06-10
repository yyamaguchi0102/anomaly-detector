import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from datetime import datetime
import sys
import logging

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_engineering import LogFeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(data_path, model_path, contamination=0.05):
    """
    Train an Isolation Forest model on the log data.
    
    Args:
        data_path (str): Path to the input CSV file
        model_path (str): Path to save the trained model
        contamination (float): Expected proportion of anomalies in the data
    """
    try:
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Initialize feature engineer
        feature_engineer = LogFeatureEngineer()
        
        # Transform data
        logger.info("Engineering features...")
        X = feature_engineer.fit_transform(df)
        
        # Train model
        logger.info("Training Isolation Forest model...")
        model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X)
        
        # Save model and feature engineer
        logger.info(f"Saving model to {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump({
            'model': model,
            'feature_engineer': feature_engineer,
            'timestamp': datetime.now()
        }, model_path)
        
        # Calculate and log anomaly scores
        scores = model.score_samples(X)
        anomaly_scores = -scores  # Convert to positive scale where higher means more anomalous
        
        # Log some statistics
        logger.info(f"Number of samples: {len(X)}")
        logger.info(f"Mean anomaly score: {anomaly_scores.mean():.3f}")
        logger.info(f"Std anomaly score: {anomaly_scores.std():.3f}")
        logger.info(f"Max anomaly score: {anomaly_scores.max():.3f}")
        
        return model, feature_engineer
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def main():
    # Define paths
    data_path = 'data/raw/sample_logs.csv'
    model_path = 'data/processed/isolation_forest_model.joblib'
    
    # Train model
    train_model(data_path, model_path)

if __name__ == "__main__":
    main() 