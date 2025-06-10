import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

class LogFeatureEngineer:
    def __init__(self):
        self.label_encoders = {}
        self.ip_encoder = LabelEncoder()
        self.user_encoder = LabelEncoder()
        self.action_encoder = LabelEncoder()
        
    def _extract_time_features(self, df):
        """Extract time-based features from timestamp."""
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
        return df
    
    def _encode_categorical(self, df):
        """Encode categorical variables."""
        # Encode IP addresses
        df['ip_encoded'] = self.ip_encoder.fit_transform(df['ip_address'])
        
        # Encode user IDs
        df['user_encoded'] = self.user_encoder.fit_transform(df['user_id'])
        
        # Encode actions
        df['action_encoded'] = self.action_encoder.fit_transform(df['action'])
        
        return df
    
    def _calculate_rolling_features(self, df):
        """Calculate rolling window features."""
        # Sort by timestamp to ensure correct rolling calculations
        df = df.sort_values('timestamp')
        
        # Calculate rolling features for each user
        for user in df['user_id'].unique():
            user_mask = df['user_id'] == user
            user_data = df[user_mask]
            
            # Rolling count of actions in last hour
            df.loc[user_mask, 'actions_last_hour'] = user_data.rolling(
                window='1H', on='timestamp'
            )['action'].count()
            
            # Rolling mean response time in last hour
            df.loc[user_mask, 'avg_response_time_last_hour'] = user_data.rolling(
                window='1H', on='timestamp'
            )['response_time'].mean()
            
            # Rolling count of failed attempts in last hour
            df.loc[user_mask, 'failed_attempts_last_hour'] = user_data.rolling(
                window='1H', on='timestamp'
            )['status_code'].apply(lambda x: (x >= 400).sum())
        
        return df
    
    def transform(self, df):
        """Transform the log data into features."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract time features
        df = self._extract_time_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df)
        
        # Calculate rolling features
        df = self._calculate_rolling_features(df)
        
        # Fill NaN values with 0 for rolling features
        df = df.fillna(0)
        
        # Select features for model
        feature_columns = [
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'ip_encoded', 'user_encoded', 'action_encoded',
            'status_code', 'response_time',
            'actions_last_hour', 'avg_response_time_last_hour',
            'failed_attempts_last_hour'
        ]
        
        return df[feature_columns]
    
    def fit_transform(self, df):
        """Fit the encoders and transform the data."""
        return self.transform(df) 