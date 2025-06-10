import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_ip():
    """Generate a random IP address."""
    return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"

def generate_normal_logs(n_samples=1000):
    """Generate normal log entries."""
    # Generate timestamps
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    timestamps = [start_time + timedelta(seconds=random.randint(0, 86400)) for _ in range(n_samples)]
    timestamps.sort()

    # Generate user IDs
    user_ids = [f"user_{random.randint(1, 50)}" for _ in range(n_samples)]

    # Generate IP addresses (some users have consistent IPs)
    ip_map = {user: generate_ip() for user in set(user_ids)}
    ip_addresses = [ip_map[user] for user in user_ids]

    # Generate actions
    actions = random.choices(
        ["login", "logout", "access_file", "download", "upload"],
        weights=[0.4, 0.2, 0.2, 0.1, 0.1],
        k=n_samples
    )

    # Generate status codes
    status_codes = random.choices(
        [200, 201, 400, 401, 403, 404, 500],
        weights=[0.7, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02],
        k=n_samples
    )

    # Generate response times (in milliseconds)
    response_times = np.random.normal(200, 50, n_samples).clip(50, 1000)

    return pd.DataFrame({
        'timestamp': timestamps,
        'user_id': user_ids,
        'ip_address': ip_addresses,
        'action': actions,
        'status_code': status_codes,
        'response_time': response_times
    })

def inject_anomalies(df, n_anomalies=50):
    """Inject anomalies into the normal logs."""
    anomaly_indices = random.sample(range(len(df)), n_anomalies)
    
    for idx in anomaly_indices:
        # Randomly choose anomaly type
        anomaly_type = random.choice(['time', 'ip', 'action', 'status', 'response'])
        
        if anomaly_type == 'time':
            # Rapid successive actions
            df.iloc[idx, df.columns.get_loc('timestamp')] = df.iloc[idx-1]['timestamp'] + timedelta(seconds=1)
        elif anomaly_type == 'ip':
            # Unusual IP address
            df.iloc[idx, df.columns.get_loc('ip_address')] = f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}"
        elif anomaly_type == 'action':
            # Unusual action
            df.iloc[idx, df.columns.get_loc('action')] = random.choice(['delete', 'modify_permissions', 'system_access'])
        elif anomaly_type == 'status':
            # Unusual status code pattern
            df.iloc[idx, df.columns.get_loc('status_code')] = random.choice([418, 429, 503])
        else:  # response
            # Unusual response time
            df.iloc[idx, df.columns.get_loc('response_time')] = random.choice([0, 5000, 10000])

    return df

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate normal logs
    print("Generating normal logs...")
    df = generate_normal_logs(n_samples=1000)
    
    # Inject anomalies
    print("Injecting anomalies...")
    df = inject_anomalies(df, n_anomalies=50)
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Save to CSV
    output_path = 'data/raw/sample_logs.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated sample logs saved to {output_path}")

if __name__ == "__main__":
    main() 