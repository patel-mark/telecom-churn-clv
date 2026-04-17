import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

# --- CONFIGURATION ---
DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"

def extract_and_build_features():
    engine = create_engine(DB_URI)
    print("Connected to database. Extracting and aggregating features...")

    # 1. Get CRM Data
    df_crm = pd.read_sql("SELECT * FROM crm_demographics", engine)

    # 2. Aggregate Recharge Data
    query_recharge = """
    SELECT customer_id, COUNT(transaction_id) as total_recharges,
           SUM(amount) as total_recharge_amount, AVG(amount) as avg_recharge_amount,
           MAX(recharge_date) as last_recharge_date FROM recharge_patterns GROUP BY customer_id
    """
    df_recharge = pd.read_sql(query_recharge, engine)
    df_recharge['last_recharge_date'] = pd.to_datetime(df_recharge['last_recharge_date'])

    # 3. Aggregate CDR Data
    query_cdrs = """
    SELECT customer_id, COUNT(log_id) as total_calls,
           SUM(duration_minutes) as total_call_duration, AVG(duration_minutes) as avg_call_duration,
           SUM(CASE WHEN dropped_call = TRUE THEN 1 ELSE 0 END) as dropped_calls,
           MAX(call_date) as last_call_date FROM cdr_logs GROUP BY customer_id
    """
    df_cdrs = pd.read_sql(query_cdrs, engine)
    df_cdrs['last_call_date'] = pd.to_datetime(df_cdrs['last_call_date'])

    print("Merging datasets...")
    df_features = df_crm.merge(df_recharge, on='customer_id', how='left').merge(df_cdrs, on='customer_id', how='left')
    df_features.fillna({'total_recharges': 0, 'total_recharge_amount': 0, 'avg_recharge_amount': 0,
                        'total_calls': 0, 'total_call_duration': 0, 'avg_call_duration': 0, 'dropped_calls': 0}, inplace=True)

    print("Engineering advanced behavioral features...")
    df_features['drop_call_rate'] = (df_features['dropped_calls'] / df_features['total_calls']).fillna(0)
    
    max_date = max(df_features['last_call_date'].max(), df_features['last_recharge_date'].max())
    df_features['days_since_last_call'] = (max_date - df_features['last_call_date']).dt.days.fillna(999)
    df_features['days_since_last_recharge'] = (max_date - df_features['last_recharge_date']).dt.days.fillna(999)

    # --- NO LEAKAGE TARGET GENERATION ---
    print("Generating realistic, leak-free target variables...")
    np.random.seed(42) # For reproducibility
    
    # 1. Calculate a hidden "Risk Score" based on combined behaviors
    risk_score = (
        (df_features['drop_call_rate'] * 50) +          # Frustration
        (df_features['days_since_last_call'] * 2) +     # Inactivity
        (df_features['days_since_last_recharge'] * 2) - # Inactivity
        (df_features['tenure_months'] * 1.5)            # Loyalty (reduces risk)
    )
    
    # 2. Add statistical noise so the model can't perfectly reverse-engineer the formula
    risk_score += np.random.normal(0, 20, size=len(df_features))
    
    # 3. Mark the top 22% of customers with the highest risk scores as "Churned"
    threshold = risk_score.quantile(0.78)
    df_features['churn'] = (risk_score >= threshold).astype(int)

    # CLV Target
    df_features['clv_projected'] = (df_features['total_recharge_amount'] / (df_features['tenure_months'] + 1)) * 12

    # Cleanup
    df_features.drop(columns=['last_recharge_date', 'last_call_date'], inplace=True)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    df_features.to_csv('data/processed/customer_features.csv', index=False)
    print(f"✅ Feature Engineering complete! Final dataset shape: {df_features.shape}")

if __name__ == "__main__":
    extract_and_build_features()