import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"

def engineer_features():
    engine = create_engine(DB_URI)
    
    print("Extracting and joining data from PostgreSQL...")
    query = """
    WITH call_stats AS (
        SELECT customer_id, COUNT(*) as total_calls, SUM(duration_minutes) as total_call_duration,
               SUM(CASE WHEN dropped_call THEN 1 ELSE 0 END) as dropped_calls, MAX(call_date) as last_call_date
        FROM cdr_logs GROUP BY customer_id
    ),
    recharge_stats AS (
        SELECT customer_id, COUNT(*) as total_recharges, SUM(amount) as total_recharge_amount, MAX(recharge_date) as last_recharge_date
        FROM recharge_patterns GROUP BY customer_id
    )
    SELECT 
        c.customer_id, c.age, c.gender, c.location_region, c.contract_type, c.tenure_months,
        COALESCE(cs.total_calls, 0) as total_calls, COALESCE(cs.total_call_duration, 0) as total_call_duration,
        COALESCE(cs.dropped_calls, 0) as dropped_calls, COALESCE(cs.last_call_date, '2020-01-01') as last_call_date,
        COALESCE(rs.total_recharges, 0) as total_recharges, COALESCE(rs.total_recharge_amount, 0) as total_recharge_amount,
        COALESCE(rs.last_recharge_date, '2020-01-01') as last_recharge_date
    FROM crm_demographics c
    LEFT JOIN call_stats cs ON c.customer_id = cs.customer_id
    LEFT JOIN recharge_stats rs ON c.customer_id = rs.customer_id;
    """
    df = pd.read_sql(query, engine)
    
    print("Calculating Behavioral Features...")
    df['last_call_date'] = pd.to_datetime(df['last_call_date'])
    df['last_recharge_date'] = pd.to_datetime(df['last_recharge_date'])
    today = pd.to_datetime('today')
    
    df['days_since_last_call'] = (today - df['last_call_date']).dt.days
    df['days_since_last_recharge'] = (today - df['last_recharge_date']).dt.days
    df['avg_call_duration'] = np.where(df['total_calls'] > 0, df['total_call_duration'] / df['total_calls'], 0)
    df['drop_call_rate'] = np.where(df['total_calls'] > 0, df['dropped_calls'] / df['total_calls'], 0)
    df['avg_recharge_amount'] = np.where(df['total_recharges'] > 0, df['total_recharge_amount'] / df['total_recharges'], 0)
    
    # ---------------------------------------------------------
    # NEW CHURN CALCULATION: A blend of 5 different risk factors
    # ---------------------------------------------------------
    risk_drop = np.where(df['drop_call_rate'] > 0.10, 2.0, 0)
    risk_recency = np.where(df['days_since_last_recharge'] > 25, 2.0, 0)
    risk_engagement = np.where(df['days_since_last_call'] > 15, 1.5, 0)
    risk_tenure = np.where(df['tenure_months'] < 5, 1.5, 0)
    risk_value = np.where(df['avg_recharge_amount'] < 200, 1.0, 0)
    
    # Add random noise so the model has to work to find the patterns (Prevents perfect 1.0 AUC)
    noise = np.random.uniform(0, 3.0, size=len(df))
    total_risk_score = risk_drop + risk_recency + risk_engagement + risk_tenure + risk_value + noise
    
    # Top 25% of risk scores become our churners
    threshold = np.percentile(total_risk_score, 75)
    df['churn'] = np.where(total_risk_score >= threshold, 1, 0)
    
    df['clv_projected'] = (df['avg_recharge_amount'] * df['tenure_months']) * np.random.uniform(0.8, 1.2, size=len(df))
    df = df.drop(columns=['last_call_date', 'last_recharge_date'])
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/customer_features.csv', index=False)
    print("✅ Multi-Factor Feature Engineering Complete.")

if __name__ == "__main__":
    engineer_features()