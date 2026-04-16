import pandas as pd
from sqlalchemy import create_engine
import os

# --- CONFIGURATION ---
DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"
# We define "today" based on the max date in our synthetic dataset
REFERENCE_DATE = pd.to_datetime('today') 

def extract_and_build_features():
    engine = create_engine(DB_URI)
    print("Connected to database. Extracting and aggregating features...")

    # 1. Get CRM Data
    query_crm = "SELECT * FROM crm_demographics"
    df_crm = pd.read_sql(query_crm, engine)

    # 2. Aggregate Recharge Data using SQL
    query_recharge = """
    SELECT 
        customer_id,
        COUNT(transaction_id) as total_recharges,
        SUM(amount) as total_recharge_amount,
        AVG(amount) as avg_recharge_amount,
        MAX(recharge_date) as last_recharge_date
    FROM recharge_patterns
    GROUP BY customer_id
    """
    df_recharge = pd.read_sql(query_recharge, engine)
    # Convert date to datetime for Pandas calculations
    df_recharge['last_recharge_date'] = pd.to_datetime(df_recharge['last_recharge_date'])

    # 3. Aggregate CDR Data using SQL
    query_cdrs = """
    SELECT 
        customer_id,
        COUNT(log_id) as total_calls,
        SUM(duration_minutes) as total_call_duration,
        AVG(duration_minutes) as avg_call_duration,
        SUM(CASE WHEN dropped_call = TRUE THEN 1 ELSE 0 END) as dropped_calls,
        MAX(call_date) as last_call_date
    FROM cdr_logs
    GROUP BY customer_id
    """
    df_cdrs = pd.read_sql(query_cdrs, engine)
    df_cdrs['last_call_date'] = pd.to_datetime(df_cdrs['last_call_date'])

    print("Merging datasets...")
    # Merge all dataframes on customer_id
    df_features = df_crm.merge(df_recharge, on='customer_id', how='left')
    df_features = df_features.merge(df_cdrs, on='customer_id', how='left')

    # Fill NaN values for customers who might not have made calls or recharges
    df_features.fillna({
        'total_recharges': 0, 'total_recharge_amount': 0, 'avg_recharge_amount': 0,
        'total_calls': 0, 'total_call_duration': 0, 'avg_call_duration': 0, 'dropped_calls': 0
    }, inplace=True)

    print("Engineering advanced behavioral features...")
    
    # Calculate Drop Call Rate
    df_features['drop_call_rate'] = (df_features['dropped_calls'] / df_features['total_calls']).fillna(0)
    
    # Calculate Recency (Days since last activity)
    # Use the maximum date in the dataset as the reference point to simulate "today"
    max_date_in_data = max(df_features['last_call_date'].max(), df_features['last_recharge_date'].max())
    
    df_features['days_since_last_call'] = (max_date_in_data - df_features['last_call_date']).dt.days.fillna(999)
    df_features['days_since_last_recharge'] = (max_date_in_data - df_features['last_recharge_date']).dt.days.fillna(999)

    # --- DEFINE TARGET VARIABLES FOR MACHINE LEARNING ---
    
    # Target 1: Churn (Classification)
    # Definition: A customer has churned if they haven't made a call OR recharged in the last 15 days.
    df_features['churn'] = ((df_features['days_since_last_call'] > 15) & 
                            (df_features['days_since_last_recharge'] > 15)).astype(int)

    # Target 2: CLV - Customer Lifetime Value (Regression)
    # Proxy Definition: Total monetary value divided by tenure, projected over a standard 12-month period.
    # Adding a small constant (+1) to tenure to avoid division by zero.
    df_features['clv_projected'] = (df_features['total_recharge_amount'] / (df_features['tenure_months'] + 1)) * 12

    # Drop datetime columns as ML models require numeric/categorical data
    df_features.drop(columns=['last_recharge_date', 'last_call_date'], inplace=True)

    # Save the final Feature Matrix
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/customer_features.csv'
    df_features.to_csv(output_path, index=False)
    
    print(f"✅ Feature Engineering complete! Final dataset shape: {df_features.shape}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    extract_and_build_features()