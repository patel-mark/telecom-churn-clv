import pandas as pd
from sqlalchemy import create_engine, text

# Update with your PostgreSQL credentials
DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"

def load_data():
    engine = create_engine(DB_URI)
    
    # --- NEW: Safely clear out the old data before loading ---
    with engine.begin() as conn:
        print("Clearing old data to prevent duplicate keys...")
        conn.execute(text("TRUNCATE TABLE crm_demographics, cdr_logs, recharge_patterns CASCADE;"))
    # ---------------------------------------------------------
    
    print("Loading CRM Demographics...")
    pd.read_csv('data/raw/crm_demographics.csv').to_sql('crm_demographics', engine, if_exists='append', index=False)
    
    print("Loading CDR Logs...")
    pd.read_csv('data/raw/cdr_logs.csv').to_sql('cdr_logs', engine, if_exists='append', index=False)
    
    print("Loading Recharge Patterns...")
    pd.read_csv('data/raw/recharge_patterns.csv').to_sql('recharge_patterns', engine, if_exists='append', index=False)
    
    print("✅ Raw Data successfully loaded into PostgreSQL Database!")

if __name__ == "__main__":
    load_data()