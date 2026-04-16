import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Initialize Faker and set seeds for reproducibility
fake = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

NUM_CUSTOMERS = 1000
DAYS_OF_DATA = 90

def generate_demographics():
    print("Generating CRM Demographics...")
    customer_ids = [f"CUST_{str(i).zfill(5)}" for i in range(1, NUM_CUSTOMERS + 1)]
    
    data = {
        "customer_id": customer_ids,
        "age": np.random.randint(18, 75, NUM_CUSTOMERS),
        "gender": np.random.choice(["Male", "Female", "Other"], NUM_CUSTOMERS, p=[0.48, 0.48, 0.04]),
        "location_region": np.random.choice(
            ["Nairobi", "Mombasa", "Kisumu", "Nakuru", "Kiambu", "Uasin Gishu"], 
            NUM_CUSTOMERS, 
            p=[0.4, 0.15, 0.1, 0.1, 0.15, 0.1]
        ),
        "tenure_months": np.random.randint(1, 60, NUM_CUSTOMERS),
        "contract_type": np.random.choice(["Prepaid", "Postpaid"], NUM_CUSTOMERS, p=[0.85, 0.15])
    }
    return pd.DataFrame(data)

def generate_recharges(customers_df):
    print("Generating Recharge Patterns...")
    recharge_records = []
    
    # Only prepaid customers get frequent recharges
    prepaid_customers = customers_df[customers_df['contract_type'] == 'Prepaid']['customer_id'].tolist()
    
    for cust_id in prepaid_customers:
        # Random number of recharges over the 90-day period
        num_recharges = np.random.randint(2, 15)
        for _ in range(num_recharges):
            recharge_records.append({
                "customer_id": cust_id,
                "recharge_date": fake.date_between(start_date=f"-{DAYS_OF_DATA}d", end_date="today"),
                "amount": np.random.choice([50, 100, 250, 500, 1000, 2000], p=[0.3, 0.3, 0.2, 0.1, 0.05, 0.05]),
                "payment_method": np.random.choice(["M-Pesa", "Airtel Money", "Bank App", "Card"], p=[0.7, 0.1, 0.15, 0.05])
            })
    return pd.DataFrame(recharge_records)

def generate_cdrs(customers_df):
    print("Generating Call Detail Records (CDRs)...")
    cdr_records = []
    
    for cust_id in customers_df['customer_id']:
        # High-value customers make more calls
        num_calls = np.random.randint(10, 150)
        for _ in range(num_calls):
            cdr_records.append({
                "customer_id": cust_id,
                "call_date": fake.date_time_between(start_date=f"-{DAYS_OF_DATA}d", end_date="now"),
                "duration_minutes": round(np.random.exponential(scale=3.5), 2), # Exponential distribution for call lengths
                "call_type": np.random.choice(["On-net", "Off-net", "International"], p=[0.6, 0.35, 0.05]),
                "dropped_call": np.random.choice([True, False], p=[0.02, 0.98]) # 2% drop rate
            })
    return pd.DataFrame(cdr_records)

if __name__ == "__main__":
    # Generate DataFrames
    df_crm = generate_demographics()
    df_recharge = generate_recharges(df_crm)
    df_cdrs = generate_cdrs(df_crm)
    
    # Save to raw data folder
    os.makedirs('data/raw', exist_ok=True)
    df_crm.to_csv('data/raw/crm_demographics.csv', index=False)
    df_recharge.to_csv('data/raw/recharge_patterns.csv', index=False)
    df_cdrs.to_csv('data/raw/cdr_logs.csv', index=False)
    
    print("✅ Data generation complete! Files saved to data/raw/")