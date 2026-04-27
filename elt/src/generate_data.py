import pandas as pd
import numpy as np
import datetime
import os

def generate_telecom_data(num_customers=1000):
    os.makedirs('data/raw', exist_ok=True)
    customer_ids = [f'CUST_{str(i).zfill(5)}' for i in range(1, num_customers + 1)]

    # Distribute 5 distinct reasons for churning
    personas = np.random.choice(
        ['Healthy', 'Network_Issue', 'Financial_Issue', 'Low_Engagement', 'Newbie_Flight'],
        size=num_customers,
        p=[0.60, 0.10, 0.10, 0.10, 0.10]
    )

    crm_data, cdr_data, recharge_data = [], [], []
    today = datetime.date.today()

    for i, cid in enumerate(customer_ids):
        persona = personas[i]

        # Base CRM
        age = np.random.randint(18, 70)
        gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.48, 0.04])
        region = np.random.choice(['Nairobi', 'Mombasa', 'Kiambu', 'Kisumu', 'Nakuru', 'Uasin Gishu'])
        contract = np.random.choice(['Prepaid', 'Postpaid'], p=[0.8, 0.2])
        tenure = np.random.randint(6, 60) if persona != 'Newbie_Flight' else np.random.randint(1, 4)

        crm_data.append({
            'customer_id': cid, 'age': age, 'gender': gender,
            'location_region': region, 'contract_type': contract, 'tenure_months': tenure
        })

        # Base Behavior
        call_days_ago = np.random.randint(0, 30, size=np.random.randint(15, 60))
        recharge_days_ago = np.random.randint(0, 20, size=np.random.randint(3, 10))
        drop_rate = np.random.uniform(0.01, 0.05)
        call_duration_mean = 15.0
        recharge_mean = 500

        # Inject specific bad behaviors based on Persona
        if persona == 'Network_Issue':
            drop_rate = np.random.uniform(0.15, 0.40) # High drop rate
        elif persona == 'Financial_Issue':
            recharge_days_ago = np.random.randint(20, 60, size=np.random.randint(1, 3))
            recharge_mean = 100 # Low recharge amounts
        elif persona == 'Low_Engagement':
            call_days_ago = np.random.randint(15, 60, size=np.random.randint(1, 10))
            call_duration_mean = 2.0 # Infrequent, very short calls

        # Generate logs
        for days in call_days_ago:
            cdr_data.append({
                'customer_id': cid, 'call_date': today - datetime.timedelta(days=int(days)),
                'duration_minutes': round(np.random.exponential(call_duration_mean), 2),
                'dropped_call': np.random.choice([True, False], p=[drop_rate, 1 - drop_rate])
            })
            
        for days in recharge_days_ago:
            recharge_data.append({
                'customer_id': cid, 'recharge_date': today - datetime.timedelta(days=int(days)),
                'amount': round(np.random.normal(recharge_mean, recharge_mean*0.2), 2)
            })

    pd.DataFrame(crm_data).to_csv('data/raw/crm_demographics.csv', index=False)
    pd.DataFrame(cdr_data).to_csv('data/raw/cdr_logs.csv', index=False)
    pd.DataFrame(recharge_data).to_csv('data/raw/recharge_patterns.csv', index=False)
    print("✅ 5-Persona Synthetic Data Generated!")

if __name__ == "__main__":
    generate_telecom_data()