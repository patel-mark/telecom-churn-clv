-- Create core tables for raw telecom data
CREATE TABLE IF NOT EXISTS crm_demographics (
    customer_id VARCHAR(50) PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    location_region VARCHAR(50),
    contract_type VARCHAR(20),
    tenure_months INT
);

CREATE TABLE IF NOT EXISTS cdr_logs (
    log_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    call_date DATE,
    duration_minutes FLOAT,
    dropped_call BOOLEAN
);

CREATE TABLE IF NOT EXISTS recharge_patterns (
    recharge_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    recharge_date DATE,
    amount DECIMAL
);

-- Create the output table that Tableau will read from
CREATE TABLE IF NOT EXISTS prediction_outputs (
    customer_id VARCHAR(50),
    prediction_date DATE,
    churn_probability FLOAT,
    predicted_clv DECIMAL,
    top_risk_factor VARCHAR(50)
);