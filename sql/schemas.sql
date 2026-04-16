-- CRM Demographics
CREATE TABLE crm_demographics (
    customer_id VARCHAR(50) PRIMARY KEY,
    age INT,
    gender VARCHAR(10),
    location_region VARCHAR(50),
    tenure_months INT,
    contract_type VARCHAR(20)
);

-- Recharge Patterns
CREATE TABLE recharge_patterns (
    transaction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES crm_demographics(customer_id),
    recharge_date DATE,
    amount DECIMAL(10, 2),
    payment_method VARCHAR(30)
);

-- Call Detail Records (CDRs)
CREATE TABLE cdr_logs (
    log_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES crm_demographics(customer_id),
    call_date TIMESTAMP,
    duration_minutes DECIMAL(10, 2),
    call_type VARCHAR(20), -- e.g., On-net, Off-net, International
    dropped_call BOOLEAN
);

-- Prediction Output (For Tableau later)
CREATE TABLE prediction_outputs (
    prediction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50),
    prediction_date DATE,
    churn_probability DECIMAL(5, 4),
    predicted_clv DECIMAL(10, 2),
    top_risk_factor VARCHAR(100)
);