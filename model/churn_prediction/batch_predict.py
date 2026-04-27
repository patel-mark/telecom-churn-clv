import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import joblib
import shap
import warnings
import datetime
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"

def run_batch_predictions():
    print("Loading engineered features for batch scoring...")
    df = pd.read_csv('data/processed/customer_features.csv')
    
    # Keep customer_id for the final output
    customer_ids = df['customer_id']
    
    print("Loading saved models and encoders...")
    xgb_churn = joblib.load('model/churn_prediction/saved_models/xgboost_churn_model.pkl')
    xgb_clv = joblib.load('model/churn_prediction/saved_models/xgboost_clv_model.pkl')
    encoders = joblib.load('model/churn_prediction/saved_models/encoders.pkl')
    feature_names = joblib.load('model/churn_prediction/saved_models/feature_names.pkl')
    
    # Prepare data for models
    X = df.drop(columns=['customer_id', 'churn', 'clv_projected'], errors='ignore')
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.transform(X[col])
            
    # Ensure column order matches training exactly
    X = X[feature_names]
    
    print("Generating predictions...")
    # Get probability of churn (Class 1)
    churn_probs = xgb_churn.predict_proba(X)[:, 1]
    # Get predicted CLV
    clv_preds = xgb_clv.predict(X)
    
    print("Calculating SHAP values to find the top risk factor per customer...")
    # We use SHAP to explain exactly WHY each specific customer is at risk
    explainer = shap.TreeExplainer(xgb_churn)
    shap_values = explainer.shap_values(X)
    
    # Find the feature name with the highest SHAP value for each row
    # This represents the biggest factor pushing them toward churning
    top_risk_indices = np.argmax(shap_values, axis=1)
    top_risk_factors = [feature_names[i] for i in top_risk_indices]
    
    print("Structuring final output table...")
    output_df = pd.DataFrame({
        'customer_id': customer_ids,
        'prediction_date': datetime.date.today(),
        'churn_probability': np.round(churn_probs, 4),
        'predicted_clv': np.round(clv_preds, 2),
        'top_risk_factor': top_risk_factors
    })
    
    print("Pushing predictions to PostgreSQL database...")
    engine = create_engine(DB_URI)
    
    # Append to the prediction_outputs table we created in Phase 1
    output_df.to_sql('prediction_outputs', engine, if_exists='append', index=False)
    
    print(f"✅ Batch prediction complete! Successfully scored and saved {len(output_df)} customers.")

if __name__ == "__main__":
    run_batch_predictions()