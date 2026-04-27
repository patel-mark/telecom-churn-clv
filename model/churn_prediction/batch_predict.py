import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import shap
import datetime
import warnings
import joblib

# -- MLFLOW IMPORTS --
import mlflow
from mlflow.tracking import MlflowClient

warnings.filterwarnings('ignore')
DB_URI = "postgresql://postgres:SecurePassword@localhost:5432/telecom_db"
mlflow.set_tracking_uri("sqlite:///mlruns.db")

def run_batch_predictions():
    print("Locating latest production model in MLflow Registry...")
    experiment = mlflow.get_experiment_by_name("Telecom_Retention_Pipeline")
    # Automatically get the most recent successful run
    latest_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["start_time DESC"], 
        max_results=1
    ).iloc[0]
    run_id = latest_run.run_id
    
    print(f"Loading Models & Encoders from Run ID: {run_id}...")
    xgb_churn = mlflow.xgboost.load_model(f"runs:/{run_id}/churn_model")
    xgb_clv = mlflow.xgboost.load_model(f"runs:/{run_id}/clv_model")
    
    # Download preprocessing artifacts dynamically
    client = MlflowClient()
    local_dir = client.download_artifacts(run_id, "preprocessing", ".")
    encoders = joblib.load(f"{local_dir}/encoders.pkl")
    feature_names = joblib.load(f"{local_dir}/feature_names.pkl")
    
    print("Loading data for batch scoring...")
    df = pd.read_csv('../../elt/src/data/processed/customer_features.csv')
    customer_ids = df['customer_id']
    
    # Prepare data
    X = df.drop(columns=['customer_id', 'churn', 'clv_projected'], errors='ignore')
    for col, le in encoders.items():
        if col in X.columns:
            X[col] = le.transform(X[col])
    X = X[feature_names] # Ensure perfect column alignment
    
    print("Scoring base and calculating SHAP values...")
    churn_probs = xgb_churn.predict_proba(X)[:, 1]
    clv_preds = xgb_clv.predict(X)
    
    explainer = shap.TreeExplainer(xgb_churn)
    shap_values = explainer.shap_values(X)
    top_risk_factors = [feature_names[i] for i in np.argmax(shap_values, axis=1)]
    
    output_df = pd.DataFrame({
        'customer_id': customer_ids,
        'prediction_date': datetime.date.today(),
        'churn_probability': np.round(churn_probs, 4),
        'predicted_clv': np.round(clv_preds, 2),
        'top_risk_factor': top_risk_factors
    })
    
    print("Pushing to PostgreSQL...")
    engine = create_engine(DB_URI)
    output_df.to_sql('prediction_outputs', engine, if_exists='replace', index=False)
    print(f"✅ Batch prediction complete! Successfully scored {len(output_df)} customers.")

if __name__ == "__main__":
    run_batch_predictions()