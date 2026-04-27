import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor

# -- MLFLOW IMPORTS --
import mlflow
import mlflow.xgboost
import mlflow.sklearn

# Set MLflow tracking to a local database
mlflow.set_tracking_uri("sqlite:///mlruns.db")
EXPERIMENT_NAME = "Telecom_Retention_Pipeline"
mlflow.set_experiment(EXPERIMENT_NAME)

def load_and_preprocess_data():
    df = pd.read_csv('../../elt/src/data/processed/customer_features.csv')
    categorical_cols = ['gender', 'location_region', 'contract_type']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    # Save encoders to a temp folder so MLflow can log them as artifacts
    os.makedirs('temp_artifacts', exist_ok=True)
    joblib.dump(encoders, 'temp_artifacts/encoders.pkl')
    return df, categorical_cols

def train_and_log_models():
    df, cat_cols = load_and_preprocess_data()
    
    X = df.drop(columns=['customer_id', 'churn', 'clv_projected'])
    y_churn = df['churn']
    y_clv = df['clv_projected']
    
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, 'temp_artifacts/feature_names.pkl')
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_churn, test_size=0.2, random_state=42, stratify=y_churn)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_clv, test_size=0.2, random_state=42)
    
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_c, y_train_c)

    # -- START MLFLOW RUN --
    with mlflow.start_run(run_name="XGBoost_SMOTE_Run"):
        print("Training Churn Classifier...")
        clf_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05, 'reg_lambda': 1.5}
        xgb_clf = XGBClassifier(**clf_params, random_state=42)
        xgb_clf.fit(X_train_smote, y_train_smote)
        
        y_proba = xgb_clf.predict_proba(X_test_c)[:, 1]
        auc = roc_auc_score(y_test_c, y_proba)
        
        print("Training CLV Regressor...")
        reg_params = {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05}
        xgb_reg = XGBRegressor(**reg_params, random_state=42)
        xgb_reg.fit(X_train_r, y_train_r)
        
        y_pred = xgb_reg.predict(X_test_r)
        rmse = np.sqrt(mean_squared_error(y_test_r, y_pred))

        # -- LOGGING TO REGISTRY --
        print("Logging Models and Metrics to MLflow...")
        mlflow.log_params({"clf_" + k: v for k, v in clf_params.items()})
        mlflow.log_params({"reg_" + k: v for k, v in reg_params.items()})
        mlflow.log_metric("churn_roc_auc", auc)
        mlflow.log_metric("clv_rmse", rmse)
        
        # Log actual models
        mlflow.xgboost.log_model(xgb_clf, "churn_model")
        mlflow.xgboost.log_model(xgb_reg, "clv_model")
        
        # Log encoders & feature names so prediction script can use them
        mlflow.log_artifacts("temp_artifacts", artifact_path="preprocessing")
        
        print(f"✅ Run Successful! ROC-AUC: {auc:.4f} | CLV RMSE: {rmse:.2f}")

if __name__ == "__main__":
    train_and_log_models()