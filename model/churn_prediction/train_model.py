import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier, XGBRegressor
import joblib
import os

def load_and_preprocess_data():
    print("Loading clean, engineered features...")
    df = pd.read_csv('data/processed/customer_features.csv')
    
    # Categorical columns to encode
    categorical_cols = ['gender', 'location_region', 'contract_type']
    
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        
    os.makedirs('model/churn_prediction/saved_models', exist_ok=True)
    joblib.dump(encoders, 'model/churn_prediction/saved_models/encoders.pkl')
    
    return df

def train_churn_model(df):
    print("\n--- Training Churn Model (XGBoost Classification) ---")
    
    X = df.drop(columns=['customer_id', 'churn', 'clv_projected'])
    y = df['churn']
    
    # Stratified split ensures the 22% churn rate is maintained in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE to balance churn classes for training...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("Training XGBoost Classifier...")
    # Added slight regularization (reg_lambda) to prevent overfitting on our noisy data
    xgb_clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, reg_lambda=1.5, random_state=42)
    xgb_clf.fit(X_train_smote, y_train_smote)
    
    y_pred = xgb_clf.predict(X_test)
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    joblib.dump(xgb_clf, 'model/churn_prediction/saved_models/xgboost_churn_model.pkl')
    joblib.dump(X.columns.tolist(), 'model/churn_prediction/saved_models/feature_names.pkl')
    print("Churn model saved!")

def train_clv_model(df):
    print("\n--- Training CLV Model (XGBoost Regression) ---")
    
    X = df.drop(columns=['customer_id', 'churn', 'clv_projected'])
    y = df['clv_projected']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost Regressor...")
    xgb_reg = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
    xgb_reg.fit(X_train, y_train)
    
    y_pred = xgb_reg.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"Mean Absolute Error (MAE): KES {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): KES {rmse:.2f}")
    
    joblib.dump(xgb_reg, 'model/churn_prediction/saved_models/xgboost_clv_model.pkl')
    print("CLV model saved!")

if __name__ == "__main__":
    df_features = load_and_preprocess_data()
    train_churn_model(df_features)
    train_clv_model(df_features)
    print("\n✅ Phase 4 Complete: Leak-Free Models trained and saved!")