# Telecom Customer Churn & CLV Prediction Pipeline

![Build Status](https://img.shields.io/badge/Status-Complete-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Tableau](https://img.shields.io/badge/Tableau-Cloud-orange) ![Machine Learning](https://img.shields.io/badge/Model-XGBoost%20%7C%20SMOTE%20%7C%20SHAP-green)

![Executive Dashboard](https://i.postimg.cc/Qd8DYMzK/telecom-dashboard-merged.png)

## 📖 Overview & Business Problem
Telecom operators lose millions annually to preventable customer churn. The primary challenge is identifying high-value customers *before* they leave, enabling targeted retention campaigns rather than inefficient, broad-spectrum marketing. 

This end-to-end Machine Learning and Business Intelligence pipeline transforms raw telecommunications logs (Call Detail Records, Recharge Patterns, CRM) into an automated, executive-facing dashboard. By predicting *who* is going to leave, *how much* they are worth, and *why* they are leaving, retention teams can deploy surgical, high-ROI interventions.

## 🎯 Objectives
1. **CLV Segmentation:** Develop a regression model to predict Customer Lifetime Value (CLV) and segment customers by projected long-term value.
2. **Churn Prediction:** Build a predictive classification model using CDRs and usage patterns.
3. **Interactive BI:** Create an analyst-friendly dashboard to track segment health, monitor model predictions, and drive executive decision-making.

## 📊 Key Executive Insights
Based on the initial batch run of 1,000 customers, the model surfaced the following critical business intelligence:
* **The Financial Impact:** With a baseline churn rate of 25.33%, over **KES 150,460** in projected lifetime value is actively at risk in this sample alone.
* **Geographic Strategy:** **Nairobi** operates as a stable stronghold with high revenue and retention. Conversely, **Kiambu** and **Uasin Gishu** flash red with the highest average churn probabilities, necessitating immediate local investigation.
* **Core Drivers of Churn:** Contrary to assumptions about network quality (dropped calls), churn is primarily behavioral. The #1 indicator of flight risk is `days_since_last_recharge`, followed by low `tenure_months`.

## 🏗️ Architecture & Workflow

![Architecture Flow](https://i.postimg.cc/CM4nRRbJ/Telecom-Customer-Lifetime-Value-(CLV)-Churn-Prediction.webp)

The pipeline is built for scalability and automated batch prediction, simulating a production environment:
1. **Data Ingestion & Cleaning:** Raw synthesized data (CDRs, Mobile Data Usage, CRM logs) is joined and preprocessed via SQL in a PostgreSQL data warehouse.
2. **Feature Engineering:** Python (`pandas`) synthesizes behavioral features such as average recharge value, network drop call rates, and recency of interactions.
3. **Modeling:** Trained and evaluated multiple classification models (Logistic Regression, Random Forest, XGBoost) for churn probability, and regression models for CLV.
4. **Batch Scoring & Deployment:** A monthly Python batch script scores the active base, logs versions via MLflow, and writes predictions back to a SQL Output Table.
5. **BI Visualization:** Tableau Cloud connects to the output layer, rendering an interactive "Customer Health Dashboard."

## 🚀 Key Features & Advanced Concepts
* **Handling Imbalanced Data (SMOTE):** Addressed highly skewed churn classes using Synthetic Minority Over-sampling Technique to ensure robust model performance.
* **Explainable AI (SHAP):** Extracted feature importance analysis that explicitly explains *why* a specific customer is predicted to churn, allowing for personalized interventions.
* **What-If Analysis:** Interactive tool within Tableau for retention teams to simulate the financial impact of promotional offers.
* **Surgical Interventions:** The dashboard provides an **At-Risk Watchlist**, sorting accounts by Highest CLV and Highest Churn Risk for immediate action.

## 🛠️ Tech Stack
* **Data Science & ML:** Python, Pandas, Scikit-Learn, XGBoost, imbalanced-learn (SMOTE), SHAP
* **Database & Querying:** PostgreSQL, DataGrip (SQL)
* **MLOps:** MLflow
* **Data Visualization:** Tableau Cloud / Tableau Server

## 📂 Repository Structure
```text
├── data/
│   ├── raw/                  # Raw CSV dumps (CDR, CRM, Recharge)
│   └── processed/            # Final joined and engineered datasets
├── notebooks/                # Jupyter notebooks for EDA, SMOTE, and model training
├── src/
│   ├── sql/                  # PostgreSQL scripts for table creation and joins
│   ├── features.py           # Pandas feature engineering scripts
│   ├── train.py              # XGBoost training and SHAP extraction
│   └── batch_predict.py      # Monthly scoring script 
├── visuals/                  # Tableau workbook files and architecture diagrams
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ How to Run the Pipeline
1. **Database Setup:** Run the SQL scripts in `src/sql/` to initialize the PostgreSQL schema and load raw data.
2. **Environment Setup:** ```bash
   pip install -r requirements.txt
   ```
3. **Train the Model:** ```bash
   python src/train.py
   ```
   *(Note: This registers the best model in your local MLflow tracking server.)*
4. **Run Batch Predictions:**
   ```bash
   python src/batch_predict.py
   ```
   *(This outputs the `prediction_outputs.csv` which feeds the Tableau dashboard.)*

## 📈 Business Impact
This pipeline directly informs targeted marketing spend. By focusing resources on high-value, high-risk accounts, it drastically improves the ROI on retention campaigns and measurably reduces the overall churn rate.

---
## 👨‍💻 Author
**Mark Patel** | [LinkedIn Profile](#) | [GitHub Profile](#)
