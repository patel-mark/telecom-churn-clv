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
1. **The Financial Impact** (The "What")
In my analysis of this 1,000-customer cohort, I identified a baseline churn rate that is actively threatening 28.56% of total revenue. Out of a KES 15.3M portfolio, I found exactly KES 4,374,321 in projected Customer Lifetime Value (CLV) is actively at risk. To put this in perspective, scaling my model to a standard telecom base of 1 million customers exposes a potential revenue loss of over KES 4.3 Billion. I built this dashboard to provide the exact targeting capabilities required to mitigate that loss.

2. **Regional Performance** (The "Where")
I isolated Nairobi as the undisputed high-stakes hub, generating the highest total revenue (over KES 4.7M) with a massive "safe" baseline. However, my data raises a critical red flag for Mombasa. I discovered this region carries an inverted health ratio where significantly more CLV is actively at risk (~KES 648K) than safe (~KES 380K). Based on this regional breakdown, I strongly advise an investigation into the Mombasa Prepaid market to determine if a competitor has launched an aggressive local campaign in the coastal region.

3. **Core Drivers of Churn** (The "Why")
While initial assumptions often point to network quality as a churn driver, my machine learning model proved otherwise. I found that network issues and dropped calls sit at the absolute bottom of the risk factors, driving only 9 of the flagged churners. Instead, I uncovered that the churn is almost entirely behavioral. The absolute highest indicator of flight risk I identified is days_since_last_recharge, which is solely responsible for 230 of the 256 at-risk customers. My conclusion is that customers are not leaving due to bad service; they are losing the habit of topping up and slipping away due to a lack of ongoing financial engagement.

4. **Strategic Action Plan** (The "How")
To counter this behavioral drop-off, I propose an automated SMS win-back campaign targeting the Prepaid segment, which my analysis shows holds KES 3.5M of the total risk. Triggering a targeted SMS top-up bonus (e.g., "100% data bonus on your next recharge") right at a 15-day recharge drought intercepts the behavior before my model classifies the account as High Risk.

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
