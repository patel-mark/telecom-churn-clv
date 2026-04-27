# Telecom Customer Lifetime Value (CLV) & Churn Prediction

![Build Status](https://img.shields.io/badge/Status-Complete-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Tableau](https://img.shields.io/badge/Tableau-Server-orange) ![Machine Learning](https://img.shields.io/badge/Model-XGBoost%20%7C%20SMOTE%20%7C%20SHAP-green)

> **Note:** Replace the image path below with the actual path to your Tableau dashboard screenshot.
> Example: `![Executive Dashboard](./images/tableau_dashboard.png)`

![Executive Dashboard](./path/to/your/tableau_dashboard_image.jpg)

## 📖 Overview
An end-to-end Machine Learning and Business Intelligence pipeline designed for the **ICT & Telecom** industry. This project addresses the core business need of customer analytics and revenue protection by identifying high-value customers at risk of leaving, enabling targeted retention campaigns.

## 💼 Business Problem
Telecom operators lose millions annually to preventable customer churn. The primary challenge is identifying high-value customers *before* they leave, in order to target retention campaigns effectively and maximize revenue, rather than relying on inefficient, broad-spectrum marketing.

## 🎯 Objectives
1. **CLV Segmentation:** Develop a regression model to segment customers by their projected long-term value.
2. **Churn Prediction:** Build a predictive classification model using call detail records (CDRs) and usage patterns.
3. **Interactive BI:** Create an analyst-friendly dashboard to track segment health, monitor model predictions, and drive executive decision-making.

## 🏗️ Architecture & Workflow

> **Note:** Replace the image path below with the actual path to your Draw.io architecture flow diagram.
> Example: `![Architecture Flow](./images/architecture_diagram.png)`

![Architecture Flow](./path/to/your/architecture_flow_image.png)

1. **Data Ingestion & Cleaning:** SQL queries are utilized to preprocess, clean, and join massive datasets directly within the data warehouse.
2. **Feature Engineering:** Python (`pandas`) is used to synthesize behavioral features such as average recharge value, network drop call rates, and recency of interactions.
3. **Modeling:** Trained and evaluated multiple classification models (Logistic Regression, Random Forest, XGBoost) for churn probability, and regression models for CLV.
4. **Deployment & Visualization:** Model predictions and key feature coefficients are exported to Tableau to populate the interactive "Customer Health Dashboard."

## 📊 Data Sources
The models are trained on synthesized, comprehensive telecom datasets:
* **Call Detail Records (CDRs):** Call duration, frequency, and roaming status.
* **Usage Data:** Mobile data consumption and recharge patterns.
* **CRM Data:** Customer service interaction logs and demographic information.

## 🚀 Key Features & Advanced Concepts
* **What-If Analysis:** An interactive tool for retention teams to simulate the financial impact of various promotional offers and retention strategies.
* **Explainable AI (SHAP):** Feature importance analysis that explicitly explains *why* a specific customer is predicted to churn, allowing for personalized interventions.
* **Advanced Techniques:** Handled highly imbalanced churn classes using **SMOTE** (Synthetic Minority Over-sampling Technique) and executed detailed cohort analysis.

## 🛠️ Tech Stack
* **Data Science & ML:** Python (Pandas, Scikit-learn, XGBoost, SHAP)
* **Database & Querying:** SQL
* **Data Visualization:** Tableau
* **MLOps:** MLflow

## ⚙️ Deployment Strategy
* **Model Management:** Models are packaged, versioned, and tracked using **MLflow**.
* **Automation:** The prediction pipeline is scheduled as a monthly batch process to score the active customer base.
* **BI Deployment:** The final dashboard is deployed to **Tableau Server** for secure, internal access by analysts and retention teams.

## 📈 Business Impact
This pipeline directly informs targeted marketing spend. By focusing resources on high-value, high-risk accounts, it drastically improves the ROI on retention campaigns and measurably reduces the overall churn rate.

---
## 👨‍💻 Author
**Mark Patel** [LinkedIn Profile](#) | [GitHub Profile](#)
