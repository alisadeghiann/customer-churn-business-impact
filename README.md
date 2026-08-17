Customer Churn Prediction & Business Impact Analysis
An end-to-end customer churn analytics solution integrating SQL Server, Python (Machine Learning), Business Impact Modeling, and an interactive Power BI dashboard.
The project demonstrates how raw data from a real-world environment can be transformed into actionable business decisions using modern analytics techniques.

📌 Dataset Source (Kaggle)
This project uses the publicly available Telco Customer Churn Dataset from Kaggle.
All preprocessing, feature engineering, modeling, and business impact calculations were performed on top of this dataset.

📌 Business Problem
Customer churn directly affects recurring revenue and customer lifetime value.
A churn prediction model alone does not answer the key business question:

Which customers should be targeted, and at what prediction threshold, to maximize financial return?

This project bridges the gap between predictive analytics and business decision-making by evaluating multiple intervention thresholds and financial scenarios.

📊 Key Business Insights
3,179 customers predicted as high-risk

$252K estimated monthly revenue at risk

0.55 optimal prediction threshold

Base Scenario yields the highest net benefit

Medium-risk customers represent the largest recoverable revenue segment

🔍 Project Architecture
The solution consists of three integrated layers:

1. SQL Server — Data Extraction
Raw customer data was stored and queried using SQL Server.
Key tasks:

Data import

Cleaning and validation

SQL transformations

Export to Python for modeling

2. Python Machine Learning Pipeline
The ML pipeline predicts the probability of churn for each customer.

Key outputs:

Churn probability

Risk segmentation (Low / Medium / High)

Predicted churn population

Estimated revenue at risk

Customer-level prioritization

Models evaluated:

Logistic Regression

Logistic Regression + SMOTE

Weighted Logistic Regression

Random Forest + SMOTE

XGBoost + SMOTE

The best-performing model was saved as:

Code
best_model_FINAL.pkl
3. Business Impact & Threshold Optimization
Instead of using a default classification threshold, the project evaluates multiple thresholds based on financial impact.

For each threshold:

True Positives (TP)

False Negatives (FN)

Intervention cost

Recovered revenue

Net Benefit

Three financial scenarios were analyzed:

Base Scenario

Low Scenario

High Scenario

Optimal Threshold: 0.55
Maximum Net Benefit: ~$1.60K

4. Power BI Dashboard
The final dashboard translates model outputs into executive-level insights.

📈 Dashboard Pages
Page 1 — Customer Churn Overview
High-risk customers

Predicted churn population

Monthly revenue at risk

Risk distribution

Probability distribution

High/medium-risk customer table

Page 2 — Business Impact & Threshold Analysis
Optimal threshold

Net benefit comparison

Scenario analysis

Threshold impact chart

Financial outcome table

Page 3 — Executive Summary
Key findings

Recommended retention strategy

Targeting priorities

Action plan

🧠 Analytical Workflow
Code
SQL Data → Python Preprocessing → ML Modeling → Risk Segmentation → 
Threshold Optimization → Financial Modeling → Power BI Dashboard → Business Strategy
This workflow demonstrates the transition from predictive analytics to prescriptive decision-making.

🛠️ Tech Stack
SQL Server — Data storage & extraction

Python — ML pipeline & business modeling

Pandas — Data manipulation

Scikit-learn — Model development

Power BI — Dashboard & executive reporting

DAX — KPI calculations

GitHub — Version control & documentation

📁 Repository Structure
Code
customer-churn-business-impact/
│
├── churn_prediction_pipeline.py
├── clean_data_FINAL.csv
├── model_output_FINAL.csv
├── powerbi_predictions_FINAL.csv
├── threshold_business_FINAL.csv
├── best_model_FINAL.pkl
│
├── dashboard/
│   └── dashboard.pbix
│
├── images/
│   ├── overview.png
│   ├── business-impact.png
│   └── executive-summary.png
│
└── README.md
🎯 Project Goals
This project aims to determine:

Who is likely to churn

Who should be targeted

How many customers should be targeted

At what probability threshold

Expected financial impact

Which scenario provides the best return

👤 Author
Ali  
Data Analyst | BI Developer
