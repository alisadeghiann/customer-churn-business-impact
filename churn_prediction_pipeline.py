import pandas as pd
import pyodbc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np
import pickle
import os

# ============================
# 0) Desktop Path
# ============================
desktop = r"C:\Users\Ali\Desktop"

# ============================
# 1) Load Data From SQL
# ============================
conn = pyodbc.connect(
    "DRIVER={SQL Server};"
    "SERVER=localhot\\SQL2025;"
    "DATABASE=TelcoDBB;"
    "UID=;"
    "PWD=
)

df = pd.read_sql("SELECT * FROM TelcoClean", conn)
df_raw = df.copy()

# ============================
# 2) Clean + Encode
# ============================
df_clean = df.dropna()
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# Save Clean Data
clean_path = os.path.join(desktop, "clean_data_FINAL.csv")
df_encoded.to_csv(clean_path, index=False)
print("✔ clean_data_FINAL.csv ساخته شد")

X = df_encoded.drop(columns=['Churn'])
y = df_encoded['Churn']

# ============================
# 3) Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 4) Define Models
# ============================
lr_plain = LogisticRegression(max_iter=2000)

pipe_lr_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', LogisticRegression(max_iter=2000))
])

lr_weighted = LogisticRegression(max_iter=2000, class_weight='balanced')

pipe_rf_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(
        n_estimators=300, max_depth=10,
        random_state=42, max_features='sqrt'
    ))
])

pipe_xgb_smote = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', XGBClassifier(
        n_estimators=300, max_depth=6,
        learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, objective='binary:logistic',
        eval_metric='logloss', random_state=42
    ))
])

# ============================
# 5) Fit Models
# ============================
def fit_and_predict(model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    return pred, prob

lr_plain_pred, lr_plain_prob = fit_and_predict(lr_plain)
lr_smote_pred, lr_smote_prob = fit_and_predict(pipe_lr_smote)
lr_weighted_pred, lr_weighted_prob = fit_and_predict(lr_weighted)
rf_smote_pred, rf_smote_prob = fit_and_predict(pipe_rf_smote)
xgb_smote_pred, xgb_smote_prob = fit_and_predict(pipe_xgb_smote)

# ============================
# 6) Metrics
# ============================
def compute_metrics(name, y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_prob)
    business_score = 0.4 * rec + 0.3 * f1 + 0.3 * pr_auc
    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "PR_AUC": pr_auc,
        "BusinessScore": business_score
    }

results = [
    compute_metrics("LR Plain", y_test, lr_plain_pred, lr_plain_prob),
    compute_metrics("LR + SMOTE", y_test, lr_smote_pred, lr_smote_prob),
    compute_metrics("LR Weighted", y_test, lr_weighted_pred, lr_weighted_prob),
    compute_metrics("RF + SMOTE", y_test, rf_smote_pred, rf_smote_prob),
    compute_metrics("XGB + SMOTE", y_test, xgb_smote_pred, xgb_smote_prob)
]

results_df = pd.DataFrame(results)
print(results_df)

# ============================
# 7) Best Model
# ============================
best_model_name = results_df.sort_values("BusinessScore", ascending=False).iloc[0]["Model"]

if best_model_name == "LR Plain":
    final_model = lr_plain
elif best_model_name == "LR + SMOTE":
    final_model = pipe_lr_smote
elif best_model_name == "LR Weighted":
    final_model = lr_weighted
elif best_model_name == "RF + SMOTE":
    final_model = pipe_rf_smote
else:
    final_model = pipe_xgb_smote

# ============================
# 8) Test Predictions Output
# ============================
test_prob = final_model.predict_proba(X_test)[:, 1]
test_pred = final_model.predict(X_test)

test_output = pd.DataFrame({
    "Actual": y_test,
    "Predicted": test_pred,
    "Probability": test_prob
})

test_output_path = os.path.join(desktop, "model_output_FINAL.csv")
test_output.to_csv(test_output_path, index=False)
print("✔ model_output_FINAL.csv ساخته شد")

# ============================
# 9) Threshold + Business Impact
# ============================
thresholds = np.arange(0.3, 0.81, 0.05)
offer_cost_scenarios = {"Low": 10, "Base": 20, "High": 30}

th_results = []

for th in thresholds:
    y_pred_th = (test_prob >= th).astype(int)
    cm_th = confusion_matrix(y_test, y_pred_th)
    tn, fp, fn, tp = cm_th.ravel()

    rec = recall_score(y_test, y_pred_th)
    prec = precision_score(y_test, y_pred_th)
    f1 = f1_score(y_test, y_pred_th)

    revenue_at_risk = df_raw.loc[y_test.index[y_test == 1], "MonthlyCharges"].sum()

    for scenario, cost in offer_cost_scenarios.items():
        campaign_cost = fp * cost
        retention_value = tp * df_raw.loc[y_test.index[y_test == 1], "MonthlyCharges"].mean() * 0.20
        net_benefit = retention_value - campaign_cost

        th_results.append({
            "Threshold": th,
            "Scenario": scenario,
            "OfferCost": cost,
            "Recall": rec,
            "Precision": prec,
            "F1": f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "RevenueAtRisk": revenue_at_risk,
            "RetentionValue": retention_value,
            "CampaignCost": campaign_cost,
            "NetBenefit": net_benefit
        })

th_df = pd.DataFrame(th_results)

th_path = os.path.join(desktop, "threshold_business_FINAL.csv")
th_df.to_csv(th_path, index=False)
print("✔ threshold_business_FINAL.csv ساخته شد")

# ============================
# 10) Full Predictions For Power BI
# ============================
final_model.fit(X, y)
full_prob = final_model.predict_proba(X)[:, 1]
full_pred = final_model.predict(X)

df_output = df_raw.copy()
df_output["PredictedChurn"] = full_pred
df_output["ChurnProbability"] = full_prob

def risk(p):
    if p >= 0.8:
        return "High"
    elif p >= 0.5:
        return "Medium"
    else:
        return "Low"

df_output["RiskLevel"] = df_output["ChurnProbability"].apply(risk)

full_path = os.path.join(desktop, "powerbi_predictions_FINAL.csv")
df_output.to_csv(full_path, index=False)
print("✔ powerbi_predictions_FINAL.csv ساخته شد")

# ============================
# 11) Save Final Model
# ============================
model_path = os.path.join(desktop, "best_model_FINAL.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_model, f)

print("✔ best_model_FINAL.pkl ذخیره شد")
