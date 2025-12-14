import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import shap
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. Load & Clean Data
# ------------------------------------------------------------
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
df.replace(" ", np.nan, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(subset=["TotalCharges"], inplace=True)

# ------------------------------------------------------------
# 2. Feature Engineering
# ------------------------------------------------------------
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]

df["num_services"] = df[service_cols].apply(lambda x: (x == "Yes").sum(), axis=1)
df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

df["tenure_bucket"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, df["tenure"].max()],
    labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"]
)

# ------------------------------------------------------------
# 3. Prepare Data
# ------------------------------------------------------------
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].map({"Yes": 1, "No": 0})

num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# ------------------------------------------------------------
# 4. Logistic Regression (Final Model)
# ------------------------------------------------------------
lr_model = Pipeline([
    ("prep", preprocessor),
    ("lr", LogisticRegression(max_iter=500))
])

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_prob = lr_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression ===")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Precision:", precision_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("F1:", f1_score(y_test, lr_pred))
print("AUC:", roc_auc_score(y_test, lr_prob))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

# ------------------------------------------------------------
# 5. Logistic Regression Feature Importance
# ------------------------------------------------------------
ohe = lr_model.named_steps["prep"].named_transformers_["cat"]
encoded_feature_names = list(ohe.get_feature_names_out(cat_cols))
all_features = list(num_cols) + encoded_feature_names

coefficients = lr_model.named_steps["lr"].coef_[0]

lr_importance = pd.DataFrame({
    "Feature": all_features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nTop Logistic Regression Features:\n")
print(lr_importance.head(20))


# ------------------------------------------------------------
# 6. SHAP for Logistic Regression
# ------------------------------------------------------------
explainer = shap.LinearExplainer(lr_model.named_steps["lr"],
                                 lr_model.named_steps["prep"].transform(X_train))

shap_values = explainer.shap_values(lr_model.named_steps["prep"].transform(X_test))

shap.summary_plot(shap_values, feature_names=all_features)

# ------------------------------------------------------------
# 7. Random Forest & XGBoost for Comparison (Fixed)
# ------------------------------------------------------------

# Use the same preprocessing transformation already fitted
X_train_trans = lr_model.named_steps["prep"].transform(X_train)
X_test_trans = lr_model.named_steps["prep"].transform(X_test)

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_trans, y_train)

rf_pred = rf.predict(X_test_trans)
rf_prob = rf.predict_proba(X_test_trans)[:, 1]

print("\n=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1:", f1_score(y_test, rf_pred))
print("AUC:", roc_auc_score(y_test, rf_prob))


# ------------------------------------------------------------
# XGBoost
# ------------------------------------------------------------
xgb = XGBClassifier(
    n_estimators=300, learning_rate=0.1, max_depth=6,
    subsample=0.9, colsample_bytree=0.9,
    eval_metric="logloss", random_state=42
)

xgb.fit(X_train_trans, y_train)

xgb_pred = xgb.predict(X_test_trans)
xgb_prob = xgb.predict_proba(X_test_trans)[:, 1]

print("\n=== XGBoost ===")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Precision:", precision_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("F1:", f1_score(y_test, xgb_pred))
print("AUC:", roc_auc_score(y_test, xgb_prob))


import joblib
import os
from ..config import PREPROCESSOR_PATH, MODEL_PATH


# Create artifact directory
os.makedirs("data/artifacts", exist_ok=True)

# Extract objects from your pipeline
preprocessor = lr_model.named_steps["prep"]
final_model = lr_model.named_steps["lr"]


# Save artifacts
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump(final_model, MODEL_PATH)

print("Preprocessor saved to:", PREPROCESSOR_PATH)
print("Logistic Regression model saved to:", MODEL_PATH)