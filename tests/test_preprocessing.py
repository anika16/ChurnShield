# chapter5_statistical_tests.py
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import os

DATA_CLEAN = "data/processed/telco_processed.csv"

def load():
    df = pd.read_csv(DATA_CLEAN)
    return df

def t_tests(df):
    churn = df[df["Churn_flag"]==1]
    not_churn = df[df["Churn_flag"]==0]

    numeric_vars = ["tenure", "MonthlyCharges", "TotalCharges", "avg_charge_per_month"]
    print("=== Independent t-tests (Churn vs Non-Churn) ===")
    for col in numeric_vars:
        tstat, pval = ttest_ind(churn[col], not_churn[col], equal_var=False)
        print(f"{col}: t-stat = {tstat:.4f}, p-value = {pval:.4e}")

def chi_square_tests(df):
    categorical_vars = ["Contract", "PaymentMethod", "InternetService", "tenure_bucket"]
    print("\n=== Chi-square tests (Categorical) ===")
    for col in categorical_vars:
        contingency = pd.crosstab(df[col], df["Churn_flag"])
        chi2, p, dof, ex = chi2_contingency(contingency)
        print(f"{col}: chi2 = {chi2:.4f}, p-value = {p:.4e}, dof = {dof}")

def main():
    if not os.path.exists(DATA_CLEAN):
        raise FileNotFoundError(f"{DATA_CLEAN} not found. Run chapter4_data_analysis.py first.")
    df = load()
    t_tests(df)
    chi_square_tests(df)

if __name__ == "__main__":
    main()
