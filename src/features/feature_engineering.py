
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = "data/raw/Telco-Customer-Churn.csv"

def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    # replace empty strings with NaN and convert TotalCharges to numeric
    df.replace(" ", np.nan, inplace=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def feature_engineer(df):
    # Derived features
    service_features = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                        "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[service_features].apply(lambda x: (x == "Yes").sum(), axis=1)
    df["avg_charge_per_month"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    df["tenure_bucket"] = pd.cut(df["tenure"],
                                 bins=[-1,12,24,48,df["tenure"].max()],
                                 labels=["0-1 year","1-2 years","2-4 years","4+ years"])
    # Ensure Churn is binary mapping for later chapters
    df["Churn_flag"] = df["Churn"].map({"Yes":1, "No":0})
    return df

def descriptive_stats(df):
    print("\n=== DESCRIPTIVE STATISTICS (NUMERIC) ===\n")
    print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe().T)

def eda_plots(df, show=True, save_dir="outputs/figures"):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(6,4))
    sns.countplot(x="Churn", data=df)
    plt.title("Churn Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "churn_distribution.png"))
    if show: plt.show()
    plt.close()

    plt.figure(figsize=(7,4))
    sns.histplot(df["tenure"], kde=True)
    plt.title("Tenure Distribution")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "tenure_distribution.png"))
    if show: plt.show()
    plt.close()

    plt.figure(figsize=(7,4))
    sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
    plt.title("Monthly Charges vs Churn")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "monthlycharges_by_churn.png"))
    if show: plt.show()
    plt.close()

    plt.figure(figsize=(8,4))
    sns.countplot(x="Contract", hue="Churn", data=df)
    plt.title("Churn by Contract Type")
    plt.tight_layout(); plt.savefig(os.path.join(save_dir, "churn_by_contract.png"))
    if show: plt.show()
    plt.close()

def main():
    df = load_and_clean()
    df = feature_engineer(df)
    descriptive_stats(df)
    eda_plots(df)
    # Save cleaned dataset for next chapters
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/telco_processed.csv", index=False)
    print("\nSaved processed dataset to data/processed/telco_processed.csv")

if __name__ == "__main__":
    main()
