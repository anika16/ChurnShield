import pandas as pd
import numpy as np

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['avg_charge_per_month'] = df.apply(
        lambda r: (r['TotalCharges'] / r['tenure']) if (r.get('tenure',0) and r.get('TotalCharges', None) is not None and r['tenure']>0) else r.get('MonthlyCharges', 0),
        axis=1
    )
    bins = [0, 12, 24, 36, 48, 60, 72, np.inf]
    labels = ['0-12','13-24','25-36','37-48','49-60','61-72','72+']
    if 'tenure' in df.columns:
        df['tenure_bucket'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True)
    svc_cols = ['PhoneService','InternetService','OnlineSecurity','OnlineBackup',
                'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']
    def count_services(row):
        c = 0
        for col in svc_cols:
            if col in row.index:
                val = row[col]
                if isinstance(val, str) and val.lower() not in ('no','no internet service','none','0','nan'):
                    c += 1
                elif pd.notna(val) and (not isinstance(val, str)) and val != 0:
                    c += 1
        return c
    df['num_services'] = df.apply(count_services, axis=1)
    return df
