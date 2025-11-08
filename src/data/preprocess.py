from typing import Tuple, List
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from ..config import PREPROCESSOR_PATH, PROCESSED_DIR, RANDOM_STATE, TEST_SIZE, VAL_SIZE
from loguru import logger

def basic_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    return df

def build_preprocessor(df: pd.DataFrame):
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in df.columns if c not in num_cols + ['customerID', 'Churn']]
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
    cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])    
    preprocessor = ColumnTransformer([('num', num_pipe, num_cols),('cat', cat_pipe, cat_cols)])
    return preprocessor, num_cols, cat_cols

def preprocess_and_split(df_raw: pd.DataFrame, save: bool = True):
    df = basic_cleanup(df_raw.copy())
    if 'TotalCharges' in df.columns:
        before = df.shape[0]
        df = df[~df['TotalCharges'].isna()]
        after = df.shape[0]
        if before != after:
            logger.warning(f"Dropped {before-after} rows with missing TotalCharges after coercion")
    if 'customerID' in df.columns:
        df_model = df.drop(columns=['customerID'])
    else:
        df_model = df.copy()
    y = (df_model['Churn'] == 'Yes').astype(int)
    X = df_model.drop(columns=['Churn'])
    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_SIZE, stratify=y_train_val, random_state=RANDOM_STATE)
    preprocessor.fit(X_train)
    if save:
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        X_train.assign(Churn=y_train).to_csv(PROCESSED_DIR / "train.csv", index=False)
        X_val.assign(Churn=y_val).to_csv(PROCESSED_DIR / "val.csv", index=False)
        X_test.assign(Churn=y_test).to_csv(PROCESSED_DIR / "test.csv", index=False)
    logger.info("Preprocessing complete and artifacts saved")
    meta = { "num_cols": num_cols, "cat_cols": cat_cols }
    return preprocessor, (X_train, X_val, X_test, y_train, y_val, y_test), meta
