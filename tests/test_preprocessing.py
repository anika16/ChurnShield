import pandas as pd
from src.data.preprocess import basic_cleanup, build_preprocessor

def test_basic_cleanup_totalcharges():
    df = pd.DataFrame({
        'TotalCharges': ['100.5', ' 200 ', 'bad', ''],
        'tenure': [1,2,3,4],
        'MonthlyCharges': [10,20,30,40],
        'Churn': ['No','Yes','No','No']
    })
    df2 = basic_cleanup(df)
    assert 'TotalCharges' in df2.columns

def test_build_preprocessor_cols():
    df = pd.DataFrame({
        'tenure':[1],
        'MonthlyCharges':[20.0],
        'TotalCharges':[20.0],
        'gender':['Male'],
        'Churn':['No']
    })
    pre, num_cols, cat_cols = build_preprocessor(df)
    assert 'tenure' in num_cols
