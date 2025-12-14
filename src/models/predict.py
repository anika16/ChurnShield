import joblib
import pandas as pd
from config import PREPROCESSOR_PATH, MODEL_PATH

def predict_single(sample: dict):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([sample])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0,1] if hasattr(model, "predict_proba") else None
    pred = int(model.predict(X)[0])
    return {"prediction": bool(pred), "probability": float(prob) if prob is not None else None}
