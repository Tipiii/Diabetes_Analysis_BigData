from pathlib import Path
import joblib
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ml" / "linear_model.pkl"
COLS_PATH = BASE_DIR / "ml" / "feature_cols.pkl"

_model = None
_cols = None

def load_artifacts():
    global _model, _cols
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _cols is None:
        _cols = joblib.load(COLS_PATH)
    return _model, _cols

def predict_risk_score(input_dict: dict) -> float:
    model, cols = load_artifacts()
    df = pd.DataFrame([input_dict])[cols]
    pred = model.predict(df)
    return float(np.ravel(pred)[0])
