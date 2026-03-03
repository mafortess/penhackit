
from pathlib import Path

import joblib
import json

def load_decision_model(model_path: Path, metrics_path: Path):
    # model = joblib.load(model_path) # MAL; DA ERROR
    # with open(model_path, "r", encoding="utf-8") as f:
    #     model = json.load(f)
    try:
        model = joblib.load(model_path)  # CORRECTO; el modelo se carga correctamente usando joblib.load() en lugar de json.load(), ya que el modelo está serializado en formato joblib, no JSON.
    except Exception as e:
        print(f"Error loading model with joblib: {e}")
        raise e
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]
    except Exception as e:
        print(f"Error loading feature names from metrics: {e}")
        raise e
    
    return model, feature_names
