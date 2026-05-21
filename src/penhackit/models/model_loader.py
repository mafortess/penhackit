
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

def load_decision_model_bundle(models_dir: Path, model_id: str) -> dict:
    model_dir = models_dir / model_id
    model_path = model_dir / "model.joblib"
    metrics_path = model_dir / "metrics.json"

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    model, feature_names = load_decision_model(
        model_path=model_path,
        metrics_path=metrics_path,
    )

    return {
        "model_id": model_id,
        "model_dir": model_dir,
        "model_path": model_path,
        "metrics_path": metrics_path,
        "model": model,
        "feature_names": feature_names,
    }


def list_available_models(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []

    available_models = []

    for item in models_dir.iterdir():
        if not item.is_dir():
            continue

        model_file = item / "model.joblib"
        metrics_file = item / "metrics.json"

        if model_file.exists() and metrics_file.exists():
            available_models.append(item.name)

    return sorted(available_models)


def model_exists(models_dir: Path, model_id: str) -> bool:
    model_dir = models_dir / model_id
    return (
        model_dir.exists()
        and (model_dir / "model.joblib").exists()
        and (model_dir / "metrics.json").exists()
    )


def get_model_dir(models_dir: Path, model_id: str) -> Path:
    return models_dir / model_id
