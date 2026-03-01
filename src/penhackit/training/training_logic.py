from pathlib import Path
import json
import time

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib
from collections import Counter

from penhackit.common.paths import Paths

MODEL_CHOICES = {
    "logreg": ("logreg", "Logistic Regression (multinomial)", lambda: LogisticRegression(max_iter=2000)),
    "decision_tree": ("decision_tree", "Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
    "random_forest": ("random_forest", "Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
    "mlp": ("mlp", "MLP (2 hidden layers)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)),
}

def training_model(training_settings: dict, dataset_path: Path, model_key: str, model_factory, paths: Paths) -> None:
    """
    Interactive training:       
    - select dataset (from dataset_path)
    - select model type (from MODEL_CHOICES)
    - train + evaluate
    - save model + metrics under models_dir/<dataset>/<model>_<n>/
    """
    # Convert dataset_path to Path if it's a string
    if isinstance(dataset_path, str):
       dataset_path = Path(dataset_path)
    
    print(f"Training settings:")
    for k, v in training_settings.items():
        print(f"  {k}: {v}")
    
    print(f"\nSelected dataset: {dataset_path}")
    
    # Load dataset
    try:
        rows = load_dataset_jsonl(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Vectorize dataset
    try:
        X, y, feature_names = vectorize_bc_rows(rows)
    except Exception as e:
        print(f"Error vectorizing dataset: {e}")
        return
    
    # split dataset
    counts = Counter(y.tolist())
    min_count = min(counts.values())

    strat = y if len(set(y.tolist())) > 1 and min_count >= 2 else None

    # split with stratify if we have at least 2 samples in each class, otherwise just split without stratify
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=strat
        )
    except Exception as e:
        print(f"Error during train/test split: {e}")
        return
    
    model = model_factory()
    
    # Train the model
    print("\nTraining...")
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return
    # Evaluate the model
    print("Evaluating...")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return
    
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # output dir
    print(f"Saving trained models to: {paths.models_dir}")

    models_dir = paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    out_parent = models_dir
    out_parent.mkdir(parents=True, exist_ok=True)

    # out_dir = next_available_path(out_parent, f"{model_key}", "")  # placeholder
    # next_available_path expects ext; we'll just do our own for dirs:
    # create model_key, model_key_1, ...
    if (out_parent / model_key).exists():
        i = 1
        while (out_parent / f"{model_key}_{i}").exists():
            i += 1
        out_dir = out_parent / f"{model_key}_{i}"
    else:
        out_dir = out_parent / model_key

    out_dir.mkdir(parents=True, exist_ok=False)

    # save
    print(f"\nSaving model and metrics to: {out_dir} ...")
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "schema": "penhackit.training.v1",
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(dataset_path),
        "model_type": model_key,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": rep,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Output dir: {out_dir}")


def vectorize_dataset(rows: list[dict]):
    all_keys = set()
    for r in rows:
        x = r.get("x") or {}
        all_keys.update(x.keys())
    feature_names = sorted(all_keys)

    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        x = r.get("x") or {}
        for j, k in enumerate(feature_names):
            X[i, j] = float(x.get(k, 0.0))
        y[i] = int(r.get("y"))

    return X, y, feature_names

def load_dataset_jsonl(dataset_path: Path | str) -> list[dict]:
       
    print(f"Loading dataset: {dataset_path} ...")
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    rows = []
    try:
        with dataset_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line) # expected: {"schema_id":..., "t":..., "state":{...}, "action_id":int}
                rows.append(obj)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in dataset file: {dataset_path} at line {e.lineno}: {e.msg}")
    if not rows:
        raise RuntimeError("Dataset is empty.")
    return rows


def load_dataset_jsonl_dir(dataset_dir: Path) -> list[dict]:
    # Show dataset options in this directort (dataset_dir)
    jsonl_files = sorted(
        [p.name for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"],
        key=str.lower,
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in: {dataset_dir}")
    
    dataset_choice = prompt(f"Load dataset from: {dataset_dir} > ", completer=WordCompleter(jsonl_files))

    
    path = dataset_dir / dataset_choice
    if not path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found: {path}")

    rows = []
    print(f"Loading dataset: {path} ...")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # expected: {"schema_id":..., "t":..., "state":{...}, "action_id":int}
            rows.append(obj)
    if not rows:
        raise RuntimeError("Dataset is empty.")
    return rows

def vectorize_bc_rows(rows: list[dict]):
    # Collect all keys from "state"
    keys = set()
    for r in rows:
        s = r.get("state") or {}
        if not isinstance(s, dict):
            raise TypeError("Each row must contain a dict field 'state'.")
        keys.update(s.keys())

    feature_names = sorted(keys)
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        s = r.get("state") or {}
        for j, k in enumerate(feature_names):
            v = s.get(k, 0)
            if isinstance(v, bool):
                v = 1 if v else 0
            if v is None:
                v = 0
            if not isinstance(v, (int, float)):
                raise TypeError(f"Non-numeric feature in state: key={k} value={v!r}")
            X[i, j] = float(v)

        if "action_id" not in r:
            raise KeyError("Missing 'action_id' in dataset row.")
        y[i] = int(r["action_id"])

    return X, y, feature_names


# =========================
# Training (single function)
# =========================
def next_available_path(dirpath: Path, base_name: str, ext: str) -> Path:
    """
    Returns a non-existing path in dirpath.
    Example: base_name="report", ext=".md" -> report.md, report_1.md, report_2.md, ...
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    p0 = dirpath / f"{base_name}{ext}"
    if not p0.exists():
        return p0

    i = 1
    while True:
        pi = dirpath / f"{base_name}_{i}{ext}"
        if not pi.exists():
            return pi
        i += 1

# DEPRECATED VERSION
def train_models_from_dataset(dataset_dir: Path, models_dir: Path) -> Path:
    rows = load_dataset_jsonl(dataset_dir)
    X, y, feature_names = vectorize_dataset(rows)

    model_id = f"models_{dataset_dir.name}"
    out_dir = models_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }

    metrics = {
        "dataset_dir": str(dataset_dir),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }

    for name, model in models.items():
        print(f"\n[train] {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"[eval] {name} accuracy: {acc:.4f}")

        model_path = out_dir / f"{name}.joblib"
        joblib.dump(model, model_path)

        metrics["models"][name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "model_path": str(model_path),
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved models to: {out_dir}")
    return out_dir
