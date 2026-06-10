from pathlib import Path
import json
import csv
import time
from time import perf_counter

import numpy as np

from penhackit.training.vectorization import vectorize_bc_rows
from penhackit.training.training_storage import load_dataset_jsonl,create_model_output_dir, write_offline_comparison_csv, save_json, save_confusion_matrix_csv, save_confusion_matrix_png, save_predictions_csv, append_offline_comparison_row

import joblib
from collections import Counter

from penhackit.common.paths import Paths

# Problema con esta estructura: carga/importa los archivos enormes de sklearn incluso si solo quieres usar la parte de report generation, que no tiene nada que ver con sklearn.
# Solución: lazy import dentro de la función de training, así solo se cargan esos módulos si realmente se va a usar la parte de training. 
# Está acoplado, depende de sklearn
# MODEL_CHOICES = {
#     "logreg": ("logreg", "Logistic Regression (multinomial)", lambda: LogisticRegression(max_iter=2000)),
#     "decision_tree": ("decision_tree", "Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
#     "random_forest": ("random_forest", "Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
#     "mlp": ("mlp", "MLP (2 hidden layers)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)),
# }

MODEL_CHOICES = {
    # Modelos baseline:
    "decision_tree": ("decision_tree", "Decision Tree"),
    "random_forest": ("random_forest", "Random Forest"),
    
    # Modelos modernos de boosting para datos tabulares
    "catboost": ("catboost", "CatBoost Classifier"),
    "lightgbm": ("lightgbm", "LightGBM Classifier"),
    "xgboost": ("xgboost", "XGBoost Classifier"),
    
    # "mlp": ("mlp", "MLP (2 hidden layers)"),
    # "logreg": ("logreg", "Logistic Regression (multinomial)"),
}


def training_model(training_settings: dict, dataset_path: Path, model_key: str, paths: Paths) -> None:
    """
    Interactive training:       
    - select dataset (from dataset_path)
    - select model type (from MODEL_CHOICES)
    - train + evaluate
    - save model + metrics under models_dir/<dataset>/<model>_<n>/
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split

    # Convert dataset_path to Path if it's a string
    if isinstance(dataset_path, str):
       dataset_path = Path(dataset_path)
    
    print(f"Training settings:")
    for k, v in training_settings.items():
        print(f"  {k}: {v}")
    
    print(f"\nSelected dataset: {dataset_path}")
    
    # ==============================================================
    # Load dataset
    try:
        rows = load_dataset_jsonl(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # ==============================================================
    # Vectorize dataset
    try:
        X, y, feature_names = vectorize_bc_rows(rows)
    except Exception as e:
        print(f"Error vectorizing dataset: {e}")
        return
    
    # ==============================================================
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
    
    # ==============================================================
    # Create model instance
    print(f"Creating model instance for: {model_key} ...")
    model_factory = get_model_factory(model_key)
    model = model_factory()
    

    # ==============================================================
    # Prepare output dir before saving evaluation artifacts (output dir)
    print(f"Saving trained models to: {paths.models_dir}")

    models_dir = paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    out_dir = create_model_output_dir(models_dir, model_key)

    # out_parent = models_dir
    # out_parent.mkdir(parents=True, exist_ok=True)

    # # out_dir = next_available_path(out_parent, f"{model_key}", "")  # placeholder
    # # next_available_path expects ext; we'll just do our own for dirs:
    # # create model_key, model_key_1, ...
    # if (out_parent / model_key).exists():
    #     i = 1
    #     while (out_parent / f"{model_key}_{i}").exists():
    #         i += 1
    #     out_dir = out_parent / f"{model_key}_{i}"
    # else:
    #     out_dir = out_parent / model_key

    # out_dir.mkdir(parents=True, exist_ok=False)


    # ==============================================================
    # Train the model
    print(f"Training model: {model_key} ...")
    try:
        train_start = perf_counter()
        model.fit(X_train, y_train)
        training_time_seconds = perf_counter() - train_start
    except Exception as e:
        print(f"Error during model training: {e}")
        return

    # ==============================================================
    # Evaluate the model
    print("\nEvaluating model on test set ...")
    try:
        inference_start = perf_counter()
        y_pred = model.predict(X_test)
        y_pred = normalize_label_vector(y_pred)
        y_test = normalize_label_vector(y_test)
        inference_time_seconds = perf_counter() - inference_start
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return

    # ==============================================================
    # Calculate metrics
    try:
        y = normalize_label_vector(y)
        y_train = normalize_label_vector(y_train)
        y_test = normalize_label_vector(y_test)
        y_pred = normalize_label_vector(y_pred)

        labels = sorted(int(v) for v in set(y.tolist()))

        metrics = build_offline_metrics(model_key=model_key, dataset_path=dataset_path, X=X, y=y,
            X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, y_pred=y_pred, 
            feature_names=feature_names, labels=labels, training_time_seconds=training_time_seconds, 
            inference_time_seconds=inference_time_seconds,
        )
    except Exception as e:
        print(f"Error during offline metrics calculation: {e}")
        print(f"type(y): {type(y)}")
        print(f"type(y_test): {type(y_test)}")
        print(f"type(y_pred): {type(y_pred)}")
        try:
            print(f"np.asarray(y).shape: {np.asarray(y, dtype=object).shape}")
            print(f"np.asarray(y_test).shape: {np.asarray(y_test, dtype=object).shape}")
            print(f"np.asarray(y_pred).shape: {np.asarray(y_pred, dtype=object).shape}")
            print(f"y_pred sample: {np.asarray(y_pred, dtype=object).reshape(-1)[:5].tolist()}")
        except Exception:
            pass
        return

    print_offline_metrics_summary(metrics)
    # ==============================================================
    # # insuficientes métricas, añadir más: precision, recall, f1 (macro y weighted), support, etc.
    # acc = float(accuracy_score(y_test, y_pred))
    # cm = confusion_matrix(y_test, y_pred).tolist()
    # rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # print(f"Accuracy: {acc:.4f}")
    # print("Confusion matrix:")
    # print(cm)

    # metrics = {
    #     "schema": "penhackit.training.v1",
    #     "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    #     "dataset_path": str(dataset_path),
    #     "model_type": model_key,
    #     "n_samples": int(len(y)),
    #     "n_features": int(X.shape[1]),
    #     "feature_names": feature_names,
    #     "accuracy": acc,
    #     "confusion_matrix": cm,
    #     "classification_report": rep,
    # }
    # (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    # ==============================================================

    # ==============================================================
    # Save model
    print(f"\nSaving model and metrics to: {out_dir} ...")
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics["model_path"] = str(model_path)
    metrics["output_dir"] = str(out_dir)

    # ==============================================================
    # Save evaluation artifacts
    save_json(out_dir / "metrics.json", metrics)
    save_json(out_dir / "classification_report.json", metrics["classification_report"])
    save_confusion_matrix_csv(
        out_dir / "confusion_matrix.csv",
        metrics["labels"],
        metrics["confusion_matrix"],
    )
    save_confusion_matrix_png(
        out_dir / "confusion_matrix.png",
        metrics["labels"],
        metrics["confusion_matrix"],
        title=f"Confusion matrix - {model_key}",
    )
    save_predictions_csv(
        out_dir / "predictions.csv",
        y_test=y_test,
        y_pred=y_pred,
    )
    append_offline_comparison_row(
        models_dir / "offline_model_comparison.csv",
        metrics,
    )
    
    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved classification report: {out_dir / 'classification_report.json'}")
    print(f"Saved confusion matrix CSV: {out_dir / 'confusion_matrix.csv'}")
    print(f"Saved confusion matrix PNG: {out_dir / 'confusion_matrix.png'}")
    print(f"Saved predictions: {out_dir / 'predictions.csv'}")
    print(f"Updated comparison CSV: {models_dir / 'offline_model_comparison.csv'}")
    print(f"Output dir: {out_dir}")


# ===========================================================================
# Helper functions for training
# ===========================================================================

def get_model_factory(model_key: str):
    if model_key == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier
        return lambda: DecisionTreeClassifier(
            random_state=42)

    elif model_key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return lambda: RandomForestClassifier(
            n_estimators=200, 
            random_state=42)

    if model_key == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError(
                "CatBoost is not installed. Install it with: pip install catboost"
            ) from e

        return lambda: CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            loss_function="MultiClass",
            random_seed=42,
            verbose=False,
        )

    if model_key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as e:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            ) from e

        return lambda: LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
            class_weight="balanced",
        )

    if model_key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "XGBoost is not installed. Install it with: pip install xgboost"
            ) from e

        return lambda: XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
        )

    elif model_key == "logreg":
        from sklearn.linear_model import LogisticRegression
        return lambda: LogisticRegression(max_iter=2000)

    elif model_key == "mlp":
        from sklearn.neural_network import MLPClassifier
        return lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)

    else:
        raise ValueError(f"Unknown model_key: {model_key}")
    



def build_offline_metrics(model_key: str, dataset_path: Path, X, y, X_train, X_test, y_train, y_test, y_pred,
    feature_names: list[str], labels: list[int], training_time_seconds: float, inference_time_seconds: float,) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y = normalize_label_vector(y)
    y_train = normalize_label_vector(y_train)
    y_test = normalize_label_vector(y_test)
    y_pred = normalize_label_vector(y_pred)

    y_list = [int(v) for v in y.tolist()]
    y_train_list = [int(v) for v in y_train.tolist()]
    y_test_list = [int(v) for v in y_test.tolist()]
    y_pred_list = [int(v) for v in y_pred.tolist()]

    cm = confusion_matrix(y_test, y_pred, labels=labels).tolist()

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    accuracy = float(accuracy_score(y_test, y_pred))

    precision_macro = float(precision_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=labels, average="weighted", zero_division=0))

    class_distribution = {str(k): int(v) for k, v in sorted(Counter(y_list).items())}
    train_class_distribution = {str(k): int(v) for k, v in sorted(Counter(y_train_list).items())}
    test_class_distribution = {str(k): int(v) for k, v in sorted(Counter(y_test_list).items())}

    return {
        "schema": "penhackit.offline_evaluation.v1",
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(dataset_path),
        "dataset_name": dataset_path.name,
        "model_type": model_key,

        "n_samples": int(len(y)),
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(labels)),

        "labels": labels,
        "feature_names": feature_names,
        "class_distribution": class_distribution,
        "train_class_distribution": train_class_distribution,
        "test_class_distribution": test_class_distribution,

        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,

        "training_time_seconds": float(training_time_seconds),
        "inference_time_seconds": float(inference_time_seconds),
        "avg_inference_time_seconds": float(
            inference_time_seconds / len(y_test) if len(y_test) > 0 else 0.0
        ),

        "confusion_matrix": cm,
        "classification_report": report,

        "y_test": y_test_list,
        "y_pred": y_pred_list,
    }


def print_offline_metrics_summary(metrics: dict) -> None:
    print("\nOffline evaluation summary")
    print("--------------------------")
    print(f"Model: {metrics['model_type']}")
    print(f"Dataset: {metrics['dataset_name']}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"Train samples: {metrics['n_train_samples']}")
    print(f"Test samples: {metrics['n_test_samples']}")
    print(f"Features: {metrics['n_features']}")
    print(f"Classes: {metrics['n_classes']}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision macro: {metrics['precision_macro']:.4f}")
    print(f"Recall macro: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Training time: {metrics['training_time_seconds']:.4f} s")
    print(f"Inference time: {metrics['inference_time_seconds']:.6f} s")
    print(f"Average inference time: {metrics['avg_inference_time_seconds']:.8f} s/sample")
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])



def evaluate_saved_models(paths: Paths) -> None:
    """
    Reads saved metrics.json files from models_dir and builds an offline
    comparison CSV without retraining models.
    """
    models_dir = paths.models_dir

    if not models_dir.exists():
        print(f"Models directory does not exist: {models_dir}")
        return

    metric_files = sorted(models_dir.glob("**/metrics.json"))

    if not metric_files:
        print(f"No metrics.json files found under: {models_dir}")
        return

    rows = []

    for metrics_path in metric_files:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping invalid metrics file {metrics_path}: {e}")
            continue

        row = {
            "trained_at_utc": metrics.get("trained_at_utc", ""),
            "dataset_name": metrics.get("dataset_name", Path(metrics.get("dataset_path", "")).name),
            "dataset_path": metrics.get("dataset_path", ""),
            "run_id": metrics.get("run_id", metrics_path.parent.name),
            "model_dir": metrics.get("model_dir", str(metrics_path.parent.parent)),
            "model_type": metrics.get("model_type", ""),
            "n_samples": metrics.get("n_samples", ""),
            "n_train_samples": metrics.get("n_train_samples", ""),
            "n_test_samples": metrics.get("n_test_samples", ""),
            "n_features": metrics.get("n_features", ""),
            "n_classes": metrics.get("n_classes", ""),
            "accuracy": metrics.get("accuracy", ""),
            "precision_macro": metrics.get("precision_macro", ""),
            "recall_macro": metrics.get("recall_macro", ""),
            "macro_f1": metrics.get("macro_f1", ""),
            "weighted_f1": metrics.get("weighted_f1", ""),
            "training_time_seconds": metrics.get("training_time_seconds", ""),
            "inference_time_seconds": metrics.get("inference_time_seconds", ""),
            "avg_inference_time_seconds": metrics.get("avg_inference_time_seconds", ""),
            "output_dir": str(metrics_path.parent),
            "model_path": metrics.get("model_path", str(metrics_path.parent / "model.joblib")),
        }
        rows.append(row)

    if not rows:
        print("No valid metrics found.")
        return

    comparison_path = models_dir / "offline_model_comparison.csv"
    write_offline_comparison_csv(comparison_path, rows)

    rows_sorted = sorted(
        rows,
        key=lambda r: float(r["macro_f1"]) if r["macro_f1"] != "" else -1.0,
        reverse=True,
    )

    print("\nOffline model comparison")
    print("------------------------")
    print(f"{'Model':<18} {'Accuracy':>10} {'Macro F1':>10} {'Weighted F1':>12} {'Train(s)':>10} {'Infer(s)':>10}")

    for row in rows_sorted:
        print(
            f"{row['model_type']:<18} "
            f"{format_float(row['accuracy']):>10} "
            f"{format_float(row['macro_f1']):>10} "
            f"{format_float(row['weighted_f1']):>12} "
            f"{format_float(row['training_time_seconds']):>10} "
            f"{format_float(row['inference_time_seconds']):>10}"
        )

    print(f"\nSaved comparison CSV: {comparison_path}")


def format_float(value) -> str:
    if value == "" or value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def normalize_label_vector(values) -> np.ndarray:
    """
    Converts labels/predictions to a 1D int numpy array.

    Handles:
      - sklearn outputs: shape (n,)
      - CatBoost outputs: shape (n, 1)
      - object arrays containing lists/arrays
      - Python lists
    """
    arr = np.asarray(values, dtype=object).reshape(-1)

    normalized = []

    for value in arr:
        v = value

        while isinstance(v, (list, tuple, np.ndarray)):
            if isinstance(v, np.ndarray):
                if v.ndim == 0:
                    v = v.item()
                    break
                v = v.tolist()

            if len(v) != 1:
                raise ValueError(f"Cannot normalize label value with length != 1: {v!r}")

            v = v[0]

        normalized.append(int(v))

    return np.asarray(normalized, dtype=np.int64)


# def vectorize_dataset(rows: list[dict]):
#     all_keys = set()
#     for r in rows:
#         x = r.get("x") or {}
#         all_keys.update(x.keys())
#     feature_names = sorted(all_keys)

#     X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
#     y = np.zeros((len(rows),), dtype=np.int64)

#     for i, r in enumerate(rows):
#         x = r.get("x") or {}
#         for j, k in enumerate(feature_names):
#             X[i, j] = float(x.get(k, 0.0))
#         y[i] = int(r.get("y"))

#     return X, y, feature_names