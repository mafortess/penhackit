from pathlib import Path
import json
from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

import csv
import time

def list_dataset_candidates(datasets_dir: Path) -> list[Path]:
    """
    Devuelve una lista de 'dataset_dir' candidatos.
    - Añade cada subdirectorio que contenga dataset.jsonl.
    - Añade el propio datasets_dir si contiene dataset.jsonl (caso plano).
    Sort by mtime (últimos primero).
    """
    candidates = []

    # caso plano: datasets/dataset.jsonl
    if (datasets_dir / "dataset.jsonl").exists():
        candidates.append(datasets_dir)

    # caso normal: datasets/<name>/dataset.jsonl
    if datasets_dir.exists():
        for p in datasets_dir.iterdir():
            if p.is_dir() and (p / "dataset.jsonl").exists():
                candidates.append(p)

    # orden por mtime (últimos primero)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

def load_dataset_jsonl(dataset_path: Path | str) -> list[dict]:
    """
    
    """
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


def create_model_output_dir(models_dir: Path, model_key: str) -> Path:
    """
    Creates a unique output directory for one training run.

    New structure:
      models/<model_key>/<run_id>/

    Example:
      models/random_forest/20260610_132455/
      models/random_forest/20260610_132455_1/
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    model_dir = models_dir / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = model_dir / run_id

    if out_dir.exists():
        i = 1
        while (model_dir / f"{run_id}_{i}").exists():
            i += 1
        out_dir = model_dir / f"{run_id}_{i}"

    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion_matrix_csv(path: Path,labels: list[int], confusion_matrix_data: list[list[int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)

        header = ["true\\pred"] + [str(label) for label in labels]
        writer.writerow(header)

        for label, row in zip(labels, confusion_matrix_data):
            writer.writerow([str(label)] + row)


def save_confusion_matrix_png(path: Path, labels: list[int], confusion_matrix_data: list[list[int]], title: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"Could not save confusion matrix PNG because matplotlib/numpy is missing: {e}")
        return

    cm = np.array(confusion_matrix_data, dtype=np.int64)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm)

    ax.set_title(title)
    ax.set_xlabel("Predicted action_id")
    ax.set_ylabel("True action_id")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels([str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticklabels([str(label) for label in labels])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_predictions_csv(path: Path, y_test, y_pred) -> None:
    y_test = _normalize_label_vector_for_storage(y_test)
    y_pred = _normalize_label_vector_for_storage(y_pred)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_index", "y_true", "y_pred", "correct"])

        for i, (true_value, pred_value) in enumerate(zip(y_test.tolist(), y_pred.tolist())):
            true_value = int(true_value)
            pred_value = int(pred_value)
            writer.writerow([
                i,
                true_value,
                pred_value,
                int(true_value == pred_value),
            ])


def append_offline_comparison_row(path: Path, metrics: dict) -> None:
    fieldnames = [
        "trained_at_utc",
        "dataset_name",
        "dataset_path",
        "model_type",
        "n_samples",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "n_classes",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "training_time_seconds",
        "inference_time_seconds",
        "avg_inference_time_seconds",
        "output_dir",
        "model_path",
    ]

    file_exists = path.exists()

    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "trained_at_utc": metrics.get("trained_at_utc"),
            "dataset_name": metrics.get("dataset_name"),
            "dataset_path": metrics.get("dataset_path"),
            "model_type": metrics.get("model_type"),
            "n_samples": metrics.get("n_samples"),
            "n_train_samples": metrics.get("n_train_samples"),
            "n_test_samples": metrics.get("n_test_samples"),
            "n_features": metrics.get("n_features"),
            "n_classes": metrics.get("n_classes"),
            "accuracy": metrics.get("accuracy"),
            "precision_macro": metrics.get("precision_macro"),
            "recall_macro": metrics.get("recall_macro"),
            "macro_f1": metrics.get("macro_f1"),
            "weighted_f1": metrics.get("weighted_f1"),
            "training_time_seconds": metrics.get("training_time_seconds"),
            "inference_time_seconds": metrics.get("inference_time_seconds"),
            "avg_inference_time_seconds": metrics.get("avg_inference_time_seconds"),
            "output_dir": metrics.get("output_dir"),
            "model_path": metrics.get("model_path"),
        })



def write_offline_comparison_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "trained_at_utc",
        "dataset_name",
        "dataset_path",
        "model_type",
        "n_samples",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "n_classes",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "training_time_seconds",
        "inference_time_seconds",
        "avg_inference_time_seconds",
        "output_dir",
        "model_path",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def get_offline_comparison_fieldnames() -> list[str]:
    return [
        "trained_at_utc",
        "run_id",
        "dataset_name",
        "dataset_path",
        "model_type",
        "n_samples",
        "n_train_samples",
        "n_test_samples",
        "n_features",
        "n_classes",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "weighted_f1",
        "training_time_seconds",
        "inference_time_seconds",
        "avg_inference_time_seconds",
        "model_dir",
        "output_dir",
        "model_path",
    ]


def load_saved_metrics(models_dir: Path) -> list[tuple[Path, dict]]:
    """
    Loads all metrics.json files under models_dir/<model_run>/metrics.json.
    """
    if not models_dir.exists():
        return []

    metric_files = sorted(models_dir.glob("**/metrics.json"))
    loaded = []

    for metrics_path in metric_files:
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skipping invalid metrics file {metrics_path}: {e}")
            continue

        loaded.append((metrics_path, metrics))

    return loaded

def _normalize_label_vector_for_storage(values):
    import numpy as np

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


def merge_jsonl_files(input_paths: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for path in input_paths:
            path = Path(path)

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    out.write(line + "\n")


def merge_datasets_quick(paths: Path) -> None:
    print("\nMerge datasets")
    print("--------------")

    dataset_files = sorted(
        [p for p in paths.datasets_dir.rglob("*.jsonl") if p.is_file()],
        key=lambda p: str(p).lower(),
    )

    if not dataset_files:
        print(f"No .jsonl datasets found in: {paths.datasets_dir}")
        return

    dataset_names = []

    for path in dataset_files:
        try:
            dataset_names.append(str(path.relative_to(paths.datasets_dir)))
        except ValueError:
            dataset_names.append(str(path))

    completer = WordCompleter(dataset_names, ignore_case=True)

    print("\nAvailable datasets:")
    for name in dataset_names:
        print(f"  - {name}")

    print("\nSelect datasets one by one. Press ENTER without text to finish.\n")

    input_paths = []

    while True:
        choice = prompt("Dataset to add > ", completer=completer).strip()

        if not choice:
            break

        selected_path = paths.datasets_dir / choice

        if not selected_path.exists():
            print(f"Dataset not found: {selected_path}")
            continue

        if selected_path in input_paths:
            print(f"Already selected: {choice}")
            continue

        input_paths.append(selected_path)
        print(f"Added: {choice}")

    if not input_paths:
        print("No datasets selected.")
        return

    print("\nSelected datasets:")
    for path in input_paths:
        print(f"  - {path.relative_to(paths.datasets_dir)}")

    output_name = prompt("\nOutput dataset name [merged_dataset.jsonl] > ").strip()

    if not output_name:
        output_name = "merged_dataset.jsonl"

    if not output_name.endswith(".jsonl"):
        output_name += ".jsonl"

    output_path = paths.datasets_dir / output_name

    try:
        merge_jsonl_files(input_paths, output_path)
    except Exception as e:
        print(f"Error merging datasets: {e}")
        return

    print(f"\nMerged dataset created: {output_path}")