from pathlib import Path
import json
import time

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

def choose_dataset_dir(datasets_dir: Path) -> Path | None:
    items = list_dataset_candidates(datasets_dir)
    if not items:
        print(f"No datasets found in: {datasets_dir}")
        print("Expected either:")
        print(f" - {datasets_dir / 'dataset.jsonl'}")
        print(f" - {datasets_dir / '<dataset_id>' / 'dataset.jsonl'}")
        return None

    print("\nAvailable datasets:")
    for i, p in enumerate(items, start=1):
        label = p.name if p != datasets_dir else "(root) datasets/"
        print(f"{i}) {label}")
    raw = prompt("Select dataset> ", completer=WordCompleter([str(i) for i in range(1, len(items) + 1)] + ["0"])).strip()
    if raw == "0":
        return None
    if not raw.isdigit():
        print("Invalid input.")
        return None
    idx = int(raw)
    if idx < 1 or idx > len(items):
        print("Out of range.")
        return None
    return items[idx - 1]


def load_dataset_jsonl(dataset_dir: Path) -> list[dict]:
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
# Model helpers
# =========================

MODEL_CHOICES = {
    "1": ("logreg", "Logistic Regression (multinomial)", lambda: LogisticRegression(max_iter=2000)),
    "2": ("decision_tree", "Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
    "3": ("random_forest", "Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
    "4": ("mlp", "MLP (2 hidden layers)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)),
}

def choose_model_type() -> tuple[str, str, callable] | None:
    print("\nModel types:")
    for k, (name, desc, _) in MODEL_CHOICES.items():
        print(f"{k}) {name} - {desc}")
    # raw = input("Select model (0 cancel)> ").strip()
    raw = prompt("Select model> ", completer=WordCompleter(list(MODEL_CHOICES.keys()) + ["0"])).strip()
    if raw == "0":
        return None
    if raw not in MODEL_CHOICES:
        print("Invalid option.")
        return None
    return MODEL_CHOICES[raw]

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
        
def run_training_interactive(datasets_dir: Path, models_dir: Path) -> None:
    """
    Interactive training:
    - choose model type
    - choose dataset folder (must contain dataset.jsonl)
    - train + evaluate
    - save model + metrics under models_dir/<dataset>/<model>_<n>/
    """

    # 1) elegir modelo
    choice = choose_model_type()
    if not choice:
        return
    model_key, model_desc, model_factory = choice
    print(f"Selected model: {model_key} ({model_desc})")

    # 2) elegir dataset
    dataset_dir = choose_dataset_dir(datasets_dir)
    if not dataset_dir:
        return    
    print(f"\nSelected dataset session: {dataset_dir.name}")
    
    rows = load_dataset_jsonl(dataset_dir)
    X, y, feature_names = vectorize_bc_rows(rows)

    # split
    counts = Counter(y.tolist())
    min_count = min(counts.values())

    strat = y if len(set(y.tolist())) > 1 and min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    model = model_factory()

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # output dir
    models_dir.mkdir(parents=True, exist_ok=True)
    out_parent = models_dir / dataset_dir.name
    out_parent.mkdir(parents=True, exist_ok=True)

    out_dir = next_available_path(out_parent, f"{model_key}", "")  # placeholder
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
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "schema": "penhackit.training.v1",
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_dir": str(dataset_dir),
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