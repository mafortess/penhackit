from penhackit.common.paths import Paths

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.training.training_storage import list_dataset_candidates
from penhackit.training.training_logic import MODEL_CHOICES

from pathlib import Path

def train_model_wizard(training_settings: dict, path: Paths) -> dict | None:
    print("Starting training wizard...")
    
    # SELECT DATASET
    dataset_path = wizard_select_dataset(training_settings, path)
    if dataset_path is None:
        return None

    # SELECT MODEL TYPE
    model_type = choose_model_type(training_settings, path)
    print(f"Selected model type: {model_type}")
    model_key, model_desc, model_factory = model_type
    print(f"Selected model: {model_key} ({model_desc}) {model_factory}")

    if model_type is None:
        return None

    confirmed = wizard_confirm_training(
        dataset_path=dataset_path,
        model_type=model_type,
    )
    if not confirmed:
        return None

    return {
        "dataset_path": dataset_path,
        "model_type": model_type,
        "test_size": training_settings["default_test_size"], # Placeholder, could be another wizard step
        "random_state": training_settings["default_random_state"], # Placeholder, could be another wizard step
    }


def wizard_select_dataset(settings: dict, paths: Paths) -> str | None:
    datasets_dir = paths.datasets_dir
    print(f"Looking for datasets in: {datasets_dir}")
    datasets = [f.name for f in datasets_dir.glob("*.jsonl") if f.is_file()]
    if not datasets:
        print("No datasets available. Please create a dataset first.")
        return None

    print("\nAvailable datasets:")
    for i, ds in enumerate(datasets, 1):
        print(f"{i}) {ds}")
    choice = prompt("Select dataset> ", completer=WordCompleter([str(i) for i in range(1, len(datasets)+1)]))
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(datasets):
            return str(datasets_dir / datasets[idx])
    except ValueError:
        pass

    print("Invalid selection.")
    return None

def wizard_select_model_type(settings: dict, paths: Paths) -> str | None:
    model_types = ["Decision Tree", "Random Forest", "MLP"]
    print("\nAvailable model types:")
    for i, mt in enumerate(model_types, 1):
        print(f"{i}) {mt}")
    choice = prompt("Select model type> ", completer=WordCompleter([str(i) for i in range(1, len(model_types)+1)]))
    if choice == "0":
        return None
    if choice not in MODEL_CHOICES:
        print("Invalid option.")
        return None
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(model_types):
            return model_types[idx]

    except ValueError:
        pass

    print("Invalid selection.")
    return None

def wizard_confirm_training(dataset_path: str, model_type: str) -> bool:
    print("\nPlease confirm the training configuration:")
    print(f"Dataset: {dataset_path}")
    print(f"Model type: {model_type}")
    choice = prompt("Confirm? (y/n)> ", completer=WordCompleter(["y", "n"])).strip().lower()
    return choice == "y"

# =========================
# Model helpers
# =========================

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

def choose_model_type(training_settings: dict, paths: Paths) -> tuple[str, str, callable] | None:
    print("\nModel types:")
    for k, (name, desc, _) in MODEL_CHOICES.items():
        print(f"{k}) {name} - {desc}")
    # raw = input("Select model (0 cancel)> ").strip()
    try:
        choice = prompt("Select model> ", completer=WordCompleter(list(MODEL_CHOICES.keys()) + ["0"])).strip()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return None
    
    if choice == "0":
        return None
    if choice not in MODEL_CHOICES:
        print("Invalid option.")
        return None
    
    return MODEL_CHOICES[choice]

