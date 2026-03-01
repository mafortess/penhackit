from zipfile import Path

from penhackit.training.training_logic import MODEL_CHOICES, training_model

from penhackit.common.paths import Paths
from penhackit.training.training_wizard import train_model_wizard

def run_train_model_service(app_context: dict) -> None:
    print(f"Running training service...") # with context: {app_context}")

    # Esta es la opcion equivocada, no tiene mucho sentido poner un fallback silencioso a ".", 
    # porque eso puede provocar: guardar cosas en la carpeta equivocada y un comportamiento inesperado
    # working_dir = app_context.get("workspace_dir", ".")
    
    # Esta es la opción a usar, porque workspace_dir es obligatorio, si no existe, debe lanza KeyError
    # su ausencia sería un error real de programación o de bootstrap.
    # workspace_dir = app_context["workspace_dir"]
    
    # Load default settings and paths
    training_settings = app_context["settings"]["training"]
    print(f"Default training settings: {training_settings}")
    paths = app_context["paths"]
    # workspace_dir = paths.workspace_dir
    # print(f"Using working directory: {workspace_dir}")

    # Wizard for training configuration
    wizard_data = train_model_wizard(training_settings, paths)
    if wizard_data is None:
        print("Training cancelled.")
        return

    test_size = wizard_data["test_size"] if wizard_data else training_settings["default_test_size"]
    model_type = wizard_data["model_type"] if wizard_data else training_settings["default_model_type"]
    random_state = wizard_data["random_state"] if wizard_data else training_settings["default_random_state"]
    dataset_path = wizard_data["dataset_path"] if wizard_data else None

    print(f"test_size: {test_size}")
    print(f"model_type: {model_type}")
    print(f"random_state: {random_state}")
    print(f"dataset_path: {dataset_path}")

    # Asegurarse de que las carpetas de datasets y modelos existen
    datasets_dir = paths.datasets_dir
    print(f"Looking for datasets in: {datasets_dir}")
    models_dir = paths.models_dir
    print(f"Saving trained models to: {models_dir}")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("model_type:", model_type)
    print("MODEL_CHOICES keys:", list(MODEL_CHOICES.keys()))
    print("MODEL_CHOICES keys and descriptions:")
    for k, (name, desc, _) in MODEL_CHOICES.items():
        print(f"  {k}: {name} - {desc}")

    try:
        training_model(training_settings, dataset_path, model_type[0], model_type[2], paths)
    except Exception as e:
        print(f"Error during training: {e}")


def run_evaluate_model_service(app_context: dict) -> None:
    # Aquí iría la lógica para evaluar el modelo, como cargar un conjunto de prueba, calcular métricas, etc.
    print(f"Running model evaluation service...") # with context: {app_context}")
    # wizard_data = evaluate_model_wizard(settings)
    # if wizard_data is None:
    #     print("Evaluation cancelled.")
    #     return

    # try:
    #     evaluate_model(
    #         model_path=wizard_data["model_path"],
    #         dataset_path=wizard_data["dataset_path"],
    #         settings=settings,
    #     )
    # except Exception as exc:
    #     print(f"Error evaluating model: {exc}")

def list_datasets_and_models(app_context: dict) -> None:
    print("Listing datasets and models...")
    # Aquí iría la lógica para listar los datasets y modelos disponibles, mostrando información relevante.

def rebuild_dataset_from_sessions(app_context: dict) -> None:
    print("Rebuilding dataset from sessions...")
    # Aquí iría la lógica para reconstruir un dataset a partir de las sesiones, probablemente pidiendo criterios o filtros.
    # Por ahora, solo es un placeholder.