from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.training.training_services import train_model, evaluate_model, list_datasets_and_models, rebuild_dataset_from_sessions
def run_training_menu(app_context: dict) -> None:
    while True:
        choice = show_training_menu()
        if choice == "1":
            train_model(app_context)
        elif choice == "2":
            evaluate_model(app_context)
        elif choice == "3":
            list_datasets_and_models(app_context)
        elif choice == "4":
            rebuild_dataset_from_sessions(app_context)
        elif choice == "0":
            return
        else:
            print("Invalid option.")

def show_training_menu() -> None:
    print("\n--- Training ---")
    print("1) Train model")
    print("2) Evaluate model")
    print("3) List datasets/models")
    print("4) Rebuild dataset from sessions")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "4", "0"])).strip()