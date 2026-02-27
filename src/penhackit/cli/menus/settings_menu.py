from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

def run_settings_menu() -> None:
    while True:
        choice = show_settings_menu()
        if choice == "1":
            _not_implemented("Show current configuration")
        elif choice == "2":
            _not_implemented("Update setting")
        elif choice == "3":
            _not_implemented("Reset settings to default")
        elif choice == "0":
            return
        else:
            print("Invalid option.")

def show_settings_menu() -> None:
    print("\n--- Settings ---")
    print("1) Show current configuration")
    print("2) Update setting")
    print("3) Reset settings to default")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "0"])).strip()

def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")