from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.session.session_services import run_session_service, list_sessions, show_session_details, delete_session

def run_session_menu(app_context: dict) -> None:
    while True:
        choice = show_session_menu()
        if choice == "1":
            run_session_service(app_context)
        elif choice == "2":
            list_sessions()
        elif choice == "3":
            show_session_details()
        elif choice == "4":
            delete_session()
        elif choice == "0":
            return
        else:
            print("Invalid option.")

def show_session_menu() -> None:
    print("\n--- Session ---")
    print("1) Run new session")
    print("2) List sessions")
    print("3) Show session details")
    print("4) Delete session")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "4", "0"])).strip()

def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")