from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.session.session_services import run_session_service, run_view_session_service
def run_session_menu(app_context: dict) -> None:
    while True:
        choice = show_session_menu()
        if choice == "1":
            run_session_service(app_context)
        elif choice == "2":
            run_view_session_service(app_context)
        elif choice == "0":
            return
        else:
            print("Invalid option.")

def show_session_menu() -> None:
    print("\n--- Session ---")
    print("1) Run new session")
    print("2) View sessions")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "0"])).strip()
