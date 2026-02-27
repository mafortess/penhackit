from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.report.report_services import generate_report, list_reports, show_report_details
def run_report_menu() -> None:
    while True:
        choice = show_report_menu()
        if choice == "1":
            session_id = prompt("Enter session ID> ")
            generate_report(session_id)
        elif choice == "2":
            list_reports()
        elif choice == "3":
            report_id = prompt("Enter report ID> ")
            show_report_details(report_id)
        elif choice == "0":
            return
        else:
            print("Invalid option.")

def show_report_menu() -> None:
    print("\n--- Reports ---")
    print("1) Generate report")
    print("2) List reports")
    print("3) Show report details")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "0"])).strip()

def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")