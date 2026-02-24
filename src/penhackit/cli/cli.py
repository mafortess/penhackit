# from penhackit.app.bootstrap import AppContext
from penhackit.cli.menus.main_menu import print_main_menu
from penhackit.cli.menus.session_menu import run_session_menu
from penhackit.cli.menus.training_menu import run_training_menu
from penhackit.cli.menus.analysis_menu import run_analysis_menu


# from penhackit.cli.menu_train import menu_train_models
# from cli.menu_analyze import menu_analyze


def run_cli() -> None:
    """
    Main CLI loop. Numeric navigation.
    """
    while True:
        print_main_menu()
        choice = input("> ").strip()

        if choice == "1":
            run_session_menu()
        elif choice == "2":
            run_training_menu()
        elif choice == "3":
            print("Report generation not implemented yet.")
        # elif choice == "3":
        #     run_analysis_menu()
        elif choice == "0":
            print("Exiting...")
            return
        else:
            print("Invalid option.")

# def run_cli(app: AppContext) -> int:
# def _read_choice(prompt: str, valid: set[str]) -> str:
#     while True:
#         s = input(prompt).strip()
#         if s in valid:
#             return s
#         print("Opción inválida.")

# def run_cli():
#     while True:
#         print("=== AGENT COMMAND-LINE INTERFACE (CLI) ===")
#         print("1) Run session")
#         print("2) Train models")
#         print("3) Analyze data/metrics")
#         print("0) Exit")
#         choice = _read_choice("> ", {"1", "2", "3", "0"})

#         if choice == "1":
#             # menu_run_session(app)
#             menu_run_session()
#         elif choice == "2":
#             # menu_train_models(app)
#             menu_train_models()
#         # elif choice == "3":
#         #     menu_analyze(app)
#         else:
#             return 0
