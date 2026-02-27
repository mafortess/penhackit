import random

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.cli.menus.session_menu import run_session_menu
from penhackit.cli.menus.training_menu import run_training_menu
from penhackit.cli.menus.report_menu import run_report_menu
from penhackit.cli.menus.settings_menu import run_settings_menu

def run_main_menu(app_context: dict) -> None:
    while True:
        choice = show_main_menu()
        if choice == "1":
            run_session_menu(app_context)
        elif choice == "2":
            run_training_menu(app_context)
        elif choice == "3":
            run_report_menu(app_context)
        elif choice == "4":
            run_settings_menu(app_context)
        elif choice == "0":
            print("Exiting...")
            return
        else:
            print("Invalid option.")

def show_main_menu() -> None:
    show_banner()
    print("\n=== Penhackit CLI ===")
    print("1) Sessions")
    print("2) Training")
    print("3) Reports")
    print("4) Settings")
    print("0) Exit")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "4", "0"])).strip()

def show_banner() -> None:
    print(random.choice(BANNERS))

BANNERS = [
r"""
   ____            __  __            _    _ __
  / __ \___  ____ / /_/ /_  ___ ____| |  (_) /_
 / /_/ / _ \/ __ `/ __/ __ \/ _ `/ __| | / / __/
/ ____/  __/ /_/ / /_/ / / /  __/ /  | |/ / /_
/_/    \___/\__,_/\__/_/ /_/\___/_/   |___/\__/

 PenHackIt — controlled pentesting automation (research prototype)
""" ,

r"""
 ____            _   _            _    ___ _____
|  _ \ ___ _ __ | | | | __ _  ___| | _|_ _|_   _|
| |_) / _ \ '_ \| |_| |/ _` |/ __| |/ /| |  | |
|  __/  __/ | | |  _  | (_| | (__|   < | |  | |
|_|   \___|_| |_|_| |_|\__,_|\___|_|\_\___| |_|

          PenHackIt — Pentest agent (BC + local LLM reporting)
"""
]


# def _read_choice(prompt: str, valid: set[str]) -> str:
#     while True:
#         s = input(prompt).strip()
#         if s in valid:
#             return s
#         print("Opción inválida.")
