def run_session_menu() -> None:
    while True:
        _print_menu()
        choice = input("> ").strip()

        if choice == "1":
            _not_implemented("Run session")
        elif choice == "2":
            _not_implemented("Observation mode")
        elif choice == "3":
            _not_implemented("Suggestion mode")
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def _print_menu() -> None:
    print("\n--- Run session ---")
    print("1) Run session (autonomous)")
    print("2) Observation mode")
    print("3) Suggestion mode")
    print("0) Back")


def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")