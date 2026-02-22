def run_analysis_menu() -> None:
    while True:
        _print_menu()
        choice = input("> ").strip()

        if choice == "1":
            _not_implemented("Session metrics summary")
        elif choice == "2":
            _not_implemented("Dataset statistics")
        elif choice == "3":
            _not_implemented("Confusion matrices")
        elif choice == "4":
            _not_implemented("Compare runs")
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def _print_menu() -> None:
    print("\n--- Analyze data/metrics ---")
    print("1) Session metrics summary")
    print("2) Dataset statistics")
    print("3) Confusion matrices")
    print("4) Compare runs")
    print("0) Back")


def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")