def run_training_menu() -> None:
    while True:
        _print_menu()
        choice = input("> ").strip()

        if choice == "1":
            _not_implemented("Build dataset")
        elif choice == "2":
            _not_implemented("Train model")
        elif choice == "3":
            _not_implemented("Evaluate model")
        elif choice == "4":
            _not_implemented("List datasets/models")
        elif choice == "0":
            return
        else:
            print("Invalid option.")


def _print_menu() -> None:
    print("\n--- Datasets % Models ---")
    print("1) Build dataset")
    print("2) Train model")
    print("3) Evaluate model")
    print("4) List datasets/models")
    print("0) Back")


def _not_implemented(feature: str) -> None:
    print(f"{feature}: not implemented yet.")