import time

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

def new_session_wizard(session_settings: dict) -> dict | None:
    print("Starting session wizard...")

    mode = choose_session_mode(session_settings)
    if mode is None:
        return None

    goal_type = choose_goal_type(session_settings)
    if goal_type is None:
        return None

    target = choose_target(session_settings)
    if target is None:
        return None

    name = choose_session_name(session_settings)
    if name is None:
        return None

    max_steps = choose_max_steps(session_settings)
    if max_steps is None:
        return None

    decider = None
    if mode in ("autonomous", "suggestion"):
        decider = choose_decider(session_settings)
        if decider is None:
            return None

    launch_kb_monitor = choose_launch_kb_monitor(session_settings)
    if launch_kb_monitor is None:
        return None

    confirmed = confirm_session_creation(
        mode=mode,
        goal_type=goal_type,
        target=target,
        name=name,
        max_steps=max_steps,
        decider=decider,
        launch_kb_monitor=launch_kb_monitor,
    )
    if not confirmed:
        return None

    return {
        "mode": mode,
        "goal_type": goal_type,
        "target": target,
        "name": name,
        "max_steps": max_steps,
        "decider": decider,
        "launch_kb_monitor": launch_kb_monitor,
    }


def choose_session_mode(session_settings: dict) -> str | None:
    default_mode = session_settings["default_mode"]

    print("\n--- Select session mode ---")
    print(f"Default mode: {default_mode}")
    print("1) Use default")
    print("2) autonomous")
    print("3) observation")
    print("4) suggestion")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "4", "0"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return default_mode
        if raw == "2":
            return "autonomous"
        if raw == "3":
            return "observation"
        if raw == "4":
            return "suggestion"
        print("Invalid option.")


def choose_goal_type(session_settings: dict) -> str | None:
    default_goal_type = session_settings["default_goal_type"]

    print("\n--- Select goal type ---")
    print(f"Default goal type: {default_goal_type}")
    print("1) Use default")
    print("2) recon")
    print("3) enumeration")
    print("4) vulnerability_discovery")
    print("5) exploitation")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "4", "5", "0"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return default_goal_type
        if raw == "2":
            return "recon"
        if raw == "3":
            return "enumeration"
        if raw == "4":
            return "vulnerability_discovery"
        if raw == "5":
            return "exploitation"
        print("Invalid option.")


def choose_target(session_settings: dict) -> str | None:
    default_target = session_settings["default_target"]

    print("\n--- Select target ---")
    print(f"Default target: {default_target}")

    raw = prompt("Target [enter=default, 0=cancel]> ").strip()
    if raw == "0":
        return None
    if raw == "":
        return default_target
    return raw


def choose_session_name(session_settings: dict) -> str | None:
    default_name = session_settings["default_name"]

    print("\n--- Select session name ---")
    print(f"Default name: {default_name}")

    raw = prompt("Name [enter=default, 0=cancel]> ").strip()
    if raw == "0":
        return None
    if raw == "":
        return default_name
    return raw


def choose_max_steps(session_settings: dict) -> int | None:
    default_max_steps = session_settings["default_max_steps"]

    print("\n--- Select max steps ---")
    print(f"Default max steps: {default_max_steps}")

    raw = prompt("Max steps [enter=default, 0=cancel]> ").strip()
    if raw == "0":
        return None
    if raw == "":
        return int(default_max_steps)
    if raw.isdigit() and int(raw) > 0:
        return int(raw)

    print("Invalid max steps.")
    return None


def choose_decider(session_settings: dict) -> str | None:
    default_decider = session_settings.get("default_decider", "scripted")

    print("\n--- Select autonomous decider ---")
    print(f"Default decider: {default_decider}")
    print("1) Use default")
    print("2) scripted")
    print("3) rules")
    print("4) model")
    print("0) Cancel")

    completer = WordCompleter(["0", "1", "2", "3", "4"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return default_decider
        if raw == "2":
            return "scripted"
        if raw == "3":
            return "rules"
        if raw == "4":
            return "model"
        print("Invalid option.")


def choose_launch_kb_monitor(session_settings: dict) -> bool | None:
    default_value = session_settings["launch_kb_monitor"]

    print("\n--- Launch KB monitor window? ---")
    print(f"Default: {'yes' if default_value else 'no'}")
    print("1) Use default")
    print("2) yes")
    print("3) no")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "0"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return bool(default_value)
        if raw == "2":
            return True
        if raw == "3":
            return False
        print("Invalid option.")


def confirm_session_creation(
    mode: str,
    goal_type: str,
    target: str,
    name: str,
    max_steps: int,
    decider: str | None,
    launch_kb_monitor: bool,
) -> bool:
    print("\n--- Confirm session creation ---")
    print(f"Mode: {mode}")
    print(f"Goal type: {goal_type}")
    print(f"Target: {target}")
    print(f"Name: {name}")
    print(f"Max steps: {max_steps}")
    print(f"Decider: {decider if decider is not None else '-'}")
    print(f"Launch KB monitor: {launch_kb_monitor}")

    completer = WordCompleter(["yes", "no"], ignore_case=True)

    while True:
        raw = prompt("Confirm? (y/n): ", completer=completer).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Invalid option.")