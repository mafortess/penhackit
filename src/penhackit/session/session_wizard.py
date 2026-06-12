import time

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from pathlib import Path
from penhackit.common.paths import Paths

from penhackit.models.model_loader import list_available_models
from penhackit.session.decision.scripted_sequences import SCRIPTED_SEQUENCES

def new_session_wizard(session_settings: dict, path: Paths) -> dict | None:
    print("Starting session wizard...")

    mode = choose_session_mode(session_settings)
    if mode is None:
        return None

    goal_type = choose_goal_type(session_settings)
    if goal_type is None:
        return None

    # target = choose_target(session_settings)
    # if target is None:
    #     return None

    target_type = choose_target_type(session_settings)
    if target_type is None:
        return None

    target = choose_target(session_settings, target_type)
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

    attack_name = None
    scripted_sequence = None

    if decider == "scripted":
        attack_name, scripted_sequence = choose_scripted_attack_sequence(
            goal_type=goal_type,
            target_type=target_type,
            session_settings=session_settings,
        )
        if scripted_sequence is None:
            return None

    model_id = None
    if decider == "model":
        model_id = choose_model(session_settings, path)
        if model_id is None:
            return None
    


    # launch_kb_monitor = choose_launch_kb_monitor(session_settings)
    # if launch_kb_monitor is None:
    #     return None

    confirmed = confirm_session_creation(
        mode=mode,
        goal_type=goal_type,
        target_type=target_type,
        target=target,
        name=name,
        max_steps=max_steps,
        decider=decider,
        scripted_sequence=scripted_sequence,
        attack_name=attack_name,
        # launch_kb_monitor=launch_kb_monitor,
    )
    if not confirmed:
        return None

    return {
        "mode": mode,
        "goal_type": goal_type,
        "target_type": target_type,
        "target": target,
        "name": name,
        "max_steps": max_steps,
        "decider": decider,
        "model_id": model_id,
        "attack_name": attack_name,
        "scripted_sequence": scripted_sequence,
        # "launch_kb_monitor": launch_kb_monitor,
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
    print("6) obtain_session")
    print("7) full_exploit")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "4", "5", "6", "7", "0"], ignore_case=True)

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
        if raw == "6":
            return "obtain_session"
        if raw == "7":
            return "full_exploit"
        print("Invalid option.")

def choose_target_type(session_settings: dict) -> str | None:
    default_target_type = session_settings.get("default_target_type", "host")

    print("\n--- Select target type ---")
    print(f"Default target type: {default_target_type}")
    print("1) Use default")
    print("2) network")
    print("3) host")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "0"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return default_target_type
        if raw == "2":
            return "network"
        if raw == "3":
            return "host"
        print("Invalid option.")


def choose_target(session_settings: dict, target_type: str) -> str | None:
    if target_type == "network":
        default_target = session_settings["default_network_target"]
        detected_targets = detect_networks()
    else:
        default_target = session_settings["default_host_target"]
        detected_targets = []

    target_options = [default_target]

    for target in detected_targets:
        if target not in target_options:
            target_options.append(target)

    completer = WordCompleter(target_options + ["0"], ignore_case=True)

    print(f"\n--- Select {target_type} target ---")
    print(f"Default {target_type} target: {default_target}")

    if target_options:
        print("Available targets:")
        for idx, option in enumerate(target_options, start=1):
            print(f"{idx}) {option}")

    raw = prompt("Target [TAB=suggestions, enter=default, 0=cancel]> ", completer=completer).strip()

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
    print("3) model")
    print("0) Cancel")

    completer = WordCompleter(["1", "2", "3", "0"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()
        if raw == "0":
            return None
        if raw == "1" or raw == "":
            return default_decider
        if raw == "2":
            return "scripted"
        if raw == "3":
            return "model"
        print("Invalid option.")



ATTACK_OPTIONS = [
    ("vsftpd_msf", "VSFTPD 2.3.4 backdoor - Metasploit - 601"),
    ("vsftpd_manual", "VSFTPD 2.3.4 backdoor - manual - 610"),
    ("samba_usermap_msf", "Samba usermap_script - Metasploit - 600"),
    ("distcc_msf", "DistCC exec - Metasploit - 602"),
    ("postgres_msf", "PostgreSQL payload - Metasploit - 604"),
    ("unreal_ircd_msf", "UnrealIRCd backdoor - Metasploit - 605"),
    ("ingreslock_bind_shell", "Ingreslock bind shell - manual - 606"),
    ("ssh_weak_creds_manual", "SSH weak credentials - manual - 520"),
    ("telnet_weak_creds_manual", "Telnet weak credentials - manual - 521"),
    ("ssh_weak_creds_msf", "SSH weak credentials - Metasploit - 611"),
    ("ftp_weak_creds_msf", "FTP weak credentials - Metasploit - 612"),
    ("ftp_weak_creds_hydra", "FTP weak credentials - Hydra + manual validation - 613/614"),
    ("exploit_smoke_test", "Exploit smoke test (all attacks)"),
]


def choose_scripted_attack_sequence(goal_type: str, target_type: str, session_settings: dict) -> tuple[str | None, str | None]:
    default_attack_name = session_settings.get("default_attack_name", "vsftpd_msf")

    attack_names = [name for name, _ in ATTACK_OPTIONS]

    if default_attack_name not in attack_names:
        default_attack_name = "vsftpd_msf"

    print("\n--- Select scripted attack sequence ---")
    print(f"Target type: {target_type}")
    print(f"Goal type: {goal_type}")
    print(f"Default attack: {default_attack_name}")
    print("1) Use default")

    for idx, (attack_name, description) in enumerate(ATTACK_OPTIONS, start=2):
        print(f"{idx}) {description} [{attack_name}]")

    print("0) Cancel")

    valid_options = ["1"] + [
        str(i)
        for i in range(2, len(ATTACK_OPTIONS) + 2)
    ] + ["0"]

    completer = WordCompleter(valid_options, ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()

        if raw == "0":
            return None, None

        if raw == "1" or raw == "":
            attack_name = default_attack_name
        elif raw.isdigit():
            selected_idx = int(raw)
            attack_idx = selected_idx - 2

            if 0 <= attack_idx < len(ATTACK_OPTIONS):
                attack_name = ATTACK_OPTIONS[attack_idx][0]
            else:
                print("Invalid option.")
                continue
        else:
            print("Invalid option.")
            continue

        sequence_name = resolve_scripted_sequence_name(
            target_type=target_type,
            goal_type=goal_type,
            attack_name=attack_name,
        )

        if sequence_name is None:
            print(
                "No scripted sequence found for "
                f"target_type={target_type}, goal_type={goal_type}, attack={attack_name}."
            )
            print("Available matching sequence names should follow one of these patterns:")
            print(f"- {target_type}_{goal_type}_{attack_name}")
            print(f"- attack_{attack_name}")
            print(f"- network_attack_{attack_name}")
            return None, None

        print(f"Selected scripted sequence: {sequence_name}")

        return attack_name, sequence_name


def resolve_scripted_sequence_name(target_type: str, goal_type: str, attack_name: str) -> str | None:
    """
    Resuelve el nombre de secuencia scripted.

    Prioridad:
    1. Si attack_name ya es una secuencia directa, se usa tal cual.
       Ejemplo: exploit_smoke_test

    2. Para full_exploit, usar secuencia global:
       host_full_exploit
       network_full_exploit

    3. Para obtain_session, usar secuencia específica por ataque:
       host_obtain_session_vsftpd_msf
       network_obtain_session_vsftpd_msf

    4. Compatibilidad con nombres antiguos:
       attack_vsftpd_msf
       network_attack_vsftpd_msf
    """

    if attack_name in SCRIPTED_SEQUENCES:
        return attack_name

    if goal_type == "full_exploit":
        sequence_name = f"{target_type}_full_exploit"

        if sequence_name in SCRIPTED_SEQUENCES:
            return sequence_name

    if goal_type == "obtain_session":
        candidates = [
            f"{target_type}_{goal_type}_{attack_name}",
        ]

        if target_type == "host":
            candidates.append(f"attack_{attack_name}")

        if target_type == "network":
            candidates.append(f"network_attack_{attack_name}")

        for candidate in candidates:
            if candidate in SCRIPTED_SEQUENCES:
                return candidate

    sequence_name = f"{target_type}_{goal_type}_{attack_name}"

    if sequence_name in SCRIPTED_SEQUENCES:
        return sequence_name

    return None


def choose_model(session_settings: dict, paths: Paths) -> str | None:
    default_model_id = session_settings.get("default_model_id")
    available_models = list_available_models(paths.models_dir)

    print("\n--- Select model ---")

    if not available_models:
        print("No trained models found.")
        print("Train a model before running a session with decider_type='model'.\n")
        return None

    options = []

    if default_model_id is not None and default_model_id in available_models:
        options.append(default_model_id)

    for model_id in available_models:
        if model_id != default_model_id:
            options.append(model_id)

    for idx, model_id in enumerate(options, start=1):
        if model_id == default_model_id:
            print(f"{idx}) Use default: {model_id}")
        else:
            print(f"{idx}) {model_id}")

    print("0) Cancel")

    completer = WordCompleter(
        [str(i) for i in range(1, len(options) + 1)] + ["0"],
        ignore_case=True,
    )

    while True:
        raw = prompt("> ", completer=completer).strip()

        if raw == "0":
            return None

        if raw == "" and default_model_id in options:
            return default_model_id

        if raw.isdigit():
            selected_idx = int(raw)
            if 1 <= selected_idx <= len(options):
                return options[selected_idx - 1]

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
    target_type: str,
    target: str,
    name: str,
    max_steps: int,
    decider: str | None,
    scripted_sequence: str | None,
    attack_name: str | None,
    # launch_kb_monitor: bool,
) -> bool:
    print("\n--- Confirm session creation ---")
    print(f"Mode: {mode}")
    print(f"Goal type: {goal_type}")
    print(f"Target type: {target_type}")
    print(f"Target: {target}")
    print(f"Name: {name}")
    print(f"Max steps: {max_steps}")
    print(f"Decider: {decider if decider is not None else '-'}")
    print(f"Attack name: {attack_name if attack_name is not None else '-'}")
    print(f"Scripted sequence: {scripted_sequence if scripted_sequence is not None else '-'}")
    # print(f"Launch KB monitor: {launch_kb_monitor}")

    completer = WordCompleter(["yes", "no"], ignore_case=True)

    while True:
        raw = prompt("Confirm? (y/n): ", completer=completer).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Invalid option.")


import subprocess
import ipaddress


def detect_networks():
    try:
        result = subprocess.run(
            ["ip", "-o", "-4", "addr", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )
    except Exception as e:
        print(f"Error detecting networks: {e}")
        return []

    networks = []

    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue

        cidr = parts[3]  # ej: 10.7.7.5/24

        try:
            ip_interface = ipaddress.ip_interface(cidr)
            network = str(ip_interface.network)
        except ValueError:
            continue

        # ignorar loopback
        if network.startswith("127."):
            continue

        if network not in networks:
            networks.append(network)

    return networks