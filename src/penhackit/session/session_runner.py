import copy
import time
from penhackit.common.paths import Paths

from penhackit.session.command.command_builder import command_builder
from penhackit.session.event.event_builder import parse_command_result
from penhackit.session.kb.kb_updater import update_kb, compute_kb_progress_simple, save_kb
from penhackit.session.state.state_builder import build_state
from penhackit.session.action.action_ids import ACTIONS, extract_action_id_from_cmd

from penhackit.session.decision.policies import  scripted_policy_decide_action, model_policy_decide_action, rules_policy_decide_action

from penhackit.session.execution.execute import execute_command

from penhackit.session.logging.logger import log_command_output, log_step, log_dataset_row, log_freeform_row


def run_session_loop(session_settings: dict, session_info: dict, paths: Paths):
    mode = session_settings["mode"]
     # Dependiendo del modo de la sesión, ejecuta la lógica correspondiente (autonomous, observation, suggestion).
    if mode == "autonomous":
        run_session_autonomous(session_settings, session_info, paths)
        return
    elif mode == "observation":
        new_session_observation(session_settings, session_info, paths)
        return
    elif mode == "suggestion":
        new_session_suggestion(session_settings, session_info, paths)
        return
    
    raise ValueError(f"Invalid session mode: {mode}")


# ============================================================================================================
# ============================================================================================================
# ============================================================================================================

def run_session_autonomous(session_settings: dict, session_info: dict, paths: Paths) -> None:
    """
    Modo autónomo: el sistema decide la acción a ejecutar (usando la política elegida) 
    y ejecuta el comando construido sin intervención del pentester.
    """
    print("Starting autonomous session logic...")

    # Initatlize sessión_context
    # Initialize KB (knowledge base)
    # Setup inicial: cargar KB, contexto, modelo (si aplica), etc. desde session_info.
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]
    model = session_info["model"]
    fn = session_info["feature_names"]

    print(f"Session ID: {session_info['session_id']}")

    print("Initial KB:")
    print(kb)
    
    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n==========================")
        print(f"-------- Step {t+1} --------")

        prev_kb = copy.deepcopy(kb)

        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making.
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
        action_id = decide_autonomous_action(session_settings=session_settings, state=state, step=t, model=model, feature_names=fn)

        if action_id == 0:
            print("STOP action selected. Finishing session loop.")
            break

        # Based on the decided action_id, build the command to execute using command_builder(action_id, kb).
        action_name, result, events = execute_autonomous_action(action_id, kb)

        print(f"Events generated from command result: {events}")

        log_command_output(session_info["session_dir"], session_info["session_id"], action_id, action_name, result)
     
        # Update KB
        kb = update_kb_with_events(kb, events, result, action_id, action_name, t)

        print(f"Updated KB: {kb}")
        save_kb(session_info["session_dir"], kb)

        # Update sessión_context       
        progress = compute_kb_progress_simple(prev_kb, kb)
        print_autonomous_progress(progress)

        # Save/log step(state, action, command, result, kb, context, state)
        log_step(session_info["session_dir"], session_info["session_id"], {"t": t, "state": state, "action_id": action_id, "command": result.get("cmd")})

        time.sleep(1)
        
    print("Session finished loop")

# ============================================================================================================

def execute_autonomous_action(action_id: int, kb: dict) -> tuple:
    """
    Ejecuta el comando correspondiente al action_id dado, parsea el resultado y devuelve el resultado y los eventos generados para actualizar la KB.
    Devuelve una tupla con (action_name, result, events) donde:
      - action_name: str con el nombre de la acción ejecutada
      - result: dict con rc, stdout, stderr, cmd
      - events: list de dicts con eventos extraídos del resultado para actualizar la KB
    """
    print("\nCOMMAND BUILDING...")
    # Build commmand to execute based on the decided action_id, using from KB and session context. 
    # action_id = 1  # INSPECT_IPCONFIG
    action_data = ACTIONS.get(action_id, ACTIONS[0])
    action_name = action_data["name"]

    print(f"Decided action: {action_name} (ID: {action_id})")

    command_ctx = command_builder(action_data, kb)
    print(f"Built command: {command_ctx }")

    if command_ctx is None:
        print(f"No command to execute for action: {action_name} (ID: {action_id}), skipping execution.")
        return action_name, {
            "rc": None,
            "stdout": "",
            "stderr": "",
            "cmd": None
        }, []

    else:
        print("\nACTION EXECUTION...")
        command_to_run = command_ctx["command"]
        print(f"Executing command for action: {action_name} (ID: {action_id}): {command_to_run}")
        rc, stdout, stderr = execute_command(command_to_run)

        result = {
            "rc": rc,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": command_to_run,
            "target": command_ctx.get("target"),
            "target_ip": command_ctx.get("target_ip"),
            "target_port": command_ctx.get("target_port"),
            "known_open_ports_csv": command_ctx.get("known_open_ports_csv"),
            "service_version_string": command_ctx.get("service_version_string"),
            "parser_family": command_ctx.get("parser_family"),
        }
        events = parse_command_result(action_name, result)
    
    return action_name, result, events


def decide_autonomous_action(session_settings: dict, state: dict, step: int, model=None, feature_names=None) -> int:
    """
    Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
    Devuelve el action_id decidido.
    """
    print("\nDECIDING ACTION...")
    decider = session_settings["decider"]
    if decider == "scripted":
        print("Using scripted policy to decide action...")
        action_id = scripted_policy_decide_action(state, step)
    elif decider == "model":
        print("Using model-based policy to decide action...")
        action_id = model_policy_decide_action(state, model, feature_names)
    elif decider == "rules":
        print("Using rules-based policy to decide action...")
        action_id = rules_policy_decide_action(state)
    else:
        raise ValueError(f"Invalid decider type: {decider}")

    if action_id not in ACTIONS:
        print(f"Invalid decider type: {session_settings['decider']}, defaulting to 0 (NONE)")
        return 0
    
    return action_id

def update_kb_with_events(kb: dict, events, result: dict, action_id: int, action_name: str, step: int) -> dict:
    """
    Actualiza la KB con los eventos generados a partir del resultado de ejecutar un comando.
    Devuelve la KB actualizada.
    """
    print("\nUPDATING KB WITH EVENTS...")

    kb = update_kb(kb, events)

    kb.setdefault("commands", [])
    if result.get("cmd"):
        kb["commands"].append(result["cmd"])

    kb["step_idx"] = step
    kb["last_action_id"] = action_id
    kb["last_action_name"] = action_name
    kb["last_rc"] = result.get("rc")
    kb["last_event_type"] = events[0].get("type") if events else None


    return kb


def print_autonomous_progress(progress: dict):
    if progress["has_progress"]:
        print(
            f"PROGRESS: +hosts={progress['new_hosts_count']} "
            f"+ports={progress['new_ports_count']} "
            f"+services={progress['new_services_count']} "
            f"+findings={progress['new_findings_count']}"
        )
    else:
        print("NO PROGRESS")

# ============================================================================================================

def new_session_observation(session_settings: dict, session_info: dict, paths: Paths) -> None:
    """
    Modo observación: el sistema muestra el estado actual y sugiere una acción 
    (usando la política elegida),
    pero no la ejecuta automáticamente. En su lugar, muestra la acción sugerida al pentester 
    y le da la opción de aceptarla (ejecutar el comando sugerido) o escribir un comando 
    alternativo. El sistema registra la acción sugerida, la acción final ejecutada 
    (si es diferente) y el resultado de la ejecución para análisis posterior.
    """
    print("Starting observation session logic...")

    # Setup inicial similar a modo autónomo: inicializar sesión_context, KB, cargar modelo si es necesario, etc.
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]  
    session_id = session_info["session_id"]
    dataset_dir = paths.datasets_dir / session_id

    print(f"Session ID: {session_info['session_id']}")
    
    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n--- Step {t} ---")

        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Pentestir elige acción (ID) o mete comando directo
        raw = input("OBS> action_id (num) OR type a command (0 stop)> ").strip()

        # El pentester quiere parar la sesión
        if raw == "0":
            print("Stopping session as per user request.")
            action_id = 0
            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
            command_to_run = None
            break

        # El pentester ha decidido ejecutar una acción predefinida (action_id) y el sistema construye el comando a ejecutar con command_builder(action_id, kb)
        elif raw.isdigit():
            print("Interpreting input as action ID...")
            action_id = int(raw)
            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
            command_to_run = command_builder(action_id, kb)
            print(f"Built command from action: {command_to_run}")
        
        # El pentester ha decidido escribir un comando libre (raw) y el sistema lo ejecuta tal cual (sin pasar por command_builder ni acciones predefinidas)
        else:
            print("Interpreting input as freeform command...")
            # Comando directo (sin acción)
            # action_id = -1
            # action_name = "USER_COMMAND"
            # cmd_template = None
            command_to_run = raw
            try:
                print("Trying to extract action_id from freeform command for logging...")
                action_id = extract_action_id_from_cmd(command_to_run)
                print(f"Extracted action_id: {action_id} from command: {command_to_run}")
            except Exception:
                print("Error extracting action_id from command, treating as freeform.")
                action_id = None

            if action_id is None:
                print("No match -> FREEFORM (not added to dataset)")
                log_freeform_row(session_dir, session_info["session_id"], {
                    "type": "FREEFORM",
                    "t": t,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "state": state,
                    "cmd": command_to_run,
                })
                continue

            action_name, _ = ACTIONS.get(action_id, ("UNKNOWN", None))

        # ---- aquí ya tienes (state, action_id) => DATASET PURO
        print(f"Logging dataset row for state and action...")
        log_dataset_row(session_dir, session_id, dataset_dir, {
            # "schema": "penhackit.bc.v1",
            "t": t,
            # "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": state,
            "action_id": action_id,
        })

        print(f"Decided action: {action_name} (ID: {action_id})")
        print(f"Command to run: {command_to_run}")

        # EJECUCIÓN
        rc, stdout, stderr = execute_command(command_to_run)
        result = {
            "rc": rc,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": command_to_run
        }
        # Para coherencia en los logs
        command = command_to_run

        log_command_output(session_dir, session_id, action_id, action_name, result)

        # PARSEAR RESULTADO Y ACTUALIZAR KB
        events = parse_command_result(action_name, result)
        print(f"Events generated from command result: {events}")

        # MEMORY (KB) UPDATE
        kb = update_kb(kb, events)

        kb.setdefault("commands", [])
        if result["cmd"]:
            kb["commands"].append(result["cmd"])

        kb["step_idx"] = t
        kb["last_action_id"] = action_id
        kb["last_action_name"] = action_name
        kb["last_rc"] = result.get("rc")
        kb["last_event_type"] = events[0].get("type") if events else None

        print(f"Updated KB: {kb}")
        save_kb(session_dir, kb)

        # Logging del paso completo (estado, acción, comando, resultado) para trazabilidad y posible entrenamiento futuro
        log_step(session_dir, session_id,{
        "t": t,
        "state": state,
        "action_id": action_id,
        "command": command,
        })

        time.sleep(0.5)

    print("Session finished loop")


def new_session_suggestion(session_settings: dict, session_info: dict, paths: Paths) -> None:
    """
    Modo sugerencia: el sistema sugiere una acción (usando la política elegida) 
    pero no la ejecuta automáticamente.
    En su lugar, muestra la acción sugerida al pentester y le da la opción de aceptarla 
    (ejecutar el comando sugerido) o escribir un comando alternativo.
    """
    print("Starting suggestion session logic...")

    # Setup inicial similar a modo autónomo: inicializar sesión_context, KB, cargar modelo si es necesario, etc. desde session_info.
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]
    model = session_info["model"]
    fn = session_info["feature_names"]

    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n--- Step {t} ---")

        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        if session_settings["decider"] == "scripted":
            print("Using scripted policy to decide action...")
            action_id = policy_decide_action(state, t)
        elif session_settings["decider"] == "model":
            print("Using model-based policy to decide action...")
            action_id = model_policy_decide_action(state, model, fn)
        elif session_settings["decider"] == "rules":
            print("Using rules-based policy to decide action...")
            action_id = rules_policy_decide_action(state)

        # action_id = 1  # INSPECT_IPCONFIG
        action_name, _ = ACTIONS.get(action_id, ("NONE", None))
        prev_kb = copy.deepcopy(kb)

        print(f"Decided action: {action_name} (ID: {action_id})")

        command = command_builder(action_id, kb)
        print(f"Built command: {command}")

        # Diferencias con modo autónomo (inicio)
        print(f"SUGGESTED: {action_name} (ID: {action_id})")
        user_cmd = input("Enter=run suggested | type cmd=override | 0=stop > ").strip()

        if user_cmd == "0":
            break
        elif user_cmd == "":
            command_to_run = command
            accepted = True
        else:
            command_to_run = user_cmd
            accepted = False

        kb["last_suggested_action_id"] = action_id
        kb["last_suggested_action_name"] = action_name
        kb["last_suggested_command"] = command
        kb["last_accepted_suggestion"] = accepted
        # Diferencias con modo autónomo (fin)

        result = execute_command(command_to_run)
        log_command_output(session_info["session_dir"], session_info["session_id"], action_id, action_name, result)

        events = parse_command_result(action_name, result)
        print(f"Events generated from command result: {events}")

        kb = update_kb(kb, events)

        kb.setdefault("commands", [])
        if result.get("cmd"):
            kb["commands"].append(result["cmd"])

        kb["step_idx"] = t
        kb["last_action_id"] = action_id
        kb["last_action_name"] = action_name
        kb["last_rc"] = result.get("rc")
        kb["last_event_type"] = events[0].get("type") if events else None

        print(f"Updated KB: {kb}")
        save_kb(session_info["session_dir"], kb)

        progress = compute_kb_progress_simple(prev_kb, kb)
        if progress["has_progress"]:
            print(
                f"PROGRESS: +hosts={progress['new_hosts_count']} "
                f"+ports={progress['new_ports_count']} "
                f"+services={progress['new_services_count']} "
                f"+findings={progress['new_findings_count']}"
            )
        else:
            print("NO PROGRESS")

        log_step(session_info["session_dir"], session_info["session_id"], {
            "t": t,
            "state": state,
            "action_id": action_id,
            "command": command_to_run,
        })

        time.sleep(1)
    
    print("Session finished loop")
            
def step_session(session_info: dict) -> dict:
    """
    Ejecuta un paso de la sesión: decide acción, construye comando, ejecuta, parsea resultado, actualiza KB, etc.

    Devuelve un dict con:
      - action_id: int
      - action_name: str
      - cmd: str o None
      - cmd_result: dict con rc, stdout, stderr (si se ejecutó comando)
      - events: list de dicts con eventos extraídos del resultado para actualizar la KB
    """
    print("Stepping through session logic...")
