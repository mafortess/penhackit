import copy
import time
from time import perf_counter

from penhackit.common.paths import Paths

from penhackit.session.command.command_builder import command_builder
from penhackit.session.parser.parser_mapping import parse_command_result
from penhackit.session.kb.kb_updater import update_kb, compute_kb_progress_simple, save_kb
from penhackit.session.state.state_builder import build_state
from penhackit.session.action.command_mapping import extract_action_id_from_cmd
from penhackit.session.action.action_catalog import ACTIONS

from penhackit.session.decision.policies import  scripted_policy_decide_action, model_policy_decide_action, rules_policy_decide_action

from penhackit.session.execution.execute import execute_command

from penhackit.session.logging.logger import log_command_output, log_step, log_dataset_row, log_freeform_row, init_online_summary, finish_online_summary, update_online_summary, build_step_outcome


def run_session_loop(session_settings: dict, session_info: dict, paths: Paths):
    mode = session_settings["mode"]

    # Dependiendo del modo de la sesión, ejecuta la lógica correspondiente (autonomous, observation, suggestion).
    if mode == "autonomous":
        run_session_autonomous(session_settings, session_info, paths)
        return
    elif mode == "observation":
        run_session_observation(session_settings, session_info, paths)
        return
    elif mode == "suggestion":
        run_session_suggestion(session_settings, session_info, paths)
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
    print("\nStarting autonomous session logic...")

    # Setup inicial similar a modo autónomo: inicializar sesión_context, KB, cargar modelo si es necesario, etc.
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_id = session_info["session_id"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]
    model = session_info["model"]
    fn = session_info["feature_names"]

    print(f"Session ID: {session_id}")
    online_summary = init_online_summary(session_id, session_settings, session_info)
    stop_reason = "max_steps_reached"  # valor por defecto, se actualizará si se alcanza la meta o se para por otro motivo
    # Initialize KB (knowledge base)
    print("Initial KB:")
    print(kb)
    
    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n==========================")
        print(f"-------- Step {t+1} --------")

        # Take timestamp for step start to compute active time later in online summary
        step_start = perf_counter()
        
        prev_kb = copy.deepcopy(kb)


        # First step: construir representación del estado actual a partir de la KB y el contexto de la sesión para alimentar a la política/modelo para la toma de decisiones.
        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making.
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
        action_id = decide_auto_sug_action(session_settings=session_settings, state=state, step=t, model=model, feature_names=fn)

        if action_id == 0:
            duration_seconds = perf_counter() - step_start

            outcome = build_step_outcome(
                events=[],
                progress=False,
                result={"rc": 0},
                previous_action_id=online_summary.get("_previous_action_id"),
                current_action_id=0,
                duration_seconds=duration_seconds,
            )

            update_online_summary(
                summary=online_summary,
                action_id=0,
                outcome=outcome,
            )

            log_step(
                session_dir,
                session_id,
                {
                    "type": "STEP",
                    "t": t,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "state": state,
                    "decision": {
                        "requested_action_id": action_id,
                    },
                    "execution": {
                        "executed_action_id": 0,
                        "action_name": "STOP",
                        "command_log_ref": None,
                    },
                    "outcome": outcome,
                    "stop_reason": "policy_stop",
                },
            )

            stop_reason = "policy_stop"
            break
            print("STOP action selected. Finishing session loop.")
            break

        # Based on the decided action_id, build the command to execute using command_builder(action_id, kb).
        execution = execute_autonomous(action_id, kb)

        executed_action_id = execution["executed_action_id"]
        action_name = execution["executed_action_name"]
        command = execution["command"]
        result = execution["result"]
        events = execution["events"]

        print(f"Events generated from command result: {events}")

        log_command_output(session_dir, session_id, executed_action_id, action_name, result, t=t)
     
        # Update KB
        kb = update_kb_with_events(kb, events, result, executed_action_id, action_name, t)

        print(f"Updated KB: {kb}")
        save_kb(session_dir, kb)

        # Update sessión_context       
        progress = compute_kb_progress_simple(prev_kb, kb)
        print_autonomous_progress(progress)

        duration_seconds = perf_counter() - step_start

        action_for_metrics = executed_action_id

        outcome = build_step_outcome(
            events=events,
            progress=progress,
            result=result,
            previous_action_id=online_summary.get("_previous_action_id"),
            current_action_id=action_for_metrics,
            duration_seconds=duration_seconds,
        )

        update_online_summary(
            summary=online_summary,
            action_id=action_for_metrics,
            outcome=outcome,
        )

        # Save/log step(state, action, command, result, kb, context, state)
        # log_step(session_dir, session_id, {"t": t, "state": state, "action_id": action_id, "command": result.get("cmd")})
        log_step(
            session_dir,
            session_id,
            {
                "type": "STEP",
                "t": t,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "state": state,
                "decision": {
                    "requested_action_id": action_id,
                },
                "execution": {
                    "executed_action_id": executed_action_id,
                    "action_name": action_name,
                    "command_log_ref": {
                        "file": "command_outputs.jsonl",
                        "t": t,
                    },
                },
                "outcome": outcome,
            },
        ) 

        if outcome["goal_reached"]:
            print("Goal reached. Finishing session loop.")
            stop_reason = "goal_reached"
            break

        time.sleep(1)
    
    summary = finish_online_summary(session_dir, online_summary, stop_reason)

    print("\nOnline session summary")
    print("----------------------")
    print(f"Success: {summary['success']}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Steps total: {summary['steps_total']}")
    print(f"Steps to goal: {summary['steps_to_goal']}")
    print(f"Progress rate: {summary['progress_rate']:.4f}")
    print(f"Repeated action rate: {summary['repeated_action_rate']:.4f}")
    print(f"Tool error rate: {summary['tool_error_rate']:.4f}")
    print(f"Active time: {summary['active_time_seconds']:.4f} s")
    print(f"Wall time: {summary['wall_time_seconds']:.4f} s")

    print("Session finished loop")


def run_session_observation(session_settings: dict, session_info: dict, paths: Paths) -> None:
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
    session_id = session_info["session_id"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]  
    session_id = session_info["session_id"]
    dataset_dir = paths.datasets_dir / session_id

    print(f"Session ID: {session_id}")
    online_summary = init_online_summary(session_id, session_settings, session_info)
    stop_reason = "max_steps_reached"  # valor por defecto, se actualizará si se alcanza la meta o se para por otro motivo

    # Initialize KB (knowledge base)
    print("Initial KB:")
    print(kb)

    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n==========================")
        print(f"\n--- Step {t+1} ---")

        # Take timestamp for step start to compute active time later in online summary
        step_start = perf_counter()

        prev_kb = copy.deepcopy(kb)

        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making.
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

        # PARSEAR RESULTADO, PARA GENERAR EVENTOS Y ACTUALIZAR KB
        events = parse_command_result(action_name, result)
        print(f"Events generated from command result: {events}")

        # Update KB with events
        kb = update_kb_with_events(kb, events, result, action_id, action_name, t)

        print(f"Updated KB: {kb}")
        save_kb(session_dir, kb)

        # Logging del paso completo (estado, acción, comando, resultado) para trazabilidad y posible entrenamiento futuro
        log_step(session_dir, session_id,{"t": t, "state": state, "action_id": action_id, "command": command,})

        time.sleep(1)

    summary = finish_online_summary(session_dir, online_summary, stop_reason)

    print("\nOnline session summary")
    print("----------------------")
    print(f"Success: {summary['success']}")
    print(f"Stop reason: {summary['stop_reason']}")
    print(f"Steps total: {summary['steps_total']}")
    print(f"Steps to goal: {summary['steps_to_goal']}")
    print(f"Progress rate: {summary['progress_rate']:.4f}")
    print(f"Repeated action rate: {summary['repeated_action_rate']:.4f}")
    print(f"Tool error rate: {summary['tool_error_rate']:.4f}")
    print(f"Active time: {summary['active_time_seconds']:.4f} s")
    print(f"Wall time: {summary['wall_time_seconds']:.4f} s")
    print(f"Saved online summary: {session_dir / 'online_summary.json'}")

    print("Session finished loop")


def run_session_suggestion(session_settings: dict, session_info: dict, paths: Paths) -> None:
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
    session_id = session_info["session_id"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]
    model = session_info["model"]
    fn = session_info["feature_names"]

    print(f"Session ID: {session_id}")

    # Initialize KB (knowledge base)
    print("Initial KB:")
    print(kb)

    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n==========================")
        print(f"-------- Step {t+1} --------")

        prev_kb = copy.deepcopy(kb)

        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
        action_id = decide_auto_sug_action(session_settings=session_settings, state=state, step=t, model=model, feature_names=fn)

    
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
        log_command_output(session_dir, session_id, action_id, action_name, result)

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

        log_step(session_info["session_dir"], session_info["session_id"], {"t": t, "state": state, "action_id": action_id, "command": command_to_run})

        time.sleep(1)
    
    print("Session finished loop")
            

# ============================================================================================================
# ============================================================================================================
# ============================================================================================================


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



# ============================================================================================================
# Helpers 

def get_action_data(action_id: int) -> tuple:
    """
    Devuelve (effective_action_id, action_data).

    Si action_id no existe en ACTIONS, usa la acción 0.
    """
    if action_id not in ACTIONS:
        print(f"Unknown action_id={action_id}. Falling back to action 0.")
        effective_action_id = 0
    else:
        effective_action_id = action_id

    action_data = ACTIONS[effective_action_id]

    return effective_action_id, action_data


def get_action_and_build_command(action_id: int, kb: dict) -> dict:
    """
    Obtiene la acción asociada a action_id y construye el command_ctx.

    Devuelve un dict con:
      - requested_action_id
      - action_id
      - action_data
      - command_ctx
    """
    print("\nCOMMAND BUILDING...")
    effective_action_id, action_data = get_action_data(action_id)

    action_name = action_data["name"]

    print(f"Decided action: {action_name} (ID: {effective_action_id})")

    command_ctx = command_builder(action_data, kb)

    print(f"Built command: {command_ctx }")

    return {
        "requested_action_id": action_id,
        "action_id": effective_action_id,
        "action_data": action_data,
        "action_name": action_name,
        "command_ctx": command_ctx,
    }


def decide_auto_sug_action(session_settings: dict, state: dict, step: int, model=None, feature_names=None) -> int:
    """
    Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
    Devuelve el action_id decidido.
    """
    print("\nDECIDING ACTION...")
    decider = session_settings["decider"]
    if decider == "scripted":
        print("Using scripted policy to decide action...")
        action_id = scripted_policy_decide_action(state, step, sequence_name=session_settings.get("scripted_sequence"))
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


def execute_command_ctx(command_ctx: dict) -> dict:
    """
    Ejecuta un command_ctx construido desde action catalog.
    Devuelve result normalizado.
    """
    if command_ctx is None:
        print("No command to execute, returning empty result.")

        return {
            "rc": None,
            "stdout": "",
            "stderr": "",
            "cmd": None
        }

    command_to_run = command_ctx["command"]

    if not command_to_run:
        return {
            "rc": None,
            "stdout": "",
            "stderr": "",
            "cmd": None,
        }
    
    print("\nACTION EXECUTION...")
    print(f"Executing command: {command_to_run}")
    rc, stdout, stderr = execute_command(command_to_run)

    return {
        "rc": rc,
        "stdout": stdout,
        "stderr": stderr,
        "cmd": command_to_run,
        "target": command_ctx.get("target"),
        "target_ip": command_ctx.get("target_ip"),
        "target_port": command_ctx.get("target_port"),
        "known_open_ports_csv": command_ctx.get("known_open_ports_csv"),
        "service_version_string": command_ctx.get("service_version_string"),
        "service_name": command_ctx.get("service_name"),
        "parser_family": command_ctx.get("parser_family"),
        "exploit": command_ctx.get("exploit"),
    }


def execute_autonomous(action_id: int, kb: dict) -> tuple:
    """
    Ejecuta el comando correspondiente al action_id dado, parsea el resultado y devuelve el resultado y los eventos generados para actualizar la KB.
    Devuelve una tupla con (action_name, result, events) donde:
      - action_name: str con el nombre de la acción ejecutada
      - result: dict con rc, stdout, stderr, cmd
      - events: list de dicts con eventos extraídos del resultado para actualizar la KB
    """
    build_info = get_action_and_build_command(action_id, kb)

    executed_action_id = build_info["action_id"]
    action_name = build_info["action_name"]
    command_ctx = build_info["command_ctx"]

    result = execute_command_ctx(command_ctx)

    if result.get("cmd") is None:
        events = []
    else:
        events = parse_command_result(action_name, result)
    
    return {
        "requested_action_id": action_id,
        "executed_action_id": executed_action_id,
        "executed_action_name": action_name,
        "command": result.get("cmd"),
        "result": result,
        "events": events,
    }


# def execute_observation(action_id: int, kb: dict) -> tuple:



# def execute_observation(action_id: int, kb: dict) -> tuple:



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
