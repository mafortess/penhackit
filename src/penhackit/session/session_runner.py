import copy
import time
from time import perf_counter

from penhackit.common.paths import Paths

from penhackit.session.command.command_builder import command_builder
from penhackit.session.parser.parser_mapping import parse_command_result
from penhackit.session.kb.kb_updater import enrich_events_with_execution_context, update_kb, compute_kb_progress_simple, save_kb
from penhackit.session.kb.kb_progress import compute_kb_progress_simple, print_autonomous_progress
from penhackit.session.state.state_builder import build_state
from penhackit.session.action.command_mapping import extract_action_id_from_cmd
from penhackit.session.action.action_catalog import ACTIONS

from penhackit.session.focus.focus_manager import update_focus

from penhackit.session.decision.policies import  scripted_policy_decide_action, model_policy_decide_action, rules_policy_decide_action
from penhackit.session.decision.stop_policy import evaluate_goal_and_stop

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
    
    # ------------------------------
    # Initialize KB (knowledge base)
    # ------------------------------
    print("Initial KB:")
    print(kb)
    
    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n==========================")
        print(f"-------- Step {t+1} --------")

        # Take timestamp for step start to compute active time later in online summary
        step_start = perf_counter()
        
        prev_kb = copy.deepcopy(kb)

        # ------------------------------
        # Focus
        # ------------------------------
        kb = update_focus(kb=kb, session_context=session_context, session_config=session_config)
        print(f"FOCUS: {kb.get('focus')}")
        
        # ------------------------------
        # State
        # ------------------------------
        # First step: construir representación del estado actual a partir de la KB y el contexto de la sesión para alimentar a la política/modelo para la toma de decisiones.
        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making.
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")


        # ------------------------------
        # Decide action
        # ------------------------------
        # Decide action_id based on the chosen policy (scripted, model-based, rules-based) for autonomous mode.
        action_id = decide_auto_sug_action(session_settings=session_settings, state=state, step=t, model=model, feature_names=fn)

        action_id = sanitize_decided_action(action_id, state, kb)

        dataset_row = {
            "session_id": session_id,
            "t": t,
            "state": state,
            "action_id": action_id,
        }

        log_dataset_row(session_dir=session_dir, session_id=session_id, dataset_dir=paths.datasets_dir, row=dataset_row)


        # ------------------------------
        # STOP BRANCH
        # ------------------------------
        if action_id == 0:
            duration_seconds = perf_counter() - step_start

            outcome = build_step_outcome(events=[], progress=False, result={"rc": 0},
                previous_action_id=online_summary.get("_previous_action_id"),
                current_action_id=0, duration_seconds=duration_seconds)

            outcome["goal_reached"] = False
            outcome["should_stop"] = True
            outcome["stop_reason"] = "policy_stop"

            update_online_summary(summary=online_summary, action_id=0, outcome=outcome,)

            step_record = {
                "type": "STEP",
                "t": t,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),

                # Flat compatibility fields
                "state": state,
                "action_id": action_id,
                "executed_action_id": 0,
                "action_name": "STOP",
                "command": None,
                "rc": 0,

                # Structured fields
                "decision": {
                    "requested_action_id": action_id,
                },
                "execution": {
                    "executed_action_id": 0,
                    "action_name": "STOP",
                    "command": None,
                    "command_log_ref": None,
                },
                "outcome": outcome,
                "stop_reason": "policy_stop",
            }
                    
            log_step(session_dir, session_id, step_record)

            save_kb(session_dir, kb)

            stop_reason = "policy_stop"
            print("STOP action selected. Finishing session loop.")
            break

        # ------------------------------
        # Execute action and get result
        # ------------------------------
        # Based on the decided action_id, build the command to execute using command_builder(action_id, kb).
        execution = execute_autonomous(action_id, kb)

        executed_action_id = execution["executed_action_id"]
        action_name = execution["executed_action_name"]
        command = execution["command"]
        result = execution["result"]
        events = execution["events"]
        command_ctx = execution.get("command_ctx", {})
        
        print(f"Events generated from command result: {events}")

        log_command_output(session_dir, session_id, executed_action_id, action_name, result, t=t)


        # ========================================================
        # Update KB with events
        # ========================================================
        # Update KB
        kb = update_kb_with_events(kb=kb, events=events, result=result, action_id=executed_action_id, action_name=action_name, step=t, command_ctx=command_ctx)

        # print(f"Updated KB: {kb}")

        # Update sessión_context       
        progress = compute_kb_progress_simple(prev_kb, kb)
        print_autonomous_progress(progress)

        duration_seconds = perf_counter() - step_start
        action_for_metrics = executed_action_id

        outcome = build_step_outcome(events=events, progress=progress,result=result,
            previous_action_id=online_summary.get("_previous_action_id"),
            current_action_id=action_for_metrics,
            duration_seconds=duration_seconds,
        )


        goal_status = evaluate_goal_and_stop(kb=kb, session_context=session_context, session_config=session_config, outcome=outcome)

        outcome["goal_reached"] = goal_status["goal_reached"]
        outcome["should_stop"] = goal_status["should_stop"]
        outcome["stop_reason"] = goal_status["stop_reason"]

        update_online_summary(summary=online_summary, action_id=action_for_metrics,outcome=outcome)

        command_value = result.get("cmd") or result.get("command") or command
        
        step_record = {
            "type": "STEP",
            "t": t,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": state,
            "action_id": action_id,
            "executed_action_id": executed_action_id,
            "action_name": action_name,
            "command": command_value,
            "rc": result.get("rc"),
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
            "stop_reason": outcome.get("stop_reason"),
        }

        # Save/log step(state, action, command, result, kb, context, state)
        # log_step(session_dir, session_id, {"t": t, "state": state, "action_id": action_id, "command": result.get("cmd")})
        log_step(session_dir, session_id, step_record)

        save_kb(session_dir, kb)

        if goal_status["should_stop"]:
            print(f"Stop condition reached: {goal_status['stop_reason']}")
            stop_reason = goal_status["stop_reason"]
            break
        else:
            print(f"Continuing to next step. Goal status: {goal_status}")

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

        # ------------------------------
        # Focus
        # ------------------------------
        kb = update_focus(
            kb=kb,
            session_context=session_context,
            session_config=session_config,
        )
        print(f"FOCUS: {kb.get('focus')}")

        # ------------------------------
        # State
        # ------------------------------
        print(f"STATE:")
        # Build state representation from KB and session context to feed into the policy/model for decision making.
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # ------------------------------
        # Manual decision
        # ------------------------------
        # Pentestir elige acción (ID) o mete comando directo
        raw = input("OBS> action_id (num) OR type a command (0 stop)> ").strip()


        # ------------------------------
        # STOP BRANCH
        # ------------------------------
        # El pentester quiere parar la sesión
        if raw == "0":
            duration_seconds = perf_counter() - step_start
            outcome = build_step_outcome(events=[], progress=False, result={"rc": 0},
                previous_action_id=online_summary.get("_previous_action_id"),
                current_action_id=0, duration_seconds=duration_seconds)
            outcome["goal_reached"] = False
            outcome["should_stop"] = True   
            outcome["stop_reason"] = "user_stop"

            update_online_summary(summary=online_summary, action_id=0, outcome=outcome,)
            step_record = {
                "type": "STEP",
                "t": t,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "state": state,
                "action_id": 0,
                "executed_action_id": 0,
                "action_name": "STOP",
                "command": None,
                "rc": 0,
                "decision": {
                    "requested_action_id": 0,
                },
                "execution": {
                    "executed_action_id": 0,
                    "action_name": "STOP",
                    "command_log_ref": None,
                },
                "outcome": outcome,
                "stop_reason": "user_stop",
            }
            log_step(session_dir, session_id, step_record)
            save_kb(session_dir, kb)   

            stop_reason = "user_stop"
            print("Stopping session as per user request.")
            break


        action_id = None
        action_data = None
        action_name = "USER_COMMAND"
        command_to_run = None

        # ------------------------------
        # Action ID mode
        # ------------------------------
        # El pentester ha decidido ejecutar una acción predefinida (action_id) y el sistema construye el comando a ejecutar con command_builder(action_id, kb)
        if raw.isdigit():
            print("Interpreting input as action ID...")
            action_id = int(raw)
            action_data = ACTIONS.get(action_id)

            if not action_data:
                print(f"Invalid action ID: {action_id}. Treating input as freeform command.")
                action_id = None

            if isinstance(action_data, tuple):
                action_name, cmd_template = action_data
                action_data = {
                    "name": action_name,
                    "cmd_template": cmd_template,
                }
            else:
                action_name = action_data.get("name", f"ACTION_{action_id}")
                cmd_template = action_data.get("cmd_template")

            command_ctx = command_builder(action_id, kb)
            print(f"Built command from action: {command_ctx}")
        
            if not command_ctx:
                print(f"Failed to build command for action ID: {action_id}. Treating input as freeform command.")
                action_id = None

            command_to_run = command_ctx.get("command_ctx")

            if not command_to_run:
                print(f"Command builder did not return a command to run for action ID: {action_id}. Treating input as freeform command.")
                continue

            print(f"Decided to run command for action ID {action_id}: {command_to_run}")

        # ------------------------------
        # Freeform command mode
        # ------------------------------
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

        # print(f"Updated KB: {kb}")
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

        # print(f"Updated KB: {kb}")
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

def sanitize_decided_action(action_id: int, state: dict, kb: dict) -> int:
    """
    Evita que el modelo repita acciones ya completadas y fuerza el avance
    mínimo del flujo: contexto local -> recon -> ataque -> stop.
    """

    if state.get("should_stop_now") or state.get("goal_obtained"):
        return 0

    # ============================================================
    # Bootstrap local
    # ============================================================

    if not state.get("done_inspect_hostname"):
        if action_id == 100:
            return 100

    if not state.get("done_inspect_ip_a"):
        return 101

    if not state.get("done_inspect_ip_r"):
        return 102

    if not state.get("done_inspect_ip_neigh"):
        return 103

    # ============================================================
    # Host recon
    # ============================================================

    if state.get("target_type") == "host":
        if not state.get("done_ping"):
            return 105

    if state.get("target_type") == "network":
        if not state.get("done_host_discovery"):
            return 200

     # Escaneo por host. Estas acciones pueden repetirse en network,
    # pero no sobre el mismo host si ya están hechas.
    if action_already_done_in_current_scope(action_id, kb):
        next_action = choose_next_attack_action(state)

        if next_action is not None:
            return next_action

        return 0

    return action_id


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
        print(
            f"Invalid action_id returned by policy: {action_id}. "
            f"Decider={session_settings['decider']}. "
            f"Defaulting to 0 (STOP/NONE)."
        )
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
        "phase": command_ctx.get("phase"),
        "host_id": command_ctx.get("host_id"),
        "port_id": command_ctx.get("port_id"),
        "service_id": command_ctx.get("service_id"),
        "vulnerability_id": command_ctx.get("vulnerability_id"),
        "credential_id": command_ctx.get("credential_id"),
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
        "command_ctx": command_ctx,
    }


# def execute_observation(action_id: int, kb: dict) -> tuple:



# def execute_observation(action_id: int, kb: dict) -> tuple:


def update_kb_with_events(kb: dict, events, result: dict, action_id: int, action_name: str, step: int, command_ctx: dict | None = None) -> dict:
    """
    Enriquece los eventos con contexto de ejecución y actualiza la KB.

    Esta función actúa como adaptador entre el loop de sesión y el updater puro
    de la KB. No debe guardar stdout/stderr completo ni duplicar estructuras
    antiguas como kb["commands"].
    """
    print("\nUPDATING KB WITH EVENTS...")

    command_ctx = command_ctx or {}

    enriched_events = enrich_events_with_execution_context(
        events=events,
        t=step,
        executed_action_id=action_id,
        action_name=action_name,
        result=result,
        command_ctx=command_ctx,
    )

    kb = update_kb(kb, enriched_events)

    kb.setdefault("last", {})
    kb["last"]["step_idx"] = step + 1
    kb["last"]["action_id"] = action_id
    kb["last"]["action_name"] = action_name
    kb["last"]["rc"] = result.get("rc")
    kb["last"]["success"] = None
    kb["last"]["event_types"] = [
        ev.get("type")
        for ev in enriched_events
        if ev.get("type")
    ]

    return kb


def is_action_already_done(action_id: int, state: dict) -> bool:
    action_to_flag = {
        100: "done_inspect_hostname",
        101: "done_inspect_ip_a",
        102: "done_inspect_ip_r",
        103: "done_inspect_ip_neigh",
        105: "done_ping",
        200: "done_host_discovery",
        210: "done_top_portscan",
        211: "done_full_portscan",
        220: "done_service_detection",

        330: "done_ftp_banner",
        331: "done_ftp_anonymous",
        332: "done_ftp_nmap_scripts",
        413: "done_ftp_vuln_check",

        320: "done_smb_shares",
        321: "done_smb_basic_enum",
        322: "done_smb_null_users",
        323: "done_smb_os_discovery",
        324: "done_smb_protocols",
        410: "done_smb_vuln_check",

        340: "done_ssh_banner",
        341: "done_ssh_nmap_scripts",

        371: "done_postgres_info",
        523: "done_postgres_creds_check",

        400: "done_service_version_vulns",
        401: "done_nmap_vuln_scripts",

        520: "done_ssh_creds_manual",
        521: "done_telnet_creds_manual",
        611: "done_ssh_creds_msf",
        612: "done_ftp_creds_msf",
        613: "done_ftp_creds_hydra",
        614: "done_ftp_creds_manual",

        600: "done_exploit_samba",
        601: "done_exploit_vsftpd_msf",
        602: "done_exploit_distcc",
        604: "done_exploit_postgres",
        605: "done_exploit_unreal_ircd",
        606: "done_exploit_ingreslock",
        610: "done_exploit_vsftpd_manual",
    }

    flag = action_to_flag.get(action_id)

    if not flag:
        return False

    return bool(state.get(flag))


def choose_next_attack_action(state: dict) -> int:
    """
    Fallback simple para avanzar cuando el modelo repite una acción ya hecha.
    No sustituye al modelo; solo evita bucles tontos.
    """

    if state.get("should_stop_now") or state.get("goal_obtained"):
        return 0

    # VSFTPD
    if state.get("host_has_vsftpd_234") or state.get("current_service_is_vsftpd_234"):
        if not state.get("done_ftp_banner"):
            return 330
        if not state.get("done_ftp_vuln_check"):
            return 413
        if not state.get("done_exploit_vsftpd_msf"):
            return 601
        return 0

    # Samba
    if state.get("host_has_samba"):
        if not state.get("done_smb_shares"):
            return 320
        if not state.get("done_smb_vuln_check"):
            return 410
        if not state.get("done_exploit_samba"):
            return 600
        return 0

    # DistCC
    if state.get("host_has_distcc") or state.get("host_has_port_3632"):
        if not state.get("done_service_version_vulns"):
            return 400
        if not state.get("done_exploit_distcc"):
            return 602
        return 0

    # PostgreSQL
    if state.get("host_has_postgres") or state.get("host_has_port_5432"):
        if not state.get("done_postgres_info"):
            return 371
        if not state.get("done_postgres_creds_check"):
            return 523
        if not state.get("done_exploit_postgres"):
            return 604
        return 0

    # UnrealIRCd
    if state.get("host_has_unreal_ircd") or state.get("host_has_port_6667"):
        if not state.get("done_service_version_vulns"):
            return 400
        if not state.get("done_exploit_unreal_ircd"):
            return 605
        return 0

    # Ingreslock
    if state.get("host_has_ingreslock") or state.get("host_has_port_1524"):
        if not state.get("done_exploit_ingreslock"):
            return 606
        return 0

    # SSH creds
    if state.get("host_has_ssh") or state.get("host_has_port_22"):
        if not state.get("done_ssh_creds_msf"):
            return 611
        return 0

    # Telnet creds
    if state.get("host_has_telnet") or state.get("host_has_port_23"):
        if not state.get("done_telnet_creds_manual"):
            return 521
        return 0

    # FTP weak creds
    if state.get("host_has_ftp") or state.get("host_has_port_21") or state.get("host_has_port_2121"):
        if not state.get("done_ftp_banner"):
            return 330
        if not state.get("done_ftp_creds_hydra"):
            return 613
        if not state.get("done_ftp_creds_manual"):
            return 614
        return 0

    return 

def action_already_done_in_current_scope(action_id: int, kb: dict) -> bool:
    """
    Comprueba si una acción ya se ha ejecutado en el mismo contexto operativo.

    Para acciones globales, se compara solo action_id.
    Para acciones sobre host/servicio/vulnerabilidad, se compara:
        action_id + host_id + port_id + service_id + vulnerability_id
    """

    current_scope = get_current_action_scope(kb)

    for event in kb.get("history", []):
        event_action_id = event.get("action_id") or event.get("executed_action_id")

        if safe_int(event_action_id) != int(action_id):
            continue

        previous_scope = get_event_action_scope(event)

        if is_global_action(action_id):
            return True

        if scopes_match(current_scope, previous_scope):
            return True

    for attempt in kb.get("attempts", {}).values():
        attempt_action_id = attempt.get("action_id") or attempt.get("executed_action_id")

        if safe_int(attempt_action_id) != int(action_id):
            continue

        previous_scope = get_event_action_scope(attempt)

        if is_global_action(action_id):
            return True

        if scopes_match(current_scope, previous_scope):
            return True

    return False


def get_current_action_scope(kb: dict) -> dict:
    focus = kb.get("focus", {})

    return {
        "host_id": focus.get("host_id"),
        "port_id": focus.get("port_id"),
        "service_id": focus.get("service_id"),
        "vulnerability_id": focus.get("vulnerability_id"),
        "credential_id": focus.get("credential_id"),
    }


def get_event_action_scope(record: dict) -> dict:
    host_id = record.get("host_id")
    port_id = record.get("port_id")
    service_id = record.get("service_id")
    vulnerability_id = record.get("vulnerability_id")
    credential_id = record.get("credential_id")

    host = record.get("host") or record.get("target_ip")
    port = record.get("port") or record.get("target_port")

    if not host_id and host:
        host_id = f"host:{host}"

    if not port_id and host and port:
        port_id = f"port:{host}:tcp:{int(port)}"

    if not service_id and host and port:
        service_id = f"svc:{host}:tcp:{int(port)}"

    return {
        "host_id": host_id,
        "port_id": port_id,
        "service_id": service_id,
        "vulnerability_id": vulnerability_id,
        "credential_id": credential_id,
    }


def scopes_match(current_scope: dict, previous_scope: dict) -> bool:
    """
    Dos acciones son repetidas solo si coinciden en el mismo ámbito útil.
    """

    current_host = current_scope.get("host_id")
    previous_host = previous_scope.get("host_id")

    if current_host and previous_host and current_host != previous_host:
        return False

    current_service = current_scope.get("service_id")
    previous_service = previous_scope.get("service_id")

    if current_service and previous_service:
        return current_service == previous_service

    current_port = current_scope.get("port_id")
    previous_port = previous_scope.get("port_id")

    if current_port and previous_port:
        return current_port == previous_port

    current_vuln = current_scope.get("vulnerability_id")
    previous_vuln = previous_scope.get("vulnerability_id")

    if current_vuln and previous_vuln:
        return current_vuln == previous_vuln

    if current_host and previous_host:
        return current_host == previous_host

    return False


def is_global_action(action_id: int) -> bool:
    """
    Acciones que no dependen de host concreto.
    Estas sí deben ejecutarse una sola vez por sesión.
    """

    return int(action_id) in {
        100,  # hostname
        101,  # ip a
        102,  # ip route
        103,  # ip neigh
        200,  # host discovery de red
    }


def safe_int(value, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default