import json
import time
from pathlib import Path

import copy

from penhackit.common.paths import Paths
from penhackit.session.command import command_builder
from penhackit.session.event.event_builder import parse_command_result
from penhackit.session.kb.kb_updater import update_kb, compute_kb_progress_simple, save_kb, build_initial_kb
from penhackit.session.state.state_builder import build_state
from penhackit.session.action.action_ids import ACTIONS, extract_action_id_from_cmd

from penhackit.session.decision.policies import policy_decide_action, model_policy_decide_action, rules_policy_decide_action
from penhackit.session.decision.model_loader import load_decision_model
from penhackit.session.kb.kb_updater import launch_kb_monitor_window_windows

from penhackit.session.execution.execute import execute_command

from penhackit.session.logging.logger import log_command_output, log_step, log_dataset_row, log_freeform_row

import numpy as np

def new_session_logic(session_settings: dict, env_profile: dict, paths: Paths) -> dict:
    """
    kb: dict con el conocimiento actual (hosts, servicios, etc.)
    session_context: dict con info de la sesión (goal_type, focus_level, etc.)
    model + feature_names: si usas policy basada en ML, el modelo cargado y el orden de features esperado.

    Retorna un dict con:
      - action_id: int
      - action_name: str
      - cmd: str o None
      - cmd_result: dict con rc, stdout, stderr (si se ejecutó comando)
      - events: list de dicts con eventos extraídos del resultado para actualizar la KB
    """
    print("Starting session logic...")

    session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + session_settings["name"].replace(" ", "_")
    session_dir = paths.sessions_dir / session_id
    
    # Crear la carpeta session_dir en el sistema de archivos.
    print(f"Creating session directory: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=False)
    
    print("Creation of session config and context files...")
    session_config = {
        "id": session_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    session_context = {
        "id": session_id,
        "mode": session_settings["mode"],
        "goal_type": session_settings["goal_type"],
        "target": session_settings["target"],
        "max_steps": session_settings["max_steps"],
    }
    
    # parents=True: si faltan carpetas “padre” en la ruta, también las crea. Ejemplo: si data/ o data/sessions/ no existen, los crea automáticamente.
    # exist_ok=True: si la carpeta ya existe, no da error. Sin esto, mkdir() lanzaría una excepción si la carpeta ya existe.
    
    # Creación de los archivos de configuración y contexto de la sesión (session_config.json y session_context.json) con los datos de la sesión.
    # 1) session_config.json (operativo)
    (session_dir / "session_config.json").write_text(
        json.dumps(session_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 2) session_context.json (tarea/objetivo)
    (session_dir / "session_context.json").write_text(
        json.dumps(session_context, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Initializing KB...")
    kb = build_initial_kb(session_id)

    # 3) kb.json (conocimiento, inicialmente vacío o con datos predeterminados)
    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Loading model for session (if applicable)...")
    model = None
    feature_names = None
    
    if session_settings["mode"] in ["autonomous", "suggestion"]:
        model_path = paths.models_dir / "decision_tree" / "model.joblib"
        metrics_path = model_path.parent / "metrics.json"
        
        if not model_path.exists() or not metrics_path.exists():
            print(f"Model files not found: {model_path}, {metrics_path}")
            print("Make sure to train a model first and place the files in mvp/models/")
            return
        
        model, feature_names = load_decision_model(model_path, metrics_path)
        print(f"Loaded model from {model_path} with features: {feature_names}")

    session_info = {
        "session_id": session_id,
        "session_dir": session_dir, 
        "session_config": session_config,
        "session_context": session_context,
        "kb": kb,
        "model": model,
        "feature_names": feature_names,
    }

    # Si la configuración de la sesión indica que se debe lanzar el monitor de KB, lo lanza pasando la ruta de session_dir para que pueda leer/escribir los archivos de KB y contexto.
    if session_settings["launch_kb_monitor"]:
        launch_kb_monitor_window_windows(session_dir)

    # Dependiendo del modo de la sesión, ejecuta la lógica correspondiente (autonomous, observation, suggestion).
    if session_settings["mode"] == "autonomous":
        return new_session_autonomous(session_settings, paths, session_info)
    elif session_settings["mode"] == "observation":
        return new_session_observation(session_settings, paths, session_info)
    elif session_settings["mode"] == "suggestion":
        return new_session_suggestion(session_settings, paths, session_info)
    
    print("Session finished")

def new_session_autonomous(session_settings: dict, paths: Paths, session_info: dict) -> dict:
    """
    Versión simple de new_session_logic que ignora la política y siempre devuelve la misma acción (para testing).
    """
    print("Starting autonomous session logic...")

    # Initatlize sessión_context
    # Initialize KB (knowledge base)

    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    session_dir = session_info["session_dir"]
    model = session_info["model"]
    fn = session_info["feature_names"]

    print(f"Session ID: {session_info['session_id']}")

    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n--- Step {t} ---")

        prev_kb = copy.deepcopy(kb)

        # Build state representation from KB and session context to feed into the policy/model for decision making.
        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Decide action_id based on the chosen policy (scripted, model-based, rules-based)
        if session_settings["decider"] == "scripted":
            print("Using scripted policy to decide action...")
            action_id = policy_decide_action(state, t)
        elif session_settings["decider"] == "model":
            print("Using model-based policy to decide action...")
            action_id = model_policy_decide_action(state, model, fn)
        elif session_settings["decider"] == "rules":
            print("Using rules-based policy to decide action...")
            action_id = rules_policy_decide_action(state)

        # Build commmand to execute based on the decided action_id, using from KB and session context. 
        # action_id = 1  # INSPECT_IPCONFIG
        action_name, _ = ACTIONS.get(action_id, ("NONE", None))
        
        print(f"Decided action: {action_name} (ID: {action_id})")

        command = command_builder(action_id, kb)
        print(f"Built command: {command}")

        command_to_run = command

        print(f"Executing command: {command_to_run}")
        rc, stdout, stderr = execute_command(command_to_run)
        result = {
            "rc": rc,
            "stdout": stdout,
            "stderr": stderr,
            "cmd": command_to_run
        }
        
        log_command_output(session_info["session_dir"], session_info["session_id"], action_id, action_name, result)

        events = parse_command_result(action_name, result)
        print(f"Events generated from command result: {events}")

        
        # Update KB
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
        save_kb(session_info["session_dir"], kb)

        # Update sessión_context
        
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

        # Save/log step(state, action, command, result, kb, context, state)
        
        log_step(session_info["session_dir"], session_info["session_id"], {
            "t": t,
            "state": state,
            "action_id": action_id,
            "command": command_to_run,
        })

        time.sleep(1)

def new_session_observation(session_settings: dict, paths: Paths, session_info: dict) -> dict:
    """
    Versión de new_session_logic que no ejecuta comandos, solo decide una acción y devuelve un evento simulado.
    Útil para testing de la parte de policy sin ejecutar comandos reales.
    """
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

def new_session_suggestion(session_settings: dict, paths: Paths, session_info: dict) -> dict:
    """
    Versión de new_session_logic que decide la acción usando un modelo de ML (si se proporciona).
    Para testing, puede usar un modelo dummy que siempre devuelve la misma acción.
    """
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
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
            
