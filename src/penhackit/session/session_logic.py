import json
import os
import re
import subprocess
import time
from pathlib import Path

import joblib
import copy

from penhackit.common.paths import Paths
from penhackit.session.command import command_builder

def new_session_logic(session_settings: dict, env_settings: dict, paths: Paths) -> dict:
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
    
    # Crear la carpeta session_dir en el sistema de archivos.
    # # parents=True: si faltan carpetas “padre” en la ruta, también las crea. Ejemplo: si data/ o data/sessions/ no existen, los crea automáticamente.
    # exist_ok=True: si la carpeta ya existe, no da error. Sin esto, mkdir() lanzaría una excepción si la carpeta ya existe.
    session_dir.mkdir(parents=True, exist_ok=True)
    
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


def load_decision_model(model_path: Path, metrics_path: Path):
    # model = joblib.load(model_path) # MAL; DA ERROR
    # with open(model_path, "r", encoding="utf-8") as f:
    #     model = json.load(f)
    try:
        model = joblib.load(model_path)  # CORRECTO; el modelo se carga correctamente usando joblib.load() en lugar de json.load(), ya que el modelo está serializado en formato joblib, no JSON.
    except Exception as e:
        print(f"Error loading model with joblib: {e}")
        raise e
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]
    except Exception as e:
        print(f"Error loading feature names from metrics: {e}")
        raise e
    
    return model, feature_names

def new_session_autonomous(session_settings: dict, paths: Paths, session_info: dict) -> dict:
    """
    Versión simple de new_session_logic que ignora la política y siempre devuelve la misma acción (para testing).
    """
    print("Starting autonomous session logic...")

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

        command_to_run = command

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

def new_session_observation(session_settings: dict, paths: Paths, session_info: dict) -> dict:
    """
    Versión de new_session_logic que no ejecuta comandos, solo decide una acción y devuelve un evento simulado.
    Útil para testing de la parte de policy sin ejecutar comandos reales.
    """
    max_steps = session_settings["max_steps"]
    kb = session_info["kb"]
    session_context = session_info["session_context"]
    session_config = session_info["session_config"]
    action_id = 1  # INSPECT_IPCONFIG
    action_name = ACTIONS.get(action_id, ("UNKNOWN", None))[0]
    cmd = command_builder(action_id, kb)

    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
        print(f"\n--- Step {t} ---")

        state = build_state(kb, session_context)
        print(f"State at step {t}: {state}")

        # Pentestir elige acción (ID) o mete comando directo
        raw = input("OBS> action_id (num) OR type a command (0 stop)> ").strip()

        # El pentester quiere parar la sesión
        if raw == "0":
            action_id = 0
            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
            command_to_run = None
            break

        # El pentester ha decidido ejecutar una acción predefinida (action_id) y el sistema construye el comando a ejecutar con command_builder(action_id, kb)
        elif raw.isdigit():
            action_id = int(raw)
            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
            command_to_run = command_builder(action_id, kb)
            print(f"Built command from action: {command_to_run}")
        
        # El pentester ha decidido escribir un comando libre (raw) y el sistema lo ejecuta tal cual (sin pasar por command_builder ni acciones predefinidas)
        else:
            # Comando directo (sin acción)
            # action_id = -1
            # action_name = "USER_COMMAND"
            # cmd_template = None
            command_to_run = raw
            action_id = extract_action_id_from_cmd(command_to_run)
            if action_id is None:
                print("No match -> FREEFORM (not added to dataset)")
                log_freeform_row(session_config["session_dir"], session_config["session_id"], {
                    "type": "FREEFORM",
                    "t": t,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "state": state,
                    "cmd": command_to_run,
                })
                continue
            action_name, _ = ACTIONS.get(action_id, ("UNKNOWN", None))

        # ---- aquí ya tienes (state, action_id) => DATASET PURO
        log_dataset_row(session_config["session_dir"], session_config["session_id"], {
            # "schema": "penhackit.bc.v1",
            "t": t,
            # "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "state": state,
            "action_id": action_id,
        })

        print(f"Decided action: {action_name} (ID: {action_id})")
        print(f"Command to run: {command_to_run}")

        # EJECUCIÓN
        result = execute_command(command_to_run)

        # Para coherencia en los logs
        command = command_to_run

        log_command_output(session_config["session_dir"], session_config["session_id"], action_id, action_name, result)

        # PARSEAR RESULTADO Y ACTUALIZAR KB
        events = parse_command_result(action_name, result)
        print(f"Events generated from command result: {events}")

        # MEMORY (KB) UPDATE
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
        save_kb(session_config["session_dir"], kb)

        # Logging del paso completo (estado, acción, comando, resultado) para trazabilidad y posible entrenamiento futuro
        log_step(session_config["session_dir"], session_config["session_id"],{
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
            

def build_initial_kb(session_id: str) -> dict:
    """
    Construye una KB inicial vacía o con datos predeterminados para el inicio de la sesión.
    """
    return {
        "session_id": session_id,
        "name_enterprise": "ITIS",
        "networks": {},
        "hosts": [],
        "services": [],
        "findings": [],
        "notes": [],
        "net": {
            "interfaces": [],
            "ipv4": [],
            "default_gw": [],
            "arp_neighbors": [],
            "routes": [],
        },
        "focus": {"level": "global", "host": "", "service": ""},
        "commands": [],
        "step_idx": 0,
        "last_action_id": None,
        "last_action_name": None,
        "last_rc": None,
        "last_event_type": None,
    }
        

# action_id -> (name, command)
ACTIONS = {
    0: ("NONE", None),

    # Windows (elige las que te interesen)
    1: ("INSPECT_IPCONFIG", "ipconfig /all"),
    2: ("INSPECT_ARP", "arp -a"),
    3: ("INSPECT_ROUTE", "route print"),
    # : ("INSPECT_NETSTAT", "netstat -ano"),

    4: ("PING_FOCUS_HOST", "ping -n 1 {ip}"),

    # Linux/Kali (si lo ejecutas allí)
    101: ("INSPECT_IP_A", "ip a"),
    102: ("INSPECT_IP_R", "ip r"),
    103: ("INSPECT_IP_NEIGH", "ip neigh"),
    104: ("INSPECT_SS", "ss -tulpn"),
}

def extract_action_id_from_cmd(cmd: str) -> int:
    """
    MVP extractor: mapea command raw -> action_id usando heurísticas simples.
    Soporta tus ACTIONS actuales.
    """
    if not cmd:
        return None

    s = cmd.strip().lower()
    s = re.sub(r"\s+", " ", s)

    # Windows (acepta "ipconfig" y "ipconfig /all")
    if s == "ipconfig" or s.startswith("ipconfig "):
        return 1
    if s == "arp" or s.startswith("arp "):
        return 2
    if s == "route" or s.startswith("route "):
        return 3
    # ping -n 1 <ipv4>
    if re.fullmatch(r"ping -n 1 (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4

    # ping -n 1 <ipv4> (y también ping <ipv4> como fallback MVP)
    if re.fullmatch(r"ping -n 1 (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    if re.fullmatch(r"ping (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    
    # Linux/Kali
    if s == "ip a":
        return 101
    if s == "ip r":
        return 102
    if s == "ip neigh":
        return 103
    if s == "ss -tulpn":
        return 104

    return None

def build_state(kb: dict, session_context: dict) -> dict:
    print("Building state from KB and session context...")

    net = kb.get("net", {}) or {}
    focus = kb.get("focus", {}) or {}

    hosts = kb.get("hosts", []) or []
    services = kb.get("services", []) or []
    findings = kb.get("findings", []) or []

    arp_neighbors = net.get("arp_neighbors", []) or []
    ipv4 = net.get("ipv4", []) or []
    default_gw = net.get("default_gw", []) or []
    interfaces = net.get("interfaces", []) or []
    routes = net.get("routes", []) or []

    state = {
        # Goal / task
        "goal_type": session_context.get("goal_type", "demo"),

        # Focus (nivel y si hay algo seleccionado)
        "focus_level": focus.get("level", "global"),
        "has_focus_host": bool(focus.get("host")),
        "has_focus_service": bool(focus.get("service")),

        # Features por “nivel” (resumen, no datos crudos)
        "net_ipv4_count": len(ipv4),
        "net_gw_count": len(default_gw),
        "net_if_count": len(interfaces),
        "net_arp_count": len(arp_neighbors),
        "net_routes_count": len(routes),

        "hosts_count": len(hosts),
        "services_count": len(services),
        "findings_count": len(findings),

        # Last transition (para que la policy no repita/pueda detectar error)
        "last_action_id": kb.get("last_action_id"),
        "last_action_name": kb.get("last_action_name"),
        "last_rc": kb.get("last_rc"),
        "last_event_type": kb.get("last_event_type"),

        # Progreso / estancamiento (mínimo)
        "step_idx": kb.get("step_idx", 0),
    }
    return state

def policy_decide_action(state, t):
    print("Deciding action based on state...")
    action = t

    return action

def command_builder(action, kb):
    print("Building command from action and KB...")
    cmd = ACTIONS.get(action, (None, None))[1]

    if not cmd:
        return None

    # Reemplaza placeholders en cmd con datos de KB (ejemplo simple)
    if "{" not in cmd:
        return cmd
    hosts = kb.get("hosts", [])
    ip = hosts[0].get("ip", None) if hosts else None  # Ejemplo: toma la primera IP de la KB
    if "{ip}" in cmd and not ip:
        print("No IP available in KB to build command.")
        return None
    cmd = cmd.format(ip=ip)  # Reemplaza {ip} en el comando

    return cmd

def execute_command(cmd):
    print(f"Executing command: {cmd} and capturing result...")
    if not cmd:  # None o "" => no ejecutar
        return {"rc": 0, "stdout": "", "stderr": "", "cmd": cmd}
    o  = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {
        "cmd": cmd,
        "rc": int(o.returncode),
        "stdout": o.stdout or "",
        "stderr": o.stderr or "",
    }

def parse_command_result(action_name: str, result: dict) -> list[dict]:
    """
    action_name: p.ej. "INSPECT_IPCONFIG", "INSPECT_ARP"
    result: {"cmd": str|None, "rc": int, "stdout": str, "stderr": str}

    Return a list of events for updating the KB. Each event is a dict with a "type" field and other relevant data.
    """
    print("Building event from command result...")

    rc = int(result.get("rc", 0))
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""

    if rc != 0:
        return [{"type": "COMMAND_ERROR", "action": action_name, "rc": rc, "stderr": (result.get("stderr", "") or "")[:500]}]

    if action_name == "INSPECT_IPCONFIG":
        # Muy simple y robusto: extrae IPv4 y Default Gateway en plano
        ipv4s = re.findall(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
        gws = re.findall(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
        ipv4s = list(dict.fromkeys(ipv4s))
        gws = list(dict.fromkeys([g for g in gws if g]))

        # Extrae interfaces (bloques) de ipconfig /all (Windows)
        # Nota: esto NO pretende ser perfecto; es suficiente para MVP.
        interfaces = []
        blocks = re.split(r"\r?\n\r?\n", stdout)
        for b in blocks:
            # Heurística: bloque que tiene IPv4 Address y algún nombre de "adapter"
            if "IPv4 Address" not in b:
                continue

            # Nombre de interfaz (línea tipo: "Ethernet adapter Ethernet:")
            name = None
            m = re.search(r"^(.*adapter.*):\s*$", b, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                name = m.group(1).strip()

            ipv4 = None
            m = re.search(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
            if m:
                ipv4 = m.group(1)

            gw = None
            m = re.search(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
            if m:
                gw = m.group(1)

            mac = None
            m = re.search(r"\bPhysical Address[^\n]*:\s*([0-9A-Fa-f\-]{11,})", b)
            if m:
                mac = m.group(1)

            interfaces.append({
                "name": name or "",
                "ipv4": ipv4 or "",
                "default_gw": gw or "",
                "mac": mac or "",
            })

        return [{
            "type": "NET_INFO",
            "ipv4": ipv4s,
            "default_gw": gws,
            "interfaces": interfaces,
        }]

    if action_name == "INSPECT_ARP":
        # arp -a (Windows): líneas típicas
        #   192.168.1.1           00-11-22-33-44-55     dynamic
        arp_neighbors = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+(?P<mac>[0-9A-Fa-f\-]{11,}|[0-9A-Fa-f:]{11,})\s+(?P<kind>\w+)",
                line,
            )
            if m:
                arp_neighbors.append({
                    "ip": m.group("ip"),
                    "mac": m.group("mac"),
                    "type": m.group("kind"),
                })

        # Fallback: si no matchea MAC, al menos extrae IPs
        if not arp_neighbors:
            ips = re.findall(r"\b((?:\d{1,3}\.){3}\d{1,3})\b", stdout)
            ips = list(dict.fromkeys(ips))
            arp_neighbors = [{"ip": ip, "mac": "", "type": ""} for ip in ips]

        return [{
            "type": "ARP_TABLE",
            "arp_neighbors": arp_neighbors,
        }]

    return [{"type": "NO_EVENT", "action": action_name}]

def update_kb(kb: dict, events: list[dict]) -> dict:
    print("Updating KB with new event...")

    # Estructura mínima esperada
    kb.setdefault("hosts", [])
    kb.setdefault("services", [])
    kb.setdefault("findings", [])
    kb.setdefault("notes", [])
    kb.setdefault("focus", {"level": "global", "host": "", "service": ""})
    kb.setdefault("commands", [])
    kb.setdefault("net", {
        "interfaces": [],
        "ipv4": [],
        "default_gw": [],
        "arp_neighbors": [],
        "routes": [],
    })

    for ev in events:
        et = ev.get("type")

        if et == "NET_INFO":
            # ipv4 / default_gw (listas planas)
            for ip in ev.get("ipv4", []):
                if ip and ip not in kb["net"]["ipv4"]:
                    kb["net"]["ipv4"].append(ip)

            for gw in ev.get("default_gw", []):
                if gw and gw not in kb["net"]["default_gw"]:
                    kb["net"]["default_gw"].append(gw)

            # interfaces (lista de dicts)
            existing_if = {(i.get("name"), i.get("ipv4")) for i in kb["net"]["interfaces"] if isinstance(i, dict)}
            for iface in ev.get("interfaces", []):
                if not isinstance(iface, dict):
                    continue
                key = (iface.get("name"), iface.get("ipv4"))
                if key not in existing_if:
                    kb["net"]["interfaces"].append(iface)
                    existing_if.add(key)

        elif et == "ARP_TABLE":
            # OJO: tu parser devuelve "arp_neighbors", no "neighbors"
            existing_arp = {n.get("ip") for n in kb["net"]["arp_neighbors"] if isinstance(n, dict)}
            for n in ev.get("arp_neighbors", []):
                if not isinstance(n, dict):
                    continue
                ip = n.get("ip")
                if ip and ip not in existing_arp:
                    kb["net"]["arp_neighbors"].append(n)
                    existing_arp.add(ip)

            # (Opcional) también refleja vecinos ARP como "hosts"
            existing_hosts = {h.get("ip") for h in kb["hosts"] if isinstance(h, dict)}
            for n in ev.get("arp_neighbors", []):
                if not isinstance(n, dict):
                    continue
                ip = n.get("ip")
                if ip and ip not in existing_hosts:
                    kb["hosts"].append({"ip": ip, "source": "arp"})
                    existing_hosts.add(ip)

        elif et == "COMMAND_ERROR":
            kb["notes"].append(ev)

        elif et == "NO_EVENT":
            # Puedes ignorarlo o guardarlo; MVP: ignorar
            pass

        else:
            kb["notes"].append(ev)

    return kb

def save_kb(session_dir, kb: dict) -> None:
    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

from typing import Any, Dict, Set, Tuple


def compute_kb_progress_simple(prev_kb: Dict[str, Any], new_kb: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple progress detector. Assumes a simple KB shape:

      kb["hosts"] -> iterable of host strings (or dicts with "ip")
      kb["open_ports"] -> iterable of {"host": "...", "port": 80}
      kb["services"] -> iterable of {"host": "...", "port": 80, "name": "http"}
      kb["findings"] -> iterable of strings

    Returns counts + has_progress.
    """

    def _host(x: Any) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            return str(x.get("ip") or x.get("host") or "").strip()
        return str(x).strip()

    prev_hosts: Set[str] = set(_host(h) for h in prev_kb.get("hosts", []) if _host(h))
    new_hosts: Set[str] = set(_host(h) for h in new_kb.get("hosts", []) if _host(h))

    prev_ports: Set[Tuple[str, int]] = set(
        (_host(p.get("host") or p.get("ip")), int(p.get("port")))
        for p in prev_kb.get("open_ports", [])
        if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
    )
    new_ports: Set[Tuple[str, int]] = set(
        (_host(p.get("host") or p.get("ip")), int(p.get("port")))
        for p in new_kb.get("open_ports", [])
        if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
    )

    prev_services: Set[Tuple[str, int, str]] = set(
        (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
        for s in prev_kb.get("services", [])
        if isinstance(s, dict)
        and _host(s.get("host") or s.get("ip"))
        and str(s.get("port", "")).isdigit()
        and str(s.get("name", "")).strip()
    )
    new_services: Set[Tuple[str, int, str]] = set(
        (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
        for s in new_kb.get("services", [])
        if isinstance(s, dict)
        and _host(s.get("host") or s.get("ip"))
        and str(s.get("port", "")).isdigit()
        and str(s.get("name", "")).strip()
    )

    prev_findings: Set[str] = set(str(f).strip() for f in prev_kb.get("findings", []) if str(f).strip())
    new_findings: Set[str] = set(str(f).strip() for f in new_kb.get("findings", []) if str(f).strip())

    added_hosts = new_hosts - prev_hosts
    added_ports = new_ports - prev_ports
    added_services = new_services - prev_services
    added_findings = new_findings - prev_findings

    return {
        "has_progress": bool(added_hosts or added_ports or added_services or added_findings),
        "new_hosts_count": len(added_hosts),
        "new_ports_count": len(added_ports),
        "new_services_count": len(added_services),
        "new_findings_count": len(added_findings),
    }

def log_step(session_dir, session_id: str, record: dict) -> None:
    path = session_dir / "steps.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)

    # Escribir cabecera META una sola vez (si el fichero no existe o está vacío)
    meta = (not path.exists()) or (path.stat().st_size == 0)

    with path.open("a", encoding="utf-8") as f:
        if meta:
            meta = {
                "type": "META",
                "session_id": session_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def log_command_output(session_dir: Path, session_id: str, action_id: int, action_name: str, result: dict) -> None:
    path = session_dir / "command_outputs.jsonl"
    session_dir.mkdir(parents=True, exist_ok=True)

    need_meta = (not path.exists()) or (path.stat().st_size == 0)
    with path.open("a", encoding="utf-8") as f:
        if need_meta:
            meta = {"type": "META", "session_id": session_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        rec = {
            "type": "CMD",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action_id": action_id,
            "action_name": action_name,
            "cmd": result.get("cmd"),
            "rc": result.get("rc"),
            # para MVP: guarda todo; si te preocupa tamaño, usa [:5000]
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def rules_policy_decide_action(state: dict) -> int:
    # Reglas mínimas y deterministas para tu MVP (Windows-centric)
    # 1) Si aún no tenemos IPv4, primero ipconfig /all
    if int(state.get("net_ipv4_count", 0) or 0) == 0:
        return 1  # INSPECT_IPCONFIG

    # 2) Si aún no tenemos vecinos ARP, pedir arp -a
    if int(state.get("net_arp_count", 0) or 0) == 0:
        return 2  # INSPECT_ARP

    # 3) Si ya hay hosts, prueba ping al foco/primer host
    if int(state.get("hosts_count", 0) or 0) > 0:
        return 4  # PING_FOCUS_HOST

    # 4) Fallback: nada que hacer
    return 0  # NONE

def model_policy_decide_action(state: dict, model, feature_names: list[str]) -> int:
    """
    model: cualquier sklearn classifier ya entrenado (joblib.load(...))
    feature_names: lista de features en el orden usado al entrenar (metrics.json["feature_names"])
    """
    x = np.zeros((1, len(feature_names)), dtype=np.float32)

    for j, k in enumerate(feature_names):
        v = state.get(k, 0)
        if isinstance(v, bool):
            v = 1 if v else 0
        if v is None:
            v = 0
        x[0, j] = float(v)

    y_pred = model.predict(x)
    return int(y_pred[0])

def log_dataset_row(session_dir, session_id: str, row: dict) -> None:
    path = DATASETS_DIR / f"dataset_{session_id}" / "dataset.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def log_freeform_row(session_dir: Path, session_id: str, row: dict) -> None:
    path = session_dir / "dataset_freeform.jsonl"
    need_meta = (not path.exists()) or (path.stat().st_size == 0)
    with path.open("a", encoding="utf-8") as f:
        if need_meta:
            f.write(json.dumps({"type":"META","schema_id":"penhackit.freeform.v1","session_id":session_id}, ensure_ascii=False) + "\n")
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def launch_kb_monitor_window_windows(session_dir: Path, cols: int = 60, rows: int = 14) -> None:
    """
    Opens a separate small PowerShell window that continuously shows kb.json.
    Windows-only. No interaction with the core; read-only.
    """
    if os.name != "nt":
        return
    session_dir = session_dir.resolve()
    kb_path = (session_dir / "kb.json").resolve()
    ps1 = (session_dir / "_kb_monitor.ps1").resolve()

    # # PowerShell script to set window size and refresh output
    ps_script = rf"""
    $kb = '{str(kb_path)}'
    $ErrorActionPreference='Continue'
    try {{
      $raw = $Host.UI.RawUI
      $raw.WindowTitle = 'PenHackIt KB'
      $size = New-Object System.Management.Automation.Host.Size({cols},{rows})
      $raw.WindowSize = $size
      $raw.BufferSize = New-Object System.Management.Automation.Host.Size({cols}, 3000)
    }} catch {{Write-Host "ERROR: $($_.Exception.Message)"}}
    
    while ($true) {{
      Clear-Host
      if (Test-Path -LiteralPath $kb) {{
        Get-Content -LiteralPath $kb -Raw
      }} else {{
        Write-Host "Waiting for kb.json: $kb"
      }}
      Start-Sleep -Milliseconds 500
    }}
    """.strip()
        # IMPORTANT: start "" ...  (empty title), otherwise args get mis-parsed and it exits.
   
    # Launch new console window
    # ps_script = "while ($true) { Clear-Host; 'alive'; Start-Sleep -Milliseconds 500 }"
    
    ps1.write_text(
        f"$kb='{kb_path}'; while($true){{cls; if(Test-Path $kb){{gc $kb -Raw}} else {{'Waiting for kb.json'}}; sleep -m 500}}",
        encoding="utf-8",
    )

    subprocess.Popen(["cmd", "/c", "start", "", "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])
    # subprocess.Popen(
    #     ["cmd.exe", "/c", "start", "", "powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL,
    #     stdin=subprocess.DEVNULL,
    #     creationflags=subprocess.CREATE_NEW_CONSOLE,
    # )