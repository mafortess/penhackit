from pathlib import Path
import json
import platform
import subprocess
import os
import time
import re
import requests

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib
from collections import Counter

import markdown
# from weasyprint import HTML
import matplotlib.pyplot as plt # para gráficos simples en el reporte (p.ej. hosts descubiertos, servicios, etc)

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

import torch # para modelos LLM locales (p.ej. llama.cpp con bindings de Python, o modelos más pequeños)
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer # para cargar modelos LLM locales con HuggingFace (si no usas ollama)  

import threading
import random

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

BANNERS = [
r"""
   ____            __  __            _    _ __
  / __ \___  ____ / /_/ /_  ___ ____| |  (_) /_
 / /_/ / _ \/ __ `/ __/ __ \/ _ `/ __| | / / __/
/ ____/  __/ /_/ / /_/ / / /  __/ /  | |/ / /_
/_/    \___/\__,_/\__/_/ /_/\___/_/   |___/\__/

 PenHackIt — controlled pentesting automation (research prototype)
""" ,

r"""
 ____            _   _            _    ___ _____
|  _ \ ___ _ __ | | | | __ _  ___| | _|_ _|_   _|
| |_) / _ \ '_ \| |_| |/ _` |/ __| |/ /| |  | |
|  __/  __/ | | |  _  | (_| | (__|   < | |  | |
|_|   \___|_| |_|_| |_|\__,_|\___|_|\_\___| |_|

          PenHackIt — Pentest agent (BC + local LLM reporting)
"""
]

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


# Ruta base para sesiones, modelos, reportes.
BASE_DIR = Path("mvp")
SESSIONS_DIR = BASE_DIR / "workspace/data/sessions"
MODELS_DIR = BASE_DIR / "workspace/models"
DATASETS_DIR = BASE_DIR / "workspace/data/datasets"
REPORTS_DIR = BASE_DIR / "workspace/data/reports"
LLM_MODELS_DIR = BASE_DIR / "workspace/llm_models" 

# Menu principal: sesiones, entrenamiento, reporte.
def main_menu():
    print("=== AGENT COMMAND-LINE INTERFACE (CLI) ===")
    print("1) Run session")
    print("2) Train models")
    print("3) Generate report")
    print("0) Exit")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "0"])).strip()
# Menu de sesiones: run session en autonomous mode.
def session_menu():
    print("--- Run session ---")
    print("1) Run session (autonomous)")
    print("2) Run session (observation mode)")
    print("3) Run session (suggestion mode)")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "3", "0"])).strip()
# Menu de entrenamiento: entrenar modelo arbol de decisión con dataset creado por mi.
def training_menu():
    print("--- Train models ---")
    print("1) Train decision tree model")
    print("0) Back")
    return prompt("Select option> ", completer=WordCompleter(["1", "0"])).strip()
# Menu de reporte: generar reporte con métricas de la sesión (en KB creada por mi).
def report_menu():
    print("--- Generate report ---")
    print("1) Generate session report")
    print("2) Export report.pdf from report.md")
    print("0) Back")   
    return prompt("Select option> ", completer=WordCompleter(["1", "2", "0"])).strip()
# Menu de selección de backend LLM para generación de reportes (ollama vs transformers local). 
def choose_llm_backend_menu() -> str | None:
    print("\nReport LLM backend:")
    print("1) Baseline (no LLM)")
    print("2) Ollama (HTTP local)")
    print("3) Transformers (torch local)")
    print("0) Cancel")
    return prompt("Select backend> ", completer=WordCompleter(["1", "2", "3", "0"])).strip()

# Clase sesión:
class Session:
    def __init__(self, session_id: str):
        self.id = session_id  # Se asignará al crear la sesión
                  # self.session_dir = session_dir

def session_wizard():
    print("Session wizard: guide user through session creation.")
    print("1) Choose session type (autonomous, observation, suggestion)")

    mode = input("> ").strip()
    if mode not in ["autonomous", "observation", "suggestion"]:
        print("Invalid mode.")
        return None

    print("2) Enter goal type:")
    goal_type = input("> ").strip()

    print("3) Enter target:")
    target = input("> ").strip()

    session_context = {"mode": mode}
    session_config = {"goal_type": goal_type, "target": target}

    return session_context, session_config

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

BLOCKED = {"powershell", "cmd", "bash", "sh", "zsh"}

def is_interactive_shell(cmd: str) -> bool:
    c = cmd.strip().lower()
    return c in BLOCKED or c.startswith("powershell ") or c.startswith("cmd ") or c.startswith("bash ")

# Clase modelo:
class Model:
    def __init__(self, name: str):
        self.name = name
        self.parameters = {}

def vectorize_dataset(rows: list[dict]):
    all_keys = set()
    for r in rows:
        x = r.get("x") or {}
        all_keys.update(x.keys())
    feature_names = sorted(all_keys)

    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        x = r.get("x") or {}
        for j, k in enumerate(feature_names):
            X[i, j] = float(x.get(k, 0.0))
        y[i] = int(r.get("y"))

    return X, y, feature_names

def train_models_from_dataset(dataset_dir: Path, models_dir: Path) -> Path:
    rows = load_dataset_jsonl(dataset_dir)
    X, y, feature_names = vectorize_dataset(rows)

    model_id = f"models_{dataset_dir.name}"
    out_dir = models_dir / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
    )

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "decision_tree": DecisionTreeClassifier(random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }

    metrics = {
        "dataset_dir": str(dataset_dir),
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": {},
    }

    for name, model in models.items():
        print(f"\n[train] {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred).tolist()
        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        print(f"[eval] {name} accuracy: {acc:.4f}")

        model_path = out_dir / f"{name}.joblib"
        joblib.dump(model, model_path)

        metrics["models"][name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "model_path": str(model_path),
        }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved models to: {out_dir}")
    return out_dir

def list_dataset_candidates(datasets_dir: Path) -> list[Path]:
    """
    Devuelve una lista de 'dataset_dir' candidatos.
    - Añade cada subdirectorio que contenga dataset.jsonl.
    - Añade el propio datasets_dir si contiene dataset.jsonl (caso plano).
    Sort by mtime (últimos primero).
    """
    candidates = []

    # caso plano: datasets/dataset.jsonl
    if (datasets_dir / "dataset.jsonl").exists():
        candidates.append(datasets_dir)

    # caso normal: datasets/<name>/dataset.jsonl
    if datasets_dir.exists():
        for p in datasets_dir.iterdir():
            if p.is_dir() and (p / "dataset.jsonl").exists():
                candidates.append(p)

    # orden por mtime (últimos primero)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates

def choose_dataset_dir(datasets_dir: Path) -> Path | None:
    items = list_dataset_candidates(datasets_dir)
    if not items:
        print(f"No datasets found in: {datasets_dir}")
        print("Expected either:")
        print(f" - {datasets_dir / 'dataset.jsonl'}")
        print(f" - {datasets_dir / '<dataset_id>' / 'dataset.jsonl'}")
        return None

    print("\nAvailable datasets:")
    for i, p in enumerate(items, start=1):
        label = p.name if p != datasets_dir else "(root) datasets/"
        print(f"{i}) {label}")
    raw = prompt("Select dataset> ", completer=WordCompleter([str(i) for i in range(1, len(items) + 1)] + ["0"])).strip()
    if raw == "0":
        return None
    if not raw.isdigit():
        print("Invalid input.")
        return None
    idx = int(raw)
    if idx < 1 or idx > len(items):
        print("Out of range.")
        return None
    return items[idx - 1]


def load_dataset_jsonl(dataset_dir: Path) -> list[dict]:
    # Show dataset options in this directort (dataset_dir)
    jsonl_files = sorted(
        [p.name for p in dataset_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jsonl"],
        key=str.lower,
    )
    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in: {dataset_dir}")
    
    dataset_choice = prompt(f"Load dataset from: {dataset_dir} > ", completer=WordCompleter(jsonl_files))

    
    path = dataset_dir / dataset_choice
    if not path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found: {path}")

    rows = []
    print(f"Loading dataset: {path} ...")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # expected: {"schema_id":..., "t":..., "state":{...}, "action_id":int}
            rows.append(obj)
    if not rows:
        raise RuntimeError("Dataset is empty.")
    return rows

def vectorize_bc_rows(rows: list[dict]):
    # Collect all keys from "state"
    keys = set()
    for r in rows:
        s = r.get("state") or {}
        if not isinstance(s, dict):
            raise TypeError("Each row must contain a dict field 'state'.")
        keys.update(s.keys())

    feature_names = sorted(keys)
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, r in enumerate(rows):
        s = r.get("state") or {}
        for j, k in enumerate(feature_names):
            v = s.get(k, 0)
            if isinstance(v, bool):
                v = 1 if v else 0
            if v is None:
                v = 0
            if not isinstance(v, (int, float)):
                raise TypeError(f"Non-numeric feature in state: key={k} value={v!r}")
            X[i, j] = float(v)

        if "action_id" not in r:
            raise KeyError("Missing 'action_id' in dataset row.")
        y[i] = int(r["action_id"])

    return X, y, feature_names

# =========================
# Model helpers
# =========================

MODEL_CHOICES = {
    "1": ("logreg", "Logistic Regression (multinomial)", lambda: LogisticRegression(max_iter=2000)),
    "2": ("decision_tree", "Decision Tree", lambda: DecisionTreeClassifier(random_state=42)),
    "3": ("random_forest", "Random Forest", lambda: RandomForestClassifier(n_estimators=200, random_state=42)),
    "4": ("mlp", "MLP (2 hidden layers)", lambda: MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)),
}

def choose_model_type() -> tuple[str, str, callable] | None:
    print("\nModel types:")
    for k, (name, desc, _) in MODEL_CHOICES.items():
        print(f"{k}) {name} - {desc}")
    # raw = input("Select model (0 cancel)> ").strip()
    raw = prompt("Select model> ", completer=WordCompleter(list(MODEL_CHOICES.keys()) + ["0"])).strip()
    if raw == "0":
        return None
    if raw not in MODEL_CHOICES:
        print("Invalid option.")
        return None
    return MODEL_CHOICES[raw]

# =========================
# Training (single function)
# =========================

def run_training_interactive(datasets_dir: Path, models_dir: Path) -> None:
    """
    Interactive training:
    - choose model type
    - choose dataset folder (must contain dataset.jsonl)
    - train + evaluate
    - save model + metrics under models_dir/<dataset>/<model>_<n>/
    """

    # 1) elegir modelo
    choice = choose_model_type()
    if not choice:
        return
    model_key, model_desc, model_factory = choice
    print(f"Selected model: {model_key} ({model_desc})")

    # 2) elegir dataset
    dataset_dir = choose_dataset_dir(datasets_dir)
    if not dataset_dir:
        return    
    print(f"\nSelected dataset session: {dataset_dir.name}")
    
    rows = load_dataset_jsonl(dataset_dir)
    X, y, feature_names = vectorize_bc_rows(rows)

    # split
    counts = Counter(y.tolist())
    min_count = min(counts.values())

    strat = y if len(set(y.tolist())) > 1 and min_count >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )

    model = model_factory()

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    print(f"Accuracy: {acc:.4f}")
    print("Confusion matrix:")
    print(cm)

    # output dir
    models_dir.mkdir(parents=True, exist_ok=True)
    out_parent = models_dir / dataset_dir.name
    out_parent.mkdir(parents=True, exist_ok=True)

    out_dir = next_available_path(out_parent, f"{model_key}", "")  # placeholder
    # next_available_path expects ext; we'll just do our own for dirs:
    # create model_key, model_key_1, ...
    if (out_parent / model_key).exists():
        i = 1
        while (out_parent / f"{model_key}_{i}").exists():
            i += 1
        out_dir = out_parent / f"{model_key}_{i}"
    else:
        out_dir = out_parent / model_key
    out_dir.mkdir(parents=True, exist_ok=False)

    # save
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "schema": "penhackit.training.v1",
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_dir": str(dataset_dir),
        "model_type": model_key,
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "accuracy": acc,
        "confusion_matrix": cm,
        "classification_report": rep,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nSaved model: {model_path}")
    print(f"Saved metrics: {out_dir / 'metrics.json'}")
    print(f"Output dir: {out_dir}")

def baseline_section_body(section_title: str, kb_compact: dict) -> str:
    """
    Baseline deterministic generator: no LLM.
    Uses only kb_compact, does not invent facts.
    """
    counts = kb_compact.get("counts", {}) or {}
    net = kb_compact.get("net", {}) or {}
    focus = kb_compact.get("focus", {}) or {}
    sid = kb_compact.get("session_id") or ""
    goal_type = kb_compact.get("goal_type") or "unknown"
    target = kb_compact.get("target") or "unknown"

    hosts_n = int(counts.get("hosts", 0) or 0)
    services_n = int(counts.get("services", 0) or 0)
    findings_n = int(counts.get("findings", 0) or 0)
    notes_n = int(counts.get("notes", 0) or 0)
    commands_n = int(counts.get("commands", 0) or 0)

    ipv4 = net.get("ipv4", []) or []
    gws = net.get("default_gw", []) or []
    arps = net.get("arp_neighbors", []) or []

    commands_tail = kb_compact.get("commands_tail", []) or []
    findings_sample = kb_compact.get("findings_sample", []) or []
    hosts_sample = kb_compact.get("hosts_sample", []) or []
    services_sample = kb_compact.get("services_sample", []) or []
    notes_tail = kb_compact.get("notes_tail", []) or []

    def bullet_list(items: list[str], empty_msg: str = "No data captured in KB.") -> str:
        if not items:
            return empty_msg
        return "\n".join([f"- {x}" for x in items])

    def fmt_arp(n: dict) -> str:
        ip = (n or {}).get("ip", "") or ""
        mac = (n or {}).get("mac", "") or ""
        kind = (n or {}).get("type", "") or ""
        s = ip
        if mac:
            s += f" ({mac})"
        if kind:
            s += f" [{kind}]"
        return s.strip() or "(unknown)"

    def fmt_host(h: dict) -> str:
        ip = (h or {}).get("ip", "") or ""
        src = (h or {}).get("source", "") or ""
        return f"{ip} [{src}]" if src else ip

    def fmt_service(svc: dict) -> str:
        # si tu KB no tiene estructura de services aún, esto no inventa nada
        if not isinstance(svc, dict):
            return str(svc)
        parts = []
        for k in ("host", "ip", "port", "proto", "name", "product", "version"):
            v = svc.get(k)
            if v is None or v == "":
                continue
            parts.append(f"{k}={v}")
        return ", ".join(parts) if parts else str(svc)

    if section_title == "Executive Summary":
        lines = []
        lines.append(f"Session {sid} executed as a baseline report (no LLM).")
        lines.append(f"Goal: {goal_type}. Target: {target}.")
        lines.append(f"Captured: {commands_n} commands, {hosts_n} hosts, {services_n} services, {findings_n} findings, {notes_n} notes.")
        if findings_n == 0:
            lines.append("Outcome: no findings recorded in KB for this session.")
        else:
            lines.append("Outcome: findings were recorded in KB; see Findings section for details.")
        return " ".join(lines)

    if section_title == "Scope and Context":
        lines = []
        lines.append(f"Scope is limited to the data captured in the session KB and command outputs.")
        lines.append(f"Goal type: {goal_type}. Target: {target}.")
        fl = (focus or {}).get("level", "global")
        fh = (focus or {}).get("host", "") or ""
        fs = (focus or {}).get("service", "") or ""
        lines.append(f"Focus: level={fl}, host={fh or 'none'}, service={fs or 'none'}.")
        lines.append("Environment details (OS, tooling versions, constraints) are not fully captured unless stored in KB.")
        return "\n".join(lines)

    if section_title == "Environment Observations":
        lines = []
        lines.append("Network observations captured from KB:")
        lines.append("")
        lines.append(f"- Local IPv4(s): {', '.join(ipv4) if ipv4 else 'Not captured'}")
        lines.append(f"- Default gateway(s): {', '.join(gws) if gws else 'Not captured'}")
        if arps:
            lines.append(f"- ARP neighbors ({min(len(arps), 30)} shown):")
            for n in arps[:30]:
                lines.append(f"  - {fmt_arp(n)}")
        else:
            lines.append("- ARP neighbors: Not captured")
        return "\n".join(lines)

    if section_title == "Actions Performed":
        # lista solo lo que existe en KB, sin interpretación
        if not commands_tail:
            return "No commands recorded in KB."
        items = [c for c in commands_tail if isinstance(c, str) and c.strip()]
        return bullet_list(items, empty_msg="No commands recorded in KB.")

    if section_title == "Findings":
        if findings_n == 0:
            return "No findings in this session (KB.findings is empty)."
        # si findings son dicts, los serializamos de forma segura
        out = []
        for f in findings_sample[:30]:
            if isinstance(f, dict):
                out.append(json.dumps(f, ensure_ascii=False))
            else:
                out.append(str(f))
        return bullet_list(out, empty_msg="Findings present but not readable.")

    if section_title == "Next Steps":
        steps = []
        # heurísticas simples basadas en "qué falta"
        if hosts_n == 0:
            steps.append("Capture discovery output to populate KB.hosts (e.g., ARP/scan results).")
        elif services_n == 0:
            steps.append("Capture service enumeration to populate KB.services (ports/protocols/banners).")
        if commands_n == 0:
            steps.append("Ensure executed commands are appended to KB.commands for traceability.")
        if findings_n == 0:
            steps.append("If the goal is vulnerability assessment, add steps that produce findings and store them in KB.findings.")
        if not ipv4 and not gws:
            steps.append("Capture basic network context (IPv4/default gateway/interfaces) to support environment section.")
        if not steps:
            steps.append("Refine KB schema for the target scenario and increase coverage of actions/parsers.")
            steps.append("Run additional sessions to build a larger dataset for BC training and evaluation.")
        return bullet_list(steps)

    # fallback (shouldn't happen)
    return "No baseline implementation for this section."

REPORT_SECTIONS = [
    ("Executive Summary", "Resumen ejecutivo: 5-8 líneas, objetivo y resultado general."),
    ("Scope and Context", "Alcance: objetivo, target(s), entorno (Kali VM + contenedores), restricciones."),
    ("Environment Observations", "Observaciones del entorno: red local, interfaces, gateways, vecinos ARP relevantes."),
    ("Actions Performed", "Acciones ejecutadas: lista concisa de comandos y propósito."),
    ("Findings", "Hallazgos: si no hay, indicar 'No findings in this session' y por qué."),
    ("Next Steps", "Siguientes pasos concretos: 5-10 bullets priorizados."),
]

def compact_kb_for_report(kb: dict) -> dict:
    net = kb.get("net", {}) or {}
    return {
        "session_id": kb.get("session_id"),
        "goal_type": (kb.get("session_context", {}) or {}).get("goal_type"),  # si existe
        "target": (kb.get("session_context", {}) or {}).get("target"),        # si existe
        "focus": kb.get("focus", {}),
        "counts": {
            "hosts": len(kb.get("hosts", []) or []),
            "services": len(kb.get("services", []) or []),
            "findings": len(kb.get("findings", []) or []),
            "notes": len(kb.get("notes", []) or []),
            "commands": len(kb.get("commands", []) or []),
        },
        "net": {
            "ipv4": (net.get("ipv4", []) or [])[:10],
            "default_gw": (net.get("default_gw", []) or [])[:10],
            "arp_neighbors": (net.get("arp_neighbors", []) or [])[:30],
        },
        "hosts_sample": (kb.get("hosts", []) or [])[:30],
        "services_sample": (kb.get("services", []) or [])[:30],
        "findings_sample": (kb.get("findings", []) or [])[:30],
        "commands_tail": (kb.get("commands", []) or [])[-40:],
        "notes_tail": (kb.get("notes", []) or [])[-20:],
    }

def build_section_prompt(section_title: str, section_guidance: str, kb_compact: dict) -> str:
    return f"""You are writing ONLY the BODY for a pentest report section.

Section title: {section_title}
Guidance: {section_guidance}

Rules:
- Do NOT include any headings or the section title.
- Be concise and technical.
- Do not invent facts. Use only the KB.
- If there is insufficient data, explicitly say so and suggest what data is missing.

KB (JSON):
{json.dumps(kb_compact, ensure_ascii=False, indent=2)}
"""

# Clase generate_report:
class Report:
    def __init__(self, session: Session):
        self.session = session
        self.content = ""


def ollama_list_models_http(timeout_s: int = 10) -> list[str]:
    """
    Returns installed model names from Ollama.
    Uses /api/tags so it reflects what's available locally.
    """
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=timeout_s)
        r.raise_for_status()
        data = r.json() or {}
        models = data.get("models", []) or []
        names = []
        for m in models:
            name = (m or {}).get("name")
            if name:
                names.append(name)
        # stable-ish order
        names.sort()
        return names
    except Exception:
        return []

def ollama_list_models_cli(timeout_s: int = 5) -> list[str]:
    """
    Fallback: calls `ollama list` and parses the first column (NAME).
    """
    try:
        o = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if o.returncode != 0:
            return []
        lines = (o.stdout or "").splitlines()
        if not lines:
            return []
        # header: NAME  ID  SIZE  MODIFIED
        out = []
        for ln in lines[1:]:
            ln = ln.strip()
            if not ln:
                continue
            # NAME is the first token
            out.append(ln.split()[0])
        out = sorted(set(out))
        return out
    except Exception:
        return []
    
def choose_ollama_model_interactive(default: str | None = None) -> str | None:
    """
    Interactive chooser:
    - fetches local models via HTTP (preferred), CLI fallback
    - provides:
      * numbered list selection
      * optional text filter
      * best-effort TAB autocomplete (if readline is available)
    Returns selected model name or None if cancelled.
    """
    models = ollama_list_models_http()
    if not models:
        models = ollama_list_models_cli()

    if not models:
        print("No Ollama models found (is Ollama running, and are models installed?).")
        print("Try: ollama list  |  ollama pull <model>")
        return None

    # best-effort TAB completion (works on Linux/macOS; on Windows may require pyreadline3)
    try:
        import readline  # type: ignore

        def _completer(text, state):
            matches = [m for m in models if m.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    print("\nAvailable Ollama models:")
    for i, m in enumerate(models, start=1):
        tag = " (default)" if default and m == default else ""
        print(f"{i}) {m}{tag}")

    while True:
        raw = input("Select model [number | name | /filter | 0 cancel]> ").strip()
        if raw == "0":
            return None

        # filter mode: /text
        if raw.startswith("/"):
            q = raw[1:].strip().lower()
            if not q:
                continue
            filtered = [m for m in models if q in m.lower()]
            if not filtered:
                print("No matches.")
                continue
            print("\nMatches:")
            for i, m in enumerate(filtered, start=1):
                print(f"{i}) {m}")
            raw2 = input("Select from matches (number | name | 0 back)> ").strip()
            if raw2 == "0":
                continue
            if raw2.isdigit():
                idx = int(raw2)
                if 1 <= idx <= len(filtered):
                    return filtered[idx - 1]
                print("Out of range.")
                continue
            if raw2 in filtered:
                return raw2
            print("Invalid selection.")
            continue

        # numeric selection
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print("Out of range.")
            continue

        # exact name selection (autocomplete helps here)
        if raw in models:
            return raw

        # small convenience: if user typed prefix and it's unique
        pref = [m for m in models if m.startswith(raw)]
        if len(pref) == 1:
            return pref[0]

        print("Invalid selection. Tip: type / to filter, or TAB to autocomplete (if supported).")

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

def ollama_generate_http(model: str, prompt: str, timeout_s: int = 180) -> str:
    r = requests.post(
        OLLAMA_GENERATE_URL,
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,
        timeout=timeout_s,
    )
    r.raise_for_status()

    chunks = []
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        data = json.loads(line)
        token = data.get("response", "")
        if token:
            print(token, end="", flush=True)  # feedback inmediato
            chunks.append(token)
        if data.get("done"):
            break

    print()  # newline final
    return "".join(chunks).strip()

def sanitize_llm_section(text: str, section_title: str) -> str:
    t = (text or "").strip()

    # 1) remove triple-backtick fences (keep inner text)
    # removes ```lang\n ... \n``` blocks
    t = re.sub(r"```[a-zA-Z0-9_-]*\n", "", t)
    t = t.replace("```", "")

    # 2) remove accidental repeated headings at start (e.g. "## Executive Summary")
    lines = [ln.rstrip() for ln in t.splitlines()]
    while lines and re.match(r"^#{1,6}\s+", lines[0]):
        # if it's the same section title or any heading, drop it
        h = re.sub(r"^#{1,6}\s+", "", lines[0]).strip().lower()
        if h == section_title.strip().lower():
            lines.pop(0)
            # drop following blank lines
            while lines and lines[0].strip() == "":
                lines.pop(0)
        else:
            break

    t = "\n".join(lines).strip()
    return t

def plot_hosts_kpi(kb: dict, out_png: Path) -> None:
    n = len(kb.get("hosts", []) or [])
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.axis("off")
    plt.text(0.5, 0.6, "Hosts discovered", ha="center", va="center", fontsize=18)
    plt.text(0.5, 0.35, str(n), ha="center", va="center", fontsize=48)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

def plot_counts(kb: dict, out_png: Path) -> None:
    hosts = len(kb.get("hosts", []) or [])
    services = len(kb.get("services", []) or [])
    findings = len(kb.get("findings", []) or [])
    commands = len(kb.get("commands", []) or [])

    labels = ["hosts", "services", "findings", "commands"]
    values = [hosts, services, findings, commands]

    # Asegura que el directorio existe para guardar la figura
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("count")
    plt.title("Session summary counts")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def generate_report_md_sectionwise(session_dir: Path, kb: dict, model: str) -> Path:
    kb_compact = compact_kb_for_report(kb)

    # report_path = session_dir / "report.md" # estándar, pero si quieres evitar sobreescribir: next_available_path(session_dir, "report", ".md")
    report_path = next_available_path(session_dir, "report", ".md") # evita sobreescribir

    # 1) cabecera fija PRIMERO (esto crea/sobrescribe el archivo)
    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    header.append(f"- Model: {model}")
    header.append("")

    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    # 2) generar figuras y AÑADIRLAS (append) DESPUÉS
    fig_dir = session_dir / "figures"

    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with (report_path).open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    # 3) secciones fijas (append)
    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")

        prompt = build_section_prompt(title, guidance, kb_compact)
        body = ollama_generate_http(model, prompt, timeout_s=180).strip()
        body = sanitize_llm_section(body, title)

        with report_path.open("a", encoding="utf-8") as f:
            f.write(body + "\n\n")

    return report_path

def next_available_path(dirpath: Path, base_name: str, ext: str) -> Path:
    """
    Returns a non-existing path in dirpath.
    Example: base_name="report", ext=".md" -> report.md, report_1.md, report_2.md, ...
    """
    ext = ext if ext.startswith(".") else f".{ext}"
    p0 = dirpath / f"{base_name}{ext}"
    if not p0.exists():
        return p0

    i = 1
    while True:
        pi = dirpath / f"{base_name}_{i}{ext}"
        if not pi.exists():
            return pi
        i += 1

def md_to_pdf(md_path: Path, pdf_path: Path) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    html_body = markdown.markdown(md_text, extensions=["fenced_code", "tables"])
    html = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{ font-family: Arial, sans-serif; font-size: 12px; }}
          h1, h2 {{ margin-top: 20px; }}
          pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
          code {{ font-family: Consolas, monospace; }}
        </style>
      </head>
      <body>{html_body}</body>
    </html>
    """
    HTML(string=html).write_pdf(str(pdf_path))

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


def md_to_pdf_simple(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8").splitlines()

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    x = 2 * cm
    y = height - 2 * cm
    line_h = 14  # puntos

    def new_page():
        nonlocal y
        c.showPage()
        y = height - 2 * cm

    for line in text:
        # estilo básico por heurística
        if line.startswith("# "):
            c.setFont("Helvetica-Bold", 16)
            line = line[2:].strip()
            y -= 6
        elif line.startswith("## "):
            c.setFont("Helvetica-Bold", 13)
            line = line[3:].strip()
            y -= 4
        elif line.startswith("- "):
            c.setFont("Helvetica", 10)
            line = "• " + line[2:].strip()
        else:
            c.setFont("Helvetica", 10)

        # salto de página si hace falta
        if y < 2 * cm:
            new_page()

        # dibuja línea
        c.drawString(x, y, line[:120])  # recorte simple MVP
        y -= line_h

    c.save()

# 4) List local HF models (directories under mvp/llm_models/)
def list_hf_model_candidates(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []
    items = []
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        # heuristics: a HF model folder usually has config.json
        if (p / "config.json").exists():
            items.append(p)
    items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return items

def choose_hf_model_dir_interactive(models_dir: Path, default_name: str | None = None) -> Path | None:
    items = list_hf_model_candidates(models_dir)
    if not items:
        print(f"No HF models found in: {models_dir}")
        print("Expected folders like: mvp/llm_models/<model_id>/config.json")
        return None

    print("\nAvailable HF models (local):")
    for i, p in enumerate(items, start=1):
        tag = " (default)" if default_name and p.name == default_name else ""
        print(f"{i}) {p.name}{tag}")

    raw = prompt("Select model [num | name | 0 cancel]> ").strip()
    if raw == "0":
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(items):
            return items[idx - 1]
        print("Out of range.")
        return None

    # name selection
    for p in items:
        if p.name == raw:
            return p

    # prefix unique
    pref = [p for p in items if p.name.startswith(raw)]
    if len(pref) == 1:
        return pref[0]

    print("Invalid selection.")
    return None

def choose_hf_device_interactive() -> str:
    # keep it simple for MVP
    # - "cuda" if available
    # - else "cpu"
    default = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice (default: {default})")
    print("1) auto")
    print("2) cuda")
    print("3) cpu")
    raw = prompt("Select device> ", completer=WordCompleter(["1", "2", "3"])).strip()
    if raw == "2":
        return "cuda"
    if raw == "3":
        return "cpu"
    return default

# 5) Transformers generator with caching (load once, reuse per section)
_HF_CACHE = {
    "model_dir": None,
    "device": None,
    "tokenizer": None,
    "model": None,
}

def hf_load_model(model_dir: Path, device: str):
    global _HF_CACHE
    if _HF_CACHE["model_dir"] == str(model_dir) and _HF_CACHE["device"] == device and _HF_CACHE["model"] is not None:
        return _HF_CACHE["tokenizer"], _HF_CACHE["model"]

    # Clean previous (optional)
    _HF_CACHE = {"model_dir": str(model_dir), "device": device, "tokenizer": None, "model": None}

    print(f"\n[HF] Loading model from: {model_dir}")
    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    # If tokenizer has no pad_token (common), set it to eos to avoid warnings in generation
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # For small models: float16 on cuda; float32 on cpu
    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    model.eval()

    _HF_CACHE["tokenizer"] = tok
    _HF_CACHE["model"] = model
    return tok, model

def hf_generate_stream(model_dir: Path, prompt_text: str, max_new_tokens: int = 350) -> str:
    print(f"[HF] Loading model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto",
    )
    model.eval()

    inputs = tokenizer(prompt_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    chunks = []
    for tok in streamer:
        print(tok, end="", flush=True)
        chunks.append(tok)

    print()
    t.join()
    return "".join(chunks).strip()

def hf_generate_text(
    model_dir: Path,
    prompt_text: str,
    device: str,
    max_new_tokens: int = 350,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    tok, model = hf_load_model(model_dir, device)

    inputs = tok(prompt_text, return_tensors="pt")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    else:
        inputs = {k: v.to("cpu") for k, v in inputs.items()}

    # Stream output tokens to console (similar feel to your Ollama streaming)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=True if temperature > 0 else False,
        temperature=float(temperature),
        top_p=float(top_p),
        streamer=streamer,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # Run generation in a background thread so streamer can iterate

    t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
    t.start()

    chunks = []
    for new_text in streamer:
        print(new_text, end="", flush=True)
        chunks.append(new_text)
    print()
    t.join()

    return "".join(chunks).strip()

# 6) Make report generation accept either backend
def generate_report_md_sectionwise_llm(
    session_dir: Path,
    kb: dict,
    backend: str,
    ollama_model: str | None = None,
    hf_model_dir: Path | None = None,
    hf_device: str = "cpu",
) -> Path:
    kb_compact = compact_kb_for_report(kb)
    report_path = next_available_path(session_dir, "report", ".md")

    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")

    if backend == "ollama":
        header.append(f"- Backend: ollama")
        header.append(f"- Model: {ollama_model or ''}")
    else:
        header.append(f"- Backend: transformers")
        header.append(f"- Model dir: {str(hf_model_dir) if hf_model_dir else ''}")
        header.append(f"- Device: {hf_device}")

    header.append("")
    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    fig_dir = session_dir / "figures"
    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")

        section_prompt = build_section_prompt(title, guidance, kb_compact)

        if backend == "ollama":
            if not ollama_model:
                raise ValueError("ollama_model is required for backend='ollama'")
            body = ollama_generate_http(ollama_model, section_prompt, timeout_s=180).strip()
        else:
            if not hf_model_dir:
                raise ValueError("hf_model_dir is required for backend='transformers'")
            body = hf_generate_text(
                hf_model_dir,
                section_prompt,
                device=hf_device,
                max_new_tokens=350,
                temperature=0.2,
                top_p=0.95,
            ).strip()

        body = sanitize_llm_section(body, title)

        with report_path.open("a", encoding="utf-8") as f:
            f.write(body + "\n\n")

    return report_path

def generate_report_md_baseline(session_dir: Path, kb: dict) -> Path:
    kb_compact = compact_kb_for_report(kb)
    report_path = next_available_path(session_dir, "report", ".md")

    header = []
    header.append("# PenHackIt Report")
    header.append("")
    header.append(f"- Session ID: {kb.get('session_id','')}")
    header.append(f"- Generated at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    header.append(f"- Backend: baseline (no LLM)")
    header.append("")

    report_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    fig_dir = session_dir / "figures"
    plot_counts(kb, fig_dir / "counts.png")
    plot_hosts_kpi(kb, fig_dir / "hosts.png")

    with report_path.open("a", encoding="utf-8") as f:
        f.write("## Figures\n\n")
        f.write("![](figures/counts.png)\n\n")
        f.write("![](figures/hosts.png)\n\n")

    for title, guidance in REPORT_SECTIONS:
        with report_path.open("a", encoding="utf-8") as f:
            f.write(f"## {title}\n\n")
        body = baseline_section_body(title, kb_compact).strip()
        with report_path.open("a", encoding="utf-8") as f:
            f.write(body + "\n\n")

    return report_path

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

# def log_bc_row(session_dir: Path, session_id: str, row: dict) -> None:
#     path = session_dir / "dataset.jsonl"
#     session_dir.mkdir(parents=True, exist_ok=True)

#     need_meta = (not path.exists()) or (path.stat().st_size == 0)
#     with path.open("a", encoding="utf-8") as f:
#         if need_meta:
#             meta = {
#                 "type": "META",
#                 "schema_id": "penhackit.bc.v1",
#                 "session_id": session_id,
#                 "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#             }
#             f.write(json.dumps(meta, ensure_ascii=False) + "\n")

#         f.write(json.dumps(row, ensure_ascii=False) + "\n")

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

def main():
    # Lógica navegación entre menús
    while True:
        print(random.choice(BANNERS))
        print()

        print("Operative System:", platform.system())  # solo para mostrar que se puede usar info del sistema en el menú si se quiere
        print("Architecture:", platform.machine() )
        print()

        choice = main_menu()
 
        if choice == "1":
            # Lógica de menu session
            while True:
                sub_choice = session_menu()
                # Modo autónomo y modo sugerencia: el sistema decide qué acción ejecutar en cada paso para avanzar hacia el objetivo
                # Puede ser con una función de decisión basada en reglas (rules_policy_decide_action) 
                # o un modelo de ML (model_policy_decide_action).
                # Para la demo se usó una secuencia fija de acciones, pero la idea es que el sistema tome decisiones autónomas basadas en el estado actual (KB) y el objetivo de la sesión.
                # Modo sugerencia es igual, excepto que el sistema sugiere la acción y el usuario confirma antes de ejecutarla.
                if sub_choice == "1" or sub_choice == "3":
                    # Aquí irá un wizard para crear la sesión (tipo, objetivo, etc.)
                    # y luego se creará la sesión con esos parámetros.
                    # session_wizard()
                    if sub_choice == "1":
                        mode = "autonomous" # suggestion, observation
                    elif sub_choice == "3":
                        mode = "suggestion"
                    # mode = "suggestion"
                    goal_type = "recon"
                    target = "127.0.0.1"
                    name = "mvp"
                    max_steps = 5

                    #Se crea sesión con parámetros por defecto (autonomous, goal_type y target hardcodeados)
                    # para poder avanzar con el desarrollo del MVP.
                    session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + name
                    session_dir = SESSIONS_DIR / f"session_{session_id}"
                    print(f"Creating session directory: {session_dir}")
                    session_dir.mkdir(parents=True, exist_ok=True)

                    session_config = {
                        "id": session_id,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    session_context = {
                        "id": session_id,
                        "mode": mode,
                        "goal_type": goal_type,
                        "target": target,
                        "max_steps": max_steps,
                    }

                    s = Session(session_id)

                    print("Running session...")
                    
                    # Creación de KB inicial vacía en la carpeta de la sesión.
                    session_dir.mkdir(parents=True, exist_ok=True)
                    # Esta linea anterior crea la carpeta session_dir en el sistema de archivos.
                    # # parents=True: si faltan carpetas “padre” en la ruta, también las crea. Ejemplo: si data/ o data/sessions/ no existen, los crea automáticamente.
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
                    kb = {
                        "session_id": session_id,

                        # Memoria “pentest”
                        "name_enterprise": "ITIS",
                        "networks": {},
                        "hosts": [],
                        "services": [],
                        "findings": [],
                        "notes": [],

                        # Memoria "entorno local"
                        "net": {
                            "interfaces": [],      # lista de interfaces (name, ipv4, mask, gw, mac opcional)
                            "ipv4": [],            # lista plana (comodín)
                            "default_gw": [],      # lista de gateways
                            "arp_neighbors": [],   # lista de vecinos ARP (ip, mac opcional)
                            "routes": [],          # opcional: ruta (dest, mask/prefix, gateway, if, metric)
                        },
                        "focus": {"level": "global", "host": "", "service": ""},
                        "commands": [],

                        # runtime / trazabilidad mínima (para el estado)
                        "step_idx": 0,
                        "last_action_id": None,
                        "last_action_name": None,
                        "last_rc": None,
                        "last_event_type": None,
                    }

                    (session_dir / "kb.json").write_text(
                        json.dumps(kb, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    # La función de arriba crea/sobrescribe kb.json en la carpeta de la sesión y guarda dentro la KB inicial en formato JSON legible.

                    launch_kb_monitor_window_windows(session_dir)
                    time.sleep(2)
                    
                    # Para el MVP, la decisión de acción se hará con una función policy_decide_action(state, t) que implementaremos con lógica fija (scripted) o reglas simples. T
                    # También podría usar un modelo de ML entrenado con los datos de sesiones anteriores.
                    autonomous_decider = "scripted"  # "scripted" | "rules" | "model"
                    model = None
                    feature_names = None
                    # model = joblib.load("mvp/models/<...>/model.joblib")
                    # feature_names = json.loads(Path("mvp/models/<...>/metrics.json").read_text())["feature_names"]
                    
                    if autonomous_decider == "model":
                        print("Loading ML model for autonomous decision...")
                        model_path = MODELS_DIR / "datasets" /"decision_tree" / "model.joblib"
                        metrics_path = model_path.parent / "metrics.json"

                        if not model_path.exists() or not metrics_path.exists():
                            print(f"Model files not found: {model_path}, {metrics_path}")
                            print("Make sure to train a model first and place the files in mvp/models/")
                            return

                        model = joblib.load(model_path)
                        feature_names = json.loads(metrics_path.read_text())["feature_names"]
                        print(f"Loaded model: {model_path}, features: {feature_names}")

                    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
                        print(f"\n--- Step {t} ---")

                        # ESTADO
                        state = build_state(kb, session_context)
                        print("Current state:", state)

                        # DECISIÓN DE ACCIÓN
                        if mode == "observation":
                            # Pentestir elige acción (ID) o mete comando directo
                            raw = input("OBS> action_id (num) OR type a command (0 stop)> ").strip()

                            if raw == "0":
                                action_id = 0
                                action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
                                command = None

                            elif raw.isdigit():
                                action_id = int(raw)
                                action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
                                command = command_builder(action_id, kb)

                            else:
                                # Comando directo (sin acción)
                                action_id = -1
                                action_name = "USER_COMMAND"
                                cmd_template = None
                                command = raw
                        else:
                            if autonomous_decider == "rules":
                                print("Deciding action with rules policy...")
                                action_id = rules_policy_decide_action(state)
                            elif autonomous_decider == "model":
                                print("Deciding action with ML model...")
                                action_id = model_policy_decide_action(state, model, feature_names)
                            else:
                                print("Deciding action with scripted policy, only for MVP demo...")
                                action_id = policy_decide_action(state, t)

                            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))
                        
                        import copy
                        # Esto no hace una copia, solo referencia.
                        # prev_kb = kb 
                        prev_kb = copy.deepcopy(kb) # para comparar antes/después y calcular progreso

                        print(f"Decided action: {action_name} (ID: {action_id})")

                        # COMANDO
                        command = command_builder(action_id, kb)
                        print(f"Built command: {command}")

                        # CONFIRMACIÓN (solo en modo sugerencia)
                        if mode == "suggestion":
                            print(f"SUGGESTED: {action_name} (ID: {action_id})")
                            user_cmd = input("Enter=run suggested | type cmd=override | 0=stop > ").strip()

                            if user_cmd == "0":
                                command_to_run = None
                                accepted = False
                            elif user_cmd == "":
                                command_to_run = command
                                accepted = True
                            else:
                                command_to_run = user_cmd
                                accepted = False

                            # registra sugerencia
                            kb["last_suggested_action_id"] = action_id
                            kb["last_suggested_action_name"] = action_name
                            kb["last_suggested_command"] = command
                            kb["last_accepted_suggestion"] = accepted

                        else:
                            command_to_run = command

                        # EJECUCIÓN
                        result = execute_command(command_to_run)

                        # Para coherencia en los logs
                        command = command_to_run

                        log_command_output(session_dir, session_id, action_id, action_name, result)

                        # PARSEAR RESULTADO Y ACTUALIZAR KB
                        events = parse_command_result(action_name, result)
                        print(f"Events generated from command result: {events}")

                        # MEMORY (KB) UPDATE
                        new_kb = update_kb(kb, events)

                        kb = new_kb

                        kb.setdefault("commands", [])
                        if result.get("cmd"):
                            kb["commands"].append(result["cmd"])

                        kb["step_idx"] = t
                        kb["last_action_id"] = action_id
                        kb["last_action_name"] = action_name
                        kb["last_rc"] = result.get("rc")
                        kb["last_event_type"] = events[0].get("type") if events else None

                        print(f"Updated KB: {kb}")
                        save_kb(session_dir, kb)

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

                        # Logging del paso completo (estado, acción, comando, resultado) para trazabilidad y posible entrenamiento futuro
                        log_step(session_dir, session_id,{
                        "t": t,
                        "state": state,
                        "action_id": action_id,
                        "command": command,
                        })

                        time.sleep(1)

                    print("Session finished\n")
                    break

                # Modo observación: el sistema solo observa la sesión sin sugerir ni tomar decisiones. El pentester ejecuta comandos libremente y el sistema actualiza la KB y logs con lo que ocurre.
                elif sub_choice == "2":
                    print("Entering observation mode...")
                    # Aquí irá un wizard para crear la sesión (tipo, objetivo, etc.)
                    # y luego se creará la sesión con esos parámetros.
                    # session_wizard()
                    
                    mode = "observation" # suggestion, observation
                    goal_type = "recon"
                    target = "127.0.0.1"
                    name = "mvp"
                    max_steps = 5

                    #Se crea sesión con parámetros por defecto (autonomous, goal_type y target hardcodeados)
                    # para poder avanzar con el desarrollo del MVP.
                    session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + name
                    session_dir = SESSIONS_DIR / f"session_{session_id}"
                    print(f"Creating session directory: {session_dir}")
                    session_dir.mkdir(parents=True, exist_ok=True)

                    session_config = {
                        "id": session_id,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    session_context = {
                        "id": session_id,
                        "mode": mode,
                        "goal_type": goal_type,
                        "target": target,
                        "max_steps": max_steps,
                    }

                    s = Session(session_id)

                    print("Running session...")

                    # launch_kb_monitor_window_windows(session_dir)
                    
                    # Creación de KB inicial vacía en la carpeta de la sesión.
                    session_dir.mkdir(parents=True, exist_ok=True)
                    # Esta linea anterior crea la carpeta session_dir en el sistema de archivos.
                    # # parents=True: si faltan carpetas “padre” en la ruta, también las crea. Ejemplo: si data/ o data/sessions/ no existen, los crea automáticamente.
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
                    kb = {
                        "session_id": session_id,

                        # Memoria “pentest”
                        "name_enterprise": "ITIS",
                        "networks": {},
                        "hosts": [],
                        "services": [],
                        "findings": [],
                        "notes": [],

                        # Memoria "entorno local"
                        "net": {
                            "interfaces": [],      # lista de interfaces (name, ipv4, mask, gw, mac opcional)
                            "ipv4": [],            # lista plana (comodín)
                            "default_gw": [],      # lista de gateways
                            "arp_neighbors": [],   # lista de vecinos ARP (ip, mac opcional)
                            "routes": [],          # opcional: ruta (dest, mask/prefix, gateway, if, metric)
                        },
                        "focus": {"level": "global", "host": "", "service": ""},
                        "commands": [],

                        # runtime / trazabilidad mínima (para el estado)
                        "step_idx": 0,
                        "last_action_id": None,
                        "last_action_name": None,
                        "last_rc": None,
                        "last_event_type": None,
                    }

                    (session_dir / "kb.json").write_text(
                        json.dumps(kb, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    # La función de arriba crea/sobrescribe kb.json en la carpeta de la sesión y guarda dentro la KB inicial en formato JSON legible.

                    for t in range(max_steps):  # Simulación de max_steps pasos de la sesión
                        print(f"\n--- Step {t} ---")

                        # ESTADO
                        state = build_state(kb, session_context)
                        print("Current state:", state)

                        # DECISIÓN DE ACCIÓN
                    
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
                                log_freeform_row(session_dir, session_id, {
                                    "type": "FREEFORM",
                                    "t": t,
                                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                    "state": state,
                                    "cmd": command_to_run,
                                })
                                continue
                            action_name, _ = ACTIONS.get(action_id, ("UNKNOWN", None))

                        # ---- aquí ya tienes (state, action_id) => DATASET PURO
                        log_dataset_row(session_dir, session_id, {
                            # "schema": "penhackit.bc.v1",
                            "t": t,
                            # "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "state": state,
                            "action_id": action_id,
                        })

                        print(f"Decided action: {action_name} (ID: {action_id})")
                        print(f"Command to run: {command_to_run}")

                        # DATASET (Behavioral Cloning): guardar (state_t, action_t)
                        # Recomendación: NO guardes comandos libres (action_id=-1) en BC si tu modelo predice action_id.
                        # if action_id >= 0:
                        #     log_bc_row(session_dir, session_id, {
                        #         "type": "BC",
                        #         "t": t,
                        #         "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        #         "mode": "observation",
                        #         "state": state,          # dict numérico/booleano como ya lo construyes
                        #         "action_id": int(action_id),
                        #         "action_name": action_name,
                        #         "cmd": command_to_run,   # opcional, útil para debug
                        #     })
                        # if action_id == -1:
                        #     log_freeform_row(session_dir, session_id, {
                        #         "type":"FREEFORM",
                        #         "t": t,
                        #         "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        #         "state": state,
                        #         "cmd": command_to_run,
                        #     })
                        # EJECUCIÓN
                        result = execute_command(command_to_run)

                        # Para coherencia en los logs
                        command = command_to_run

                        log_command_output(session_dir, session_id, action_id, action_name, result)

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
                        save_kb(session_dir, kb)

                        # Logging del paso completo (estado, acción, comando, resultado) para trazabilidad y posible entrenamiento futuro
                        log_step(session_dir, session_id,{
                        "t": t,
                        "state": state,
                        "action_id": action_id,
                        "command": command,
                        })

                        time.sleep(0.5)
                    print("Session finished\n")
                    break

                elif sub_choice == "0":
                    break
                else:
                    print("Invalid option.")


        elif choice == "2":
            # Lógica de menu training
            while True:
                sub_choice = training_menu()
                if sub_choice == "1":
                    print("Training model...")
                    # Lógica de entrenamiento de modelo usando dataset creado por mi
                    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
                    MODELS_DIR.mkdir(parents=True, exist_ok=True)

                    try:
                        # train_models_from_dataset(DATASETS_DIR, MODELS_DIR )
                        run_training_interactive(DATASETS_DIR, MODELS_DIR)
                    except Exception as e:
                        print(f"Error training model: {e}")

                elif sub_choice == "0":
                    break
                else:
                    print("Invalid option.")


        elif choice == "3":
            # Lógica de menu report
            while True:
                sub_choice = report_menu()
                if sub_choice == "1":

                    # Mostrar listado de sesiones disponibles (carpetas dentro de SESSIONS_DIR) y elegir una para generar el PDF a partir del md generado en esa sesión.
                    sessions_list = [s for s in SESSIONS_DIR.iterdir() if s.is_dir()]
                    if not sessions_list:
                        print("No sessions found.")
                        break
                    print("Available sessions:")
                    for i, s in enumerate(sessions_list):
                        print(f"{i+1}) {s.name}")
                    # options = [str(i) for i in range(1, len(sessions_list) + 1)] + [s.name for s in sessions_list] + ["c", "C"]
                    options =  [s.name for s in sessions_list] + ["c", "C"]
                    completer = WordCompleter(options, ignore_case=True, match_middle=True)

                    session_choice = prompt("Choose session [num|name|c=cancel]> ", completer=completer).strip()
                    session_dir = SESSIONS_DIR / session_choice     # session_dir = SESSIONS_DIR / "session_20260224_000156_mvp" # default para pruebas, luego elegiré de la lista

                    if not session_dir or not session_dir.exists():
                        print(f"Session dir not found: {session_dir}")
                        break

                    kb_path = session_dir / "kb.json"
                    if not kb_path.exists():
                        print(f"kb.json not found: {kb_path}")
                        break

                    # 2) cargar kb (AQUÍ es donde faltaba)
                    kb = json.loads(kb_path.read_text(encoding="utf-8"))
                    
                    # Lógica de generación de reporte usando datos de sesión
                    backend_choice = choose_llm_backend_menu()
                    if backend_choice == 0 or backend_choice == "0":
                        print("Report generation cancelled.")
                        break
                    elif backend_choice == 1 or backend_choice == "1":
                        print("Selected LLM backend: Baseline (no LLM)")
                        try:
                            generate_report_md_baseline(session_dir, kb)
                        except Exception as e:
                            print(f"Error generating baseline report: {e}")

                    elif backend_choice == 2 or backend_choice == "2":
                        print("Selected LLM backend: Ollama (local)")
                        
                        # 3) elegir modelo LLM local (ollama) para generación de reporte
                        # Download model ollama by using: ollama pull <model_name>
                        # model_llm = "qwen3:4b" # esto casi explota

                        models = ollama_list_models_http()

                        if not models:
                            models = ollama_list_models_cli()

                        if not models:
                            print("No Ollama models found (is Ollama running, and are models installed?).")
                            print("Try: ollama list  |  ollama pull <model>")
                            return None
                        
                    
                        default_model = "gemma3:1b"
                        model_llm = choose_ollama_model_interactive(default=default_model)
                        
                        # 4) generar reporte sección por sección usando modelo LLM local (ollama)
                        try:
                            print("Generating report...")
                            generate_report_md_sectionwise(session_dir, kb, model_llm)
                        except Exception as e:
                            print(f"Error generating report: {e}")
                    
                    elif backend_choice == 3 or backend_choice == "3":
                        print("Selected LLM backend: transfomers (local)")
                        # TRANSFORMERS (torch)
                        default_hf = None  # e.g. "qwen2.5-1.5b-instruct" if you want
                        hf_dir = choose_hf_model_dir_interactive(LLM_MODELS_DIR, default_name=default_hf)
                        if not hf_dir:
                            print("Cancelled.")
                            break

                        device = choose_hf_device_interactive()

                        try:
                            print("Generating report...")
                            generate_report_md_sectionwise_llm(
                                session_dir=session_dir,
                                kb=kb,
                                backend="transformers",
                                hf_model_dir=hf_dir,
                                hf_device=device,
                            )
                        except Exception as e:
                            print(f"Error generating report: {e}")

                elif sub_choice == "2":
                    print("Converting report to PDF...")

                    # Mostrar listado de sesiones disponibles (carpetas dentro de SESSIONS_DIR) y elegir una para generar el PDF a partir del md generado en esa sesión.
                    sessions_list = [s for s in SESSIONS_DIR.iterdir() if s.is_dir()]
                    if not sessions_list:
                        print("No sessions found.")
                        break
                    print("Available sessions:")
                    for i, s in enumerate(sessions_list):
                        print(f"{i+1}) {s.name}")
                    # options = [str(i) for i in range(1, len(sessions_list) + 1)] + [s.name for s in sessions_list] + ["c", "C"]
                    options =  [s.name for s in sessions_list] + ["c", "C"]
                    completer = WordCompleter(options, ignore_case=True, match_middle=True)

                    session_choice = prompt("Choose session [num|name|c=cancel]> ", completer=completer).strip()
                    session_dir = SESSIONS_DIR / session_choice     # session_dir = SESSIONS_DIR / "session_20260224_000156_mvp" # default para pruebas, luego elegiré de la lista
                    md_path = session_dir / "report_2.md"
                    # pdf_path = session_dir / "report.pdf"
                    pdf_path = next_available_path(session_dir, "report", ".pdf")

                    try:
                        md_to_pdf_simple(md_path, pdf_path)
                        print(f"PDF generated: {pdf_path}")

                    except Exception as e:
                        print(f"Error converting to PDF: {e}")

                    break
                elif sub_choice == "0":
                    break
                else:
                    print("Invalid option.")

        elif choice == "0":
            print("Exiting...")
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    main()