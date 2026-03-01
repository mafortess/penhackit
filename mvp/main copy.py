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

BLOCKED = {"powershell", "cmd", "bash", "sh", "zsh"}

def is_interactive_shell(cmd: str) -> bool:
    c = cmd.strip().lower()
    return c in BLOCKED or c.startswith("powershell ") or c.startswith("cmd ") or c.startswith("bash ")



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



def main():
    # Lógica navegación entre menús
    while True:
        print(random.choice(BANNERS))
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
                       
                    print("Session finished\n")
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