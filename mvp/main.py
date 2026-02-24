from pathlib import Path
import json
import subprocess
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

import markdown
import matplotlib.pyplot as plt

# from weasyprint import HTML

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

# Ruta base para sesiones, modelos, reportes.
BASE_DIR = Path("mvp")
SESSIONS_DIR = BASE_DIR / "sessions"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
REPORTS_DIR = BASE_DIR / "reports"

# Menu principal: sesiones, entrenamiento, reporte.
def main_menu():
    print("=== AGENT COMMAND-LINE INTERFACE (CLI) ===")
    print("1) Run session")
    print("2) Train models")
    print("3) Generate report")
    print("0) Exit")
# Menu de sesiones: run session en autonomous mode.
def session_menu():
    print("--- Run session ---")
    print("1) Run session (autonomous)")
    print("0) Back")
# Menu de entrenamiento: entrenar modelo arbol de decisión con dataset creado por mi.
def training_menu():
    print("--- Train models ---")
    print("1) Train decision tree model")
    print("0) Back")
# Menu de reporte: generar reporte con métricas de la sesión (en KB creada por mi).
def report_menu():
    print("--- Generate report ---")
    print("1) Generate session report")
    print("2) Export report.pdf from report.md")
    print("0) Back")

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

# def train_models_from_dataset(dataset_dir: Path, models_dir: Path) -> Path:
#     rows = load_dataset_jsonl(dataset_dir)
#     X, y, feature_names = vectorize_dataset(rows)

#     model_id = f"models_{dataset_dir.name}"
#     out_dir = models_dir / model_id
#     out_dir.mkdir(parents=True, exist_ok=True)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.25, random_state=42, stratify=y if len(set(y)) > 1 else None
#     )

#     models = {
#         "logreg": LogisticRegression(max_iter=2000),
#         "decision_tree": DecisionTreeClassifier(random_state=42),
#         "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
#         "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
#     }

#     metrics = {
#         "dataset_dir": str(dataset_dir),
#         "n_samples": int(len(y)),
#         "n_features": int(X.shape[1]),
#         "feature_names": feature_names,
#         "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#         "models": {},
#     }

#     for name, model in models.items():
#         print(f"\n[train] {name} ...")
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)

#         acc = float(accuracy_score(y_test, y_pred))
#         cm = confusion_matrix(y_test, y_pred).tolist()
#         rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

#         print(f"[eval] {name} accuracy: {acc:.4f}")

#         model_path = out_dir / f"{name}.joblib"
#         joblib.dump(model, model_path)

#         metrics["models"][name] = {
#             "accuracy": acc,
#             "confusion_matrix": cm,
#             "model_path": str(model_path),
#         }

#     (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
#     print(f"\nSaved models to: {out_dir}")
#     return out_dir

def list_dataset_candidates(datasets_dir: Path) -> list[Path]:
    """
    Devuelve una lista de 'dataset_dir' candidatos.
    - Si existe datasets/dataset.jsonl, añade datasets/ como candidato.
    - Añade cada subdirectorio que contenga dataset.jsonl.
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

    raw = input("Select number (0 cancel)> ").strip()
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
    path = dataset_dir / "dataset.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset.jsonl not found: {path}")

    rows = []
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
    raw = input("Select model (0 cancel)> ").strip()
    if raw == "0":
        return None
    if raw not in MODEL_CHOICES:
        print("Invalid option.")
        return None
    return MODEL_CHOICES[raw]

# =========================
# Training (single function)
# =========================

from collections import Counter

def run_training_interactive(datasets_dir: Path, models_dir: Path) -> None:
    """
    Interactive training:
    - choose dataset folder (must contain dataset.jsonl)
    - choose model type
    - train + evaluate
    - save model + metrics under models_dir/<dataset>/<model>_<n>/
    """
    dataset_dir = choose_dataset_dir(datasets_dir)
    if not dataset_dir:
        return

    choice = choose_model_type()
    if not choice:
        return
    model_key, model_desc, model_factory = choice

    print(f"\nSelected dataset: {dataset_dir.name}")
    print(f"Selected model: {model_key} ({model_desc})")

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


def main():

    # Lógica navegación entre menús
    while True:
        main_menu()
        choice = input("> ").strip()

        if choice == "1":
            # Lógica de menu session
            while True:
                session_menu()
                sub_choice = input("> ").strip()
                if sub_choice == "1":
                    # Aquí irá un wizard para crear la sesión (tipo, objetivo, etc.)
                    # y luego se creará la sesión con esos parámetros.
                    # session_wizard()
                    mode = "autonomous" # suggestion, observation
                    # mode = "suggestion"
                    goal_type = "recon"
                    target = "127.0.0.1"
                    name = "mvp"

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
                        "target": target
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

                    for t in range(5):  # Simulación de 10 pasos de la sesión
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
                            action_id = policy_decide_action(state, t)
                            action_name, cmd_template = ACTIONS.get(action_id, ("NONE", None))

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
                training_menu()
                sub_choice = input("> ").strip()
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
                report_menu()
                sub_choice = input("> ").strip()
                if sub_choice == "1":
                    print("Generating report...")
                    # Lógica de generación de reporte usando datos de sesión
                    session_dir = SESSIONS_DIR / "session_20260224_000156_mvp"

                    if not session_dir or not session_dir.exists():
                        print(f"Session dir not found: {session_dir}")
                        break

                    kb_path = session_dir / "kb.json"
                    if not kb_path.exists():
                        print(f"kb.json not found: {kb_path}")
                        break

                    # 2) cargar kb (AQUÍ es donde faltaba)
                    kb = json.loads(kb_path.read_text(encoding="utf-8"))

                    # 3) generar reporte sección por sección usando modelo LLM local (ollama)
                    # Download model ollama by using: ollama pull <model_name>
                    # model_llm = "qwen3:4b" # esto casi explota
                    model_llm = "gemma3:1b"
                    try:
                        generate_report_md_sectionwise(session_dir, kb, model_llm)
                    except Exception as e:
                        print(f"Error generating report: {e}")

                elif sub_choice == "2":
                    print("Converting report to PDF...")
                    session_dir = SESSIONS_DIR / "session_20260224_000156_mvp"
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