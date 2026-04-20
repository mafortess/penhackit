import json
import time
from pathlib import Path

from penhackit.common.paths import Paths
from penhackit.session.kb.kb_updater import  build_initial_kb

from penhackit.session.decision.model_loader import load_decision_model
from penhackit.session.kb.kb_updater import launch_kb_monitor_window_windows


def create_session_runtime(session_settings: dict, env_profile: dict, paths: Paths) -> None:
    """
    Lógica principal para ejecutar una sesión de pentesting en modo autónomo, observación o sugerencia.
    """
    print("Creating session runtime...")

    # 1) Crear carpeta de sesión (session_dir) con un nombre único basado en la fecha/hora y el nombre de la sesión.
    session_id, session_dir = create_session_workspace(session_settings, paths)

    # 2) Crear archivos de configuración y contexto de la sesión (session_config.json y session_context.json) con los datos de la sesión.
    session_config, session_context = build_session_metadata(session_id, session_settings)
    persist_session_metadata(session_dir, session_config, session_context)

    # 3) Crear archivo de KB inicial (kb.json) con datos predeterminados o vacío.    
    kb = initialize_kb(session_id, session_dir)

    # 4) Si la política de decisión de la sesión es basada en modelo, cargar el modelo de decisión entrenado y su metadata (feature_names) para usarlo durante la sesión.    
    model, feature_names = load_model_if_needed(session_settings, paths)
        
    # 5) Construir un dict session_info con toda la información relevante de la sesión (session_id, session_dir, session_config, session_context, kb, model, feature_names) para pasarla a las funciones de lógica de cada modo de sesión (autonomous, observation, suggestion).
    session_info = build_session_info(session_id, session_dir, session_config, session_context, kb, model, feature_names)

    # Si la configuración de la sesión indica que se debe lanzar el monitor de KB, lo lanza pasando la ruta de session_dir para que pueda leer/escribir los archivos de KB y contexto.
    if session_settings["launch_kb_monitor"]:
        launch_kb_monitor_window_windows(session_dir)

    return session_info

    # dispatch_session_mode(session_settings, paths, session_info)

# ============================================================================================================
# Subfunciones para cada parte de la lógica de new_session_logic, para mantener el código organizado y modular:

def create_session_workspace(session_settings: dict, paths: Paths) -> tuple[str, Path]:
    print("Creating session workspace...")
    session_id = time.strftime("%Y%m%d_%H%M%S") + "_" + session_settings["name"].replace(" ", "_")
    session_dir = paths.sessions_dir / session_id
    
    # parents=True: si faltan carpetas “padre” en la ruta, también las crea. Ejemplo: si data/ o data/sessions/ no existen, los crea automáticamente.
    # exist_ok=True: si la carpeta ya existe, no da error. Sin esto, mkdir() lanzaría una excepción si la carpeta ya existe.
    
    # Crear la carpeta session_dir en el sistema de archivos.
    print(f"Creating session directory: {session_dir}")
    session_dir.mkdir(parents=True, exist_ok=False)

    return session_id, session_dir

def build_session_metadata(session_id: str, session_settings: dict):
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
    return session_config, session_context

def persist_session_metadata(session_dir: Path, session_config: dict, session_context: dict):
    print("Saving session config and context...")

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

def initialize_kb(session_id: str, session_dir: Path):
    print("Initializing KB...")
    kb = build_initial_kb(session_id)

    # 3) kb.json (conocimiento, inicialmente vacío o con datos predeterminados)
    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return kb

def load_model_if_needed(session_settings: dict, paths: Paths):
    print("Loading model for session (if applicable)...")
    model = None
    feature_names = None
    
    if session_settings["mode"] in ["autonomous", "suggestion"]:
        model_path = paths.models_dir / "decision_tree" / "model.joblib"
        metrics_path = model_path.parent / "metrics.json"
        
        if not model_path.exists() or not metrics_path.exists():
            print(f"Model files not found: {model_path}, {metrics_path}")
            print("Make sure to train a model first and place the files in mvp/models/")
            return None, None
        
        model, feature_names = load_decision_model(model_path, metrics_path)
        print(f"Loaded model from {model_path} with features: {feature_names}")

    return model, feature_names

def build_session_info(session_id, session_dir, session_config, session_context, kb, model, feature_names):
    return {
        "session_id": session_id,
        "session_dir": session_dir,
        "session_config": session_config,
        "session_context": session_context,
        "kb": kb,
        "model": model,
        "feature_names": feature_names,
    }