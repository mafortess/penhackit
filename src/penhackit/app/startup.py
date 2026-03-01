from datetime import datetime, timezone
from pathlib import Path
import os
import platform
import json
import shutil
import sys

SETTINGS_FILENAME = "settings.json"
ENVIRONMENT_PROFILE_FILENAME = "environment_profile.json"

from penhackit.common.paths import Paths

def bootstrap_app() -> dict:
    print("Bootstrapping application...")

    print("0) Resolving workspace directory...")
    workspace_dir = resolve_workspace_dir()
    paths = Paths(workspace_dir)

    print("1) Loading settings...")
    settings = setup_settings(workspace_dir)
    # print(f"Settings loaded: {settings}")

    print("2) Setting up workspace directories...")   
    setup_workspace_dirs(paths)
    
    print("3) Loading environment profiles...")
    enviroment_profile = setup_environment_profile(workspace_dir, settings)

    return {
        "paths": paths,
        "settings": settings,
        "enviroment_profile": enviroment_profile,
    }

# ===============================================================
# Functions for setting up workspace
def resolve_workspace_dir() -> Path:
    workspace_dir_str = os.getenv("WORKSPACE_DIR", "workspace").strip()

    if not workspace_dir_str:
        workspace_dir_str = "workspace"

    return Path(workspace_dir_str)


# Function version using atribute from Paths class
def setup_workspace_dirs(paths: Path) -> None:
    workspace_dirs = [
        paths.config_dir,
        paths.sessions_dir,
        paths.datasets_dir,
        paths.env_dir,
        paths.models_dir,
        paths.logs_dir,
        paths.llm_models_dir,
    ]

    for directory in workspace_dirs:
        directory.mkdir(parents=True, exist_ok=True)

# Function version using helper functions from Paths class
# def setup_workspace_dirs(settings: dict) -> None:
#     workspace_dirs = [
#         get_config_dir(settings),
#         get_sessions_dir(settings),
#         get_datasets_dir(settings),
#         get_env_dir(settings),
#         get_models_dir(settings),
#         get_logs_dir(settings),
#         get_llm_models_dir(settings),
#     ]
#     for directory in workspace_dirs:
#         directory.mkdir(parents=True, exist_ok=True)

# Functions for setting up settings
def build_default_settings(workspace_dir: Path) -> dict:
    return {
        "schema_id": "settings.v1",
        "app": {
            "show_banner": True,
            "verbose": True,
            "refresh_environment_on_startup": False,
            "active_profile": "default"
        },
        "paths": {
            "workspace_dir": str(workspace_dir),   
        },
        "session": {
            "default_name": "mvp",
            "default_mode": "autonomous",
            "default_goal_type": "recon",
            "default_target": "127.0.0.1",
            "default_max_steps": 5,
            "launch_kb_monitor": True,
        },
        "training": {
            "default_test_size": 0.25,
            "default_random_state": 42,
            "default_model_type": "decision_tree",
        },
        "report": {
            "default_backend": "baseline",
            "default_output_format": "md",
            "default_ollama_model": "gemma3:1b",
            "default_transformers_model": "",
            "default_transformers_device": "cpu",
            "export_pdf_after_generation": False,
        },
    }

def load_settings(workspace_dir: Path) -> dict | None:
    settings_path = workspace_dir / "config" / SETTINGS_FILENAME

    if not settings_path.exists():
        return None

    with settings_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(workspace_dir: Path, settings: dict) -> Path:
    settings_path = workspace_dir / "config" / SETTINGS_FILENAME

    with settings_path.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2, ensure_ascii=False)

    return settings_path

def setup_settings(workspace_dir: Path) -> dict:
    # Aquí se cargarían las configuraciones desde archivos o variables de entorno
    settings = load_settings(workspace_dir)

    if settings is None:
        print("No settings found, creating default settings.")
        settings = build_default_settings(workspace_dir)
        save_settings(workspace_dir, settings)
    
    return settings

# ===============================================================
# Functions for setting up environment profiles
def detect_system_info() -> dict:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "hostname": platform.node(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
    }

def detect_tool_groups() -> dict:
    detected = {}
    from penhackit.common.environment import TOOL_CATALOG
    for group_name, tool_names in TOOL_CATALOG.items():
        detected[group_name] = {}

        for tool_name in tool_names:
            detected[group_name][tool_name] = shutil.which(tool_name) is not None

    return detected

def build_capabilities(tool_groups: dict) -> dict:
    basic_network = tool_groups.get("basic_network", {})
    host_discovery = tool_groups.get("host_discovery", {})
    service_enumeration = tool_groups.get("service_enumeration", {})
    web_enumeration = tool_groups.get("web_enumeration", {})
    vulnerability_analysis = tool_groups.get("vulnerability_analysis", {})
    exploitation = tool_groups.get("exploitation", {})

    return {
        "can_do_basic_network_checks": any(basic_network.values()),
        "can_do_host_discovery": any(host_discovery.values()),
        "can_do_service_enumeration": any(service_enumeration.values()),
        "can_do_web_enumeration": any(web_enumeration.values()),
        "can_do_vulnerability_analysis": any(vulnerability_analysis.values()),
        "can_do_exploitation": any(exploitation.values()),
    }

def build_goal_support(capabilities: dict) -> dict:
    return {
        "recon": (
            capabilities["can_do_basic_network_checks"]
            or capabilities["can_do_host_discovery"]
        ),
        "enumeration": (
            capabilities["can_do_service_enumeration"]
            or capabilities["can_do_web_enumeration"]
        ),
        "vulnerability_discovery": capabilities["can_do_vulnerability_analysis"],
        "exploitation": capabilities["can_do_exploitation"],
    }

def detect_environment_profile() -> dict:
    system_info = detect_system_info()  
    detected_tools = detect_tool_groups()
    capabilities = build_capabilities(detected_tools)
    goal_support = build_goal_support(capabilities)

    return {
        "schema_id": "environment_profile.v1",
        "detected_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system": system_info,
        "tools": detected_tools,
        "capabilities": capabilities,
        "goal_support": goal_support,
    }

def load_environment_profile(workspace_dir: Path) -> dict | None:
    profile_path = workspace_dir / "data" / "env" / ENVIRONMENT_PROFILE_FILENAME

    if not profile_path.exists():
        return None

    with profile_path.open("r", encoding="utf-8") as f:
        return json.load(f)
    
def save_environment_profile(workspace_dir: Path, profile: dict) -> Path:
    profile_path = workspace_dir / "data" / "env" / ENVIRONMENT_PROFILE_FILENAME

    with profile_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    return profile_path

def setup_environment_profile(workspace_dir: Path, settings: dict) -> dict:
    refresh = settings["app"]["refresh_environment_on_startup"]
    
    if not refresh:
        profile = load_environment_profile(workspace_dir)
        if profile is not None:
            print("Environment profile loaded from disk.")
            return profile
    profile = detect_environment_profile()
    
    save_environment_profile(workspace_dir, profile)
    print("Environment profile detected and saved to disk.")
    return profile