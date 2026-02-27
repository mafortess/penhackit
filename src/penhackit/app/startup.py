from pathlib import Path
import os

def bootstrap_app() -> dict:
    print("Bootstrapping application...")

    print("1) Setting up workspace directories...")
    workspace_dir = resolve_workspace_dir()
    setup_workspace_dirs(workspace_dir)
    
    print("2) Loading settings...")
    settings = setup_settings(workspace_dir)
    
    print("3) Loading environment profiles...")
    enviroment_profile = setup_environment_profile(workspace_dir)

    return {
        "workspace_dir": workspace_dir,
        "settings": settings,
        "enviroment_profile": enviroment_profile,
    }


def setup_workspace_dirs(workspace_dir: Path) -> None:
    workspace_dirs = [
        workspace_dir / "config",
        workspace_dir / "data" / "sessions",
        workspace_dir / "data" / "datasets",
        workspace_dir / "data" / "env",
        workspace_dir / "models",
        workspace_dir / "logs",
    ]

    for directory in workspace_dirs:
        directory.mkdir(parents=True, exist_ok=True)

def setup_settings(workspace_dir: Path) -> None:
    # Aquí se cargarían las configuraciones desde archivos o variables de entorno
    pass

def setup_environment_profile(workspace_dir: Path) -> None:
    # Aquí se cargarían los perfiles de entorno (por ejemplo, para diferentes tipos de pruebas)
    pass
    

def resolve_workspace_dir() -> Path:
    workspace_dir_str = os.getenv("WORKSPACE_DIR", "workspace").strip()

    if not workspace_dir_str:
        workspace_dir_str = "workspace"

    return Path(workspace_dir_str)