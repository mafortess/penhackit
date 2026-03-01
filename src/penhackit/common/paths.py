from pathlib import Path

CONFIG_DIR = "config"
DATA_DIR = "data"
SESSIONS_DIR = "sessions"
DATASETS_DIR = "datasets"
ENV_DIR = "env"
MODELS_DIR = "models"
LOGS_DIR = "logs"
LLM_MODELS_DIR = "llm_models"

class Paths:
    def __init__(self, workspace_dir: Path):
        self.workspace_dir = Path(workspace_dir)
        self.config_dir = self.workspace_dir / CONFIG_DIR
        self.data_dir = self.workspace_dir / DATA_DIR
        self.sessions_dir = self.data_dir / SESSIONS_DIR
        self.datasets_dir = self.data_dir / DATASETS_DIR
        self.env_dir = self.data_dir / ENV_DIR
        self.models_dir = self.workspace_dir / MODELS_DIR
        self.logs_dir = self.workspace_dir / LOGS_DIR
        self.llm_models_dir = self.workspace_dir / LLM_MODELS_DIR

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


    # def get_workspace_dir(settings: dict) -> Path:
    #     return Path(settings["paths"]["workspace_dir"])

    # def get_config_dir(settings: dict) -> Path:
    #     return get_workspace_dir(settings) / CONFIG_DIR

    # def get_data_dir(settings: dict) -> Path:
    #     return get_workspace_dir(settings) / DATA_DIR

    # def get_sessions_dir(settings: dict) -> Path:
    #     return get_data_dir(settings) / SESSIONS_DIR    

    # def get_datasets_dir(settings: dict) -> Path:
    #     return get_data_dir(settings) / DATASETS_DIR

    # def get_env_dir(settings: dict) -> Path:
    #     return get_data_dir(settings) / ENV_DIR

    # def get_models_dir(settings: dict) -> Path:
    #     return get_workspace_dir(settings) / MODELS_DIR

    # def get_logs_dir(settings: dict) -> Path:
    #     return get_workspace_dir(settings) / LOGS_DIR

    # def get_llm_models_dir(settings: dict) -> Path:
    #     return get_workspace_dir(settings) / LLM_MODELS_DIR
