import json
import subprocess
from pathlib import Path

import requests

from penhackit.common.paths import next_available_path

def list_reportable_sessions(sessions_dir: Path) -> list[str]:
    if not sessions_dir.exists() or not sessions_dir.is_dir():
        return []

    sessions = []
    for path in sessions_dir.iterdir():
        if path.is_dir():
            # Check if the session has the expected reportable structure ("kb.json")
            kb_path = path / "kb.json"
            # print(f"Checking session '{path.name}' for reportability: looking for {kb_path}")
            # print(f"  Exists: {kb_path.exists()}, Is file: {kb_path.is_file()}")
            if kb_path.exists() and kb_path.is_file():  
                sessions.append(path.name)
            else:
                print(f"Skipping session '{path.name}' - missing kb.json")

    sessions.sort()
    return sessions

# ===========================================   ======================
# List local Ollama models
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

def list_local_ollama_models() -> list[str]:
    """
    Returns a list of local Ollama model names, trying HTTP API first and falling back to CLI if needed.
    """
    models = ollama_list_models_http()
    if not models:
        models = ollama_list_models_cli()
    return models


def ollama_list_models_http(timeout_s: int = 10) -> list[str]:
    """
    Returns installed model names from Ollama.
    Uses /api/tags so it reflects what's available locally.
    """
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=timeout_s)
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


# =================================================================
# List local Transformers models (directories under mvp/llm_models/)
# def list_local_transformers_models(models_dir: str) -> list[str]:
#     if not models_dir:
#         return []

#     root = Path(models_dir)
#     if not root.exists() or not root.is_dir():
#         return []

#     models = []
#     for path in root.iterdir():
#         if path.is_dir():
#             models.append(path.name)

#     models.sort()
#     return models

# 4) List local HF models (directories under mvp/llm_models/)
def list_local_hf_transformers_models(models_dir: Path) -> list[Path]:
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
