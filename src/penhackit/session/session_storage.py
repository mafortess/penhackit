from pathlib import Path
import json
import shutil

def list_sessions(sessions_dir: Path) -> list[str]:
    if not sessions_dir.exists() or not sessions_dir.is_dir():
        return []

    sessions = [p.name for p in sessions_dir.iterdir() if p.is_dir()]           
    sessions.sort()
    return sessions


def load_session_details(sessions_dir: Path, session_id: str) -> dict | None:
    session_dir = sessions_dir / session_id
    if not session_dir.exists():
        return None

    kb_path = session_dir / "kb.json"
    if kb_path.exists():
        return json.loads(kb_path.read_text(encoding="utf-8"))

    return {"session_id": session_id, "status": "No kb.json found"}


def delete_session(sessions_dir: Path, session_id: str) -> None:
    session_dir = sessions_dir / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir)