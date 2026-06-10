import time
import json
from pathlib import Path

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

def log_command_output(session_dir: Path, session_id: str, action_id: int, action_name: str, result: dict, t: int) -> None:
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
            "t": t,
            "action_id": action_id,
            "action_name": action_name,
            "cmd": result.get("cmd"),
            "rc": result.get("rc"),
            # para MVP: guarda todo; si te preocupa tamaño, usa [:5000]
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")



def log_dataset_row(session_dir, session_id: str, dataset_dir: Path, row: dict) -> None:
    path = dataset_dir / "dataset.jsonl"
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


def init_online_summary(session_id: str, session_settings: dict, session_info: dict) -> dict:
    session_context = session_info.get("session_context", {})
    session_config = session_info.get("session_config", {})

    policy_name = (
        session_settings.get("policy_name")
        or session_settings.get("decider")
        or session_settings.get("policy")
        or "unknown"
    )

    scenario_id = (
        session_config.get("scenario_id")
        or session_config.get("scenario")
        or session_context.get("scenario_id")
        or "unknown"
    )

    goal_type = (
        session_config.get("goal_type")
        or session_context.get("goal_type")
        or session_settings.get("goal_type")
        or "unknown"
    )

    return {
        "schema": "penhackit.online_summary.v1",
        "session_id": session_id,
        "scenario_id": scenario_id,
        "policy_name": policy_name,
        "goal_type": goal_type,
        "model_path": str(session_info.get("model_path")) if session_info.get("model_path") else None,

        "started_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "finished_at_utc": None,
        "_wall_start": time.perf_counter(),

        "success": False,
        "stop_reason": None,
        "steps_total": 0,
        "steps_to_goal": None,

        "progress_steps": 0,
        "repeated_actions": 0,
        "tool_errors": 0,
        "timeouts": 0,

        "active_time_seconds": 0.0,
        "event_type_counts": {},

        "_previous_action_id": None,
    }


def build_step_outcome(
    events: list[dict],
    progress,
    result: dict | None,
    previous_action_id: int | None,
    current_action_id: int | None,
    duration_seconds: float,
) -> dict:
    result = result or {}
    events = events or []

    event_types = extract_event_types(events)
    progress_bool = progress_to_bool(progress)
    repeated = previous_action_id == current_action_id and current_action_id is not None and not progress_bool
    tool_error = is_tool_error(result, event_types)
    timeout = is_timeout(result, event_types)
    goal_reached = "SESSION_OPENED" in event_types

    return {
        "event_types": event_types,
        "events_count": len(events),
        "progress": progress_bool,
        "repeated": repeated,
        "tool_error": tool_error,
        "timeout": timeout,
        "goal_reached": goal_reached,
        "duration_seconds": float(duration_seconds),
    }


def update_online_summary(summary: dict, action_id: int | None, outcome: dict) -> None:
    summary["steps_total"] += 1
    summary["active_time_seconds"] += float(outcome.get("duration_seconds", 0.0))

    if outcome.get("progress"):
        summary["progress_steps"] += 1

    if outcome.get("repeated"):
        summary["repeated_actions"] += 1

    if outcome.get("tool_error"):
        summary["tool_errors"] += 1

    if outcome.get("timeout"):
        summary["timeouts"] += 1

    for event_type in outcome.get("event_types", []):
        summary["event_type_counts"][event_type] = summary["event_type_counts"].get(event_type, 0) + 1

    if outcome.get("goal_reached") and not summary["success"]:
        summary["success"] = True
        summary["steps_to_goal"] = summary["steps_total"]
        summary["stop_reason"] = "goal_reached"

    summary["_previous_action_id"] = action_id


def finish_online_summary(session_dir: Path, summary: dict, stop_reason: str) -> dict:
    if not summary.get("stop_reason"):
        summary["stop_reason"] = stop_reason

    summary["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    wall_time_seconds = time.perf_counter() - summary["_wall_start"]
    summary["wall_time_seconds"] = float(wall_time_seconds)

    steps_total = summary["steps_total"]

    summary["progress_rate"] = safe_div(summary["progress_steps"], steps_total)
    summary["repeated_action_rate"] = safe_div(summary["repeated_actions"], steps_total)
    summary["tool_error_rate"] = safe_div(summary["tool_errors"], steps_total)
    summary["timeout_rate"] = safe_div(summary["timeouts"], steps_total)

    clean_summary = {
        k: v for k, v in summary.items()
        if not k.startswith("_")
    }

    path = session_dir / "online_summary.json"
    path.write_text(
        json.dumps(clean_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return clean_summary


def extract_event_types(events: list[dict]) -> list[str]:
    event_types = []

    for ev in events:
        if not isinstance(ev, dict):
            continue

        event_type = ev.get("type") or ev.get("event_type")
        if event_type:
            event_types.append(str(event_type))

    return event_types


def progress_to_bool(progress) -> bool:
    if progress is None:
        return False

    if isinstance(progress, bool):
        return progress

    if isinstance(progress, (int, float)):
        return progress > 0

    if isinstance(progress, dict):
        return any(progress_to_bool(v) for v in progress.values())

    if isinstance(progress, (list, tuple, set)):
        return any(progress_to_bool(v) for v in progress)

    return False


def is_tool_error(result: dict, event_types: list[str]) -> bool:
    if "TOOL_ERROR" in event_types or "ACTION_FAILED" in event_types or "COMMAND_FAILED" in event_types:
        return True

    rc = result.get("rc")

    if rc is None:
        return False

    try:
        return int(rc) != 0
    except Exception:
        return False


def is_timeout(result: dict, event_types: list[str]) -> bool:
    if "TIMEOUT" in event_types or "COMMAND_TIMEOUT" in event_types:
        return True

    return bool(result.get("timeout", False))


def safe_div(num: int | float, den: int | float) -> float:
    return float(num / den) if den else 0.0