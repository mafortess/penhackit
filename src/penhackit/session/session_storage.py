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



def load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read JSON file {path}: {exc}")
        return {}

def load_session_online_summary(sessions_dir: Path, session_id: str) -> dict:
    session_dir = sessions_dir / session_id
    return load_json_if_exists(session_dir / "online_summary.json")

def count_jsonl_records(path: Path, skip_meta: bool = True) -> int:
    if not path.exists():
        return 0

    count = 0

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                if skip_meta:
                    try:
                        obj = json.loads(line)
                        if obj.get("type") == "META":
                            continue
                    except Exception:
                        pass

                count += 1

    except Exception as exc:
        print(f"Could not count JSONL records in {path}: {exc}")
        return 0

    return count


def build_session_view_rows(sessions_dir: Path) -> list[dict]:
    rows = []

    for session_id in list_sessions(sessions_dir):
        session_dir = sessions_dir / session_id

        kb = load_session_details(sessions_dir, session_id) or {}
        online_summary = load_json_if_exists(session_dir / "online_summary.json")

        session_block = kb.get("session", {}) if isinstance(kb, dict) else {}
        scope_block = kb.get("scope", {}) if isinstance(kb, dict) else {}

        row = {
            "session_id": session_id,
            "session_dir": str(session_dir),

            "name": session_block.get("name", "-"),
            "mode": session_block.get("mode", "-"),

            "goal_type": (
                online_summary.get("goal_type")
                or scope_block.get("goal_type")
                or scope_block.get("goal")
                or "-"
            ),

            "target": (
                scope_block.get("target")
                or "-"
            ),

            "steps": count_jsonl_records(session_dir / "steps.jsonl"),
            "commands": count_jsonl_records(session_dir / "command_outputs.jsonl"),
            "freeform_rows": count_jsonl_records(session_dir / "dataset_freeform.jsonl"),

            "has_kb": (session_dir / "kb.json").exists(),
            "has_steps": (session_dir / "steps.jsonl").exists(),
            "has_commands": (session_dir / "command_outputs.jsonl").exists(),
            "has_online_summary": (session_dir / "online_summary.json").exists(),

            "success": online_summary.get("success", "-"),
            "stop_reason": online_summary.get("stop_reason", "-"),
            "steps_total": online_summary.get("steps_total", "-"),
            "progress_rate": online_summary.get("progress_rate", "-"),
            "repeated_action_rate": online_summary.get("repeated_action_rate", "-"),
            "tool_error_rate": online_summary.get("tool_error_rate", "-"),
            "active_time_seconds": online_summary.get("active_time_seconds", "-"),
        }

        rows.append(row)

    return rows

def list_session_online_summaries(sessions_dir: Path) -> list[dict]:
    rows = []

    for session_id in list_sessions(sessions_dir):
        summary = load_session_online_summary(sessions_dir, session_id)

        if not summary:
            continue

        summary["_session_id"] = session_id
        summary["_session_dir"] = str(sessions_dir / session_id)
        rows.append(summary)

    return rows

import csv
from collections import defaultdict


def get_online_evaluations_dir(paths) -> Path:
    data_dir = getattr(paths, "data_dir", None)

    if data_dir is None:
        data_dir = paths.sessions_dir.parent

    out_dir = data_dir / "evaluations" / "online"
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def save_online_runs_csv(path: Path, summaries: list[dict]) -> None:
    fieldnames = [
        "_session_id",
        "scenario_id",
        "policy_name",
        "goal_type",
        "model_path",
        "success",
        "stop_reason",
        "steps_total",
        "steps_to_goal",
        "progress_steps",
        "repeated_actions",
        "tool_errors",
        "timeouts",
        "progress_rate",
        "repeated_action_rate",
        "tool_error_rate",
        "timeout_rate",
        "active_time_seconds",
        "wall_time_seconds",
        "started_at_utc",
        "finished_at_utc",
        "_session_dir",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in summaries:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_grouped_online_results(summaries: list[dict]) -> list[dict]:
    groups = defaultdict(list)

    for row in summaries:
        key = (
            row.get("policy_name", "unknown"),
            row.get("scenario_id", "unknown"),
            row.get("goal_type", "unknown"),
        )
        groups[key].append(row)

    grouped_rows = []

    for (policy_name, scenario_id, goal_type), rows in groups.items():
        runs = len(rows)
        successes = sum(1 for row in rows if row.get("success") is True)

        grouped_rows.append({
            "policy_name": policy_name,
            "scenario_id": scenario_id,
            "goal_type": goal_type,
            "runs": runs,
            "successes": successes,
            "success_rate": safe_avg([1.0 if row.get("success") is True else 0.0 for row in rows]),
            "avg_steps_total": safe_avg_field(rows, "steps_total"),
            "avg_steps_to_goal": safe_avg_field(rows, "steps_to_goal"),
            "avg_progress_rate": safe_avg_field(rows, "progress_rate"),
            "avg_repeated_action_rate": safe_avg_field(rows, "repeated_action_rate"),
            "avg_tool_error_rate": safe_avg_field(rows, "tool_error_rate"),
            "avg_timeout_rate": safe_avg_field(rows, "timeout_rate"),
            "avg_active_time_seconds": safe_avg_field(rows, "active_time_seconds"),
            "avg_wall_time_seconds": safe_avg_field(rows, "wall_time_seconds"),
        })

    grouped_rows.sort(
        key=lambda row: (
            str(row["scenario_id"]),
            str(row["goal_type"]),
            str(row["policy_name"]),
        )
    )

    return grouped_rows


def save_online_grouped_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "policy_name",
        "scenario_id",
        "goal_type",
        "runs",
        "successes",
        "success_rate",
        "avg_steps_total",
        "avg_steps_to_goal",
        "avg_progress_rate",
        "avg_repeated_action_rate",
        "avg_tool_error_rate",
        "avg_timeout_rate",
        "avg_active_time_seconds",
        "avg_wall_time_seconds",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def safe_avg_field(rows: list[dict], field: str):
    values = []

    for row in rows:
        value = row.get(field)

        if value is None or value == "" or value == "-":
            continue

        try:
            values.append(float(value))
        except Exception:
            continue

    return safe_avg(values)


def safe_avg(values: list[float]):
    if not values:
        return ""

    return sum(values) / len(values)