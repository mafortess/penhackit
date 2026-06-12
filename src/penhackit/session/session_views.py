from penhackit.session.session_storage import (
    list_sessions,
    delete_session,
    load_session_details,
    load_session_online_summary,
    count_jsonl_records,
    list_session_online_summaries,
    get_online_evaluations_dir,
    save_online_grouped_csv,
    build_grouped_online_results,
    save_online_runs_csv
)

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from pathlib import Path

def run_view_sessions_view(paths: dict) -> None:
    print("Viewing sessions...")
    
    sessions = list_sessions(paths.sessions_dir)
    
    if not sessions:
        print("No sessions found.")
        return
    
    print("\n--- Available Sessions ---")
    for index, session in enumerate(sessions, start=1):
        print(f"{index}. {session}")

    options = [str(i) for i in range(1, len(sessions) + 1)]
    selection = prompt("Select session to view (or 0 to go back)> ", completer=WordCompleter(options + ["0"])).strip()
    if selection == "0":
        return
    if not selection.isdigit() or int(selection) < 1 or int(selection) > len(sessions):
        print("Invalid selection.")
        return
    
    selected_session = sessions[int(selection) - 1]
    print(f"Selected session: {selected_session}")
    run_selected_session_view(paths, selected_session)


def run_selected_session_view(paths, session_id: str) -> None:
    while True:
        print(f"\n--- Session: {session_id} ---")
        print("1) Show summary")
        print("2) Show details")
        print("3) Show online summary")
        print("4) Delete session")
        print("0) Back")

        choice = prompt("Select option> ", completer=WordCompleter(["1", "2","3", "4", "0"])).strip()

        if choice == "1":
            show_session_summary_view(paths, session_id)            
        elif choice == "2": 
            show_session_details_view(paths, session_id)
        elif choice == "3":
            show_session_online_summary_view(paths, session_id)
        elif choice == "4":
            confirm = prompt(f"Delete session '{session_id}'? [y/N]> ", completer=WordCompleter(["y", "N"])).strip().lower()
            if confirm == "y":
                delete_session(paths.sessions_dir, session_id)
                print("Session deleted.")
                return

        elif choice == "0":
            return

        else:
            print("Invalid option.")

def show_session_summary_view(paths, session_id: str) -> None:
    kb = load_session_details(paths.sessions_dir, session_id)

    if kb is None:
        print("Session not found.")
        return

    online_summary = load_session_online_summary(paths.sessions_dir, session_id)
    summary = build_session_summary_data(session_id, kb, paths.sessions_dir, online_summary)

    print(f"\n--- Session summary: {session_id} ---")
    print(f"Session ID: {summary['session_id']}")
    print(f"Name: {summary['name']}")
    print(f"Mode: {summary['mode']}")
    print(f"Goal type: {summary['goal_type']}")
    print(f"Target: {summary['target']}")
    # print(f"Status: {summary['status']}")
    # print(f"Steps executed: {summary['steps']}")
    print(f"Hosts discovered: {summary['hosts']}")
    print(f"Services discovered: {summary['services']}")
    print(f"Findings: {summary['findings']}")
    print(f"Reports available: {summary['reports_available']}")
    

def build_session_summary_data(session_id: str, kb: dict, sessions_dir: Path, online_summary: dict) -> dict:
    online_summary = online_summary or {}

    session_context = kb.get("session_context", {}) or {}
    session_block = kb.get("session", {}) or {}
    scope_block = kb.get("scope", {}) or {}

    session_dir = sessions_dir / session_id
    reports_available = has_session_reports(sessions_dir, session_id)

    return {
        "session_id": session_id,

        "name": (
            session_context.get("name")
            or session_block.get("name")
            or session_id
        ),

        "mode": (
            session_context.get("mode")
            or session_block.get("mode")
            or "unknown"
        ),

        "goal_type": (
            session_context.get("goal_type")
            or scope_block.get("goal_type")
            or online_summary.get("goal_type")
            or "unknown"
        ),

        "target": (
            session_context.get("target")
            or scope_block.get("target")
            or "unknown"
        ),

        "hosts": len(kb.get("hosts", {}) or {}),
        "services": len(kb.get("services", {}) or {}),
        "findings": len(kb.get("findings", []) or []),

        "steps": count_jsonl_records(session_dir / "steps.jsonl"),
        "commands": count_jsonl_records(session_dir / "command_outputs.jsonl"),

        "has_online_summary": "yes" if online_summary else "no",
        "success": online_summary.get("success", "-"),
        "stop_reason": online_summary.get("stop_reason", "-"),
        "steps_total": online_summary.get("steps_total", "-"),
        "progress_rate": online_summary.get("progress_rate", "-"),
        "active_time_seconds": online_summary.get("active_time_seconds", "-"),

        "reports_available": "yes" if reports_available else "no",
    }



def has_session_reports(sessions_dir: Path, session_id: str) -> bool:
    session_dir = sessions_dir / session_id
    if not session_dir.exists():
        return False

    for path in session_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".md", ".pdf"} and "report" in path.name.lower():
            return True

    return False


def show_session_details_view(paths, session_id: str) -> None:
    details = load_session_details(paths.sessions_dir, session_id)

    if details is None:
        print("Session not found.")
        return

    print(f"\n--- Session details: {session_id} ---")
    for key, value in details.items():
        print(f"{key}: {value}")


def show_session_online_summary_view(paths, session_id: str) -> None:
    summary = load_session_online_summary(paths.sessions_dir, session_id)

    if not summary:
        print("No online_summary.json found for this session.")
        return

    print(f"\n--- Online summary: {session_id} ---")
    print(f"Scenario ID: {summary.get('scenario_id', '-')}")
    print(f"Policy name: {summary.get('policy_name', '-')}")
    print(f"Goal type: {summary.get('goal_type', '-')}")
    print(f"Model path: {summary.get('model_path', '-')}")
    print(f"Success: {format_bool_or_dash(summary.get('success', '-'))}")
    print(f"Stop reason: {summary.get('stop_reason', '-')}")
    print(f"Steps total: {summary.get('steps_total', '-')}")
    print(f"Steps to goal: {summary.get('steps_to_goal', '-')}")
    print(f"Progress steps: {summary.get('progress_steps', '-')}")
    print(f"Repeated actions: {summary.get('repeated_actions', '-')}")
    print(f"Tool errors: {summary.get('tool_errors', '-')}")
    print(f"Timeouts: {summary.get('timeouts', '-')}")
    print(f"Progress rate: {format_float(summary.get('progress_rate', '-'))}")
    print(f"Repeated action rate: {format_float(summary.get('repeated_action_rate', '-'))}")
    print(f"Tool error rate: {format_float(summary.get('tool_error_rate', '-'))}")
    print(f"Active time seconds: {format_float(summary.get('active_time_seconds', '-'))}")
    print(f"Wall time seconds: {format_float(summary.get('wall_time_seconds', '-'))}")

    event_counts = summary.get("event_type_counts", {})
    if event_counts:
        print("\nEvent type counts:")
        for event_type, count in event_counts.items():
            print(f"  {event_type}: {count}")


def format_float(value) -> str:
    if value == "-" or value is None:
        return "-"

    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def format_bool_or_dash(value) -> str:
    if value == "-" or value is None:
        return "-"

    if value is True:
        return "yes"

    if value is False:
        return "no"

    return str(value)

def shorten(value, width: int) -> str:
    value = str(value)

    if len(value) <= width:
        return value

    if width <= 3:
        return value[:width]

    return value[: width - 3] + "..."


def run_view_online_results_view(paths) -> None:
    summaries = list_session_online_summaries(paths.sessions_dir)

    if not summaries:
        print("No online summaries found.")
        return

    out_dir = get_online_evaluations_dir(paths)

    runs_csv = out_dir / "online_runs.csv"
    grouped_csv = out_dir / "online_grouped.csv"

    save_online_runs_csv(runs_csv, summaries)

    grouped_rows = build_grouped_online_results(summaries)
    save_online_grouped_csv(grouped_csv, grouped_rows)

    print("\n--- Online evaluation results ---")
    print(f"Runs found: {len(summaries)}")
    print(f"Saved runs CSV: {runs_csv}")
    print(f"Saved grouped CSV: {grouped_csv}")

    print("\n--- Individual runs ---")
    print(
        f"{'Session':<24} "
        f"{'Policy':<16} "
        f"{'Model/Seq':<24} "
        f"{'Success':>8} ",
        f"{'Steps':>6} "
        f"{'ProgRate':>9} "
        f"{'RepRate':>8} "
        f"{'ErrRate':>8} "
        f"{'Time(s)':>9}"
    )

    for row in summaries:
        run_type = get_run_type_label(row)
        print(
            f"{shorten(row.get('_session_id', '-'), 24):<24} "
            f"{shorten(row.get('policy_name', '-'), 16):<16} "
             f"{shorten(run_type, 24):<24} "
            f"{format_bool_or_dash(row.get('success', '-')):>8} "
            f"{str(row.get('steps_total', '-')):>6} "
            f"{format_float(row.get('progress_rate', '-')):>9} "
            f"{format_float(row.get('repeated_action_rate', '-')):>8} "
            f"{format_float(row.get('tool_error_rate', '-')):>8} "
            f"{format_float(row.get('active_time_seconds', '-')):>9}"
        )

    print("\n--- Grouped comparison ---")
    print(
        f"{'Policy':<16} "
        f"{'Model/Seq':<24} "
        f"{'Goal':<16} "
        f"{'Runs':>5} "
        f"{'Succ':>5} "
        f"{'SuccRate':>9} "
        f"{'AvgSteps':>9} "
        f"{'AvgProg':>9} "
        f"{'AvgRep':>8} "
        f"{'AvgErr':>8} "
        f"{'AvgTime':>9}"
    )

    for row in grouped_rows:
        print(
            f"{shorten(row.get('policy_name', '-'), 16):<16} "
            f"{shorten(run_type, 24):<24} "
            f"{shorten(row.get('goal_type', '-'), 16):<16} "
            f"{str(row.get('runs', '-')):>5} "
            f"{str(row.get('successes', '-')):>5} "
            f"{format_float(row.get('success_rate', '-')):>9} "
            f"{format_float(row.get('avg_steps_total', '-')):>9} "
            f"{format_float(row.get('avg_progress_rate', '-')):>9} "
            f"{format_float(row.get('avg_repeated_action_rate', '-')):>8} "
            f"{format_float(row.get('avg_tool_error_rate', '-')):>8} "
            f"{format_float(row.get('avg_active_time_seconds', '-')):>9}"
        )

def get_run_type_label(row: dict) -> str:
    policy_name = str(row.get("policy_name", "unknown")).lower()

    raw_run_type = row.get("run_type")
    if raw_run_type:
        raw_run_type = str(raw_run_type)

        # Si ya viene bien construido, lo usamos.
        if raw_run_type not in ("model", "scripted"):
            return raw_run_type

    if policy_name == "scripted":
        sequence_type = (
            row.get("sequence_type")
            or row.get("sequence_name")
            or row.get("scripted_sequence")
            or "unknown_sequence"
        )
        return f"scripted:{sequence_type}"

    if policy_name == "model":
        model_type = (
            row.get("model_type")
            or infer_model_type(row.get("model_path"))
            or "unknown_model"
        )
        return f"model:{model_type}"

    model_type = (
        row.get("model_type")
        or infer_model_type(row.get("model_path"))
    )

    if model_type:
        return f"model:{model_type}"

    return str(row.get("policy_name", "unknown"))


def infer_model_type(model_path) -> str | None:
    if not model_path:
        return None

    text = str(model_path).lower()

    if "catboost" in text:
        return "catboost"

    if "random_forest" in text or "randomforest" in text:
        return "random_forest"

    if "decision_tree" in text or "decisiontree" in text:
        return "decision_tree"

    if "xgboost" in text:
        return "xgboost"

    if "lightgbm" in text:
        return "lightgbm"

    if "mlp" in text:
        return "mlp"

    return "unknown_model"