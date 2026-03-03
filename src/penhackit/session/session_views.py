from penhackit.session.session_storage import list_sessions, delete_session, load_session_details

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
        print("3) Delete session")
        print("0) Back")

        choice = prompt("Select option> ", completer=WordCompleter(["1", "2","3", "0"])).strip()

        if choice == "1":
            show_session_summary_view(paths, session_id)            
        elif choice == "2": 
            show_session_details_view(paths, session_id)
        elif choice == "3":
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

    summary = build_session_summary_data(session_id, kb, paths.sessions_dir)

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
    

def build_session_summary_data(session_id: str, kb: dict, sessions_dir: Path) -> dict:
    session_context = kb.get("session_context", {}) or {}
    reports_available = has_session_reports(sessions_dir, session_id)

    return {
        "session_id": session_id,
        "name": session_context.get("name", session_id),
        "mode": session_context.get("mode", "unknown"),
        "goal_type": session_context.get("goal_type", "unknown"),
        "target": session_context.get("target", "unknown"),
        # "status": infer_session_status(kb),
        # "steps": infer_step_count(kb),
        "hosts": len(kb.get("hosts", []) or []),
        "services": len(kb.get("services", []) or []),
        "findings": len(kb.get("findings", []) or []),
        "commands": len(kb.get("commands", []) or []),
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