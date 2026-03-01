from penhackit.common.paths import Paths

from prompt_toolkit import prompt # input mejorada (historial, autocompletado, multilinea, etc)
from prompt_toolkit.completion import WordCompleter # autcompletado para menus y opciones

from penhackit.report.report_storage import list_reportable_sessions, list_local_ollama_models, list_local_hf_transformers_models

import torch

def generate_report_wizard(report_settings: dict, paths: Paths) -> dict | None:
    print("Starting report generation wizard...")

    sessions_dir = paths.sessions_dir
    print(f"Looking for sessions in: {sessions_dir}")
    
    # SELECT SESSION
    # Show available sessions to select one for the report
    sessions = list_reportable_sessions(sessions_dir)
    session_id = wizard_select_session(sessions)
    if session_id is None:
        return None
    
    session_dir = sessions_dir / session_id
    print(f"Selected session: {session_id} ({session_dir})")

    # SELECT BACKEND
    backend = wizard_select_backend(report_settings)
    if backend is None:
        return None

    # SELECT MODEL (if applicable) AND DEVICE (for transformers)
    model_name = None
    device = None

    if backend == "ollama":
        ollama_models = list_local_ollama_models()
        model_name = choose_ollama_model_interactive(ollama_models, default_model =report_settings["default_ollama_model"])
        if model_name is None:
            return None

    elif backend == "transformers":
        llm_models = list_local_hf_transformers_models(paths.llm_models_dir)
        model_name = choose_hf_model_dir_interactive(llm_models, default_name=report_settings["default_transformers_model"])
        if model_name is None:
            return None
    
        device = choose_hf_device_interactive()
        if device is None:
            return None
        
    # PDF GENERATION
    output_format = choose_output_format(report_settings)
    # pdf_generation = wizard_pdf_generation()

    confirmed = wizard_confirm(
        session_id=session_id,
        backend=backend,
        model_name=model_name,
        output_format=output_format,
        device=device,
    )
    if not confirmed:
        return None

    return {
        "session_id": session_id,
        "output_format": output_format,
        "backend": backend,
        "ollama_model_name": model_name if backend == "ollama" else None,
        "transformers_model_name": model_name if backend == "transformers" else None,
        "device": device if backend == "transformers" else None,
    }


def wizard_select_session(sessions: list[str]) -> str | None:
    print("\n--- Generate report / Select session ---")
    
    if not sessions:
        print("No sessions available.")
        return None
    
    print("Available sessions:")
    for idx, session_id in enumerate(sessions, start=1):
        print(f"{idx}) {session_id}")
    print("0) Cancel")

    options = [ str(idx) for idx in range(1, len(sessions) + 1) ] + ["0"]
    completer = WordCompleter(options, ignore_case=True, match_middle=True)
    
    while True:
        session_choice = prompt("Choose a session> ", completer=completer).strip()

        print(f"User input: {session_choice}")  # Debug print

        if session_choice == "0":
            return None

        if session_choice.isdigit():
            index = int(session_choice)
            if 1 <= index <= len(sessions):
                return sessions[index - 1]

        print("Invalid option.")

def wizard_select_backend(report_settings: dict) -> str | None:
    default_backend = report_settings["default_backend"]

    print("\n--- Generate report / Select backend ---")
    print(f"Default backend: {default_backend}")
    print("1) Use default")
    print("1) Baseline (no LLM)")
    print("2) Ollama (HTTP local)")
    print("4) Transformers (local, HF models)")
    print("0) Cancel")

    options = ["1", "2", "3", "4", "0"]
    completer = WordCompleter(options, ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()

        if raw == "0":
            return None
        if raw == "1":
            return default_backend
        if raw == "2":
            return "baseline"
        if raw == "3":
            return "ollama"
        if raw == "4":
            return "transformers"

        print("Invalid option.")

def wizard_pdf_generation() -> bool:
    print("\n--- Generate report / PDF generation ---")
    print("Generate PDF report? (y/n)")

    options = ["yes", "no"]
    completer = WordCompleter(options, ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Invalid option.")

def wizard_confirm(session_id: str, backend: str, model_name: str | None, output_format: str | None, device: str | None) -> bool:
    print("\n--- Generate report / Confirm ---")
    print(f"Session: {session_id}")
    print(f"Backend: {backend}")
    print(f"Model: {model_name if model_name is not None else '-'}")
    print(f"Output format: {output_format if output_format is not None else '-'}")
    print(f"Device: {device if device is not None else '-'}")

    while True:
        options = ["yes", "no"]
        completer = WordCompleter(options, ignore_case=True)
        raw = prompt("Confirm? (y/n): ", completer=completer).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Invalid option.")


# FROM MVP

def choose_ollama_model_interactive(models: list[str], default_model: str | None = None) -> str | None:
    """
    Interactive chooser:
    - fetches local models via HTTP (preferred), CLI fallback
    - provides:
      * numbered list selection
      * optional text filter
      * best-effort TAB autocomplete (if readline is available)
    Returns selected model name or None if cancelled.
    """
    if not models:
        print("No Ollama models found (is Ollama running, and are models installed?).")
        print("Try: ollama list  |  ollama pull <model>")
        return None

    # best-effort TAB completion (works on Linux/macOS; on Windows may require pyreadline3)
    try:
        import readline  # type: ignore

        def _completer(text, state):
            matches = [m for m in models if m.startswith(text)]
            return matches[state] if state < len(matches) else None

        readline.set_completer(_completer)
        readline.parse_and_bind("tab: complete")
    except Exception:
        pass

    print("\nAvailable Ollama models:")
    for i, m in enumerate(models, start=1):
        tag = " (default)" if default_model and m == default_model else ""
        print(f"{i}) {m}{tag}")

    while True:
        raw = input("Select model [number | name | /filter | 0 cancel]> ").strip()
        if raw == "0":
            return None

        # filter mode: /text
        if raw.startswith("/"):
            q = raw[1:].strip().lower()
            if not q:
                continue
            filtered = [m for m in models if q in m.lower()]
            if not filtered:
                print("No matches.")
                continue
            print("\nMatches:")
            for i, m in enumerate(filtered, start=1):
                print(f"{i}) {m}")
            raw2 = input("Select from matches (number | name | 0 back)> ").strip()
            if raw2 == "0":
                continue
            if raw2.isdigit():
                idx = int(raw2)
                if 1 <= idx <= len(filtered):
                    return filtered[idx - 1]
                print("Out of range.")
                continue
            if raw2 in filtered:
                return raw2
            print("Invalid selection.")
            continue

        # numeric selection
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print("Out of range.")
            continue

        # exact name selection (autocomplete helps here)
        if raw in models:
            return raw

        # small convenience: if user typed prefix and it's unique
        pref = [m for m in models if m.startswith(raw)]
        if len(pref) == 1:
            return pref[0]

        print("Invalid selection. Tip: type / to filter, or TAB to autocomplete (if supported).")


def choose_hf_model_dir_interactive(models: list[str], default_name: str | None = None) -> str | None:
    """
    Interactive chooser for local HF model directories.
    - models: list of model directory names (not full paths)
    - default_name: if provided, marks the default model in the list
    Returns selected model name or None if cancelled.
    """
    if not models:
        print("No HF models found.")
        print("Expected folders like: mvp/llm_models/<model_id>/config.json")
        return None

    print("\nAvailable HF models (local):")
    for i, m in enumerate(models, start=1):
        tag = " (default)" if default_name and m == default_name else ""
        print(f"{i}) {m}{tag}")

    options = [ str(i) for i in range(1, len(models) + 1) ] + ["0"]
    completer = WordCompleter(options, ignore_case=True, match_middle=True)
    raw = prompt("Select model [num | name | 0 cancel]> ", completer=completer).strip()
    if raw == "0":
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(models):
            return models[idx - 1]
        print("Out of range.")
        return None

    # name selection
    for m in models:
        if m == raw:
            return m

    # prefix unique
    pref = [p for p in models if p.startswith(raw)]
    if len(pref) == 1:
        return pref[0]

    print("Invalid selection.")
    return None

def choose_hf_device_interactive() -> str:
    # keep it simple for MVP
    # - "cuda" if available
    # - else "cpu"
    default = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice (default: {default})")
    print("1) auto")
    print("2) cuda")
    print("3) cpu")
    options = ["1", "2", "3"]
    completer = WordCompleter(options, ignore_case=True)    
    raw = prompt("Select device> ", completer=completer).strip()
    if raw == "2":
        return "cuda"
    if raw == "3":
        return "cpu"
    return default

def choose_output_format(report_settings: dict) -> str | None:
    default_output_format = report_settings["default_output_format"]

    print("\n--- Select output format ---")
    print(f"Default output format: {default_output_format}")
    print("1) Use default")
    print("2) md")
    print("3) md+pdf")
    print("0) Cancel")

    completer = WordCompleter(["0", "1", "2", "3"], ignore_case=True)

    while True:
        raw = prompt("> ", completer=completer).strip()

        if raw == "0":
            return None
        if raw == "1":
            return default_output_format
        if raw == "2":
            return "md"
        if raw == "3":
            return "md+pdf"

        print("Invalid option.")
