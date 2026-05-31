from penhackit import settings
from penhackit.session.session_wizard import new_session_wizard
from penhackit.session.session_builder import create_session_runtime 
from penhackit.session.session_runner import run_session_loop
from penhackit.session.session_views import run_view_sessions_view


def run_session_service(app_context: dict) -> None:
    """Servicio para ejecutar una nueva sesión de pentesting en modo autónomo, observación o sugerencia."""
    
    print("New session service...")
    
    # Load default settings and paths
    default_session_settings = app_context["settings"]["session"]
    env_profile = app_context["enviroment_profile"]
    paths = app_context["paths"]

    # default_name = session_settings["default_name"]# default_goal_type = session_settings["default_goal_type"]
    # default_target = session_settings["default_target"]  # default_max_steps = session_settings["default_max_steps"]
    # launch_kb_monitor = session_settings["launch_kb_monitor"]

    # Wizard for new session creation
    wizard_data = new_session_wizard(default_session_settings, paths)
    if wizard_data is None:
        print("Session creation cancelled.")
        print("===========================\n")
        return
    
    session_settings = build_session_settings(default_session_settings, wizard_data)

    # session_settings = {    #     "name": wizard_data["name"] if "name" in wizard_data else session_settings["default_name"],#     "mode": wizard_data["mode"] if "mode" in wizard_data else session_settings["default_mode"],
    #     "decider": wizard_data["decider"] if "decider" in wizard_data else session_settings["default_decider"],    #     "goal_type": wizard_data["goal_type"] if "goal_type" in wizard_data else session_settings["default_goal_type"],
    #     "target": wizard_data["target"] if "target" in wizard_data else session_settings["default_target"],    #     "max_steps": wizard_data["max_steps"] if "max_steps" in wizard_data else session_settings["default_max_steps"],
    #     "launch_kb_monitor": wizard_data["launch_kb_monitor"] if "launch_kb_monitor" in wizard_data else session_settings["default_launch_kb_monitor"]    # }
    
    try:
        print("Creating session...")
          
        session_info = create_session_runtime(
            session_settings=session_settings,
            env_profile=env_profile,
            paths=paths,
        )

        print("Running session loop...")
        run_session_loop(
            session_settings=session_settings,
            session_info=session_info,
            paths=paths
        )
        print("Session finished service.")
        
    except Exception as e:
        print(f"Error during session execution: {e}")


def build_session_settings(default_session_settings: dict, wizard_data: dict) -> dict:

    return {
        "name": wizard_data["name"] if "name" in wizard_data else default_session_settings["default_name"],
        "mode": wizard_data["mode"] if "mode" in wizard_data else default_session_settings["default_mode"],
        "goal_type": wizard_data["goal_type"] if "goal_type" in wizard_data else default_session_settings["default_goal_type"],
        "target_type": wizard_data["target_type"] if "target_type" in wizard_data else default_session_settings["default_target_type"],
        "target": wizard_data["target"] if "target" in wizard_data else default_session_settings["default_target"],
        "max_steps": wizard_data["max_steps"] if "max_steps" in wizard_data else default_session_settings["default_max_steps"],
        "decider": wizard_data["decider"] if "decider" in wizard_data else default_session_settings["default_decider"],
        "model_id": wizard_data["model_id"] if "model_id" in wizard_data else default_session_settings["default_model_id"],
        "scripted_sequence": default_session_settings["default_scripted_sequence"],
        "launch_kb_monitor": wizard_data["launch_kb_monitor"] if "launch_kb_monitor" in wizard_data else default_session_settings["default_launch_kb_monitor"]
    }

def run_view_session_service(app_context: dict) -> None:
    print("View sessions service...")
    run_view_sessions_view(app_context["paths"])