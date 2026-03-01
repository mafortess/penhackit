from penhackit import settings
from penhackit.session.session_wizard import new_session_wizard
from penhackit.session.session_logic import new_session_logic
def run_session_service(app_context: dict) -> None:
    print("Starting new session...")
    
    # Load default settings and paths
    session_settings = app_context["settings"]["session"]
    env_settings = app_context["enviroment_profile"]
    paths = app_context["paths"]

    # default_name = session_settings["default_name"]
    # default_goal_type = session_settings["default_goal_type"]
    # default_target = session_settings["default_target"]
    # default_max_steps = session_settings["default_max_steps"]
    # launch_kb_monitor = session_settings["launch_kb_monitor"]

    # Wizard for new session creation
    wizard_data = new_session_wizard(session_settings)
    if wizard_data is None:
        print("Session creation cancelled.")
        return
    
    session_settings = {
        "name": wizard_data["name"] if "name" in wizard_data else session_settings["default_name"],
        "mode": wizard_data["mode"] if "mode" in wizard_data else session_settings["default_mode"],
        "decider": wizard_data["decider"] if "decider" in wizard_data else session_settings["default_decider"],
        "goal_type": wizard_data["goal_type"] if "goal_type" in wizard_data else session_settings["default_goal_type"],
        "target": wizard_data["target"] if "target" in wizard_data else session_settings["default_target"],
        "max_steps": wizard_data["max_steps"] if "max_steps" in wizard_data else session_settings["default_max_steps"],
        "launch_kb_monitor": wizard_data["launch_kb_monitor"] if "launch_kb_monitor" in wizard_data else session_settings["default_launch_kb_monitor"]
    }
    
    try:
        print("Running session...")
        new_session_logic(
            session_settings=session_settings,
            env_settings=env_settings,
            paths=paths)
        
    except Exception as e:
        print(f"Error during session execution: {e}")

    # print(f"default_name: {default_name}")
    # print(f"default_goal_type: {default_goal_type}")
    # print(f"default_target: {default_target}")
    # print(f"default_max_steps: {default_max_steps}")


def list_sessions() -> None:
    print("Listing sessions...")
    # Aquí iría la lógica para listar las sesiones existentes, mostrando información relevante.

def show_session_details() -> None:
    print("Showing session details...")
    # Aquí iría la lógica para mostrar los detalles de una sesión específica, probablemente pidiendo un ID o nombre.

def delete_session() -> None:
    print("Deleting session...")
    # Aquí iría la lógica para eliminar una sesión, probablemente pidiendo un ID o nombre y confirmando la acción。

