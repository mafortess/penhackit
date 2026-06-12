from penhackit.session.action.control import CONTROL_ACTIONS
from penhackit.session.action.local_context import LOCAL_CONTEXT_ACTIONS
from penhackit.session.action.recon import RECON_ACTIONS
from penhackit.session.action.enumeration import ENUMERATION_ACTIONS
from penhackit.session.action.vulnerability import VULNERABILITY_ACTIONS
from penhackit.session.action.credentials import CREDENTIALS_ACTIONS
from penhackit.session.action.exploitation import EXPLOIT_ACTIONS
from penhackit.session.action.post_exploitation import POST_EXPLOIT_ACTIONS

# 000-099  Control
# 100-199  Local attacker context
# 200-299  Recon / discovery / scan
# 300-399  Enumeration
# 400-499  Vulnerability discovery / validation
# 500-599  Credential attacks
# 600-699  Exploitation
# 700-799  Post-exploitation

ACTIONS = {
    **CONTROL_ACTIONS,
    **LOCAL_CONTEXT_ACTIONS,
    **RECON_ACTIONS,
    **ENUMERATION_ACTIONS,
    **VULNERABILITY_ACTIONS,
    **CREDENTIALS_ACTIONS,
    **EXPLOIT_ACTIONS,  
    **POST_EXPLOIT_ACTIONS,
}


def get_action(action_id: int) -> dict:
    return ACTIONS.get(action_id, ACTIONS[0])


def get_action_name(action_id: int) -> str:
    return get_action(action_id)["name"]


def get_command_template(action_id: int) -> str | None:
    return get_action(action_id).get("command_template")


def get_parser_family(action_id: int) -> str | None:
    return get_action(action_id).get("parser_family")

def get_expected_events(action_id: int) -> list[str]:
    return get_action(action_id).get("expected_events", [])


def get_placeholders(action_id: int) -> list[str]:
    return get_action(action_id).get("placeholders", [])


def get_preconditions(action_id: int) -> dict:
    return get_action(action_id).get("preconditions", {})