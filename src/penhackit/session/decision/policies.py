import numpy as np
from penhackit.training.vectorization import vectorize_bc_rows, vectorize_state
from penhackit.session.decision.scripted_sequences import get_scripted_sequence

def scripted_policy_decide_action(state: dict, t: int, sequence_name: str) -> int:
    """
    Deterministic scripted policy.

    It follows a predefined action_id sequence. This is useful for:
    - end-to-end pipeline validation
    - controlled lab execution
    - dataset generation
    - baseline comparison against rules/model policies
    """

    print("Deciding action based on state...")
    # Política muy simple y determinista basada en el estado actual (ejemplo MVP)
    # action = t+1

    print("Deciding action based on {t} step of scripted sequence '{sequence_name}'...".format(t=t, sequence_name=sequence_name))

    sequence = get_scripted_sequence(sequence_name)
   
    # Secuencia predefinida de acciones para probar el pipeline end-to-end
    if t < len(sequence):
        action = sequence[t]
        return action
    
    return 0  # STOP

def rules_policy_decide_action(state: dict) -> int:
    """
    Política basada en reglas heurísticas mínimas y deterministas (Kali)
    """
    if int(state.get("hosts_count", 0) or 0) == 0:
        return 200  # DISCOVER_HOSTS

    if int(state.get("open_ports_count", 0) or 0) == 0:
        return 210  # SCAN_TOP_TCP_PORTS

    if int(state.get("services_count", 0) or 0) == 0:
        return 220  # DETECT_SERVICES

    if bool(state.get("has_http", False)) and not bool(state.get("http_headers_done", False)):
        return 300  # ENUM_HTTP_HEADERS

    if bool(state.get("has_http", False)) and not bool(state.get("http_dirs_done", False)):
        return 310  # ENUM_HTTP_DIRS

    if bool(state.get("has_smb", False)) and not bool(state.get("smb_enum_done", False)):
        return 320  # ENUM_SMB_SHARES

    if int(state.get("candidate_vulns_count", 0) or 0) == 0:
        return 400  # CHECK_SERVICE_VERSION_VULNS

    return 0  # STOP

    # Reglas mínimas y deterministas para tu MVP (Windows-centric)
    # 1) Si aún no tenemos IPv4, primero ipconfig /all
    if int(state.get("net_ipv4_count", 0) or 0) == 0:
        return 1  # INSPECT_IPCONFIG

    # 2) Si aún no tenemos vecinos ARP, pedir arp -a
    if int(state.get("net_arp_count", 0) or 0) == 0:
        return 2  # INSPECT_ARP

    # 3) Si ya hay hosts, prueba ping al foco/primer host
    if int(state.get("hosts_count", 0) or 0) > 0:
        return 4  # PING_FOCUS_HOST

    # 4) Fallback: nada que hacer
    return 0  # NONE


def model_policy_decide_action(state: dict, model, feature_names: list[str]) -> int:
    """
    model: cualquier sklearn classifier ya entrenado (joblib.load(...))
    feature_names: lista de features en el orden usado al entrenar (metrics.json["feature_names"])
    """
    x = [vectorize_state(state)]

    y_pred = model.predict(x)
    return int(y_pred[0])
