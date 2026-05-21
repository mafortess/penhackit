from typing import Any
import numpy as np

GOAL_TYPE_TO_ID = {
    None: 0,
    "unknown": 0,
    "recon": 1,
    "enumeration": 2,
    "vulnerability_discovery": 3,
    "exploitation": 4,
    "obtain_session": 5,
}


FOCUS_LEVEL_TO_ID = {
    None: 0,
    "unknown": 0,
    "global": 1,
    "network": 2,
    "subnet": 3,
    "host": 4,
    "service": 5,
    "vuln": 6,
    "session": 7,
}


EVENT_TYPE_TO_ID = {
    None: 0,
    "unknown": 0,
    "NO_EVENT": 1,
    "ACTION_COMPLETED": 2,
    "TOOL_ERROR": 3,
    "TIMEOUT": 4,
    "NO_MEANINGFUL_OUTPUT": 5,

    "NET_INFO": 10,
    "ARP_TABLE": 11,
    "ROUTE_TABLE": 12,

    "HOST_DISCOVERED": 20,
    "HOST_UNREACHABLE": 21,
    "SUBNET_SCAN_COMPLETED": 22,

    "PORT_OPEN": 30,
    "PORT_CLOSED": 31,
    "SERVICE_DETECTED": 32,
    "SERVICE_VERSION_DETECTED": 33,
    "OS_GUESS_DETECTED": 34,

    "WEB_PATH_FOUND": 40,
    "WEB_TECH_DETECTED": 41,
    "HTTP_HEADER_DETECTED": 42,
    "LOGIN_FORM_DETECTED": 43,

    "SMB_SHARE_FOUND": 50,
    "SMB_USER_FOUND": 51,
    "SMB_GROUP_FOUND": 52,
    "SMB_GUEST_ACCESS_ALLOWED": 53,

    "CANDIDATE_VULN_FOUND": 60,
    "VULN_VALIDATED": 61,
    "VULN_REJECTED": 62,

    "VALID_CREDENTIAL_FOUND": 70,
    "LOGIN_SUCCESS": 71,
    "LOGIN_FAILED": 72,

    "EXPLOIT_ATTEMPTED": 80,
    "SESSION_OPENED": 81,
}


ACTION_NAME_TO_ID = {
    None: 0,
    "unknown": 0,
    "NO_OP": 1,
    "STOP": 2,

    "INSPECT_ARP": 10,
    "INSPECT_IP_A": 11,
    "INSPECT_IP_ROUTE": 12,

    "DISCOVER_HOSTS_PING_SWEEP": 100,
    "PORTSCAN_TOP_TCP": 110,
    "PORTSCAN_FULL_TCP": 111,
    "SERVICE_DETECT": 120,
    "OS_FINGERPRINT": 130,

    "ENUM_HTTP_HEADERS": 200,
    "ENUM_HTTP_TECH": 201,
    "ENUM_HTTP_DIR": 202,
    "ENUM_HTTP_NIKTO": 203,
    "ENUM_SMB_BASIC": 210,
    "ENUM_SMB_SHARES": 211,
    "ENUM_FTP_BASIC": 220,
    "ENUM_SSH_BASIC": 230,

    "CHECK_SERVICE_VERSION_VULNS": 300,
    "CHECK_HTTP_COMMON_VULNS": 310,
    "CHECK_SMB_COMMON_VULNS": 320,
    "VALIDATE_CANDIDATE_VULN": 330,

    "TRY_DEFAULT_CREDS": 400,
    "TRY_HTTP_LOGIN": 410,
    "TRY_SMB_LOGIN": 420,
    "TRY_FTP_LOGIN": 430,
    "TRY_METASPLOIT_MODULE": 440,
    "VERIFY_SHELL": 450,
}


STATE_FEATURE_NAMES = [
    "goal_type",
    "focus_level",
    "has_focus_host",
    "has_focus_service",
    "net_ipv4_count",
    "net_gw_count",
    "net_if_count",
    "net_arp_count",
    "net_routes_count",
    "hosts_count",
    "services_count",
    "findings_count",
    "last_action_id",
    "last_action_name",
    "last_rc",
    "last_event_type",
    "step_idx",
]


def encode_categorical(value: Any, mapping: dict) -> int:
    if value in mapping:
        return mapping[value]

    return mapping.get("unknown", 0)


def encode_bool(value: Any) -> int:
    if value is None:
        return 0

    return 1 if bool(value) else 0


def encode_number(value: Any, default: int | float = 0) -> int | float:
    if value is None:
        return default

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (int, float)):
        return value

    raise ValueError(f"Expected numeric value, got {type(value).__name__}: {value!r}")


def vectorize_state(state: dict) -> list[int | float]:
    return [
        encode_categorical(state.get("goal_type"), GOAL_TYPE_TO_ID),
        encode_categorical(state.get("focus_level"), FOCUS_LEVEL_TO_ID),
        encode_bool(state.get("has_focus_host")),
        encode_bool(state.get("has_focus_service")),

        encode_number(state.get("net_ipv4_count")),
        encode_number(state.get("net_gw_count")),
        encode_number(state.get("net_if_count")),
        encode_number(state.get("net_arp_count")),
        encode_number(state.get("net_routes_count")),
        encode_number(state.get("hosts_count")),
        encode_number(state.get("services_count")),
        encode_number(state.get("findings_count")),

        encode_number(state.get("last_action_id"), default=0),
        encode_categorical(state.get("last_action_name"), ACTION_NAME_TO_ID),
        encode_number(state.get("last_rc"), default=-1),
        encode_categorical(state.get("last_event_type"), EVENT_TYPE_TO_ID),

        encode_number(state.get("step_idx")),
    ]


def vectorize_bc_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.zeros((len(rows), len(STATE_FEATURE_NAMES)), dtype=np.float32)
    y = np.zeros((len(rows),), dtype=np.int64)

    for i, row in enumerate(rows):
        state = row.get("state")
        action_id = row.get("action_id")

        if state is None:
            raise ValueError(f"Dataset row without state: {row}")

        if action_id is None:
            raise ValueError(f"Dataset row without action_id: {row}")

        X[i, :] = np.array(vectorize_state(state), dtype=np.float32)
        y[i] = int(action_id)

    return X, y, STATE_FEATURE_NAMES