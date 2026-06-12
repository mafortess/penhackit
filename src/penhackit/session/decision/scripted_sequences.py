"""
Predefined scripted action sequences for deterministic execution.

This module generates scripted sequences for dataset generation.

Supported matrix:

    12 attacks x 2 target types x 2 goal types = 48 sequences

Target types:
    - host
    - network

Goal types:
    - obtain_session
    - full_exploit

Important:
    - full_exploit does not include post-exploitation.
    - full_exploit means a more complete pre-exploitation path before launching the exploit.
    - The sequence stops after exploit/login because opened sessions can block later commands.
"""

from typing import Optional


# ============================================================
# Common prefixes
# ============================================================

SEQ_LOCAL_CONTEXT = [
    100,  # INSPECT_LOCAL_HOSTNAME
    101,  # INSPECT_IP_A
    102,  # INSPECT_IP_R
    103,  # INSPECT_IP_NEIGH
]


SEQ_HOST_BASE_RECON = [
    *SEQ_LOCAL_CONTEXT,
    105,  # PING_FOCUS_HOST
    210,  # SCAN_TOP_TCP_PORTS
    211,  # SCAN_FULL_TCP_PORTS
    220,  # DETECT_SERVICES
]


SEQ_NETWORK_BASE_RECON = [
    *SEQ_LOCAL_CONTEXT,
    200,  # DISCOVER_HOSTS
    210,  # SCAN_TOP_TCP_PORTS on focused host
    211,  # SCAN_FULL_TCP_PORTS on focused host
    220,  # DETECT_SERVICES
]


SEQ_HOST_RECON = [
    *SEQ_HOST_BASE_RECON,
    0,
]


SEQ_NETWORK_RECON = [
    *SEQ_NETWORK_BASE_RECON,
    0,
]

NETWORK_SCRIPTED_PASSES = 4


SEQ_FULL_EXPLOIT_BLOCK = [
    # FTP / VSFTPD
    330,
    331,
    332,
    413,
    400,
    601,

    # SMB / Samba
    320,
    322,
    323,
    324,
    410,
    400,
    600,

    # DistCC
    401,
    400,
    602,

    # PostgreSQL
    371,
    523,
    400,
    604,

    # UnrealIRCd
    401,
    400,
    605,

    # Ingreslock bind shell
    606,

    # SSH / Telnet / FTP credentials
    340,
    341,
    611,
    521,
    613,
    614,
]


SEQ_OBTAIN_SESSION_ALL_BLOCK = [
    # FTP / VSFTPD
    330,
    413,
    601,

    # SMB / Samba
    320,
    410,
    600,

    # DistCC
    400,
    602,

    # PostgreSQL
    371,
    523,
    604,

    # UnrealIRCd
    400,
    605,

    # Ingreslock bind shell
    606,

    # Credential based sessions
    611,
    521,
    613,
    614,
]


SEQ_HOST_FULL_EXPLOIT = [
    *SEQ_HOST_BASE_RECON,
    *SEQ_FULL_EXPLOIT_BLOCK,
    0,
]


SEQ_NETWORK_OBTAIN_SESSION_ALL = [
    *SEQ_LOCAL_CONTEXT,
    200,
]

for _ in range(NETWORK_SCRIPTED_PASSES):
    SEQ_NETWORK_OBTAIN_SESSION_ALL.extend([
        210,
        211,
        220,
        *SEQ_OBTAIN_SESSION_ALL_BLOCK,
    ])

SEQ_NETWORK_OBTAIN_SESSION_ALL.append(0)


SEQ_NETWORK_FULL_EXPLOIT = [
    *SEQ_LOCAL_CONTEXT,
    200,
]

for _ in range(NETWORK_SCRIPTED_PASSES):
    SEQ_NETWORK_FULL_EXPLOIT.extend([
        210,
        211,
        220,
        *SEQ_FULL_EXPLOIT_BLOCK,
    ])

SEQ_NETWORK_FULL_EXPLOIT.append(0)

# ============================================================
# Attack specs
# ============================================================

ATTACK_SPECS = {
    "vsftpd_msf": {
        "obtain_session": [330, 413, 601],
        "full_exploit": [330, 331, 332, 413, 400, 601],
    },
    "vsftpd_manual": {
        "obtain_session": [330, 413, 610],
        "full_exploit": [330, 331, 332, 413, 400, 610],
    },
    "samba_usermap_msf": {
        "obtain_session": [320, 410, 600],
        "full_exploit": [320, 322, 323, 324, 410, 400, 600],
    },
    "distcc_msf": {
        "obtain_session": [400, 602],
        "full_exploit": [401, 400, 602],
    },
    "postgres_msf": {
        "obtain_session": [371, 523, 604],
        "full_exploit": [371, 523, 400, 604],
    },
    "unreal_ircd_msf": {
        "obtain_session": [400, 605],
        "full_exploit": [401, 400, 605],
    },
    "ingreslock_bind_shell": {
        "obtain_session": [606],
        "full_exploit": [400, 606],
    },
    "ssh_weak_creds_manual": {
        "obtain_session": [520],
        "full_exploit": [340, 341, 520],
    },
    "telnet_weak_creds_manual": {
        "obtain_session": [521],
        "full_exploit": [521],
    },
    "ssh_weak_creds_msf": {
        "obtain_session": [611],
        "full_exploit": [340, 341, 611],
    },
    "ftp_weak_creds_msf": {
        "obtain_session": [330, 612],
        "full_exploit": [330, 331, 332, 612],
    },
    "ftp_weak_creds_hydra": {
        "obtain_session": [330, 613, 614],
        "full_exploit": [330, 331, 332, 613, 614],
    },
}


# ============================================================
# Sequence builder
# ============================================================

def build_sequence(target_type: str, goal_type: str, attack_name: str) -> list[int]:
    """
    Build a scripted sequence from:
        target_type + goal_type + attack_name

    Rules:
    - host + obtain_session: one selected attack path.
    - network + obtain_session: broad sequence to obtain sessions across hosts.
    - full_exploit: broad sequence, not attack-specific.
    """
    if goal_type == "full_exploit":
        if target_type == "host":
            return list(SEQ_HOST_FULL_EXPLOIT)

        if target_type == "network":
            return list(SEQ_NETWORK_FULL_EXPLOIT)

        raise ValueError(f"Invalid target_type: {target_type}")

    if goal_type == "obtain_session" and target_type == "network":
        return list(SEQ_NETWORK_OBTAIN_SESSION_ALL)

    if target_type == "host":
        seq = list(SEQ_HOST_BASE_RECON)
    elif target_type == "network":
        seq = list(SEQ_NETWORK_BASE_RECON)
    else:
        raise ValueError(f"Invalid target_type: {target_type}")

    if attack_name not in ATTACK_SPECS:
        raise ValueError(f"Invalid attack_name: {attack_name}")

    attack_spec = ATTACK_SPECS[attack_name]

    if goal_type not in attack_spec:
        raise ValueError(f"Invalid goal_type for attack {attack_name}: {goal_type}")

    seq.extend(attack_spec[goal_type])
    seq.append(0)

    return seq


# ============================================================
# Smoke-test sequence
# ============================================================

SEQ_EXPLOIT_SMOKE_TEST = [
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR
    610,  # MANUAL_EXPLOIT_VSFTPD_234_BACKDOOR
    600,  # MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT
    602,  # MSF_EXPLOIT_DISTCC_EXEC
    604,  # MSF_EXPLOIT_POSTGRES_PAYLOAD
    605,  # MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR
    606,  # CONNECT_INGRESLOCK_BIND_SHELL
    520,  # CHECK_SSH_KNOWN_CREDS
    521,  # CHECK_TELNET_KNOWN_CREDS
    611,  # MSF_SSH_LOGIN
    612,  # MSF_FTP_LOGIN
    613,  # HYDRA_FTP_LOGIN
    614,  # CHECK_FTP_KNOWN_CREDS_MANUAL
    0,
]


# ============================================================
# Registry
# ============================================================

SCRIPTED_SEQUENCES = {
    "host_recon": SEQ_HOST_RECON,
    "network_recon": SEQ_NETWORK_RECON,
    "exploit_smoke_test": SEQ_EXPLOIT_SMOKE_TEST,
    
    "host_full_exploit": SEQ_HOST_FULL_EXPLOIT,
    "network_obtain_session_all": SEQ_NETWORK_OBTAIN_SESSION_ALL,
    "network_full_exploit": SEQ_NETWORK_FULL_EXPLOIT,
}


for attack_name in ATTACK_SPECS:
    SCRIPTED_SEQUENCES[f"host_obtain_session_{attack_name}"] = build_sequence(
        target_type="host",
        goal_type="obtain_session",
        attack_name=attack_name,
    )

    SCRIPTED_SEQUENCES[f"network_obtain_session_{attack_name}"] = build_sequence(
        target_type="network",
        goal_type="obtain_session",
        attack_name=attack_name,
    )

    SCRIPTED_SEQUENCES[f"host_full_exploit_{attack_name}"] = build_sequence(
        target_type="host",
        goal_type="full_exploit",
        attack_name=attack_name,
    )

    SCRIPTED_SEQUENCES[f"network_full_exploit_{attack_name}"] = build_sequence(
        target_type="network",
        goal_type="full_exploit",
        attack_name=attack_name,
    )


# ============================================================
# Backward-compatible aliases
# ============================================================

SCRIPTED_SEQUENCES["attack_vsftpd_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_vsftpd_msf"]
SCRIPTED_SEQUENCES["attack_vsftpd_manual"] = SCRIPTED_SEQUENCES["host_obtain_session_vsftpd_manual"]
SCRIPTED_SEQUENCES["attack_samba_usermap_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_samba_usermap_msf"]
SCRIPTED_SEQUENCES["attack_distcc_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_distcc_msf"]
SCRIPTED_SEQUENCES["attack_postgres_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_postgres_msf"]
SCRIPTED_SEQUENCES["attack_unreal_ircd_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_unreal_ircd_msf"]
SCRIPTED_SEQUENCES["attack_ingreslock_bind_shell"] = SCRIPTED_SEQUENCES["host_obtain_session_ingreslock_bind_shell"]
SCRIPTED_SEQUENCES["attack_ssh_weak_creds_manual"] = SCRIPTED_SEQUENCES["host_obtain_session_ssh_weak_creds_manual"]
SCRIPTED_SEQUENCES["attack_telnet_weak_creds_manual"] = SCRIPTED_SEQUENCES["host_obtain_session_telnet_weak_creds_manual"]
SCRIPTED_SEQUENCES["attack_ssh_weak_creds_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_ssh_weak_creds_msf"]
SCRIPTED_SEQUENCES["attack_ftp_weak_creds_msf"] = SCRIPTED_SEQUENCES["host_obtain_session_ftp_weak_creds_msf"]
SCRIPTED_SEQUENCES["attack_ftp_weak_creds_hydra"] = SCRIPTED_SEQUENCES["host_obtain_session_ftp_weak_creds_hydra"]

SCRIPTED_SEQUENCES["network_attack_vsftpd_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_vsftpd_msf"]
SCRIPTED_SEQUENCES["network_attack_vsftpd_manual"] = SCRIPTED_SEQUENCES["network_obtain_session_vsftpd_manual"]
SCRIPTED_SEQUENCES["network_attack_samba_usermap_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_samba_usermap_msf"]
SCRIPTED_SEQUENCES["network_attack_distcc_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_distcc_msf"]
SCRIPTED_SEQUENCES["network_attack_postgres_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_postgres_msf"]
SCRIPTED_SEQUENCES["network_attack_unreal_ircd_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_unreal_ircd_msf"]
SCRIPTED_SEQUENCES["network_attack_ingreslock_bind_shell"] = SCRIPTED_SEQUENCES["network_obtain_session_ingreslock_bind_shell"]
SCRIPTED_SEQUENCES["network_attack_ssh_weak_creds_manual"] = SCRIPTED_SEQUENCES["network_obtain_session_ssh_weak_creds_manual"]
SCRIPTED_SEQUENCES["network_attack_telnet_weak_creds_manual"] = SCRIPTED_SEQUENCES["network_obtain_session_telnet_weak_creds_manual"]
SCRIPTED_SEQUENCES["network_attack_ssh_weak_creds_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_ssh_weak_creds_msf"]
SCRIPTED_SEQUENCES["network_attack_ftp_weak_creds_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_ftp_weak_creds_msf"]
SCRIPTED_SEQUENCES["network_attack_ftp_weak_creds_hydra"] = SCRIPTED_SEQUENCES["network_obtain_session_ftp_weak_creds_hydra"]

SCRIPTED_SEQUENCES["vsftpd"] = SCRIPTED_SEQUENCES["network_obtain_session_vsftpd_msf"]
SCRIPTED_SEQUENCES["vsftpd_manual"] = SCRIPTED_SEQUENCES["network_obtain_session_vsftpd_manual"]
SCRIPTED_SEQUENCES["samba"] = SCRIPTED_SEQUENCES["network_obtain_session_samba_usermap_msf"]
SCRIPTED_SEQUENCES["distcc"] = SCRIPTED_SEQUENCES["network_obtain_session_distcc_msf"]
SCRIPTED_SEQUENCES["postgres"] = SCRIPTED_SEQUENCES["network_obtain_session_postgres_msf"]
SCRIPTED_SEQUENCES["unreal_ircd"] = SCRIPTED_SEQUENCES["network_obtain_session_unreal_ircd_msf"]
SCRIPTED_SEQUENCES["ingreslock"] = SCRIPTED_SEQUENCES["network_obtain_session_ingreslock_bind_shell"]
SCRIPTED_SEQUENCES["ssh"] = SCRIPTED_SEQUENCES["network_obtain_session_ssh_weak_creds_manual"]
SCRIPTED_SEQUENCES["telnet"] = SCRIPTED_SEQUENCES["network_obtain_session_telnet_weak_creds_manual"]
SCRIPTED_SEQUENCES["ssh_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_ssh_weak_creds_msf"]
SCRIPTED_SEQUENCES["ftp_msf"] = SCRIPTED_SEQUENCES["network_obtain_session_ftp_weak_creds_msf"]
SCRIPTED_SEQUENCES["ftp_hydra"] = SCRIPTED_SEQUENCES["network_obtain_session_ftp_weak_creds_hydra"]

SCRIPTED_SEQUENCES["standard"] = SCRIPTED_SEQUENCES["host_obtain_session_vsftpd_msf"]


# ============================================================
# Defaults
# ============================================================

DEFAULT_SCRIPTED_SEQUENCE_NAME = "host_obtain_session_vsftpd_msf"
DEFAULT_SCRIPTED_SEQUENCE = SCRIPTED_SEQUENCES[DEFAULT_SCRIPTED_SEQUENCE_NAME]


# ============================================================
# Public helpers
# ============================================================

def get_scripted_sequence(name: Optional[str] = None) -> list[int]:
    """
    Return a predefined scripted action sequence.

    If name is None or unknown, return the default sequence.
    """
    if not name:
        return DEFAULT_SCRIPTED_SEQUENCE

    return SCRIPTED_SEQUENCES.get(name, DEFAULT_SCRIPTED_SEQUENCE)


def list_scripted_sequences() -> list[str]:
    """
    Return the available scripted sequence names.
    """
    return sorted(SCRIPTED_SEQUENCES.keys())


def get_default_scripted_sequence_name() -> str:
    """
    Return the default scripted sequence name.
    """
    return DEFAULT_SCRIPTED_SEQUENCE_NAME


def list_attack_names() -> list[str]:
    """
    Return the available attack names.
    """
    return sorted(ATTACK_SPECS.keys())