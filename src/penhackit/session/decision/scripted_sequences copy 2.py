"""
Predefined scripted action sequences for deterministic execution.

These sequences are useful for:
- end-to-end pipeline testing
- dataset generation
- observation-mode baseline sessions
- comparing scripted/rules/model policies

The model should eventually learn:
    state_t -> action_id

The scripted policy simply replays:
    t -> action_id

Sequence categories:
- Smoke tests: execute exploits directly. Useful for debugging actions, not for training.
- Recon sequences: build KB progressively.
- Dataset sequences: context -> recon -> enum/check -> attack -> optional post -> stop.
- Legacy sequences: kept for compatibility, not part of the final 12 supported attacks.
"""

from typing import Optional 

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
    0,    # STOP
]

# ============================================================
# Common recon prefixes
# ============================================================

SEQ_LOCAL_CONTEXT = [
    100,  # INSPECT_LOCAL_HOSTNAME
    101,  # INSPECT_IP_A
    102,  # INSPECT_IP_R
    103,  # INSPECT_IP_NEIGH
]

# Prefijo común si el target es una red
SEQ_HOST_BASE_RECON = [
    *SEQ_LOCAL_CONTEXT,
    105,  # PING_FOCUS_HOST
    210,  # SCAN_TOP_TCP_PORTS
    211,  # SCAN_FULL_TCP_PORTS
    220,  # DETECT_SERVICES
]

# Prefijo común si el target es un host
SEQ_NETWORK_BASE_RECON = [
    *SEQ_LOCAL_CONTEXT,
    200,  # DISCOVER_HOSTS
    210,  # SCAN_TOP_TCP_PORTS on focused host
    211,  # SCAN_FULL_TCP_PORTS on focused host
    220,  # DETECT_SERVICES
]

# Prefijo común si el target es un host
SEQ_COMMON_HOST_RECON = [
    *SEQ_HOST_BASE_RECON,
    401,  # CHECK_NMAP_VULN_SCRIPTS
    0,    # STOP
]

SEQ_COMMON_NETWORK_RECON = [
    *SEQ_NETWORK_BASE_RECON,
    401,  # CHECK_NMAP_VULN_SCRIPTS
    0,    # STOP
]

# ============================================================
# Common post-exploitation tail
# ============================================================

SEQ_BASIC_POST_EXPLOIT = [
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    703,  # POST_ENUM_HOSTNAME
    701,  # POST_ENUM_UNAME
    704,  # POST_ENUM_IP_ADDR
]


# ============================================================
# Attack tails
# ============================================================

# 1. VSFTPD 2.3.4 backdoor - Metasploit - 601
TAIL_ATTACK_VSFTPD_MSF = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    413,  # CHECK_FTP_VULNS
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR
    0,    # STOP
]


# 2. VSFTPD 2.3.4 backdoor - Manual - 610
TAIL_ATTACK_VSFTPD_MANUAL = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    413,  # CHECK_FTP_VULNS
    610,  # MANUAL_EXPLOIT_VSFTPD_234_BACKDOOR
    0,    # STOP
]


# 3. Samba usermap_script - Metasploit - 600
TAIL_ATTACK_SAMBA_USERMAP_MSF = [
    320,  # ENUM_SMB_SHARES
    321,  # ENUM_SMB_BASIC_ENUM4LINUX
    322,  # ENUM_SMB_NULL_SESSION_USERS
    323,  # ENUM_SMB_OS_DISCOVERY
    324,  # ENUM_SMB_PROTOCOLS
    410,  # CHECK_SMB_VULNS
    600,  # MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT
    0,    # STOP
]


# 4. DistCC exec - Metasploit - 602
TAIL_ATTACK_DISTCC_MSF = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    602,  # MSF_EXPLOIT_DISTCC_EXEC
    0,    # STOP
]


# 5. PostgreSQL payload - Metasploit - 604
TAIL_ATTACK_POSTGRES_MSF = [
    371,  # ENUM_POSTGRES_INFO
    523,  # CHECK_POSTGRES_KNOWN_CREDS
    604,  # MSF_EXPLOIT_POSTGRES_PAYLOAD
    0,    # STOP
]


# 6. UnrealIRCd backdoor - Metasploit - 605
TAIL_ATTACK_UNREAL_IRCD_MSF = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    605,  # MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR
    0,    # STOP
]


# 7. Ingreslock bind shell - Manual - 606
TAIL_ATTACK_INGRESLOCK_BIND_SHELL = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    606,  # CONNECT_INGRESLOCK_BIND_SHELL
    0,    # STOP
]


# 8. SSH weak credentials - Manual - 520
TAIL_ATTACK_SSH_WEAK_CREDS_MANUAL = [
    340,  # ENUM_SSH_BANNER
    341,  # ENUM_SSH_NMAP_SCRIPTS
    520,  # CHECK_SSH_KNOWN_CREDS
    0,    # STOP
]


# 9. Telnet weak credentials - Manual - 521
TAIL_ATTACK_TELNET_WEAK_CREDS_MANUAL = [
    521,  # CHECK_TELNET_KNOWN_CREDS
    0,    # STOP
]


# 10. SSH weak credentials - Metasploit - 611
TAIL_ATTACK_SSH_WEAK_CREDS_MSF = [
    340,  # ENUM_SSH_BANNER
    341,  # ENUM_SSH_NMAP_SCRIPTS
    611,  # MSF_SSH_LOGIN
    0,    # STOP
]


# 11. FTP weak credentials - Metasploit - 612
TAIL_ATTACK_FTP_WEAK_CREDS_MSF = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    612,  # MSF_FTP_LOGIN
    0,    # STOP
]


# 12. FTP weak credentials - Hydra + manual validation - 613 + 614
TAIL_ATTACK_FTP_WEAK_CREDS_HYDRA = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    613,  # HYDRA_FTP_LOGIN
    614,  # CHECK_FTP_KNOWN_CREDS_MANUAL
    0,    # STOP
]

# ============================================================
# Host-target dataset sequences
# ============================================================

SEQ_ATTACK_VSFTPD_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_VSFTPD_MSF
SEQ_ATTACK_VSFTPD_MANUAL = SEQ_HOST_BASE_RECON + TAIL_ATTACK_VSFTPD_MANUAL
SEQ_ATTACK_SAMBA_USERMAP_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_SAMBA_USERMAP_MSF
SEQ_ATTACK_DISTCC_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_DISTCC_MSF
SEQ_ATTACK_POSTGRES_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_POSTGRES_MSF
SEQ_ATTACK_UNREAL_IRCD_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_UNREAL_IRCD_MSF
SEQ_ATTACK_INGRESLOCK_BIND_SHELL = SEQ_HOST_BASE_RECON + TAIL_ATTACK_INGRESLOCK_BIND_SHELL
SEQ_ATTACK_SSH_WEAK_CREDS_MANUAL = SEQ_HOST_BASE_RECON + TAIL_ATTACK_SSH_WEAK_CREDS_MANUAL
SEQ_ATTACK_TELNET_WEAK_CREDS_MANUAL = SEQ_HOST_BASE_RECON + TAIL_ATTACK_TELNET_WEAK_CREDS_MANUAL
SEQ_ATTACK_SSH_WEAK_CREDS_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_SSH_WEAK_CREDS_MSF
SEQ_ATTACK_FTP_WEAK_CREDS_MSF = SEQ_HOST_BASE_RECON + TAIL_ATTACK_FTP_WEAK_CREDS_MSF
SEQ_ATTACK_FTP_WEAK_CREDS_HYDRA = SEQ_HOST_BASE_RECON + TAIL_ATTACK_FTP_WEAK_CREDS_HYDRA


# ============================================================
# Network-target dataset sequences
# ============================================================

SEQ_NETWORK_ATTACK_VSFTPD_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_VSFTPD_MSF
SEQ_NETWORK_ATTACK_VSFTPD_MANUAL = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_VSFTPD_MANUAL
SEQ_NETWORK_ATTACK_SAMBA_USERMAP_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_SAMBA_USERMAP_MSF
SEQ_NETWORK_ATTACK_DISTCC_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_DISTCC_MSF
SEQ_NETWORK_ATTACK_POSTGRES_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_POSTGRES_MSF
SEQ_NETWORK_ATTACK_UNREAL_IRCD_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_UNREAL_IRCD_MSF
SEQ_NETWORK_ATTACK_INGRESLOCK_BIND_SHELL = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_INGRESLOCK_BIND_SHELL
SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MANUAL = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_SSH_WEAK_CREDS_MANUAL
SEQ_NETWORK_ATTACK_TELNET_WEAK_CREDS_MANUAL = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_TELNET_WEAK_CREDS_MANUAL
SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_SSH_WEAK_CREDS_MSF
SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_MSF = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_FTP_WEAK_CREDS_MSF
SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_HYDRA = SEQ_NETWORK_BASE_RECON + TAIL_ATTACK_FTP_WEAK_CREDS_HYDRA


# ============================================================
# Recon-only sequences
# ============================================================

SEQ_HOST_RECON = [
    *SEQ_HOST_BASE_RECON,
    0,
]


SEQ_NETWORK_RECON = [
    *SEQ_NETWORK_BASE_RECON,
    0,
]


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
    # --------------------------------------------------------
    # Recon only
    # --------------------------------------------------------
    "host_recon": SEQ_HOST_RECON,
    "network_recon": SEQ_NETWORK_RECON,

    # --------------------------------------------------------
    # Host-target attack sequences
    # --------------------------------------------------------
    "attack_vsftpd_msf": SEQ_ATTACK_VSFTPD_MSF,
    "attack_vsftpd_manual": SEQ_ATTACK_VSFTPD_MANUAL,
    "attack_samba_usermap_msf": SEQ_ATTACK_SAMBA_USERMAP_MSF,
    "attack_distcc_msf": SEQ_ATTACK_DISTCC_MSF,
    "attack_postgres_msf": SEQ_ATTACK_POSTGRES_MSF,
    "attack_unreal_ircd_msf": SEQ_ATTACK_UNREAL_IRCD_MSF,
    "attack_ingreslock_bind_shell": SEQ_ATTACK_INGRESLOCK_BIND_SHELL,
    "attack_ssh_weak_creds_manual": SEQ_ATTACK_SSH_WEAK_CREDS_MANUAL,
    "attack_telnet_weak_creds_manual": SEQ_ATTACK_TELNET_WEAK_CREDS_MANUAL,
    "attack_ssh_weak_creds_msf": SEQ_ATTACK_SSH_WEAK_CREDS_MSF,
    "attack_ftp_weak_creds_msf": SEQ_ATTACK_FTP_WEAK_CREDS_MSF,
    "attack_ftp_weak_creds_hydra": SEQ_ATTACK_FTP_WEAK_CREDS_HYDRA,

    # --------------------------------------------------------
    # Network-target attack sequences
    # --------------------------------------------------------
    "network_attack_vsftpd_msf": SEQ_NETWORK_ATTACK_VSFTPD_MSF,
    "network_attack_vsftpd_manual": SEQ_NETWORK_ATTACK_VSFTPD_MANUAL,
    "network_attack_samba_usermap_msf": SEQ_NETWORK_ATTACK_SAMBA_USERMAP_MSF,
    "network_attack_distcc_msf": SEQ_NETWORK_ATTACK_DISTCC_MSF,
    "network_attack_postgres_msf": SEQ_NETWORK_ATTACK_POSTGRES_MSF,
    "network_attack_unreal_ircd_msf": SEQ_NETWORK_ATTACK_UNREAL_IRCD_MSF,
    "network_attack_ingreslock_bind_shell": SEQ_NETWORK_ATTACK_INGRESLOCK_BIND_SHELL,
    "network_attack_ssh_weak_creds_manual": SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MANUAL,
    "network_attack_telnet_weak_creds_manual": SEQ_NETWORK_ATTACK_TELNET_WEAK_CREDS_MANUAL,
    "network_attack_ssh_weak_creds_msf": SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MSF,
    "network_attack_ftp_weak_creds_msf": SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_MSF,
    "network_attack_ftp_weak_creds_hydra": SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_HYDRA,

    # --------------------------------------------------------
    # Short aliases
    # --------------------------------------------------------
    "vsftpd": SEQ_NETWORK_ATTACK_VSFTPD_MSF,
    "vsftpd_manual": SEQ_NETWORK_ATTACK_VSFTPD_MANUAL,
    "samba": SEQ_NETWORK_ATTACK_SAMBA_USERMAP_MSF,
    "distcc": SEQ_NETWORK_ATTACK_DISTCC_MSF,
    "postgres": SEQ_NETWORK_ATTACK_POSTGRES_MSF,
    "unreal_ircd": SEQ_NETWORK_ATTACK_UNREAL_IRCD_MSF,
    "ingreslock": SEQ_NETWORK_ATTACK_INGRESLOCK_BIND_SHELL,
    "ssh": SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MANUAL,
    "telnet": SEQ_NETWORK_ATTACK_TELNET_WEAK_CREDS_MANUAL,
    "ssh_msf": SEQ_NETWORK_ATTACK_SSH_WEAK_CREDS_MSF,
    "ftp_msf": SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_MSF,
    "ftp_hydra": SEQ_NETWORK_ATTACK_FTP_WEAK_CREDS_HYDRA,

    # --------------------------------------------------------
    # Debug only
    # --------------------------------------------------------
    "standard": SEQ_NETWORK_ATTACK_VSFTPD_MSF,
    "exploit_smoke_test": SEQ_EXPLOIT_SMOKE_TEST,
}



# Secuencia FTP: vsftpd 2.3.4 -> sesión
SEQ_NETWORK_TO_VSFTPD_SESSION = [
    100, 101, 102, 103,
    # 200, 210, 211, 401,
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS: ftp-anon,ftp-syst only
    413,  # CHECK_FTP_VULNS: ftp-vsftpd-backdoor
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR

    # ya se ejecutan dentro de 601 mediante: sessions -c "whoami && id && hostname && uname -a" -i 1
    # 700, 702, 703, 701, 704,
    0,
]


# Secuencia multi-sesión completa
SEQ_NETWORK_TO_MULTIPLE_SESSIONS = [
    # Local attacker context
    100,  # INSPECT_LOCAL_HOSTNAME
    101,  # INSPECT_IP_A
    102,  # INSPECT_IP_R
    103,  # INSPECT_IP_NEIGH

    # Initial recon
    200,  # DISCOVER_HOSTS
    210,  # SCAN_TOP_TCP_PORTS
    211,  # SCAN_FULL_TCP_PORTS
    220,  # DETECT_SERVICES
    401,  # CHECK_NMAP_VULN_SCRIPTS

    # FTP / vsftpd
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    413,  # CHECK_FTP_VULNS
    400,  # CHECK_SERVICE_VERSION_VULNS
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # SMB / Samba
    320,  # ENUM_SMB_SHARES
    321,  # ENUM_SMB_BASIC_ENUM4LINUX
    322,  # ENUM_SMB_NULL_SESSION_USERS
    323,  # ENUM_SMB_OS_DISCOVERY
    324,  # ENUM_SMB_PROTOCOLS
    410,  # CHECK_SMB_VULNS
    600,  # MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # distcc
    400,  # CHECK_SERVICE_VERSION_VULNS
    602,  # MSF_EXPLOIT_DISTCC_EXEC
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # PostgreSQL
    371,  # ENUM_POSTGRES_INFO
    523,  # CHECK_POSTGRES_KNOWN_CREDS
    604,  # MSF_EXPLOIT_POSTGRES_PAYLOAD
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # Tomcat
    300,  # ENUM_HTTP_HEADERS
    301,  # ENUM_HTTP_INDEX
    313,  # ENUM_HTTP_TECHNOLOGIES
    312,  # ENUM_HTTP_NIKTO
    411,  # CHECK_HTTP_VULNS_NIKTO
    524,  # CHECK_TOMCAT_MANAGER_CREDS
    603,  # MSF_EXPLOIT_TOMCAT_MGR_UPLOAD
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # UnrealIRCd
    400,  # CHECK_SERVICE_VERSION_VULNS
    605,  # MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # ingreslock bind shell
    606,  # CONNECT_INGRESLOCK_BIND_SHELL
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    701,  # POST_ENUM_UNAME

    # Final stop
    0,    # STOP
]

# Secuencia con post-explotación y pivoting
SEQ_POST_EXPLOIT_AND_PIVOT = [
    # Basic post-exploitation
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID
    703,  # POST_ENUM_HOSTNAME
    701,  # POST_ENUM_UNAME
    704,  # POST_ENUM_IP_ADDR
    705,  # POST_ENUM_IP_ROUTE
    706,  # POST_ENUM_SS_LISTENERS
    708,  # POST_ENUM_USERS_PASSWD
    709,  # POST_ENUM_HOME_USERS
    710,  # POST_ENUM_PROCESSES
    711,  # POST_ENUM_ENV

    # Privilege escalation discovery
    712,  # POST_CHECK_SUDO_PRIVS
    713,  # POST_CHECK_SUDOERS_PERMS
    714,  # POST_FIND_SUID_BINARIES
    733,  # MSF_LOCAL_EXPLOIT_SUGGESTER

    # Privilege escalation attempt
    715,  # POST_PRIVESC_NMAP_INTERACTIVE

    # Root validation
    700,  # POST_ENUM_WHOAMI
    702,  # POST_ENUM_ID

    # Sensitive evidence
    717,  # POST_READ_SHADOW
    718,  # POST_LIST_SSH_HOST_KEYS
    719,  # POST_READ_CRONTAB

    # Metasploit post modules
    730,  # MSF_POST_ENUM_SYSTEM
    731,  # MSF_POST_ENUM_CONFIGS
    732,  # MSF_POST_ENUM_NETWORK

    # Pivoting
    760,  # MSF_AUTOROUTE_ADD_INTERNAL_NET
    761,  # MSF_START_SOCKS_PROXY
    762,  # PIVOT_SCAN_INTERNAL_NET_PROXYCHAINS
    763,  # MSF_PORTFWD_ADD
    764,  # PIVOT_MYSQL_LOGIN_LOCAL_PORTFWD

    # Evidence / exfiltration
    780,  # POST_SEARCH_WEB_PASSWORDS
    781,  # POST_SEARCH_WEB_DB_CONFIGS
    782,  # POST_READ_DVWA_CONFIG
    783,  # POST_READ_PHPMYADMIN_CONFIG
    784,  # POST_READ_TIKIWIKI_DB_CONFIG
    790,  # POST_ARCHIVE_MYSQL_DATA_DIR
    791,  # POST_ARCHIVE_POSTGRES_DATA_DIR

    0,    # STOP
]



DEFAULT_SCRIPTED_SEQUENCE_NAME = "attack_vsftpd_msf"
DEFAULT_SCRIPTED_SEQUENCE = SCRIPTED_SEQUENCES[DEFAULT_SCRIPTED_SEQUENCE_NAME]

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
