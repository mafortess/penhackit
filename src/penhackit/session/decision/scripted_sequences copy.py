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
# Dataset attack tails
# ============================================================

# ------------------------------------------------------------
# 1. VSFTPD 2.3.4 backdoor - Metasploit - 601
# ------------------------------------------------------------

TAIL_ATTACK_VSFTPD_MSF = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    413,  # CHECK_FTP_VULNS
    400,  # CHECK_SERVICE_VERSION_VULNS
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 2. VSFTPD 2.3.4 backdoor - Manual - 610
# ------------------------------------------------------------

TAIL_ATTACK_VSFTPD_MANUAL = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    413,  # CHECK_FTP_VULNS
    400,  # CHECK_SERVICE_VERSION_VULNS
    610,  # MANUAL_EXPLOIT_VSFTPD_234_BACKDOOR
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 3. Samba usermap_script - Metasploit - 600
# ------------------------------------------------------------

TAIL_ATTACK_SAMBA_USERMAP_MSF = [
    320,  # ENUM_SMB_SHARES
    321,  # ENUM_SMB_BASIC_ENUM4LINUX
    322,  # ENUM_SMB_NULL_SESSION_USERS
    323,  # ENUM_SMB_OS_DISCOVERY
    324,  # ENUM_SMB_PROTOCOLS
    410,  # CHECK_SMB_VULNS
    400,  # CHECK_SERVICE_VERSION_VULNS
    600,  # MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 4. DistCC exec - Metasploit - 602
# ------------------------------------------------------------

TAIL_ATTACK_DISTCC_MSF = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    602,  # MSF_EXPLOIT_DISTCC_EXEC
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 5. PostgreSQL payload - Metasploit - 604
# ------------------------------------------------------------

TAIL_ATTACK_POSTGRES_MSF = [
    371,  # ENUM_POSTGRES_INFO
    523,  # CHECK_POSTGRES_KNOWN_CREDS
    400,  # CHECK_SERVICE_VERSION_VULNS
    604,  # MSF_EXPLOIT_POSTGRES_PAYLOAD
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 6. UnrealIRCd backdoor - Metasploit - 605
# ------------------------------------------------------------

TAIL_ATTACK_UNREAL_IRCD_MSF = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    605,  # MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 7. Ingreslock bind shell - Manual / netcat - 606
# ------------------------------------------------------------

TAIL_ATTACK_INGRESLOCK_BIND_SHELL = [
    400,  # CHECK_SERVICE_VERSION_VULNS
    606,  # CONNECT_INGRESLOCK_BIND_SHELL
    *SEQ_BASIC_POST_EXPLOIT,
    0,
]


# ------------------------------------------------------------
# 8. SSH weak credentials - Manual - 520
# ------------------------------------------------------------

TAIL_ATTACK_SSH_WEAK_CREDS_MANUAL = [
    340,  # ENUM_SSH_BANNER
    341,  # ENUM_SSH_NMAP_SCRIPTS
    520,  # CHECK_SSH_KNOWN_CREDS
    0,
]


# ------------------------------------------------------------
# 9. Telnet weak credentials - Manual - 521
# ------------------------------------------------------------

TAIL_ATTACK_TELNET_WEAK_CREDS_MANUAL = [
    521,  # CHECK_TELNET_KNOWN_CREDS
    0,
]


# ------------------------------------------------------------
# 10. SSH weak credentials - Metasploit - 611
# ------------------------------------------------------------

TAIL_ATTACK_SSH_WEAK_CREDS_MSF = [
    340,  # ENUM_SSH_BANNER
    341,  # ENUM_SSH_NMAP_SCRIPTS
    611,  # MSF_SSH_LOGIN
    *SEQ_BASIC_POST_EXPLOIT,
    705,  # POST_ENUM_IP_ROUTE
    0,
]


# ------------------------------------------------------------
# 11. FTP weak credentials - Metasploit - 612
# ------------------------------------------------------------

TAIL_ATTACK_FTP_WEAK_CREDS_MSF = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    612,  # MSF_FTP_LOGIN
    0,
]


# ------------------------------------------------------------
# 12. FTP weak credentials - Hydra - 613
# 12b. FTP manual validation - 614
# ------------------------------------------------------------

TAIL_ATTACK_FTP_WEAK_CREDS_HYDRA = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    613,  # HYDRA_FTP_LOGIN
    614,  # CHECK_FTP_KNOWN_CREDS_MANUAL
    0,
]

TAIL_ATTACK_FTP_MANUAL_VALIDATION = [
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS
    614,  # CHECK_FTP_KNOWN_CREDS_MANUAL
    0,
]



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

# Secuencia SMB: Samba usermap_script -> sesión
SEQ_NETWORK_TO_SAMBA_SESSION = [
    100, 101, 102, 103,
    200, 210, 211, 401, 320,
    321,
    322,
    323,
    324,
    410,
    400,
    600,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia distcc -> sesión
SEQ_NETWORK_TO_DISTCC_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    400,
    602,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia PostgreSQL -> sesión
SEQ_NETWORK_TO_POSTGRES_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    371,
    523,
    604,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia Tomcat Manager -> sesión
SEQ_NETWORK_TO_TOMCAT_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    300,
    301,
    313,
    312,
    411,
    524,
    603,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia UnrealIRCd -> sesión
SEQ_NETWORK_TO_UNREAL_IRCD_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    400,
    605,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia ingreslock bind shell -> sesión
SEQ_NETWORK_TO_INGRESLOCK_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    606,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia SSH credenciales débiles -> sesión
SEQ_NETWORK_TO_SSH_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    341,
    520,
    700,
    702,
    703,
    701,
    704,
    705,
    0,
]

# Secuencia Telnet credenciales débiles -> sesión
SEQ_NETWORK_TO_TELNET_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    521,
    700,
    702,
    703,
    701,
    704,
    0,
]

# Secuencia para sesión Java RMI
SEQ_NETWORK_TO_JAVA_RMI_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    400,
    608,
    700,
    702,
    703,
    701,
    704,
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

# SCRIPTED_SEQUENCE = [
#     # ============================================================
#     # LOCAL CONTEXT
#     # ============================================================
#     101,  # INSPECT_IP_A
#     102,  # INSPECT_IP_R
#     103,  # INSPECT_IP_NEIGH

#     # ============================================================
#     # HOST DISCOVERY
#     # ============================================================
#     200,  # DISCOVER_HOSTS / DISCOVER_HOSTS_NMAP_PING_SWEEP
#     201,  # DISCOVER_HOSTS_ARP_LOCALNET

#     # ============================================================
#     # PORT SCANNING
#     # ============================================================
#     210,  # SCAN_TOP_TCP_PORTS
#     211,  # SCAN_FULL_TCP_PORTS

#     # ============================================================
#     # SERVICE DETECTION
#     # ============================================================
#     220,  # DETECT_SERVICES
#     230,  # ENUM_NMAP_DEFAULT_SCRIPTS

#     # ============================================================
#     # HTTP ENUMERATION
#     # ============================================================
#     300,  # ENUM_HTTP_HEADERS
#     301,  # ENUM_HTTP_INDEX
#     303,  # ENUM_HTTP_ROBOTS
#     313,  # ENUM_HTTP_TECHNOLOGIES
#     310,  # ENUM_HTTP_DIRS_GOBUSTER
#     312,  # ENUM_HTTP_NIKTO

#     # ============================================================
#     # SMB ENUMERATION
#     # ============================================================
#     320,  # ENUM_SMB_SHARES
#     323,  # ENUM_SMB_OS_DISCOVERY
#     324,  # ENUM_SMB_PROTOCOLS
#     321,  # ENUM_SMB_BASIC_ENUM4LINUX
#     322,  # ENUM_SMB_NULL_SESSION_USERS

#     # ============================================================
#     # FTP ENUMERATION
#     # ============================================================
#     330,  # ENUM_FTP_BANNER
#     331,  # ENUM_FTP_ANONYMOUS
#     332,  # ENUM_FTP_NMAP_SCRIPTS

#     # ============================================================
#     # SSH ENUMERATION
#     # ============================================================
#     340,  # ENUM_SSH_BANNER
#     341,  # ENUM_SSH_NMAP_SCRIPTS

#     # ============================================================
#     # DNS / RPC / NFS ENUMERATION
#     # ============================================================
#     350,  # ENUM_DNS_VERSION_BIND
#     351,  # ENUM_DNS_ANY
#     361,  # ENUM_RPCINFO
#     360,  # ENUM_NFS_EXPORTS

#     # ============================================================
#     # DATABASE / REMOTE ACCESS ENUMERATION
#     # ============================================================
#     370,  # ENUM_MYSQL_INFO
#     371,  # ENUM_POSTGRES_INFO
#     380,  # ENUM_RDP_INFO
#     381,  # ENUM_VNC_INFO

#     # ============================================================
#     # VULNERABILITY DISCOVERY
#     # ============================================================
#     400,  # CHECK_SERVICE_VERSION_VULNS
#     401,  # CHECK_NMAP_VULN_SCRIPTS
#     410,  # CHECK_SMB_VULNS
#     411,  # CHECK_HTTP_VULNS_NIKTO
#     412,  # CHECK_SSL_TLS_CIPHERS
#     413,  # CHECK_FTP_VULNS

#     # ============================================================
#     # STOP
#     # ============================================================
#     0,    # STOP
# ]



# ============================================================
# Registry
# ============================================================

SCRIPTED_SEQUENCES = {
    # Smoke tests
    "exploit_smoke_test": SEQ_EXPLOIT_SMOKE_TEST,

    # Recon
    "network_recon": SEQ_COMMON_NETWORK_RECON,
    "host_recon": SEQ_COMMON_HOST_RECON,

    # Dataset attacks
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
    "attack_ftp_manual_validation": SEQ_ATTACK_FTP_MANUAL_VALIDATION,

    "vsftpd": SEQ_NETWORK_TO_VSFTPD_SESSION,
    "samba": SEQ_NETWORK_TO_SAMBA_SESSION,
    "distcc": SEQ_NETWORK_TO_DISTCC_SESSION,
    "postgres": SEQ_NETWORK_TO_POSTGRES_SESSION,
    # "tomcat": SEQ_NETWORK_TO_TOMCAT_SESSION,
    "unreal_ircd": SEQ_NETWORK_TO_UNREAL_IRCD_SESSION,
    "ingreslock": SEQ_NETWORK_TO_INGRESLOCK_SESSION,
    "ssh": SEQ_NETWORK_TO_SSH_SESSION,
    "telnet": SEQ_NETWORK_TO_TELNET_SESSION,
    "java_rmi": SEQ_NETWORK_TO_JAVA_RMI_SESSION,

    "multi_session": SEQ_NETWORK_TO_MULTIPLE_SESSIONS,
    "post_exploit_pivot": SEQ_POST_EXPLOIT_AND_PIVOT,
}

DEFAULT_SCRIPTED_SEQUENCE_NAME = "vsftpd"
DEFAULT_SCRIPTED_SEQUENCE = SCRIPTED_SEQUENCES[DEFAULT_SCRIPTED_SEQUENCE_NAME]


def get_scripted_sequence(name: str | None = None) -> list[int]:
    """
    Return a predefined scripted action sequence.

    If name is None or unknown, return the default sequence.
    """
    if not name:
        return DEFAULT_SCRIPTED_SEQUENCE

    return SCRIPTED_SEQUENCES.get(name, DEFAULT_SCRIPTED_SEQUENCE)

