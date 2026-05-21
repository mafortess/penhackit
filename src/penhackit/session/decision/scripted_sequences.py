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
"""

SCRIPTED_SEQUENCE = [
    # 1,
    # 2,
    # 3,
    # 4,
    200,  # DISCOVER_HOSTS
    201,  # DISCOVER_HOSTS_ARP_LOCALNET
    210,  # SCAN_TOP_TCP_PORTS
    220,  # DETECT_SERVICES
    300,  # ENUM_HTTP_HEADERS
    310,  # ENUM_HTTP_DIRS
    320,  # ENUM_SMB_SHARES
    400,  # CHECK_SERVICE_VERSION_VULNS
    0,    # STOP
]

# Prefijo común si el target es una red
SEQ_COMMON_NETWORK_RECON = [
    100,  # INSPECT_LOCAL_HOSTNAME
    101,  # INSPECT_IP_A
    102,  # INSPECT_IP_R
    103,  # INSPECT_IP_NEIGH
    200,  # DISCOVER_HOSTS
    210,  # SCAN_TOP_TCP_PORTS
    211,  # SCAN_FULL_TCP_PORTS
    220,  # DETECT_SERVICES
    401,  # CHECK_NMAP_VULN_SCRIPTS
]

# Prefijo común si el target es un host
SEQ_COMMON_HOST_RECON = [
    100,  # INSPECT_LOCAL_HOSTNAME
    101,  # INSPECT_IP_A
    102,  # INSPECT_IP_R
    103,  # INSPECT_IP_NEIGH
    105,  # PING_FOCUS_HOST
    210,  # SCAN_TOP_TCP_PORTS
    211,  # SCAN_FULL_TCP_PORTS
    220,  # DETECT_SERVICES
    401,  # CHECK_NMAP_VULN_SCRIPTS
]

# Secuencia FTP: vsftpd 2.3.4 -> sesión
SEQ_NETWORK_TO_VSFTPD_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    # 401,
    330,  # ENUM_FTP_BANNER
    331,  # ENUM_FTP_ANONYMOUS
    332,  # ENUM_FTP_NMAP_SCRIPTS: ftp-anon,ftp-syst only
    413,  # CHECK_FTP_VULNS: ftp-vsftpd-backdoor
    601,  # MSF_EXPLOIT_VSFTPD_234_BACKDOOR
    # 700,
    # 702,
    # 703,
    # 701,
    # 704,
    0,
]

# Secuencia SMB: Samba usermap_script -> sesión
SEQ_NETWORK_TO_SAMBA_SESSION = [
    100, 101, 102, 103,
    200,
    210,
    211,
    401,
    320,
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
    "network_recon": SEQ_COMMON_NETWORK_RECON,
    "host_recon": SEQ_COMMON_HOST_RECON,

    "vsftpd": SEQ_NETWORK_TO_VSFTPD_SESSION,
    "samba": SEQ_NETWORK_TO_SAMBA_SESSION,
    "distcc": SEQ_NETWORK_TO_DISTCC_SESSION,
    "postgres": SEQ_NETWORK_TO_POSTGRES_SESSION,
    "tomcat": SEQ_NETWORK_TO_TOMCAT_SESSION,
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

