
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

def scripted_policy_decide_action(state, t):
    """
    Política de ejemplo muy simple y determinista: sigue una secuencia predefinida de acciones (SCRIPTED_SEQUENCE).
    Esto es útil para probar tu pipeline end-to-end con un comportamiento conocido.
    """
    print("Deciding action based on state...")
    # Política muy simple y determinista basada en el estado actual (ejemplo MVP)
    # action = t+1

    # Secuencia predefinida de acciones para probar el pipeline end-to-end
    if t < len(SCRIPTED_SEQUENCE):
        action = SCRIPTED_SEQUENCE[t]
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
    x = np.zeros((1, len(feature_names)), dtype=np.float32)

    for j, k in enumerate(feature_names):
        v = state.get(k, 0)
        if isinstance(v, bool):
            v = 1 if v else 0
        if v is None:
            v = 0
        x[0, j] = float(v)

    y_pred = model.predict(x)
    return int(y_pred[0])
