def build_state(kb: dict, session_context: dict) -> dict:
    print("Building state from KB and session context...")

    net = kb.get("net", {}) or {}
    focus = kb.get("focus", {}) or {}

    hosts = kb.get("hosts", []) or []
    services = kb.get("services", []) or []
    findings = kb.get("findings", []) or []

    if not isinstance(hosts, dict):
        hosts = {}

    arp_neighbors = net.get("arp_neighbors", []) or []
    ipv4 = net.get("ipv4", []) or []
    default_gw = net.get("default_gw", []) or []
    interfaces = net.get("interfaces", []) or []
    routes = net.get("routes", []) or []

    focus_host_ip = focus.get("host") or ""
    focus_service = focus.get("service") or ""

    focus_host = hosts.get(focus_host_ip, {}) if focus_host_ip else {}

    ports = focus_host.get("ports", {}) or {}
    host_services = focus_host.get("services", {}) or {}
    web_paths = focus_host.get("web_paths", []) or []
    smb_shares = focus_host.get("smb_shares", []) or []
    candidate_vulns = focus_host.get("candidate_vulns", []) or []

    if not isinstance(ports, dict):
        ports = {}

    if not isinstance(host_services, dict):
        host_services = {}

    service_names = set()
    open_tcp_ports = []
    open_udp_ports = []

    for port_key, port_data in ports.items():
        if not isinstance(port_data, dict):
            continue

        service_name = str(port_data.get("service", "") or "").lower()
        proto = str(port_data.get("proto", "") or "").lower()
        port = port_data.get("port")

        if service_name:
            service_names.add(service_name)

        if proto == "tcp":
            open_tcp_ports.append(port)

        if proto == "udp":
            open_udp_ports.append(port)

    for service_key, service_data in host_services.items():
        if not isinstance(service_data, dict):
            continue

        service_name = str(service_data.get("service", "") or service_data.get("name", "") or "").lower()
        if service_name:
            service_names.add(service_name)

    current_service_name = ""
    current_service_port = 0
    current_service_has_version = False

    if focus_service:
        service_data = host_services.get(str(focus_service), {}) or {}

        if isinstance(service_data, dict):
            current_service_name = str(
                service_data.get("service", "") or service_data.get("name", "") or ""
            ).lower()
            current_service_port = int(service_data.get("port", 0) or 0)
            current_service_has_version = bool(service_data.get("version"))

    if not current_service_name and ports:
        first_port_data = None

        for _, port_data in ports.items():
            if isinstance(port_data, dict):
                first_port_data = port_data
                break

        if first_port_data:
            current_service_name = str(first_port_data.get("service", "") or "").lower()
            current_service_port = int(first_port_data.get("port", 0) or 0)
            current_service_has_version = bool(first_port_data.get("version"))

    has_http = (
        "http" in service_names
        or "http-proxy" in service_names
        or 80 in open_tcp_ports
        or 8080 in open_tcp_ports
        or 8000 in open_tcp_ports
        or 8888 in open_tcp_ports
    )

    has_https = (
        "https" in service_names
        or "ssl/http" in service_names
        or 443 in open_tcp_ports
        or 8443 in open_tcp_ports
    )

    has_ssh = "ssh" in service_names or 22 in open_tcp_ports
    has_ftp = "ftp" in service_names or 21 in open_tcp_ports

    has_smb = (
        "microsoft-ds" in service_names
        or "netbios-ssn" in service_names
        or "smb" in service_names
        or 445 in open_tcp_ports
        or 139 in open_tcp_ports
    )

    has_dns = (
        "domain" in service_names
        or "dns" in service_names
        or 53 in open_tcp_ports
        or 53 in open_udp_ports
    )

    has_rdp = (
        "ms-wbt-server" in service_names
        or "rdp" in service_names
        or 3389 in open_tcp_ports
    )

    has_nfs = (
        "nfs" in service_names
        or "rpcbind" in service_names
        or 111 in open_tcp_ports
        or 2049 in open_tcp_ports
    )

    has_mysql = "mysql" in service_names or 3306 in open_tcp_ports
    has_postgres = "postgresql" in service_names or 5432 in open_tcp_ports
    has_vnc = "vnc" in service_names or 5900 in open_tcp_ports

    state = {
        # Goal / task
        "goal_type": session_context.get("goal_type", "demo"),

        # Focus (nivel y si hay algo seleccionado)
        "focus_level": focus.get("level", "global"),
        "has_focus_host": bool(focus.get("host")),
        "has_focus_service": bool(focus.get("service")),

        # Features por “nivel” (resumen, no datos crudos)
        "net_ipv4_count": len(ipv4),
        "net_gw_count": len(default_gw),
        "net_if_count": len(interfaces),
        "net_arp_count": len(arp_neighbors),
        "net_routes_count": len(routes),

        # Global KB summary
        "hosts_count": len(hosts),
        "services_count": len(services),
        "findings_count": len(findings),

        # Service presence flags
        "focus_host_has_http": has_http,
        "focus_host_has_https": has_https,
        "focus_host_has_ssh": has_ssh,
        "focus_host_has_ftp": has_ftp,
        "focus_host_has_smb": has_smb,
        "focus_host_has_dns": has_dns,
        "focus_host_has_rdp": has_rdp,
        "focus_host_has_nfs": has_nfs,
        "focus_host_has_mysql": has_mysql,
        "focus_host_has_postgres": has_postgres,
        "focus_host_has_vnc": has_vnc,

        # Current service summary
        "current_service_port": current_service_port,
        "current_service_name": current_service_name,
        "current_service_has_version": current_service_has_version,

        # Last transition (para que la policy no repita/pueda detectar error)
        "last_action_id": kb.get("last_action_id"),
        "last_action_name": kb.get("last_action_name"),
        "last_rc": kb.get("last_rc"),
        "last_event_type": kb.get("last_event_type"),

        # Progreso / estancamiento (mínimo)
        "step_idx": kb.get("step_idx", 0),
    }
    return state