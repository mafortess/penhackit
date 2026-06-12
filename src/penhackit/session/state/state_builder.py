# state_builder.py
#
# Construcción del estado tabular del agente a partir de la KB v2.
#
# Regla principal:
#   state_t se construye ANTES de ejecutar action_t.
#
# Este módulo no modifica la KB, no ejecuta comandos y no decide acciones.

from typing import Any, Optional


SERVICE_FLAGS = {
    "ftp": ["ftp", "vsftpd", "proftpd"],
    "ssh": ["ssh", "openssh"],
    "telnet": ["telnet"],
    "smb": ["smb", "samba", "netbios", "microsoft-ds"],
    "postgres": ["postgres", "postgresql"],
    "distcc": ["distcc", "distccd"],
    "irc": ["irc", "ircd", "unreal"],
    "ingreslock": ["ingreslock", "1524"],
}


def build_state(kb: dict, session_context: Optional[dict] = None) -> dict:
    """
    Construye un estado plano y estable a partir de la KB v2.

    El estado está pensado para datasets de Behavioral Cloning:
        state_t -> action_id

    No debe contener listas completas de hosts, servicios o vulnerabilidades.
    Solo resume el conocimiento disponible mediante contadores, flags, foco,
    firmas ofensivas y cobertura de acciones relevantes.
    """
    session_context = session_context or {}

    scope = kb.get("scope", {})
    target = kb.get("target", {})
    attacker = kb.get("attacker", {})
    coverage = kb.get("coverage", {})
    focus = kb.get("focus", {})
    last = kb.get("last", {})

    hosts = target.get("hosts", {})
    ports = target.get("ports", {})
    services = target.get("services", {})
    vulnerabilities = kb.get("vulnerabilities", {})
    credentials = kb.get("credentials", {})
    sessions = kb.get("sessions", {})
    attempts = kb.get("attempts", {})
    findings = kb.get("findings", {})

    focus_host = get_focus_host(kb)
    focus_service = get_focus_service(kb)
    focus_vuln = get_focus_vulnerability(kb)
    focus_session = get_focus_session(kb)

    focus_host_services = get_services_for_host(kb, focus_host)
    focus_host_ports = get_ports_for_host(kb, focus_host)
    focus_host_vulns = get_vulns_for_host(kb, focus_host)
    focus_host_sessions = get_sessions_for_host(kb, focus_host)

    focus_service_vulns = get_vulns_for_service(kb, focus_service)
    focus_service_credentials = get_credentials_for_service(kb, focus_service)
    focus_service_sessions = get_sessions_for_service(kb, focus_service)

    goal_type = (
        session_context.get("goal_type")
        or scope.get("goal")
        or "unknown"
    )

    target_type = (
        session_context.get("target_type")
        or scope.get("target_type")
        or "unknown"
    )

    state = {
        # ========================================================
        # Scope / goal
        # ========================================================
        "goal_type": goal_type,
        "target_type": target_type,
        "focus_level": focus.get("level") or "global",
        "focus_reason": focus.get("reason") or "",

        # ========================================================
        # Step / temporal context
        # ========================================================
        "step_idx": safe_int(last.get("step_idx"), default=0),
        "history_count": len(kb.get("history", [])),
        "last_action_id": safe_int(last.get("action_id"), default=0),
        "last_action_name": last.get("action_name") or "",
        "last_rc": safe_int(last.get("rc"), default=-999),
        "last_event_type": first_event_type(last.get("event_types")),
        "last_progress": bool(last.get("progress")),
        "recent_no_progress_count": 0,
        "recent_repeated_action_count": 0,

        # ========================================================
        # Attacker / local context
        # ========================================================
        "net_if_count": len(attacker.get("interfaces", [])),
        "net_ipv4_count": count_attacker_ipv4(attacker),
        "net_gw_count": len(attacker.get("default_gw", [])),
        "net_routes_count": len(attacker.get("routes", [])),
        "net_arp_count": len(attacker.get("arp_neighbors", [])),
        "has_local_hostname": bool(attacker.get("hostname")),
        "has_lhost": bool(attacker.get("lhost") or attacker.get("ipv4")),

        # ========================================================
        # Global KB counts
        # ========================================================
        "networks_count": len(target.get("networks", {})),
        "hosts_count": len(hosts),
        "alive_hosts_count": count_alive_hosts(hosts),
        "ports_count": len(ports),
        "open_ports_count": count_open_ports(ports),
        "services_count": len(services),
        "services_with_version_count": count_services_with_version(services),
        "vulns_count": len(vulnerabilities),
        "vulns_pending_count": count_vulns_by_status(vulnerabilities, {"candidate", "validated"}),
        "vulns_validated_count": count_vulns_by_status(vulnerabilities, {"validated"}),
        "credentials_count": len(credentials),
        "valid_credentials_count": count_valid_credentials(credentials),
        "sessions_count": len(sessions),
        "open_sessions_count": count_open_sessions(sessions),
        "attempts_count": len(attempts),
        "findings_count": len(findings),

        # ========================================================
        # General coverage
        # ========================================================
        "has_host_discovery": bool(coverage.get("hosts_discovered")),
        "has_port_scan": bool(coverage.get("hosts_port_scanned")),
        "has_service_detection": bool(coverage.get("hosts_service_scanned")),
        "has_service_enumeration": bool(coverage.get("services_enumerated")),
        "has_vuln_discovery": bool(vulnerabilities),
        "has_credentials": bool(credentials),
        "has_valid_credentials": count_valid_credentials(credentials) > 0,
        "has_exploit_attempt": bool(attempts) or bool(coverage.get("exploits_attempted")),
        "has_session": bool(sessions),

        # ========================================================
        # Pending work
        # ========================================================
        "pending_hosts_portscan_count": count_pending_hosts_portscan(kb),
        "pending_hosts_servicescan_count": count_pending_hosts_servicescan(kb),
        "pending_services_enum_count": count_pending_services_enum(kb),
        "pending_vulns_count": count_pending_vulns(kb),

        # ========================================================
        # Focus host
        # ========================================================
        "has_focus_host_id": bool(focus.get("host_id")),
        "has_focus_host": focus_host is not None,
        "focus_host_alive": bool(focus_host.get("alive")) if focus_host else False,
        "focus_host_ports_count": len(focus_host_ports),
        "focus_host_open_ports_count": count_open_ports_list(focus_host_ports),
        "focus_host_services_count": len(focus_host_services),
        "focus_host_services_with_version_count": count_services_with_version_list(focus_host_services),
        "focus_host_vulns_count": len(focus_host_vulns),
        "focus_host_vulns_pending_count": count_vulns_by_status_list(
            focus_host_vulns,
            {"candidate", "validated"},
        ),
        "focus_host_credentials_count": len(get_credentials_for_host(kb, focus_host)),
        "focus_host_sessions_count": len(focus_host_sessions),
        "focus_host_has_session": len(focus_host_sessions) > 0,

        # ========================================================
        # Focus service
        # ========================================================
        "has_focus_service": focus_service is not None,
        "current_service_port": safe_int(focus_service.get("port"), default=0) if focus_service else 0,
        "current_service_name": service_name(focus_service) if focus_service else "",
        "current_service_family": service_family(focus_service) if focus_service else "",
        "current_service_has_version": service_has_version(focus_service),
        "current_service_has_pending_vuln": count_vulns_by_status_list(
            focus_service_vulns,
            {"candidate", "validated"},
        ) > 0,
        "current_service_has_valid_credentials": count_valid_credentials_list(focus_service_credentials) > 0,
        "current_service_has_session": len(focus_service_sessions) > 0,

        # ========================================================
        # Focus vulnerability / session
        # ========================================================
        "has_focus_vulnerability": focus_vuln is not None,
        "focus_vulnerability_status": focus_vuln.get("status", "") if focus_vuln else "",
        "has_focus_session": focus_session is not None,
        "focus_session_status": focus_session.get("status", "") if focus_session else "",
    }

    add_service_presence_flags(state, kb, focus_host, focus_service)
    add_offensive_signature_flags(state, kb, focus_host, focus_service)
    add_action_coverage_flags(state, kb)
    add_host_attack_surface_flags(state, focus_host_ports, focus_host_services)
    add_goal_status_flags(state, goal_type, target_type, focus_host, focus_host_sessions)
    add_attack_candidate_flags(state)

    return state


# ============================================================
# Focus helpers
# ============================================================

def get_focus_host(kb: dict) -> Optional[dict]:
    focus = kb.get("focus", {})
    host_id = focus.get("host_id")

    if not host_id:
        return None

    return kb.get("target", {}).get("hosts", {}).get(host_id)


def get_focus_service(kb: dict) -> Optional[dict]:
    focus = kb.get("focus", {})
    service_id = focus.get("service_id")

    if not service_id:
        return None

    return kb.get("target", {}).get("services", {}).get(service_id)


def get_focus_vulnerability(kb: dict) -> Optional[dict]:
    focus = kb.get("focus", {})
    vulnerability_id = focus.get("vulnerability_id")

    if not vulnerability_id:
        return None

    return kb.get("vulnerabilities", {}).get(vulnerability_id)


def get_focus_session(kb: dict) -> Optional[dict]:
    focus = kb.get("focus", {})
    session_id = focus.get("session_id")

    if not session_id:
        return None

    return kb.get("sessions", {}).get(session_id)


# ============================================================
# Entity collection helpers
# ============================================================

def get_ports_for_host(kb: dict, host: Optional[dict]) -> list:
    if not host:
        return []

    ports = []
    all_ports = kb.get("target", {}).get("ports", {})

    for port_id in host.get("port_ids", []):
        port = all_ports.get(port_id)
        if port:
            ports.append(port)

    if ports:
        return ports

    host_id = host.get("id")
    ip = host.get("ip")

    for port in all_ports.values():
        if port.get("host_id") == host_id or port.get("ip") == ip:
            ports.append(port)

    return ports


def get_services_for_host(kb: dict, host: Optional[dict]) -> list:
    if not host:
        return []

    services = []
    all_services = kb.get("target", {}).get("services", {})

    for service_id in host.get("service_ids", []):
        service = all_services.get(service_id)
        if service:
            services.append(service)

    if services:
        return services

    host_id = host.get("id")
    ip = host.get("ip")

    for service in all_services.values():
        if service.get("host_id") == host_id or service.get("ip") == ip:
            services.append(service)

    return services


def get_vulns_for_host(kb: dict, host: Optional[dict]) -> list:
    if not host:
        return []

    vulns = []
    all_vulns = kb.get("vulnerabilities", {})

    for vuln_id in host.get("vulnerability_ids", []):
        vuln = all_vulns.get(vuln_id)
        if vuln:
            vulns.append(vuln)

    if vulns:
        return vulns

    host_id = host.get("id")
    ip = host.get("ip")

    for vuln in all_vulns.values():
        if vuln.get("host_id") == host_id or vuln.get("host") == ip:
            vulns.append(vuln)

    return vulns


def get_credentials_for_host(kb: dict, host: Optional[dict]) -> list:
    if not host:
        return []

    credentials = []
    all_credentials = kb.get("credentials", {})

    for credential_id in host.get("credential_ids", []):
        credential = all_credentials.get(credential_id)
        if credential:
            credentials.append(credential)

    if credentials:
        return credentials

    host_id = host.get("id")
    ip = host.get("ip")

    for credential in all_credentials.values():
        if credential.get("host_id") == host_id or credential.get("host") == ip:
            credentials.append(credential)

    return credentials


def get_sessions_for_host(kb: dict, host: Optional[dict]) -> list:
    if not host:
        return []

    sessions = []
    all_sessions = kb.get("sessions", {})

    for session_id in host.get("session_ids", []):
        session = all_sessions.get(session_id)
        if session:
            sessions.append(session)

    if sessions:
        return sessions

    host_id = host.get("id")
    ip = host.get("ip")

    for session in all_sessions.values():
        if session.get("host_id") == host_id or session.get("host") == ip:
            sessions.append(session)

    return sessions


def get_vulns_for_service(kb: dict, service: Optional[dict]) -> list:
    if not service:
        return []

    vulns = []
    all_vulns = kb.get("vulnerabilities", {})

    for vuln_id in service.get("vulnerability_ids", []):
        vuln = all_vulns.get(vuln_id)
        if vuln:
            vulns.append(vuln)

    if vulns:
        return vulns

    service_id = service.get("id")

    for vuln in all_vulns.values():
        if vuln.get("service_id") == service_id:
            vulns.append(vuln)

    return vulns


def get_credentials_for_service(kb: dict, service: Optional[dict]) -> list:
    if not service:
        return []

    credentials = []
    all_credentials = kb.get("credentials", {})

    for credential_id in service.get("credential_ids", []):
        credential = all_credentials.get(credential_id)
        if credential:
            credentials.append(credential)

    if credentials:
        return credentials

    service_id = service.get("id")

    for credential in all_credentials.values():
        if credential.get("service_id") == service_id:
            credentials.append(credential)

    return credentials


def get_sessions_for_service(kb: dict, service: Optional[dict]) -> list:
    if not service:
        return []

    sessions = []
    all_sessions = kb.get("sessions", {})

    for session_id in service.get("session_ids", []):
        session = all_sessions.get(session_id)
        if session:
            sessions.append(session)

    if sessions:
        return sessions

    service_id = service.get("id")

    for session in all_sessions.values():
        if session.get("service_id") == service_id:
            sessions.append(session)

    return sessions


# ============================================================
# Service flags
# ============================================================

def add_service_presence_flags(
    state: dict,
    kb: dict,
    focus_host: Optional[dict],
    focus_service: Optional[dict],
) -> None:
    focus_host_services = get_services_for_host(kb, focus_host)

    for flag_name, aliases in SERVICE_FLAGS.items():
        state[f"focus_host_has_{flag_name}"] = any(
            service_matches(service, aliases)
            for service in focus_host_services
        )

        state[f"current_service_is_{flag_name}"] = (
            service_matches(focus_service, aliases)
            if focus_service
            else False
        )


def add_offensive_signature_flags(
    state: dict,
    kb: dict,
    focus_host: Optional[dict],
    focus_service: Optional[dict],
) -> None:
    focus_host_services = get_services_for_host(kb, focus_host)

    state["current_service_is_vsftpd_234"] = service_matches_all(
        focus_service,
        ["vsftpd", "2.3.4"],
    )
    state["focus_host_has_vsftpd_234"] = any(
        service_matches_all(service, ["vsftpd", "2.3.4"])
        for service in focus_host_services
    )

    state["current_service_is_samba"] = service_matches(
        focus_service,
        ["samba", "smb", "netbios", "microsoft-ds"],
    )
    state["focus_host_has_samba"] = any(
        service_matches(service, ["samba", "smb", "netbios", "microsoft-ds"])
        for service in focus_host_services
    )

    state["current_service_is_unreal_ircd"] = service_matches(
        focus_service,
        ["unreal", "unrealircd", "ircd"],
    )
    state["focus_host_has_unreal_ircd"] = any(
        service_matches(service, ["unreal", "unrealircd", "ircd"])
        for service in focus_host_services
    )

    state["current_service_is_ingreslock"] = service_matches(
        focus_service,
        ["ingreslock", "1524"],
    )
    state["focus_host_has_ingreslock"] = any(
        service_matches(service, ["ingreslock", "1524"])
        for service in focus_host_services
    )

    state["current_service_is_postgres_target"] = service_matches(
        focus_service,
        ["postgres", "postgresql"],
    )
    state["current_service_is_distcc_target"] = service_matches(
        focus_service,
        ["distcc", "distccd"],
    )

    state["current_service_is_ftp_weak_creds_target"] = (
        service_matches(focus_service, ["ftp"])
        and safe_int(focus_service.get("port"), default=0) in {21, 2121}
        if focus_service
        else False
    )


def service_matches(service: Optional[dict], aliases: list) -> bool:
    if not service:
        return False

    text = service_text(service)

    for alias in aliases:
        if str(alias).lower() in text:
            return True

    return False


def service_matches_all(service: Optional[dict], aliases: list) -> bool:
    if not service:
        return False

    text = service_text(service)

    for alias in aliases:
        if str(alias).lower() not in text:
            return False

    return True


def service_text(service: Optional[dict]) -> str:
    if not service:
        return ""

    return " ".join([
        str(service.get("name", "")),
        str(service.get("service", "")),
        str(service.get("family", "")),
        str(service.get("product", "")),
        str(service.get("version", "")),
        str(service.get("banner", "")),
        str(service.get("port", "")),
    ]).lower()


def service_name(service: Optional[dict]) -> str:
    if not service:
        return ""

    return (
        service.get("name")
        or service.get("service")
        or ""
    )


def service_family(service: Optional[dict]) -> str:
    if not service:
        return ""

    return (
        service.get("family")
        or service.get("name")
        or service.get("service")
        or ""
    )


def service_has_version(service: Optional[dict]) -> bool:
    if not service:
        return False

    return bool(
        service.get("version")
        or service.get("product")
        or service.get("banner")
    )


# ============================================================
# Action coverage flags
# ============================================================

def add_action_coverage_flags(state: dict, kb: dict) -> None:
    done_actions = collect_done_action_ids(kb)

    # ============================================================
    # Local context actions
    # ============================================================

    state["done_inspect_hostname"] = any_action_done(done_actions, {100})
    state["done_inspect_ip_a"] = any_action_done(done_actions, {101})
    state["done_inspect_ip_r"] = any_action_done(done_actions, {102})
    state["done_inspect_ip_neigh"] = any_action_done(done_actions, {103})

    state["done_local_context"] = (
        state["done_inspect_ip_a"]
        and state["done_inspect_ip_r"]
        and state["done_inspect_ip_neigh"]
    )

    # ============================================================
    # Recon actions
    # ============================================================

    state["done_ping"] = any_action_done(done_actions, {105})
    state["done_host_discovery"] = any_action_done(done_actions, {200})
    state["done_top_portscan"] = any_action_done(done_actions, {210})
    state["done_full_portscan"] = any_action_done(done_actions, {211})
    state["done_service_detection"] = any_action_done(done_actions, {220})

    # ============================================================
    # FTP / VSFTPD enumeration
    # ============================================================

    state["done_ftp_banner"] = any_action_done(done_actions, {330})
    state["done_ftp_anonymous"] = any_action_done(done_actions, {331})
    state["done_ftp_nmap_scripts"] = any_action_done(done_actions, {332})
    state["done_ftp_vuln_check"] = any_action_done(done_actions, {413})

    # ============================================================
    # SMB / Samba enumeration
    # ============================================================

    state["done_smb_shares"] = any_action_done(done_actions, {320})
    state["done_smb_basic_enum"] = any_action_done(done_actions, {321})
    state["done_smb_null_users"] = any_action_done(done_actions, {322})
    state["done_smb_os_discovery"] = any_action_done(done_actions, {323})
    state["done_smb_protocols"] = any_action_done(done_actions, {324})
    state["done_smb_vuln_check"] = any_action_done(done_actions, {410})

    # ============================================================
    # SSH enumeration
    # ============================================================

    state["done_ssh_banner"] = any_action_done(done_actions, {340})
    state["done_ssh_nmap_scripts"] = any_action_done(done_actions, {341})

    # ============================================================
    # PostgreSQL enumeration / credentials
    # ============================================================

    state["done_postgres_info"] = any_action_done(done_actions, {371})
    state["done_postgres_creds_check"] = any_action_done(done_actions, {523})

    # ============================================================
    # Generic vulnerability checks
    # ============================================================

    state["done_service_version_vulns"] = any_action_done(done_actions, {400})
    state["done_nmap_vuln_scripts"] = any_action_done(done_actions, {401})

    # ============================================================
    # Credential attacks
    # ============================================================

    state["done_ssh_creds_manual"] = any_action_done(done_actions, {520})
    state["done_telnet_creds_manual"] = any_action_done(done_actions, {521})
    state["done_ssh_creds_msf"] = any_action_done(done_actions, {611})
    state["done_ftp_creds_msf"] = any_action_done(done_actions, {612})
    state["done_ftp_creds_hydra"] = any_action_done(done_actions, {613})
    state["done_ftp_creds_manual"] = any_action_done(done_actions, {614})

    # ============================================================
    # Exploitation actions
    # ============================================================

    state["done_exploit_samba"] = any_action_done(done_actions, {600})
    state["done_exploit_vsftpd_msf"] = any_action_done(done_actions, {601})
    state["done_exploit_distcc"] = any_action_done(done_actions, {602})
    state["done_exploit_postgres"] = any_action_done(done_actions, {604})
    state["done_exploit_unreal_ircd"] = any_action_done(done_actions, {605})
    state["done_exploit_ingreslock"] = any_action_done(done_actions, {606})
    state["done_exploit_vsftpd_manual"] = any_action_done(done_actions, {610})

    # ============================================================
    # Aggregate flags
    # ============================================================

    # ============================================================
    # Aggregate flags
    # ============================================================

    state["done_any_credential_attack"] = any_action_done(
        done_actions,
        {520, 521, 523, 611, 612, 613, 614},
    )

    state["done_any_exploit"] = any_action_done(
        done_actions,
        {600, 601, 602, 604, 605, 606, 610},
    )

    state["done_any_attack_step"] = any_action_done(
        done_actions,
        {
            330, 331, 332, 413,
            320, 321, 322, 323, 324, 410,
            340, 341,
            371, 523,
            400, 401,
            520, 521, 611, 612, 613, 614,
            600, 601, 602, 604, 605, 606, 610,
        },
    )

def action_done(done_actions: set, action_id: int) -> bool:
    return int(action_id) in done_actions

def collect_done_action_ids(kb: dict) -> set:
    action_ids = set()

    last = kb.get("last", {})
    add_action_id(action_ids, last.get("action_id"))
    add_action_id(action_ids, last.get("executed_action_id"))
    add_action_id(action_ids, last.get("requested_action_id"))

    for event in kb.get("history", []):
        add_action_id(action_ids, event.get("action_id"))
        add_action_id(action_ids, event.get("executed_action_id"))
        add_action_id(action_ids, event.get("requested_action_id"))
        add_action_id(action_ids, event.get("source_action_id"))

    for attempt in kb.get("attempts", {}).values():
        add_action_id(action_ids, attempt.get("action_id"))
        add_action_id(action_ids, attempt.get("executed_action_id"))
        add_action_id(action_ids, attempt.get("requested_action_id"))
    return action_ids


def add_action_id(action_ids: set, value: Any) -> None:
    if value is None:
        return

    try:
        action_ids.add(int(value))
    except (TypeError, ValueError):
        return


def any_action_done(done_actions: set, action_ids: set) -> bool:
    return bool(done_actions.intersection(action_ids))


# ============================================================
# Counters
# ============================================================

def count_attacker_ipv4(attacker: dict) -> int:
    count = 0

    for item in attacker.get("ipv4", []):
        if isinstance(item, dict) and item.get("ip"):
            count += 1
        elif isinstance(item, str) and item:
            count += 1

    for interface in attacker.get("interfaces", []):
        for ipv4 in interface.get("ipv4", []):
            if ipv4.get("ip"):
                count += 1

    return count


def count_alive_hosts(hosts: dict) -> int:
    total = 0

    for host in hosts.values():
        if host.get("alive") is True:
            total += 1

    return total


def count_open_ports(ports: dict) -> int:
    total = 0

    for port in ports.values():
        if port.get("state") == "open":
            total += 1

    return total


def count_open_ports_list(ports: list) -> int:
    total = 0

    for port in ports:
        if port.get("state") == "open":
            total += 1

    return total


def count_services_with_version(services: dict) -> int:
    total = 0

    for service in services.values():
        if service_has_version(service):
            total += 1

    return total


def count_services_with_version_list(services: list) -> int:
    total = 0

    for service in services:
        if service_has_version(service):
            total += 1

    return total


def count_vulns_by_status(vulns: dict, statuses: set) -> int:
    total = 0

    for vuln in vulns.values():
        if vuln.get("status") in statuses:
            total += 1

    return total


def count_vulns_by_status_list(vulns: list, statuses: set) -> int:
    total = 0

    for vuln in vulns:
        if vuln.get("status") in statuses:
            total += 1

    return total


def count_valid_credentials(credentials: dict) -> int:
    total = 0

    for credential in credentials.values():
        if credential.get("valid") is True:
            total += 1

    return total


def count_valid_credentials_list(credentials: list) -> int:
    total = 0

    for credential in credentials:
        if credential.get("valid") is True:
            total += 1

    return total


def count_open_sessions(sessions: dict) -> int:
    total = 0

    for session in sessions.values():
        if session.get("status") in {"opened", "open", "active"}:
            total += 1

    return total


# ============================================================
# Pending work counters
# ============================================================

def count_pending_hosts_portscan(kb: dict) -> int:
    coverage = kb.get("coverage", {})
    scanned = set(coverage.get("hosts_port_scanned", []))

    total = 0

    for host_id, host in kb.get("target", {}).get("hosts", {}).items():
        if host.get("alive") is False:
            continue

        if host_id not in scanned:
            total += 1

    return total


def count_pending_hosts_servicescan(kb: dict) -> int:
    coverage = kb.get("coverage", {})
    scanned = set(coverage.get("hosts_service_scanned", []))

    total = 0

    for host_id, host in kb.get("target", {}).get("hosts", {}).items():
        if host.get("alive") is False:
            continue

        if not host.get("port_ids"):
            continue

        if host_id not in scanned:
            total += 1

    return total


def count_pending_services_enum(kb: dict) -> int:
    coverage = kb.get("coverage", {})
    enumerated = set(coverage.get("services_enumerated", []))

    total = 0

    for service_id in kb.get("target", {}).get("services", {}):
        if service_id not in enumerated:
            total += 1

    return total


def count_pending_vulns(kb: dict) -> int:
    total = 0

    attempted = set(kb.get("coverage", {}).get("vulns_attempted", []))

    for vuln_id, vuln in kb.get("vulnerabilities", {}).items():
        if vuln.get("status") not in {"candidate", "validated"}:
            continue

        if vuln_id in attempted:
            continue

        if vuln.get("attempt_ids"):
            continue

        total += 1

    return total


# ============================================================
# Generic helpers
# ============================================================

def first_event_type(event_types: Any) -> str:
    if isinstance(event_types, list) and event_types:
        return str(event_types[0])

    if isinstance(event_types, str):
        return event_types

    return ""


def safe_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default

    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Alias por compatibilidad
build_state_from_kb = build_state


def add_host_attack_surface_flags(
    state: dict,
    focus_host_ports: list,
    focus_host_services: list,
) -> None:
    open_ports = set()

    for port in focus_host_ports:
        if port.get("state") != "open":
            continue

        port_number = safe_int(port.get("port"), default=0)

        if port_number > 0:
            open_ports.add(port_number)

    state["host_has_port_21"] = 21 in open_ports
    state["host_has_port_22"] = 22 in open_ports
    state["host_has_port_23"] = 23 in open_ports
    state["host_has_port_139"] = 139 in open_ports
    state["host_has_port_445"] = 445 in open_ports
    state["host_has_port_1524"] = 1524 in open_ports
    state["host_has_port_2121"] = 2121 in open_ports
    state["host_has_port_3632"] = 3632 in open_ports
    state["host_has_port_5432"] = 5432 in open_ports
    state["host_has_port_6667"] = 6667 in open_ports

    state["host_has_vsftpd_234"] = any(
        service_matches_all(service, ["vsftpd", "2.3.4"])
        for service in focus_host_services
    )

    state["host_has_samba"] = any(
        service_matches(service, ["samba", "smb", "netbios", "microsoft-ds"])
        for service in focus_host_services
    )

    state["host_has_distcc"] = any(
        service_matches(service, ["distcc", "distccd"])
        for service in focus_host_services
    )

    state["host_has_postgres"] = any(
        service_matches(service, ["postgres", "postgresql"])
        for service in focus_host_services
    )

    state["host_has_unreal_ircd"] = any(
        service_matches(service, ["unreal", "unrealircd", "ircd"])
        for service in focus_host_services
    )

    state["host_has_ingreslock"] = any(
        service_matches(service, ["ingreslock", "1524"])
        for service in focus_host_services
    )

    state["host_has_ssh"] = any(
        service_matches(service, ["ssh", "openssh"])
        for service in focus_host_services
    )

    state["host_has_telnet"] = any(
        service_matches(service, ["telnet"])
        for service in focus_host_services
    )

    state["host_has_ftp"] = any(
        service_matches(service, ["ftp", "vsftpd", "proftpd"])
        for service in focus_host_services
    )

    state["host_has_ftp_weak_creds_surface"] = (
        state["host_has_port_21"]
        or state["host_has_port_2121"]
        or state["host_has_ftp"]
    )


def add_goal_status_flags(
    state: dict,
    goal_type: str,
    target_type: str,
    focus_host: Optional[dict],
    focus_host_sessions: list,
) -> None:
    has_open_session_for_focus_host = False

    for session in focus_host_sessions:
        if session.get("status") in {"opened", "open", "active"}:
            has_open_session_for_focus_host = True
            break

    state["has_open_session_for_focus_host"] = has_open_session_for_focus_host

    if goal_type == "obtain_session":
        state["goal_obtained"] = has_open_session_for_focus_host or state.get("has_session", False)
    elif goal_type == "full_exploit":
        state["goal_obtained"] = state.get("done_any_exploit", False) and state.get("has_session", False)
    else:
        state["goal_obtained"] = False

    state["should_stop_now"] = state["goal_obtained"]


def add_attack_candidate_flags(state: dict) -> None:
    service_detection_done = state.get("done_service_detection", False)

    state["can_try_vsftpd_msf"] = (
        service_detection_done
        and state.get("host_has_vsftpd_234", False)
        and not state.get("done_exploit_vsftpd_msf", False)
        and not state.get("has_session", False)
    )

    state["can_try_vsftpd_manual"] = (
        service_detection_done
        and state.get("host_has_vsftpd_234", False)
        and not state.get("done_exploit_vsftpd_manual", False)
        and not state.get("has_session", False)
    )

    state["can_try_samba_usermap"] = (
        service_detection_done
        and state.get("host_has_samba", False)
        and (state.get("host_has_port_139", False) or state.get("host_has_port_445", False))
        and not state.get("done_exploit_samba", False)
        and not state.get("has_session", False)
    )

    state["can_try_distcc"] = (
        service_detection_done
        and state.get("host_has_distcc", False)
        and state.get("host_has_port_3632", False)
        and not state.get("done_exploit_distcc", False)
        and not state.get("has_session", False)
    )

    state["can_try_postgres"] = (
        service_detection_done
        and state.get("host_has_postgres", False)
        and state.get("host_has_port_5432", False)
        and not state.get("done_exploit_postgres", False)
        and not state.get("has_session", False)
    )

    state["can_try_unreal_ircd"] = (
        service_detection_done
        and state.get("host_has_unreal_ircd", False)
        and state.get("host_has_port_6667", False)
        and not state.get("done_exploit_unreal_ircd", False)
        and not state.get("has_session", False)
    )

    state["can_try_ingreslock"] = (
        service_detection_done
        and state.get("host_has_ingreslock", False)
        and state.get("host_has_port_1524", False)
        and not state.get("done_exploit_ingreslock", False)
        and not state.get("has_session", False)
    )

    state["can_try_ssh_creds"] = (
        service_detection_done
        and state.get("host_has_ssh", False)
        and state.get("host_has_port_22", False)
        and not state.get("done_ssh_creds_manual", False)
        and not state.get("done_ssh_creds_msf", False)
        and not state.get("has_session", False)
    )

    state["can_try_telnet_creds"] = (
        service_detection_done
        and state.get("host_has_telnet", False)
        and state.get("host_has_port_23", False)
        and not state.get("done_telnet_creds_manual", False)
        and not state.get("has_session", False)
    )

    state["can_try_ftp_creds"] = (
        service_detection_done
        and state.get("host_has_ftp_weak_creds_surface", False)
        and not state.get("done_ftp_creds_msf", False)
        and not state.get("done_ftp_creds_hydra", False)
        and not state.get("done_ftp_creds_manual", False)
        and not state.get("has_session", False)
    )