# kb_updater.py
#
# Recibe eventos normalizados, decide qué entidades de la KB actualizar
# y mantiene las relaciones entre redes, hosts, puertos, servicios,
# vulnerabilidades, credenciales, sesiones, intentos, evidencias y findings.

import json
from pathlib import Path
from typing import Any, Optional

from penhackit.session.kb.kb_schema import (
    ensure_kb_collections, refresh_stats, make_network, make_host, make_port, make_service, make_vulnerability, 
    make_credential, make_session, make_attempt, make_evidence,make_finding, make_network_id, make_host_id, 
    make_port_id, make_service_id, make_vulnerability_id, make_credential_id, make_attempt_id, make_step
)


def update_kb(kb: dict, events: list[dict]) -> dict:
    """
    Actualiza la KB a partir de una lista de eventos normalizados.

    La KB v2 usa colecciones relacionales:
    - kb["target"]["networks"]
    - kb["target"]["hosts"]
    - kb["target"]["ports"]
    - kb["target"]["services"]
    - kb["vulnerabilities"]
    - kb["credentials"]
    - kb["sessions"]
    - kb["attempts"]
    - kb["evidence"]
    - kb["findings"]

    El stdout/stderr completo no debe guardarse aquí, sino en logs.
    """
    print("Updating KB with new events...")

    ensure_kb_collections(kb)

    for ev in events:
        et = ev.get("type")

        if et == "HOST_DISCOVERED":
            _handle_host_discovered(kb, ev)

        elif et == "PORT_OPEN":
            _handle_port_open(kb, ev)

        elif et == "SERVICE_DETECTED":
            _handle_service_detected(kb, ev)

        elif et == "SERVICE_VERSION_DETECTED":
            _handle_service_version_detected(kb, ev)

        elif et == "OS_GUESS_DETECTED":
            _handle_os_guess_detected(kb, ev)

        elif et == "CANDIDATE_VULN_FOUND":
            _handle_candidate_vuln_found(kb, ev)

        elif et == "VULN_VALIDATED":
            _handle_vuln_validated(kb, ev)

        elif et == "VULN_REJECTED":
            _handle_vuln_rejected(kb, ev)

        elif et in {"VALID_CREDENTIAL_FOUND", "LOGIN_SUCCESS"}:
            _handle_valid_credential_found(kb, ev)

        elif et == "LOGIN_FAILED":
            _handle_login_failed(kb, ev)

        elif et == "EXPLOIT_ATTEMPTED":
            _handle_exploit_attempted(kb, ev)

        elif et in {"EXPLOIT_FAILED", "SESSION_NOT_CREATED"}:
            _handle_exploit_failed(kb, ev)

        elif et == "SESSION_OPENED":
            _handle_session_opened(kb, ev)

        elif et == "SESSION_CLOSED":
            _handle_session_closed(kb, ev)

        elif et == "SESSION_USER_DETECTED":
            _handle_session_field_detected(kb, ev, "user")

        elif et == "SESSION_PRIVILEGES_DETECTED":
            _handle_session_field_detected(kb, ev, "privilege")

        elif et == "SESSION_HOSTNAME_DETECTED":
            _handle_session_field_detected(kb, ev, "hostname")

        elif et == "SESSION_SYSTEM_DETECTED":
            _handle_session_field_detected(kb, ev, "system")

        elif et == "NET_INFO":
            _handle_net_info(kb, ev)

        elif et == "ROUTE_TABLE":
            _handle_route_table(kb, ev)

        elif et == "ARP_TABLE":
            _handle_arp_table(kb, ev)

        elif et == "HOST_UNREACHABLE":
            _handle_host_unreachable(kb, ev)

        elif et == "SUBNET_SCAN_COMPLETED":
            _handle_subnet_scan_completed(kb, ev)

        elif et in {
            "COMMAND_ERROR",
            "TOOL_ERROR",
            "TIMEOUT",
            "NO_MEANINGFUL_OUTPUT",
            "NO_COMMAND_EXECUTED",
            "NO_EVENT",
            "ACTION_COMPLETED",
        }:
            _append_history(kb, ev)

        else:
            _append_history(kb, ev)

        _update_last_from_event(kb, ev)

    refresh_stats(kb)
    return kb


# ============================================================
# Event handlers
# ============================================================

def _handle_host_discovered(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    if not ip:
        _append_history(kb, ev)
        return

    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    host = ensure_host(kb, cidr, ip)
    host["alive"] = True

    _append_unique(kb["coverage"]["hosts_discovered"], host["id"])

    network_id = host.get("network_id")
    if network_id:
        network = kb["target"]["networks"].get(network_id)
        if network:
            network["discovered"] = True

    _set_focus_host(kb, host["id"])
    _append_history(kb, ev)


def _handle_host_unreachable(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    if not ip:
        _append_history(kb, ev)
        return

    cidr = ev.get("network") or infer_network_for_host(kb, ip)
    host = ensure_host(kb, cidr, ip)
    host["alive"] = False

    _append_history(kb, ev)


def _handle_port_open(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    if not ip or port is None:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    port_obj = ensure_port(kb, cidr, ip, int(port), protocol)
    port_obj["state"] = "open"

    service_name = ev.get("service")
    if service_name:
        service = ensure_service(kb, cidr, ip, int(port), protocol)
        service["name"] = service_name
        service["family"] = service.get("family") or service_name.lower()

    _append_unique(kb["coverage"]["hosts_port_scanned"], make_host_id(ip))
    _set_focus_service(kb, make_service_id(ip, int(port), protocol))
    _append_history(kb, ev)


def _handle_service_detected(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    if not ip or port is None:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    service = ensure_service(kb, cidr, ip, int(port), protocol)

    service_name = ev.get("service")
    if service_name:
        service["name"] = service_name
        service["family"] = service.get("family") or service_name.lower()

    _append_unique(kb["coverage"]["hosts_service_scanned"], make_host_id(ip))
    _set_focus_service(kb, service["id"])
    _append_history(kb, ev)


def _handle_service_version_detected(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    if not ip or port is None:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    service = ensure_service(kb, cidr, ip, int(port), protocol)

    service["name"] = ev.get("service") or service.get("name")
    service["product"] = ev.get("product") or service.get("product")
    service["version"] = ev.get("version") or service.get("version")
    service["banner"] = ev.get("banner") or service.get("banner")

    if service.get("name"):
        service["family"] = service.get("family") or service["name"].lower()

    _append_unique(kb["coverage"]["services_enumerated"], service["id"])
    _set_focus_service(kb, service["id"])
    _append_history(kb, ev)


def _handle_os_guess_detected(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    if not ip:
        _append_history(kb, ev)
        return

    cidr = ev.get("network") or infer_network_for_host(kb, ip)
    host = ensure_host(kb, cidr, ip)

    host["os"] = ev.get("os") or ev.get("os_guess") or host.get("os")

    _append_history(kb, ev)


def _handle_candidate_vuln_found(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    vuln_name = (
        ev.get("vuln")
        or ev.get("exploit")
        or ev.get("name")
        or ev.get("title")
        or "unknown_vulnerability"
    )

    if not ip:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    host = ensure_host(kb, cidr, ip)

    service_id = None
    if port is not None:
        service = ensure_service(kb, cidr, ip, int(port), protocol)
        service_id = service["id"]

    vuln = make_vulnerability(
        name=vuln_name,
        host=ip,
        port=int(port) if port is not None else None,
        protocol=protocol,
        service_id=service_id,
        status="candidate",
        source=ev.get("source"),
        confidence=ev.get("confidence") or "medium",
    )

    vuln_id = vuln["id"]
    existing = kb["vulnerabilities"].get(vuln_id, {})
    existing.update({k: v for k, v in vuln.items() if v is not None})
    kb["vulnerabilities"][vuln_id] = existing or vuln

    _append_unique(host["vulnerability_ids"], vuln_id)

    if service_id:
        service_obj = kb["target"]["services"].get(service_id)
        if service_obj:
            _append_unique(service_obj["vulnerability_ids"], vuln_id)
            _append_unique(kb["coverage"]["services_checked_for_vulns"], service_id)

    _set_focus_vulnerability(kb, vuln_id)
    _append_history(kb, ev)


def _handle_vuln_validated(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    vuln_name = (
        ev.get("vuln")
        or ev.get("exploit")
        or ev.get("name")
        or ev.get("title")
        or "unknown_vulnerability"
    )

    if not ip:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    vuln_id = make_vulnerability_id(
        vuln_name,
        ip,
        int(port) if port is not None else None,
        protocol,
    )

    if vuln_id not in kb["vulnerabilities"]:
        _handle_candidate_vuln_found(kb, ev)

    vuln = kb["vulnerabilities"].get(vuln_id)
    if vuln:
        vuln["status"] = "validated"
        vuln["confidence"] = ev.get("confidence") or "high"

    finding = make_finding(
        title=ev.get("title") or vuln_name,
        severity=ev.get("severity") or "high",
        host=ip,
        port=int(port) if port is not None else None,
        service=ev.get("service"),
        source=ev.get("source"),
        status="confirmed",
        finding_type="vulnerability",
    )

    finding_id = finding["id"]
    kb["findings"][finding_id] = finding

    if vuln:
        _append_unique(vuln["finding_ids"], finding_id)

    _link_finding_to_host_service(kb, finding_id)
    _set_focus_vulnerability(kb, vuln_id)
    _append_history(kb, ev)


def _handle_vuln_rejected(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    vuln_name = (
        ev.get("vuln")
        or ev.get("exploit")
        or ev.get("name")
        or ev.get("title")
        or "unknown_vulnerability"
    )

    if not ip:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    vuln_id = make_vulnerability_id(
        vuln_name,
        ip,
        int(port) if port is not None else None,
        protocol,
    )

    if vuln_id in kb["vulnerabilities"]:
        kb["vulnerabilities"][vuln_id]["status"] = "rejected"

    _append_history(kb, ev)


def _handle_valid_credential_found(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")
    username = ev.get("username")
    password = ev.get("password")

    if not ip or port is None or username is None:
        _append_history(kb, ev)
        return

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    service_name = ev.get("service") or "unknown"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    host = ensure_host(kb, cidr, ip)
    service = ensure_service(kb, cidr, ip, int(port), protocol)

    if service_name != "unknown":
        service["name"] = service_name
        service["family"] = service.get("family") or service_name.lower()

    credential = make_credential(
        username=username,
        password=password,
        service=service_name,
        host=ip,
        port=int(port),
        source=ev.get("source"),
        protocol=protocol,
        valid=True,
    )

    credential_id = credential["id"]
    kb["credentials"][credential_id] = credential

    _append_unique(host["credential_ids"], credential_id)
    _append_unique(service["credential_ids"], credential_id)
    _append_unique(kb["coverage"]["credentials_tested"], credential_id)

    finding = make_finding(
        title=ev.get("title") or f"Valid credentials for {service_name}",
        severity=ev.get("severity") or "high",
        host=ip,
        port=int(port),
        service=service_name,
        source=ev.get("source"),
        status="confirmed",
        finding_type="credential",
    )

    finding_id = finding["id"]
    kb["findings"][finding_id] = finding
    _append_unique(finding["credential_ids"], credential_id)

    _append_unique(credential["finding_ids"], finding_id)
    _link_finding_to_host_service(kb, finding_id)

    _set_focus_credential(kb, credential_id)
    _append_history(kb, ev)


def _handle_login_failed(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")
    username = ev.get("username")

    if ip and port is not None and username:
        service_name = ev.get("service") or "unknown"
        credential_id = make_credential_id(ip, service_name, username, int(port))
        _append_unique(kb["coverage"]["credentials_tested"], credential_id)

    _append_history(kb, ev)


def _handle_exploit_attempted(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")
    exploit = ev.get("exploit")

    step_idx = _event_step_idx(kb, ev)
    action_id = _event_action_id(ev)
    action_name = ev.get("action_name") or exploit or "EXPLOIT_ATTEMPTED"

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"

    host_id = make_host_id(ip) if ip else None
    service_id = make_service_id(ip, int(port), protocol) if ip and port is not None else None
    vuln_id = None

    if ip and exploit:
        vuln_id = make_vulnerability_id(
            exploit,
            ip,
            int(port) if port is not None else None,
            protocol,
        )

    attempt = make_attempt(
        step_idx=step_idx,
        action_id=action_id,
        action_name=action_name,
        phase=ev.get("phase") or "exploit",
        target_id=service_id or host_id,
        host_id=host_id,
        service_id=service_id,
        vulnerability_id=vuln_id,
        credential_id=ev.get("credential_id"),
        command=ev.get("command"),
        rc=ev.get("rc"),
        success=None,
        event_types=[ev.get("type")],
    )

    attempt_id = attempt["id"]
    kb["attempts"][attempt_id] = attempt

    if ip and port is not None:
        cidr = ev.get("network") or infer_network_for_host(kb, ip)
        host = ensure_host(kb, cidr, ip)
        service = ensure_service(kb, cidr, ip, int(port), protocol)

        _append_unique(host["attempt_ids"], attempt_id)
        _append_unique(service["attempt_ids"], attempt_id)
        _append_unique(kb["coverage"]["exploits_attempted"], attempt_id)

    if vuln_id and vuln_id in kb["vulnerabilities"]:
        _append_unique(kb["vulnerabilities"][vuln_id]["attempt_ids"], attempt_id)
        _append_unique(kb["coverage"]["vulns_attempted"], vuln_id)

    kb["stats"]["exploits_attempted"] = kb["stats"].get("exploits_attempted", 0) + 1

    _append_history(kb, ev)


def _handle_exploit_failed(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")
    exploit = ev.get("exploit")

    step_idx = _event_step_idx(kb, ev)
    action_id = _event_action_id(ev)
    action_name = ev.get("action_name") or exploit or ev.get("type") or "EXPLOIT_FAILED"

    protocol = ev.get("proto") or ev.get("protocol") or "tcp"

    host_id = make_host_id(ip) if ip else None
    service_id = make_service_id(ip, int(port), protocol) if ip and port is not None else None
    vuln_id = None

    if ip and exploit:
        vuln_id = make_vulnerability_id(
            exploit,
            ip,
            int(port) if port is not None else None,
            protocol,
        )

    attempt = make_attempt(
        step_idx=step_idx,
        action_id=action_id,
        action_name=action_name,
        phase=ev.get("phase") or "exploit",
        target_id=service_id or host_id,
        host_id=host_id,
        service_id=service_id,
        vulnerability_id=vuln_id,
        credential_id=ev.get("credential_id"),
        command=ev.get("command"),
        rc=ev.get("rc"),
        success=False,
        event_types=[ev.get("type")],
    )

    attempt["error"] = ev.get("error")

    attempt_id = attempt["id"]
    kb["attempts"][attempt_id] = attempt

    if ip and port is not None:
        cidr = ev.get("network") or infer_network_for_host(kb, ip)
        host = ensure_host(kb, cidr, ip)
        service = ensure_service(kb, cidr, ip, int(port), protocol)

        _append_unique(host["attempt_ids"], attempt_id)
        _append_unique(service["attempt_ids"], attempt_id)

    if vuln_id and vuln_id in kb["vulnerabilities"]:
        _append_unique(kb["vulnerabilities"][vuln_id]["attempt_ids"], attempt_id)

    kb["stats"]["exploits_failed"] = kb["stats"].get("exploits_failed", 0) + 1

    _append_history(kb, ev)


def _handle_session_opened(kb: dict, ev: dict) -> None:
    ip = ev.get("host")
    port = ev.get("port")

    if not ip or port is None:
        _append_history(kb, ev)
        return

    step_idx = _event_step_idx(kb, ev)
    protocol = ev.get("proto") or ev.get("protocol") or "tcp"
    cidr = ev.get("network") or infer_network_for_host(kb, ip)

    host = ensure_host(kb, cidr, ip)
    service = ensure_service(kb, cidr, ip, int(port), protocol)

    exploit = ev.get("exploit")
    vulnerability_id = None
    if exploit:
        vulnerability_id = make_vulnerability_id(exploit, ip, int(port), protocol)

    attempt_id = _find_recent_attempt_for_service(kb, service["id"], exploit)

    session = make_session(
        session_type=ev.get("session_type") or "shell",
        host=ip,
        port=int(port),
        service=service.get("name") or ev.get("service") or "unknown",
        tool=ev.get("source") or ev.get("tool"),
        step_idx=step_idx,
        protocol=protocol,
        user=ev.get("user"),
        privilege=ev.get("privilege"),
        status="opened",
        vulnerability_id=vulnerability_id,
        credential_id=ev.get("credential_id"),
        attempt_id=attempt_id,
    )

    external_session_id = ev.get("session_id")
    if external_session_id is not None:
        session["external_id"] = external_session_id
        session["id"] = f"sess:{external_session_id}"

    session_id = session["id"]
    kb["sessions"][session_id] = session

    _append_unique(host["session_ids"], session_id)
    _append_unique(service["session_ids"], session_id)
    _append_unique(kb["coverage"]["sessions_validated"], session_id)

    if attempt_id and attempt_id in kb["attempts"]:
        kb["attempts"][attempt_id]["success"] = True
        _append_unique(kb["attempts"][attempt_id]["event_types"], ev.get("type"))

    if vulnerability_id and vulnerability_id in kb["vulnerabilities"]:
        kb["vulnerabilities"][vulnerability_id]["status"] = "validated"
        _append_unique(kb["vulnerabilities"][vulnerability_id]["attempt_ids"], attempt_id)
        _append_unique(kb["vulnerabilities"][vulnerability_id]["finding_ids"], session_id)

    evidence = make_evidence(
        step_idx=step_idx,
        kind="session",
        title="Session opened",
        summary=ev.get("summary"),
        host_id=host["id"],
        service_id=service["id"],
        attempt_id=attempt_id,
        suffix=session_id.replace(":", "_"),
    )

    evidence_id = evidence["id"]
    kb["evidence"][evidence_id] = evidence

    _append_unique(session["evidence_ids"], evidence_id)
    _append_unique(host["evidence_ids"], evidence_id)
    _append_unique(service["evidence_ids"], evidence_id)

    finding = make_finding(
        title=ev.get("title") or f"Session opened via {exploit or service.get('name') or 'service'}",
        severity=ev.get("severity") or "critical",
        host=ip,
        port=int(port),
        service=service.get("name"),
        source=ev.get("source"),
        status="confirmed",
        finding_type="session",
    )

    finding_id = finding["id"]
    kb["findings"][finding_id] = finding

    _append_unique(finding["session_ids"], session_id)
    _append_unique(finding["evidence_ids"], evidence_id)

    if vulnerability_id:
        _append_unique(finding["vulnerability_ids"], vulnerability_id)

    _append_unique(session["finding_ids"], finding_id)
    _append_unique(host["finding_ids"], finding_id)
    _append_unique(service["finding_ids"], finding_id)

    _set_focus_session(kb, session_id)
    _append_history(kb, ev)


def _handle_session_closed(kb: dict, ev: dict) -> None:
    session_id = ev.get("session_id")

    if session_id is None:
        _append_history(kb, ev)
        return

    candidates = [
        str(session_id),
        f"sess:{session_id}",
    ]

    step_idx = _event_step_idx(kb, ev)

    for candidate in candidates:
        session = kb["sessions"].get(candidate)
        if session:
            session["status"] = "closed"
            session["closed_at_step"] = step_idx

    _append_history(kb, ev)


def _handle_session_field_detected(kb: dict, ev: dict, field_name: str) -> None:
    session_id = ev.get("session_id")
    value = ev.get(field_name)

    if session_id is None or value is None:
        _append_history(kb, ev)
        return

    candidates = [
        str(session_id),
        f"sess:{session_id}",
    ]

    for candidate in candidates:
        session = kb["sessions"].get(candidate)
        if session:
            session[field_name] = value

    _append_history(kb, ev)


def _handle_net_info(kb: dict, ev: dict) -> None:
    attacker = kb["attacker"]

    hostname = ev.get("hostname")
    if hostname:
        attacker["hostname"] = hostname

    for iface in ev.get("interfaces", []):
        _append_unique(attacker["interfaces"], iface)

    for ip in ev.get("ipv4", []):
        if ip:
            _append_unique(attacker["ipv4"], ip)

    for gw in ev.get("default_gw", []):
        if gw:
            _append_unique(attacker["default_gw"], gw)

    lhost = ev.get("lhost")
    if lhost:
        attacker["lhost"] = lhost

    _append_history(kb, ev)


def _handle_route_table(kb: dict, ev: dict) -> None:
    attacker = kb["attacker"]

    for route in ev.get("routes", []):
        _append_unique(attacker["routes"], route)

    _append_history(kb, ev)


def _handle_arp_table(kb: dict, ev: dict) -> None:
    attacker = kb["attacker"]

    for neighbor in ev.get("arp_neighbors", []):
        _append_unique(attacker["arp_neighbors"], neighbor)

        ip = neighbor.get("ip")
        if not ip:
            continue

        if not _should_promote_arp_neighbor_to_target_host(kb, ip):
            continue

        cidr = ev.get("network") or infer_network_for_host(kb, ip)
        host = ensure_host(kb, cidr, ip)

        host["alive"] = True

        mac = neighbor.get("mac")
        if mac:
            host["mac"] = mac

        dev = neighbor.get("dev")
        if dev:
            host["interface"] = dev

        state = neighbor.get("state")
        if state:
            host["arp_state"] = state

        _append_unique(kb["coverage"]["hosts_discovered"], host["id"])

    _append_history(kb, ev)


def _handle_subnet_scan_completed(kb: dict, ev: dict) -> None:
    cidr = ev.get("network") or ev.get("cidr")

    if cidr:
        network = ensure_network(kb, cidr)
        network["host_discovery_done"] = True
        _append_unique(kb["coverage"]["networks_scanned"], network["id"])

    _append_history(kb, ev)


# ============================================================
# Ensure / upsert helpers
# ============================================================

def ensure_network(kb: dict, cidr: str) -> dict:
    ensure_kb_collections(kb)

    network_id = make_network_id(cidr)
    networks = kb["target"]["networks"]

    if network_id not in networks:
        networks[network_id] = make_network(cidr)

    return networks[network_id]


def ensure_host(kb: dict, cidr: str, ip: str) -> dict:
    ensure_kb_collections(kb)

    network = ensure_network(kb, cidr)
    host_id = make_host_id(ip)

    hosts = kb["target"]["hosts"]

    if host_id not in hosts:
        hosts[host_id] = make_host(ip, network_id=network["id"])

    host = hosts[host_id]

    if not host.get("network_id"):
        host["network_id"] = network["id"]

    _append_unique(network["host_ids"], host_id)
    _append_unique(kb["coverage"]["hosts_discovered"], host_id)

    return host


def ensure_port(
    kb: dict,
    cidr: str,
    ip: str,
    port: int,
    protocol: str = "tcp",
) -> dict:
    ensure_kb_collections(kb)

    host = ensure_host(kb, cidr, ip)
    port_id = make_port_id(ip, port, protocol)

    ports = kb["target"]["ports"]

    if port_id not in ports:
        ports[port_id] = make_port(ip, port, protocol)

    port_obj = ports[port_id]

    _append_unique(host["port_ids"], port_id)

    return port_obj


def ensure_service(
    kb: dict,
    cidr: str,
    ip: str,
    port: int,
    protocol: str = "tcp",
) -> dict:
    ensure_kb_collections(kb)

    host = ensure_host(kb, cidr, ip)
    port_obj = ensure_port(kb, cidr, ip, port, protocol)

    service_id = make_service_id(ip, port, protocol)
    services = kb["target"]["services"]

    if service_id not in services:
        services[service_id] = make_service(ip, port, protocol)

    service = services[service_id]

    port_obj["service_id"] = service_id

    _append_unique(host["service_ids"], service_id)

    return service


# ============================================================
# Focus helpers
# ============================================================

def _set_focus_host(kb: dict, host_id: str) -> None:
    kb["focus"]["level"] = "host"
    kb["focus"]["host_id"] = host_id
    kb["focus"]["network_id"] = kb["target"]["hosts"].get(host_id, {}).get("network_id")
    kb["focus"]["port_id"] = None
    kb["focus"]["service_id"] = None
    kb["focus"]["vulnerability_id"] = None
    kb["focus"]["credential_id"] = None
    kb["focus"]["session_id"] = None


def _set_focus_service(kb: dict, service_id: str) -> None:
    service = kb["target"]["services"].get(service_id, {})

    kb["focus"]["level"] = "service"
    kb["focus"]["network_id"] = None
    kb["focus"]["host_id"] = service.get("host_id")
    kb["focus"]["port_id"] = service.get("port_id")
    kb["focus"]["service_id"] = service_id
    kb["focus"]["vulnerability_id"] = None
    kb["focus"]["credential_id"] = None
    kb["focus"]["session_id"] = None


def _set_focus_vulnerability(kb: dict, vulnerability_id: str) -> None:
    vuln = kb["vulnerabilities"].get(vulnerability_id, {})

    kb["focus"]["level"] = "vuln"
    kb["focus"]["network_id"] = None
    kb["focus"]["host_id"] = vuln.get("host_id")
    kb["focus"]["port_id"] = None
    kb["focus"]["service_id"] = vuln.get("service_id")
    kb["focus"]["vulnerability_id"] = vulnerability_id
    kb["focus"]["credential_id"] = None
    kb["focus"]["session_id"] = None


def _set_focus_credential(kb: dict, credential_id: str) -> None:
    cred = kb["credentials"].get(credential_id, {})

    kb["focus"]["level"] = "service"
    kb["focus"]["network_id"] = None
    kb["focus"]["host_id"] = cred.get("host_id")
    kb["focus"]["port_id"] = None
    kb["focus"]["service_id"] = cred.get("service_id")
    kb["focus"]["vulnerability_id"] = None
    kb["focus"]["credential_id"] = credential_id
    kb["focus"]["session_id"] = None


def _set_focus_session(kb: dict, session_id: str) -> None:
    session = kb["sessions"].get(session_id, {})

    kb["focus"]["level"] = "session"
    kb["focus"]["network_id"] = None
    kb["focus"]["host_id"] = session.get("host_id")
    kb["focus"]["port_id"] = None
    kb["focus"]["service_id"] = session.get("service_id")
    kb["focus"]["vulnerability_id"] = session.get("vulnerability_id")
    kb["focus"]["credential_id"] = session.get("credential_id")
    kb["focus"]["session_id"] = session_id


# ============================================================
# Generic helpers
# ============================================================

def infer_network_for_host(kb: dict, ip: str) -> str:
    scope = kb.get("scope", {})
    target = scope.get("target")
    target_type = scope.get("target_type")

    if target_type == "network" and target:
        return target

    if target_type == "host":
        return f"{ip}/24"

    return "unknown"


def _append_history(kb: dict, ev: dict) -> None:
    kb.setdefault("history", [])
    kb["history"].append(ev)


def _append_unique(items: list, value: Any) -> None:
    if value is None:
        return

    if value not in items:
        items.append(value)


def _event_step_idx(kb: dict, ev: dict) -> int:
    step_idx = ev.get("step_idx")

    if step_idx is not None:
        try:
            return int(step_idx)
        except ValueError:
            pass

    last_step = kb.get("last", {}).get("step_idx")
    if last_step is not None:
        try:
            return int(last_step)
        except ValueError:
            pass

    return len(kb.get("history", [])) + 1


def _event_action_id(ev: dict) -> int:
    action_id = ev.get("action_id")

    if action_id is None:
        return -1

    try:
        return int(action_id)
    except ValueError:
        return -1


def _update_last_from_event(kb: dict, ev: dict) -> None:
    kb.setdefault("last", {})

    step_idx = ev.get("step_idx")
    if step_idx is not None:
        kb["last"]["step_idx"] = step_idx

    if ev.get("action_id") is not None:
        kb["last"]["action_id"] = ev.get("action_id")

    if ev.get("action_name") is not None:
        kb["last"]["action_name"] = ev.get("action_name")

    if ev.get("rc") is not None:
        kb["last"]["rc"] = ev.get("rc")

    if ev.get("success") is not None:
        kb["last"]["success"] = ev.get("success")

    et = ev.get("type")
    if et:
        kb["last"]["event_types"] = [et]

    if ev.get("progress") is not None:
        kb["last"]["progress"] = ev.get("progress")


def _find_recent_attempt_for_service(
    kb: dict,
    service_id: Optional[str],
    exploit: Optional[str] = None,
) -> Optional[str]:
    if not service_id:
        return None

    attempts = list(kb.get("attempts", {}).values())
    attempts.sort(key=lambda item: item.get("step_idx", 0), reverse=True)

    for attempt in attempts:
        if attempt.get("service_id") != service_id:
            continue

        if exploit and exploit not in {
            attempt.get("action_name"),
            attempt.get("vulnerability_id"),
        }:
            # No se descarta siempre, porque action_name puede no coincidir exactamente.
            pass

        return attempt.get("id")

    return None


def _link_finding_to_host_service(kb: dict, finding_id: str) -> None:
    finding = kb["findings"].get(finding_id)
    if not finding:
        return

    for host_id in finding.get("host_ids", []):
        host = kb["target"]["hosts"].get(host_id)
        if host:
            _append_unique(host["finding_ids"], finding_id)

    for service_id in finding.get("service_ids", []):
        service = kb["target"]["services"].get(service_id)
        if service:
            _append_unique(service["finding_ids"], finding_id)


# ============================================================
# Persistence
# ============================================================

def save_kb(session_dir: Path, kb: dict) -> None:
    session_dir = Path(session_dir)

    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ============================================================
# Progress tracking
# ============================================================

def compute_kb_progress_simple(
    prev_kb: dict[str, Any],
    new_kb: dict[str, Any],
) -> dict[str, Any]:
    prev_hosts = _host_set(prev_kb)
    new_hosts = _host_set(new_kb)

    prev_ports = _ports_set(prev_kb)
    new_ports = _ports_set(new_kb)

    prev_services = _services_set(prev_kb)
    new_services = _services_set(new_kb)

    prev_vulns = _vulns_set(prev_kb)
    new_vulns = _vulns_set(new_kb)

    prev_credentials = _credentials_set(prev_kb)
    new_credentials = _credentials_set(new_kb)

    prev_sessions = _sessions_set(prev_kb)
    new_sessions = _sessions_set(new_kb)

    prev_findings = _findings_set(prev_kb)
    new_findings = _findings_set(new_kb)

    added_hosts = new_hosts - prev_hosts
    added_ports = new_ports - prev_ports
    added_services = new_services - prev_services
    added_vulns = new_vulns - prev_vulns
    added_credentials = new_credentials - prev_credentials
    added_sessions = new_sessions - prev_sessions
    added_findings = new_findings - prev_findings

    has_progress = bool(
        added_hosts
        or added_ports
        or added_services
        or added_vulns
        or added_credentials
        or added_sessions
        or added_findings
    )

    return {
        "has_progress": has_progress,
        "new_hosts_count": len(added_hosts),
        "new_ports_count": len(added_ports),
        "new_services_count": len(added_services),
        "new_vulns_count": len(added_vulns),
        "new_credentials_count": len(added_credentials),
        "new_sessions_count": len(added_sessions),
        "new_findings_count": len(added_findings),
    }


def _host_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("hosts", {}).keys())


def _ports_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("ports", {}).keys())


def _services_set(kb: dict[str, Any]) -> set[str]:
    services = kb.get("target", {}).get("services", {})
    return {
        service_id
        for service_id, service in services.items()
        if service.get("name")
    }


def _vulns_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("vulnerabilities", {}).keys())


def _credentials_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("credentials", {}).keys())


def _sessions_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("sessions", {}).keys())


def _findings_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("findings", {}).keys())


def append_kb_step(
    kb: dict,
    step_record: dict,
    result: dict | None = None,
    command_ctx: dict | None = None,
) -> None:
    """
    Añade a kb["steps"] un resumen compacto del paso ejecutado.

    La traza completa se conserva en steps.jsonl y command_outputs.jsonl.
    La KB solo guarda lo necesario para estado futuro, progreso y trazabilidad.
    """
    result = result or {}
    command_ctx = command_ctx or {}

    kb.setdefault("steps", [])

    execution = step_record.get("execution", {})
    decision = step_record.get("decision", {})
    outcome = step_record.get("outcome", {})

    t = step_record.get("t")
    step_idx = int(t) + 1 if t is not None else len(kb["steps"]) + 1

    executed_action_id = execution.get("executed_action_id")
    action_name = execution.get("action_name")

    command = (
        result.get("cmd")
        or result.get("command")
        or command_ctx.get("command")
    )

    compact_step = {
        "step_idx": step_idx,
        "t": t,
        "ts": step_record.get("ts"),

        "requested_action_id": decision.get("requested_action_id"),
        "executed_action_id": executed_action_id,
        "action_name": action_name,

        "phase": command_ctx.get("phase"),
        "target_id": command_ctx.get("target_id"),
        "host_id": command_ctx.get("host_id"),
        "port_id": command_ctx.get("port_id"),
        "service_id": command_ctx.get("service_id"),
        "vulnerability_id": command_ctx.get("vulnerability_id"),
        "credential_id": command_ctx.get("credential_id"),
        "session_id": command_ctx.get("session_id"),

        "target_ip": command_ctx.get("target_ip"),
        "target_port": command_ctx.get("target_port"),
        "service_name": command_ctx.get("service_name"),

        "command": command,
        "rc": result.get("rc"),

        "event_types": outcome.get("event_types", []),
        "events_count": outcome.get("events_count"),
        "progress": outcome.get("progress"),
        "repeated": outcome.get("repeated"),
        "tool_error": outcome.get("tool_error"),
        "timeout": outcome.get("timeout"),
        "goal_reached": outcome.get("goal_reached"),
        "duration_seconds": outcome.get("duration_seconds"),

        "command_log_ref": execution.get("command_log_ref"),
        "stop_reason": step_record.get("stop_reason"),
    }

    kb["steps"].append(compact_step)

    kb.setdefault("stats", {})
    kb["stats"]["steps"] = len(kb["steps"])
    kb["stats"]["commands"] = count_commands_from_steps(kb)


def count_commands_from_steps(kb: dict) -> int:
    total = 0

    for step in kb.get("steps", []):
        if step.get("command"):
            total += 1

    return total

def enrich_events_with_execution_context(
    events: list[dict],
    t: int,
    executed_action_id: int,
    action_name: str,
    result: dict,
    command_ctx: dict | None = None,
) -> list[dict]:
    """
    Añade contexto común de ejecución a cada evento generado por un comando.
    Esto permite que el kb_updater relacione eventos con host, servicio,
    acción, comando y step.
    """
    command_ctx = command_ctx or {}
    result = result or {}

    enriched_events = []

    for ev in events or []:
        new_ev = dict(ev)

        new_ev.setdefault("step_idx", t + 1)
        new_ev.setdefault("t", t)
        new_ev.setdefault("action_id", executed_action_id)
        new_ev.setdefault("action_name", action_name)
        new_ev.setdefault("command", result.get("cmd") or command_ctx.get("command"))
        new_ev.setdefault("rc", result.get("rc"))

        new_ev.setdefault("target_id", command_ctx.get("target_id"))
        new_ev.setdefault("host_id", command_ctx.get("host_id"))
        new_ev.setdefault("port_id", command_ctx.get("port_id"))
        new_ev.setdefault("service_id", command_ctx.get("service_id"))
        new_ev.setdefault("vulnerability_id", command_ctx.get("vulnerability_id"))
        new_ev.setdefault("credential_id", command_ctx.get("credential_id"))
        new_ev.setdefault("session_id", command_ctx.get("session_id"))

        new_ev.setdefault("host", command_ctx.get("target_ip"))
        new_ev.setdefault("port", command_ctx.get("target_port"))
        new_ev.setdefault("service", command_ctx.get("service_name"))

        enriched_events.append(new_ev)

    return enriched_events


def _should_promote_arp_neighbor_to_target_host(kb: dict, ip: str) -> bool:
    if _looks_like_gateway_or_noise(ip):
        return False

    return _is_ip_in_scope(kb, ip)


def _looks_like_gateway_or_noise(ip: str | None) -> bool:
    if not ip:
        return True

    # Evita gateways típicos del lab/NAT.
    if ip.endswith(".1"):
        return True

    return False


def _is_ip_in_scope(kb: dict, ip: str) -> bool:
    scope = kb.get("scope", {})
    target_type = scope.get("target_type")
    target = scope.get("target")

    if not target:
        return False

    if target_type == "host":
        return ip == target

    if target_type == "network":
        return _ip_belongs_to_cidr(ip, target)

    return False


def _ip_belongs_to_cidr(ip: str, cidr: str) -> bool:
    try:
        import ipaddress

        return ipaddress.ip_address(ip) in ipaddress.ip_network(cidr, strict=False)
    except ValueError:
        return False