from typing import Any

# Selección del foco operativo de la sesión.

# El foco determina sobre qué entidad se construye el estado:
# - red
# - host
# - servicio
# - vulnerabilidad
# - credencial
# - sesión

# No decide la acción. Solo decide el "objeto activo" sobre el que el state_builder y la política deben razonar.



def update_focus(kb: dict[str, Any], session_context: dict[str, Any] | None = None, session_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Actualiza kb["focus"] en función del objetivo, el tipo de target,
    el target concreto y el conocimiento disponible en la KB.
    """
    session_context = session_context or {}
    session_config = session_config or {}

    goal_type = _get_goal_type(kb, session_context, session_config)
    target_type = _get_target_type(kb, session_context, session_config)
    target = _get_target(kb, session_context, session_config)

    kb.setdefault("focus", {})

    if target_type == "host":
        focus = _select_focus_for_host_target(kb, goal_type, target)
    elif target_type == "network":
        focus = _select_focus_for_network_target(kb, goal_type, target)
    else:
        focus = _make_global_focus(reason="unknown_target_type")

    kb["focus"].update(focus)

    return kb


# ============================================================
# Host target
# ============================================================

def _select_focus_for_host_target(
    kb: dict[str, Any],
    goal_type: str,
    target: str | None,
) -> dict[str, Any]:
    host = _find_host_by_target(kb, target)

    if host is None:
        return _make_host_focus(
            host_id=_host_id_from_target(target),
            reason="target_host_not_discovered_yet",
        )

    host_id = host.get("id")

    pending_vuln = _first_pending_vuln_for_host(kb, host_id)
    if pending_vuln is not None and goal_type in {"exploitation", "full_exploit", "obtain_session"}:
        return _make_vuln_focus(pending_vuln, reason="pending_vulnerability_on_target_host")

    pending_service = _first_pending_service_for_host(kb, host_id)
    if pending_service is not None and goal_type in {
        "enumeration",
        "vulnerability_discovery",
        "exploitation",
        "full_exploit",
        "obtain_session",
    }:
        return _make_service_focus(pending_service, reason="pending_service_on_target_host")

    if _host_needs_port_or_service_scan(kb, host_id):
        return _make_host_focus(host_id, reason="target_host_needs_scan")

    active_session = _first_session_for_host(kb, host_id)
    if active_session is not None:
        return _make_session_focus(active_session, reason="session_available_on_target_host")

    return _make_host_focus(host_id, reason="target_host_default")


# ============================================================
# Network target
# ============================================================

def _select_focus_for_network_target(
    kb: dict[str, Any],
    goal_type: str,
    target: str | None,
) -> dict[str, Any]:
    network = _find_network_by_target(kb, target)

    if network is None:
        return _make_network_focus(
            network_id=_network_id_from_target(target),
            reason="target_network_not_discovered_yet",
        )

    network_id = network.get("id")
    host_ids = network.get("host_ids", [])

    if not host_ids:
        return _make_network_focus(network_id, reason="network_has_no_hosts_yet")

    if goal_type == "obtain_session":
        host_ids = _host_ids_without_session(kb, host_ids)

        if not host_ids:
            return _make_network_focus(network_id, reason="all_network_hosts_have_session")

    if goal_type in {"exploitation", "full_exploit"}:
        host_ids = _host_ids_with_pending_full_exploit_work(kb, host_ids)

        if not host_ids:
            return _make_network_focus(network_id, reason="network_full_exploit_completed")

    if goal_type in {"exploitation", "full_exploit", "obtain_session"}:
        pending_vuln = _first_pending_vuln_for_hosts(kb, host_ids)
        if pending_vuln is not None:
            return _make_vuln_focus(pending_vuln, reason="pending_vulnerability_in_network")

    if goal_type in {
        "enumeration",
        "vulnerability_discovery",
        "exploitation",
        "full_exploit",
        "obtain_session",
    }:
        pending_service = _first_pending_service_for_hosts(kb, host_ids)
        if pending_service is not None:
            return _make_service_focus(pending_service, reason="pending_service_in_network")

    pending_host = _first_host_needing_scan(kb, host_ids)
    if pending_host is not None:
        return _make_host_focus(pending_host.get("id"), reason="host_in_network_needs_scan")

    first_host = _first_host(kb, host_ids)
    if first_host is not None:
        return _make_host_focus(first_host.get("id"), reason="network_default_first_unprocessed_host")

    return _make_network_focus(network_id, reason="network_default")

# def _select_focus_for_network_target(
#     kb: dict[str, Any],
#     goal_type: str,
#     target: str | None,
# ) -> dict[str, Any]:
#     network = _find_network_by_target(kb, target)

#     if network is None:
#         return _make_network_focus(
#             network_id=_network_id_from_target(target),
#             reason="target_network_not_discovered_yet",
#         )

#     network_id = network.get("id")
#     host_ids = network.get("host_ids", [])

#     if not host_ids:
#         return _make_network_focus(network_id, reason="network_has_no_hosts_yet")

#     if goal_type in {"exploitation", "full_exploit", "obtain_session"}:
#         pending_vuln = _first_pending_vuln_for_hosts(kb, host_ids)
#         if pending_vuln is not None:
#             return _make_vuln_focus(pending_vuln, reason="pending_vulnerability_in_network")

#     if goal_type in {
#         "enumeration",
#         "vulnerability_discovery",
#         "exploitation",
#         "full_exploit",
#         "obtain_session",
#     }:
#         pending_service = _first_pending_service_for_hosts(kb, host_ids)
#         if pending_service is not None:
#             return _make_service_focus(pending_service, reason="pending_service_in_network")

#     pending_host = _first_host_needing_scan(kb, host_ids)
#     if pending_host is not None:
#         return _make_host_focus(pending_host.get("id"), reason="host_in_network_needs_scan")

#     active_session = _first_session_for_hosts(kb, host_ids)
#     if active_session is not None:
#         return _make_session_focus(active_session, reason="session_available_in_network")

#     first_host = _first_host(kb, host_ids)
#     if first_host is not None:
#         return _make_host_focus(first_host.get("id"), reason="network_default_first_host")

#     return _make_network_focus(network_id, reason="network_default")


# ============================================================
# Focus object builders
# ============================================================

def _make_global_focus(reason: str) -> dict[str, Any]:
    return {
        "level": "global",
        "network_id": None,
        "host_id": None,
        "port_id": None,
        "service_id": None,
        "vulnerability_id": None,
        "credential_id": None,
        "session_id": None,
        "reason": reason,
    }


def _make_network_focus(network_id: str | None, reason: str) -> dict[str, Any]:
    return {
        "level": "network",
        "network_id": network_id,
        "host_id": None,
        "port_id": None,
        "service_id": None,
        "vulnerability_id": None,
        "credential_id": None,
        "session_id": None,
        "reason": reason,
    }


def _make_host_focus(host_id: str | None, reason: str) -> dict[str, Any]:
    return {
        "level": "host",
        "network_id": None,
        "host_id": host_id,
        "port_id": None,
        "service_id": None,
        "vulnerability_id": None,
        "credential_id": None,
        "session_id": None,
        "reason": reason,
    }


def _make_service_focus(service: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "level": "service",
        "network_id": None,
        "host_id": service.get("host_id"),
        "port_id": service.get("port_id"),
        "service_id": service.get("id"),
        "vulnerability_id": None,
        "credential_id": None,
        "session_id": None,
        "reason": reason,
    }


def _make_vuln_focus(vuln: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "level": "vuln",
        "network_id": None,
        "host_id": vuln.get("host_id"),
        "port_id": None,
        "service_id": vuln.get("service_id"),
        "vulnerability_id": vuln.get("id"),
        "credential_id": None,
        "session_id": None,
        "reason": reason,
    }


def _make_session_focus(session: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "level": "session",
        "network_id": None,
        "host_id": session.get("host_id"),
        "port_id": None,
        "service_id": session.get("service_id"),
        "vulnerability_id": session.get("vulnerability_id"),
        "credential_id": session.get("credential_id"),
        "session_id": session.get("id"),
        "reason": reason,
    }


# ============================================================
# Selection helpers
# ============================================================

def _first_pending_vuln_for_host(kb: dict[str, Any], host_id: str | None) -> dict[str, Any] | None:
    if not host_id:
        return None

    for vuln in kb.get("vulnerabilities", {}).values():
        if vuln.get("host_id") != host_id:
            continue

        if _is_pending_vuln(kb, vuln):
            return vuln

    return None


def _first_pending_vuln_for_hosts(kb: dict[str, Any], host_ids: list[str]) -> dict[str, Any] | None:
    host_id_set = set(host_ids)

    for vuln in kb.get("vulnerabilities", {}).values():
        if vuln.get("host_id") not in host_id_set:
            continue

        if _is_pending_vuln(kb, vuln):
            return vuln

    return None


def _is_pending_vuln(kb: dict[str, Any], vuln: dict[str, Any]) -> bool:
    vuln_id = vuln.get("id")
    status = vuln.get("status")

    if not vuln_id:
        return False

    if status not in {"candidate", "validated"}:
        return False

    attempted_vulns = set(kb.get("coverage", {}).get("vulns_attempted", []))
    if vuln_id in attempted_vulns:
        return False

    if vuln.get("attempt_ids"):
        return False

    return True


def _first_pending_service_for_host(kb: dict[str, Any], host_id: str | None) -> dict[str, Any] | None:
    if not host_id:
        return None

    services = kb.get("target", {}).get("services", {})
    coverage = kb.get("coverage", {})
    services_enumerated = set(coverage.get("services_enumerated", []))
    services_checked = set(coverage.get("services_checked_for_vulns", []))

    for service in services.values():
        if service.get("host_id") != host_id:
            continue

        service_id = service.get("id")

        if service_id not in services_enumerated:
            return service

        if service_id not in services_checked:
            return service

    return None


def _first_pending_service_for_hosts(kb: dict[str, Any], host_ids: list[str]) -> dict[str, Any] | None:
    host_id_set = set(host_ids)

    services = kb.get("target", {}).get("services", {})
    coverage = kb.get("coverage", {})
    services_enumerated = set(coverage.get("services_enumerated", []))
    services_checked = set(coverage.get("services_checked_for_vulns", []))

    for service in services.values():
        if service.get("host_id") not in host_id_set:
            continue

        service_id = service.get("id")

        if service_id not in services_enumerated:
            return service

        if service_id not in services_checked:
            return service

    return None


def _first_host_needing_scan(kb: dict[str, Any], host_ids: list[str]) -> dict[str, Any] | None:
    for host_id in host_ids:
        host = kb.get("target", {}).get("hosts", {}).get(host_id)
        if not host:
            continue

        if _host_needs_port_or_service_scan(kb, host_id):
            return host

    return None


def _host_needs_port_or_service_scan(kb: dict[str, Any], host_id: str | None) -> bool:
    if not host_id:
        return False

    host = kb.get("target", {}).get("hosts", {}).get(host_id)
    if not host:
        return True

    coverage = kb.get("coverage", {})

    hosts_port_scanned = set(coverage.get("hosts_port_scanned", []))
    hosts_service_scanned = set(coverage.get("hosts_service_scanned", []))

    if host_id not in hosts_port_scanned:
        return True

    if host_id not in hosts_service_scanned:
        return True

    return False


def _first_session_for_host(kb: dict[str, Any], host_id: str | None) -> dict[str, Any] | None:
    if not host_id:
        return None

    for session in kb.get("sessions", {}).values():
        if session.get("host_id") == host_id:
            return session

    return None


def _first_session_for_hosts(kb: dict[str, Any], host_ids: list[str]) -> dict[str, Any] | None:
    host_id_set = set(host_ids)

    for session in kb.get("sessions", {}).values():
        if session.get("host_id") in host_id_set:
            return session

    return None


def _first_host(kb: dict[str, Any], host_ids: list[str]) -> dict[str, Any] | None:
    for host_id in host_ids:
        host = kb.get("target", {}).get("hosts", {}).get(host_id)
        if host:
            return host

    return None


# ============================================================
# Target helpers
# ============================================================

def _get_goal_type(
    kb: dict[str, Any],
    session_context: dict[str, Any],
    session_config: dict[str, Any],
) -> str:
    goal_type = (
        session_context.get("goal_type")
        or session_config.get("goal_type")
        or kb.get("scope", {}).get("goal")
        or "obtain_session"
    )

    return str(goal_type).replace(" ", "_")


def _get_target_type(
    kb: dict[str, Any],
    session_context: dict[str, Any],
    session_config: dict[str, Any],
) -> str:
    return (
        session_context.get("target_type")
        or session_config.get("target_type")
        or kb.get("scope", {}).get("target_type")
        or "host"
    )


def _get_target(
    kb: dict[str, Any],
    session_context: dict[str, Any],
    session_config: dict[str, Any],
) -> str | None:
    return (
        session_context.get("target")
        or session_config.get("target")
        or kb.get("scope", {}).get("target")
    )


def _find_host_by_target(kb: dict[str, Any], target: str | None) -> dict[str, Any] | None:
    if not target:
        return None

    for host in kb.get("target", {}).get("hosts", {}).values():
        if host.get("ip") == target:
            return host

        if host.get("id") == target:
            return host

    return None


def _find_network_by_target(kb: dict[str, Any], target: str | None) -> dict[str, Any] | None:
    if not target:
        return None

    for network in kb.get("target", {}).get("networks", {}).values():
        if network.get("cidr") == target:
            return network

        if network.get("id") == target:
            return network

    return None


def _host_id_from_target(target: str | None) -> str | None:
    if not target:
        return None

    if target.startswith("host:"):
        return target

    return f"host:{target}"


def _network_id_from_target(target: str | None) -> str | None:
    if not target:
        return None

    if target.startswith("net:"):
        return target

    return f"net:{target}"

def _host_ids_without_session(kb: dict[str, Any], host_ids: list[str]) -> list[str]:
    result = []

    for host_id in host_ids:
        if not _host_has_session(kb, host_id):
            result.append(host_id)

    return result


def _host_ids_with_pending_full_exploit_work(kb: dict[str, Any], host_ids: list[str]) -> list[str]:
    result = []

    for host_id in host_ids:
        if _host_needs_port_or_service_scan(kb, host_id):
            result.append(host_id)
            continue

        if _first_pending_service_for_host(kb, host_id) is not None:
            result.append(host_id)
            continue

        if _first_pending_vuln_for_host(kb, host_id) is not None:
            result.append(host_id)
            continue

    return result


def _host_has_session(kb: dict[str, Any], host_id: str | None) -> bool:
    if not host_id:
        return False

    for session in kb.get("sessions", {}).values():
        if session.get("host_id") == host_id:
            return True

    return False