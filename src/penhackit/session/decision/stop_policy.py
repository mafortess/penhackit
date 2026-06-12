# Política centralizada de parada de sesión.
# La decisión depende del objetivo, del tipo de target y de la KB actual.

# Importante:
# - La parada depende de goal_type + target_type + target + estado de la KB.

from typing import Any


def evaluate_goal_and_stop(kb: dict[str, Any], session_context: dict[str, Any] | None = None, session_config: dict[str, Any] | None = None, outcome: dict[str, Any] | None = None) -> dict[str, Any]:
    session_context = session_context or {}
    session_config = session_config or {}
    outcome = outcome or {}

    goal_type = _get_goal_type(kb, session_context, session_config)
    target_type = _get_target_type(kb, session_context, session_config)
    target = _get_target(kb, session_context, session_config)

    if goal_type == "obtain_session":
        return _evaluate_obtain_session(kb, target_type, target)

    if goal_type in {"exploitation", "full_exploit"}:
        return _evaluate_full_exploit(kb, target_type, target)

    if goal_type == "vulnerability_discovery":
        return _evaluate_vulnerability_discovery(kb, target_type, target)

    if goal_type == "enumeration":
        return _evaluate_enumeration(kb, target_type, target)

    if goal_type == "recon":
        return _evaluate_recon(kb, target_type, target)

    return {
        "goal_type": goal_type,
        "target_type": target_type,
        "target": target,
        "goal_reached": False,
        "should_stop": False,
        "stop_reason": None,
    }


def _evaluate_obtain_session(kb: dict[str, Any], target_type: str, target: str | None) -> dict[str, Any]:
    """
    obtain_session:
    - host: parar cuando se obtiene sesión en ese host.
    - network: NO parar necesariamente con la primera sesión; continuar si queda trabajo pendiente.
    """
    sessions = _sessions_in_scope(kb, target_type, target)
    has_session = len(sessions) > 0

    if target_type == "host":
        return {
            "goal_type": "obtain_session",
            "target_type": target_type,
            "target": target,
            "goal_reached": has_session,
            "should_stop": has_session,
            "stop_reason": "target_host_session_obtained" if has_session else None,
        }

    if target_type == "network":
        pending_host_ids = _alive_host_ids_without_session(kb, target_type, target)
        goal_reached = len(pending_host_ids) == 0 and len(_alive_hosts_in_scope(kb, target_type, target)) > 0

        return {
            "goal_type": "obtain_session",
            "target_type": target_type,
            "target": target,
            "goal_reached": goal_reached,
            "should_stop": goal_reached,    
            "stop_reason": "all_network_hosts_have_session" if goal_reached else None,
            "pending_session_hosts_count": len(pending_host_ids),
            "pending_session_host_ids": sorted(pending_host_ids),
        }

    return {
        "goal_type": "obtain_session",
        "target_type": target_type,
        "target": target,
        "goal_reached": has_session,
        "should_stop": has_session,
        "stop_reason": "session_obtained" if has_session else None,
    }


def _evaluate_full_exploit(kb: dict[str, Any], target_type: str, target: str | None) -> dict[str, Any]:
    """
    full_exploit:
    - host: parar cuando el host objetivo ya tiene sesión o intento de explotación y no quedan vulns pendientes.
    - network: parar solo cuando todos los hosts vivos del scope tienen sesión o intento registrado.
    """
    pending_vulns = _pending_candidate_vulns_in_scope(kb, target_type, target)
    sessions = _sessions_in_scope(kb, target_type, target)
    attempts = _attempts_in_scope(kb, target_type, target)

    has_sessions = len(sessions) > 0
    has_attempts = len(attempts) > 0

    if target_type == "host":
        should_stop = not pending_vulns and (has_sessions or has_attempts)

        return {
            "goal_type": "full_exploit",
            "target_type": target_type,
            "target": target,
            "goal_reached": has_sessions or has_attempts,
            "should_stop": should_stop,
            "stop_reason": "target_host_full_exploit_completed" if should_stop else None,
            "pending_vulns_count": len(pending_vulns),
        }

    if target_type == "network":
        pending_host_ids = _alive_host_ids_without_session_or_attempt(kb, target_type, target)
        goal_reached = len(pending_host_ids) == 0 and len(_alive_hosts_in_scope(kb, target_type, target)) > 0
        should_stop = goal_reached and not pending_vulns

        return {
            "goal_type": "full_exploit",
            "target_type": target_type,
            "target": target,
            "goal_reached": goal_reached,
            "should_stop": should_stop,
            "stop_reason": "all_network_hosts_processed" if should_stop else None,
            "pending_vulns_count": len(pending_vulns),
            "pending_exploit_hosts_count": len(pending_host_ids),
            "pending_exploit_host_ids": sorted(pending_host_ids),
        }

    should_stop = not pending_vulns and (has_sessions or has_attempts)

    return {
        "goal_type": "full_exploit",
        "target_type": target_type,
        "target": target,
        "goal_reached": has_sessions or has_attempts,
        "should_stop": should_stop,
        "stop_reason": "full_exploit_completed" if should_stop else None,
        "pending_vulns_count": len(pending_vulns),
    }


def _evaluate_vulnerability_discovery(kb: dict[str, Any], target_type: str, target: str | None) -> dict[str, Any]:
    """
    vulnerability_discovery:
    - Objetivo alcanzado si hay vulnerabilidades candidatas.
    - Para cuando no queda trabajo razonable de enumeración/vuln discovery.
    """
    vulns = _vulns_in_scope(kb, target_type, target)
    pending_enum = _has_pending_enumeration_work(kb, target_type, target)

    goal_reached = len(vulns) > 0
    should_stop = goal_reached and not pending_enum

    return {
        "goal_type": "vulnerability_discovery",
        "target_type": target_type,
        "target": target,
        "goal_reached": goal_reached,
        "should_stop": should_stop,
        "stop_reason": "vulnerability_discovery_completed" if should_stop else None,
        "vulns_count": len(vulns),
    }


def _evaluate_enumeration(kb: dict[str, Any], target_type: str, target: str | None) -> dict[str, Any]:
    pending_enum = _has_pending_enumeration_work(kb, target_type, target)
    has_services = len(_services_in_scope(kb, target_type, target)) > 0

    should_stop = has_services and not pending_enum

    return {
        "goal_type": "enumeration",
        "target_type": target_type,
        "target": target,
        "goal_reached": has_services,
        "should_stop": should_stop,
        "stop_reason": "enumeration_completed" if should_stop else None,
    }


def _evaluate_recon(kb: dict[str, Any], target_type: str, target: str | None) -> dict[str, Any]:
    hosts = _hosts_in_scope(kb, target_type, target)
    has_hosts = len(hosts) > 0

    if target_type == "network":
        network = _network_by_cidr(kb, target)
        recon_done = bool(network and network.get("host_discovery_done"))
    else:
        recon_done = has_hosts

    should_stop = has_hosts and recon_done

    return {
        "goal_type": "recon",
        "target_type": target_type,
        "target": target,
        "goal_reached": has_hosts,
        "should_stop": should_stop,
        "stop_reason": "recon_completed" if should_stop else None,
    }


# ============================================================
# Scope helpers
# ============================================================

def _get_goal_type(kb: dict[str, Any], session_context: dict[str, Any], session_config: dict[str, Any]) -> str:
    goal_type = (
        session_context.get("goal_type")
        or session_config.get("goal_type")
        or kb.get("scope", {}).get("goal")
        or "obtain_session"
    )

    return str(goal_type).replace(" ", "_")


def _get_target_type(kb: dict[str, Any], session_context: dict[str, Any], session_config: dict[str, Any]) -> str:
    return (
        session_context.get("target_type")
        or session_config.get("target_type")
        or kb.get("scope", {}).get("target_type")
        or "host"
    )


def _get_target(kb: dict[str, Any], session_context: dict[str, Any], session_config: dict[str, Any]) -> str | None:
    return (
        session_context.get("target")
        or session_config.get("target")
        or kb.get("scope", {}).get("target")
    )


def _network_by_cidr(kb: dict[str, Any], cidr: str | None) -> dict[str, Any] | None:
    if not cidr:
        return None

    for network in kb.get("target", {}).get("networks", {}).values():
        if network.get("cidr") == cidr:
            return network

    return None


def _hosts_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    hosts = list(kb.get("target", {}).get("hosts", {}).values())

    if target_type == "host" and target:
        return [host for host in hosts if host.get("ip") == target or host.get("id") == target]

    if target_type == "network" and target:
        network = _network_by_cidr(kb, target)
        if not network:
            return []

        host_ids = set(network.get("host_ids", []))
        return [host for host in hosts if host.get("id") in host_ids]

    return hosts


def _services_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    scoped_host_ids = {host.get("id") for host in _hosts_in_scope(kb, target_type, target)}
    services = list(kb.get("target", {}).get("services", {}).values())

    if scoped_host_ids:
        return [service for service in services if service.get("host_id") in scoped_host_ids]

    return services


def _vulns_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    scoped_host_ids = {host.get("id") for host in _hosts_in_scope(kb, target_type, target)}
    vulns = list(kb.get("vulnerabilities", {}).values())

    if scoped_host_ids:
        return [vuln for vuln in vulns if vuln.get("host_id") in scoped_host_ids]

    return vulns


def _sessions_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    scoped_host_ids = {host.get("id") for host in _hosts_in_scope(kb, target_type, target)}
    sessions = list(kb.get("sessions", {}).values())

    if scoped_host_ids:
        return [session for session in sessions if session.get("host_id") in scoped_host_ids]

    return sessions


def _attempts_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    scoped_host_ids = {host.get("id") for host in _hosts_in_scope(kb, target_type, target)}
    attempts = list(kb.get("attempts", {}).values())

    if scoped_host_ids:
        return [attempt for attempt in attempts if attempt.get("host_id") in scoped_host_ids]

    return attempts


# ============================================================
# Pending work
# ============================================================

def _pending_candidate_vulns_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[str]:
    pending = []
    attempted = set(kb.get("coverage", {}).get("vulns_attempted", []))

    for vuln in _vulns_in_scope(kb, target_type, target):
        vuln_id = vuln.get("id")
        status = vuln.get("status")

        if not vuln_id:
            continue

        if status not in {"candidate", "validated"}:
            continue

        if vuln_id in attempted:
            continue

        if vuln.get("attempt_ids"):
            continue

        pending.append(vuln_id)

    return pending


def _has_pending_network_work(kb: dict[str, Any], target: str | None) -> bool:
    return (
        _has_pending_enumeration_work(kb, "network", target)
        or bool(_pending_candidate_vulns_in_scope(kb, "network", target))
    )


def _has_pending_enumeration_work(kb: dict[str, Any], target_type: str, target: str | None) -> bool:
    coverage = kb.get("coverage", {})

    hosts = _hosts_in_scope(kb, target_type, target)
    services = _services_in_scope(kb, target_type, target)

    hosts_service_scanned = set(coverage.get("hosts_service_scanned", []))
    services_enumerated = set(coverage.get("services_enumerated", []))

    for host in hosts:
        host_id = host.get("id")
        if host_id and host_id not in hosts_service_scanned:
            return True

    for service in services:
        service_id = service.get("id")
        if service_id and service_id not in services_enumerated:
            return True

    return False


def _alive_hosts_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> list[dict[str, Any]]:
    hosts = _hosts_in_scope(kb, target_type, target)
    return [host for host in hosts if host.get("alive") is True]


def _session_host_ids_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> set[str]:
    host_ids = set()

    for session in _sessions_in_scope(kb, target_type, target):
        host_id = session.get("host_id")

        if not host_id:
            host = session.get("host") or session.get("ip")
            if host:
                host_id = f"host:{host}"

        if host_id:
            host_ids.add(host_id)

    return host_ids


def _attempt_host_ids_in_scope(kb: dict[str, Any], target_type: str, target: str | None) -> set[str]:
    host_ids = set()

    for attempt in _attempts_in_scope(kb, target_type, target):
        host_id = attempt.get("host_id")

        if not host_id:
            host = attempt.get("host") or attempt.get("ip")
            if host:
                host_id = f"host:{host}"

        if host_id:
            host_ids.add(host_id)

    return host_ids


def _alive_host_ids_without_session(kb: dict[str, Any], target_type: str, target: str | None) -> set[str]:
    alive_host_ids = {
        host.get("id")
        for host in _alive_hosts_in_scope(kb, target_type, target)
        if host.get("id")
    }

    session_host_ids = _session_host_ids_in_scope(kb, target_type, target)

    return alive_host_ids - session_host_ids


def _alive_host_ids_without_session_or_attempt(kb: dict[str, Any], target_type: str, target: str | None) -> set[str]:
    alive_host_ids = {
        host.get("id")
        for host in _alive_hosts_in_scope(kb, target_type, target)
        if host.get("id")
    }

    processed_host_ids = (
        _session_host_ids_in_scope(kb, target_type, target)
        | _attempt_host_ids_in_scope(kb, target_type, target)
    )

    return alive_host_ids - processed_host_ids