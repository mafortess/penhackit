from typing import Any


def compute_kb_progress_simple(prev_kb: dict[str, Any], new_kb: dict[str, Any]) -> dict[str, Any]:
    """
    Calcula progreso entre dos snapshots de KB siguiendo el schema v2.

    Distingue entre:
    - progreso de conocimiento: nuevos hosts, puertos, servicios, vulnerabilidades,
      credenciales, sesiones, evidencias, findings o enriquecimiento de entidades.
    - actividad: nuevos steps o attempts, que no cuentan por sí solos como progreso
      porque una acción fallida también genera actividad.

    Devuelve claves compatibles con el código anterior:
    - has_progress
    - new_hosts_count
    - new_ports_count
    - new_services_count
    - new_findings_count
    """

    prev_networks = _network_set(prev_kb)
    new_networks = _network_set(new_kb)

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

    prev_evidence = _evidence_set(prev_kb)
    new_evidence = _evidence_set(new_kb)

    prev_findings = _findings_set(prev_kb)
    new_findings = _findings_set(new_kb)

    prev_attempts = _attempts_set(prev_kb)
    new_attempts = _attempts_set(new_kb)

    prev_steps = _steps_set(prev_kb)
    new_steps = _steps_set(new_kb)

    added_networks = new_networks - prev_networks
    added_hosts = new_hosts - prev_hosts
    added_ports = new_ports - prev_ports
    added_services = new_services - prev_services
    added_vulns = new_vulns - prev_vulns
    added_credentials = new_credentials - prev_credentials
    added_sessions = new_sessions - prev_sessions
    added_evidence = new_evidence - prev_evidence
    added_findings = new_findings - prev_findings

    added_attempts = new_attempts - prev_attempts
    added_steps = new_steps - prev_steps

    changed_hosts = _changed_existing_profiles(
        _host_profile_map(prev_kb),
        _host_profile_map(new_kb),
        added_hosts,
    )

    changed_services = _changed_existing_profiles(
        _service_profile_map(prev_kb),
        _service_profile_map(new_kb),
        added_services,
    )

    changed_vulns = _changed_existing_profiles(
        _vuln_profile_map(prev_kb),
        _vuln_profile_map(new_kb),
        added_vulns,
    )

    changed_credentials = _changed_existing_profiles(
        _credential_profile_map(prev_kb),
        _credential_profile_map(new_kb),
        added_credentials,
    )

    changed_sessions = _changed_existing_profiles(
        _session_profile_map(prev_kb),
        _session_profile_map(new_kb),
        added_sessions,
    )

    has_progress = bool(
        added_networks
        or added_hosts
        or added_ports
        or added_services
        or added_vulns
        or added_credentials
        or added_sessions
        or added_evidence
        or added_findings
        or changed_hosts
        or changed_services
        or changed_vulns
        or changed_credentials
        or changed_sessions
    )

    return {
        "has_progress": has_progress,

        # Compatibilidad con código anterior
        "new_hosts_count": len(added_hosts),
        "new_ports_count": len(added_ports),
        "new_services_count": len(added_services),
        "new_findings_count": len(added_findings),

        # Nuevos contadores v2
        "new_networks_count": len(added_networks),
        "new_vulns_count": len(added_vulns),
        "new_credentials_count": len(added_credentials),
        "new_sessions_count": len(added_sessions),
        "new_evidence_count": len(added_evidence),

        # Cambios sobre entidades ya existentes
        "changed_hosts_count": len(changed_hosts),
        "changed_services_count": len(changed_services),
        "changed_vulns_count": len(changed_vulns),
        "changed_credentials_count": len(changed_credentials),
        "changed_sessions_count": len(changed_sessions),

        # Actividad registrada, no necesariamente progreso
        "new_attempts_count": len(added_attempts),
        "new_steps_count": len(added_steps),

        # IDs útiles para debug, logs o futuro estado
        "new_networks": sorted(added_networks),
        "new_hosts": sorted(added_hosts),
        "new_ports": sorted(added_ports),
        "new_services": sorted(added_services),
        "new_vulns": sorted(added_vulns),
        "new_credentials": sorted(added_credentials),
        "new_sessions": sorted(added_sessions),
        "new_evidence": sorted(added_evidence),
        "new_findings": sorted(added_findings),

        "changed_hosts": sorted(changed_hosts),
        "changed_services": sorted(changed_services),
        "changed_vulns": sorted(changed_vulns),
        "changed_credentials": sorted(changed_credentials),
        "changed_sessions": sorted(changed_sessions),
    }


def _network_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("networks", {}).keys())


def _host_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("hosts", {}).keys())


def _ports_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("ports", {}).keys())


def _services_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("target", {}).get("services", {}).keys())


def _vulns_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("vulnerabilities", {}).keys())


def _credentials_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("credentials", {}).keys())


def _sessions_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("sessions", {}).keys())


def _attempts_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("attempts", {}).keys())


def _evidence_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("evidence", {}).keys())


def _findings_set(kb: dict[str, Any]) -> set[str]:
    return set(kb.get("findings", {}).keys())


def _steps_set(kb: dict[str, Any]) -> set[str]:
    steps = kb.get("steps", [])
    result = set()

    for step in steps:
        if not isinstance(step, dict):
            continue

        step_idx = step.get("step_idx")
        if step_idx is None:
            step_idx = step.get("t")

        if step_idx is not None:
            result.add(str(step_idx))

    return result


def _host_profile_map(kb: dict[str, Any]) -> dict[str, tuple]:
    hosts = kb.get("target", {}).get("hosts", {})
    profiles = {}

    for host_id, host in hosts.items():
        profiles[host_id] = (
            host.get("alive"),
            host.get("os"),
            tuple(sorted(host.get("hostnames", []))),
            tuple(sorted(host.get("port_ids", []))),
            tuple(sorted(host.get("service_ids", []))),
            tuple(sorted(host.get("vulnerability_ids", []))),
            tuple(sorted(host.get("credential_ids", []))),
            tuple(sorted(host.get("session_ids", []))),
            tuple(sorted(host.get("finding_ids", []))),
        )

    return profiles


def _service_profile_map(kb: dict[str, Any]) -> dict[str, tuple]:
    services = kb.get("target", {}).get("services", {})
    profiles = {}

    for service_id, service in services.items():
        profiles[service_id] = (
            service.get("name"),
            service.get("family"),
            service.get("product"),
            service.get("version"),
            service.get("banner"),
            tuple(sorted(service.get("technology", []))),
            service.get("enumerated"),
            service.get("vuln_checked"),
            tuple(sorted(service.get("vulnerability_ids", []))),
            tuple(sorted(service.get("credential_ids", []))),
            tuple(sorted(service.get("session_ids", []))),
            tuple(sorted(service.get("finding_ids", []))),
        )

    return profiles


def _vuln_profile_map(kb: dict[str, Any]) -> dict[str, tuple]:
    vulns = kb.get("vulnerabilities", {})
    profiles = {}

    for vuln_id, vuln in vulns.items():
        profiles[vuln_id] = (
            vuln.get("name"),
            vuln.get("status"),
            vuln.get("confidence"),
            vuln.get("source"),
            tuple(sorted(vuln.get("evidence_ids", []))),
            tuple(sorted(vuln.get("attempt_ids", []))),
            tuple(sorted(vuln.get("finding_ids", []))),
        )

    return profiles


def _credential_profile_map(kb: dict[str, Any]) -> dict[str, tuple]:
    credentials = kb.get("credentials", {})
    profiles = {}

    for credential_id, credential in credentials.items():
        profiles[credential_id] = (
            credential.get("username"),
            credential.get("service"),
            credential.get("host"),
            credential.get("port"),
            credential.get("valid"),
            credential.get("source"),
            tuple(sorted(credential.get("evidence_ids", []))),
            tuple(sorted(credential.get("attempt_ids", []))),
            tuple(sorted(credential.get("finding_ids", []))),
        )

    return profiles


def _session_profile_map(kb: dict[str, Any]) -> dict[str, tuple]:
    sessions = kb.get("sessions", {})
    profiles = {}

    for session_id, session in sessions.items():
        profiles[session_id] = (
            session.get("type"),
            session.get("status"),
            session.get("host_id"),
            session.get("service_id"),
            session.get("vulnerability_id"),
            session.get("credential_id"),
            session.get("attempt_id"),
            session.get("user"),
            session.get("privilege"),
            session.get("hostname"),
            session.get("system"),
            session.get("opened_at_step"),
            session.get("closed_at_step"),
            tuple(sorted(session.get("evidence_ids", []))),
            tuple(sorted(session.get("finding_ids", []))),
        )

    return profiles


def _changed_existing_profiles(
    prev_profiles: dict[str, tuple],
    new_profiles: dict[str, tuple],
    added_ids: set[str],
) -> set[str]:
    changed = set()

    common_ids = set(prev_profiles.keys()) & set(new_profiles.keys())

    for entity_id in common_ids:
        if entity_id in added_ids:
            continue

        if prev_profiles[entity_id] != new_profiles[entity_id]:
            changed.add(entity_id)

    return changed


def print_autonomous_progress(progress: dict):
    if not progress["has_progress"]:
        print("NO PROGRESS")
        return

    print(
        "PROGRESS: "
        f"+networks={progress.get('new_networks_count', 0)} "
        f"+hosts={progress.get('new_hosts_count', 0)} "
        f"+ports={progress.get('new_ports_count', 0)} "
        f"+services={progress.get('new_services_count', 0)} "
        f"+vulns={progress.get('new_vulns_count', 0)} "
        f"+creds={progress.get('new_credentials_count', 0)} "
        f"+sessions={progress.get('new_sessions_count', 0)} "
        f"+findings={progress.get('new_findings_count', 0)} "
        f"+evidence={progress.get('new_evidence_count', 0)} "
        f"changed_services={progress.get('changed_services_count', 0)} "
        f"changed_vulns={progress.get('changed_vulns_count', 0)} "
        f"changed_sessions={progress.get('changed_sessions_count', 0)}"
    )