# kb_schema.py
# Estructura inicial de la Knowledge Base (KB).

# Diseño v2:
# - KB híbrida grafo-relacional en JSON.
# - Las entidades principales tienen IDs estables.
# - Las relaciones se expresan mediante referencias entre entidades.
# - La KB completa es memoria rica.
# - El estado del modelo debe ser una vista compacta derivada de esta KB.

from typing import Any, Optional


KB_SCHEMA_VERSION = "2.0"


# ============================================================
# ID helpers
# ============================================================

def make_network_id(cidr: str) -> str:
    return f"net:{cidr}"


def make_host_id(ip: str) -> str:
    return f"host:{ip}"


def make_port_id(host: str, port: int, protocol: str = "tcp") -> str:
    return f"port:{host}:{protocol}:{int(port)}"


def make_service_id(host: str, port: int, protocol: str = "tcp") -> str:
    return f"svc:{host}:{protocol}:{int(port)}"


def make_vulnerability_id(name: str, host: str, port: Optional[int] = None, protocol: str = "tcp") -> str:
    if port is None:
        return f"vuln:{name}:{host}"
    return f"vuln:{name}:{host}:{protocol}:{int(port)}"


def make_credential_id(host: str, service: str, username: str, port: Optional[int] = None) -> str:
    if port is None:
        return f"cred:{host}:{service}:{username}"
    return f"cred:{host}:{service}:{int(port)}:{username}"


def make_session_id(step_idx: int, host: str, port: Optional[int] = None) -> str:
    if port is None:
        return f"sess:{int(step_idx)}:{host}"
    return f"sess:{int(step_idx)}:{host}:{int(port)}"


def make_attempt_id(step_idx: int, action_id: Optional[int] = None) -> str:
    if action_id is None:
        return f"attempt:{int(step_idx):04d}"
    return f"attempt:{int(step_idx):04d}:{int(action_id)}"


def make_evidence_id(step_idx: int, kind: str, suffix: Optional[str] = None) -> str:
    if suffix:
        return f"evd:{int(step_idx):04d}:{kind}:{suffix}"
    return f"evd:{int(step_idx):04d}:{kind}"


def make_finding_id(title: str, host: Optional[str] = None, port: Optional[int] = None) -> str:
    safe_title = (
        title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("-", "_")
    )

    if host is None:
        return f"finding:{safe_title}"

    if port is None:
        return f"finding:{safe_title}:{host}"

    return f"finding:{safe_title}:{host}:{int(port)}"


# ============================================================
# Initial KB
# ============================================================

def build_initial_kb(session_context: dict) -> dict:
    """
    Construye una KB inicial vacía para una sesión de pentesting.

    La KB almacena memoria rica,trazabilidad y relaciones. 
    El estado se deriva posteriormente de esta estructura mediante el state_builder.
    """
    session_id = session_context.get("id")
    target_type = session_context.get("target_type") or "host"
    target = session_context.get("target") or "10.6.6.10"
    goal_type = session_context.get("goal_type") or "obtain_session"
    name = session_context.get("name")

    return {
        "schema_version": KB_SCHEMA_VERSION,

        # Metadata de la sesión actual.
        "session": {
            "id": session_id,
            "name": name,
            "mode": session_context.get("mode"),
            "created_at": session_context.get("created_at"),
            "status": "created",
        },

        # Alcance autorizado y objetivo operativo.
        "scope": {
            "target_type": target_type,
            "target": target,
            "goal": goal_type,
            "authorized": True,
            "allow_exploitation": True,
            "allow_pivoting": False,
            "max_steps": session_context.get("max_steps"),
        },

        # Contexto local del atacante/Kali.
        "attacker": {
            "hostname": None,
            "interfaces": [],
            "ipv4": [],
            "default_gw": [],
            "arp_neighbors": [],
            "routes": [],
            "lhost": None,
        },

        # Inventario técnico del objetivo.
        # Entidades separadas y relacionadas mediante IDs.
        "target": {
            "networks": {},
            "hosts": {},
            "ports": {},
            "services": {},
        },

        # Vulnerabilidades candidatas, validadas o rechazadas.
        "vulnerabilities": {},

        # Credenciales descubiertas o validadas.
        "credentials": {},

        # Sesiones obtenidas mediante explotación o login.
        "sessions": {},

        # Intentos de acciones relevantes.
        "attempts": {},

        # Evidencias técnicas extraídas de resultados.
        "evidence": {},

        # Hallazgos reportables.
        "findings": {},

        # Cobertura operativa.
        # Sirve para evitar repeticiones y para construir features del estado.
        "coverage": {
            "networks_discovered": [],
            "networks_scanned": [],
            "hosts_discovered": [],
            "hosts_port_scanned": [],
            "hosts_service_scanned": [],
            "services_enumerated": [],
            "services_checked_for_vulns": [],
            "vulns_attempted": [],
            "credentials_tested": [],
            "exploits_attempted": [],
            "sessions_validated": [],
        },

        # Historial semántico compacto.
        # stdout/stderr completo debe guardarse en logs, no aquí.
        "history": [],

        # Pasos ejecutados durante la sesión.
        # Cada entrada representa una acción/comando ejecutado y sus eventos asociados.
        "steps": [],

        # Resumen del último paso ejecutado.
        "last": {
            "step_idx": 0,
            "action_id": None,
            "action_name": None,
            "rc": None,
            "success": None,
            "event_types": [],
            "progress": None,
        },

        # Foco operativo actual.
        # El state_builder debe construir el estado principalmente sobre este foco.
        "focus": {
            "level": "global",
            "network_id": None,
            "host_id": None,
            "port_id": None,
            "service_id": None,
            "vulnerability_id": None,
            "credential_id": None,
            "session_id": None,
        },

        # Contadores compactos para progreso, estado y reporting.
        "stats": {
            "steps": 0,
            "commands": 0,
            "networks": 0,
            "hosts": 0,
            "ports": 0,
            "services": 0,
            "vulns": 0,
            "credentials": 0,
            "sessions": 0,
            "attempts": 0,
            "evidence": 0,
            "findings": 0,
            "exploits_attempted": 0,
            "exploits_failed": 0,
        },
    }


# ============================================================
# Entity constructors
# ============================================================

def make_network(cidr: str, in_scope: bool = True) -> dict:
    network_id = make_network_id(cidr)

    return {
        "id": network_id,
        "cidr": cidr,
        "in_scope": in_scope,
        "reachable": None,
        "discovered": False,
        "host_discovery_done": False,
        "port_scan_done": False,
        "host_ids": [],
        "evidence_ids": [],
    }


def make_host(
    ip: str,
    network_id: Optional[str] = None,
    in_scope: bool = True,
    alive: Optional[bool] = None,
) -> dict:
    host_id = make_host_id(ip)

    return {
        "id": host_id,
        "ip": ip,
        "network_id": network_id,
        "in_scope": in_scope,
        "alive": alive,
        "hostnames": [],
        "os": None,
        "port_ids": [],
        "service_ids": [],
        "credential_ids": [],
        "vulnerability_ids": [],
        "session_ids": [],
        "finding_ids": [],
        "evidence_ids": [],
        "attempt_ids": [],
    }


def make_port(
    host: str,
    port: int,
    protocol: str = "tcp",
    state: str = "open",
) -> dict:
    host_id = make_host_id(host)
    port_id = make_port_id(host, port, protocol)
    service_id = make_service_id(host, port, protocol)

    return {
        "id": port_id,
        "host_id": host_id,
        "ip": host,
        "port": int(port),
        "protocol": protocol,
        "state": state,
        "service_id": service_id,
        "evidence_ids": [],
        "attempt_ids": [],
    }


def make_service(
    host: str,
    port: int,
    protocol: str = "tcp",
    name: Optional[str] = None,
    product: Optional[str] = None,
    version: Optional[str] = None,
    banner: Optional[str] = None,
) -> dict:
    host_id = make_host_id(host)
    port_id = make_port_id(host, port, protocol)
    service_id = make_service_id(host, port, protocol)

    return {
        "id": service_id,
        "host_id": host_id,
        "port_id": port_id,
        "ip": host,
        "port": int(port),
        "protocol": protocol,
        "name": name,
        "family": infer_service_family(name),
        "product": product,
        "version": version,
        "banner": banner,
        "technology": [],
        "enumerated": False,
        "vuln_checked": False,
        "credential_ids": [],
        "vulnerability_ids": [],
        "session_ids": [],
        "finding_ids": [],
        "evidence_ids": [],
        "attempt_ids": [],
    }


def make_vulnerability(
    name: str,
    host: str,
    port: Optional[int] = None,
    protocol: str = "tcp",
    service_id: Optional[str] = None,
    status: str = "candidate",
    source: Optional[str] = None,
    confidence: Optional[str] = "medium",
) -> dict:
    vulnerability_id = make_vulnerability_id(name, host, port, protocol)
    host_id = make_host_id(host)

    if service_id is None and port is not None:
        service_id = make_service_id(host, port, protocol)

    return {
        "id": vulnerability_id,
        "name": name,
        "status": status,
        "host_id": host_id,
        "service_id": service_id,
        "source": source,
        "confidence": confidence,
        "evidence_ids": [],
        "attempt_ids": [],
        "finding_ids": [],
    }


def make_credential(
    username: str,
    password: str,
    service: str,
    host: str,
    port: int,
    source: str,
    protocol: str = "tcp",
    valid: Optional[bool] = True,
) -> dict:
    credential_id = make_credential_id(host, service, username, port)
    host_id = make_host_id(host)
    service_id = make_service_id(host, port, protocol)

    return {
        "id": credential_id,
        "host_id": host_id,
        "service_id": service_id,
        "username": username,
        "password": password,
        "service": service,
        "host": host,
        "port": int(port),
        "protocol": protocol,
        "source": source,
        "valid": valid,
        "evidence_ids": [],
        "attempt_ids": [],
        "finding_ids": [],
    }


def make_session(
    session_type: str,
    host: str,
    port: int,
    service: str,
    tool: str,
    step_idx: int,
    protocol: str = "tcp",
    user: Optional[str] = None,
    privilege: Optional[str] = None,
    status: str = "opened",
    vulnerability_id: Optional[str] = None,
    credential_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
) -> dict:
    session_id = make_session_id(step_idx, host, port)
    host_id = make_host_id(host)
    service_id = make_service_id(host, port, protocol)

    return {
        "id": session_id,
        "type": session_type,
        "status": status,
        "host_id": host_id,
        "service_id": service_id,
        "vulnerability_id": vulnerability_id,
        "credential_id": credential_id,
        "attempt_id": attempt_id,
        "host": host,
        "port": int(port),
        "protocol": protocol,
        "service": service,
        "tool": tool,
        "user": user,
        "privilege": privilege,
        "hostname": None,
        "system": None,
        "opened_at_step": step_idx,
        "closed_at_step": None,
        "evidence_ids": [],
        "finding_ids": [],
    }


def make_attempt(
    step_idx: int,
    action_id: int,
    action_name: str,
    phase: Optional[str] = None,
    target_id: Optional[str] = None,
    host_id: Optional[str] = None,
    service_id: Optional[str] = None,
    vulnerability_id: Optional[str] = None,
    credential_id: Optional[str] = None,
    command: Optional[str] = None,
    rc: Optional[int] = None,
    success: Optional[bool] = None,
    event_types: Optional[list[str]] = None,
) -> dict:
    attempt_id = make_attempt_id(step_idx, action_id)

    return {
        "id": attempt_id,
        "step_idx": int(step_idx),
        "action_id": int(action_id),
        "action_name": action_name,
        "phase": phase,
        "target_id": target_id,
        "host_id": host_id,
        "service_id": service_id,
        "vulnerability_id": vulnerability_id,
        "credential_id": credential_id,
        "command": command,
        "rc": rc,
        "success": success,
        "event_types": event_types or [],
        "evidence_ids": [],
    }


def make_evidence(
    step_idx: int,
    kind: str,
    title: str,
    summary: Optional[str] = None,
    host_id: Optional[str] = None,
    service_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    event_ref: Optional[str] = None,
    command_ref: Optional[str] = None,
    suffix: Optional[str] = None,
) -> dict:
    evidence_id = make_evidence_id(step_idx, kind, suffix)

    return {
        "id": evidence_id,
        "step_idx": int(step_idx),
        "kind": kind,
        "title": title,
        "summary": summary,
        "host_id": host_id,
        "service_id": service_id,
        "attempt_id": attempt_id,
        "event_ref": event_ref,
        "command_ref": command_ref,
    }


def make_finding(
    title: str,
    severity: Optional[str],
    host: Optional[str] = None,
    port: Optional[int] = None,
    service: Optional[str] = None,
    source: Optional[str] = None,
    status: str = "confirmed",
    finding_type: str = "vulnerability",
) -> dict:
    finding_id = make_finding_id(title, host, port)

    host_ids = []
    service_ids = []

    if host is not None:
        host_ids.append(make_host_id(host))

    if host is not None and port is not None:
        service_ids.append(make_service_id(host, port))

    return {
        "id": finding_id,
        "title": title,
        "type": finding_type,
        "status": status,
        "severity": severity,
        "host_ids": host_ids,
        "service_ids": service_ids,
        "vulnerability_ids": [],
        "credential_ids": [],
        "session_ids": [],
        "evidence_ids": [],
        "source": source,
        "service": service,
        "description": None,
        "recommendation": None,
    }


def make_history_entry(
    step_idx: int,
    action_id: int,
    action_name: str,
    event_types: list[str],
    success: Optional[bool],
    progress: Optional[bool] = None,
) -> dict:
    return {
        "step_idx": int(step_idx),
        "action_id": int(action_id),
        "action_name": action_name,
        "event_types": event_types,
        "success": success,
        "progress": progress,
    }

def make_step(
    step_idx: int,
    action_id: int | None,
    action_name: str | None,
    command: str | None = None,
    phase: str | None = None,
    target_id: str | None = None,
    host_id: str | None = None,
    port_id: str | None = None,
    service_id: str | None = None,
    vulnerability_id: str | None = None,
    credential_id: str | None = None,
    session_id: str | None = None,
    rc: int | None = None,
    success: bool | None = None,
    progress: bool | None = None,
    event_types: list[str] | None = None,
) -> dict:
    return {
        "step_idx": int(step_idx),
        "action_id": action_id,
        "action_name": action_name,
        "phase": phase,
        "target_id": target_id,
        "host_id": host_id,
        "port_id": port_id,
        "service_id": service_id,
        "vulnerability_id": vulnerability_id,
        "credential_id": credential_id,
        "session_id": session_id,
        "command": command,
        "rc": rc,
        "success": success,
        "progress": progress,
        "event_types": event_types or [],
    }

# ============================================================
# Small helpers
# ============================================================

def infer_service_family(service_name: Optional[str]) -> Optional[str]:
    if service_name is None:
        return None

    name = service_name.lower()

    if "http" in name or "www" in name:
        return "http"

    if "ftp" in name:
        return "ftp"

    if "ssh" in name:
        return "ssh"

    if "smb" in name or "microsoft-ds" in name or "netbios" in name:
        return "smb"

    if "mysql" in name:
        return "mysql"

    if "postgres" in name or "postgresql" in name:
        return "postgresql"

    if "distcc" in name or "distccd" in name:
        return "distcc"

    if "irc" in name:
        return "irc"

    if "tomcat" in name:
        return "tomcat"

    return name


def refresh_stats(kb: dict) -> None:
    """
    Recalcula contadores básicos de la KB.

    Esta función no sustituye a contadores específicos de ejecución,
    pero evita que stats quede desincronizado respecto a las entidades.
    """
    target = kb.get("target", {})

    kb.setdefault("stats", {})
    kb["stats"]["networks"] = len(target.get("networks", {}))
    kb["stats"]["hosts"] = len(target.get("hosts", {}))
    kb["stats"]["ports"] = len(target.get("ports", {}))
    kb["stats"]["services"] = len(target.get("services", {}))
    kb["stats"]["vulns"] = len(kb.get("vulnerabilities", {}))
    kb["stats"]["credentials"] = len(kb.get("credentials", {}))
    kb["stats"]["sessions"] = len(kb.get("sessions", {}))
    kb["stats"]["attempts"] = len(kb.get("attempts", {}))
    kb["stats"]["evidence"] = len(kb.get("evidence", {}))
    kb["stats"]["findings"] = len(kb.get("findings", {}))
    kb["stats"]["steps"] = len(kb.get("steps", []))
    kb["stats"]["commands"] = count_commands(kb)

def ensure_kb_collections(kb: dict) -> None:
    """
    Garantiza que existan las colecciones principales.
    Útil para cargar KB antiguas o parcialmente creadas.
    """
    kb.setdefault("target", {})
    kb["target"].setdefault("networks", {})
    kb["target"].setdefault("hosts", {})
    kb["target"].setdefault("ports", {})
    kb["target"].setdefault("services", {})

    kb.setdefault("vulnerabilities", {})
    kb.setdefault("credentials", {})
    kb.setdefault("sessions", {})
    kb.setdefault("attempts", {})
    kb.setdefault("evidence", {})
    kb.setdefault("findings", {})
    kb.setdefault("coverage", {})
    kb.setdefault("history", [])
    kb.setdefault("steps", [])
    kb.setdefault("last", {})
    kb.setdefault("focus", {})
    kb.setdefault("stats", {})

def count_commands(kb: dict) -> int:
    total = 0

    for step in kb.get("steps", []):
        if step.get("command"):
            total += 1

    return total