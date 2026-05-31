# estructura inicial de la KB
# - constructores make_network, make_host, make_port, make_session...

KB_SCHEMA_VERSION = "1.0"

def build_initial_kb(session_context: dict) -> dict:
    """
    Construye una KB inicial vacía o con datos predeterminados para el inicio de la sesión.
    """
    session_id = session_context.get("id")
    target_type = session_context.get("target_type")
    target = session_context.get("target")
    goal_type = session_context.get("goal_type")
    name = session_context.get("name")
    return {
        # Version of the KB schema
        "schema_version": KB_SCHEMA_VERSION,

         # Current pentesting session metadata.
        "session": {
            "id": session_id,
            "name_enterprise": name,
        },

        "scope": {
            "target_type": target_type or "host",
            "target": target or "10.6.6.10",
            # High-level objective of the session, e.g. obtain_session.
            "goal": goal_type or "obtain session",
    
            # "target_type": target_type or "network",
            # "target": target or "10.7.7.0/24",
        },
        
        # Local attacker/Kali context: interfaces, routes, ARP table and LHOST.
        "attacker": {
            "hostname": None,
            "interfaces": [],
            "ipv4": [],
            "default_gw": [],
            "arp_neighbors": [],
            "routes": [],
        },

        # Access-related knowledge discovered during the session.
        # Discovered target-side information. target.networks[cidr].hosts[ip].ports[port]
        # Jerarquía de redes -> hosts -> puertos -> servicios, con datos asociados (alive, banners, etc.)
        "target": {
            "networks": {},  # cidr -> { "hosts": { ip -> { "alive": bool, "ports": { port -> { "state": "open|closed|filtered", "service": str, ... } }, ... } }, ... }
        },

        # Security findings relevant for the final report.
        # Examples: weak credentials, vulnerable service, exposed bind shell.
        "findings": [],
        
        # Compact semantic history. Full stdout/stderr must go to logs.
        "history": [],

        # Counters useful for progress tracking and later state building.
        "stats": {
            "steps": 0,
            "networks": 0,
            "hosts": 0,
            "ports": 0,
            "services": 0,
            "vulns": 0,
            "credentials": 0,
            "sessions": 0,
            "exploits_attempted": 0,
            "exploits_failed": 0,
        },

        # Summary of the latest action/result.
        # Useful for debugging and for state features such as last_action_id.
        "last":{
            "step_idx": 0,
            "action_id": None,
            "action_name": None,
            "rc": None,
            "success": None,
            "event_types": None,
        },

    
        "focus": {
            "level": "global", 
            "host": "", 
            "service": ""
        },       
    }

def make_network(cidr: str) -> dict:
    return {
        "cidr": cidr,
        "host_discovery_done": False,
        "hosts": {},
    }


def make_host(ip: str) -> dict:
    return {
        "ip": ip,
        "hostname": None,
        "alive": True,
        "ports": {},
        "findings": [],
    }


def make_port(port: int, protocol: str = "tcp", state: str = "open") -> dict:
    return {
        "port": int(port),
        "protocol": protocol,
        "state": state,

        "service": {
            "name": None,
            "product": None,
            "version": None,
            "banner": None,
        },

        "candidate_vulns": [],
        "credentials": [],
        "sessions": [],
        "attempts": [],
    }


def make_credential(
    username: str,
    password: str,
    service: str,
    host: str,
    port: int,
    source: str,
) -> dict:
    return {
        "username": username,
        "password": password,
        "service": service,
        "host": host,
        "port": int(port),
        "source": source,
        "valid": True,
    }


def make_session(
    session_type: str,
    host: str,
    port: int,
    service: str,
    tool: str,
    user: str | None = None,
    privilege: str | None = None,
    closed: bool = True,
) -> dict:
    return {
        "type": session_type,
        "host": host,
        "port": int(port),
        "service": service,
        "tool": tool,
        "user": user,
        "privilege": privilege,
        "closed": closed,
        "evidence": {},
    }


def make_finding(
    title: str,
    severity: str,
    host: str | None = None,
    port: int | None = None,
    service: str | None = None,
    source: str | None = None,
) -> dict:
    return {
        "title": title,
        "severity": severity,
        "host": host,
        "port": port,
        "service": service,
        "source": source,
    }


def make_history_entry(
    step_idx: int,
    action_id: int,
    action_name: str,
    event_types: list[str],
    success: bool | None,
) -> dict:
    return {
        "step_idx": step_idx,
        "action_id": action_id,
        "action_name": action_name,
        "event_types": event_types,
        "success": success,
    }