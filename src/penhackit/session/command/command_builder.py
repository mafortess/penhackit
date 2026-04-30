
from penhackit.session.action.action_ids import ACTIONS
import re

def command_builder(action_data, kb: dict) -> dict | None:
    print("Building command from action and KB...")
    cmd_template = action_data.get("command_template")
    # try:
    #     action_data = ACTIONS.get(action_id)
    #     if not action_data:
    #         print(f"Action ID {action_id} not found in catalog.")
    #         return None
        
        # name, cmd_template = action_data.get("name"), action_data.get("command_template")
    
    # except Exception as e:
    #     print(f"Error retrieving action from catalog: {e}")
    #     return None

    if not cmd_template:
        print(f"No command template found for action: {action_data.get('name')}")
        return None

    # Reemplaza placeholders en cmd con datos de KB (ejemplo simple)
    # if "{" not in cmd_template:
    #     print("No placeholders in command template, returning as is.")
    #     return cmd_template
    
    values = build_placeholder_values(kb)

    try:
        cmd = cmd_template.format(**values)  # Reemplaza placeholders en el comando
    except  KeyError as e:
        print(f"Missing value for placeholder: {e}")
        return None
    
    if "None" in cmd:
        print(f"Command has unresolved placeholders after formatting: {cmd}")
        return None
    
    # try:
    #     # Extrae los placeholders del comando
    #     hosts = kb.get("hosts", [])
    #     ip = hosts[0].get("ip", None) if hosts else None  # Ejemplo: toma la primera IP de la KB
        
    #     if "{ip}" in cmd_template and not ip:
    #         print("No IP available in KB to build command.")
    #         return None
    # except Exception as e:
    #     print(f"Error extracting data from KB: {e}")
    #     return None
    
    return {
        "command": cmd,
        "parser_family": action_data.get("parser_family"),
        "target": values.get("target"),
        "target_ip": values.get("target_ip"),
        "target_port": values.get("target_port"),
        "known_open_ports_csv": values.get("known_open_ports_csv"),
        "service_version_string": values.get("service_version_string"),
    }


def build_placeholder_values(kb: dict) -> dict:
    return {
        "ip": resolve_target_ip(kb),
        "target": resolve_target(kb),
        "target_ip": resolve_target_ip(kb),
        "target_port": resolve_target_port(kb),
        "known_open_ports_csv": resolve_known_open_ports_csv(kb),
        "service_version_string": resolve_service_version_string(kb),
    }

def resolve_target(kb: dict) -> str | None:
    """
    Para acciones tipo:
    nmap -sn {target}

    Prioridad:
    1. kb["scope"]["target"]
    2. kb["target"]                # compatibilidad antigua
    3. kb["session_context"]["target"]
    4. error explícito
    
    """
    if kb.get("target"):
        return kb["target"]

    scope = kb.get("scope", {})
    if scope.get("target"):
        return scope["target"]

    session_context = kb.get("session_context", {})
    if session_context.get("target"):
        return session_context["target"]

    raise ValueError("Missing target: expected kb['scope']['target']")


def resolve_target_ip(kb: dict) -> str | None:
    """
    Devuelve un host objetivo.
    Funciona tanto si kb["hosts"] es dict como si es list.
    """
    focus = kb.get("focus", {})
    if focus.get("host"):
        return focus["host"]

    hosts = kb.get("hosts", {})

    if isinstance(hosts, dict):
        for ip in hosts.keys():
            if ip and not ip.endswith(".1"):
                return ip

    if isinstance(hosts, list):
        for host in hosts:
            ip = host.get("ip")
            if ip and not ip.endswith(".1"):
                return ip

    return None


def resolve_target_port(kb: dict) -> int | None:
    """
    Prioridad:
    1. puerto HTTP/HTTPS si existe
    2. primer puerto abierto
    """
    target_ip = resolve_target_ip(kb)
    if not target_ip:
        return None

    host = get_host(kb, target_ip)
    if not host:
        return None

    ports = host.get("ports", {})

    for port, data in ports.items():
        service = data.get("service", "")
        if service in {"http", "https", "http-alt"}:
            return int(port)

    for port in ports.keys():
        return int(port)

    return None


def resolve_known_open_ports_csv(kb: dict) -> str | None:
    target_ip = resolve_target_ip(kb)
    if not target_ip:
        return None

    host = get_host(kb, target_ip)
    if not host:
        return None

    ports = host.get("ports", {})
    if not ports:
        return None

    return ",".join(str(port) for port in ports.keys())


def resolve_service_version_string(kb: dict) -> str | None:
    target_ip = resolve_target_ip(kb)
    if not target_ip:
        return None

    host = get_host(kb, target_ip)
    if not host:
        return None

    services = host.get("services", {})

    for service_data in services.values():
        service = service_data.get("service", "")
        version = service_data.get("version", "")

        if service and version:
            return f"{service} {version}"

    for service_data in services.values():
        service = service_data.get("service", "")
        if service:
            return service

    return None


def get_host(kb: dict, target_ip: str) -> dict | None:
    hosts = kb.get("hosts", {})

    if isinstance(hosts, dict):
        return hosts.get(target_ip)

    if isinstance(hosts, list):
        for host in hosts:
            if host.get("ip") == target_ip:
                return host

    return None