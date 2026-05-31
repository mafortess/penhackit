
from penhackit.session.action.action_ids import ACTIONS
import re

def command_builder(action_data, kb: dict) -> dict | None:
    print("Building command from action and KB...")

    cmd_template = action_data.get("command_template")
    if not cmd_template:
        print(f"No command template found for action: {action_data.get('name')}")
        return None

    # Reemplaza placeholders en cmd con datos de KB (ejemplo simple)
    # if "{" not in cmd_template:
    #     print("No placeholders in command template, returning as is.")
    #     return cmd_template
    try:
        values = build_placeholder_values(kb, action_data)
    except Exception as e:
        print(f"Cannot build command for {action_data.get('name')}: {e}")
        return None
    
    try:
        cmd = cmd_template.format(**values)  # Reemplaza placeholders en el comando
    except  KeyError as e:
        print(f"Missing value for placeholder: {e}")
        return None
    
    if "None" in cmd:
        print(f"Command has unresolved placeholders after formatting: {cmd}")
        return None
    
    return {
        "command": cmd,
        "action_name": action_data.get("name"),
        "parser_family": action_data.get("parser_family"),
        "target": values.get("target"),
        "target_ip": values.get("target_ip"),
        "target_port": values.get("target_port"),
        "known_open_ports_csv": values.get("known_open_ports_csv"),
        "service_version_string": values.get("service_version_string"),
        "service_name": values.get("service_name"),
        "exploit": action_data.get("name"),
    }


def build_placeholder_values(kb: dict, action_data: dict) -> dict:
    return {
        "ip": resolve_target_ip(kb),
        "target": resolve_target(kb),
        "target_ip": resolve_target_ip(kb),
        "target_port": resolve_target_port(kb, action_data),
        "known_open_ports_csv": resolve_known_open_ports_csv(kb),
        "service_version_string": resolve_service_version_string(kb),
        # "service_name": resolve_service_name(kb, action_data),
        "lhost": resolve_lhost(kb, action_data),
        "lport": resolve_lport(action_data),
        "username": resolve_username(kb, action_data),
        "password": resolve_password(kb, action_data),
        "userlist_path": resolve_userlist_path(kb, action_data),
        "passwordlist_path": resolve_passwordlist_path(kb, action_data),
    }

def resolve_target(kb: dict) -> str | None:
    """
    Para acciones tipo:    nmap -sn {target}
    """
    scope = kb.get("scope", {})
    target = scope.get("target")
    if target:
        return target

    raise ValueError("Missing target: expected kb['scope']['target']")


def resolve_target_ip(kb: dict) -> str | None:
    """
    Devuelve un host objetivo.
    Funciona tanto si kb["hosts"] es dict como si es list.
    """
    scope = kb.get("scope", {})
    target = scope.get("target")
    target_type = scope.get("target_type")

    if target_type == "host":
        return target

    if target_type == "network":
        return target
        # return find_first_host_in_kb(kb)

    return None


def find_first_host_in_kb(kb: dict) -> str | None:
    networks = kb.get("target", {}).get("networks", {})

    for network in networks.values():
        hosts = network.get("hosts", {})
        for ip in hosts.keys():
            return ip

    return None

# def resolve_target_ip(kb: dict) -> str | None:
#     """
#     Devuelve un host objetivo.
#     Funciona tanto si kb["hosts"] es dict como si es list.
#     """
#     focus = kb.get("focus", {})
#     if focus.get("host"):
#         return focus["host"]

#     hosts = kb.get("hosts", {})

#     if isinstance(hosts, dict):
#         for ip in hosts.keys():
#             if ip and not ip.endswith(".1"):
#                 return ip

#     if isinstance(hosts, list):
#         for host in hosts:
#             ip = host.get("ip")
#             if ip and not ip.endswith(".1"):
#                 return ip

#     return None


def resolve_target_port(kb: dict, action_data: dict) -> int | None:
    service_names = [s.lower() for s in action_data.get("service_names", [])]

    target_ip = resolve_target_ip(kb)
    host = get_host(kb, target_ip) if target_ip else None

    if host:
        for port, port_data in host.get("ports", {}).items():
            service = port_data.get("service", {})
            text = " ".join([
                str(service.get("name", "")),
                str(service.get("product", "")),
                str(service.get("version", "")),
                str(service.get("banner", "")),
            ]).lower()

            if any(name in text for name in service_names):
                return int(port)

    return action_data.get("default_port")

# def resolve_target_port(kb: dict, action_data: dict) -> int | None:
#     """
#     Prioridad:
#     1. puerto HTTP/HTTPS si existe
#     2. primer puerto abierto
#     """
#     target_ip = resolve_target_ip(kb)
#     if not target_ip:
#         return None

#     host = get_host(kb, target_ip)
#     if not host:
#         return None

#     ports = host.get("ports", {})

#     for port, data in ports.items():
#         service = data.get("service", "")
#         if service in {"http", "https", "http-alt"}:
#             return int(port)

#     for port in ports.keys():
#         return int(port)

#     return None


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


def resolve_lhost(kb: dict, action_data: dict) -> str | None:
    if action_data.get("default_lhost"):
        return action_data.get("default_lhost")

    attacker = kb.get("attacker", {})
    ipv4_list = attacker.get("ipv4", [])

    if ipv4_list:
        first = ipv4_list[0]

        if isinstance(first, dict):
            return first.get("ip")

        if isinstance(first, str):
            return first

    # fallback para VirtualBox NAT típico
    return "10.6.6.1"


def resolve_lport(action_data: dict) -> int:
    return int(action_data.get("default_lport", 4444))


def resolve_username(kb: dict, action_data: dict) -> str | None:
    if action_data.get("default_username"):
        return action_data.get("default_username")

    credential = find_first_credential(kb)
    if credential:
        return credential.get("username")

    return None


def resolve_password(kb: dict, action_data: dict) -> str | None:
    if action_data.get("default_password"):
        return action_data.get("default_password")

    credential = find_first_credential(kb)
    if credential:
        return credential.get("password")

    return None


def resolve_userlist_path(kb: dict, action_data: dict) -> str:
    return action_data.get("default_userlist_path", "users.txt")


def resolve_passwordlist_path(kb: dict, action_data: dict) -> str:
    return action_data.get("default_passwordlist_path", "passwords.txt")


def find_first_credential(kb: dict) -> dict | None:
    networks = kb.get("target", {}).get("networks", {})

    for network in networks.values():
        for host in network.get("hosts", {}).values():
            for port_data in host.get("ports", {}).values():
                for credential in port_data.get("credentials", []):
                    if credential.get("username") and credential.get("password"):
                        return credential

    return None



def get_host(kb: dict, target_ip: str) -> dict | None:
    networks = kb.get("target", {}).get("networks", {})

    for network in networks.values():
        hosts = network.get("hosts", {})
        if target_ip in hosts:
            return hosts[target_ip]

    return None