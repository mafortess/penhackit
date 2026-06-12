
import re
from typing import Any, Optional

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
        validate_required_placeholders(values, action_data)
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
    
    target_ip = values.get("target_ip")
    target_port = values.get("target_port")

    return {
        "command": cmd,
        "action_name": action_data.get("name"),
        "parser_family": action_data.get("parser_family"),
        "target": values.get("target"),
        "target_ip": target_ip,
        "target_port": target_port,
        "known_open_ports_csv": values.get("known_open_ports_csv"),
        "service_version_string": values.get("service_version_string"),
        "service_name": values.get("service_name"),
        "exploit": action_data.get("name"),
        "phase": action_data.get("phase"),
        "host_id": make_host_id_from_ip(target_ip),
        "port_id": make_port_id_from_values(target_ip, target_port),
        "service_id": make_service_id_from_values(target_ip, target_port),
        "vulnerability_id": values.get("vulnerability_id"),
        "credential_id": values.get("credential_id"),
    }


def build_placeholder_values(kb: dict, action_data: dict) -> dict:
    return {
        "ip": resolve_target_ip(kb),
        "target": resolve_target(kb),
        "target_ip": resolve_target_ip(kb),
        "target_port": resolve_target_port(kb, action_data),
        "known_open_ports_csv": resolve_known_open_ports_csv(kb),
        "service_version_string": resolve_service_version_string(kb, action_data),
        "service_name": resolve_service_name(kb, action_data),
        "lhost": resolve_lhost(kb, action_data),
        "lport": resolve_lport(action_data),
        "username": resolve_username(kb, action_data),
        "password": resolve_password(kb, action_data),
        "userlist_path": resolve_userlist_path(kb, action_data),
        "passwordlist_path": resolve_passwordlist_path(kb, action_data),
        "vulnerability_id": resolve_vulnerability_id(kb, action_data),
        "credential_id": resolve_credential_id(kb, action_data),
    }

def validate_required_placeholders(values: dict, action_data: dict) -> None:
    placeholders = action_data.get("placeholders", [])

    for placeholder in placeholders:
        value = values.get(placeholder)

        if value is None or value == "":
            raise ValueError(
                f"Missing required placeholder '{placeholder}' for action {action_data.get('name')}"
            )

    if "target_ip" in placeholders:
        target_ip = values.get("target_ip")
        if target_ip and "/" in str(target_ip):
            raise ValueError(
                f"Invalid target_ip '{target_ip}'. Port/service actions require a concrete host, not a network."
            )


def resolve_target(kb: dict) -> str | None:
    """
    Para acciones tipo:    nmap -sn {target}
    """
    scope = kb.get("scope", {})
    target = scope.get("target")
    if target:
        return target

    raise ValueError("Missing target: expected kb['scope']['target']")


def resolve_target_ip(kb: dict) -> Optional[str]:
    """
    Devuelve un host objetivo concreto.

    Regla:
    - Si target_type == host, devuelve scope.target.
    - Si target_type == network, NO devuelve la red.
      Devuelve el host enfocado o el primer host conocido en la KB.
    """
    scope = kb.get("scope", {})
    target = scope.get("target")
    target_type = scope.get("target_type")

    focus = kb.get("focus", {})
    focus_host_id = focus.get("host_id")

    if focus_host_id:
        host = get_host_by_id(kb, focus_host_id)
        if host:
            return host.get("ip")

        if str(focus_host_id).startswith("host:"):
            return str(focus_host_id).replace("host:", "", 1)

    if target_type == "host":
        return target

    if target_type == "network":
        return find_first_host_in_kb(kb)

    return None


def find_first_host_in_kb(kb: dict) -> Optional[str]:
    hosts = kb.get("target", {}).get("hosts", {})

    for host in hosts.values():
        ip = host.get("ip")
        if not ip:
            continue

        if str(ip).endswith(".1"):
            continue

        if host.get("alive") is False:
            continue

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


def resolve_target_port(kb: dict, action_data: dict) -> Optional[int]:
    target_ip = resolve_target_ip(kb)

    service_names = [
        str(s).lower()
        for s in action_data.get("service_names", [])
        if s
    ]

    default_port = action_data.get("default_port")

    # 1. Si la acción declara service_names, manda la acción, no el foco.
    if target_ip and service_names:
        service = find_service_by_names(kb, target_ip, service_names)

        if service and service.get("port") is not None:
            return int(service.get("port"))

        # Si no encuentra servicio compatible, usar default_port.
        # NO caer al foco, porque puede estar en FTP/vsftpd, SMB, etc.
        if default_port is not None:
            return int(default_port)

        return None

    # 2. Si no hay service_names, usar foco.
    focus = kb.get("focus", {})

    focus_port_id = focus.get("port_id")
    if focus_port_id:
        port_obj = kb.get("target", {}).get("ports", {}).get(focus_port_id)
        if port_obj and port_obj.get("port") is not None:
            return int(port_obj.get("port"))

    focus_service_id = focus.get("service_id")
    if focus_service_id:
        service_obj = kb.get("target", {}).get("services", {}).get(focus_service_id)
        if service_obj and service_obj.get("port") is not None:
            return int(service_obj.get("port"))

    focus_vuln_id = focus.get("vulnerability_id")
    if focus_vuln_id:
        vuln_obj = kb.get("vulnerabilities", {}).get(focus_vuln_id)
        if vuln_obj and vuln_obj.get("port") is not None:
            return int(vuln_obj.get("port"))

    if default_port is not None:
        return int(default_port)

    return None

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


def resolve_known_open_ports_csv(kb: dict) -> Optional[str]:
    target_ip = resolve_target_ip(kb)
    if not target_ip:
        return None

    ports = []

    # Fuente principal KB v2: índice global de puertos
    for port_obj in kb.get("target", {}).get("ports", {}).values():
        if port_obj.get("ip") != target_ip:
            continue

        if port_obj.get("state") != "open":
            continue

        port = port_obj.get("port")
        if port is not None:
            ports.append(int(port))

    # Fuente secundaria: host["port_ids"]
    if not ports:
        host = get_host(kb, target_ip)

        if host:
            for port_id in host.get("port_ids", []):
                port_obj = kb.get("target", {}).get("ports", {}).get(port_id)
                if not port_obj:
                    continue

                if port_obj.get("state") != "open":
                    continue

                port = port_obj.get("port")
                if port is not None:
                    ports.append(int(port))

    if not ports:
        return None

    return ",".join(str(port) for port in sorted(set(ports)))


def resolve_service_version_string(kb: dict, action_data: dict) -> Optional[str]:
    target_ip = resolve_target_ip(kb)
    if not target_ip:
        return None

    service_names = [
        str(s).lower()
        for s in action_data.get("service_names", [])
        if s
    ]

    services = get_services_for_host(kb, target_ip)

    if service_names:
        service = find_service_by_names(kb, target_ip, service_names)
        if not service:
            return None

        services = [service]

    for service in services:
        name = service.get("name") or service.get("service")
        product = service.get("product")
        version = service.get("version")
        banner = service.get("banner")

        text = " ".join(
            str(x)
            for x in [name, product, version, banner]
            if x
        ).strip()

        if text:
            return text

    return None


def resolve_service_name(kb: dict, action_data: dict) -> Optional[str]:
    target_ip = resolve_target_ip(kb)

    service_names = [
        str(s).lower()
        for s in action_data.get("service_names", [])
        if s
    ]

    # Si la acción declara service_names, no devolver servicio del foco si no coincide.
    if target_ip and service_names:
        service = find_service_by_names(kb, target_ip, service_names)

        if service:
            return service.get("name") or service.get("service") or service.get("family")

        return service_names[0]

    focus = kb.get("focus", {})
    service_id = focus.get("service_id")

    if service_id:
        service = kb.get("target", {}).get("services", {}).get(service_id)
        if service:
            return service.get("name") or service.get("service") or service.get("family")

    if not target_ip:
        return None

    services = get_services_for_host(kb, target_ip)

    for service in services:
        name = service.get("name") or service.get("service") or service.get("family")
        if name:
            return name

    return None


def resolve_lhost(kb: dict, action_data: dict) -> str:
    if action_data.get("default_lhost"):
        return action_data.get("default_lhost")

    attacker = kb.get("attacker", {})
    ipv4_list = attacker.get("ipv4", [])

    for item in ipv4_list:
        if isinstance(item, dict):
            ip = item.get("ip")
            if ip:
                return ip

        if isinstance(item, str):
            return item
    interfaces = attacker.get("interfaces", [])
    for interface in interfaces:
        for ipv4 in interface.get("ipv4", []):
            ip = ipv4.get("ip")
            if ip and not ip.startswith("127."):
                return ip

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

def resolve_vulnerability_id(kb: dict, action_data: dict) -> Optional[str]:
    target_ip = resolve_target_ip(kb)
    target_port = resolve_target_port(kb, action_data)

    exploit_name = action_data.get("exploit_name")

    if target_ip and target_port and exploit_name:
        return f"vuln:{exploit_name}:{target_ip}:tcp:{int(target_port)}"

    # Si la acción tiene service_names, no heredar vulnerability_id del foco.
    # Evita cosas como UnrealIRCd usando vuln de vsftpd:21.
    if action_data.get("service_names"):
        return None

    focus = kb.get("focus", {})
    if focus.get("vulnerability_id"):
        return focus.get("vulnerability_id")

    return None
    
def resolve_credential_id(kb: dict, action_data: dict) -> Optional[str]:
    focus = kb.get("focus", {})
    if focus.get("credential_id"):
        return focus.get("credential_id")

    credential = find_first_credential(kb)
    if credential:
        return credential.get("id")

    return None


def find_first_credential(kb: dict) -> dict | None:
    for credential in kb.get("credentials", {}).values():
        username = credential.get("username")
        password = credential.get("password")

        if username and password:
            return credential

    return None


def get_host(kb: dict, target_ip: str) -> dict | None:
    host_id = make_host_id_from_ip(target_ip)

    if host_id:
        host = kb.get("target", {}).get("hosts", {}).get(host_id)
        if host:
            return host

    for host in kb.get("target", {}).get("hosts", {}).values():
        if host.get("ip") == target_ip:
            return host

    return None


def get_host_by_id(kb: dict, host_id: str) -> Optional[dict]:
    return kb.get("target", {}).get("hosts", {}).get(host_id)


def get_services_for_host(kb: dict, target_ip: str) -> list[dict]:
    host = get_host(kb, target_ip)
    if not host:
        return []

    service_ids = host.get("service_ids", [])
    services = []

    for service_id in service_ids:
        service = kb.get("target", {}).get("services", {}).get(service_id)
        if service:
            services.append(service)

    if services:
        return services

    for service in kb.get("target", {}).get("services", {}).values():
        if service.get("ip") == target_ip:
            services.append(service)

    return services


def find_service_by_names(kb: dict, target_ip: str, service_names: list[str]) -> Optional[dict]:
    services = get_services_for_host(kb, target_ip)

    for service in services:
        text = " ".join([
            str(service.get("name", "")),
            str(service.get("service", "")),
            str(service.get("product", "")),
            str(service.get("version", "")),
            str(service.get("banner", "")),
            str(service.get("family", "")),
        ]).lower()

        if any(name in text for name in service_names):
            return service

    return None


def make_host_id_from_ip(ip: Optional[str]) -> Optional[str]:
    if not ip:
        return None

    if str(ip).startswith("host:"):
        return str(ip)

    return f"host:{ip}"


def make_port_id_from_values(ip: Optional[str], port: Optional[int], proto: str = "tcp") -> Optional[str]:
    if not ip or port is None:
        return None

    return f"port:{ip}:{proto}:{int(port)}"


def make_service_id_from_values(ip: Optional[str], port: Optional[int], proto: str = "tcp") -> Optional[str]:
    if not ip or port is None:
        return None

    return f"svc:{ip}:{proto}:{int(port)}"