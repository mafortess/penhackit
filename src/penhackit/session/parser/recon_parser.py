# recon_parser.py

import re

# ============================================================
# RECON / DISCOVERY
# ============================================================

def parse_discover_hosts_nmap_ping_sweep(stdout: str) -> list[dict]:
    return parse_nmap_host_discovery(stdout)


def parse_discover_hosts_arp_localnet(stdout: str) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+"
            r"(?P<mac>[0-9a-fA-F:]{17})\s+"
            r"(?P<vendor>.*)$",
            line,
        )

        if not m:
            continue

        events.append({
            "type": "HOST_DISCOVERED",
            "host": m.group("ip"),
            "mac": m.group("mac"),
            "vendor": m.group("vendor").strip(),
            "discovery_method": "arp-scan",
        })

    return events


def parse_discover_hosts_arp_range(stdout: str, target: str | None) -> list[dict]:
    events = parse_discover_hosts_arp_localnet(stdout)

    for event in events:
        event["target"] = target

    return events


def parse_discover_hosts_netdiscover(stdout: str) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+"
            r"(?P<mac>[0-9a-fA-F:]{17})\s+"
            r"\d+\s+\d+\s+"
            r"(?P<vendor>.*)$",
            line,
        )

        if not m:
            continue

        events.append({
            "type": "HOST_DISCOVERED",
            "host": m.group("ip"),
            "mac": m.group("mac"),
            "vendor": m.group("vendor").strip(),
            "discovery_method": "netdiscover",
        })

    return events


def parse_discover_hosts_fping(stdout: str) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})$", line)
        if not m:
            continue

        events.append({
            "type": "HOST_DISCOVERED",
            "host": m.group("ip"),
            "discovery_method": "fping",
        })

    return events



# ============================================================
# PORT SCANNING
# ============================================================

def parse_scan_top_tcp_ports(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_portscan(stdout, target_ip)


def parse_scan_full_tcp_ports(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_portscan(stdout, target_ip)


def parse_scan_quick_tcp_ports(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_portscan(stdout, target_ip)


def parse_scan_top_udp_ports(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_portscan(stdout, target_ip)


# ============================================================
# SERVICE DETECTION
# ============================================================

def parse_detect_services(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_service_detection(stdout, target_ip)


def parse_detect_services_light(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_service_detection(stdout, target_ip)


def parse_detect_services_aggressive(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_nmap_service_detection(stdout, target_ip)


def parse_enum_nmap_default_scripts(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    current_port = None
    current_script = None
    buffer = []

    for line in stdout.splitlines():
        line_stripped = line.strip()

        port_match = re.match(r"^(\d+)\/(tcp|udp)\s+open\s+(\S+)", line_stripped)
        if port_match:
            if current_script and buffer:
                events.append({
                    "type": "SCRIPT_RESULT",
                    "host": target_ip,
                    "port": current_port,
                    "script": current_script,
                    "output": "\n".join(buffer).strip()[:2000],
                })

            current_port = int(port_match.group(1))
            current_script = None
            buffer = []
            continue

        script_match = re.match(r"^\|_?(?P<script>[a-zA-Z0-9_\-]+):\s*(?P<text>.*)", line)
        if script_match:
            if current_script and buffer:
                events.append({
                    "type": "SCRIPT_RESULT",
                    "host": target_ip,
                    "port": current_port,
                    "script": current_script,
                    "output": "\n".join(buffer).strip()[:2000],
                })

            current_script = script_match.group("script")
            buffer = [script_match.group("text")]
            continue

        if current_script and line.startswith("|"):
            buffer.append(line.strip("|_ "))

    if current_script and buffer:
        events.append({
            "type": "SCRIPT_RESULT",
            "host": target_ip,
            "port": current_port,
            "script": current_script,
            "output": "\n".join(buffer).strip()[:2000],
        })

    return events

# Funciones de parsing específicas para cada comando (ejemplo MVP)
def parse_nmap_host_discovery(stdout: str) -> list[dict]:
    events = []

    for match in re.finditer(r"Nmap scan report for (?P<ip>\d{1,3}(?:\.\d{1,3}){3})", stdout):
        ip = match.group("ip")

        # Ignorar gateways Docker típicos
        if ip.endswith(".1"):
            continue

        events.append({
            "type": "HOST_DISCOVERED",
            "host": ip,
        })

    return events


def parse_nmap_portscan(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<port>\d+)\/(?P<proto>tcp|udp)\s+open\s+(?P<service>\S+)",
            line,
        )

        if not m:
            continue

        events.append({
            "type": "PORT_OPEN",
            "host": target_ip,
            "port": int(m.group("port")),
            "proto": m.group("proto"),
            "service": m.group("service"),
        })

    return events


def parse_nmap_service_detection(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<port>\d+)\/(?P<proto>tcp|udp)\s+open\s+(?P<service>\S+)\s*(?P<version>.*)$",
            line,
        )

        if not m:
            continue

        port = int(m.group("port"))
        proto = m.group("proto")
        service = m.group("service")
        version = m.group("version").strip()

        events.append({
            "type": "SERVICE_DETECTED",
            "host": target_ip,
            "port": port,
            "proto": proto,
            "service": service,
        })

        if version:
            events.append({
                "type": "SERVICE_VERSION_DETECTED",
                "host": target_ip,
                "port": port,
                "proto": proto,
                "service": service,
                "version": version,
            })

    return events


RECON_PARSERS = {
    # ============================================================
    # By action name lower()
    # ============================================================
    "discover_hosts": parse_discover_hosts_nmap_ping_sweep,
    "discover_hosts_nmap_ping_sweep": parse_discover_hosts_nmap_ping_sweep,
    "discover_hosts_arp_localnet": parse_discover_hosts_arp_localnet,
    "discover_hosts_arp_range": parse_discover_hosts_arp_range,
    "discover_hosts_netdiscover": parse_discover_hosts_netdiscover,
    "discover_hosts_fping": parse_discover_hosts_fping,

    "scan_top_tcp_ports": parse_scan_top_tcp_ports,
    "scan_full_tcp_ports": parse_scan_full_tcp_ports,
    "scan_quick_tcp_ports": parse_scan_quick_tcp_ports,
    "scan_top_udp_ports": parse_scan_top_udp_ports,

    "detect_services": parse_detect_services,
    "detect_services_light": parse_detect_services_light,
    "detect_services_aggressive": parse_detect_services_aggressive,

    "enum_nmap_default_scripts": parse_enum_nmap_default_scripts,

    # ============================================================
    # By parser_family
    # ============================================================
    "nmap_host_discovery": parse_nmap_host_discovery,
    "arp_scan": parse_discover_hosts_arp_localnet,
    "netdiscover": parse_discover_hosts_netdiscover,
    "fping": parse_discover_hosts_fping,

    "nmap_portscan": parse_nmap_portscan,
    "nmap_service_detection": parse_nmap_service_detection,
    "nmap_scripts": parse_enum_nmap_default_scripts,
}