# local_context_parser.py
import re

# ============================================================
# LOCAL ATTACKER CONTEXT
# ============================================================

def parse_inspect_local_hostname(stdout: str) -> list[dict]:
    hostname = stdout.strip()

    if not hostname:
        return []

    return [{
        "type": "LOCAL_HOSTNAME_DETECTED",
        "hostname": hostname,
    }]


def parse_inspect_ip_a(stdout: str) -> list[dict]:
    interfaces = []
    current = None

    for line in stdout.splitlines():
        line = line.rstrip()

        m = re.match(r"^\d+:\s+([^:]+):\s+<([^>]*)>", line)
        if m:
            current = {
                "name": m.group(1).strip(),
                "flags": m.group(2).split(","),
                "ipv4": [],
                "ipv6": [],
                "mac": "",
            }
            interfaces.append(current)
            continue

        if current is None:
            continue

        m = re.search(r"\blink/ether\s+([0-9a-fA-F:]{17})", line)
        if m:
            current["mac"] = m.group(1)

        m = re.search(r"\binet\s+((?:\d{1,3}\.){3}\d{1,3})/(\d+)", line)
        if m:
            current["ipv4"].append({
                "ip": m.group(1),
                "prefix": int(m.group(2)),
            })

        m = re.search(r"\binet6\s+([0-9a-fA-F:]+)/(\d+)", line)
        if m:
            current["ipv6"].append({
                "ip": m.group(1),
                "prefix": int(m.group(2)),
            })

    return [{
        "type": "NET_INFO",
        "interfaces": interfaces,
    }]


def parse_inspect_ip_r(stdout: str) -> list[dict]:
    routes = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        route = {
            "raw": line,
            "default": line.startswith("default "),
            "gateway": "",
            "dev": "",
            "src": "",
        }

        m = re.search(r"\bvia\s+((?:\d{1,3}\.){3}\d{1,3})", line)
        if m:
            route["gateway"] = m.group(1)

        m = re.search(r"\bdev\s+(\S+)", line)
        if m:
            route["dev"] = m.group(1)

        m = re.search(r"\bsrc\s+((?:\d{1,3}\.){3}\d{1,3})", line)
        if m:
            route["src"] = m.group(1)

        routes.append(route)

    return [{
        "type": "ROUTE_TABLE",
        "routes": routes,
    }]


def parse_inspect_ip_neigh(stdout: str) -> list[dict]:
    neighbors = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+dev\s+(?P<dev>\S+)"
            r"(?:\s+lladdr\s+(?P<mac>[0-9a-fA-F:]{17}))?\s+(?P<state>\S+)",
            line,
        )

        if not m:
            continue

        neighbors.append({
            "ip": m.group("ip"),
            "dev": m.group("dev"),
            "mac": m.group("mac") or "",
            "state": m.group("state"),
        })

    return [{
        "type": "ARP_TABLE",
        "arp_neighbors": neighbors,
    }]


def parse_inspect_ss_listeners(stdout: str) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line or line.lower().startswith("netid"):
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        proto = parts[0]
        local_addr = parts[4]
        process = " ".join(parts[6:]) if len(parts) > 6 else ""

        port = None
        m = re.search(r":(\d+)$", local_addr)
        if m:
            port = int(m.group(1))

        events.append({
            "type": "PORT_LISTENER_DETECTED",
            "proto": proto,
            "local_addr": local_addr,
            "port": port,
            "process": process,
        })

    return events


def parse_ping_focus_host(stdout: str, target_ip: str | None) -> list[dict]:
    text = stdout.lower()

    alive = (
        "bytes from" in text
        or "ttl=" in text
        or "1 received" in text
        or "0% packet loss" in text
    )

    return [{
        "type": "PING_RESPONSE",
        "host": target_ip,
        "alive": alive,
        "raw": stdout.strip()[:500],
    }]


def parse_trace_route_to_host(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(r"^(?P<hop>\d+)\s+(?P<rest>.+)$", line)
        if not m:
            continue

        ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line)

        events.append({
            "type": "ROUTE_HOP_DETECTED",
            "target": target_ip,
            "hop": int(m.group("hop")),
            "ips": ips,
            "raw": line,
        })

    return events

LOCAL_CONTEXT_PARSERS = {
    # ============================================================
    # By action name lower()
    # ============================================================
    "inspect_local_hostname": parse_inspect_local_hostname,
    "inspect_ip_a": parse_inspect_ip_a,
    "inspect_ip_r": parse_inspect_ip_r,
    "inspect_ip_neigh": parse_inspect_ip_neigh,
    "inspect_ss_listeners": parse_inspect_ss_listeners,
    "ping_focus_host": parse_ping_focus_host,
    "trace_route_to_host": parse_trace_route_to_host,

    # ============================================================
    # By parser_family
    # ============================================================
    "generic_text": parse_inspect_local_hostname,
    "linux_ip_addr": parse_inspect_ip_a,
    "linux_ip_route": parse_inspect_ip_r,
    "linux_ip_neigh": parse_inspect_ip_neigh,
    "linux_ss": parse_inspect_ss_listeners,
    "generic_ping": parse_ping_focus_host,
    "traceroute": parse_trace_route_to_host,
}