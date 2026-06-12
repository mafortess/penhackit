# - recibe eventos
# - decide qué parte de la KB actualizar
# - llama a los helpers de kb_schema.py

import subprocess  
import json
import os
from pathlib import Path


def update_kb(kb: dict, events: list[dict]) -> dict:
    print("Updating KB with new events...")

    for ev in events:
        et = ev.get("type")


        if et == "HOST_DISCOVERED":
            ip = ev.get("host")
            if not ip:
                continue

            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            host = ensure_host(kb, cidr, ip)

            host["alive"] = True
            # host["source"] = ev.get("source", "nmap")

            kb["history"].append(ev)
        
        elif et == "PORT_OPEN":
            ip = ev.get("host")
            port = ev.get("port")

            if not ip or port is None:
                kb["history"].append(ev)
                continue
            
            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            port_obj = ensure_port(kb, cidr, ip, int(port))

            port_obj["proto"] = ev.get("proto", "tcp")
            port_obj["state"] = "open"

            service_name = ev.get("service")
            if service_name:
                port_obj["service"]["name"] = service_name
                
            kb["history"].append(ev)

        elif et == "SERVICE_DETECTED":
            ip = ev.get("host")
            port = ev.get("port")

            if not ip or port is None:
                kb["history"].append(ev)
                continue

            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            port_obj = ensure_port(kb, cidr, ip, int(port))

            service_name = ev.get("service")
            if service_name:
                port_obj["service"]["name"] = service_name

            kb["history"].append(ev)

        elif et == "SERVICE_VERSION_DETECTED":
            ip = ev.get("host")
            port = ev.get("port")

            if not ip or port is None:
                kb["history"].append(ev)
                continue
            
            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            port_obj = ensure_port(kb, cidr, ip, int(port))

            service = port_obj["service"]
            service["name"] = ev.get("service") or service.get("name")
            service["product"] = ev.get("product") or service.get("product")
            service["version"] = ev.get("version") or service.get("version")
            service["banner"] = ev.get("banner") or service.get("banner")

            kb["history"].append(ev)

        elif et == "CANDIDATE_VULN_FOUND":
            ip = ev.get("host")
            port = ev.get("port")

            finding = {
                "title": ev.get("title") or ev.get("vuln") or ev.get("exploit"),
                "severity": ev.get("severity", "unknown"),
                "host": ip,
                "port": int(port) if port is not None else None,
                "service": ev.get("service"),
                "version": ev.get("version"),
                "source": ev.get("source"),
            }

            if ip and port is not None:
                cidr = ev.get("network") or infer_network_for_host(kb, ip)
                port_obj = ensure_port(kb, cidr, ip, int(port))

                if finding not in port_obj["candidate_vulns"]:
                    port_obj["candidate_vulns"].append(finding)

            if finding not in kb["findings"]:
                kb["findings"].append(finding)

            kb["history"].append(ev)

        elif et == "VALID_CREDENTIAL_FOUND":
            ip = ev.get("host")
            port = ev.get("port")

            if not ip or port is None:
                kb["history"].append(ev)
                continue

            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            port_obj = ensure_port(kb, cidr, ip, int(port))
                
            credential = {
                "host": ip,
                "port": int(port) if port is not None else None,
                "service": ev.get("service"),
                "username": ev.get("username"),
                "password": ev.get("password"),
                "source": ev.get("source"),
                "valid": True,
            }

            if credential not in kb["access"]["credentials"]:
                kb["access"]["credentials"].append(credential)

            kb["history"].append(ev)

        elif et == "EXPLOIT_ATTEMPTED":
            ip = ev.get("host")
            port = ev.get("port")
            exploit = ev.get("exploit")

            attempt = {
                "type": "exploit",
                "exploit": exploit,
                "source": ev.get("source"),
                "success": None,
            }

            if ip and port is not None:
                cidr = ev.get("network") or infer_network_for_host(kb, ip)
                port_obj = ensure_port(kb, cidr, ip, int(port))
                port_obj["attempts"].append(attempt)

            kb["stats"]["exploits_attempted"] += 1
            kb["history"].append(ev)

        elif et in {"EXPLOIT_FAILED", "SESSION_NOT_CREATED"}:
            ip = ev.get("host")
            port = ev.get("port")

            if ip and port is not None:
                cidr = ev.get("network") or infer_network_for_host(kb, ip)
                port_obj = ensure_port(kb, cidr, ip, int(port))

                port_obj["attempts"].append({
                    "type": "exploit",
                    "exploit": ev.get("exploit"),
                    "source": ev.get("source"),
                    "success": False,
                    "error": ev.get("error"),
                })

            kb["stats"]["exploits_failed"] += 1
            kb["history"].append(ev)


        elif et == "SESSION_OPENED":
            ip = ev.get("host")
            session_id = ev.get("session_id")
            exploit = ev.get("exploit")
            source = ev.get("source")

            if not ip:
                kb["history"].append(ev)
                continue

            cidr = ev.get("network") or infer_network_for_host(kb, ip)
            port_obj = ensure_port(kb, cidr, ip, int(port))

            session = {
                "id": session_id,
                "host": ip,
                "port": int(port),
                "service": port_obj.get("service", {}).get("name") or "ftp",
                "type": ev.get("session_type", "shell"),
                "tool": source,
                "exploit": exploit,
                "user": ev.get("user"),
                "privilege": ev.get("privilege"),
                "closed": False,
                "evidence": {},
            }

            if session not in port_obj["sessions"]:
                port_obj["sessions"].append(session)
            
            finding = {
                "title": f"Session opened via {exploit}",
                "severity": "critical",
                "host": ip,
                "port": int(port),
                "service": session["service"],
                "source": source,
            }

            if finding not in kb["findings"]:
                kb["findings"].append(finding)

            kb["history"].append(ev)
            

        elif et == "SESSION_CLOSED":
            session_id = ev.get("session_id")

            if not session_id:
                kb["history"].append(ev)
                continue

            for network in kb["target"]["networks"].values():
                for host in network["hosts"].values():
                    for port_obj in host["ports"].values():
                        for session in port_obj.get("sessions", []):
                            if session.get("id") == session_id:
                                session["closed"] = True

            kb["history"].append(ev)


        # elif et == "HTTP_HEADER_DETECTED":
        #     ip = ev.get("host")
        #     if not ip:
        #         kb["notes"].append(ev)
        #         continue
            
        #     cidr = ev.get("network") or infer_network_for_host(kb, ip)

        #     host = ensure_host(kb, cidr, ip)
        #     host.setdefault("http_headers", [])
        #     host["http_headers"].append({
        #         "port": ev.get("port"),
        #         "header": ev.get("header"),
        #         "value": ev.get("value"),
        #     })

        # elif et == "WEB_PATH_FOUND":
        #     ip = ev.get("host")
        #     if not ip:
        #         kb["notes"].append(ev)
        #         continue

        #     cidr = ev.get("network") or infer_network_for_host(kb, ip)

        #     host = ensure_host(kb, cidr, ip)
        #     entry = {
        #         "port": ev.get("port"),
        #         "path": ev.get("path"),
        #         "status": ev.get("status"),
        #     }

        #     if entry not in host["web_paths"]:
        #         host["web_paths"].append(entry)

        # elif et == "SMB_SHARE_FOUND":
        #     ip = ev.get("host")
        #     if not ip:
        #         kb["notes"].append(ev)
        #         continue

        #     cidr = ev.get("network") or infer_network_for_host(kb, ip)

        #     host = ensure_host(kb, cidr, ip)
        #     entry = {
        #         "share": ev.get("share"),
        #         "share_type": ev.get("share_type"),
        #     }

        #     if entry not in host["smb_shares"]:
        #         host["smb_shares"].append(entry)


        elif et == "NET_INFO":
            for ip in ev.get("ipv4", []):
                if ip and ip not in kb["net"]["ipv4"]:
                    kb["attacker"]["ipv4"].append(ip)

            for gw in ev.get("default_gw", []):
                if gw and gw not in kb["net"]["default_gw"]:
                    kb["attacker"]["default_gw"].append(gw)

            for iface in ev.get("interfaces", []):
                if iface not in kb["net"]["interfaces"]:
                    kb["attacker"]["interfaces"].append(iface)

        elif et == "ROUTE_TABLE":
            for route in ev.get("routes", []):
                if route not in kb["attacker"]["routes"]:
                    kb["attacker"]["routes"].append(route)

            kb["history"].append(ev)
        
        elif et == "ARP_TABLE":
            for n in ev.get("arp_neighbors", []):
                if n not in kb["net"]["arp_neighbors"]:
                    kb["attacker"]["arp_neighbors"].append(n)

            kb["history"].append(ev)

            # Importante: NO meter ARP en hosts del PoC de pentesting.
            # Si lo haces, contaminarás el foco con 10.0.2.2, gateways, NAT, etc.
        elif et in {"COMMAND_ERROR", "NO_MEANINGFUL_OUTPUT", "NO_COMMAND_EXECUTED", "NO_EVENT"}:
            kb["history"].append(ev)
        
        else:
            kb["history"].append(ev)

    refresh_stats(kb)
    return kb

# ============= Helpers for KB update logic =============

from penhackit.session import kb
from penhackit.session.kb.kb_schema import make_network, make_host, make_port

def refresh_stats(kb: dict) -> None:
    stats = kb["stats"]

    stats["networks"] = count_networks(kb)
    stats["hosts"] = count_hosts(kb)
    stats["ports"] = count_ports(kb)
    stats["services"] = count_services(kb)
    stats["vulns"] = count_vulns(kb)
    stats["credentials"] = count_credentials(kb)
    stats["sessions"] = count_sessions(kb)

def ensure_network(kb: dict, cidr: str) -> dict:
    networks = kb["target"]["networks"]

    if cidr not in networks:
        networks[cidr] = make_network(cidr)

    return networks[cidr]


def ensure_host(kb: dict, cidr: str, ip: str) -> dict:
    network = ensure_network(kb, cidr)
    hosts = network["hosts"]

    if ip not in hosts:
        hosts[ip] = make_host(ip)

    return hosts[ip]


def ensure_port(kb: dict, cidr: str, ip: str, port: int) -> dict:
    host = ensure_host(kb, cidr, ip)
    ports = host["ports"]
    port_key = str(port)

    if port_key not in ports:
        ports[port_key] = make_port(port)

    return ports[port_key]


def infer_network_for_host(kb: dict, ip: str) -> str:
    scope = kb.get("scope", {})
    target = scope.get("target")
    target_type = scope.get("target_type")

    if target_type == "network" and target:
        return target

    if target_type == "host":
        return f"{ip}/24"

    return "unknown"



def count_networks(kb: dict) -> int:
    return len(kb.get("target", {}).get("networks", {}))


def count_hosts(kb: dict) -> int:
    total = 0
    for network in kb.get("target", {}).get("networks", {}).values():
        total += len(network.get("hosts", {}))
    return total


def count_ports(kb: dict) -> int:
    total = 0
    for network in kb.get("target", {}).get("networks", {}).values():
        for host in network.get("hosts", {}).values():
            total += len(host.get("ports", {}))
    return total


def count_services(kb: dict) -> int:
    total = 0
    for network in kb.get("target", {}).get("networks", {}).values():
        for host in network.get("hosts", {}).values():
            for port_data in host.get("ports", {}).values():
                service = port_data.get("service", {})
                if isinstance(service, dict) and service.get("name"):
                    total += 1
                elif isinstance(service, str) and service:
                    total += 1
    return total


def count_vulns(kb: dict) -> int:
    total = 0
    for network in kb.get("target", {}).get("networks", {}).values():
        for host in network.get("hosts", {}).values():
            for port_data in host.get("ports", {}).values():
                total += len(port_data.get("candidate_vulns", []))
    return total

def count_credentials(kb: dict) -> int:
    total = 0

    for network in kb.get("target", {}).get("networks", {}).values():
        for host in network.get("hosts", {}).values():
            for port_data in host.get("ports", {}).values():
                total += len(port_data.get("credentials", []))

    return total


def count_sessions(kb: dict) -> int:
    total = 0

    for network in kb.get("target", {}).get("networks", {}).values():
        for host in network.get("hosts", {}).values():
            for port_data in host.get("ports", {}).values():
                total += len(port_data.get("sessions", []))

    return total

# ===============================A========================0


def save_kb(session_dir, kb: dict) -> None:
    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

from typing import Any, Dict, Set, Tuple


# Esta función compara dos KBs y calcula un resumen simple del progreso basado en nuevos hosts, puertos, servicios y hallazgos encontrados.

def compute_kb_progress_simple(prev_kb: dict[str, Any], new_kb: dict[str, Any]) -> dict[str, Any]:
    prev_hosts = _host_set(prev_kb)
    new_hosts = _host_set(new_kb)

    prev_ports = _ports_set(prev_kb)
    new_ports = _ports_set(new_kb)

    prev_services = _services_set(prev_kb)
    new_services = _services_set(new_kb)

    prev_findings = _findings_set(prev_kb)
    new_findings = _findings_set(new_kb)

    added_hosts = new_hosts - prev_hosts
    added_ports = new_ports - prev_ports
    added_services = new_services - prev_services
    added_findings = new_findings - prev_findings

    return {
        "has_progress": bool(added_hosts or added_ports or added_services or added_findings),
        "new_hosts_count": len(added_hosts),
        "new_ports_count": len(added_ports),
        "new_services_count": len(added_services),
        "new_findings_count": len(added_findings),
    }

# Helpers para compute_kb_progress_simple

def _host_set(kb: dict) -> set[str]:
    hosts = kb.get("hosts", {})

    if isinstance(hosts, dict):
        return set(hosts.keys())

    if isinstance(hosts, list):
        return set(
            h.get("ip")
            for h in hosts
            if isinstance(h, dict) and h.get("ip")
        )

    return set()

def _ports_set(kb: dict) -> set[tuple[str, int]]:
    result = set()

    hosts = kb.get("hosts", {})
    if not isinstance(hosts, dict):
        return result

    for ip, host in hosts.items():
        for port in host.get("ports", {}):
            result.add((ip, int(port)))

    return result


def _services_set(kb: dict) -> set[tuple[str, int, str]]:
    result = set()

    hosts = kb.get("hosts", {})
    if not isinstance(hosts, dict):
        return result

    for ip, host in hosts.items():
        for port, service_data in host.get("services", {}).items():
            service = service_data.get("service", "")
            if service:
                result.add((ip, int(port), service.lower()))

    return result

def _findings_set(kb: dict) -> set[str]:
    return set(
        json.dumps(finding, sort_keys=True)
        for finding in kb.get("findings", [])
    )


# def update_kb(kb: dict, events: list[dict]) -> dict:
#     print("Updating KB with new event...")

#     # Estructura mínima esperada
#     kb.setdefault("hosts", [])
#     kb.setdefault("services", [])
#     kb.setdefault("findings", [])
#     kb.setdefault("notes", [])
#     kb.setdefault("focus", {"level": "global", "host": "", "service": ""})
#     kb.setdefault("commands", [])
#     kb.setdefault("net", {
#         "interfaces": [],
#         "ipv4": [],
#         "default_gw": [],
#         "arp_neighbors": [],
#         "routes": [],
#     })

#     for ev in events:
#         et = ev.get("type")

#         if et == "NET_INFO":
#             # ipv4 / default_gw (listas planas)
#             for ip in ev.get("ipv4", []):
#                 if ip and ip not in kb["net"]["ipv4"]:
#                     kb["net"]["ipv4"].append(ip)

#             for gw in ev.get("default_gw", []):
#                 if gw and gw not in kb["net"]["default_gw"]:
#                     kb["net"]["default_gw"].append(gw)

#             # interfaces (lista de dicts)
#             existing_if = {(i.get("name"), i.get("ipv4")) for i in kb["net"]["interfaces"] if isinstance(i, dict)}
#             for iface in ev.get("interfaces", []):
#                 if not isinstance(iface, dict):
#                     continue
#                 key = (iface.get("name"), iface.get("ipv4"))
#                 if key not in existing_if:
#                     kb["net"]["interfaces"].append(iface)
#                     existing_if.add(key)

#         elif et == "ARP_TABLE":
#             # OJO: tu parser devuelve "arp_neighbors", no "neighbors"
#             existing_arp = {n.get("ip") for n in kb["net"]["arp_neighbors"] if isinstance(n, dict)}
#             for n in ev.get("arp_neighbors", []):
#                 if not isinstance(n, dict):
#                     continue
#                 ip = n.get("ip")
#                 if ip and ip not in existing_arp:
#                     kb["net"]["arp_neighbors"].append(n)
#                     existing_arp.add(ip)

#             # (Opcional) también refleja vecinos ARP como "hosts"
#             existing_hosts = {h.get("ip") for h in kb["hosts"] if isinstance(h, dict)}
#             for n in ev.get("arp_neighbors", []):
#                 if not isinstance(n, dict):
#                     continue
#                 ip = n.get("ip")
#                 if ip and ip not in existing_hosts:
#                     kb["hosts"].append({"ip": ip, "source": "arp"})
#                     existing_hosts.add(ip)

#         elif et == "COMMAND_ERROR":
#             kb["notes"].append(ev)

#         elif et == "NO_EVENT":
#             # Puedes ignorarlo o guardarlo; MVP: ignorar
#             pass

#         else:
#             kb["notes"].append(ev)

#     return kb



def launch_kb_monitor_window_windows(session_dir: Path, cols: int = 60, rows: int = 14) -> None:
    """
    Opens a separate small PowerShell window that continuously shows kb.json.
    Windows-only. No interaction with the core; read-only.
    """
    if os.name != "nt":
        return
    session_dir = session_dir.resolve()
    kb_path = (session_dir / "kb.json").resolve()
    ps1 = (session_dir / "_kb_monitor.ps1").resolve()

    # # PowerShell script to set window size and refresh output
    ps_script = rf"""
    $kb = '{str(kb_path)}'
    $ErrorActionPreference='Continue'
    try {{
      $raw = $Host.UI.RawUI
      $raw.WindowTitle = 'PenHackIt KB'
      $size = New-Object System.Management.Automation.Host.Size({cols},{rows})
      $raw.WindowSize = $size
      $raw.BufferSize = New-Object System.Management.Automation.Host.Size({cols}, 3000)
    }} catch {{Write-Host "ERROR: $($_.Exception.Message)"}}
    
    while ($true) {{
      Clear-Host
      if (Test-Path -LiteralPath $kb) {{
        Get-Content -LiteralPath $kb -Raw
      }} else {{
        Write-Host "Waiting for kb.json: $kb"
      }}
      Start-Sleep -Milliseconds 500
    }}
    """.strip()
        # IMPORTANT: start "" ...  (empty title), otherwise args get mis-parsed and it exits.
   
    # Launch new console window
    # ps_script = "while ($true) { Clear-Host; 'alive'; Start-Sleep -Milliseconds 500 }"
    
    ps1.write_text(
        f"$kb='{kb_path}'; while($true){{cls; if(Test-Path $kb){{gc $kb -Raw}} else {{'Waiting for kb.json'}}; sleep -m 500}}",
        encoding="utf-8",
    )

    subprocess.Popen(["cmd", "/c", "start", "", "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)])
    # subprocess.Popen(
    #     ["cmd.exe", "/c", "start", "", "powershell.exe", "-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
    #     stdout=subprocess.DEVNULL,
    #     stderr=subprocess.DEVNULL,
    #     stdin=subprocess.DEVNULL,
    #     creationflags=subprocess.CREATE_NEW_CONSOLE,
    # )


# def compute_kb_progress_simple(prev_kb: Dict[str, Any], new_kb: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     Very simple progress detector. Assumes a simple KB shape:

#       kb["hosts"] -> iterable of host strings (or dicts with "ip")
#       kb["open_ports"] -> iterable of {"host": "...", "port": 80}
#       kb["services"] -> iterable of {"host": "...", "port": 80, "name": "http"}
#       kb["findings"] -> iterable of strings

#     Returns counts + has_progress.
#     """

    
#     prev_hosts: Set[str] = set(_host(h) for h in prev_kb.get("hosts", []) if _host(h))
#     new_hosts: Set[str] = set(_host(h) for h in new_kb.get("hosts", []) if _host(h))

#     prev_ports: Set[Tuple[str, int]] = set(
#         (_host(p.get("host") or p.get("ip")), int(p.get("port")))
#         for p in prev_kb.get("open_ports", [])
#         if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
#     )
#     new_ports: Set[Tuple[str, int]] = set(
#         (_host(p.get("host") or p.get("ip")), int(p.get("port")))
#         for p in new_kb.get("open_ports", [])
#         if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
#     )

#     prev_services: Set[Tuple[str, int, str]] = set(
#         (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
#         for s in prev_kb.get("services", [])
#         if isinstance(s, dict)
#         and _host(s.get("host") or s.get("ip"))
#         and str(s.get("port", "")).isdigit()
#         and str(s.get("name", "")).strip()
#     )
#     new_services: Set[Tuple[str, int, str]] = set(
#         (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
#         for s in new_kb.get("services", [])
#         if isinstance(s, dict)
#         and _host(s.get("host") or s.get("ip"))
#         and str(s.get("port", "")).isdigit()
#         and str(s.get("name", "")).strip()
#     )

#     prev_findings: Set[str] = set(str(f).strip() for f in prev_kb.get("findings", []) if str(f).strip())
#     new_findings: Set[str] = set(str(f).strip() for f in new_kb.get("findings", []) if str(f).strip())

#     added_hosts = new_hosts - prev_hosts
#     added_ports = new_ports - prev_ports
#     added_services = new_services - prev_services
#     added_findings = new_findings - prev_findings

#     return {
#         "has_progress": bool(added_hosts or added_ports or added_services or added_findings),
#         "new_hosts_count": len(added_hosts),
#         "new_ports_count": len(added_ports),
#         "new_services_count": len(added_services),
#         "new_findings_count": len(added_findings),
#     }

# def _host(x: Any) -> str:
#         if isinstance(x, str):
#             return x.strip()
#         if isinstance(x, dict):
#             return str(x.get("ip") or x.get("host") or "").strip()
#         return str(x).strip()
