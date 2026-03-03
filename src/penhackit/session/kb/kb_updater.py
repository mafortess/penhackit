import subprocess  
import json
import os
from pathlib import Path

def build_initial_kb(session_id: str) -> dict:
    """
    Construye una KB inicial vacía o con datos predeterminados para el inicio de la sesión.
    """
    return {
        "session_id": session_id,
        "name_enterprise": "ITIS",
        "networks": {},
        "hosts": [],
        "services": [],
        "findings": [],
        "notes": [],
        "net": {
            "interfaces": [],
            "ipv4": [],
            "default_gw": [],
            "arp_neighbors": [],
            "routes": [],
        },
        "focus": {"level": "global", "host": "", "service": ""},
        "commands": [],
        "step_idx": 0,
        "last_action_id": None,
        "last_action_name": None,
        "last_rc": None,
        "last_event_type": None,
    }


def update_kb(kb: dict, events: list[dict]) -> dict:
    print("Updating KB with new event...")

    # Estructura mínima esperada
    kb.setdefault("hosts", [])
    kb.setdefault("services", [])
    kb.setdefault("findings", [])
    kb.setdefault("notes", [])
    kb.setdefault("focus", {"level": "global", "host": "", "service": ""})
    kb.setdefault("commands", [])
    kb.setdefault("net", {
        "interfaces": [],
        "ipv4": [],
        "default_gw": [],
        "arp_neighbors": [],
        "routes": [],
    })

    for ev in events:
        et = ev.get("type")

        if et == "NET_INFO":
            # ipv4 / default_gw (listas planas)
            for ip in ev.get("ipv4", []):
                if ip and ip not in kb["net"]["ipv4"]:
                    kb["net"]["ipv4"].append(ip)

            for gw in ev.get("default_gw", []):
                if gw and gw not in kb["net"]["default_gw"]:
                    kb["net"]["default_gw"].append(gw)

            # interfaces (lista de dicts)
            existing_if = {(i.get("name"), i.get("ipv4")) for i in kb["net"]["interfaces"] if isinstance(i, dict)}
            for iface in ev.get("interfaces", []):
                if not isinstance(iface, dict):
                    continue
                key = (iface.get("name"), iface.get("ipv4"))
                if key not in existing_if:
                    kb["net"]["interfaces"].append(iface)
                    existing_if.add(key)

        elif et == "ARP_TABLE":
            # OJO: tu parser devuelve "arp_neighbors", no "neighbors"
            existing_arp = {n.get("ip") for n in kb["net"]["arp_neighbors"] if isinstance(n, dict)}
            for n in ev.get("arp_neighbors", []):
                if not isinstance(n, dict):
                    continue
                ip = n.get("ip")
                if ip and ip not in existing_arp:
                    kb["net"]["arp_neighbors"].append(n)
                    existing_arp.add(ip)

            # (Opcional) también refleja vecinos ARP como "hosts"
            existing_hosts = {h.get("ip") for h in kb["hosts"] if isinstance(h, dict)}
            for n in ev.get("arp_neighbors", []):
                if not isinstance(n, dict):
                    continue
                ip = n.get("ip")
                if ip and ip not in existing_hosts:
                    kb["hosts"].append({"ip": ip, "source": "arp"})
                    existing_hosts.add(ip)

        elif et == "COMMAND_ERROR":
            kb["notes"].append(ev)

        elif et == "NO_EVENT":
            # Puedes ignorarlo o guardarlo; MVP: ignorar
            pass

        else:
            kb["notes"].append(ev)

    return kb

def save_kb(session_dir, kb: dict) -> None:
    (session_dir / "kb.json").write_text(
        json.dumps(kb, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

from typing import Any, Dict, Set, Tuple


def compute_kb_progress_simple(prev_kb: Dict[str, Any], new_kb: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple progress detector. Assumes a simple KB shape:

      kb["hosts"] -> iterable of host strings (or dicts with "ip")
      kb["open_ports"] -> iterable of {"host": "...", "port": 80}
      kb["services"] -> iterable of {"host": "...", "port": 80, "name": "http"}
      kb["findings"] -> iterable of strings

    Returns counts + has_progress.
    """

    def _host(x: Any) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            return str(x.get("ip") or x.get("host") or "").strip()
        return str(x).strip()

    prev_hosts: Set[str] = set(_host(h) for h in prev_kb.get("hosts", []) if _host(h))
    new_hosts: Set[str] = set(_host(h) for h in new_kb.get("hosts", []) if _host(h))

    prev_ports: Set[Tuple[str, int]] = set(
        (_host(p.get("host") or p.get("ip")), int(p.get("port")))
        for p in prev_kb.get("open_ports", [])
        if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
    )
    new_ports: Set[Tuple[str, int]] = set(
        (_host(p.get("host") or p.get("ip")), int(p.get("port")))
        for p in new_kb.get("open_ports", [])
        if isinstance(p, dict) and _host(p.get("host") or p.get("ip")) and str(p.get("port", "")).isdigit()
    )

    prev_services: Set[Tuple[str, int, str]] = set(
        (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
        for s in prev_kb.get("services", [])
        if isinstance(s, dict)
        and _host(s.get("host") or s.get("ip"))
        and str(s.get("port", "")).isdigit()
        and str(s.get("name", "")).strip()
    )
    new_services: Set[Tuple[str, int, str]] = set(
        (_host(s.get("host") or s.get("ip")), int(s.get("port")), str(s.get("name", "")).strip().lower())
        for s in new_kb.get("services", [])
        if isinstance(s, dict)
        and _host(s.get("host") or s.get("ip"))
        and str(s.get("port", "")).isdigit()
        and str(s.get("name", "")).strip()
    )

    prev_findings: Set[str] = set(str(f).strip() for f in prev_kb.get("findings", []) if str(f).strip())
    new_findings: Set[str] = set(str(f).strip() for f in new_kb.get("findings", []) if str(f).strip())

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
