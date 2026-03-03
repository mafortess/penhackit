import re

def parse_command_result(action_name: str, result: dict) -> list[dict]:
    """
    action_name: p.ej. "INSPECT_IPCONFIG", "INSPECT_ARP"
    result: {"cmd": str|None, "rc": int, "stdout": str, "stderr": str}

    Return a list of events for updating the KB. Each event is a dict with a "type" field and other relevant data.
    """
    print("Building event from command result...")

    rc = int(result.get("rc", 0))
    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""

    if rc != 0:
        return [{"type": "COMMAND_ERROR", "action": action_name, "rc": rc, "stderr": (result.get("stderr", "") or "")[:500]}]

    if action_name == "INSPECT_IPCONFIG":
        # Muy simple y robusto: extrae IPv4 y Default Gateway en plano
        ipv4s = re.findall(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
        gws = re.findall(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
        ipv4s = list(dict.fromkeys(ipv4s))
        gws = list(dict.fromkeys([g for g in gws if g]))

        # Extrae interfaces (bloques) de ipconfig /all (Windows)
        # Nota: esto NO pretende ser perfecto; es suficiente para MVP.
        interfaces = []
        blocks = re.split(r"\r?\n\r?\n", stdout)
        for b in blocks:
            # Heurística: bloque que tiene IPv4 Address y algún nombre de "adapter"
            if "IPv4 Address" not in b:
                continue

            # Nombre de interfaz (línea tipo: "Ethernet adapter Ethernet:")
            name = None
            m = re.search(r"^(.*adapter.*):\s*$", b, flags=re.IGNORECASE | re.MULTILINE)
            if m:
                name = m.group(1).strip()

            ipv4 = None
            m = re.search(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
            if m:
                ipv4 = m.group(1)

            gw = None
            m = re.search(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
            if m:
                gw = m.group(1)

            mac = None
            m = re.search(r"\bPhysical Address[^\n]*:\s*([0-9A-Fa-f\-]{11,})", b)
            if m:
                mac = m.group(1)

            interfaces.append({
                "name": name or "",
                "ipv4": ipv4 or "",
                "default_gw": gw or "",
                "mac": mac or "",
            })

        return [{
            "type": "NET_INFO",
            "ipv4": ipv4s,
            "default_gw": gws,
            "interfaces": interfaces,
        }]

    if action_name == "INSPECT_ARP":
        # arp -a (Windows): líneas típicas
        #   192.168.1.1           00-11-22-33-44-55     dynamic
        arp_neighbors = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(
                r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+(?P<mac>[0-9A-Fa-f\-]{11,}|[0-9A-Fa-f:]{11,})\s+(?P<kind>\w+)",
                line,
            )
            if m:
                arp_neighbors.append({
                    "ip": m.group("ip"),
                    "mac": m.group("mac"),
                    "type": m.group("kind"),
                })

        # Fallback: si no matchea MAC, al menos extrae IPs
        if not arp_neighbors:
            ips = re.findall(r"\b((?:\d{1,3}\.){3}\d{1,3})\b", stdout)
            ips = list(dict.fromkeys(ips))
            arp_neighbors = [{"ip": ip, "mac": "", "type": ""} for ip in ips]

        return [{
            "type": "ARP_TABLE",
            "arp_neighbors": arp_neighbors,
        }]

    return [{"type": "NO_EVENT", "action": action_name}]



