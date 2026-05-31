"""
Legacy parsers not used by the current MVP.

This file temporarily stores parsers for HTTP, SMB, Windows, Gobuster,
Nikto, DNS, NFS and other families that are not wired into ACTION_PARSERS yet.
"""

# ============================================================  
# windows

def parse_windows_ipconfig(stdout: str) -> list[dict]:
    ipv4s = re.findall(
        r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
        stdout,
    )
    gws = re.findall(
        r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
        stdout,
    )

    ipv4s = list(dict.fromkeys(ipv4s))
    gws = list(dict.fromkeys([g for g in gws if g]))

    # Extrae interfaces (bloques) de ipconfig /all (Windows)
    interfaces = []
    blocks = re.split(r"\r?\n\r?\n", stdout)

    for block in blocks:
        # Heurística: bloque que tiene IPv4 Address y algún nombre de "adapter"
        if "IPv4 Address" not in block:
            continue

        name = ""
        m = re.search(
            r"^(.*adapter.*):\s*$",
            block,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if m:
            name = m.group(1).strip()

        ipv4 = ""
        m = re.search(
            r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
            block,
        )
        if m:
            ipv4 = m.group(1)

        gw = ""
        m = re.search(
            r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})",
            block,
        )
        if m:
            gw = m.group(1)

        mac = ""
        m = re.search(
            r"\bPhysical Address[^\n]*:\s*([0-9A-Fa-f\-]{11,})",
            block,
        )
        if m:
            mac = m.group(1)

        interfaces.append({
            "name": name,
            "ipv4": ipv4,
            "default_gw": gw,
            "mac": mac,
        })

    return [{
        "type": "NET_INFO",
        "ipv4": ipv4s,
        "default_gw": gws,
        "interfaces": interfaces,
    }]

def parse_windows_arp(stdout: str) -> list[dict]:
    # arp -a (Windows): líneas típicas
    #   192.168.1.1           00-11-22-33-44-55     dynamic
    arp_neighbors = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(
            r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+"
            r"(?P<mac>[0-9A-Fa-f\-]{11,}|[0-9A-Fa-f:]{11,})\s+"
            r"(?P<kind>\w+)",
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


# oTHERS


def parse_curl_http_headers(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if ":" not in line:
            continue

        header, value = line.split(":", 1)

        events.append({
            "type": "HTTP_HEADER_DETECTED",
            "host": target_ip,
            "port": target_port,
            "header": header.strip(),
            "value": value.strip(),
        })

    return events


def parse_gobuster_dirs(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<path>\/\S+)\s+\(Status:\s*(?P<status>\d+)\)",
            line,
        )

        if not m:
            continue

        events.append({
            "type": "WEB_PATH_FOUND",
            "host": target_ip,
            "port": target_port,
            "path": m.group("path"),
            "status": int(m.group("status")),
        })

    return events


def parse_smbclient_shares(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    in_table = False

    for line in stdout.splitlines():
        raw = line
        line = line.strip()

        if not line:
            continue

        if line.startswith("Sharename"):
            in_table = True
            continue

        if not in_table:
            continue

        if line.startswith("---------"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue

        share = parts[0]
        share_type = parts[1]

        if share in {"IPC$", "print$"}:
            continue

        events.append({
            "type": "SMB_SHARE_FOUND",
            "host": target_ip,
            "share": share,
            "share_type": share_type,
        })

    return events


# HTTP

def parse_enum_http_headers(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_curl_http_headers(stdout, target_ip, target_port)


def parse_enum_https_headers(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_curl_http_headers(stdout, target_ip, target_port)


def parse_enum_http_index(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    m = re.search(r"<title[^>]*>(.*?)</title>", stdout, flags=re.IGNORECASE | re.DOTALL)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
        events.append({
            "type": "HTTP_TITLE_DETECTED",
            "host": target_ip,
            "port": target_port,
            "title": title,
        })

    hints = []
    for pattern in ["login", "admin", "password", "upload", "phpmyadmin", "wordpress", "drupal", "joomla"]:
        if re.search(pattern, stdout, flags=re.IGNORECASE):
            hints.append(pattern)

    if hints:
        events.append({
            "type": "HTTP_BODY_HINT_DETECTED",
            "host": target_ip,
            "port": target_port,
            "hints": sorted(set(hints)),
        })

    return events


def parse_enum_http_robots(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    if stdout.strip():
        events.append({
            "type": "ROBOTS_TXT_FOUND",
            "host": target_ip,
            "port": target_port,
            "content": stdout.strip()[:2000],
        })

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(r"^(Disallow|Allow):\s*(?P<path>/\S*)", line, flags=re.IGNORECASE)
        if not m:
            continue

        events.append({
            "type": "WEB_PATH_FOUND",
            "host": target_ip,
            "port": target_port,
            "path": m.group("path"),
            "source": "robots.txt",
        })

    return events


def parse_enum_http_dirs_gobuster(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_gobuster_dirs(stdout, target_ip, target_port)


def parse_enum_http_dirs_feroxbuster(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.search(r"(?P<status>\d{3}).*(?P<url>https?://\S+)", line)
        if not m:
            continue

        url = m.group("url")
        path = "/"

        path_match = re.search(r"https?://[^/]+(?P<path>/\S*)", url)
        if path_match:
            path = path_match.group("path")

        events.append({
            "type": "WEB_PATH_FOUND",
            "host": target_ip,
            "port": target_port,
            "path": path,
            "status": int(m.group("status")),
            "url": url,
            "source": "feroxbuster",
        })

    return events


def parse_enum_http_nikto(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line.startswith("+"):
            continue

        finding = line.lstrip("+").strip()
        if not finding:
            continue

        event_type = "WEB_FINDING_FOUND"

        if any(word in finding.lower() for word in ["vulnerable", "cve-", "xss", "sql", "outdated"]):
            event_type = "CANDIDATE_VULN_FOUND"

        events.append({
            "type": event_type,
            "host": target_ip,
            "port": target_port,
            "finding": finding[:1000],
            "source": "nikto",
        })

    return events


def parse_enum_http_technologies(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    text = stdout.strip()

    if not text:
        return []

    technologies = []

    for token in re.findall(r"\[([^\]]+)\]", text):
        technologies.append(token.strip())

    return [{
        "type": "WEB_TECH_DETECTED",
        "host": target_ip,
        "port": target_port,
        "technologies": technologies,
        "raw": text[:1000],
    }]


def parse_enum_http_waf(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    text = stdout.strip()
    lower = text.lower()

    detected = "is behind" in lower or ("waf" in lower and "detected" in lower)

    return [{
        "type": "WAF_DETECTED",
        "host": target_ip,
        "port": target_port,
        "detected": detected,
        "raw": text[:1000],
    }]



def parse_curl_http_body(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_index(stdout, target_ip, target_port)


def parse_curl_robots(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_robots(stdout, target_ip, target_port)


def parse_feroxbuster(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_dirs_feroxbuster(stdout, target_ip, target_port)


def parse_nikto(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_nikto(stdout, target_ip, target_port)


def parse_whatweb(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_technologies(stdout, target_ip, target_port)


def parse_wafw00f(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_waf(stdout, target_ip, target_port)


# ============================================================
# SMB ENUMERATION 
# ============================================================

def parse_enum_smb_shares(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_smbclient_shares(stdout, target_ip)


def parse_enum_smb_basic_enum4linux(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.search(r"\[([^\]]+)\]\s+\(S-\d-[^)]+\)", line)
        if m:
            events.append({
                "type": "SMB_USER_FOUND",
                "host": target_ip,
                "user": m.group(1),
                "source": "enum4linux",
            })

        m = re.match(r"^(?P<share>\S+)\s+(?P<share_type>Disk|IPC|Printer)\s*(?P<comment>.*)$", line)
        if m:
            events.append({
                "type": "SMB_SHARE_FOUND",
                "host": target_ip,
                "share": m.group("share"),
                "share_type": m.group("share_type"),
                "comment": m.group("comment").strip(),
                "source": "enum4linux",
            })

    return events


def parse_enum_smb_null_session_users(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for m in re.finditer(r"user:\[(?P<user>[^\]]+)\]\s+rid:\[(?P<rid>[^\]]+)\]", stdout):
        events.append({
            "type": "SMB_USER_FOUND",
            "host": target_ip,
            "user": m.group("user"),
            "rid": m.group("rid"),
            "source": "rpcclient",
        })

    return events


def parse_enum_smb_os_discovery(stdout: str, target_ip: str | None) -> list[dict]:
    info = []

    for line in stdout.splitlines():
        if "|" in line and any(key in line.lower() for key in ["os:", "computer name:", "domain name:", "workgroup:"]):
            info.append(line.strip("|_ "))

    if not info:
        return []

    return [{
        "type": "SMB_INFO_DETECTED",
        "host": target_ip,
        "info": info,
    }]


def parse_enum_smb_protocols(stdout: str, target_ip: str | None) -> list[dict]:
    protocols = []

    for line in stdout.splitlines():
        m = re.search(r"SMBv\d[^\s:]*", line)
        if m:
            protocols.append(m.group(0))

    return [{
        "type": "SMB_PROTOCOL_DETECTED",
        "host": target_ip,
        "protocols": sorted(set(protocols)),
        "raw": stdout[:1500],
    }]

def parse_enum4linux(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_smb_basic_enum4linux(stdout, target_ip)


def parse_rpcclient(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_smb_null_session_users(stdout, target_ip)


def parse_nmap_smb(stdout: str, target_ip: str | None) -> list[dict]:
    events = []
    events.extend(parse_enum_smb_os_discovery(stdout, target_ip))
    events.extend(parse_enum_smb_protocols(stdout, target_ip))
    return events
