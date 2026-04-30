import re

def parse_command_result(action: dict, result: dict) -> list[dict]:
    """
    action: action metadata dict from ACTIONS.
    result: {
        "cmd": str | None,
        "rc": int,
        "stdout": str,
        "stderr": str,
        "target_ip": optional,
        "target_port": optional,
        "target": optional,
        ...
    }
    
    Return a list of events for updating the KB. Each event is a dict with a "type" field and other relevant data.
    """
    print(f"\nEVENTS:")
    print("Building event from command result...")

    if isinstance(action, dict):
        action_name = action.get("name", "UNKNOWN_ACTION")
    elif isinstance(action, str):
        action_name = action
    else:
        action_name = "UNKNOWN_ACTION"

    try:
        rc = int(result.get("rc", 0))
    except (ValueError, TypeError):
        rc = -1

    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""

    target_ip = result.get("target_ip")
    target_port = result.get("target_port")
    target = result.get("target")
    target_domain = result.get("target_domain")

    if rc is not None and int(rc) != 0:
        return [{"type": "COMMAND_ERROR", "action": action_name, "rc": rc, "stderr": (result.get("stderr", "") or "")[:500]}]

    # Windows/local inspection
    if action_name == "INSPECT_IPCONFIG":
        return parse_windows_ipconfig(stdout)

    if action_name == "INSPECT_ARP":
        return parse_windows_arp(stdout)
    
    # ============================================================
    # CONTROL
    # ============================================================

    if action_name == "STOP":
        return [{"type": "SESSION_STOPPED"}]

    if action_name == "NO_OP":
        return [{"type": "NO_ACTION"}]

    
    # ============================================================
    # LOCAL ATTACKER CONTEXT
    # ============================================================

    if action_name == "INSPECT_LOCAL_HOSTNAME":
        return parse_inspect_local_hostname(stdout)

    if action_name == "INSPECT_IP_A":
        return parse_inspect_ip_a(stdout)

    if action_name == "INSPECT_IP_R":
        return parse_inspect_ip_r(stdout)

    if action_name == "INSPECT_IP_NEIGH":
        return parse_inspect_ip_neigh(stdout)

    if action_name == "INSPECT_SS_LISTENERS":
        return parse_inspect_ss_listeners(stdout)

    if action_name == "PING_FOCUS_HOST":
        return parse_ping_focus_host(stdout, target_ip)

    if action_name == "TRACE_ROUTE_TO_HOST":
        return parse_trace_route_to_host(stdout, target_ip)

    # ============================================================
    # RECON / DISCOVERY
    # ============================================================

    if action_name == "DISCOVER_HOSTS_NMAP_PING_SWEEP":
        return parse_discover_hosts_nmap_ping_sweep(stdout)

    if action_name == "DISCOVER_HOSTS":
        return parse_discover_hosts_nmap_ping_sweep(stdout)

    if action_name == "DISCOVER_HOSTS_ARP_LOCALNET":
        return parse_discover_hosts_arp_localnet(stdout)

    if action_name == "DISCOVER_HOSTS_ARP_RANGE":
        return parse_discover_hosts_arp_range(stdout, target)

    if action_name == "DISCOVER_HOSTS_NETDISCOVER":
        return parse_discover_hosts_netdiscover(stdout)

    if action_name == "DISCOVER_HOSTS_FPING":
        return parse_discover_hosts_fping(stdout)

    # ============================================================
    # PORT SCANNING
    # ============================================================

    if action_name == "SCAN_TOP_TCP_PORTS":
        return parse_scan_top_tcp_ports(stdout, target_ip)

    if action_name == "SCAN_FULL_TCP_PORTS":
        return parse_scan_full_tcp_ports(stdout, target_ip)

    if action_name == "SCAN_QUICK_TCP_PORTS":
        return parse_scan_quick_tcp_ports(stdout, target_ip)

    if action_name == "SCAN_TOP_UDP_PORTS":
        return parse_scan_top_udp_ports(stdout, target_ip)

    # ============================================================
    # SERVICE DETECTION
    # ============================================================

    if action_name == "DETECT_SERVICES":
        return parse_detect_services(stdout, target_ip)

    if action_name == "DETECT_SERVICES_LIGHT":
        return parse_detect_services_light(stdout, target_ip)

    if action_name == "DETECT_SERVICES_AGGRESSIVE":
        return parse_detect_services_aggressive(stdout, target_ip)

    if action_name == "ENUM_NMAP_DEFAULT_SCRIPTS":
        return parse_enum_nmap_default_scripts(stdout, target_ip)

    # ============================================================
    # HTTP / HTTPS ENUMERATION
    # ============================================================

    if action_name == "ENUM_HTTP_HEADERS":
        return parse_enum_http_headers(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTPS_HEADERS":
        return parse_enum_https_headers(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_INDEX":
        return parse_enum_http_index(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_ROBOTS":
        return parse_enum_http_robots(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_DIRS_GOBUSTER":
        return parse_enum_http_dirs_gobuster(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_DIRS":
        return parse_enum_http_dirs_gobuster(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_DIRS_FEROXBUSTER":
        return parse_enum_http_dirs_feroxbuster(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_NIKTO":
        return parse_enum_http_nikto(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_TECHNOLOGIES":
        return parse_enum_http_technologies(stdout, target_ip, target_port)

    if action_name == "ENUM_HTTP_WAF":
        return parse_enum_http_waf(stdout, target_ip, target_port)

    # ============================================================
    # SMB ENUMERATION
    # ============================================================

    if action_name == "ENUM_SMB_SHARES":
        return parse_enum_smb_shares(stdout, target_ip)

    if action_name == "ENUM_SMB_BASIC_ENUM4LINUX":
        return parse_enum_smb_basic_enum4linux(stdout, target_ip)

    if action_name == "ENUM_SMB_BASIC":
        return parse_enum_smb_basic_enum4linux(stdout, target_ip)

    if action_name == "ENUM_SMB_NULL_SESSION_USERS":
        return parse_enum_smb_null_session_users(stdout, target_ip)

    if action_name == "ENUM_SMB_OS_DISCOVERY":
        return parse_enum_smb_os_discovery(stdout, target_ip)

    if action_name == "ENUM_SMB_PROTOCOLS":
        return parse_enum_smb_protocols(stdout, target_ip)

    # ============================================================
    # FTP ENUMERATION
    # ============================================================

    if action_name == "ENUM_FTP_BANNER":
        return parse_enum_ftp_banner(stdout, target_ip, target_port)

    if action_name == "ENUM_FTP_ANONYMOUS":
        return parse_enum_ftp_anonymous(stdout, target_ip, target_port)

    if action_name == "ENUM_FTP_NMAP_SCRIPTS":
        return parse_enum_ftp_nmap_scripts(stdout, target_ip)

    # ============================================================
    # SSH ENUMERATION
    # ============================================================

    if action_name == "ENUM_SSH_BANNER":
        return parse_enum_ssh_banner(stdout, target_ip, target_port)

    if action_name == "ENUM_SSH_NMAP_SCRIPTS":
        return parse_enum_ssh_nmap_scripts(stdout, target_ip)

    # ============================================================
    # DNS ENUMERATION
    # ============================================================

    if action_name == "ENUM_DNS_VERSION_BIND":
        return parse_enum_dns_version_bind(stdout, target_ip, target_domain)

    if action_name == "ENUM_DNS_ANY":
        return parse_enum_dns_any(stdout, target_ip, target_domain)

    if action_name == "ENUM_DNS_ZONE_TRANSFER":
        return parse_enum_dns_zone_transfer(stdout, target_ip, target_domain)

    # ============================================================
    # NFS / RPC ENUMERATION
    # ============================================================

    if action_name == "ENUM_NFS_EXPORTS":
        return parse_enum_nfs_exports(stdout, target_ip)

    if action_name == "ENUM_RPCINFO":
        return parse_enum_rpcinfo(stdout, target_ip)

    # ============================================================
    # DATABASE / RDP / VNC ENUMERATION
    # ============================================================

    if action_name == "ENUM_MYSQL_INFO":
        return parse_enum_mysql_info(stdout, target_ip)

    if action_name == "ENUM_POSTGRES_INFO":
        return parse_enum_postgres_info(stdout, target_ip)

    if action_name == "ENUM_RDP_INFO":
        return parse_enum_rdp_info(stdout, target_ip)

    if action_name == "ENUM_VNC_INFO":
        return parse_enum_vnc_info(stdout, target_ip)

    # ============================================================
    # VULNERABILITY DISCOVERY
    # ============================================================

    if action_name == "CHECK_SERVICE_VERSION_VULNS":
        return parse_check_service_version_vulns(stdout)

    if action_name == "CHECK_NMAP_VULN_SCRIPTS":
        return parse_check_nmap_vuln_scripts(stdout, target_ip, target_port)

    if action_name == "CHECK_SMB_VULNS":
        return parse_check_smb_vulns(stdout, target_ip)

    if action_name == "CHECK_HTTP_VULNS_NIKTO":
        return parse_check_http_vulns_nikto(stdout, target_ip, target_port)

    if action_name == "CHECK_SSL_TLS_CIPHERS":
        return parse_check_ssl_tls_ciphers(stdout, target_ip, target_port)

    if action_name == "CHECK_FTP_VULNS":
        return parse_check_ftp_vulns(stdout, target_ip, target_port)

    # ============================================================
    # CREDENTIAL ATTACKS / AUTH CHECKS
    # ============================================================

    if action_name == "BRUTEFORCE_SSH_LAB":
        return parse_bruteforce_ssh_lab(stdout, stderr, target_ip, target_port)

    if action_name == "BRUTEFORCE_FTP_LAB":
        return parse_bruteforce_ftp_lab(stdout, stderr, target_ip, target_port)

    if action_name == "BRUTEFORCE_HTTP_LOGIN_LAB":
        return parse_bruteforce_http_login_lab(stdout, stderr, target_ip, target_port)

    if action_name == "CHECK_FTP_ANONYMOUS_LOGIN":
        return parse_check_ftp_anonymous_login(stdout, target_ip, target_port)

    # ============================================================
    # EXPLOITATION CONTROLADA
    # ============================================================

    if action_name == "MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT":
        return parse_msf_exploit_samba_usermap_script(stdout, stderr, target_ip)

    if action_name == "MSF_EXPLOIT_VSFTPD_234_BACKDOOR":
        return parse_msf_exploit_vsftpd_234_backdoor(stdout, stderr, target_ip)

    if action_name == "MSF_EXPLOIT_DISTCC_EXEC":
        return parse_msf_exploit_distcc_exec(stdout, stderr, target_ip)

    if action_name == "MSF_EXPLOIT_TOMCAT_MGR_UPLOAD":
        return parse_msf_exploit_tomcat_mgr_upload(stdout, stderr, target_ip)

    # ============================================================
    # POST-EXPLOITATION
    # ============================================================

    if action_name == "POST_ENUM_WHOAMI":
        return parse_post_enum_whoami(stdout)

    if action_name == "POST_ENUM_UNAME":
        return parse_post_enum_uname(stdout)

    if action_name == "POST_ENUM_ID":
        return parse_post_enum_id(stdout)

    if action_name == "POST_ENUM_HOSTNAME":
        return parse_post_enum_hostname(stdout)

    if action_name == "POST_ENUM_IP_ADDR":
        return parse_post_enum_ip_addr(stdout)


    return [{
        "type": "NO_EVENT",
        "action": action_name,
    }]

# ============================================================
# CONTROL
# ============================================================

def parse_stop() -> list[dict]:
    return [{
        "type": "SESSION_STOPPED",
    }]


def parse_no_op() -> list[dict]:
    return [{
        "type": "NO_ACTION",
    }]

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


# ============================================================
# FTP/SSH ENUMERATION 
# ============================================================

def parse_enum_ftp_banner(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    banner = stdout.strip()

    if not banner:
        return []

    return [{
        "type": "SERVICE_BANNER_DETECTED",
        "service_hint": "ftp",
        "host": target_ip,
        "port": target_port,
        "banner": banner[:1000],
    }]


def parse_enum_ssh_banner(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    banner = stdout.strip()

    if not banner:
        return []

    return [{
        "type": "SERVICE_BANNER_DETECTED",
        "service_hint": "ssh",
        "host": target_ip,
        "port": target_port,
        "banner": banner[:1000],
    }]


def parse_enum_ftp_anonymous(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    text = stdout.lower()

    allowed = (
        "anonymous ftp login allowed" in text
        or ("ftp-anon:" in text and "anonymous" in text)
    )

    return [{
        "type": "FTP_ANON_LOGIN_ALLOWED",
        "host": target_ip,
        "port": target_port,
        "allowed": allowed,
        "raw": stdout[:1000],
    }]


def parse_enum_ftp_nmap_scripts(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)


def parse_enum_ssh_nmap_scripts(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)

def parse_generic_banner(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    banner = stdout.strip()

    if not banner:
        return []

    return [{
        "type": "SERVICE_BANNER_DETECTED",
        "host": target_ip,
        "port": target_port,
        "banner": banner[:1000],
    }]


def parse_nmap_ftp_anon(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_ftp_anonymous(stdout, target_ip, target_port)


def parse_enum_dns_version_bind(stdout: str, target_ip: str | None, target_domain: str | None) -> list[dict]:
    records = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line or line.startswith(";"):
            continue

        if "version.bind" in line.lower() or "\t" in line or " IN " in line:
            records.append(line)

    if not records:
        return []

    return [{
        "type": "DNS_INFO_DETECTED",
        "server": target_ip,
        "domain": target_domain,
        "records": records,
    }]


def parse_enum_dns_any(stdout: str, target_ip: str | None, target_domain: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line or line.startswith(";"):
            continue

        if "\t" in line or " IN " in line:
            events.append({
                "type": "DNS_RECORD_FOUND",
                "server": target_ip,
                "domain": target_domain,
                "record": line,
            })

    return events


def parse_enum_dns_zone_transfer(stdout: str, target_ip: str | None, target_domain: str | None) -> list[dict]:
    lower = stdout.lower()

    failed = (
        "transfer failed" in lower
        or "connection timed out" in lower
        or "failed" in lower
    )

    allowed = bool(stdout.strip()) and not failed

    events = [{
        "type": "DNS_ZONE_TRANSFER_ALLOWED",
        "server": target_ip,
        "domain": target_domain,
        "allowed": allowed,
    }]

    if allowed:
        events.extend(parse_enum_dns_any(stdout, target_ip, target_domain))

    return events


def parse_enum_nfs_exports(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line or line.lower().startswith("export list"):
            continue

        parts = line.split()
        if not parts:
            continue

        export_path = parts[0]

        if export_path.startswith("/"):
            events.append({
                "type": "NFS_EXPORT_FOUND",
                "host": target_ip,
                "export": export_path,
                "allowed": parts[1:] if len(parts) > 1 else [],
            })

    return events

# DNS / NFS / RPC

def parse_enum_rpcinfo(stdout: str, target_ip: str | None) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        m = re.match(
            r"^(?P<program>\d+)\s+(?P<version>\d+)\s+"
            r"(?P<proto>tcp|udp)\s+(?P<port>\d+)\s+(?P<service>\S+)",
            line,
        )

        if not m:
            continue

        events.append({
            "type": "RPC_SERVICE_FOUND",
            "host": target_ip,
            "program": m.group("program"),
            "version": m.group("version"),
            "proto": m.group("proto"),
            "port": int(m.group("port")),
            "service": m.group("service"),
        })

    return events


# DB RDP VNC

def parse_enum_mysql_info(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)


def parse_enum_postgres_info(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)


def parse_enum_rdp_info(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)


def parse_enum_vnc_info(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_enum_nmap_default_scripts(stdout, target_ip)

# ============================================================
# VULNERABILITY DISCOVERY
# ============================================================


def parse_check_service_version_vulns(stdout: str) -> list[dict]:
    return parse_searchsploit(stdout)


def parse_check_nmap_vuln_scripts(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_vuln_script_output(stdout, target_ip, target_port, source="nmap-vuln")


def parse_check_smb_vulns(stdout: str, target_ip: str | None) -> list[dict]:
    return parse_vuln_script_output(stdout, target_ip, 445, source="nmap-smb-vuln")


def parse_check_http_vulns_nikto(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_http_nikto(stdout, target_ip, target_port)


def parse_check_ssl_tls_ciphers(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    events = [{
        "type": "TLS_INFO_DETECTED",
        "host": target_ip,
        "port": target_port,
        "raw": stdout[:3000],
    }]

    lower = stdout.lower()
    weaknesses = []

    for marker in ["sslv2", "sslv3", "tlsv1.0", "rc4", "3des", "weak"]:
        if marker in lower:
            weaknesses.append(marker)

    if weaknesses:
        events.append({
            "type": "TLS_WEAKNESS_DETECTED",
            "host": target_ip,
            "port": target_port,
            "weaknesses": sorted(set(weaknesses)),
        })

    return events


def parse_check_ftp_vulns(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_vuln_script_output(stdout, target_ip, target_port, source="nmap-ftp-vuln")


def parse_vuln_script_output(
    stdout: str,
    target_ip: str | None,
    target_port: int | None,
    source: str,
) -> list[dict]:
    events = []

    current_script = None
    buffer = []

    for line in stdout.splitlines():
        script_match = re.match(r"^\|_?(?P<script>[a-zA-Z0-9_\-]+):\s*(?P<text>.*)", line)

        if script_match:
            if current_script and buffer:
                text = "\n".join(buffer).strip()
                events.append(build_vuln_script_event(target_ip, target_port, current_script, text, source))

            current_script = script_match.group("script")
            buffer = [script_match.group("text")]
            continue

        if current_script and line.startswith("|"):
            buffer.append(line.strip("|_ "))

    if current_script and buffer:
        text = "\n".join(buffer).strip()
        events.append(build_vuln_script_event(target_ip, target_port, current_script, text, source))

    return events


def build_vuln_script_event(
    target_ip: str | None,
    target_port: int | None,
    script: str,
    text: str,
    source: str,
) -> dict:
    lower = text.lower()

    event_type = "VULN_SCRIPT_RESULT"

    if (
        "vulnerable" in lower
        or "state: vulnerable" in lower
        or "cve-" in lower
        or "exploit" in lower
    ):
        event_type = "CANDIDATE_VULN_FOUND"

    cves = re.findall(r"CVE-\d{4}-\d{4,7}", text, flags=re.IGNORECASE)

    return {
        "type": event_type,
        "host": target_ip,
        "port": target_port,
        "script": script,
        "cves": sorted(set(cve.upper() for cve in cves)),
        "output": text[:2000],
        "source": source,
    }


def parse_nmap_vuln_scripts(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_check_nmap_vuln_scripts(stdout, target_ip, target_port)


def parse_nmap_ssl(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_check_ssl_tls_ciphers(stdout, target_ip, target_port)


# CREDENTILAS ATTACKS
def parse_bruteforce_ssh_lab(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port: int | None,
) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="ssh")


def parse_bruteforce_ftp_lab(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port: int | None,
) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="ftp")


def parse_bruteforce_http_login_lab(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port: int | None,
) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="http")


def parse_hydra_credentials(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port: int | None,
    service: str,
) -> list[dict]:
    events = []
    text = stdout + "\n" + stderr

    for line in text.splitlines():
        line = line.strip()

        m = re.search(
            r"host:\s*(?P<host>\S+)\s+login:\s*(?P<login>\S+)\s+password:\s*(?P<password>\S+)",
            line,
            flags=re.IGNORECASE,
        )

        if not m:
            continue

        events.append({
            "type": "VALID_CREDENTIAL_FOUND",
            "host": m.group("host") or target_ip,
            "port": target_port,
            "service": service,
            "username": m.group("login"),
            "password": m.group("password"),
            "source": "hydra",
        })

    if not events and ("0 valid password" in text.lower() or "0 valid" in text.lower()):
        events.append({
            "type": "LOGIN_FAILED",
            "host": target_ip,
            "port": target_port,
            "service": service,
            "source": "hydra",
        })

    return events


def parse_check_ftp_anonymous_login(stdout: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_enum_ftp_anonymous(stdout, target_ip, target_port)

# METASPLOIT
# 

def parse_msf_exploit_samba_usermap_script(
    stdout: str,
    stderr: str,
    target_ip: str | None,
) -> list[dict]:
    return parse_msfconsole_exploit(stdout, stderr, target_ip, exploit_name="samba_usermap_script")


def parse_msf_exploit_vsftpd_234_backdoor(
    stdout: str,
    stderr: str,
    target_ip: str | None,
) -> list[dict]:
    return parse_msfconsole_exploit(stdout, stderr, target_ip, exploit_name="vsftpd_234_backdoor")


def parse_msf_exploit_distcc_exec(
    stdout: str,
    stderr: str,
    target_ip: str | None,
) -> list[dict]:
    return parse_msfconsole_exploit(stdout, stderr, target_ip, exploit_name="distcc_exec")


def parse_msf_exploit_tomcat_mgr_upload(
    stdout: str,
    stderr: str,
    target_ip: str | None,
) -> list[dict]:
    return parse_msfconsole_exploit(stdout, stderr, target_ip, exploit_name="tomcat_mgr_upload")


def parse_msfconsole_exploit(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    exploit_name: str,
) -> list[dict]:
    events = [{
        "type": "EXPLOIT_ATTEMPTED",
        "host": target_ip,
        "exploit": exploit_name,
        "source": "msfconsole",
    }]

    text = stdout + "\n" + stderr
    lower = text.lower()

    patterns = [
        r"Command shell session (?P<session_id>\d+) opened",
        r"Meterpreter session (?P<session_id>\d+) opened",
        r"session (?P<session_id>\d+) opened",
    ]

    for pattern in patterns:
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if m:
            events.append({
                "type": "SESSION_OPENED",
                "host": target_ip,
                "session_id": m.group("session_id"),
                "exploit": exploit_name,
                "source": "msfconsole",
            })
            return events

    if "exploit failed" in lower or "failed" in lower:
        events.append({
            "type": "EXPLOIT_FAILED",
            "host": target_ip,
            "exploit": exploit_name,
            "source": "msfconsole",
            "raw_hint": text[-1000:],
        })

    return events  


def parse_msfconsole(stdout: str, stderr: str, target_ip: str | None) -> list[dict]:
    return parse_msfconsole_exploit(stdout, stderr, target_ip, exploit_name="unknown")

# =============================================================
# POST EXPLOIT
# =============================================================

def parse_post_enum_whoami(stdout: str) -> list[dict]:
    user = stdout.strip()

    if not user:
        return []

    return [{
        "type": "SESSION_USER_DETECTED",
        "user": user,
    }]


def parse_post_enum_uname(stdout: str) -> list[dict]:
    system = stdout.strip()

    if not system:
        return []

    return [{
        "type": "SESSION_SYSTEM_DETECTED",
        "system": system,
    }]


def parse_post_enum_id(stdout: str) -> list[dict]:
    text = stdout.strip()

    if not text:
        return []

    return [{
        "type": "SESSION_PRIVILEGES_DETECTED",
        "raw": text,
    }]


def parse_post_enum_hostname(stdout: str) -> list[dict]:
    hostname = stdout.strip()

    if not hostname:
        return []

    return [{
        "type": "SESSION_HOSTNAME_DETECTED",
        "hostname": hostname,
    }]


def parse_post_enum_ip_addr(stdout: str) -> list[dict]:
    events = parse_inspect_ip_a(stdout)

    for event in events:
        event["type"] = "SESSION_NET_INFO_DETECTED"

    return events


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


def parse_searchsploit(stdout: str) -> list[dict]:
    events = []

    for line in stdout.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.startswith("-") or "Exploit Title" in line or "Path" in line:
            continue

        if "|" not in line:
            continue

        title, path = line.split("|", 1)

        events.append({
            "type": "CANDIDATE_VULN_FOUND",
            "title": title.strip(),
            "path": path.strip(),
        })

    return events




# ============================================================================================

# # Sin factorizar
# def parse_command_result(action_name: str, result: dict) -> list[dict]:
#     """
#     action_name: p.ej. "INSPECT_IPCONFIG", "INSPECT_ARP"
#     result: {"cmd": str|None, "rc": int, "stdout": str, "stderr": str}

#     Return a list of events for updating the KB. Each event is a dict with a "type" field and other relevant data.
#     """
#     print("Building event from command result...")

#     rc = int(result.get("rc", 0))
#     stdout = result.get("stdout", "") or ""
#     stderr = result.get("stderr", "") or ""

#     if rc != 0:
#         return [{"type": "COMMAND_ERROR", "action": action_name, "rc": rc, "stderr": (result.get("stderr", "") or "")[:500]}]

#     if action_name == "INSPECT_IPCONFIG":
#         # Muy simple y robusto: extrae IPv4 y Default Gateway en plano
#         ipv4s = re.findall(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
#         gws = re.findall(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", stdout)
#         ipv4s = list(dict.fromkeys(ipv4s))
#         gws = list(dict.fromkeys([g for g in gws if g]))

#         # Extrae interfaces (bloques) de ipconfig /all (Windows)
#         # Nota: esto NO pretende ser perfecto; es suficiente para MVP.
#         interfaces = []
#         blocks = re.split(r"\r?\n\r?\n", stdout)
#         for b in blocks:
#             # Heurística: bloque que tiene IPv4 Address y algún nombre de "adapter"
#             if "IPv4 Address" not in b:
#                 continue

#             # Nombre de interfaz (línea tipo: "Ethernet adapter Ethernet:")
#             name = None
#             m = re.search(r"^(.*adapter.*):\s*$", b, flags=re.IGNORECASE | re.MULTILINE)
#             if m:
#                 name = m.group(1).strip()

#             ipv4 = None
#             m = re.search(r"\bIPv4 Address[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
#             if m:
#                 ipv4 = m.group(1)

#             gw = None
#             m = re.search(r"\bDefault Gateway[^\n]*:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", b)
#             if m:
#                 gw = m.group(1)

#             mac = None
#             m = re.search(r"\bPhysical Address[^\n]*:\s*([0-9A-Fa-f\-]{11,})", b)
#             if m:
#                 mac = m.group(1)

#             interfaces.append({
#                 "name": name or "",
#                 "ipv4": ipv4 or "",
#                 "default_gw": gw or "",
#                 "mac": mac or "",
#             })

#         return [{
#             "type": "NET_INFO",
#             "ipv4": ipv4s,
#             "default_gw": gws,
#             "interfaces": interfaces,
#         }]

#     if action_name == "INSPECT_ARP":
#         # arp -a (Windows): líneas típicas
#         #   192.168.1.1           00-11-22-33-44-55     dynamic
#         arp_neighbors = []
#         for line in stdout.splitlines():
#             line = line.strip()
#             if not line:
#                 continue
#             m = re.match(
#                 r"^(?P<ip>(?:\d{1,3}\.){3}\d{1,3})\s+(?P<mac>[0-9A-Fa-f\-]{11,}|[0-9A-Fa-f:]{11,})\s+(?P<kind>\w+)",
#                 line,
#             )
#             if m:
#                 arp_neighbors.append({
#                     "ip": m.group("ip"),
#                     "mac": m.group("mac"),
#                     "type": m.group("kind"),
#                 })

#         # Fallback: si no matchea MAC, al menos extrae IPs
#         if not arp_neighbors:
#             ips = re.findall(r"\b((?:\d{1,3}\.){3}\d{1,3})\b", stdout)
#             ips = list(dict.fromkeys(ips))
#             arp_neighbors = [{"ip": ip, "mac": "", "type": ""} for ip in ips]

#         return [{
#             "type": "ARP_TABLE",
#             "arp_neighbors": arp_neighbors,
#         }]

#     return [{"type": "NO_EVENT", "action": action_name}]



