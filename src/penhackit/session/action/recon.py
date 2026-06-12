 # ============================================================
# 200-299 RECON / DISCOVERY / PORT SCANNING
# ============================================================

RECON_ACTIONS = { 
    # SCAN HOSTS
    # Desactiva el escaneo de puertos. Realiza únicamente un descubrimiento de hosts
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
    # ==============
    # SCAN PORTS
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
        "command_template": "timeout 60 nmap -sV --version-light --max-retries 2 --host-timeout 75s -T3 -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_service_detection",
        "expected_events": [
            "SERVICE_DETECTED",
            "SERVICE_VERSION_DETECTED",
        ],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "low",
        "description": "Detect service names and versions on known open ports.",
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
        "command_template": "timeout 150 nmap -A --max-retries 2 --host-timeout 160s -T3 -p {known_open_ports_csv} {target_ip}",
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
        "command_template": "timeout 90 nmap -sC --max-retries 2 --host-timeout 100s -p {known_open_ports_csv} {target_ip}",
        "placeholders": ["known_open_ports_csv", "target_ip"],
        "parser_family": "nmap_scripts",
        "expected_events": ["SCRIPT_RESULT"],
        "preconditions": {
            "requires_target_ip": True,
            "requires_known_open_ports": True,
        },
        "risk_level": "low",
        "description": "Run Nmap default NSE scripts on known open ports.",
    }
}