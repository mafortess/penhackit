ACTION_CATALOG = {
    1: {
        "name": "DISCOVER_HOST",
        "category": "recon",
        "phase": "discovery",
        "command_profile_ids": ["nmap_host_discovery_basic"],
        "parser_family": "nmap_grep_host_discovery",
        "expected_event_ids": [100],
        "description": "Discover alive hosts in target network."
    },
    2: {
        "name": "SCAN_PORTS",
        "category": "recon",
        "phase": "portscan",
        "command_profile_ids": ["nmap_top_tcp_scan"],
        "parser_family": "nmap_grep_portscan",
        "expected_event_ids": [110],
        "description": "Scan top TCP ports on target host."
    }
}