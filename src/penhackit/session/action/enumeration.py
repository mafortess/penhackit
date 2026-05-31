# ============================================================
# 300-399 SERVICE ENUMERATION
# ============================================================

ENUMERATION_ACTIONS = {
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
}