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


CONTROL_PARSERS = {
    "STOP": parse_stop,
    "NO_OP": parse_no_op,
}

'''
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
        return parse_msf_exploit_samba_usermap_script(stdout, stderr, target_ip, target_port)

    if action_name == "MSF_EXPLOIT_VSFTPD_234_BACKDOOR":
        return parse_msf_exploit_vsftpd_234_backdoor(stdout, stderr, target_ip, target_port)

    if action_name == "MSF_EXPLOIT_DISTCC_EXEC":
        return parse_msf_exploit_distcc_exec(stdout, stderr, target_ip, target_port)

    if action_name == "MSF_EXPLOIT_TOMCAT_MGR_UPLOAD":
        return parse_msf_exploit_tomcat_mgr_upload(stdout, stderr, target_ip, target_port   )

    if action_name == "MSF_EXPLOIT_POSTGRES_PAYLOAD":
        return parse_msf_exploit_postgres_payload(stdout, stderr, target_ip, target_port)

    if action_name == "MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR":
        return parse_msf_exploit_unreal_ircd_backdoor(stdout, stderr, target_ip, target_port)

    if action_name == "CONNECT_INGRESLOCK_BIND_SHELL":
        return parse_shell_validation_output(
            stdout,
            stderr,
            target_ip,
            target_port,
            source="nc",
            session_type="shell",
            exploit_name="ingreslock_bind_shell",
            service="ingreslock",
        )
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


'''