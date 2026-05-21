## Semantic actions

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 0 | `STOP` | control | Ends the current session execution. |
| 1 | `NO_OP` | control | Performs no action in the current step. |

### Local attacker context

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 100 | `INSPECT_LOCAL_HOSTNAME` | attacker_context | Retrieves the hostname of the attacker machine. |
| 101 | `INSPECT_IP_A` | attacker_context | Inspects local Linux network interfaces. |
| 102 | `INSPECT_IP_R` | attacker_context | Inspects the local routing table. |
| 103 | `INSPECT_IP_NEIGH` | attacker_context | Inspects the local ARP/neighbour table. |
| 104 | `INSPECT_SS_LISTENERS` | attacker_context | Lists local TCP/UDP listening sockets. |
| 105 | `PING_FOCUS_HOST` | attacker_context | Sends one ICMP probe to the focused target host. |
| 106 | `TRACE_ROUTE_TO_HOST` | attacker_context | Traces the network path to the focused host. |

### Reconnaissance and discovery

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 200 | `DISCOVER_HOSTS` | host_discovery | Discovers live hosts in the target scope using Nmap ping scan. |
| 201 | `DISCOVER_HOSTS_ARP_LOCALNET` | host_discovery | Discovers local network hosts using ARP scan. |
| 202 | `DISCOVER_HOSTS_ARP_RANGE` | host_discovery | Discovers hosts in a target range using ARP scan. |
| 203 | `DISCOVER_HOSTS_NETDISCOVER` | host_discovery | Discovers hosts using Netdiscover. |
| 204 | `DISCOVER_HOSTS_FPING` | host_discovery | Discovers live hosts using fping sweep. |
| 210 | `SCAN_TOP_TCP_PORTS` | portscan | Scans the most common TCP ports on the focused host. |
| 211 | `SCAN_FULL_TCP_PORTS` | portscan | Scans all TCP ports on the focused host. |
| 212 | `SCAN_QUICK_TCP_PORTS` | portscan | Runs a fast TCP port scan. |
| 213 | `SCAN_TOP_UDP_PORTS` | portscan | Scans common UDP ports. |
| 220 | `DETECT_SERVICES` | service_detection | Detects service names and versions on known open ports. |
| 221 | `DETECT_SERVICES_LIGHT` | service_detection | Performs lightweight service version detection. |
| 222 | `DETECT_SERVICES_AGGRESSIVE` | service_detection | Performs aggressive service, script and OS detection. |
| 223 | `DETECT_OS_GUESS` | os_detection | Runs bounded OS detection against the focused host. |
| 230 | `ENUM_NMAP_DEFAULT_SCRIPTS` | general_enum | Runs bounded Nmap default scripts against known open ports. |

### HTTP / HTTPS enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 300 | `ENUM_HTTP_HEADERS` | http_enum | Retrieves HTTP response headers. |
| 301 | `ENUM_HTTP_INDEX` | http_enum | Fetches the HTTP index page and extracts basic hints. |
| 302 | `ENUM_HTTPS_HEADERS` | http_enum | Retrieves HTTPS response headers while ignoring certificate errors. |
| 303 | `ENUM_HTTP_ROBOTS` | http_enum | Fetches `robots.txt` and extracts discovered web paths. |
| 310 | `ENUM_HTTP_DIRS_GOBUSTER` | http_enum | Discovers common HTTP paths using Gobuster. |
| 311 | `ENUM_HTTP_DIRS_FEROXBUSTER` | http_enum | Discovers common HTTP paths using Feroxbuster. |
| 312 | `ENUM_HTTP_NIKTO` | http_enum | Runs basic Nikto checks against an HTTP service. |
| 313 | `ENUM_HTTP_TECHNOLOGIES` | http_enum | Fingerprints web technologies using WhatWeb. |
| 314 | `ENUM_HTTP_WAF` | http_enum | Attempts to detect a web application firewall. |

### SMB enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 320 | `ENUM_SMB_SHARES` | smb_enum | Enumerates SMB shares anonymously. |
| 321 | `ENUM_SMB_BASIC_ENUM4LINUX` | smb_enum | Runs broad SMB enumeration using enum4linux. |
| 322 | `ENUM_SMB_NULL_SESSION_USERS` | smb_enum | Attempts anonymous SMB user enumeration through rpcclient. |
| 323 | `ENUM_SMB_OS_DISCOVERY` | smb_enum | Retrieves SMB OS information using Nmap NSE. |
| 324 | `ENUM_SMB_PROTOCOLS` | smb_enum | Enumerates supported SMB protocol versions. |

### FTP enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 330 | `ENUM_FTP_BANNER` | ftp_enum | Grabs the FTP banner and closes the connection cleanly. |
| 331 | `ENUM_FTP_ANONYMOUS` | ftp_enum | Checks whether anonymous FTP login is allowed. |
| 332 | `ENUM_FTP_NMAP_SCRIPTS` | ftp_enum | Runs selected bounded FTP enumeration scripts. |

### SSH enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 340 | `ENUM_SSH_BANNER` | ssh_enum | Grabs the SSH service banner. |
| 341 | `ENUM_SSH_NMAP_SCRIPTS` | ssh_enum | Enumerates SSH algorithms and host keys. |

### DNS enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 350 | `ENUM_DNS_VERSION_BIND` | dns_enum | Queries DNS `version.bind` information. |
| 351 | `ENUM_DNS_ANY` | dns_enum | Queries DNS ANY records for a known domain. |
| 352 | `ENUM_DNS_ZONE_TRANSFER` | dns_enum | Attempts a DNS zone transfer in the authorized lab. |

### NFS / RPC enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 360 | `ENUM_NFS_EXPORTS` | nfs_enum | Enumerates NFS exported directories. |
| 361 | `ENUM_RPC_SERVICES` | rpc_enum | Enumerates exposed RPC services. |
| 362 | `ENUM_NFS_NMAP_SCRIPTS` | nfs_enum | Enumerates NFS exports and metadata using Nmap scripts. |

### Database / remote access enumeration

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 370 | `ENUM_MYSQL_INFO` | db_enum | Enumerates MySQL service information. |
| 371 | `ENUM_POSTGRES_INFO` | db_enum | Runs PostgreSQL NSE enumeration. |
| 380 | `ENUM_RDP_INFO` | rdp_enum | Enumerates RDP encryption and NTLM information. |
| 381 | `ENUM_VNC_INFO` | vnc_enum | Enumerates VNC service information. |

### Vulnerability discovery

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 400 | `CHECK_SERVICE_VERSION_VULNS` | vuln_lookup | Searches public exploit references for detected service versions. |
| 401 | `CHECK_NMAP_VULN_SCRIPTS` | vuln_lookup | Runs bounded Nmap vulnerability scripts against known open ports. |
| 410 | `CHECK_SMB_VULNS` | vuln_lookup | Runs SMB vulnerability NSE scripts. |
| 411 | `CHECK_HTTP_VULNS_NIKTO` | vuln_lookup | Checks common HTTP vulnerabilities using Nikto. |
| 412 | `CHECK_SSL_TLS_CIPHERS` | vuln_lookup | Enumerates TLS configuration and weak ciphers. |
| 413 | `CHECK_FTP_VULNS` | vuln_lookup | Checks FTP anonymous access and the vsftpd backdoor in the lab. |

### Credential validation and authentication checks

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 500 | `BRUTEFORCE_SSH` | credential_attack | Tests SSH credentials using Hydra in an authorized lab. |
| 501 | `BRUTEFORCE_FTP` | credential_attack | Tests FTP credentials using Hydra in an authorized lab. |
| 502 | `BRUTEFORCE_HTTP_LOGIN` | credential_attack | Tests HTTP form credentials when login parameters are known. |
| 510 | `CHECK_FTP_ANONYMOUS_LOGIN` | credential_attack | Checks whether anonymous FTP login is allowed. |
| 520 | `CHECK_SSH_KNOWN_CREDS` | credential_attack | Validates known SSH credentials. |
| 521 | `CHECK_TELNET_KNOWN_CREDS` | credential_attack | Validates known Telnet credentials. |
| 522 | `CHECK_MYSQL_KNOWN_CREDS` | credential_attack | Validates known MySQL credentials. |
| 523 | `CHECK_POSTGRES_KNOWN_CREDS` | credential_attack | Validates known PostgreSQL credentials. |
| 524 | `CHECK_TOMCAT_MANAGER_CREDS` | credential_attack | Validates Tomcat Manager credentials. |

### Exploitation

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 600 | `MSF_EXPLOIT_SAMBA_USERMAP_SCRIPT` | exploit | Attempts Samba username map script exploitation in the controlled lab. |
| 601 | `MSF_EXPLOIT_VSFTPD_234_BACKDOOR` | exploit | Exploits vsftpd 2.3.4, obtains a shell, executes validation commands and closes the session. |
| 602 | `MSF_EXPLOIT_DISTCC_EXEC` | exploit | Attempts distcc remote command execution. |
| 603 | `MSF_EXPLOIT_TOMCAT_MGR_UPLOAD` | exploit | Attempts Tomcat Manager WAR upload exploitation using valid credentials. |
| 604 | `MSF_EXPLOIT_POSTGRES_PAYLOAD` | exploit | Attempts PostgreSQL payload execution using valid credentials. |
| 605 | `MSF_EXPLOIT_UNREAL_IRCD_BACKDOOR` | exploit | Attempts UnrealIRCd backdoor exploitation. |
| 606 | `CONNECT_INGRESLOCK_BIND_SHELL` | exploit | Connects to an exposed bind shell service in the lab. |
| 607 | `MSF_EXPLOIT_RLOGIN_RSH_TRUST` | exploit | Attempts rlogin/rsh-style remote login in the controlled lab. |
| 608 | `MSF_EXPLOIT_JAVA_RMI_SERVER` | exploit | Attempts Java RMI server exploitation. |
| 609 | `MSF_EXPLOIT_DOCKER_DISTCC_EXEC` | exploit | Attempts distcc exploitation with explicit reverse payload parameters. |
| 610 | `EXPLOIT_VSFTPD_234_BACKDOOR_VALIDATE` | exploit | Triggers the vsftpd backdoor, connects to the spawned shell, executes validation commands and closes. |

### Basic post-exploitation

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 700 | `POST_ENUM_WHOAMI` | post_exploit | Identifies the current user in an established session. |
| 701 | `POST_ENUM_UNAME` | post_exploit | Retrieves remote system and kernel information. |
| 702 | `POST_ENUM_ID` | post_exploit | Retrieves UID, GID and group information. |
| 703 | `POST_ENUM_HOSTNAME` | post_exploit | Retrieves the compromised host name. |
| 704 | `POST_ENUM_IP_ADDR` | post_exploit | Inspects network interfaces from the compromised host. |
| 705 | `POST_ENUM_IP_ROUTE` | post_exploit | Inspects the routing table from the compromised host. |
| 706 | `POST_ENUM_SS_LISTENERS` | post_exploit | Lists listening services using `ss`. |
| 707 | `POST_ENUM_NETSTAT_LISTENERS` | post_exploit | Lists listening services using `netstat`. |
| 708 | `POST_ENUM_USERS_PASSWD` | post_exploit | Enumerates local users from `/etc/passwd`. |
| 709 | `POST_ENUM_HOME_USERS` | post_exploit | Identifies users with home directories and likely interactive accounts. |
| 710 | `POST_ENUM_PROCESSES` | post_exploit | Enumerates running processes. |
| 711 | `POST_ENUM_ENV` | post_exploit | Inspects environment variables. |

### Privilege escalation discovery

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 712 | `POST_CHECK_SUDO_PRIVS` | privilege_escalation_discovery | Checks sudo privileges for the current compromised user. |
| 713 | `POST_CHECK_SUDOERS_PERMS` | privilege_escalation_discovery | Checks permissions on `/etc/sudoers`. |
| 714 | `POST_FIND_SUID_BINARIES` | privilege_escalation_discovery | Finds binaries with the SUID bit set. |
| 715 | `POST_PRIVESC_NMAP_INTERACTIVE` | privilege_escalation | Uses legacy SUID Nmap interactive mode to attempt a root shell in the lab. |
| 716 | `POST_PRIVESC_FIND_SUID_SHELL` | privilege_escalation | Uses a SUID `find` GTFOBins technique to attempt a privileged shell. |
| 717 | `POST_READ_SHADOW` | credential_access | Reads `/etc/shadow` after root access to collect password hash evidence. |
| 718 | `POST_LIST_SSH_HOST_KEYS` | credential_access | Lists SSH host key files as sensitive evidence. |
| 719 | `POST_READ_CRONTAB` | post_exploit | Reads system crontab for scheduled task and persistence evidence. |

### Metasploit post-exploitation modules

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 730 | `MSF_POST_ENUM_SYSTEM` | post_exploit | Runs Metasploit Linux system enumeration. |
| 731 | `MSF_POST_ENUM_CONFIGS` | post_exploit | Runs Metasploit Linux configuration enumeration. |
| 732 | `MSF_POST_ENUM_NETWORK` | post_exploit | Runs Metasploit Linux network enumeration. |
| 733 | `MSF_LOCAL_EXPLOIT_SUGGESTER` | privilege_escalation_discovery | Runs Metasploit local exploit suggester against an established session. |

### Pivoting and internal access

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 760 | `MSF_AUTOROUTE_ADD_INTERNAL_NET` | pivoting | Adds a route through the compromised host to reach an internal subnet. |
| 761 | `MSF_START_SOCKS_PROXY` | pivoting | Starts a SOCKS proxy in Metasploit for pivoting. |
| 762 | `PIVOT_SCAN_INTERNAL_NET_PROXYCHAINS` | pivoting | Scans an internal network through the SOCKS proxy using proxychains. |
| 763 | `MSF_PORTFWD_ADD` | pivoting | Forwards a remote internal service port to a local attacker port. |
| 764 | `PIVOT_MYSQL_LOGIN_LOCAL_PORTFWD` | pivoting | Accesses an internal MySQL service through local port forwarding. |
| 765 | `PIVOT_MYSQL_DUMP_ALL_DATABASES` | exfiltration | Dumps internal MySQL databases through the pivot as lab evidence. |

### Credential discovery and evidence collection

| `action_id` | Action | Phase | Description |
|---:|---|---|---|
| 780 | `POST_SEARCH_WEB_PASSWORDS` | credential_access | Searches for password references in web application files. |
| 781 | `POST_SEARCH_WEB_DB_CONFIGS` | credential_access | Searches for database configuration references in web application files. |
| 782 | `POST_READ_DVWA_CONFIG` | credential_access | Reads the DVWA database configuration file. |
| 783 | `POST_READ_PHPMYADMIN_CONFIG` | credential_access | Reads the phpMyAdmin database configuration file. |
| 784 | `POST_READ_TIKIWIKI_DB_CONFIG` | credential_access | Reads the TikiWiki database configuration file. |
| 790 | `POST_ARCHIVE_MYSQL_DATA_DIR` | exfiltration | Archives the MySQL data directory as lab evidence after root access. |
| 791 | `POST_ARCHIVE_POSTGRES_DATA_DIR` | exfiltration | Archives the PostgreSQL data directory as lab evidence after root access. |
| 792 | `MSF_DOWNLOAD_EVIDENCE_FILE` | exfiltration | Downloads a selected evidence file through Meterpreter. |