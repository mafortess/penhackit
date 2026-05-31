# enumeration_parser.py

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

ENUMERATION_PARSERS = {
    "enum_ftp_banner": parse_enum_ftp_banner,
    "enum_ssh_banner": parse_enum_ssh_banner,
    "enum_ftp_nmap_scripts": parse_enum_ftp_nmap_scripts,
    "enum_ssh_nmap_scripts": parse_enum_ssh_nmap_scripts,
    "enum_generic_banner": parse_generic_banner,
    "nmap_ftp_anon": parse_nmap_ftp_anon,
    "enum_dns_version_bind": parse_enum_dns_version_bind,
    "enum_dns_any": parse_enum_dns_any,
    "enum_dns_zone_transfer": parse_enum_dns_zone_transfer,
    "enum_nfs_exports": parse_enum_nfs_exports,
    "enum_rpcinfo": parse_enum_rpcinfo,
    "enum_mysql_info": parse_enum_mysql_info,
    "enum_postgres_info": parse_enum_postgres_info,
    "enum_rdp_info": parse_enum_rdp_info,
    "enum_vnc_info": parse_enum_vnc_info,
}