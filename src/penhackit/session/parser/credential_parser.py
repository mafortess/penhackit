import re

# CREDENTILAS ATTACKS
def parse_bruteforce_ssh_lab(stdout: str, stderr: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="ssh")


def parse_bruteforce_ftp_lab(stdout: str, stderr: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="ftp")


def parse_bruteforce_http_login_lab(stdout: str, stderr: str, target_ip: str | None, target_port: int | None) -> list[dict]:
    return parse_hydra_credentials(stdout, stderr, target_ip, target_port, service="http")


def parse_hydra_credentials(stdout: str, stderr: str, target_ip: str | None, target_port: int | None, service: str,
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


CREDENTIALS_PARSERS = {
    "bruteforce_ssh_lab": parse_bruteforce_ssh_lab,
    "bruteforce_ftp_lab": parse_bruteforce_ftp_lab,
    "bruteforce_http_login_lab": parse_bruteforce_http_login_lab,
    "check_ftp_anonymous_login": parse_check_ftp_anonymous_login,
}