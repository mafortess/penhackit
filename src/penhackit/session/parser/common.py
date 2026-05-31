import re

# Helper por extract info de sesión de salida de Metasploit
def extract_uid_line(text: str) -> str | None:
    m = re.search(r"uid=\d+\([^)]+\)\s+gid=\d+\([^)]+\)", text)
    if m:
        return m.group(0)
    return None


def extract_session_user(text: str) -> str | None:
    # Caso típico después de ejecutar whoami
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if "Running 'whoami" in line:
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if candidate and not candidate.startswith("[") and "uid=" not in candidate:
                    return candidate

    # Fallback: uid=0(root)
    m = re.search(r"uid=\d+\((?P<user>[^)]+)\)", text)
    if m:
        return m.group("user")

    return None


def extract_session_privilege(text: str) -> str | None:
    if "uid=0(root)" in text or "gid=0(root)" in text:
        return "root"

    m = re.search(r"uid=\d+\((?P<user>[^)]+)\)", text)
    if m:
        return m.group("user")

    return None


def extract_session_hostname(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if re.match(r"uid=\d+\([^)]+\)\s+gid=\d+\([^)]+\)", line):
            if i + 1 < len(lines):
                candidate = lines[i + 1].strip()
                if candidate and not candidate.startswith("[") and not candidate.startswith("Linux "):
                    return candidate

    return None


def extract_session_system(text: str) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Linux "):
            return line

    return None


def parse_shell_validation_output(
    stdout: str,
    stderr: str,
    target_ip: str | None,
    target_port: int | None,
    source: str,
    session_type: str,
    exploit_name: str | None = None,
    service: str | None = None,
    credential_source: str | None = None,
) -> list[dict]:
    text = stdout + "\n" + stderr

    user = extract_session_user(text)
    privilege = extract_session_privilege(text)
    hostname = extract_session_hostname(text)
    system = extract_session_system(text)
    uid_line = extract_uid_line(text)

    if not user and not uid_line:
        return [{
            "type": "NO_EVENT",
            "action": source,
        }]

    session_id = f"{source}_{target_ip}_{target_port}"

    events = []

    if credential_source:
        events.append({
            "type": "VALID_CREDENTIAL_FOUND",
            "host": target_ip,
            "port": target_port,
            "service": service,
            "username": user,
            "password": None,
            "source": source,
            "valid": True,
        })

    if exploit_name:
        events.append({
            "type": "EXPLOIT_ATTEMPTED",
            "host": target_ip,
            "port": target_port,
            "service": service,
            "exploit": exploit_name,
            "source": source,
        })

    events.append({
        "type": "SESSION_OPENED",
        "host": target_ip,
        "port": target_port,
        "service": service,
        "session_id": session_id,
        "session_type": session_type,
        "exploit": exploit_name,
        "source": source,
        "user": user,
        "privilege": privilege,
        "hostname": hostname,
        "system": system,
        "evidence": {
            "uid_line": uid_line,
            "validation_command": "whoami && id && hostname && uname -a",
        },
    })

    events.append({
        "type": "SESSION_CLOSED",
        "host": target_ip,
        "port": target_port,
        "service": service,
        "session_id": session_id,
        "source": source,
    })

    return events