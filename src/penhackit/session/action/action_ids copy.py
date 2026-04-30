import re

# action_id -> (name, command)
ACTIONS = {
    0: ("NONE", None),

    # Windows
    1: ("INSPECT_IPCONFIG", "ipconfig /all"),
    2: ("INSPECT_ARP", "arp -a"),
    3: ("INSPECT_ROUTE", "route print"),
    # : ("INSPECT_NETSTAT", "netstat -ano"),
    4: ("PING_FOCUS_HOST", "ping -n 1 {ip}"),

    # Linux/Kali (si lo ejecutas allí)
    101: ("INSPECT_IP_A", "ip a"),
    102: ("INSPECT_IP_R", "ip r"),
    103: ("INSPECT_IP_NEIGH", "ip neigh"),
    104: ("INSPECT_SS", "ss -tulpn"),
    105: ("PING_FOCUS_HOST", "ping -c 1 -W 2 {ip}"),

    # Pentesting MVP
    200: ("DISCOVER_HOSTS", "nmap -sn {target}"),
    210: ("SCAN_TOP_TCP_PORTS", "nmap --top-ports 1000 --open -T3 {target_ip}"),
    220: ("DETECT_SERVICES", "nmap -sV -sC -O -T3 -p {known_open_ports_csv} {target_ip}"),
    300: ("ENUM_HTTP_HEADERS", "curl -I --max-time 10 http://{target_ip}:{target_port}"),
    310: ("ENUM_HTTP_DIRS", "gobuster dir -u http://{target_ip}:{target_port} -w /usr/share/wordlists/dirb/common.txt -q"),
    320: ("ENUM_SMB_SHARES", "smbclient -L //{target_ip} -N"),
    400: ("CHECK_SERVICE_VERSION_VULNS", "searchsploit {service_version_string}"),
}

def extract_action_id_from_cmd(cmd: str) -> int:
    """
    Extracts the action ID from a command string.
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
    
    # Linux/Kali
    if s == "ip a":
        return 101
    if s == "ip r":
        return 102
    if s == "ip neigh":
        return 103
    if s == "ss -tulpn":
        return 104

     # Pentesting
    if s.startswith("nmap -sn "):
        return 200
    if s.startswith("nmap ") and "--top-ports" in s and "--open" in s:
        return 210
    if s.startswith("nmap ") and "-sv" in s:
        return 220
    if s.startswith("curl -i ") or s.startswith("curl -i") or s.startswith("curl -i "):
        return 300
    if s.startswith("curl -i") or s.startswith("curl -I".lower()):
        return 300
    if s.startswith("gobuster dir "):
        return 310
    if s.startswith("smbclient -l ") or s.startswith("smbclient -L".lower()):
        return 320
    if s.startswith("searchsploit "):
        return 400
    
    return None
