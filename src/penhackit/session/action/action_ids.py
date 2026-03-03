import re

# action_id -> (name, command)
ACTIONS = {
    0: ("NONE", None),

    # Windows (elige las que te interesen)
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
}

def extract_action_id_from_cmd(cmd: str) -> int:
    """
    MVP extractor: mapea command raw -> action_id usando heurísticas simples.
    Soporta tus ACTIONS actuales.
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
    if re.fullmatch(r"ping -n 1 (?:\d{1,3}\.){3}\d{1,3}", s):
        return 4
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

    return None
