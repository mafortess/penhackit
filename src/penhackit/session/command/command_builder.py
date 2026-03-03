
from penhackit.session.action.action_ids import ACTIONS
import re

def command_builder(action, kb):
    print("Building command from action and KB...")
    cmd = ACTIONS.get(action, (None, None))[1]

    if not cmd:
        return None

    # Reemplaza placeholders en cmd con datos de KB (ejemplo simple)
    if "{" not in cmd:
        return cmd
    hosts = kb.get("hosts", [])
    ip = hosts[0].get("ip", None) if hosts else None  # Ejemplo: toma la primera IP de la KB
    if "{ip}" in cmd and not ip:
        print("No IP available in KB to build command.")
        return None
    cmd = cmd.format(ip=ip)  # Reemplaza {ip} en el comando

    return cmd