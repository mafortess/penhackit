
from penhackit.session.action.action_ids import ACTIONS
import re

def command_builder(action, kb):
    print("Building command from action and KB...")
    try:
        name, cmd = ACTIONS.get(action, (None, None))
    except Exception as e:
        print(f"Error retrieving action from catalog: {e}")
        return None

    if not cmd:
        print(f"No command template found for action: {action}")
        return None

    # Reemplaza placeholders en cmd con datos de KB (ejemplo simple)
    if "{" not in cmd:
        print("No placeholders in command template, returning as is.")
        return cmd
    
    try:
        # Extrae los placeholders del comando
        hosts = kb.get("hosts", [])
        ip = hosts[0].get("ip", None) if hosts else None  # Ejemplo: toma la primera IP de la KB
        
        if "{ip}" in cmd and not ip:
            print("No IP available in KB to build command.")
            return None
    except Exception as e:
        print(f"Error extracting data from KB: {e}")
        return None
    
    cmd = cmd.format(ip=ip)  # Reemplaza {ip} en el comando

    return cmd