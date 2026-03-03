def build_state(kb: dict, session_context: dict) -> dict:
    print("Building state from KB and session context...")

    net = kb.get("net", {}) or {}
    focus = kb.get("focus", {}) or {}

    hosts = kb.get("hosts", []) or []
    services = kb.get("services", []) or []
    findings = kb.get("findings", []) or []

    arp_neighbors = net.get("arp_neighbors", []) or []
    ipv4 = net.get("ipv4", []) or []
    default_gw = net.get("default_gw", []) or []
    interfaces = net.get("interfaces", []) or []
    routes = net.get("routes", []) or []

    state = {
        # Goal / task
        "goal_type": session_context.get("goal_type", "demo"),

        # Focus (nivel y si hay algo seleccionado)
        "focus_level": focus.get("level", "global"),
        "has_focus_host": bool(focus.get("host")),
        "has_focus_service": bool(focus.get("service")),

        # Features por “nivel” (resumen, no datos crudos)
        "net_ipv4_count": len(ipv4),
        "net_gw_count": len(default_gw),
        "net_if_count": len(interfaces),
        "net_arp_count": len(arp_neighbors),
        "net_routes_count": len(routes),

        "hosts_count": len(hosts),
        "services_count": len(services),
        "findings_count": len(findings),

        # Last transition (para que la policy no repita/pueda detectar error)
        "last_action_id": kb.get("last_action_id"),
        "last_action_name": kb.get("last_action_name"),
        "last_rc": kb.get("last_rc"),
        "last_event_type": kb.get("last_event_type"),

        # Progreso / estancamiento (mínimo)
        "step_idx": kb.get("step_idx", 0),
    }
    return state