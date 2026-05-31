
    # ============================================================
    # 000-099 CONTROL
    # ============================================================
CONTROL_ACTIONS = { 
    0: {
        "name": "STOP",
        "category": "control",
        "phase": "stop",
        "tool": None,
        "command_template": None,
        "placeholders": [],
        "parser_family": None,
        "expected_events": ["SESSION_STOPPED"],
        "description": "Stop the session.",
    },

    1: {
        "name": "NO_OP",
        "category": "control",
        "phase": "control",
        "tool": None,
        "command_template": None,
        "placeholders": [],
        "parser_family": None,
        "expected_events": ["NO_ACTION"],
        "preconditions": {},
        "risk_level": "safe",
        "description": "Do nothing in the current step.",
    },
}