
def policy_decide_action(state, t):
    print("Deciding action based on state...")
    action = t

    return action

def rules_policy_decide_action(state: dict) -> int:
    # Reglas mínimas y deterministas para tu MVP (Windows-centric)
    # 1) Si aún no tenemos IPv4, primero ipconfig /all
    if int(state.get("net_ipv4_count", 0) or 0) == 0:
        return 1  # INSPECT_IPCONFIG

    # 2) Si aún no tenemos vecinos ARP, pedir arp -a
    if int(state.get("net_arp_count", 0) or 0) == 0:
        return 2  # INSPECT_ARP

    # 3) Si ya hay hosts, prueba ping al foco/primer host
    if int(state.get("hosts_count", 0) or 0) > 0:
        return 4  # PING_FOCUS_HOST

    # 4) Fallback: nada que hacer
    return 0  # NONE

def model_policy_decide_action(state: dict, model, feature_names: list[str]) -> int:
    """
    model: cualquier sklearn classifier ya entrenado (joblib.load(...))
    feature_names: lista de features en el orden usado al entrenar (metrics.json["feature_names"])
    """
    x = np.zeros((1, len(feature_names)), dtype=np.float32)

    for j, k in enumerate(feature_names):
        v = state.get(k, 0)
        if isinstance(v, bool):
            v = 1 if v else 0
        if v is None:
            v = 0
        x[0, j] = float(v)

    y_pred = model.predict(x)
    return int(y_pred[0])
