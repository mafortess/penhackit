import subprocess
import torch
import requests
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


# =========================
# CONFIG
# =========================

# TARGET = "192.168.56.101"
TARGET = "localhost"
MODEL_PATH = BASE_DIR / "models" / "policy.pt"
#MODEL_PATH = "policy.pt"
# DATASET_PATH = Path("dataset.jsonl")

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

MAX_STEPS = 5

ACTIONS = [
    "whoami",
    "ipconfig",
    "hostname",
    "ver",
    "arp -a",
    "route print",
    "netstat -ano",
    "tasklist",
    "systeminfo",
    "STOP"
]


# =========================
# LLM ENCODER
# =========================

def encode(text: str) -> torch.Tensor:
    r = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": text}
    )
    r.raise_for_status()
    emb = r.json()["embedding"]
    return torch.tensor(emb, dtype=torch.float32)

# =========================
# LOAD POLICY
# =========================

class DecisionMLP(torch.nn.Module):
    def __init__(self, dim, num_actions):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)

ckpt = torch.load(MODEL_PATH)
ACTIONS = ckpt["actions"]

dummy_dim = encode("test").shape[0]
model = DecisionMLP(dummy_dim, len(ACTIONS))
model.load_state_dict(ckpt["model"])
model.eval()

MAX_STEPS = 5

# =========================
# AGENT LOOP
# =========================

def main():
    facts = {
        "user": None,
        "hostname": None,
        "osver": None,
        "has_ipconfig": False,
        "has_arp": False,
        "has_routes": False,
        "has_netstat": False,
        "has_tasklist": False,
        "has_systeminfo": False,
    }
     # state = f"Target specified: {TARGET}. No scans performed yet."
    state = "Initial state. No information collected yet."
    # state = f"Target specified: {TARGET}. No commands executed yet."
    
    trace = ""

    decision_state = "Initial state. No information collected yet."

    for step in range(MAX_STEPS):

        print("\n==============================")
        print(f"[STEP {step}]")
        print("[STATE]")
        # print("\n".join(state.split("\n")[:5]))

        if trace:
            print("\n[TRACE]")
            print(trace)

        emb = encode(state)
        print("\n[EMBEDDING] Computed.")
        print("Embedding preview:", emb[:5].tolist())

        with torch.no_grad():
            scores = model(emb.unsqueeze(0))[0]

        action = ACTIONS[torch.argmax(scores).item()]
        print(f"\n[DECISION] {action}")

        if action == "STOP":
            print("[AGENT] Stopping.")
            break

        result = subprocess.run(
            action,
            shell=True,
            capture_output=True, # Capture output
            text=True,
            encoding="cp850",
            errors="replace"
        )
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()

        # Solo para mostrar/depurar, NO para decidir
        trace = trace + f"\n\n[CMD] {action}\n{out}"
        if err:
            trace = trace + f"\n[STDERR]\n{err}"

        # ---------- Actualiza "facts" según la acción ----------
        if action == "whoami" and out:
            facts["user"] = out
        elif action == "hostname" and out:
            facts["hostname"] = out
        elif action == "ver" and out:
            facts["osver"] = out
        elif action == "ipconfig":
            facts["has_ipconfig"] = True
        elif action == "arp -a":
            facts["has_arp"] = True
        elif action == "route print":
            facts["has_routes"] = True
        elif action == "netstat -ano":
            facts["has_netstat"] = True
        elif action == "tasklist":
            facts["has_tasklist"] = True
        elif action == "systeminfo":
            facts["has_systeminfo"] = True

        # ---------- Construye un estado "estilo dataset" ----------
        if facts["user"] is None:
            state = "No user identity collected yet. Run whoami."
        elif facts["hostname"] is None:
            state = "User identity is known (whoami already executed). Hostname not collected yet."
        elif facts["osver"] is None:
            state = "Hostname has been collected. Determine Windows version quickly."
        elif not facts["has_ipconfig"]:
            state = "User and hostname known. Network configuration missing. Obtain ipconfig."
        elif not facts["has_arp"]:
            state = "Network configuration collected. ARP cache missing. Run arp -a."
        elif not facts["has_routes"]:
            state = "ARP collected. Routing table missing. Run route print."
        elif not facts["has_netstat"]:
            state = "Routes known. Active connections missing. Run netstat -ano."
        elif not facts["has_tasklist"]:
            state = "Netstat executed; need process list to map PIDs. Run tasklist."
        elif not facts["has_systeminfo"]:
            state = "Basic network/process data collected. Collect systeminfo for report."
        else:
            state = "All basic information collected. Stop."

        print("\n[TERMINAL OUTPUT]")
        print(result.stdout)

        if result.stderr:
            print("[TERMINAL ERROR]")
            print(result.stderr)
        
        out = (result.stdout or "").strip()
        err = (result.stderr or "").strip()
        if not out and err:
            out = err
        if not out:
            out = "<NO_OUTPUT>"

        # state = state + f"\n\n[CMD] {action}\n[RC] {result.returncode}\n{out}" # Para mantener contexto
        # state = state + f"\n\nLast command executed: {action}\n{out}"
        # state = result.stdout.strip()
        state = state + f"\n\n[CMD] {action}\n{out}"
        
    print("\n[AGENT] Finished.")

if __name__ == "__main__":
    main()
