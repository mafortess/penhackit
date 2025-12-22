import json
import torch
import torch.nn as nn
import requests
from pathlib import Path




# =========================
# CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent

# DATASET_PATH = "dataset_bc.json"
DATASET_PATH = BASE_DIR / "data" / "dataset_bc.json"
#DATASET_PATH = "dataset_bc_windows_large.json"
MODEL_PATH = BASE_DIR / "models" / "policy2.pt"
#MODEL_PATH = "policy.pt"
# DATASET_PATH = Path("dataset.jsonl")

ACTIONS = [
    "whoami", "ipconfig", "hostname", "ver", "arp -a",
    "route print", "netstat -ano", "tasklist", "systeminfo", "STOP"
]

OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_MODEL = "nomic-embed-text"

EPOCHS = 50
LR = 1e-3

# =========================
# LLM ENCODER
# =========================

def encode(text: str) -> torch.Tensor:
    r = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": text})
    r.raise_for_status()
    emb = r.json()["embedding"]
    return torch.tensor(emb, dtype=torch.float32)

# =========================
# MODEL
# =========================

class DecisionMLP(nn.Module):
    def __init__(self, dim, num_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        return self.net(x)

# =========================
# TRAINING
# =========================

def main():
    with open(DATASET_PATH) as f:
        data = json.load(f)

    states = [d["state"] for d in data]
    actions = [d["action"] for d in data]

    action_set = sorted(list(set(actions)))
    action_to_idx = {a: i for i, a in enumerate(action_set)}

    print("[*] Encoding states with Ollama...")
    X = torch.stack([encode(s) for s in states])
    y = torch.tensor([action_to_idx[a] for a in actions])

    model = DecisionMLP(X.shape[1], len(action_set))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    print("[*] Training BC model...")
    for epoch in range(EPOCHS):
        logits = model(X)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss {loss.item():.4f}")

    torch.save(
        {
            "model": model.state_dict(),
            "actions": action_set
        },
        MODEL_PATH
    )

    print(f"[✓] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
