import json
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Path to the session log
log_path = Path("data/house_of_david_ddax_v3/20251226_140831/session_log.json")

def visualize():
    if not log_path.exists():
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, "r") as f:
        data = json.load(f)

    turns = [t["turn"] for t in data["turns"]]
    agents = ["DAVID", "MYCHAL", "MIRAB"]
    
    # Plotting setup
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("House of David - DDA-X Dynamics (Hybrid Phi-OAI)", fontsize=16)

    # 1. Surprise (Epsilon)
    for agent in agents:
        eps = [t["agents"][agent]["metrics"]["epsilon"] for t in data["turns"]]
        axs[0, 0].plot(turns, eps, label=agent, marker='o', markersize=4)
    axs[0, 0].set_title("Surprise (ε)")
    axs[0, 0].set_ylabel("Epsilon")
    axs[0, 0].legend()
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Rigidity (Rho)
    for agent in agents:
        rho = [t["agents"][agent]["metrics"]["rho_after"] for t in data["turns"]]
        axs[0, 1].plot(turns, rho, label=agent, marker='s', markersize=4)
    axs[0, 1].set_title("Rigidity (ρ)")
    axs[0, 1].set_ylabel("Rho")
    axs[0, 1].legend()
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Core Drift
    for agent in agents:
        drift = [t["agents"][agent]["metrics"]["core_drift"] for t in data["turns"]]
        axs[1, 0].plot(turns, drift, label=agent, marker='^', markersize=4)
    axs[1, 0].set_title("Core Identity Drift")
    axs[1, 0].set_ylabel("L2 Distance")
    axs[1, 0].legend()
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Social Alignment
    align_keys = ["D↔M", "D↔R", "M↔R"]
    for key in align_keys:
        vals = [t["alignment"][key] for t in data["turns"]]
        axs[1, 1].plot(turns, vals, label=key, marker='x', markersize=4)
    axs[1, 1].set_title("Pairwise Alignment (Cosine)")
    axs[1, 1].set_ylabel("Similarity")
    axs[1, 1].legend()
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save output
    output_png = log_path.parent / "dda_dynamics.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved visualization to {output_png}")

if __name__ == "__main__":
    visualize()
