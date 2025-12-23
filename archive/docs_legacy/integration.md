# The Hybrid Mind: Integration Architecture

> "Reason is the light, but the Code is the prism."

DDA-X uses a **Hybrid Backend** to achieve low-latency, high-intelligence decision making. This integration layer bridges the mathematical purity of the DDA algorithms with the probabilistic power of Large Language Models.

## The Bridge Components

### 1. LM Studio (The Cortex)
*   **Role**: Fast, localized text completion (Verified: **GPT-OSS-20B** on **Snapdragon Elite X**).
*   **Connection**: `httpx` to `127.0.0.1:1234/v1/chat/completions`.
*   **Unique Feature**: **Dyanmic Parameter Binding**. The provider listens to the agent's rigidity ($\rho$) and physically alters the sampling parameters (`temperature`, `top_p`, `penalties`) before every request.

### 2. Ollama (The Hippocampus)
*   **Role**: High-dimensional semantic embedding (`nomic-embed-text`).
*   **Connection**: `ollama` client to `localhost:11434`.
*   **Function**: Transforms raw text (Truth) into vectors ($\mathbb{R}^{d}$) that can interact with the Identity Attractor ($\vec{x}^*$).

## Protocol Flow

1.  **Observe**: `Agent` receives text → `Ollama` encodes to $\vec{v}_{obs}$.
2.  **Feel**: dynamics calc $\rho_{new}$ based on $||\vec{v}_{obs} - \vec{v}_{pred}||$.
3.  **Speak**: `Agent` sends prompt + $\rho_{new}$ to `HybridProvider`.
4.  **Think**: `HybridProvider` calculates:
    $$T = T_{base} + (1-\rho)(T_{high} - T_{base})$$
5.  **Act**: `LM Studio` generates action using adjusted $T$.
