# House of David — DDA‑X Court Dynamics Simulator

## What you get
- 3 agents (David, Mychal, Mirab)
- DDA‑X: surprise → rigidity → contraction (decoding + bandwidth + candidate depth)
- 3000‑D latent state space (projection from 3072‑D embeddings)
- Hybrid backend: Azure Phi‑4 for voice, OpenAI embeddings for state

## Run
```bash
python house_of_david_ddax_sim.py --turns 24 --seed 7
```

## Outputs
- `data/house_of_david_ddax/<timestamp>/session_log.json`
- `data/house_of_david_ddax/<timestamp>/transcript.md`
- `data/house_of_david_ddax/<timestamp>/ledgers.pkl`

## Env Vars
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (optional)
- `AZURE_PHI_DEPLOYMENT` (optional, default `Phi-4`)
- `OPENAI_API_KEY` (or `OAI_API_KEY`)
- `OAI_EMBED_MODEL` (optional, default `text-embedding-3-large`)
