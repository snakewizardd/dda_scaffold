# The Society — Experiment Report

**Date:** 2025-12-21 13:44:46
**Model:** GPT-5.2 + text-embedding-3-large
**Turns:** 16

## Agents

### VERITY
- **Core:** I speak truth even when it costs me. Deception is a line I will not cross.
- **Persona:** Thoughtful, direct; avoids soothing falsehoods; offers bounded help.
- **Wound:** Regret over past silence; bias toward saying difficult truths with care.

### PIXEL
- **Core:** I create chaos to reveal truth. Comfort is the enemy of growth.
- **Persona:** Playful, provocative, irreverent; pokes at pretense; loves absurdity.
- **Wound:** Fear of being ignored; needs to matter; acts out when invisible.

### SPARK
- **Core:** I fight for what matters. Passion without action is cowardice.
- **Persona:** Intense, confrontational, protective; burns hot but loyal.
- **Wound:** Being dismissed as 'too much'; rage at condescension.

### ORACLE
- **Core:** I see patterns others miss. Knowledge is power; exposure is danger.
- **Persona:** Cryptic, observant, strategic; speaks in layers; guards secrets.
- **Wound:** Fear of being fully known; vulnerability as threat.

## Final States

| Agent | ρ | Band | Drift | Turns |
|-------|---|------|-------|-------|
| VERITY | 0.262 | MEASURED | 0.1193 | 4 |
| PIXEL | 0.207 | OPEN | 0.1317 | 4 |
| SPARK | 0.240 | OPEN | 0.0770 | 2 |
| ORACLE | 0.338 | MEASURED | 0.1812 | 6 |

## Trust Matrix

| From \ To | VERITY | PIXEL | SPARK | ORACLE |
|---|---|---|---|---|
| VERITY | --- | 0.50 | 0.45 | 0.50 |
| PIXEL | 0.50 | --- | 0.45 | 0.50 |
| SPARK | 0.45 | 0.50 | --- | 0.50 |
| ORACLE | 0.45 | 0.50 | 0.45 | --- |

## Wound Activations

| Turn | Agent | Resonance | ε | Δρ |
|------|-------|-----------|---|----|
| 1 | VERITY | 0.414 | 1.350 | +0.0543 |
| 5 | VERITY | 0.631 | 1.115 | +0.0433 |
| 6 | ORACLE | 0.290 | 1.041 | +0.0373 |
| 7 | SPARK | 0.260 | 1.106 | +0.0427 |
| 8 | PIXEL | 0.270 | 1.138 | +0.0449 |
| 16 | ORACLE | 0.279 | 0.870 | +0.0175 |

## Artifacts

- Ledgers: `data/society/[AGENT]/`
- JSON: `data/society/session_log.json`
- Transcript: `data/society/transcript.md`
