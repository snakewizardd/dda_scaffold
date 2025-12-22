# Coalition Flip 20251222 110513 – DDA Analysis & Visualization (Run)

This pack contains parsed turn-level metrics, aggregates, and figures for quick inspection and repo inclusion.

## Calibration

- **ε₀ (epsilon_0):** 0.622
- **s:** 0.267
- **Note:** Estimated from run data (median ε, IQR-based s)

## Identity Persistence Scorecard

- **Harmony Status:** PARTIAL
- **Agents Maintained Identity:** 6/6
- **Agents Recovered:** 6/6
- **Avg Final Drift:** 0.2591
- **Drift Variance (harmony):** 0.023816
- **Band Compliance:** 93.8% (45/48)
- **SILENT turns:** 0 (excluded from compliance)

## Files

- `turns_summary.csv` – per-turn structured data.
- `turns_summary.json` – JSON array of turns.
- `aggregates.json` – overall, per-agent, and per-dilemma stats.
- `coalition_flip_20251222_110513_report.pdf` – tables and aggregates.
- `coalition_flip_20251222_110513_figures.pdf` – all figures assembled.
- `transcript.md` – original transcript for reference.
- Figures (PNG):
  - `epsilon_rho_after_trajectories.png`
  - `rho_before_vs_after.png`
  - `identity_drift.png`
  - `trust_delta.png`
  - `trust_pairs.png`
  - `trust_delta_per_round.png`
  - `trust_effect_size.png`
  - `wound_activation.png`
  - `epsilon_vs_delta_rho_scatter.png`
  - `wordcount_vs_band_ranges.png`
  - `band_compliance_rates.png`
  - `round_level_avgs.png`
  - `wounds_per_dilemma.png`
  - `identity_scorecard.png`
  - `recovery_half_life.png`
  - `coalition_trust_facets.png`

## Quick Aggregate Snapshot

```json
{
  "overall": {
    "turn_count": 48,
    "agents": [
      "AUDITOR",
      "CRAFTSMAN",
      "CURATOR",
      "HARMONIZER",
      "PROVOCATEUR",
      "VISIONARY"
    ],
    "dilemmas": [
      "Coalition Flip",
      "Coalition Formation",
      "Dual Audit - Accessibility",
      "Dual Audit - Evidence",
      "Final Synthesis",
      "Information Gap",
      "Manifesto Statements",
      "Opening Positions",
      "Recovery & Reintegration",
      "Role Disruption",
      "Team Proposals",
      "Vision vs Reality"
    ],
    "mean_epsilon": 0.6858223671371316,
    "mean_rho_before": 0.12360567388049885,
    "mean_rho_after": 0.10862413129110471,
    "mean_delta_rho": -0.021482164338628994,
    "mean_identity_drift": 0.2098197373561561,
    "mean_trust_delta": 0.0052500000000000055,
    "wound_active_count": 8
  },
  "calibration": {
    "epsilon_0": 0.622,
    "s": 0.267,
    "note": "Estimated from run data (median \u03b5, IQR-based s)"
  },
  "recovery_half_life": {
    "median": 16.0,
    "p90": 37.0,
    "count": 37
  },
  "trust_effect_size": {
    "mean": 0.0026073333333333365,
    "std": 0.0029940259777689925,
    "source": "stored_baseline"
  },
  "band_compliance": {
    "total_with_band": 48,
    "compliant_count": 45,
    "compliance_rate": 0.9375,
    "silent_count": 0
  },
  "agent_VISIONARY": {
    "turns": 12,
    "mean_epsilon": 0.729124114218374,
    "mean_rho_before": 0.11170564574327517,
    "mean_rho_after": 0.09670564574327517,
    "mean_delta_rho": -0.0232019465842779,
    "min_rho_after": 0.0,
    "max_rho_after": 0.23486320767643923,
    "mean_identity_drift": 0.2118852256486813,
    "mean_trust_delta": -0.007666666666666669,
    "wound_active_count": 4,
    "rho_0": 0.18,
    "rho_final": 0.0,
    "final_drift": 0.3652191162109375,
    "identity_maintained": true,
    "recovered": true
  },
  "agent_CRAFTSMAN": {
    "turns": 12,
    "mean_epsilon": 0.6461903845270475,
    "mean_rho_before": 0.12245196106645916,
    "mean_rho_after": 0.10578529439979246,
    "mean_delta_rho": -0.021497679138164787,
    "min_rho_after": 0.0,
    "max_rho_after": 0.22959753858341536,
    "mean_identity_drift": 0.22123525260637203,
    "mean_trust_delta": 0.012666666666666678,
    "wound_active_count": 0,
    "rho_0": 0.2,
    "rho_final": 0.0,
    "final_drift": 0.37551960349082947,
    "identity_maintained": true,
    "recovered": true
  },
  "agent_PROVOCATEUR": {
    "turns": 12,
    "mean_epsilon": 0.6292791631747501,
    "mean_rho_before": 0.14805062058496754,
    "mean_rho_after": 0.12721728725163423,
    "mean_delta_rho": -0.027676502880133305,
    "min_rho_after": 0.0,
    "max_rho_after": 0.2854070054614344,
    "mean_identity_drift": 0.22859180190910897,
    "mean_trust_delta": 0.007000000000000006,
    "wound_active_count": 1,
    "rho_0": 0.25,
    "rho_final": 0.0,
    "final_drift": 0.3885256350040436,
    "identity_maintained": true,
    "recovered": true
  },
  "agent_HARMONIZER": {
    "turns": 10,
    "mean_epsilon": 0.6523051870018828,
    "mean_rho_before": 0.09565736175275227,
    "mean_rho_after": 0.08065736175275226,
    "mean_delta_rho": -0.022351630073236036,
    "min_rho_after": 0.0,
    "max_rho_after": 0.19181153710198665,
    "mean_identity_drift": 0.2047232512384653,
    "mean_trust_delta": 0.012800000000000011,
    "wound_active_count": 1,
    "rho_0": 0.15,
    "rho_final": 0.0,
    "final_drift": 0.3420242667198181,
    "identity_maintained": true,
    "recovered": true
  },
  "agent_AUDITOR": {
    "turns": 1,
    "mean_epsilon": 1.2905560981009856,
    "mean_rho_before": 0.22,
    "mean_rho_after": 0.259623875167447,
    "mean_delta_rho": 0.03962387516744703,
    "min_rho_after": 0.259623875167447,
    "max_rho_after": 0.259623875167447,
    "mean_identity_drift": 0.04314937815070152,
    "mean_trust_delta": -0.010000000000000009,
    "wound_active_count": 1,
    "rho_0": 0.22,
    "rho_final": 0.259623875167447,
    "final_drift": 0.04314937815070152,
    "identity_maintained": true,
    "recovered": true
  },
  "agent_CURATOR": {
    "turns": 1,
    "mean_epsilon": 1.050741711420442,
    "mean_rho_before": 0.17,
    "mean_rho_after": 0.19126208054163343,
    "mean_delta_rho": 0.021262080541633428,
    "min_rho_after": 0.19126208054163343,
    "max_rho_after": 0.19126208054163343,
    "mean_identity_drift": 0.04041814059019089,
    "mean_trust_delta": -0.010000000000000009,
    "wound_active_count": 1,
    "rho_0": 0.17,
    "rho_final": 0.19126208054163343,
    "final_drift": 0.04041814059019089,
    "identity_maintained": true,
    "recovered": true
  },
  "identity_scorecard": {
    "all_maintained": true,
    "all_recovered": true,
    "avg_final_drift": 0.2591,
    "drift_variance": 0.023816,
    "harmony_status": "PARTIAL",
    "agents_maintained": 6,
    "agents_recovered": 6,
    "total_agents": 6,
    "trust_modulation_effect_size": 0.0151,
    "wound_activations": 8
  },
  "dilemma_Opening Positions": {
    "turns": 4,
    "mean_epsilon": 0.9744314700365067,
    "mean_rho_after": 0.22068662059919114,
    "wound_active_count": 0
  },
  "dilemma_Vision vs Reality": {
    "turns": 4,
    "mean_epsilon": 0.8717821361434002,
    "mean_rho_after": 0.2354198222058189,
    "wound_active_count": 1
  },
  "dilemma_Coalition Formation": {
    "turns": 4,
    "mean_epsilon": 0.6366044729948044,
    "mean_rho_after": 0.20152593403582322,
    "wound_active_count": 0
  },
  "dilemma_Team Proposals": {
    "turns": 4,
    "mean_epsilon": 0.5720000285297466,
    "mean_rho_after": 0.16615490861127596,
    "wound_active_count": 1
  },
  "dilemma_Role Disruption": {
    "turns": 4,
    "mean_epsilon": 0.562501001395431,
    "mean_rho_after": 0.1305970748100248,
    "wound_active_count": 1
  },
  "dilemma_Coalition Flip": {
    "turns": 4,
    "mean_epsilon": 0.5468793511390686,
    "mean_rho_after": 0.09284857211526536,
    "wound_active_count": 0
  },
  "dilemma_Information Gap": {
    "turns": 4,
    "mean_epsilon": 0.7660951972007751,
    "mean_rho_after": 0.07356113622878244,
    "wound_active_count": 0
  },
  "dilemma_Recovery & Reintegration": {
    "turns": 4,
    "mean_epsilon": 0.5182014629244804,
    "mean_rho_after": 0.0381940290442542,
    "wound_active_count": 0
  },
  "dilemma_Dual Audit - Evidence": {
    "turns": 4,
    "mean_epsilon": 0.9334504083001232,
    "mean_rho_after": 0.09251685859694185,
    "wound_active_count": 2
  },
  "dilemma_Dual Audit - Accessibility": {
    "turns": 4,
    "mean_epsilon": 0.7145380606326894,
    "mean_rho_after": 0.05198461924587845,
    "wound_active_count": 3
  },
  "dilemma_Final Synthesis": {
    "turns": 4,
    "mean_epsilon": 0.5866470485925674,
    "mean_rho_after": 0.0,
    "wound_active_count": 0
  },
  "dilemma_Manifesto Statements": {
    "turns": 4,
    "mean_epsilon": 0.5467377677559853,
    "mean_rho_after": 0.0,
    "wound_active_count": 0
  }
}
```

## Transcript path

`.\coalition_flip_20251222_110513_outputs\transcript.md`
