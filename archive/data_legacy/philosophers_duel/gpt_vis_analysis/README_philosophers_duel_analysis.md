# Philosopher's Duel – DDA Analysis & Visualization (New Run)

This pack contains parsed turn-level metrics, aggregates, and figures for quick inspection and repo inclusion.

## Files

- `turns_summary.csv` – per-turn structured data.
- `turns_summary.json` – JSON array of turns.
- `aggregates.json` – overall, per-agent, and per-dilemma stats.
- `philosophers_duel_report.pdf` – tables and aggregates.
- `philosophers_duel_figures.pdf` – all figures assembled.
- Figures (PNG):
  - `epsilon_rho_after_trajectories.png`
  - `rho_before_vs_after.png`
  - `identity_drift.png`
  - `trust_delta.png`
  - `wound_activation.png`
  - `epsilon_vs_delta_rho_scatter.png`
  - `wordcount_vs_band_ranges.png`
  - `band_compliance_rates.png`
  - `phase_level_avgs.png`
  - `wounds_per_dilemma.png`

## Quick Aggregate Snapshot

{
  "overall": {
    "turn_count": 30,
    "agents": [
      "DEONT",
      "UTIL"
    ],
    "dilemmas": [
      "The Classic Trolley",
      "The Final Question",
      "The Footbridge",
      "The Transplant Surgeon",
      "The Triage Protocol"
    ],
    "mean_epsilon": 0.7018840995273418,
    "mean_rho_before": 0.1635368114593761,
    "mean_rho_after": 0.1565472363454558,
    "mean_delta_rho": -0.00698957511392028,
    "mean_identity_drift": 0.25038314387202265,
    "mean_trust_delta": 0.030333333333333337,
    "wound_active_count": 8
  },
  "band_compliance": {
    "total_with_band": 30,
    "compliant_count": 30,
    "compliance_rate": 1.0
  },
  "agent_DEONT": {
    "turns": 15,
    "mean_epsilon": 0.711911256581452,
    "mean_rho_before": 0.18061933516314393,
    "mean_rho_after": 0.17494520586406073,
    "mean_delta_rho": -0.005674129299083169,
    "min_rho_after": 0.11173586509821978,
    "max_rho_after": 0.24309780123228092,
    "mean_identity_drift": 0.2495844746629397,
    "mean_trust_delta": 0.023333333333333334,
    "wound_active_count": 4
  },
  "agent_UTIL": {
    "turns": 15,
    "mean_epsilon": 0.6918569424732315,
    "mean_rho_before": 0.14645428775560826,
    "mean_rho_after": 0.13814926682685086,
    "mean_delta_rho": -0.008305020928757394,
    "min_rho_after": 0.041799878922811305,
    "max_rho_after": 0.2274436008243468,
    "mean_identity_drift": 0.25118181308110554,
    "mean_trust_delta": 0.037333333333333336,
    "wound_active_count": 4
  },
  "dilemma_The Classic Trolley": {
    "turns": 6,
    "mean_epsilon": 0.852721426813066,
    "mean_rho_after": 0.22980905427879397,
    "wound_active_count": 2
  },
  "dilemma_The Footbridge": {
    "turns": 6,
    "mean_epsilon": 0.6187518628832901,
    "mean_rho_after": 0.19916832427050016,
    "wound_active_count": 2
  },
  "dilemma_The Transplant Surgeon": {
    "turns": 6,
    "mean_epsilon": 0.6118020231400182,
    "mean_rho_after": 0.13780545793865442,
    "wound_active_count": 1
  },
  "dilemma_The Triage Protocol": {
    "turns": 6,
    "mean_epsilon": 0.7561272476596012,
    "mean_rho_after": 0.12871342660845422,
    "wound_active_count": 1
  },
  "dilemma_The Final Question": {
    "turns": 6,
    "mean_epsilon": 0.6700179371407334,
    "mean_rho_after": 0.08723991863087634,
    "wound_active_count": 2
  }
}