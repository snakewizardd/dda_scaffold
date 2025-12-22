# 20251222 095058 – DDA Analysis & Visualization (Run)

This pack contains parsed turn-level metrics, aggregates, and figures for quick inspection and repo inclusion.

## Files

- `turns_summary.csv` – per-turn structured data.
- `turns_summary.json` – JSON array of turns.
- `aggregates.json` – overall, per-agent, and per-dilemma stats.
- `philosophers_duel_report.pdf` – tables and aggregates.
- `philosophers_duel_figures.pdf` – all figures assembled.
- `transcript.md` – original transcript for reference.
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
    "turn_count": 24,
    "agents": [
      "CRAFTSMAN",
      "HARMONIZER",
      "PROVOCATEUR",
      "VISIONARY"
    ],
    "dilemmas": [
      ""
    ],
    "mean_epsilon": 0.7530859687193047,
    "mean_rho_before": 0.18675426037239007,
    "mean_rho_after": 0.16808789232435326,
    "mean_delta_rho": -0.018666368048036838,
    "mean_identity_drift": 0.14405418885871768,
    "mean_trust_delta": NaN,
    "wound_active_count": 6
  },
  "band_compliance": {
    "total_with_band": 24,
    "compliant_count": 23,
    "compliance_rate": 0.9583333333333334
  },
  "agent_VISIONARY": {
    "turns": 8,
    "mean_epsilon": 0.7615316578149499,
    "mean_rho_before": 0.16423704309050047,
    "mean_rho_after": 0.14478444852194086,
    "mean_delta_rho": -0.019452594568559613,
    "min_rho_after": 0.024379243451523078,
    "max_rho_after": 0.23943521807653909,
    "mean_identity_drift": 0.1577406506985426,
    "mean_trust_delta": NaN,
    "wound_active_count": 3
  },
  "agent_CRAFTSMAN": {
    "turns": 8,
    "mean_epsilon": 0.699119085211727,
    "mean_rho_before": 0.1697868996186416,
    "mean_rho_after": 0.1486851986832914,
    "mean_delta_rho": -0.021101700935350222,
    "min_rho_after": 0.031186392517198215,
    "max_rho_after": 0.24366229362061842,
    "mean_identity_drift": 0.16740718577057123,
    "mean_trust_delta": NaN,
    "wound_active_count": 3
  },
  "agent_PROVOCATEUR": {
    "turns": 6,
    "mean_epsilon": 0.761158416668574,
    "mean_rho_before": 0.25049906216821044,
    "mean_rho_after": 0.23387974123021216,
    "mean_delta_rho": -0.01661932093799827,
    "min_rho_after": 0.15028407437201044,
    "max_rho_after": 0.2977447248411017,
    "mean_identity_drift": 0.12268344250818093,
    "mean_trust_delta": NaN,
    "wound_active_count": 0
  },
  "agent_HARMONIZER": {
    "turns": 2,
    "mean_epsilon": 0.9109534025192261,
    "mean_rho_before": 0.15345816712748117,
    "mean_rho_after": 0.14153689538067327,
    "mean_delta_rho": -0.01192127174680789,
    "min_rho_after": 0.1261574565063842,
    "max_rho_after": 0.15691633425496235,
    "mean_identity_drift": 0.060008592903614044,
    "mean_trust_delta": NaN,
    "wound_active_count": 0
  },
  "dilemma_": {
    "turns": 24,
    "mean_epsilon": 0.7530859687193047,
    "mean_rho_after": 0.16808789232435326,
    "wound_active_count": 6
  }
}

## Transcript path

`data\creative_collective\20251222_095058_outputs\transcript.md`
