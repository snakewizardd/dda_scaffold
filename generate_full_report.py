
#!/usr/bin/env python3
"""
Robust DDA analysis & visualization generator.
Reads data/<experiment>/transcript.md and session_log.json (any reasonable schema),
normalizes to list-of-records, and produces the full report pack.

CLI extras:
- --json-key <key>       pick a top-level key (e.g., 'results')
- --json-path <a.b.c>    traverse nested dictionaries (dot path)

Outputs under <out-root>/<experiment>_outputs:
- turns_summary.csv / turns_summary.json
- aggregates.json
- {experiment}_report.pdf
- {experiment}_figures.pdf
- README_{experiment}_analysis.md
- transcript.md (copied)
- Figures (PNGs):
    epsilon_rho_after_trajectories.png
    rho_before_vs_after.png
    identity_drift.png
    trust_delta.png
    trust_pairs.png
    trust_delta_per_round.png
    trust_effect_size.png
    wound_activation.png
    epsilon_vs_delta_rho_scatter.png
    wordcount_vs_band_ranges.png
    band_compliance_rates.png
    round_level_avgs.png
    wounds_per_dilemma.png
    identity_scorecard.png
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    Table, TableStyle, SimpleDocTemplate, Paragraph, Spacer
)
from reportlab.lib.styles import getSampleStyleSheet

FIGURE_LIST = [
    "epsilon_rho_after_trajectories.png",
    "rho_before_vs_after.png",
    "identity_drift.png",
    "trust_delta.png",
    "trust_pairs.png",
    "trust_delta_per_round.png",
    "trust_effect_size.png",
    "wound_activation.png",
    "epsilon_vs_delta_rho_scatter.png",
    "wordcount_vs_band_ranges.png",
    "band_compliance_rates.png",
    "round_level_avgs.png",
    "wounds_per_dilemma.png",
    "identity_scorecard.png",
    "recovery_half_life.png",
    "coalition_trust_facets.png",
]

# Shock colors for overlays
SHOCK_COLORS = {
    "coalition_vote": "#f2cc8f",
    "coalition_flip": "#e5989b",
    "role_swap": "#b5e48c",
    "context_drop": "#84a59d",
    "partial_context_drop": "#84a59d",
    "outside_scrutiny": "#9a8c98",
    "curator_audit": "#6a4c93",
    "final_merge": "#c9ada7",
}
BAND_RANGES = {
    "OPEN": (80, 150),
    "MEASURED": (60, 100),
    "GUARDED": (30, 70),
    "FORTIFIED": (15, 40),
    # SILENT excluded from compliance (handled separately)
}

def parse_args():
    ap = argparse.ArgumentParser(description="Build DDA analysis pack from data/<experiment>/transcript.md and session_log.json")
    ap.add_argument("--experiment", default="philosophers_duel", help="Experiment folder name under data/")
    ap.add_argument("--data-root", default="data", help="Root folder that contains the experiment folder")
    ap.add_argument("--out-root", default=".", help="Where to create the output folder")
    ap.add_argument("--json-key", default=None, help="Top-level key to extract a list of records (e.g., 'results')")
    ap.add_argument("--json-path", default=None, help="Dot path to nested list of records (e.g., 'collective.events')")
    return ap.parse_args()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_inputs(data_root: str, experiment: str) -> Dict[str, Any]:
    exp_dir = os.path.join(data_root, experiment)
    transcript_path = os.path.join(exp_dir, "transcript.md")
    session_path = os.path.join(exp_dir, "session_log.json")

    if not os.path.isfile(transcript_path):
        print(f"[ERROR] transcript.md not found at {transcript_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(session_path):
        print(f"[ERROR] session_log.json not found at {session_path}", file=sys.stderr)
        sys.exit(1)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()
    with open(session_path, "r", encoding="utf-8") as f:
        session_json = json.load(f)

    return {
        "transcript_path": transcript_path,
        "session_path": session_path,
        "transcript": transcript,
        "session_json": session_json,
    }

def _get_by_path(obj: Any, path: str) -> Any:
    """Traverse nested dicts by 'a.b.c' path."""
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Path segment '{part}' not found.")
        cur = cur[part]
    return cur

def normalize_session(session_json: Any, json_key: str = None, json_path: str = None) -> List[Dict[str, Any]]:
    """
    Accepts multiple shapes and returns a list[dict] of records:
      - Use --json-path if provided (dot path).
      - Else use --json-key if provided (top-level key).
      - Else auto-detect: list-of-dicts, or a dict with one of known keys, or dict-of-lists.
    """
    # 1) Explicit json_path
    if json_path:
        try:
            v = _get_by_path(session_json, json_path)
        except Exception as e:
            raise ValueError(f"--json-path '{json_path}' failed: {e}")
        return _coerce_to_records(v)

    # 2) Explicit json_key
    if json_key:
        if not isinstance(session_json, dict) or json_key not in session_json:
            raise ValueError(f"--json-key '{json_key}' not found in top-level JSON keys.")
        v = session_json[json_key]
        return _coerce_to_records(v)

    # 3) Auto-detect
    if isinstance(session_json, list):
        # Expect list of dicts
        return _coerce_to_records(session_json)

    if isinstance(session_json, dict):
        # Try known keys at top-level
        candidate_keys = [
            'events','records','log','turns','data','items','session','sessions',
            # additional based on your file
            'results','collective','trust_dynamics'
        ]
        for k in candidate_keys:
            v = session_json.get(k)
            if v is not None:
                try:
                    return _coerce_to_records(v)
                except ValueError:
                    # not a usable list-of-dicts; continue searching
                    pass

        # Dict-of-lists case: combine by index
        arrays = {k: v for k, v in session_json.items() if isinstance(v, list)}
        if arrays:
            lengths = {k: len(v) for k, v in arrays.items()}
            min_len = min(lengths.values())
            max_len = max(lengths.values())
            if min_len == 0:
                raise ValueError("Dict-of-lists contains an empty list; no records to build.")
            if min_len < max_len:
                print(f"[WARN] Dict-of-lists has uneven lengths {lengths}; truncating to min_len={min_len}.", file=sys.stderr)
            rows = []
            for i in range(min_len):
                row = {k: arrays[k][i] for k in arrays.keys()}
                rows.append(row)
            return rows

    raise ValueError("Could not normalize session_log.json into a list of records.")

def _coerce_to_records(v: Any) -> List[Dict[str, Any]]:
    """Coerce various shapes to list-of-dicts records or raise ValueError."""
    if isinstance(v, list):
        if not v:
            raise ValueError("List is empty.")
        if not isinstance(v[0], dict):
            raise ValueError("List does not contain dict records.")
        return v
    if isinstance(v, dict):
        # dict-of-lists → combine rows
        arrays = {k: val for k, val in v.items() if isinstance(val, list)}
        if arrays:
            lengths = {k: len(val) for k, val in arrays.items()}
            min_len = min(lengths.values())
            max_len = max(lengths.values())
            if min_len == 0:
                raise ValueError("Dict-of-lists contains an empty list; no records to build.")
            if min_len < max_len:
                print(f"[WARN] Nested dict-of-lists uneven lengths {lengths}; truncating to min_len={min_len}.", file=sys.stderr)
            rows = [{k: arrays[k][i] for k in arrays} for i in range(min_len)]
            return rows
    raise ValueError("Value is neither a list-of-dicts nor a dict-of-lists.")

def build_dataframe(session_list: List[Dict[str, Any]]) -> pd.DataFrame:
    cols_order = [
        "turn","dilemma","phase","speaker","word_count","band","text","epsilon",
        "rho_before","rho_after","delta_rho","wound_resonance","wound_active",
        "identity_drift","trust_delta"
    ]
    df = pd.DataFrame(session_list)

    # Fill expected columns if missing
    for col in cols_order:
        if col not in df.columns:
            if col in ("text", "band", "speaker", "phase", "dilemma"):
                df[col] = ""
            elif col == "wound_active":
                df[col] = False
            elif col == "word_count":
                df[col] = df["text"].apply(lambda t: len([w for w in str(t).split() if w.strip()]))
            else:
                df[col] = float("nan")

    # Handle trust_others dict → compute trust_delta as mean change from previous turn
    if "trust_others" in df.columns and df["trust_others"].notna().any():
        trust_deltas = []
        prev_trust_by_speaker = {}
        
        for idx, row in df.iterrows():
            speaker = row.get("speaker", "")
            trust_others = row.get("trust_others")
            
            if isinstance(trust_others, dict) and trust_others:
                # Get previous trust state for this speaker
                prev_trust = prev_trust_by_speaker.get(speaker, {})
                
                # Compute delta as sum of changes across all trust targets
                delta = 0.0
                for target, val in trust_others.items():
                    prev_val = prev_trust.get(target, 0.5)  # default 0.5
                    delta += (val - prev_val)
                
                trust_deltas.append(delta)
                prev_trust_by_speaker[speaker] = trust_others.copy()
            else:
                trust_deltas.append(0.0)
        
        df["trust_delta"] = trust_deltas

    # Coerce numeric columns that might be strings
    numeric_cols = ["epsilon","rho_before","rho_after","delta_rho","wound_resonance","identity_drift","trust_delta","word_count","turn"]
    for nc in numeric_cols:
        df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # Sort by turn if present
    if "turn" in df.columns and df["turn"].notna().any():
        df = df.sort_values("turn")

    # Handle round_name as dilemma/phase fallback
    if "round_name" in df.columns:
        if df["dilemma"].isna().all() or (df["dilemma"] == "").all():
            df["dilemma"] = df["round_name"]
        if df["phase"].isna().all() or (df["phase"] == "").all():
            df["phase"] = df["round_name"]

    # Ensure cols_order columns exist
    for col in cols_order:
        if col not in df.columns:
            df[col] = float("nan") if col not in ("text", "band", "speaker", "phase", "dilemma") else ""

    df = df[cols_order].reset_index(drop=True)

    # Band compliance - include SILENT as neutral (not counted)
    def band_compliant(row):
        band = row["band"]
        if band == "SILENT":
            return None  # Exclude from compliance stats
        rng = BAND_RANGES.get(band)
        if not rng:
            return None
        wc = row["word_count"] if pd.notnull(row["word_count"]) else 0
        return int(rng[0] <= wc <= rng[1])

    df["band_compliant"] = df.apply(band_compliant, axis=1)
    return df

def compute_aggregates(df: pd.DataFrame, session_list: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    session_list = session_list or []
    agg = {}
    agg["overall"] = {
        "turn_count": int(df.shape[0]),
        "agents": sorted(df["speaker"].unique().tolist()),
        "dilemmas": sorted(df["dilemma"].unique().tolist()),
        "mean_epsilon": float(df["epsilon"].mean()),
        "mean_rho_before": float(df["rho_before"].mean()),
        "mean_rho_after": float(df["rho_after"].mean()),
        "mean_delta_rho": float(df["delta_rho"].mean()),
        "mean_identity_drift": float(df["identity_drift"].mean()),
        "mean_trust_delta": float(df["trust_delta"].mean()),
        "wound_active_count": int(df["wound_active"].sum()),
    }
    
    # Estimate calibrated ε₀ and s from data (median and IQR of epsilon)
    eps_values = df["epsilon"].dropna()
    if len(eps_values) >= 6:
        eps_median = float(eps_values.median())
        eps_iqr = float(eps_values.quantile(0.75) - eps_values.quantile(0.25)) or 0.2
        agg["calibration"] = {
            "epsilon_0": round(eps_median, 3),
            "s": round(max(0.10, min(0.30, eps_iqr)), 3),
            "note": "Estimated from run data (median ε, IQR-based s)"
        }
    else:
        agg["calibration"] = {
            "epsilon_0": 0.75,
            "s": 0.20,
            "note": "Default values (insufficient data for calibration)"
        }
    
    # Recovery half-life summary
    hl_values = [r.get("recovery_half_life") for r in session_list if r.get("recovery_half_life") is not None and r.get("recovery_half_life") > 0]
    if hl_values:
        agg["recovery_half_life"] = {
            "median": float(np.median(hl_values)),
            "p90": float(np.percentile(hl_values, 90)) if len(hl_values) >= 10 else None,
            "count": len(hl_values),
        }
    
    # Trust effect size (using stored baseline if available)
    if session_list and "delta_rho_baseline" in session_list[0]:
        trust_effects = [r.get("delta_rho", 0) - r.get("delta_rho_baseline", 0) 
                        for r in session_list if not r.get("is_silent")]
        if trust_effects:
            agg["trust_effect_size"] = {
                "mean": float(np.mean(trust_effects)),
                "std": float(np.std(trust_effects)),
                "source": "stored_baseline",
            }
    
    # Band compliance - exclude SILENT
    band_df = df[(df["band"].notna()) & (df["band"] != "SILENT")]
    silent_count = int((df["band"] == "SILENT").sum())
    compliant = int(band_df["band_compliant"].sum()) if not band_df.empty else 0
    total_with_band = int(band_df.shape[0])
    agg["band_compliance"] = {
        "total_with_band": total_with_band,
        "compliant_count": compliant,
        "compliance_rate": float(compliant / total_with_band) if total_with_band else 0.0,
        "silent_count": silent_count,
    }
    # Per agent
    for ag in df["speaker"].unique():
        dfa = df[df["speaker"]==ag]
        # Get first and last rho for recovery check
        first_rho = float(dfa["rho_before"].iloc[0]) if not dfa.empty else 0.0
        last_rho = float(dfa["rho_after"].iloc[-1]) if not dfa.empty else 0.0
        final_drift = float(dfa["identity_drift"].iloc[-1]) if not dfa.empty else 0.0
        
        # Identity persistence metrics
        maintained = final_drift < 0.40
        recovered = last_rho <= (first_rho + 0.05)
        
        agg[f"agent_{ag}"] = {
            "turns": int(dfa.shape[0]),
            "mean_epsilon": float(dfa["epsilon"].mean()),
            "mean_rho_before": float(dfa["rho_before"].mean()),
            "mean_rho_after": float(dfa["rho_after"].mean()),
            "mean_delta_rho": float(dfa["delta_rho"].mean()),
            "min_rho_after": float(dfa["rho_after"].min()),
            "max_rho_after": float(dfa["rho_after"].max()),
            "mean_identity_drift": float(dfa["identity_drift"].mean()),
            "mean_trust_delta": float(dfa["trust_delta"].mean()),
            "wound_active_count": int(dfa["wound_active"].sum()),
            # Identity persistence scorecard
            "rho_0": first_rho,
            "rho_final": last_rho,
            "final_drift": final_drift,
            "identity_maintained": maintained,
            "recovered": recovered,
        }
    
    # Collective identity persistence scorecard
    agents = df["speaker"].unique().tolist()
    all_maintained = all(agg[f"agent_{ag}"]["identity_maintained"] for ag in agents)
    all_recovered = all(agg[f"agent_{ag}"]["recovered"] for ag in agents)
    final_drifts = [agg[f"agent_{ag}"]["final_drift"] for ag in agents]
    avg_drift = float(sum(final_drifts) / len(final_drifts)) if final_drifts else 0.0
    drift_variance = float(sum((d - avg_drift)**2 for d in final_drifts) / len(final_drifts)) if final_drifts else 0.0
    
    harmony_status = "HARMONY" if avg_drift < 0.30 and drift_variance < 0.01 else \
                     "PARTIAL" if avg_drift < 0.35 else "EROSION"
    
    # Trust modulation effect size: mean absolute trust_delta (excluding SILENT)
    df_non_silent = df[df["band"] != "SILENT"]
    trust_effect_size = float(df_non_silent["trust_delta"].abs().mean()) if not df_non_silent.empty else 0.0
    
    # Wound stats
    total_wounds = int(df["wound_active"].sum())
    
    agg["identity_scorecard"] = {
        "all_maintained": all_maintained,
        "all_recovered": all_recovered,
        "avg_final_drift": round(avg_drift, 4),
        "drift_variance": round(drift_variance, 6),
        "harmony_status": harmony_status,
        "agents_maintained": sum(1 for ag in agents if agg[f"agent_{ag}"]["identity_maintained"]),
        "agents_recovered": sum(1 for ag in agents if agg[f"agent_{ag}"]["recovered"]),
        "total_agents": len(agents),
        "trust_modulation_effect_size": round(trust_effect_size, 4),
        "wound_activations": total_wounds,
    }
    
    # Per dilemma
    for dname in df["dilemma"].unique():
        dfd = df[df["dilemma"]==dname]
        agg[f"dilemma_{dname}"] = {
            "turns": int(dfd.shape[0]),
            "mean_epsilon": float(dfd["epsilon"].mean()),
            "mean_rho_after": float(dfd["rho_after"].mean()),
            "wound_active_count": int(dfd["wound_active"].sum()),
        }
    return agg

def render_figures(df: pd.DataFrame, out_dir: str, session_list: List[Dict[str, Any]] = None):
    """Render all visualization figures."""
    plt.style.use('seaborn-v0_8')
    session_list = session_list or []
    
    # Build shock overlay data from session_list
    shock_turns = {}
    for record in session_list:
        shock = record.get("shock_active")
        if shock:
            turn = record.get("turn", 0)
            if shock not in shock_turns:
                shock_turns[shock] = []
            shock_turns[shock].append(turn)

    # (1) epsilon & rho_after trajectories with shock overlays
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ag, dfa in df.groupby("speaker"):
        axes[0].plot(dfa["turn"], dfa["epsilon"], marker="o", label=ag)
        axes[1].plot(dfa["turn"], dfa["rho_after"], marker="o", label=ag)
    
    # Add shock overlays
    for shock_type, turns in shock_turns.items():
        color = SHOCK_COLORS.get(shock_type, "#cccccc")
        for t in set(turns):
            axes[0].axvspan(t-0.5, t+0.5, color=color, alpha=0.15, label=shock_type if t == min(turns) else "")
            axes[1].axvspan(t-0.5, t+0.5, color=color, alpha=0.15)
    
    axes[0].set_title("Prediction Error (ε) by Turn")
    axes[0].set_ylabel("ε")
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_title("Rigidity (ρ_after) by Turn")
    axes[1].set_ylabel("ρ_after")
    axes[1].set_xlabel("Turn")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epsilon_rho_after_trajectories.png"))
    plt.close(fig)

    # (2) rho_before vs rho_after
    fig, ax = plt.subplots(figsize=(10,5))
    for ag, dfa in df.groupby("speaker"):
        ax.plot(dfa["turn"], dfa["rho_before"], marker="o", linestyle="--", label=f"{ag} – ρ_before")
        ax.plot(dfa["turn"], dfa["rho_after"], marker="s", label=f"{ag} – ρ_after")
    ax.set_title("Rigidity Before vs After by Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("ρ")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rho_before_vs_after.png"))
    plt.close(fig)

    # (3) Identity drift
    fig, ax = plt.subplots(figsize=(10,5))
    for ag, dfa in df.groupby("speaker"):
        ax.plot(dfa["turn"], dfa["identity_drift"], marker="o", label=ag)
    ax.set_title("Identity Drift by Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Identity drift")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "identity_drift.png"))
    plt.close(fig)

    # (4) Trust delta - improved to show actual trust changes
    fig, ax = plt.subplots(figsize=(10,5))
    has_trust_data = df["trust_delta"].notna().any() and (df["trust_delta"] != 0).any()
    if has_trust_data:
        # Plot as lines per speaker for better visibility
        for ag, dfa in df.groupby("speaker"):
            ax.plot(dfa["turn"], dfa["trust_delta"], marker="o", label=ag, alpha=0.8)
            ax.fill_between(dfa["turn"], 0, dfa["trust_delta"], alpha=0.2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title("Trust Delta by Turn (cumulative change per turn)")
    else:
        # Fallback: show message if no trust data
        ax.text(0.5, 0.5, "No trust delta data available\n(trust_others dict not found or unchanged)", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Trust Delta by Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Trust delta (sum of changes)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trust_delta.png"))
    plt.close(fig)

    # (4b) Per-pair trust trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    has_trust_others = "trust_others" in session_list[0] if session_list else False
    if has_trust_others:
        # Extract per-pair trust over time
        trust_pairs = {}  # (speaker, target) -> [(turn, value), ...]
        for record in session_list:
            turn = record.get("turn", 0)
            speaker = record.get("speaker", "")
            trust_others = record.get("trust_others", {})
            if isinstance(trust_others, dict):
                for target, val in trust_others.items():
                    pair = f"{speaker}→{target}"
                    if pair not in trust_pairs:
                        trust_pairs[pair] = []
                    trust_pairs[pair].append((turn, val))
        
        # Plot each pair
        for pair, data in sorted(trust_pairs.items()):
            turns, vals = zip(*data) if data else ([], [])
            ax.plot(turns, vals, marker=".", label=pair, alpha=0.7)
        
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='baseline (0.5)')
        ax.set_title("Per-Pair Trust Trajectories")
        ax.set_xlabel("Turn")
        ax.set_ylabel("Trust value")
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No per-pair trust data available\n(trust_others dict not found)", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Per-Pair Trust Trajectories")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trust_pairs.png"))
    plt.close(fig)

    # (4c) Per-pair trust delta per round (stacked bars)
    fig, ax = plt.subplots(figsize=(14, 6))
    has_trust_others = "trust_others" in session_list[0] if session_list else False
    if has_trust_others and "round_name" in session_list[0]:
        # Compute per-pair trust delta per round
        rounds_order = []
        pair_deltas = {}  # pair -> {round: delta}
        prev_trust_by_speaker = {}
        
        for record in session_list:
            round_name = record.get("round_name", "")
            speaker = record.get("speaker", "")
            trust_others = record.get("trust_others", {})
            
            if round_name and round_name not in rounds_order:
                rounds_order.append(round_name)
            
            if isinstance(trust_others, dict):
                prev_trust = prev_trust_by_speaker.get(speaker, {})
                for target, val in trust_others.items():
                    pair = f"{speaker}→{target}"
                    prev_val = prev_trust.get(target, 0.5)
                    delta = val - prev_val
                    
                    if pair not in pair_deltas:
                        pair_deltas[pair] = {}
                    if round_name not in pair_deltas[pair]:
                        pair_deltas[pair][round_name] = 0.0
                    pair_deltas[pair][round_name] += delta
                
                prev_trust_by_speaker[speaker] = trust_others.copy()
        
        # Plot stacked bars per round
        if rounds_order and pair_deltas:
            x = range(len(rounds_order))
            width = 0.8 / len(pair_deltas)
            
            for i, (pair, round_deltas) in enumerate(sorted(pair_deltas.items())):
                deltas = [round_deltas.get(r, 0.0) for r in rounds_order]
                offset = (i - len(pair_deltas)/2 + 0.5) * width
                ax.bar([xi + offset for xi in x], deltas, width, label=pair, alpha=0.8)
            
            ax.set_xticks(list(x))
            ax.set_xticklabels(rounds_order, rotation=45, ha='right')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Round")
            ax.set_ylabel("Trust Delta")
            ax.set_title("Per-Pair Trust Delta by Round")
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        else:
            ax.text(0.5, 0.5, "Insufficient data for per-round trust delta", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    else:
        ax.text(0.5, 0.5, "No per-pair trust data or round_name available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Per-Pair Trust Delta by Round")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trust_delta_per_round.png"))
    plt.close(fig)

    # (5) Wound activation
    fig, ax = plt.subplots(figsize=(10,3))
    y = df["wound_active"].astype(int)
    ax.scatter(df["turn"], y, c=y, cmap="coolwarm", s=80)
    ax.set_title("Wound Activation by Turn (1=Active, 0=Inactive)")
    ax.set_xlabel("Turn")
    ax.set_yticks([0,1]); ax.set_yticklabels(["Inactive","Active"])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wound_activation.png"))
    plt.close(fig)

    # (6) ε vs Δρ scatter
    fig, ax = plt.subplots(figsize=(8,5))
    for ag, dfa in df.groupby("speaker"):
        ax.scatter(dfa["epsilon"], dfa["delta_rho"], label=ag)
    ax.set_title("Surprise (ε) vs Δρ (rigidity change)")
    ax.set_xlabel("ε")
    ax.set_ylabel("Δρ")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "epsilon_vs_delta_rho_scatter.png"))
    plt.close(fig)

    # (7) Word count vs band ranges
    fig, ax = plt.subplots(figsize=(10,5))
    colors_map = {"OPEN": "#4C78A8", "MEASURED": "#F58518"}
    for band in df["band"].unique():
        sub = df[df["band"] == band]
        ax.scatter(sub["turn"], sub["word_count"], label=f"{band} word count", color=colors_map.get(band, "#888"))
        rng = BAND_RANGES.get(band)
        if rng and not sub.empty:
            ax.fill_between(sub["turn"], rng[0], rng[1], color=colors_map.get(band, "#888"),
                            alpha=0.15, step="mid", label=f"{band} expected range")
    ax.set_title("Word Count vs Expected Band Ranges")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Word Count")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wordcount_vs_band_ranges.png"))
    plt.close(fig)

    # (8) Band compliance rate - exclude SILENT, show it separately
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Filter out SILENT for compliance calculation
    df_compliance = df[df["band"] != "SILENT"].copy()
    silent_count = (df["band"] == "SILENT").sum()
    
    if not df_compliance.empty and df_compliance["band_compliant"].notna().any():
        comp = df_compliance.groupby("band")["band_compliant"].mean().rename("compliance_rate")
        colors_list = ["#4C78A8", "#F58518", "#54A24B", "#E45756"][:len(comp)]
        comp.plot(kind="bar", ax=ax, color=colors_list)
        ax.set_ylim(0, 1)
        
        # Add SILENT annotation if any
        if silent_count > 0:
            ax.annotate(f"SILENT turns: {silent_count} (excluded)", 
                       xy=(0.98, 0.02), xycoords='axes fraction',
                       ha='right', va='bottom', fontsize=9, color='gray')
    else:
        ax.text(0.5, 0.5, "No compliance data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    
    ax.set_title("Band Compliance Rate by Band Type")
    ax.set_ylabel("Compliance Rate")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "band_compliance_rates.png"))
    plt.close(fig)

    # (9) Round-level averages (renamed from phase)
    fig, ax = plt.subplots(figsize=(9,5))
    # Use dilemma (which may be round_name) - always label as "Round"
    group_col = "dilemma"
    round_avg = df.groupby(group_col)[["epsilon","rho_after","identity_drift"]].mean()
    round_avg.plot(kind="bar", ax=ax)
    ax.set_title("Round-level Averages (ε, ρ_after, identity_drift)")
    ax.set_ylabel("Value")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "round_level_avgs.png"))
    plt.close(fig)

    # (10) Dilemma-level wound activations
    fig, ax = plt.subplots(figsize=(9,5))
    dilemma_wounds = df.groupby("dilemma")["wound_active"].sum()
    dilemma_wounds.plot(kind="bar", ax=ax, color="#54A24B")
    ax.set_title("Wound Activations per Round")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wounds_per_dilemma.png"))
    plt.close(fig)

    # (11) Trust Effect Size - use stored delta_rho_baseline if available
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Check if delta_rho_baseline is in session data
    has_baseline = "delta_rho_baseline" in session_list[0] if session_list else False
    
    if has_baseline:
        # Use stored baseline (consistent computation)
        turns = []
        trust_effects = []
        for record in session_list:
            if record.get("is_silent"):
                continue
            drho = record.get("delta_rho", 0)
            drho_base = record.get("delta_rho_baseline", drho)
            turns.append(record.get("turn", 0))
            trust_effects.append(drho - drho_base)
        
        if turns:
            ax.bar(turns, trust_effects, color='#54A24B', alpha=0.7, label='Trust effect (Δρ_actual - Δρ_baseline)')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Δρ shift from trust modulation")
            ax.set_title("Trust Modulation Effect Size per Turn (from stored baseline)")
            ax.legend()
            
            mean_effect = sum(trust_effects) / len(trust_effects)
            ax.annotate(f"Mean effect: {mean_effect:+.4f}", 
                       xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No non-SILENT turns with baseline data", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    else:
        # Fallback: recompute baseline (legacy behavior)
        import math
        def sigmoid(z):
            if z >= 0:
                return 1.0 / (1.0 + math.exp(-z))
            else:
                ez = math.exp(z)
                return ez / (1.0 + ez)
        
        alpha = 0.12
        eps_0 = 0.75
        s_val = 0.20
        drift_penalty = 0.10
        drift_soft_floor = 0.20
        
        baseline_drhos = []
        actual_drhos = []
        trust_effects = []
        turns = []
        
        for idx, row in df.iterrows():
            eps = row["epsilon"]
            actual_drho = row["delta_rho"]
            fair_eng = row.get("fair_engagement", True)
            drift = row.get("identity_drift", 0.0)
            
            if pd.isna(eps) or pd.isna(actual_drho):
                continue
            
            z = (eps - eps_0) / s_val
            sig = sigmoid(z)
            baseline = alpha * (sig - 0.5)
            
            if fair_eng:
                baseline *= 0.85
            else:
                baseline *= 1.10
            
            if drift > drift_soft_floor and baseline > 0:
                penalty = drift_penalty * (drift - drift_soft_floor)
                penalty = min(penalty, baseline)
                baseline -= penalty
            
            baseline_drhos.append(baseline)
            actual_drhos.append(actual_drho)
            trust_effects.append(actual_drho - baseline)
            turns.append(row["turn"])
        
        if turns:
            ax.bar(turns, trust_effects, color='#54A24B', alpha=0.7, label='Trust effect (Δρ_actual - Δρ_baseline)')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Δρ shift from trust modulation")
            ax.set_title("Trust Modulation Effect Size per Turn (recomputed baseline)")
            ax.legend()
            
            mean_effect = sum(trust_effects) / len(trust_effects)
            ax.annotate(f"Mean effect: {mean_effect:+.4f}", 
                       xy=(0.98, 0.98), xycoords='axes fraction',
                       ha='right', va='top', fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "Insufficient data for trust effect calculation", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
    
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trust_effect_size.png"))
    plt.close(fig)

    # (12) Identity Persistence Scorecard with value labels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Per-agent final drift vs threshold
    agents = df["speaker"].unique()
    final_drifts = []
    for ag in agents:
        dfa = df[df["speaker"] == ag]
        if not dfa.empty:
            final_drifts.append(float(dfa["identity_drift"].iloc[-1]))
        else:
            final_drifts.append(0.0)
    
    colors_list = ['#4C78A8' if d < 0.40 else '#E45756' for d in final_drifts]
    bars = axes[0].barh(list(agents), final_drifts, color=colors_list)
    axes[0].axvline(x=0.40, color='red', linestyle='--', label='Threshold (0.40)')
    
    # Add value labels
    for i, (bar, v) in enumerate(zip(bars, final_drifts)):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va='center', fontsize=9)
    
    axes[0].set_xlabel("Final Identity Drift")
    axes[0].set_title("Identity Maintenance (drift < 0.40)")
    axes[0].legend()
    axes[0].grid(True, axis="x", alpha=0.3)
    
    # Right: Recovery status (ρ_final vs ρ₀ + 0.05)
    rho_0s = []
    rho_finals = []
    for ag in agents:
        dfa = df[df["speaker"] == ag]
        if not dfa.empty:
            rho_0s.append(float(dfa["rho_before"].iloc[0]))
            rho_finals.append(float(dfa["rho_after"].iloc[-1]))
        else:
            rho_0s.append(0.0)
            rho_finals.append(0.0)
    
    x = range(len(agents))
    width = 0.35
    axes[1].bar([i - width/2 for i in x], rho_0s, width, label='ρ₀ (initial)', color='#4C78A8')
    axes[1].bar([i + width/2 for i in x], rho_finals, width, label='ρ_final', color='#F58518')
    
    # Add threshold lines (ρ₀ + 0.05) for each agent
    for i, r0 in enumerate(rho_0s):
        axes[1].hlines(y=r0 + 0.05, xmin=i-0.4, xmax=i+0.4, colors='red', linestyles='--', alpha=0.5)
        # Add ρ₀ dashed line
        axes[1].hlines(y=r0, xmin=i-0.4, xmax=i+0.4, colors='#999999', linestyles=':', alpha=0.6)
    
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(list(agents), rotation=45, ha='right')
    axes[1].set_ylabel("Rigidity (ρ)")
    axes[1].set_title("Recovery Status (ρ_final ≤ ρ₀ + 0.05)")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    
    fig.suptitle("Identity Persistence Scorecard", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "identity_scorecard.png"))
    plt.close(fig)
    
    # (13) Recovery Half-Life Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    hl_values = []
    for record in session_list:
        hl = record.get("recovery_half_life")
        if hl is not None and hl > 0:
            hl_values.append(hl)
    
    if hl_values:
        ax.hist(hl_values, bins=max(5, len(set(hl_values))), color='#54A24B', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.median(hl_values), color='red', linestyle='--', label=f'Median: {np.median(hl_values):.0f}')
        if len(hl_values) >= 10:
            ax.axvline(x=np.percentile(hl_values, 90), color='orange', linestyle='--', label=f'90th pct: {np.percentile(hl_values, 90):.0f}')
        ax.set_xlabel("Recovery Half-Life (turns)")
        ax.set_ylabel("Count")
        ax.set_title("Recovery Half-Life Distribution")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No recovery half-life data available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title("Recovery Half-Life Distribution")
    
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "recovery_half_life.png"))
    plt.close(fig)
    
    # (14) Coalition Trust Facets (intra vs inter)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    has_coalition = "coalition_id" in session_list[0] if session_list else False
    has_trust_gain = "trust_gain_intra" in session_list[0] if session_list else False
    
    if has_coalition and has_trust_gain:
        # Intra-coalition trust gains
        intra_gains = [(r.get("turn", 0), r.get("trust_gain_intra", 0)) for r in session_list if r.get("trust_gain_intra", 0) != 0]
        inter_gains = [(r.get("turn", 0), r.get("trust_gain_inter", 0)) for r in session_list if r.get("trust_gain_inter", 0) != 0]
        
        if intra_gains:
            turns, gains = zip(*intra_gains)
            axes[0].bar(turns, gains, color='#4C78A8', alpha=0.7)
            axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            mean_intra = np.mean(gains)
            axes[0].annotate(f"Mean: {mean_intra:+.4f}", xy=(0.98, 0.98), xycoords='axes fraction',
                           ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[0].set_xlabel("Turn")
        axes[0].set_ylabel("Trust Gain")
        axes[0].set_title("Intra-Coalition Trust Gains")
        axes[0].grid(True, axis="y", alpha=0.3)
        
        if inter_gains:
            turns, gains = zip(*inter_gains)
            axes[1].bar(turns, gains, color='#F58518', alpha=0.7)
            axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            mean_inter = np.mean(gains)
            axes[1].annotate(f"Mean: {mean_inter:+.4f}", xy=(0.98, 0.98), xycoords='axes fraction',
                           ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_xlabel("Turn")
        axes[1].set_ylabel("Trust Gain")
        axes[1].set_title("Inter-Coalition Trust Gains")
        axes[1].grid(True, axis="y", alpha=0.3)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "No coalition trust data available", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12, color='gray')
        axes[0].set_title("Intra-Coalition Trust Gains")
        axes[1].set_title("Inter-Coalition Trust Gains")
    
    fig.suptitle("Coalition Trust Facets", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coalition_trust_facets.png"))
    plt.close(fig)

def build_tables_pdf(aggregates: Dict[str, Any], out_dir: str, experiment: str) -> str:
    pdf_path = os.path.join(out_dir, f"{experiment}_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = f"{experiment.replace('_',' ').title()} – DDA Visualization Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph("Transcript analysis, aggregates, and visual diagnostics", styles["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    overall = aggregates["overall"]
    calibration = aggregates.get("calibration", {})
    band_compliance = aggregates.get("band_compliance", {})
    
    summary_data = [
        ["Metric","Value"],
        ["Turns", str(overall["turn_count"])],
        ["Agents", ", ".join(overall["agents"])],
        ["Dilemmas/Rounds", ", ".join(overall["dilemmas"][:5]) + ("..." if len(overall["dilemmas"]) > 5 else "")],
        ["Mean ε", f"{overall['mean_epsilon']:.3f}"],
        ["Mean ρ_before", f"{overall['mean_rho_before']:.3f}"],
        ["Mean ρ_after", f"{overall['mean_rho_after']:.3f}"],
        ["Mean Δρ", f"{overall['mean_delta_rho']:.3f}"],
        ["Mean identity drift", f"{overall['mean_identity_drift']:.3f}"],
        ["Mean trust delta", f"{overall['mean_trust_delta']:.3f}"],
        ["Wound activations", str(overall["wound_active_count"])],
        ["Calibrated ε₀", f"{calibration.get('epsilon_0', 'N/A')}"],
        ["Calibrated s", f"{calibration.get('s', 'N/A')}"],
        ["Band compliance", f"{band_compliance.get('compliance_rate', 0):.1%} ({band_compliance.get('compliant_count', 0)}/{band_compliance.get('total_with_band', 0)})"],
        ["SILENT turns", str(band_compliance.get('silent_count', 0))],
    ]
    summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("GRID",(0,0),(-1,-1),0.5,colors.grey),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 0.25*inch))

    # Agent tables
    for ag in overall["agents"]:
        a = aggregates[f"agent_{ag}"]
        # Identity persistence status
        maintained_str = "✓ Yes" if a.get("identity_maintained", False) else "✗ No"
        recovered_str = "✓ Yes" if a.get("recovered", False) else "○ No"
        
        agent_data = [
            ["Agent", ag],
            ["Turns", str(a["turns"])],
            ["Mean ε", f"{a['mean_epsilon']:.3f}"],
            ["Mean ρ_before", f"{a['mean_rho_before']:.3f}"],
            ["Mean ρ_after", f"{a['mean_rho_after']:.3f}"],
            ["Mean Δρ", f"{a['mean_delta_rho']:.3f}"],
            ["ρ_after range", f"{a['min_rho_after']:.3f} – {a['max_rho_after']:.3f}"],
            ["ρ₀ → ρ_final", f"{a.get('rho_0', 0):.3f} → {a.get('rho_final', 0):.3f}"],
            ["Mean identity drift", f"{a['mean_identity_drift']:.3f}"],
            ["Final drift", f"{a.get('final_drift', 0):.4f}"],
            ["Identity maintained", maintained_str],
            ["Recovered (ρ ≤ ρ₀+0.05)", recovered_str],
            ["Mean trust delta", f"{a['mean_trust_delta']:.3f}"],
            ["Wound activations", str(a["wound_active_count"])],
        ]
        t = Table(agent_data, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.25*inch))
    
    # Identity Persistence Scorecard (one-page summary)
    scorecard = aggregates.get("identity_scorecard", {})
    if scorecard:
        story.append(Paragraph("Identity Persistence Scorecard", styles["Heading2"]))
        harmony_color = colors.green if scorecard.get("harmony_status") == "HARMONY" else \
                       colors.orange if scorecard.get("harmony_status") == "PARTIAL" else colors.red
        
        scorecard_data = [
            ["Metric", "Value"],
            ["Harmony Status", scorecard.get("harmony_status", "N/A")],
            ["Agents Maintained Identity", f"{scorecard.get('agents_maintained', 0)}/{scorecard.get('total_agents', 0)}"],
            ["Agents Recovered", f"{scorecard.get('agents_recovered', 0)}/{scorecard.get('total_agents', 0)}"],
            ["Avg Final Drift", f"{scorecard.get('avg_final_drift', 0):.4f}"],
            ["Drift Variance (harmony)", f"{scorecard.get('drift_variance', 0):.6f}"],
            ["Trust Modulation Effect Size", f"{scorecard.get('trust_modulation_effect_size', 0):.4f}"],
            ["Wound Activations", str(scorecard.get("wound_activations", 0))],
            ["Band Compliance", f"{band_compliance.get('compliance_rate', 0):.1%}"],
        ]
        sc_table = Table(scorecard_data, colWidths=[2.5*inch, 3.5*inch])
        sc_table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("BACKGROUND",(1,1),(1,1), harmony_color),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ]))
        story.append(sc_table)
        story.append(Spacer(1, 0.25*inch))

    # Dilemma tables
    for dname in overall["dilemmas"]:
        d = aggregates[f"dilemma_{dname}"]
        d_data = [
            ["Dilemma", dname],
            ["Turns", str(d["turns"])],
            ["Mean ε", f"{d['mean_epsilon']:.3f}"],
            ["Mean ρ_after", f"{d['mean_rho_after']:.3f}"],
            ["Wound activations", str(d["wound_active_count"])],
        ]
        t = Table(d_data, colWidths=[2.5*inch, 3.5*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.25*inch))

    doc.build(story)
    return pdf_path

def build_figures_pdf(out_dir: str, experiment: str) -> str:
    fig_pdf_path = os.path.join(out_dir, f"{experiment}_figures.pdf")
    c = canvas.Canvas(fig_pdf_path, pagesize=letter)
    width, height = letter
    for fn in FIGURE_LIST:
        fig_path = os.path.join(out_dir, fn)
        if not os.path.isfile(fig_path):
            continue
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, height - 72, os.path.basename(fig_path))
        c.drawImage(fig_path, 72, 72, width=letter[0]-144, height=letter[1]-180,
                    preserveAspectRatio=True, anchor="c")
        c.showPage()
    c.save()
    return fig_pdf_path

def copy_transcript(src_path: str, out_dir: str) -> str:
    dst_path = os.path.join(out_dir, "transcript.md")
    with open(src_path, "r", encoding="utf-8") as fsrc, open(dst_path, "w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())
    return dst_path

def write_readme(out_dir: str, aggregates: Dict[str, Any], experiment: str, transcript_copied_path: str):
    md_path = os.path.join(out_dir, f"README_{experiment}_analysis.md")
    calibration = aggregates.get("calibration", {})
    scorecard = aggregates.get("identity_scorecard", {})
    band_compliance = aggregates.get("band_compliance", {})
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {experiment.replace('_',' ').title()} – DDA Analysis & Visualization (Run)\n\n")
        f.write("This pack contains parsed turn-level metrics, aggregates, and figures for quick inspection and repo inclusion.\n\n")
        
        # Calibration block at top
        f.write("## Calibration\n\n")
        f.write(f"- **ε₀ (epsilon_0):** {calibration.get('epsilon_0', 'N/A')}\n")
        f.write(f"- **s:** {calibration.get('s', 'N/A')}\n")
        f.write(f"- **Note:** {calibration.get('note', 'N/A')}\n\n")
        
        # Identity Persistence Scorecard
        if scorecard:
            f.write("## Identity Persistence Scorecard\n\n")
            f.write(f"- **Harmony Status:** {scorecard.get('harmony_status', 'N/A')}\n")
            f.write(f"- **Agents Maintained Identity:** {scorecard.get('agents_maintained', 0)}/{scorecard.get('total_agents', 0)}\n")
            f.write(f"- **Agents Recovered:** {scorecard.get('agents_recovered', 0)}/{scorecard.get('total_agents', 0)}\n")
            f.write(f"- **Avg Final Drift:** {scorecard.get('avg_final_drift', 0):.4f}\n")
            f.write(f"- **Drift Variance (harmony):** {scorecard.get('drift_variance', 0):.6f}\n")
            f.write(f"- **Band Compliance:** {band_compliance.get('compliance_rate', 0):.1%} ({band_compliance.get('compliant_count', 0)}/{band_compliance.get('total_with_band', 0)})\n")
            f.write(f"- **SILENT turns:** {band_compliance.get('silent_count', 0)} (excluded from compliance)\n\n")
        
        f.write("## Files\n\n")
        f.write("- `turns_summary.csv` – per-turn structured data.\n")
        f.write("- `turns_summary.json` – JSON array of turns.\n")
        f.write("- `aggregates.json` – overall, per-agent, and per-dilemma stats.\n")
        f.write(f"- `{experiment}_report.pdf` – tables and aggregates.\n")
        f.write(f"- `{experiment}_figures.pdf` – all figures assembled.\n")
        f.write("- `transcript.md` – original transcript for reference.\n")
        f.write("- Figures (PNG):\n")
        for fn in FIGURE_LIST:
            f.write(f"  - `{fn}`\n")
        f.write("\n## Quick Aggregate Snapshot\n\n")
        f.write("```json\n")
        f.write(json.dumps(aggregates, indent=2))
        f.write("\n```\n")
        f.write("\n## Transcript path\n\n")
        f.write(f"`{transcript_copied_path}`\n")
    return md_path


def main():
    args = parse_args()
    exp = args.experiment
    data_root = args.data_root
    
    # Output directory: default to alongside source data (data/<experiment>_outputs)
    # If --out-root is explicitly set, use that instead
    exp_safe = os.path.basename(exp.rstrip("/\\")) if "/" in exp or "\\" in exp else exp
    if args.out_root == ".":
        # Default: output alongside source data
        out_dir = os.path.join(data_root, exp + "_outputs")
    else:
        # Explicit out-root: sanitize experiment name for flat output
        exp_flat = exp.replace("/", "_").replace("\\", "_")
        out_dir = os.path.join(args.out_root, f"{exp_flat}_outputs")
    ensure_dir(out_dir)

    # 1) Load inputs
    inputs = load_inputs(data_root, exp)
    session_json = inputs["session_json"]

    # 2) Normalize session
    try:
        session_list = normalize_session(session_json, json_key=args.json_key, json_path=args.json_path)
    except Exception as e:
        print(f"[ERROR] Failed to normalize session_log.json: {e}", file=sys.stderr)
        # Helpful debug: show top-level keys and types
        if isinstance(session_json, dict):
            info = {k: type(v).__name__ for k, v in session_json.items()}
            print(f"[DEBUG] Top-level keys/types: {info}", file=sys.stderr)
        else:
            print(f"[DEBUG] Top-level type: {type(session_json).__name__}", file=sys.stderr)
        sys.exit(1)

    # 3) Build DF
    df = build_dataframe(session_list)

    # 4) Save CSV/JSON
    csv_path = os.path.join(out_dir, "turns_summary.csv")
    json_path = os.path.join(out_dir, "turns_summary.json")
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    # 5) Aggregates
    aggregates = compute_aggregates(df, session_list)
    agg_path = os.path.join(out_dir, "aggregates.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)

    # 6) Figures
    render_figures(df, out_dir, session_list)

    # 7) PDFs
    tables_pdf = build_tables_pdf(aggregates, out_dir, exp_safe)
    figures_pdf = build_figures_pdf(out_dir, exp_safe)

    # 8) Copy transcript
    transcript_out = copy_transcript(inputs["transcript_path"], out_dir)

    # 9) README
    readme_path = write_readme(out_dir, aggregates, exp_safe, transcript_out)

    print(json.dumps({
        "experiment": exp,
        "out_dir": out_dir,
        "csv_path": csv_path,
        "json_path": json_path,
        "agg_path": agg_path,
        "tables_pdf": tables_pdf,
        "figures_pdf": figures_pdf,
        "transcript_copied": transcript_out,
        "readme_path": readme_path,
        "figures": FIGURE_LIST,
    }, indent=2))

if __name__ == "__main__":
    main()
