
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
- philosophers_duel_report.pdf
- philosophers_duel_figures.pdf
- README_philosophers_duel_analysis.md
- transcript.md (copied)
- Figures (PNGs):
    epsilon_rho_after_trajectories.png
    rho_before_vs_after.png
    identity_drift.png
    trust_delta.png
    wound_activation.png
    epsilon_vs_delta_rho_scatter.png
    wordcount_vs_band_ranges.png
    band_compliance_rates.png
    phase_level_avgs.png
    wounds_per_dilemma.png
- <experiment>_outputs.zip (ZIP of folder)
"""

import argparse
import json
import os
import sys
import zipfile
from typing import Dict, List, Any

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
    "wound_activation.png",
    "epsilon_vs_delta_rho_scatter.png",
    "wordcount_vs_band_ranges.png",
    "band_compliance_rates.png",
    "phase_level_avgs.png",
    "wounds_per_dilemma.png",
]
BAND_RANGES = {
    "OPEN": (80, 150),
    "MEASURED": (60, 100),
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

    # Coerce numeric columns that might be strings
    numeric_cols = ["epsilon","rho_before","rho_after","delta_rho","wound_resonance","identity_drift","trust_delta","word_count","turn"]
    for nc in numeric_cols:
        df[nc] = pd.to_numeric(df[nc], errors="coerce")

    # Sort by turn if present
    if "turn" in df.columns and df["turn"].notna().any():
        df = df.sort_values("turn")

    df = df[cols_order].reset_index(drop=True)

    # Band compliance
    def band_compliant(row):
        rng = BAND_RANGES.get(row["band"])
        if not rng:
            return None
        wc = row["word_count"] if pd.notnull(row["word_count"]) else 0
        return int(rng[0] <= wc <= rng[1])

    df["band_compliant"] = df.apply(band_compliant, axis=1)
    return df

def compute_aggregates(df: pd.DataFrame) -> Dict[str, Any]:
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
    # Band compliance
    band_df = df[df["band"].notna()]
    compliant = int(band_df["band_compliant"].sum()) if not band_df.empty else 0
    total_with_band = int(band_df.shape[0])
    agg["band_compliance"] = {
        "total_with_band": total_with_band,
        "compliant_count": compliant,
        "compliance_rate": float(compliant / total_with_band) if total_with_band else 0.0,
    }
    # Per agent
    for ag in df["speaker"].unique():
        dfa = df[df["speaker"]==ag]
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

def render_figures(df: pd.DataFrame, out_dir: str):
    plt.style.use('seaborn-v0_8')

    # (1) epsilon & rho_after trajectories
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ag, dfa in df.groupby("speaker"):
        axes[0].plot(dfa["turn"], dfa["epsilon"], marker="o", label=ag)
        axes[1].plot(dfa["turn"], dfa["rho_after"], marker="o", label=ag)
    axes[0].set_title("Prediction Error (ε) by Turn")
    axes[0].set_ylabel("ε")
    axes[0].legend()
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

    # (4) Trust delta
    fig, ax = plt.subplots(figsize=(10,5))
    for ag, dfa in df.groupby("speaker"):
        ax.bar(dfa["turn"], dfa["trust_delta"], label=ag, alpha=0.6)
    ax.set_title("Trust Delta by Turn")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Trust delta")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trust_delta.png"))
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

    # (8) Band compliance rate
    fig, ax = plt.subplots(figsize=(8,5))
    comp = df.groupby("band")["band_compliant"].mean().rename("compliance_rate")
    comp.plot(kind="bar", ax=ax, color=["#4C78A8","#F58518"])
    ax.set_ylim(0,1)
    ax.set_title("Band Compliance Rate by Band Type")
    ax.set_ylabel("Compliance Rate")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "band_compliance_rates.png"))
    plt.close(fig)

    # (9) Phase-level averages
    fig, ax = plt.subplots(figsize=(9,5))
    phase_avg = df.groupby("phase")[["epsilon","rho_after","identity_drift"]].mean()
    phase_avg.plot(kind="bar", ax=ax)
    ax.set_title("Phase-level Averages (ε, ρ_after, identity_drift)")
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "phase_level_avgs.png"))
    plt.close(fig)

    # (10) Dilemma-level wound activations
    fig, ax = plt.subplots(figsize=(9,5))
    dilemma_wounds = df.groupby("dilemma")["wound_active"].sum()
    dilemma_wounds.plot(kind="bar", ax=ax, color="#54A24B")
    ax.set_title("Wound Activations per Dilemma")
    ax.set_ylabel("Count")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "wounds_per_dilemma.png"))
    plt.close(fig)

def build_tables_pdf(aggregates: Dict[str, Any], out_dir: str, experiment: str) -> str:
    pdf_path = os.path.join(out_dir, "philosophers_duel_report.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title = f"{experiment.replace('_',' ').title()} – DDA Visualization Report"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph("Transcript analysis, aggregates, and visual diagnostics", styles["Normal"]))
    story.append(Spacer(1, 0.25*inch))

    overall = aggregates["overall"]
    summary_data = [
        ["Metric","Value"],
        ["Turns", str(overall["turn_count"])],
        ["Agents", ", ".join(overall["agents"])],
        ["Dilemmas", ", ".join(overall["dilemmas"])],
        ["Mean ε", f"{overall['mean_epsilon']:.3f}"],
        ["Mean ρ_before", f"{overall['mean_rho_before']:.3f}"],
        ["Mean ρ_after", f"{overall['mean_rho_after']:.3f}"],
        ["Mean Δρ", f"{overall['mean_delta_rho']:.3f}"],
        ["Mean identity drift", f"{overall['mean_identity_drift']:.3f}"],
        ["Mean trust delta", f"{overall['mean_trust_delta']:.3f}"],
        ["Wound activations", str(overall["wound_active_count"])],
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
        agent_data = [
            ["Agent", ag],
            ["Turns", str(a["turns"])],
            ["Mean ε", f"{a['mean_epsilon']:.3f}"],
            ["Mean ρ_before", f"{a['mean_rho_before']:.3f}"],
            ["Mean ρ_after", f"{a['mean_rho_after']:.3f}"],
            ["Mean Δρ", f"{a['mean_delta_rho']:.3f}"],
            ["ρ_after range", f"{a['min_rho_after']:.3f} – {a['max_rho_after']:.3f}"],
            ["Mean identity drift", f"{a['mean_identity_drift']:.3f}"],
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

def build_figures_pdf(out_dir: str) -> str:
    fig_pdf_path = os.path.join(out_dir, "philosophers_duel_figures.pdf")
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
    md_path = os.path.join(out_dir, "README_philosophers_duel_analysis.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {experiment.replace('_',' ').title()} – DDA Analysis & Visualization (Run)\n\n")
        f.write("This pack contains parsed turn-level metrics, aggregates, and figures for quick inspection and repo inclusion.\n\n")
        f.write("## Files\n\n")
        f.write("- `turns_summary.csv` – per-turn structured data.\n")
        f.write("- `turns_summary.json` – JSON array of turns.\n")
        f.write("- `aggregates.json` – overall, per-agent, and per-dilemma stats.\n")
        f.write("- `philosophers_duel_report.pdf` – tables and aggregates.\n")
        f.write("- `philosophers_duel_figures.pdf` – all figures assembled.\n")
        f.write("- `transcript.md` – original transcript for reference.\n")
        f.write("- Figures (PNG):\n")
        for fn in FIGURE_LIST:
            f.write(f"  - `{fn}`\n")
        f.write("\n## Quick Aggregate Snapshot\n\n")
        f.write(json.dumps(aggregates, indent=2))
        f.write("\n\n## Transcript path\n\n")
        f.write(f"`{transcript_copied_path}`\n")
    return md_path

def zip_folder(out_dir: str) -> str:
    zip_path = out_dir.rstrip("/\\") + ".zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(out_dir):
            for fn in files:
                p = os.path.join(root, fn)
                arc = os.path.relpath(p, start=os.path.dirname(out_dir))
                z.write(p, arcname=arc)
    return zip_path

def main():
    args = parse_args()
    exp = args.experiment
    data_root = args.data_root
    out_dir = os.path.join(args.out_root, f"{exp}_outputs")
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
    aggregates = compute_aggregates(df)
    agg_path = os.path.join(out_dir, "aggregates.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregates, f, indent=2)

    # 6) Figures
    render_figures(df, out_dir)

    # 7) PDFs
    tables_pdf = build_tables_pdf(aggregates, out_dir, exp)
    figures_pdf = build_figures_pdf(out_dir)

    # 8) Copy transcript
    transcript_out = copy_transcript(inputs["transcript_path"], out_dir)

    # 9) README
    readme_path = write_readme(out_dir, aggregates, exp, transcript_out)

    # 10) ZIP
    zip_path = zip_folder(out_dir)

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
        "zip_path": zip_path,
        "figures": FIGURE_LIST,
    }, indent=2))

if __name__ == "__main__":
    main()
