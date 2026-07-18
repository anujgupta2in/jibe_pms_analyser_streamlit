"""
report_export.py
────────────────
Generates a branded PDF report and an Excel export from the filtered survey data.

Charts are rendered with Matplotlib (no browser / kaleido required).

PDF sections
  1. Cover — title, date, active filters summary
  2. Headline KPIs
  3. Satisfaction distribution
  4. NPS sentiment breakdown
  5. Data Quality breakdown
  6. Software Capability radar
  7. Top requested improvements
  8. Top data issues
"""

from __future__ import annotations

import io
import math
import tempfile
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from fpdf import FPDF

from data_loader import (
    SATISFACTION_COL, SUPPORT_COL, CORRECTION_COL, TRAINING_COL,
    RANK_COL, TENURE_COL, LIKED_COL, EXCESSIVE_COL,
    QUALITY_COLS, SOFTWARE_COLS,
    ORDINAL_MAPS,
)

# ── Colour palette ─────────────────────────────────────────────────────────────
C_BLUE   = "#1e40af"
C_LBLUE  = "#dbeafe"
C_GREEN  = "#22c55e"
C_AMBER  = "#f59e0b"
C_RED    = "#ef4444"
C_GREY   = "#64748b"
C_WHITE  = "#ffffff"

# PDF brand tuples (R, G, B int 0-255)
BRAND_BLUE  = (30, 64, 175)
BRAND_LIGHT = (219, 234, 254)
TEXT_DARK   = (15, 23, 42)
TEXT_GREY   = (100, 116, 139)

PAGE_W, PAGE_H = 210, 297
MARGIN         = 15
CONTENT_W      = PAGE_W - 2 * MARGIN


# ── Matplotlib helpers ─────────────────────────────────────────────────────────

def _mpl_style():
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "grid.color":       "#e2e8f0",
        "grid.linewidth":   0.6,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
    })


def _fig_to_png(fig: plt.Figure) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── Chart builders ─────────────────────────────────────────────────────────────

def _chart_satisfaction(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    score_counts = df[SATISFACTION_COL].value_counts().sort_index()
    scores = score_counts.index.tolist()
    counts = score_counts.values.tolist()
    colours = [
        C_RED if s <= 5 else (C_AMBER if s <= 7 else C_GREEN)
        for s in scores
    ]

    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.bar(scores, counts, color=colours, width=0.7, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(count), ha="center", va="bottom", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlabel("Satisfaction Score (1–10)", fontsize=9, color=C_GREY)
    ax.set_ylabel("Respondents",              fontsize=9, color=C_GREY)
    ax.set_title("Overall Satisfaction Distribution", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.set_xticks(scores)

    legend_handles = [
        mpatches.Patch(color=C_RED,   label="Detractors (<=5)"),
        mpatches.Patch(color=C_AMBER, label="Passives (6-7)"),
        mpatches.Patch(color=C_GREEN, label="Promoters (>=8)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left",
              framealpha=0.8, edgecolor="none")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_nps_bar(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    """Horizontal bar showing promoter / passive / detractor proportions."""
    _mpl_style()
    total      = len(df)
    promoters  = (df[SATISFACTION_COL] >= 8).sum()
    passives   = df[SATISFACTION_COL].between(6, 7).sum()
    detractors = (df[SATISFACTION_COL] <= 5).sum()
    nps        = round((promoters - detractors) / total * 100)

    p_pct = promoters  / total * 100
    pa_pct = passives  / total * 100
    d_pct = detractors / total * 100

    fig, ax = plt.subplots(figsize=(8, 1.8))
    ax.barh(["Respondents"], [p_pct],  color=C_GREEN, label=f"Promoters {p_pct:.0f}%")
    ax.barh(["Respondents"], [pa_pct], left=[p_pct],        color=C_AMBER, label=f"Passives {pa_pct:.0f}%")
    ax.barh(["Respondents"], [d_pct],  left=[p_pct + pa_pct], color=C_RED, label=f"Detractors {d_pct:.0f}%")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of respondents", fontsize=9, color=C_GREY)
    ax.set_title(f"NPS-style Sentiment  ·  Net Promoter Score: {nps:+d}",
                 fontsize=11, fontweight="bold", color=primary_hex, pad=8)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.8, edgecolor="none")
    ax.tick_params(labelsize=8)
    ax.set_yticks([])
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_quality(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    QUALITY_ORDER = ["Acceptable", "Needs Improvement", "Poor"]
    COLOURS       = {"Acceptable": C_GREEN, "Needs Improvement": C_AMBER, "Poor": C_RED}

    dims   = list(QUALITY_COLS.values())
    origs  = list(QUALITY_COLS.keys())
    x      = np.arange(len(dims))
    width  = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    for i, rating in enumerate(QUALITY_ORDER):
        pcts = [(df[orig] == rating).mean() * 100 for orig in origs]
        bars = ax.bar(x + offsets[i], pcts, width, label=rating, color=COLOURS[rating],
                      edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, pcts):
            if pct > 4:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=6.5, color=TEXT_DARK_HEX)

    ax.set_xticks(x)
    ax.set_xticklabels(dims, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("% of respondents", fontsize=9, color=C_GREY)
    ax.set_title("Data Quality Ratings by Dimension", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8, edgecolor="none")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_radar(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    ordinal_map = {"Good": 3, "Needs Improvement": 2, "Poor": 1}
    labels      = list(SOFTWARE_COLS.values())
    origs       = list(SOFTWARE_COLS.keys())
    values      = [df[orig].map(ordinal_map).mean() for orig in origs]
    values      += [values[0]]   # close the loop

    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True))
    ax.plot(angles, values,  color=primary_hex, linewidth=2)
    ax.fill(angles, values,  color=primary_hex, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=7.5, color=TEXT_DARK_HEX)
    ax.set_yticks([1, 1.5, 2, 2.5, 3])
    ax.set_yticklabels(["Poor", "", "Avg", "", "Good"], size=7, color=C_GREY)
    ax.set_ylim(1, 3)
    ax.set_title("Software Capability Radar", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=15)
    ax.grid(color="#e2e8f0", linewidth=0.6)
    ax.spines["polar"].set_visible(False)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_improvements(data: dict, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    df           = data["survey"]
    improvements = data["improvements"]
    imp_df       = improvements.head(10).reset_index()
    imp_df.columns = ["Improvement", "Votes"]
    imp_df["Pct"]  = imp_df["Votes"] / len(df) * 100
    imp_df         = imp_df.sort_values("Pct", ascending=True)

    # Truncate long labels
    labels = [
        (s[:52] + "…") if len(s) > 53 else s
        for s in imp_df["Improvement"].tolist()
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.4, 0.85, len(imp_df)))
    bars = ax.barh(range(len(labels)), imp_df["Pct"], color=colours,
                   edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, imp_df["Pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("% of respondents who selected this", fontsize=9, color=C_GREY)
    ax.set_title("Top Requested Improvements", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_issues(data: dict, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    df            = data["survey"]
    issues_counts = data["issues"]
    filtered = issues_counts[
        ~issues_counts.index.str.startswith("No significant")
    ].head(10).reset_index()
    filtered.columns = ["Issue", "Count"]
    filtered["Pct"]  = filtered["Count"] / len(df) * 100
    filtered         = filtered.sort_values("Pct", ascending=True)

    labels = [
        (s[:52] + "…") if len(s) > 53 else s
        for s in filtered["Issue"].tolist()
    ]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.4, 0.85, len(filtered)))
    bars = ax.barh(range(len(labels)), filtered["Pct"], color=colours,
                   edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, filtered["Pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("% of respondents", fontsize=9, color=C_GREY)
    ax.set_title("Most Frequently Reported Data Issues", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)





def _chart_support_dist(df: pd.DataFrame, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    score_counts = df[SUPPORT_COL].value_counts().sort_index()
    scores = score_counts.index.tolist()
    counts = score_counts.values.tolist()
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.4, 0.85, len(scores)))
    
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bars = ax.bar(scores, counts, color=colours, width=0.7, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlabel("Support Rating (1–10)", fontsize=9, color=C_GREY)
    ax.set_ylabel("Respondents",            fontsize=9, color=C_GREY)
    ax.set_title("Support Rating Distribution", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.set_xticks(scores)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_box_by_rank(df: pd.DataFrame, column: str, title: str, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    ranks = sorted(df[RANK_COL].dropna().unique().tolist())
    if not ranks:
        return b""
    data = [df[df[RANK_COL] == r][column].dropna().values for r in ranks]
    
    # Filter out empty lists
    valid_indices = [i for i, d in enumerate(data) if len(d) > 0]
    valid_ranks = [ranks[i] for i in valid_indices]
    valid_data = [data[i] for i in valid_indices]
    if not valid_data:
        return b""
        
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.boxplot(valid_data, patch_artist=True,
               boxprops=dict(facecolor="#f1f5f9", color=primary_hex, linewidth=1),
               medianprops=dict(color=primary_hex, linewidth=2),
               whiskerprops=dict(color=C_GREY),
               capprops=dict(color=C_GREY))
    ax.set_title(title, fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.set_xticks(range(1, len(valid_ranks) + 1))
    ax.set_xticklabels(valid_ranks, rotation=15, ha="right", fontsize=7.5)
    ax.set_ylabel("Rating", fontsize=9, color=C_GREY)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_training_impact(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    trained = df[df[TRAINING_COL] == "Yes"][SATISFACTION_COL].dropna()
    not_trained = df[df[TRAINING_COL] == "No"][SATISFACTION_COL].dropna()
    avg_t = trained.mean() if len(trained) > 0 else 0
    avg_nt = not_trained.mean() if len(not_trained) > 0 else 0
    
    fig, ax = plt.subplots(figsize=(7, 3.2))
    bars = ax.bar(["Attended Training", "Did Not Attend"], [avg_t, avg_nt], color=[C_GREEN, C_AMBER], width=0.4, edgecolor="white", linewidth=0.5)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.3,
                f"{height:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=TEXT_DARK_HEX)
    ax.set_ylim(0, 10.8)
    ax.set_ylabel("Avg Satisfaction (1–10)", fontsize=9, color=C_GREY)
    ax.set_title("Training Attendance vs Satisfaction", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.tick_params(labelsize=8.5)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_support_awareness(support_res: pd.Series, total_resp: int, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    if support_res.empty:
        return b""
    res_df = support_res.reset_index()
    res_df.columns = ["Resource", "Count"]
    res_df["Pct"] = res_df["Count"] / total_resp * 100
    res_df = res_df.sort_values("Pct", ascending=True)
    
    labels = [
        (s[:52] + "…") if len(s) > 53 else s
        for s in res_df["Resource"].tolist()
    ]
    
    fig, ax = plt.subplots(figsize=(9, 4))
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.4, 0.85, len(res_df)))
    bars = ax.barh(range(len(labels)), res_df["Pct"], color=colours, edgecolor="white", linewidth=0.5)
    for bar, pct in zip(bars, res_df["Pct"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.0f}%", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("% of respondents who knew about this", fontsize=9, color=C_GREY)
    ax.set_title("Support Resource Awareness", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_correlation_heatmap(enc: pd.DataFrame, primary_hex: str = C_BLUE, method: str = "spearman") -> bytes:
    _mpl_style()
    corr_matrix = enc.corr(method=method, numeric_only=True)
    if corr_matrix.empty:
        return b""
        
    fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.imshow(corr_matrix.values, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    
    labels = corr_matrix.columns
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", 
                    color="white" if abs(val) > 0.45 else "black", fontsize=7)
                    
    method_label = "Spearman" if method == "spearman" else "Pearson"
    ax.set_title(f"Correlation Heatmap ({method_label}-Encoded)", fontsize=11, fontweight="bold", color=primary_hex, pad=15)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_drivers(enc: pd.DataFrame, primary_hex: str = C_BLUE, method: str = "spearman") -> bytes:
    _mpl_style()
    corr_matrix = enc.corr(method=method, numeric_only=True)
    if "Satisfaction" not in corr_matrix.columns:
        return b""
    sat_corr = corr_matrix["Satisfaction"].drop("Satisfaction").sort_values(key=abs, ascending=True)
    features = sat_corr.index.tolist()
    corrs = sat_corr.values.tolist()
    
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = ["#22c55e" if c > 0 else "#ef4444" for c in corrs]
    bars = ax.barh(range(len(features)), corrs, color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, corrs):
        align = "left" if val < 0 else "right"
        offset = -0.01 if val < 0 else 0.01
        ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.3f}",
                va="center", ha="right" if val < 0 else "left", fontsize=7.5, color=TEXT_DARK_HEX)
                
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=7.5)
    
    method_desc = "Spearman Rank Correlation (rho)" if method == "spearman" else "Pearson Correlation Coefficient (r)"
    ax.set_xlabel(method_desc, fontsize=9, color=C_GREY)
    ax.set_title("What Drives Satisfaction Most? (Drivers ranking)", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _get_theme_counts(series: pd.Series, themes: dict) -> pd.DataFrame:
    rows = []
    import re
    for theme, keywords in themes.items():
        pattern = "|".join(re.escape(k) for k in keywords)
        matches = series[series.str.lower().str.contains(pattern, na=False)]
        rows.append({
            "Theme": theme,
            "Count": len(matches)
        })
    return pd.DataFrame(rows).sort_values("Count", ascending=False)


def _chart_theme_bar(theme_df: pd.DataFrame, colour_hex: str, title: str, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    valid = theme_df[theme_df["Count"] > 0].sort_values("Count", ascending=True)
    if valid.empty:
        return b""
    fig, ax = plt.subplots(figsize=(9, max(2.5, len(valid)*0.35)))
    bars = ax.barh(range(len(valid)), valid["Count"], color=colour_hex, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, valid["Count"]):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, str(val),
                va="center", ha="left", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_yticks(range(len(valid)))
    ax.set_yticklabels(valid["Theme"].tolist(), fontsize=7.5)
    ax.set_xlabel("Mentions", fontsize=8.5, color=C_GREY)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=primary_hex, pad=8)
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_opentext_compare(liked_themes: pd.DataFrame, excess_themes: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    top_pos = liked_themes[liked_themes["Count"] > 0].head(5).copy()
    top_neg = excess_themes[excess_themes["Count"] > 0].head(5).copy()
    
    fig, ax = plt.subplots(figsize=(9, 4))
    
    y_pos_pos = np.arange(len(top_pos))
    y_pos_neg = np.arange(len(top_neg)) + len(top_pos) + 0.5
    
    bars_pos = ax.barh(y_pos_pos, top_pos["Count"], color="#22c55e", label="Positive Themes", height=0.45)
    bars_neg = ax.barh(y_pos_neg, top_neg["Count"], color="#ef4444", label="Concern Themes", height=0.45)
    
    for bar, val in zip(bars_pos, top_pos["Count"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va="center", ha="left", fontsize=7.5)
    for bar, val in zip(bars_neg, top_neg["Count"]):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va="center", ha="left", fontsize=7.5)
        
    ticks = np.concatenate([y_pos_pos, y_pos_neg])
    labels = top_pos["Theme"].tolist() + top_neg["Theme"].tolist()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Top Themes Comparison", fontsize=10.5, fontweight="bold", color=primary_hex, pad=10)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_vessels_stacked_bar(view: pd.DataFrame, merged: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    CORR_ORDER = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
    CORR_COLOURS = ["#22c55e", "#86efac", "#f59e0b", "#ef4444"]
    
    sorted_vessels = view.sort_values("High Corr %", ascending=True)
    if sorted_vessels.empty:
        return b""
        
    vessels = sorted_vessels["Vessel"].tolist()
    
    counts = {band: [] for band in CORR_ORDER}
    for vessel in vessels:
        v_data = merged[merged["Vessel Name"] == vessel]
        for band in CORR_ORDER:
            counts[band].append((v_data[CORRECTION_COL] == band).sum())
            
    fig, ax = plt.subplots(figsize=(9, max(3, len(vessels) * 0.35)))
    
    left = np.zeros(len(vessels))
    for i, band in enumerate(CORR_ORDER):
        vals = np.array(counts[band])
        ax.barh(vessels, vals, left=left, color=CORR_COLOURS[i], label=band, height=0.6)
        left += vals
        
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("Number of respondents", fontsize=9, color=C_GREY)
    ax.set_title("Job Correction Rate Breakdown by Vessel", fontsize=11, fontweight="bold", color=primary_hex, pad=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_compare_bar(title: str, dims: list[str], scores_a: list[float], scores_b: list[float], label_a: str, label_b: str, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    x = np.arange(len(dims))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars_a = ax.bar(x - width/2, scores_a, width, label=label_a, color="#1e40af", edgecolor="white", linewidth=0.5)
    bars_b = ax.bar(x + width/2, scores_b, width, label=label_b, color="#f97316", edgecolor="white", linewidth=0.5)
    
    for bar in bars_a:
        val = bar.get_height()
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    for bar in bars_b:
        val = bar.get_height()
        if pd.notna(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
            
    ax.set_xticks(x)
    ax.set_xticklabels(dims, rotation=15, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", color=primary_hex, pad=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_scatters_grid(enc: pd.DataFrame, primary_hex: str = C_BLUE, method: str = "spearman") -> bytes:
    _mpl_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    
    scatters_info = [
        ("Satisfaction", "Support Rating", "Sat vs Support"),
        ("Satisfaction", "Correction %", "Sat vs Correction %"),
        ("Satisfaction", "PMS Tenure", "Sat vs PMS Tenure")
    ]
    
    from scipy import stats
    
    for i, (col_x, col_y, title) in enumerate(scatters_info):
        ax = axes[i]
        r_val, p_val = 0.0, 1.0
        coeff_label = "rho" if method == "spearman" else "r"
        
        if col_x in enc.columns and col_y in enc.columns:
            sub = enc[[col_x, col_y]].dropna()
            if len(sub) >= 3:
                x_vals = sub[col_x].values
                y_vals = sub[col_y].values
                
                if method == "spearman":
                    r_val, p_val = stats.spearmanr(x_vals, y_vals)
                    coeff_label = "rho"
                else:
                    r_val, p_val = stats.pearsonr(x_vals, y_vals)
                    coeff_label = "r"
                    
                ax.scatter(x_vals, y_vals, color=primary_hex, alpha=0.35, edgecolors="none")
                try:
                    m, b = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_vals.min(), x_vals.max(), 10)
                    ax.plot(x_line, m*x_line + b, color="#ef4444", linewidth=1.5)
                except Exception:
                    pass
        ax.set_title(f"{title} ({coeff_label}={r_val:.2f})", fontsize=8.5, fontweight="bold", color=primary_hex)
        ax.set_xlabel(col_x, fontsize=7.5)
        ax.set_ylabel(col_y, fontsize=7.5)
        ax.tick_params(labelsize=7)
        
    fig.tight_layout()
    return _fig_to_png(fig)


# String helper (used in charts)
TEXT_DARK_HEX = "#0f172a"


def _safe(text: str) -> str:
    """Replace non-latin-1 characters so fpdf Helvetica doesn't crash."""
    return (
        str(text)
        .replace("\u2013", "-")   # en-dash
        .replace("\u2014", "-")   # em-dash
        .replace("\u2019", "'")   # right single quote
        .replace("\u2018", "'")   # left single quote
        .replace("\u201c", '"')   # left double quote
        .replace("\u201d", '"')   # right double quote
        .replace("\u00b0", "deg") # degree sign
        .replace("\u00b7", ".")   # middle dot
        .encode("latin-1", errors="replace").decode("latin-1")
    )


# ── Additional chart builders ──────────────────────────────────────────────────

def _chart_rank_dist(df: pd.DataFrame, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    rank_counts = df[RANK_COL].value_counts().sort_values(ascending=True)
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.4, 0.85, len(rank_counts)))
    fig, ax = plt.subplots(figsize=(9, max(3, len(rank_counts) * 0.45)))
    bars = ax.barh(rank_counts.index, rank_counts.values, color=colours,
                   edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, rank_counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlabel("Respondents", fontsize=9, color=C_GREY)
    ax.set_title("Respondents by Rank", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_sat_by_rank(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    grp = df.groupby(RANK_COL)[SATISFACTION_COL].mean().sort_values(ascending=True)
    colours = [C_RED if v < 6 else (C_AMBER if v < 7.5 else C_GREEN) for v in grp.values]
    fig, ax = plt.subplots(figsize=(9, max(3, len(grp) * 0.45)))
    bars = ax.barh(grp.index, grp.values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlim(0, 10)
    ax.axvline(7, color=primary_hex, linewidth=1, linestyle="--", label="Target 7.0")
    ax.set_xlabel("Avg Satisfaction (1–10)", fontsize=9, color=C_GREY)
    ax.set_title("Avg Satisfaction by Rank", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_tenure_dist(df: pd.DataFrame, primary_hex: str = C_BLUE, cmap_name: str = "Blues") -> bytes:
    _mpl_style()
    order  = ["Less than 6 Months", "6 Months – 1 Year", "1 - 2 Years", "More than 2 Years"]
    counts = df[TENURE_COL].value_counts().reindex(order, fill_value=0)
    cmap = getattr(plt.cm, cmap_name, plt.cm.Blues)
    colours = cmap(np.linspace(0.35, 0.85, len(order)))
    fig, ax = plt.subplots(figsize=(9, 3))
    bars = ax.bar(counts.index, counts.values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(count), ha="center", va="bottom", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_ylabel("Respondents", fontsize=9, color=C_GREY)
    ax.set_title("PMS Tenure Distribution", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_sat_by_vessel_type(merged: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    if "Type" not in merged.columns:
        return b""
    grp = merged.groupby("Type")[SATISFACTION_COL].mean().sort_values(ascending=True).dropna()
    if grp.empty:
        return b""
    colours = [C_RED if v < 6 else (C_AMBER if v < 7.5 else C_GREEN) for v in grp.values]
    fig, ax = plt.subplots(figsize=(9, max(3, len(grp) * 0.55)))
    bars = ax.barh(grp.index, grp.values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlim(0, 10)
    ax.axvline(7, color=primary_hex, linewidth=1, linestyle="--", label="Target 7.0")
    ax.set_xlabel("Avg Satisfaction (1–10)", fontsize=9, color=C_GREY)
    ax.set_title("Avg Satisfaction by Vessel Type", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_sat_by_division(merged: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    if "Division Code" not in merged.columns:
        return b""
    grp = merged.groupby("Division Code")[SATISFACTION_COL].mean().sort_values(ascending=True).dropna()
    if grp.empty:
        return b""
    colours = [C_RED if v < 6 else (C_AMBER if v < 7.5 else C_GREEN) for v in grp.values]
    fig, ax = plt.subplots(figsize=(9, max(3, len(grp) * 0.5)))
    bars = ax.barh(grp.index, grp.values, color=colours, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, grp.values):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8, color=TEXT_DARK_HEX)
    ax.set_xlim(0, 10)
    ax.axvline(7, color=primary_hex, linewidth=1, linestyle="--")
    ax.set_xlabel("Avg Satisfaction (1–10)", fontsize=9, color=C_GREY)
    ax.set_title("Avg Satisfaction by Division", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _chart_correction_by_rank(df: pd.DataFrame, primary_hex: str = C_BLUE) -> bytes:
    _mpl_style()
    corr_order = ["Less than 5%", "5–10%", "10–20%", "More than 20%"]
    colours    = [C_GREEN, "#86efac", C_AMBER, C_RED]
    cross      = pd.crosstab(df[RANK_COL], df[CORRECTION_COL])
    cross      = cross.reindex(columns=corr_order, fill_value=0)
    cross_pct  = cross.div(cross.sum(axis=1), axis=0) * 100

    ranks  = cross_pct.index.tolist()
    x      = np.arange(len(ranks))
    width  = 0.18
    offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(ranks) * 0.5)))
    for i, (cat, col) in enumerate(zip(corr_order, colours)):
        vals = cross_pct[cat].values
        ax.bar(x + offsets[i], vals, width, label=cat, color=col,
               edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(ranks, fontsize=7.5, rotation=20, ha="right")
    ax.set_ylabel("% of rank", fontsize=9, color=C_GREY)
    ax.set_title("Job Correction Rate by Rank", fontsize=11, fontweight="bold",
                 color=primary_hex, pad=10)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.8, edgecolor="none")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return _fig_to_png(fig)


def _vessel_correction_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Return vessel-level correction summary sorted by high-correction %."""
    HIGH = {"10–20%", "More than 20%"}
    rows = []
    for vessel, grp in merged.groupby("Vessel Name"):
        corr = grp[CORRECTION_COL].dropna()
        if len(corr) == 0:
            continue
        high_pct = corr.isin(HIGH).sum() / len(corr) * 100
        vtype = grp["Type"].dropna().mode().iloc[0] if "Type" in grp and grp["Type"].notna().any() else "—"
        div   = grp["Division Code"].dropna().mode().iloc[0] if "Division Code" in grp and grp["Division Code"].notna().any() else "—"
        avg_s = grp[SATISFACTION_COL].mean() if grp[SATISFACTION_COL].notna().any() else None
        rows.append({
            "Vessel":     vessel,
            "Type":       vtype,
            "Division":   div,
            "Resp.":      len(grp),
            "High Corr %": round(high_pct, 1),
            "Avg Sat":    round(avg_s, 1) if avg_s and pd.notna(avg_s) else "—",
        })
    df = pd.DataFrame(rows).sort_values("High Corr %", ascending=False)
    return df.reset_index(drop=True)


# ── PDF class ──────────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    def __init__(self, brand_primary=(30, 64, 175), brand_light=(219, 234, 254), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.brand_primary = brand_primary
        self.brand_light = brand_light

    def header(self):
        self.set_fill_color(*self.brand_primary)
        self.rect(0, 0, PAGE_W, 10, "F")
        self.set_y(12)

    def footer(self):
        self.set_y(-13)
        self.set_fill_color(*self.brand_light)
        self.rect(0, PAGE_H - 13, PAGE_W, 13, "F")
        self.set_y(-10)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*TEXT_GREY)
        self.cell(0, 6,
                  f"JiBe PMS Survey Analysis  ·  Anglo Eastern  ·  Page {self.page_no()}",
                  align="C")
        self.set_text_color(*TEXT_DARK)

    def section_title(self, text: str):
        self.ln(4)
        self.set_font("Helvetica", "B", 13)
        self.set_fill_color(*self.brand_light)
        self.set_text_color(*self.brand_primary)
        self.cell(CONTENT_W, 8, _safe(text), fill=True, ln=True, align="L")
        self.set_text_color(*TEXT_DARK)
        self.ln(3)

    def kpi_row(self, kpis: list[tuple]):
        n      = len(kpis)
        cell_w = CONTENT_W / n
        x0     = MARGIN
        y0     = self.get_y()
        for label, value, note in kpis:
            self.set_xy(x0, y0)
            self.set_fill_color(*self.brand_light)
            self.rect(x0, y0, cell_w - 3, 22, "F")
            self.set_xy(x0 + 2, y0 + 2)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*TEXT_GREY)
            self.cell(cell_w - 5, 4, label, ln=True)
            self.set_xy(x0 + 2, self.get_y())
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*self.brand_primary)
            self.cell(cell_w - 5, 7, value, ln=True)
            if note:
                self.set_xy(x0 + 2, self.get_y())
                self.set_font("Helvetica", "I", 6.5)
                self.set_text_color(*TEXT_GREY)
                self.cell(cell_w - 5, 4, note, ln=True)
            x0 += cell_w
        self.set_text_color(*TEXT_DARK)
        self.set_y(y0 + 25)

    def embed_png(self, png_bytes: bytes, w: Optional[float] = None, center: bool = True):
        w = w or CONTENT_W
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name
        try:
            x = MARGIN if not center else (PAGE_W - w) / 2
            self.image(tmp_path, x=x, w=w)
        finally:
            os.unlink(tmp_path)
        self.ln(3)


# ── Public API ─────────────────────────────────────────────────────────────────

COLOR_THEMES = {
    "Classic Blue": {
        "primary_rgb": (30, 64, 175),
        "light_rgb": (219, 234, 254),
        "primary_hex": "#1e40af",
        "cmap": "Blues"
    },
    "Teal / Emerald": {
        "primary_rgb": (13, 148, 136),
        "light_rgb": (204, 251, 241),
        "primary_hex": "#0d9488",
        "cmap": "GnBu"
    },
    "Charcoal": {
        "primary_rgb": (75, 85, 99),
        "light_rgb": (243, 244, 246),
        "primary_hex": "#4b5563",
        "cmap": "Greys"
    },
    "Crimson": {
        "primary_rgb": (185, 28, 28),
        "light_rgb": (254, 226, 226),
        "primary_hex": "#b91c1c",
        "cmap": "Reds"
    }
}

DEFAULT_PDF_OPTIONS = {
    # Cover metadata
    "report_title":  "JiBe PMS Survey Analysis Report",
    "report_subtitle": "Data Quality & Product Health Assessment",
    "author_org":    "Anglo Eastern",
    "include_cover":     True,
    "executive_summary": "",
    "color_theme":       "Classic Blue",
    "vessel_limit":      35,
    "keyword_limit":     10,
    
    # Granular visual switches
    "vis_kpis":             True,
    "vis_sat_dist":         True,
    "vis_nps_bar":          True,
    "vis_sup_dist":         True,
    "vis_sat_box_rank":     True,
    "vis_sup_box_rank":     True,
    "vis_train_impact":     True,
    "vis_sup_awareness":    True,
    
    "vis_qual_chart":       True,
    "vis_qual_table":       True,
    "vis_soft_radar":       True,
    "vis_soft_table":       True,
    "vis_improvements":     True,
    "vis_issues":           True,
    
    "vis_corr_heatmap":     True,
    "vis_scatters":         True,
    "vis_chisquare_table":  True,
    "vis_drivers_chart":    True,
    
    "vis_profile_rank":     True,
    "vis_profile_tenure":   True,
    "vis_profile_sat":      True,
    
    "vis_fleet_vtype":      True,
    "vis_fleet_div":        True,
    
    "vis_corr_rank":        True,
    "vis_corr_table":       True,
    
    "vis_vessels_table":        True,
    "vis_vessels_stacked_bar":  True,
    
    "vis_opentext_lists":    True,
    "vis_opentext_pos_bar":  True,
    "vis_opentext_neg_bar":  True,
    "vis_opentext_compare":  True,
    
    "vis_compare_summary":   True,
    "vis_compare_quality":   True,
    "vis_compare_software":  True,
    "vis_compare_sat":       True,
}


def generate_pdf(data: dict, active_filters: dict, pdf_options: dict | None = None) -> bytes:
    """
    Generate a branded PDF report from filtered survey data.

    Parameters
    ----------
    data           : dict returned by load_data / apply_filters
    active_filters : dict of active sidebar filter selections (for the cover page)
    pdf_options    : optional overrides for DEFAULT_PDF_OPTIONS

    Returns
    -------
    bytes  PDF content

    Raises
    ------
    ValueError  if the filtered dataset contains zero rows.
    """
    opts   = {**DEFAULT_PDF_OPTIONS, **(pdf_options or {})}
    df     = data["survey"]
    merged = data.get("merged", df)
    total  = len(df)

    if total == 0:
        raise ValueError(
            "Cannot generate a report: the current filter combination matches "
            "zero respondents. Please broaden or clear the filters before exporting."
        )

    # ── Theme color mapping ───────────────────────────────────────────────────
    theme_opt = opts.get("color_theme", "Classic Blue")
    theme = COLOR_THEMES.get(theme_opt, COLOR_THEMES["Classic Blue"])
    primary_rgb = theme["primary_rgb"]
    light_rgb = theme["light_rgb"]
    primary_hex = theme["primary_hex"]
    cmap_name = theme["cmap"]

    # ── Headline KPIs ─────────────────────────────────────────────────────────
    vessel_count = merged["Vessel Name"].nunique() if "Vessel Name" in merged.columns else 0
    avg_sat    = df[SATISFACTION_COL].mean()
    avg_sup    = df[SUPPORT_COL].mean()
    high_corr  = df[CORRECTION_COL].isin(["10–20%", "More than 20%"]).sum() / total * 100
    trained    = (df[TRAINING_COL] == "Yes").sum() / total * 100
    promoters  = (df[SATISFACTION_COL] >= 8).sum()
    passives   = df[SATISFACTION_COL].between(6, 7).sum()
    detractors = (df[SATISFACTION_COL] <= 5).sum()
    nps        = round((promoters - detractors) / total * 100)

    # ── Build PDF ─────────────────────────────────────────────────────────────
    pdf = ReportPDF(brand_primary=primary_rgb, brand_light=light_rgb, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(MARGIN, 14, MARGIN)

    # ── Cover ─────────────────────────────────────────────────────────────────
    if opts.get("include_cover", True):
        pdf.add_page()
        pdf.set_fill_color(*primary_rgb)
        pdf.rect(0, 10, PAGE_W, 68, "F")
        pdf.set_y(24)
        pdf.set_font("Helvetica", "B", 24)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 12, _safe(opts["report_title"]), align="C", ln=True)
        pdf.set_font("Helvetica", "", 13)
        pdf.cell(0, 8, _safe(opts["report_subtitle"]), align="C", ln=True)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 7, _safe(opts["author_org"]) + "  ·  " + datetime.now().strftime("%d %B %Y"),
                 align="C", ln=True)
        pdf.set_text_color(*TEXT_DARK)

        pdf.set_y(90)
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_text_color(*primary_rgb)
        pdf.cell(0, 7, "Report Scope", ln=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*TEXT_DARK)
        pdf.cell(0, 6, f"Based on {total:,} survey responses", ln=True)

        if active_filters:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*primary_rgb)
            pdf.cell(0, 6, "Active Filters", ln=True)
            label_map = {
                "rank": "Rank", "vessel_type": "Vessel Type",
                "division": "Division Code", "tenure": "PMS Tenure",
            }
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_DARK)
            for key, vals in active_filters.items():
                pdf.cell(0, 5, f"  * {label_map.get(key, key)}: {', '.join(vals)}", ln=True)
        else:
            pdf.set_font("Helvetica", "I", 9)
            pdf.set_text_color(*TEXT_GREY)
            pdf.cell(0, 5, "No filters applied -- showing all respondents", ln=True)
            pdf.set_text_color(*TEXT_DARK)

        # Executive Summary
        exec_sum = opts.get("executive_summary", "")
        if exec_sum:
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*primary_rgb)
            pdf.cell(0, 6, "Executive Summary", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_DARK)
            pdf.multi_cell(0, 5, _safe(exec_sum))

        pdf.ln(4)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(*TEXT_GREY)
        pdf.multi_cell(0, 5,
            "Contents: Headline KPIs . Satisfaction & NPS . Data Quality . "
            "Software Capabilities . Top Issues & Improvements")
        pdf.set_text_color(*TEXT_DARK)

    # ── renumber sections dynamically ────────────────────────────────────────
    _sec = [0]
    def _next_sec(title: str) -> str:
        _sec[0] += 1
        return f"{_sec[0]} . {title}"

    # ── Section: KPIs ────────────────────────────────────────────────────────
    if opts.get("vis_kpis", True):
        pdf.add_page()
        # If cover is excluded, print executive summary at the very top of first page if present
        if not opts.get("include_cover", True) and opts.get("executive_summary", ""):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*primary_rgb)
            pdf.cell(0, 7, _safe(opts["report_title"]), ln=True)
            pdf.set_font("Helvetica", "I", 8.5)
            pdf.set_text_color(*TEXT_GREY)
            pdf.cell(0, 5, _safe(opts["report_subtitle"]) + "  ·  " + datetime.now().strftime("%d %B %Y"), ln=True)
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*primary_rgb)
            pdf.cell(0, 6, "Executive Summary", ln=True)
            pdf.set_font("Helvetica", "", 8.5)
            pdf.set_text_color(*TEXT_DARK)
            pdf.multi_cell(0, 4.5, _safe(opts["executive_summary"]))
            pdf.ln(4)

        pdf.section_title(_next_sec("Headline KPIs"))
        pdf.kpi_row([
            ("Total Responses",     f"{total:,}",               None),
            ("Vessels Covered",     f"{vessel_count:,}",        None),
            ("Avg. Satisfaction",   f"{avg_sat:.1f} / 10",      f"{avg_sat - 7:+.1f} vs target 7"),
        ])
        pdf.kpi_row([
            ("Avg. Support Rating",   f"{avg_sup:.1f} / 10",    None),
            ("Net Promoter Score",    f"{nps:+d}",
             f"Promoters {promoters/total*100:.0f}%  .  Detractors {detractors/total*100:.0f}%"),
            ("Jobs > 10% Correction", f"{high_corr:.0f}%",
             f"{high_corr - 20:+.0f}pp vs 20% benchmark"),
        ])
        pdf.kpi_row([
            ("Training Attended",      f"{trained:.0f}%",                               None),
            ("NPS Promoters (>=8)",    f"{promoters} ({promoters/total*100:.0f}%)",     None),
            ("NPS Detractors (<=5)",   f"{detractors} ({detractors/total*100:.0f}%)",   None),
        ])

    # ── Section: Satisfaction, Support & Training ─────────────────────────────
    if any(opts.get(k, True) for k in ["vis_sat_dist", "vis_nps_bar", "vis_sup_dist", "vis_sat_box_rank", "vis_sup_box_rank", "vis_train_impact", "vis_sup_awareness"]):
        pdf.section_title(_next_sec("Satisfaction, Support & Training"))
        
        if opts.get("vis_sat_dist", True):
            png_sat = _chart_satisfaction(df, primary_hex)
            pdf.embed_png(png_sat, w=CONTENT_W)
            pdf.ln(2)

        if opts.get("vis_nps_bar", True):
            png_nps = _chart_nps_bar(df, primary_hex)
            pdf.embed_png(png_nps, w=CONTENT_W)
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_GREY)
            pdf.cell(0, 5,
                f"Promoters (>=8): {promoters}  ({promoters/total*100:.0f}%)  ·  "
                f"Passives (6-7): {passives}  ({passives/total*100:.0f}%)  ·  "
                f"Detractors (<=5): {detractors}  ({detractors/total*100:.0f}%)",
                ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.ln(2)
            
        if opts.get("vis_sup_dist", True):
            png_sup = _chart_support_dist(df, primary_hex, cmap_name)
            pdf.embed_png(png_sup, w=CONTENT_W)
            pdf.ln(2)
            
        if opts.get("vis_sat_box_rank", True):
            png_box_sat = _chart_box_by_rank(df, SATISFACTION_COL, "Satisfaction Scores by Rank Distribution", primary_hex)
            if png_box_sat:
                pdf.embed_png(png_box_sat, w=CONTENT_W)
                pdf.ln(2)
                
        if opts.get("vis_sup_box_rank", True):
            png_box_sup = _chart_box_by_rank(df, SUPPORT_COL, "Support Ratings by Rank Distribution", primary_hex)
            if png_box_sup:
                pdf.embed_png(png_box_sup, w=CONTENT_W)
                pdf.ln(2)
                
        if opts.get("vis_train_impact", True):
            png_train = _chart_training_impact(df, primary_hex)
            pdf.embed_png(png_train, w=CONTENT_W)
            pdf.ln(2)
            
        if opts.get("vis_sup_awareness", True):
            png_sup_aw = _chart_support_awareness(data["support_res"], total, primary_hex, cmap_name)
            if png_sup_aw:
                pdf.embed_png(png_sup_aw, w=CONTENT_W)

    # ── Section: Data Quality ─────────────────────────────────────────────────
    if opts.get("vis_qual_chart", True) or opts.get("vis_qual_table", True):
        pdf.add_page()
        pdf.section_title(_next_sec("Data Quality Assessment"))
        
        if opts.get("vis_qual_table", True):
            pdf.set_font("Helvetica", "B", 9)
            col_w = CONTENT_W / 4
            pdf.set_fill_color(*primary_rgb)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(col_w * 1.5, 7, "Dimension",      border=0, fill=True)
            pdf.cell(col_w * 0.83, 7, "Acceptable",    align="C", border=0, fill=True)
            pdf.cell(col_w * 0.83, 7, "Needs Improv.", align="C", border=0, fill=True)
            pdf.cell(col_w * 0.83, 7, "Poor",          align="C", border=0, fill=True, ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 8)
            for i, (orig, alias) in enumerate(QUALITY_COLS.items()):
                fill = i % 2 == 0
                pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                p_acc  = (df[orig] == "Acceptable").mean() * 100
                p_ni   = (df[orig] == "Needs Improvement").mean() * 100
                p_poor = (df[orig] == "Poor").mean() * 100
                pdf.cell(col_w * 1.5,  6, alias,            fill=fill)
                pdf.cell(col_w * 0.83, 6, f"{p_acc:.0f}%",  align="C", fill=fill)
                pdf.cell(col_w * 0.83, 6, f"{p_ni:.0f}%",   align="C", fill=fill)
                pdf.cell(col_w * 0.83, 6, f"{p_poor:.0f}%", align="C", fill=fill, ln=True)
            pdf.ln(4)

        if opts.get("vis_qual_chart", True):
            png_qual = _chart_quality(df, primary_hex)
            pdf.embed_png(png_qual, w=CONTENT_W)

    # ── Section: Software Capabilities ────────────────────────────────────────
    if opts.get("vis_soft_radar", True) or opts.get("vis_soft_table", True):
        pdf.add_page()
        pdf.section_title(_next_sec("Software Capabilities"))
        
        if opts.get("vis_soft_radar", True):
            png_rad = _chart_radar(df, primary_hex)
            pdf.embed_png(png_rad, w=130)
            pdf.ln(2)

        if opts.get("vis_soft_table", True):
            ordinal_map = {"Good": 3, "Needs Improvement": 2, "Poor": 1}
            scores_list = sorted(
                [(alias, df[orig].map(ordinal_map).mean()) for orig, alias in SOFTWARE_COLS.items()],
                key=lambda x: x[1],
            )
            cw = CONTENT_W / 3
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(*primary_rgb)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(cw * 2, 7, "Capability",       border=0, fill=True)
            pdf.cell(cw,     7, "Avg Score (/3)",   align="C", border=0, fill=True, ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 8)
            for i, (alias, sc) in enumerate(scores_list):
                fill = i % 2 == 0
                pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                colour = (34, 197, 94) if sc >= 2.5 else ((245, 158, 11) if sc >= 2.0 else (239, 68, 68))
                pdf.cell(cw * 2, 6, alias, fill=fill)
                pdf.set_text_color(*colour)
                pdf.cell(cw, 6, f"{sc:.2f}", align="C", fill=fill, ln=True)
                pdf.set_text_color(*TEXT_DARK)

    # ── Section: Top Improvements & Issues ────────────────────────────────────
    if opts.get("vis_improvements", True) or opts.get("vis_issues", True):
        pdf.add_page()
        if opts.get("vis_improvements", True):
            pdf.section_title(_next_sec("Top Requested Improvements"))
            png_imp = _chart_improvements(data, primary_hex, cmap_name)
            pdf.embed_png(png_imp, w=CONTENT_W)
            pdf.ln(4)

        if opts.get("vis_issues", True):
            pdf.section_title(_next_sec("Most Reported Data Issues"))
            png_iss = _chart_issues(data, primary_hex, cmap_name)
            pdf.embed_png(png_iss, w=CONTENT_W)

    # ── Section: Advanced Correlations ────────────────────────────────────────
    if any(opts.get(k, True) for k in ["vis_corr_heatmap", "vis_scatters", "vis_chisquare_table", "vis_drivers_chart"]):
        pdf.add_page()
        pdf.section_title(_next_sec("Advanced Correlations & Drivers"))
        
        if opts.get("vis_corr_heatmap", True):
            png_corr_heat = _chart_correlation_heatmap(data["encoded"], primary_hex, method=opts.get("corr_method", "spearman"))
            if png_corr_heat:
                pdf.embed_png(png_corr_heat, w=CONTENT_W)
                pdf.ln(2)
                
        if opts.get("vis_drivers_chart", True):
            png_drivers = _chart_drivers(data["encoded"], primary_hex, method=opts.get("corr_method", "spearman"))
            if png_drivers:
                pdf.embed_png(png_drivers, w=CONTENT_W)
                pdf.ln(2)
                
        if opts.get("vis_scatters", True):
            png_scat = _chart_scatters_grid(data["encoded"], primary_hex, method=opts.get("corr_method", "spearman"))
            if png_scat:
                pdf.embed_png(png_scat, w=CONTENT_W)
                pdf.ln(4)
                
        if opts.get("vis_chisquare_table", True):
            rows = _get_chisquare_data(df)
            if rows:
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(*primary_rgb)
                pdf.cell(0, 6, "Chi-Square Association Tests (Categorical)", ln=True)
                pdf.ln(1)
                
                cw = CONTENT_W / 6
                pdf.set_fill_color(*primary_rgb)
                pdf.set_text_color(255, 255, 255)
                pdf.cell(cw * 1.5, 6, "Variable A", fill=True)
                pdf.cell(cw * 1.5, 6, "Variable B", fill=True)
                pdf.cell(cw * 0.7, 6, "Chi-Sq", fill=True, align="C")
                pdf.cell(cw * 0.5, 6, "df", fill=True, align="C")
                pdf.cell(cw * 0.8, 6, "p-value", fill=True, align="C")
                pdf.cell(cw * 1.0, 6, "Significant?", fill=True, align="C", ln=True)
                
                pdf.set_text_color(*TEXT_DARK)
                pdf.set_font("Helvetica", "", 7.5)
                for i, r in enumerate(rows):
                    fill = i % 2 == 0
                    pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                    pdf.cell(cw * 1.5, 5.5, r["var_a"], fill=fill)
                    pdf.cell(cw * 1.5, 5.5, r["var_b"], fill=fill)
                    pdf.cell(cw * 0.7, 5.5, r["chi2"], fill=fill, align="C")
                    pdf.cell(cw * 0.5, 5.5, r["df"], fill=fill, align="C")
                    pdf.cell(cw * 0.8, 5.5, r["p_val"], fill=fill, align="C")
                    pdf.cell(cw * 1.0, 5.5, r["sig"], fill=fill, align="C", ln=True)
                pdf.ln(4)

    # ── Section: Respondent Profile ───────────────────────────────────────────
    if opts.get("vis_profile_rank", True) or opts.get("vis_profile_tenure", True) or opts.get("vis_profile_sat", True):
        pdf.add_page()
        pdf.section_title(_next_sec("Respondent Profile"))
        if opts.get("vis_profile_rank", True):
            png_rank = _chart_rank_dist(df, primary_hex, cmap_name)
            pdf.embed_png(png_rank,  w=CONTENT_W)
        if opts.get("vis_profile_tenure", True):
            png_ten = _chart_tenure_dist(df, primary_hex, cmap_name)
            pdf.embed_png(png_ten,   w=CONTENT_W)
        if opts.get("vis_profile_sat", True):
            png_sat_r = _chart_sat_by_rank(df, primary_hex)
            pdf.embed_png(png_sat_r, w=CONTENT_W)

    # ── Section: Fleet Analysis ───────────────────────────────────────────────
    if (opts.get("vis_fleet_vtype", True) or opts.get("vis_fleet_div", True)) and "Type" in merged.columns and "Division Code" in merged.columns:
        pdf.add_page()
        pdf.section_title(_next_sec("Fleet Analysis"))
        if opts.get("vis_fleet_vtype", True):
            png_vtype = _chart_sat_by_vessel_type(merged, primary_hex)
            if png_vtype:
                pdf.embed_png(png_vtype, w=CONTENT_W)
        if opts.get("vis_fleet_div", True):
            png_div = _chart_sat_by_division(merged, primary_hex)
            if png_div:
                pdf.embed_png(png_div, w=CONTENT_W)

    # ── Section: Correction Rate by Rank ──────────────────────────────────────
    if opts.get("vis_corr_rank", True) or opts.get("vis_corr_table", True):
        pdf.add_page()
        pdf.section_title(_next_sec("Job Correction Rate by Rank"))
        if opts.get("vis_corr_rank", True):
            png_corr_rank = _chart_correction_by_rank(df, primary_hex)
            pdf.embed_png(png_corr_rank, w=CONTENT_W)
        
        if opts.get("vis_corr_table", True):
            corr_order  = ["Less than 5%", "5\u201310%", "10\u201320%", "More than 20%"]
            corr_counts = df[CORRECTION_COL].value_counts().reindex(corr_order, fill_value=0)
            pdf.ln(2)
            cw2 = CONTENT_W / 3
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(*primary_rgb)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(cw2 * 2, 7, "Correction Rate Band",   border=0, fill=True)
            pdf.cell(cw2,     7, "Respondents (%)", align="C", border=0, fill=True, ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 8)
            for i, band in enumerate(corr_order):
                cnt  = corr_counts[band]
                pct  = cnt / total * 100
                fill = i % 2 == 0
                pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                pdf.cell(cw2 * 2, 6, _safe(band),              fill=fill)
                pdf.cell(cw2,     6, f"{cnt}  ({pct:.0f}%)",   align="C", fill=fill, ln=True)

    # ── Section: Vessels High Correction ──────────────────────────────────────
    if (opts.get("vis_vessels_table", True) or opts.get("vis_vessels_stacked_bar", True)) and "Vessel Name" in merged.columns:
        pdf.add_page()
        pdf.section_title(_next_sec("Vessels Correction Rates"))
        vessel_tbl = _vessel_correction_table(merged)
        
        if opts.get("vis_vessels_stacked_bar", True):
            png_v_stacked = _chart_vessels_stacked_bar(vessel_tbl, merged, primary_hex)
            if png_v_stacked:
                pdf.embed_png(png_v_stacked, w=CONTENT_W)
                pdf.ln(4)
                
        if opts.get("vis_vessels_table", True):
            flagged    = vessel_tbl[vessel_tbl["High Corr %"] >= 10]
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_GREY)
            pdf.cell(0, 5, f"{len(flagged)} of {len(vessel_tbl)} vessels have >10% high-correction respondents.", ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.ln(2)
            col_widths = [52, 34, 18, 18, 22, 18]
            headers    = ["Vessel", "Type", "Div", "Resp.", "High Corr %", "Avg Sat"]
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_fill_color(*primary_rgb)
            pdf.set_text_color(255, 255, 255)
            for h, cw3 in zip(headers, col_widths):
                pdf.cell(cw3, 7, h, border=0, fill=True)
            pdf.ln()
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 7.5)
            
            limit_rows = opts.get("vessel_limit", 35)
            for i, row in vessel_tbl.head(limit_rows).iterrows():
                fill = i % 2 == 0
                pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                flag_icon = ">> " if row["High Corr %"] >= 10 else "   "
                values = [
                    _safe(flag_icon + str(row["Vessel"])[:28]),
                    _safe(str(row["Type"])[:20]),
                    _safe(str(row["Division"])[:8]),
                    str(row["Resp."]),
                    f"{row['High Corr %']:.1f}%",
                    str(row["Avg Sat"]),
                ]
                for val, cw3 in zip(values, col_widths):
                    pdf.set_text_color(180, 90, 0) if row["High Corr %"] >= 10 else pdf.set_text_color(*TEXT_DARK)
                    pdf.cell(cw3, 5.5, val, fill=fill)
                pdf.ln()
            pdf.set_text_color(*TEXT_DARK)

    # ── Section: Open Text Key Themes ─────────────────────────────────────────
    if any(opts.get(k, True) for k in ["vis_opentext_lists", "vis_opentext_pos_bar", "vis_opentext_neg_bar", "vis_opentext_compare"]):
        pdf.add_page()
        pdf.section_title(_next_sec("Open Text Feedback & Themes"))

        import re
        from collections import Counter as _Counter

        STOPWORDS_PDF = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with",
            "is","it","are","was","be","as","this","that","have","has","from","by",
            "we","i","our","its","very","also","all","can","not","no","do","so",
            "more","some","they","their","there","which","when","will","been","had",
            "would","could","should","may","than","if","up","out","about","what",
            "into","who","how","any","just","use","used","using","one","my","your",
            "his","her","he","she","us","each","other","such","does","did","were",
            "am","over","new","time","get","only","still","need","well","good",
            "like","make","most","per","etc","na","nan","pms","jibe","system",
        }

        kw_limit = opts.get("keyword_limit", 10)

        def _top_words(series, n=10):
            words = []
            for text in series.dropna():
                tokens = re.findall(r"[a-zA-Z]{4,}", str(text).lower())
                words.extend([t for t in tokens if t not in STOPWORDS_PDF])
            return _Counter(words).most_common(n)

        liked_s  = df[LIKED_COL].dropna() if LIKED_COL in df.columns else pd.Series(dtype=str)
        liked_s  = liked_s[liked_s.str.strip().str.len() > 3]
        excess_s = df[EXCESSIVE_COL].dropna() if EXCESSIVE_COL in df.columns else pd.Series(dtype=str)
        excess_s = excess_s[~excess_s.str.strip().str.lower().isin(
            {"none","nil","n/a","na","no","-","nothing","no comment"})]
        excess_s = excess_s[excess_s.str.strip().str.len() > 5]

        from page_modules.page9_text import POSITIVE_THEMES, NEGATIVE_THEMES
        liked_themes = _get_theme_counts(liked_s, POSITIVE_THEMES)
        excess_themes = _get_theme_counts(excess_s, NEGATIVE_THEMES)

        if opts.get("vis_opentext_pos_bar", True):
            png_pos_theme = _chart_theme_bar(liked_themes, "#0ea5e9", "Positive Feedback Themes - Frequency", primary_hex)
            if png_pos_theme:
                pdf.embed_png(png_pos_theme, w=CONTENT_W)
                pdf.ln(2)

        if opts.get("vis_opentext_neg_bar", True):
            png_neg_theme = _chart_theme_bar(excess_themes, "#ef4444", "Concern Themes - Frequency", primary_hex)
            if png_neg_theme:
                pdf.embed_png(png_neg_theme, w=CONTENT_W)
                pdf.ln(2)

        if opts.get("vis_opentext_compare", True):
            png_comp_theme = _chart_opentext_compare(liked_themes, excess_themes, primary_hex)
            if png_comp_theme:
                pdf.embed_png(png_comp_theme, w=CONTENT_W)
                pdf.ln(4)

        if opts.get("vis_opentext_lists", True):
            top_liked  = _top_words(liked_s,  kw_limit)
            top_excess = _top_words(excess_s, kw_limit)

            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(*TEXT_GREY)
            pdf.cell(0, 5,
                f"Positive feedback: {len(liked_s)} responses  ·  "
                f"Concerns: {len(excess_s)} responses",
                ln=True)
            pdf.set_text_color(*TEXT_DARK)
            pdf.ln(2)

            half = CONTENT_W / 2 - 3
            x_left  = MARGIN
            x_right = MARGIN + half + 6
            y_start = pdf.get_y()

            # Left column — liked
            pdf.set_xy(x_left, y_start)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(34, 197, 94)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(half, 7, "  What respondents liked (top keywords)", fill=True, ln=False)
            pdf.set_xy(x_left, y_start + 8)
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 8.5)
            for rank, (word, cnt) in enumerate(top_liked, 1):
                pdf.set_x(x_left)
                pct = cnt / max(len(liked_s), 1) * 100
                pdf.cell(half, 5.5, f"  {rank:>2}. {word.capitalize():<22}  {cnt} mentions ({pct:.0f}%)", ln=True)

            # Right column — concerns
            pdf.set_xy(x_right, y_start)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_fill_color(239, 68, 68)
            pdf.set_text_color(255, 255, 255)
            pdf.cell(half, 7, "  Main concerns (top keywords)", fill=True, ln=False)
            pdf.set_xy(x_right, y_start + 8)
            pdf.set_text_color(*TEXT_DARK)
            pdf.set_font("Helvetica", "", 8.5)
            for rank, (word, cnt) in enumerate(top_excess, 1):
                pdf.set_x(x_right)
                pct = cnt / max(len(excess_s), 1) * 100
                pdf.cell(half, 5.5, f"  {rank:>2}. {word.capitalize():<22}  {cnt} mentions ({pct:.0f}%)", ln=True)

    # ── Section: Compare Mode ─────────────────────────────────────────────────
    cmp_dim = opts.get("cmp_dimension")
    cmp_a = opts.get("cmp_val_a")
    cmp_b = opts.get("cmp_val_b")
    
    if cmp_dim and cmp_a and cmp_b:
        if any(opts.get(k, True) for k in ["vis_compare_summary", "vis_compare_quality", "vis_compare_software", "vis_compare_sat"]):
            pdf.add_page()
            pdf.section_title(_next_sec(f"Compare Mode: {cmp_a} vs {cmp_b}"))
            
            from page_modules.page10_compare import _filter_group, _quality_scores, _software_scores
            df_a = _filter_group(merged, cmp_dim, cmp_a)
            df_b = _filter_group(merged, cmp_dim, cmp_b)
            n_a, n_b = len(df_a), len(df_b)
            
            if n_a > 0 and n_b > 0:
                sat_a = df_a[SATISFACTION_COL].mean()
                sat_b = df_b[SATISFACTION_COL].mean()
                sup_a = df_a[SUPPORT_COL].mean()
                sup_b = df_b[SUPPORT_COL].mean()
                
                label_a = f"{cmp_a} (n={n_a})"
                label_b = f"{cmp_b} (n={n_b})"
                
                if opts.get("vis_compare_summary", True):
                    pdf.set_font("Helvetica", "B", 9)
                    pdf.set_fill_color(*primary_rgb)
                    pdf.set_text_color(255, 255, 255)
                    pdf.cell(CONTENT_W / 3, 7, "  Metric", fill=True)
                    pdf.cell(CONTENT_W / 3, 7, label_a, fill=True, align="C")
                    pdf.cell(CONTENT_W / 3, 7, label_b, fill=True, align="C", ln=True)
                    
                    pdf.set_text_color(*TEXT_DARK)
                    pdf.set_font("Helvetica", "", 8.5)
                    metrics = [
                        ("Avg Satisfaction Score", f"{sat_a:.2f}", f"{sat_b:.2f}"),
                        ("Avg Support Score", f"{sup_a:.2f}", f"{sup_b:.2f}"),
                        ("Respondents count", f"{n_a}", f"{n_b}")
                    ]
                    for i, (m_lbl, val_a, val_b) in enumerate(metrics):
                        fill = i % 2 == 0
                        pdf.set_fill_color(*(light_rgb if fill else (255, 255, 255)))
                        pdf.cell(CONTENT_W / 3, 6, "  " + m_lbl, fill=fill)
                        pdf.cell(CONTENT_W / 3, 6, val_a, fill=fill, align="C")
                        pdf.cell(CONTENT_W / 3, 6, val_b, fill=fill, align="C", ln=True)
                    pdf.ln(4)
                    
                if opts.get("vis_compare_quality", True):
                    qa = _quality_scores(df_a)
                    qb = _quality_scores(df_b)
                    if not qa.empty and not qb.empty:
                        q_merged = qa.merge(qb, on="Dimension", suffixes=("_a", "_b"))
                        dims_q = q_merged["Dimension"].tolist()
                        scores_qa = q_merged["Score_a"].tolist()
                        scores_qb = q_merged["Score_b"].tolist()
                        png_q = _chart_compare_bar("Data Quality Comparison", dims_q, scores_qa, scores_qb, label_a, label_b, primary_hex)
                        pdf.embed_png(png_q, w=CONTENT_W)
                        pdf.ln(2)
                        
                if opts.get("vis_compare_software", True):
                    sa = _software_scores(df_a)
                    sb = _software_scores(df_b)
                    if not sa.empty and not sb.empty:
                        s_merged = sa.merge(sb, on="Dimension", suffixes=("_a", "_b"))
                        dims_s = s_merged["Dimension"].tolist()
                        scores_sa = s_merged["Score_a"].tolist()
                        scores_sb = s_merged["Score_b"].tolist()
                        png_s = _chart_compare_bar("Software Capabilities Comparison", dims_s, scores_sa, scores_sb, label_a, label_b, primary_hex)
                        pdf.embed_png(png_s, w=CONTENT_W)
                        pdf.ln(2)
                        
                if opts.get("vis_compare_sat", True):
                    sat_dims = ["Overall Satisfaction", "Support Rating"]
                    scores_sat_a = [sat_a, sup_a]
                    scores_sat_b = [sat_b, sup_b]
                    png_sat_cmp = _chart_compare_bar("Satisfaction & Support Comparison", sat_dims, scores_sat_a, scores_sat_b, label_a, label_b, primary_hex)
                    pdf.embed_png(png_sat_cmp, w=CONTENT_W)
                    pdf.ln(4)

    # Let's add the chi-square and helper data fetching functions inside report_export.py scope
    # (they will be defined globally or we can place them here)
    return bytes(pdf.output())


def _get_chisquare_data(df: pd.DataFrame) -> list:
    from scipy import stats
    cat_cols = {
        "Rank":           RANK_COL,
        "Tenure":         TENURE_COL,
        "Correction %":   CORRECTION_COL,
        "Training":       "Have you attended any of the JiBe PMS training sessions? [either on Training Centre/ Online session with PMS Team/ Marineflix Videos]",
    }

    pairs = [
        ("Rank",         "Tenure"),
        ("Rank",         "Correction %"),
        ("Rank",         "Training"),
        ("Tenure",       "Correction %"),
        ("Tenure",       "Training"),
        ("Training",     "Correction %"),
    ]

    chi_rows = []
    for a, b in pairs:
        if a not in cat_cols or b not in cat_cols:
            continue
        col_a = cat_cols[a]
        col_b = cat_cols[b]
        if col_a not in df.columns or col_b not in df.columns:
            continue
        sub = df[[col_a, col_b]].dropna()
        if len(sub) < 5:
            continue
        ct = pd.crosstab(sub[col_a], sub[col_b])
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "No"))
            chi_rows.append({
                "var_a": a, "var_b": b,
                "chi2": f"{chi2:.2f}", "df": str(dof),
                "p_val": f"{p:.4f}", "sig": sig
            })
        except Exception:
            pass
    return chi_rows


def generate_excel(data: dict) -> bytes:
    """Return an Excel workbook with the filtered survey data and summary stats."""
    df     = data["survey"]
    merged = data["merged"]

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Survey Responses", index=False)

        fleet_cols = ["Vessel Name", "Type", "Division Code", "PMS Go-Live"]
        avail_fleet = [c for c in fleet_cols if c in merged.columns]
        survey_cols = [c for c in df.columns if c in merged.columns]
        cols_out = list(dict.fromkeys(avail_fleet + survey_cols))  # deduplicate
        merged[[c for c in cols_out if c in merged.columns]].to_excel(
            writer, sheet_name="With Fleet Info", index=False)

        issues_df = data["issues"].reset_index()
        issues_df.columns = ["Issue", "Count"]
        issues_df["% of respondents"] = (issues_df["Count"] / len(df) * 100).round(1)
        issues_df.to_excel(writer, sheet_name="Data Issues", index=False)

        imp_df = data["improvements"].reset_index()
        imp_df.columns = ["Improvement", "Votes"]
        imp_df["% of respondents"] = (imp_df["Votes"] / len(df) * 100).round(1)
        imp_df.to_excel(writer, sheet_name="Improvements", index=False)

    return output.getvalue()
