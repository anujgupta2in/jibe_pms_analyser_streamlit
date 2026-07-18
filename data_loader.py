"""
Data loading and preprocessing for JiBe PMS Survey Analyser.
All heavy parsing is cached so pages stay fast.
"""

import io
import re
import pandas as pd
import streamlit as st

# ── Column aliases ────────────────────────────────────────────────────────────
VESSEL_COL   = "Select your Vessel Name"
RANK_COL     = "Select your Rank"
TENURE_COL   = "How long have you been using JiBe PMS?"
CORRECTION_COL = "Approximately what percentage of PMS jobs require correction?"
CHANGE_FREQ_COL = "How often do you request PMS changes from the office?"
PERF_COL     = "How often do you experience slow system performance?"
PARTS_DELAY_COL = "How often do PMS missing parts delay your actual maintenance / raising requisitions?"
TRAINING_COL = "Have you attended any of the JiBe PMS training sessions? [either on Training Centre/ Online session with PMS Team/ Marineflix Videos]"
SATISFACTION_COL = "Overall, how satisfied are you with JiBe PMS?"
SUPPORT_COL  = "Rate our Support Team Response Time and Resolution Quality"
ISSUES_COL   = "Which data issues do you frequently experience? (Select all that apply)"
IMPROVEMENTS_COL = "Which THREE improvements would create the biggest benefit for your daily work? (Select exactly 3)"
SUPPORT_RESOURCES_COL = "Which of the following JiBe support resources are you aware of? (Select all that apply)"
LIKED_COL    = "What did you like and find good about the JiBe PMS system?"
EXCESSIVE_COL = "Which specific PMS jobs or routines do you feel are excessive or repetitive compared to our legacy system and could be optimized or removed?"

QUALITY_COLS = {
    "Question.Machinery Hierarchy / Equipment Structure": "Machinery Hierarchy",
    "Question.Job Descriptions & Maintenance Instructions": "Job Descriptions",
    "Question.Job Frequency & Running Hour Intervals Configuration": "Job Frequency",
    "Question.Spare Parts Mapping to Machinery": "Spare Parts Mapping",
    "Question.Critical Equipment & Jobs Marking": "Critical Equipment",
}

SOFTWARE_COLS = {
    "Rate your experience with the following JiBe PMS software capabilities:.Ease of locating data": "Ease of Locating Data",
    "Rate your experience with the following JiBe PMS software capabilities:.Job Reporting & Updating Running Hours": "Job Reporting",
    "Rate your experience with the following JiBe PMS software capabilities:.Task Reporting": "Task Reporting",
    "Rate your experience with the following JiBe PMS software capabilities:.Viewing Job History": "Job History",
    "Rate your experience with the following JiBe PMS software capabilities:.Office Approval Workflow": "Office Approvals",
    "Rate your experience with the following JiBe PMS software capabilities:.Job Postponements": "Job Postponements",
    "Rate your experience with the following JiBe PMS software capabilities:.Job Verifications": "Job Verifications",
    "Rate your experience with the following JiBe PMS software capabilities:.Spare Parts Consumption Logging": "Spare Parts Logging",
    "Rate your experience with the following JiBe PMS software capabilities:.e-form reporting": "e-form Reporting",
}

ORDINAL_MAPS = {
    "quality": {"Poor": 1, "Needs Improvement": 2, "Acceptable": 3},
    "software": {"Poor": 1, "Needs Improvement": 2, "Good": 3},
    "correction": {"Less than 5%": 1, "5–10%": 2, "10–20%": 3, "More than 20%": 4},
    "change_freq": {"Never": 1, "Rarely [Quarterly]": 2, "Sometimes [Monthly]": 3, "Often [Every 15 Days]": 4},
    "performance": {"Rarely": 1, "Occasionally": 2, "Monthly": 3, "Weekly": 4, "Daily": 5, "Always": 6},
    "tenure": {"Less than 6 Months": 1, "6 Months – 1 Year": 2, "1 - 2 Years": 3, "More than 2 Years": 4},
}

COLOUR_MAP   = {"Good": "#22c55e", "Acceptable": "#22c55e",
                "Needs Improvement": "#f59e0b", "Poor": "#ef4444"}
PLOTLY_THEME = "plotly_white"
PRIMARY      = "#1e40af"
SEQ_COLOURS  = ["#1e40af", "#0ea5e9", "#22c55e", "#f59e0b", "#ef4444",
                "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#64748b"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_multiselect(series: pd.Series) -> pd.Series:
    """Split semicolon-separated multi-select answers into lists."""
    return series.dropna().apply(
        lambda x: [i.strip() for i in str(x).split(";") if i.strip()]
    )


def _flatten_counts(series: pd.Series) -> pd.Series:
    """Explode a series of lists and return value counts."""
    return pd.Series(
        [item for sublist in series for item in sublist]
    ).value_counts()


@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> dict:
    """
    Load Sheet1 and Fleet, clean, merge, and return a dict of dataframes.
    Accepts raw bytes so Streamlit's caching works with uploaded files.
    """
    buf = io.BytesIO(file_bytes)

    # ── Sheet 1 ──
    df = pd.read_excel(buf, sheet_name="Sheet1")
    df = df.dropna(how="all")

    # ── Fleet ──
    buf.seek(0)
    raw_fleet = pd.read_excel(buf, sheet_name="Fleet", header=None)
    # Row 0 is blank, Row 1 is the header
    header_row = None
    for i, row in raw_fleet.iterrows():
        if "Vessel Name" in row.values:
            header_row = i
            break
    fleet = raw_fleet.iloc[header_row + 1:].copy()
    fleet.columns = ["Vessel Name", "Type", "Division Code", "PMS Go-Live", "Takeover Date"]
    fleet = fleet.dropna(subset=["Vessel Name"])
    fleet = fleet[fleet["Vessel Name"] != "Vessel Name"]   # drop any duplicate header rows
    fleet["PMS Go-Live"]    = pd.to_datetime(fleet["PMS Go-Live"],    errors="coerce")
    fleet["Takeover Date"]  = pd.to_datetime(fleet["Takeover Date"],  errors="coerce")
    fleet = fleet.reset_index(drop=True)

    # ── Merge ──
    merged = df.merge(
        fleet, left_on=VESSEL_COL, right_on="Vessel Name", how="left"
    )

    # ── Multi-select parsed counts ──
    issues_counts       = _flatten_counts(_parse_multiselect(df[ISSUES_COL]))
    improvements_counts = _flatten_counts(_parse_multiselect(df[IMPROVEMENTS_COL]))
    support_res_counts  = _flatten_counts(_parse_multiselect(df[SUPPORT_RESOURCES_COL]))

    # ── Ordinal-encoded frame for correlations ──
    enc = pd.DataFrame()
    for orig, alias in QUALITY_COLS.items():
        if orig in df.columns:
            enc[alias] = df[orig].map(ORDINAL_MAPS["quality"])
    for orig, alias in SOFTWARE_COLS.items():
        if orig in df.columns:
            enc[alias] = df[orig].map(ORDINAL_MAPS["software"])
    enc["Correction %"]     = df[CORRECTION_COL].map(ORDINAL_MAPS["correction"])
    enc["Change Frequency"] = df[CHANGE_FREQ_COL].map(ORDINAL_MAPS["change_freq"])
    enc["Sys Performance"]  = df[PERF_COL].map(ORDINAL_MAPS["performance"])
    enc["PMS Tenure"]       = df[TENURE_COL].map(ORDINAL_MAPS["tenure"])
    enc["Satisfaction"]     = df[SATISFACTION_COL]
    enc["Support Rating"]   = df[SUPPORT_COL]

    return {
        "survey":        df,
        "fleet":         fleet,
        "merged":        merged,
        "issues":        issues_counts,
        "improvements":  improvements_counts,
        "support_res":   support_res_counts,
        "encoded":       enc,
    }


def apply_filters(data: dict, filters: dict) -> dict:
    """
    Apply sidebar filter selections to a loaded data dict.

    filters keys (all optional, empty list = no filter):
        rank         – list[str]  values of RANK_COL
        vessel_type  – list[str]  values of fleet "Type"
        division     – list[str]  values of fleet "Division Code"
        tenure       – list[str]  values of TENURE_COL

    Returns a new dict with filtered survey/merged and recomputed
    issues, improvements, support_res, and encoded.
    Fleet is returned unchanged (it is a vessel registry, not per-respondent).
    """
    merged = data["merged"]
    survey = data["survey"]

    # Build a boolean mask over respondent rows
    mask = pd.Series(True, index=merged.index)

    if filters.get("rank"):
        mask &= merged[RANK_COL].isin(filters["rank"])
    if filters.get("vessel_type"):
        mask &= merged["Type"].isin(filters["vessel_type"])
    if filters.get("division"):
        mask &= merged["Division Code"].isin(filters["division"])
    if filters.get("tenure"):
        mask &= merged[TENURE_COL].isin(filters["tenure"])

    filtered_merged = merged[mask].reset_index(drop=True)
    # survey and merged share the same positional alignment (left join)
    filtered_survey = survey[mask.values].reset_index(drop=True)

    # Recompute multi-select counts from filtered survey
    issues_counts       = _flatten_counts(_parse_multiselect(filtered_survey[ISSUES_COL]))
    improvements_counts = _flatten_counts(_parse_multiselect(filtered_survey[IMPROVEMENTS_COL]))
    support_res_counts  = _flatten_counts(_parse_multiselect(filtered_survey[SUPPORT_RESOURCES_COL]))

    # Recompute ordinal-encoded frame for correlations
    enc = pd.DataFrame()
    for orig, alias in QUALITY_COLS.items():
        if orig in filtered_survey.columns:
            enc[alias] = filtered_survey[orig].map(ORDINAL_MAPS["quality"])
    for orig, alias in SOFTWARE_COLS.items():
        if orig in filtered_survey.columns:
            enc[alias] = filtered_survey[orig].map(ORDINAL_MAPS["software"])
    enc["Correction %"]     = filtered_survey[CORRECTION_COL].map(ORDINAL_MAPS["correction"])
    enc["Change Frequency"] = filtered_survey[CHANGE_FREQ_COL].map(ORDINAL_MAPS["change_freq"])
    enc["Sys Performance"]  = filtered_survey[PERF_COL].map(ORDINAL_MAPS["performance"])
    enc["PMS Tenure"]       = filtered_survey[TENURE_COL].map(ORDINAL_MAPS["tenure"])
    enc["Satisfaction"]     = filtered_survey[SATISFACTION_COL]
    enc["Support Rating"]   = filtered_survey[SUPPORT_COL]

    return {
        "survey":       filtered_survey,
        "fleet":        data["fleet"],
        "merged":       filtered_merged,
        "issues":       issues_counts,
        "improvements": improvements_counts,
        "support_res":  support_res_counts,
        "encoded":      enc,
    }
