"""
opm_table1.py  –  Table 1: baseline characteristics of the primary analytic cohort,
                   stratified by M_3class.

Columns: Overall | M0 | M1d | M2d | p-value

Continuous variables: median (IQR); Kruskal–Wallis test across three M classes.
Categorical variables: n (%); chi-squared test.

Outputs
-------
- Printed table to stdout
- Table1_baseline_characteristics.docx  (Word document ready for manuscript)

Usage
-----
    python opm_table1.py

Requires opm_core.py, opm_word_utils.py, and db2_bk200260525.xlsx.
"""

import numpy as np
import pandas as pd
from scipy.stats import kruskal, chi2_contingency
from lifelines import KaplanMeierFitter

from opm_core import load_cohort
from opm_word_utils import save_table_to_word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_median_iqr(series: pd.Series) -> str:
    s = series.dropna()
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"


def classify_classic7(site: str) -> str:
    s = str(site).upper()
    if s.startswith("B") and "PY" in s:
        return "Bulbar-pyramidal"
    if s.startswith("B"):
        return "Bulbar"
    if "FA" in s:
        return "Flail-arm"
    if "FL" in s:
        return "Flail-leg"
    if "RE" in s:
        return "Respiratory"
    if "PY" in s:
        return "Pyramidal"
    if s.startswith("S"):
        return "Classic spinal"
    return "Other"


# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------

cohort = load_cohort()
cohort["classic"] = cohort["site of onset1"].apply(classify_classic7)

M_CLASSES = ["M0", "M1d", "M2d"]
groups     = {g: cohort[cohort["OPM_M_3class"] == g] for g in M_CLASSES}


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------

def row_continuous(label: str, col: str) -> str:
    vals_all = cohort[col].dropna()
    cells    = [fmt_median_iqr(vals_all)]
    for g in M_CLASSES:
        cells.append(fmt_median_iqr(groups[g][col]))
    arrays = [groups[g][col].dropna() for g in M_CLASSES]
    try:
        _, pval = kruskal(*arrays)
    except Exception:
        pval = float("nan")
    return _format_row(label, cells, pval)


def row_categorical(label: str, col: str, value: str) -> str:
    def pct_str(df: pd.DataFrame) -> str:
        n = (df[col] == value).sum()
        return f"{n} ({100 * n / len(df):.1f}%)"
    cells = [pct_str(cohort)] + [pct_str(groups[g]) for g in M_CLASSES]
    ct = pd.crosstab(cohort["OPM_M_3class"], (cohort[col].astype(str) == value))
    try:
        _, pval, _, _ = chi2_contingency(ct)
    except Exception:
        pval = float("nan")
    return _format_row(label, cells, pval)


def row_events(label: str) -> str:
    def pct_str(df: pd.DataFrame) -> str:
        n = int(df["event"].sum())
        return f"{n} ({100 * n / len(df):.1f}%)"
    cells = [pct_str(cohort)] + [pct_str(groups[g]) for g in M_CLASSES]
    return _format_row(label, cells, pval=None)


def _format_row(label: str, cells: list[str], pval) -> str:
    pval_str = f"p = {pval:.3g}" if pval is not None and not np.isnan(pval) else ""
    col_w = 20
    line  = f"{label:<28}| {cells[0]:<{col_w}}|"
    for c in cells[1:]:
        line += f" {c:<{col_w}}|"
    line += f" {pval_str}"
    return line


# ---------------------------------------------------------------------------
# Print table to stdout
# ---------------------------------------------------------------------------

N_TOTAL = len(cohort)
COL_W   = 20
N_EACH  = {g: len(groups[g]) for g in M_CLASSES}

print("=" * 130)
print(f"TABLE 1 — Baseline characteristics of the primary analytic cohort (N = {N_TOTAL})")
print("=" * 130)

header = f"{'Variable':<28}| {'Overall N=' + str(N_TOTAL):<{COL_W}}|"
for g in M_CLASSES:
    header += f" {g + ' (N=' + str(N_EACH[g]) + ')':<{COL_W}}|"
print(header)
print("-" * 130)

# Demographics
print(row_categorical("Male sex, n (%)",       "sexU",   "M"))
print(row_continuous( "Age at onset (years)",  "age"))
print(row_continuous( "Survival (months)",     "surv_mo"))
print(row_events(     "Events (death/trach.), n (%)"))
print(row_continuous( "Propagation P1(n) (months)", "P_mo"))

# OPM_O distribution
print("-" * 130)
print("Onset region OPM_O:")
for o_cat in sorted(cohort["OPM_O"].dropna().unique()):
    print(row_categorical(f"  {o_cat}", "OPM_O", o_cat))

# Classical phenotype distribution
print("-" * 130)
print("Classical phenotype:")
for pheno in [
    "Bulbar", "Classic spinal", "Flail-arm", "Flail-leg",
    "Respiratory", "Bulbar-pyramidal", "Pyramidal",
]:
    print(row_categorical(f"  {pheno}", "classic", pheno))

# Staging (King's)
print("-" * 130)
print("King's stage at baseline:")
for stage in sorted(cohort["KINGS_use"].dropna().unique()):
    print(row_categorical(f"  Stage {int(stage)}", "KINGS_use", stage))

print("=" * 130)

# ---------------------------------------------------------------------------
# Kaplan-Meier medians by M_3class (reported in Results text)
# ---------------------------------------------------------------------------

print("\nKaplan-Meier median survival by M_3class (from symptom onset):")
kmf = KaplanMeierFitter()
km_medians = {}
for g in M_CLASSES:
    sub = groups[g]
    kmf.fit(sub["surv_mo"], sub["event"])
    km_medians[g] = kmf.median_survival_time_
    print(f"  {g}:  median = {km_medians[g]:.1f} months")


# ---------------------------------------------------------------------------
# Helper: build structured data for each section
# ---------------------------------------------------------------------------

def _continuous_cells(col: str):
    vals_all = cohort[col].dropna()
    cells = [fmt_median_iqr(vals_all)]
    for g in M_CLASSES:
        cells.append(fmt_median_iqr(groups[g][col]))
    arrays = [groups[g][col].dropna() for g in M_CLASSES]
    try:
        _, pval = kruskal(*arrays)
        cells.append(f"{pval:.3g}")
    except Exception:
        cells.append("")
    return cells


def _categorical_cells(col: str, value: str):
    def pct_str(df):
        n = (df[col] == value).sum()
        return f"{n} ({100 * n / len(df):.1f}%)"
    cells = [pct_str(cohort)] + [pct_str(groups[g]) for g in M_CLASSES]
    ct = pd.crosstab(cohort["OPM_M_3class"], (cohort[col].astype(str) == value))
    try:
        _, pval, _, _ = chi2_contingency(ct)
        cells.append(f"{pval:.3g}")
    except Exception:
        cells.append("")
    return cells


def _events_cells():
    def pct_str(df):
        n = int(df["event"].sum())
        return f"{n} ({100 * n / len(df):.1f}%)"
    return [pct_str(cohort)] + [pct_str(groups[g]) for g in M_CLASSES] + [""]


# ---------------------------------------------------------------------------
# Build Word table rows
# ---------------------------------------------------------------------------

word_headers = [
    "Variable",
    f"Overall (N = {N_TOTAL})",
    f"M0 (N = {N_EACH['M0']})",
    f"M1d (N = {N_EACH['M1d']})",
    f"M2d (N = {N_EACH['M2d']})",
    "p-value",
]

word_rows = []

# Section header
word_rows.append(["Demographics"] + [""] * 5)

word_rows.append(["Male sex, n (%)"]           + _categorical_cells("sexU",   "M"))
word_rows.append(["Age at onset, years"]       + _continuous_cells("age"))
word_rows.append(["Survival, months"]          + _continuous_cells("surv_mo"))
word_rows.append(["Events (death/trach.), n (%)"] + _events_cells())
word_rows.append(["Propagation P1(n), months"] + _continuous_cells("P_mo"))

word_rows.append(["Onset region (OPM_O)"] + [""] * 5)
for o_cat in sorted(cohort["OPM_O"].dropna().unique()):
    word_rows.append([f"  {o_cat}"] + _categorical_cells("OPM_O", o_cat))

word_rows.append(["Classical phenotype"] + [""] * 5)
for pheno in [
    "Bulbar", "Classic spinal", "Flail-arm", "Flail-leg",
    "Respiratory", "Bulbar-pyramidal", "Pyramidal",
]:
    word_rows.append([f"  {pheno}"] + _categorical_cells("classic", pheno))

word_rows.append(["King's stage at baseline"] + [""] * 5)
for stage in sorted(cohort["KINGS_use"].dropna().unique()):
    word_rows.append([f"  Stage {int(stage)}"] + _categorical_cells("KINGS_use", stage))

# ---------------------------------------------------------------------------
# Save Word document
# ---------------------------------------------------------------------------

CAPTION = (
    f"Baseline characteristics of the primary analytic cohort (N = {N_TOTAL}) "
    "stratified by motor-neuron pattern class (M_3class). "
    "Continuous variables are presented as median (IQR); Kruskal–Wallis test. "
    "Categorical variables are presented as n (%); chi-squared test. "
    "M0 = balanced upper and lower motor-neuron involvement; "
    "M1d = predominant upper motor-neuron involvement (definite/probable); "
    "M2d = predominant lower motor-neuron involvement (definite/probable). "
    "Events = death or permanent invasive ventilation (tracheostomy)."
)

FOOTNOTES = [
    "Abbreviations: IQR = interquartile range; OPM = ALS Onco-Propagation-Motor pattern system v3.3.",
    f"Kaplan-Meier median survival from symptom onset: "
    f"M0 = {km_medians['M0']:.1f} mo, M1d = {km_medians['M1d']:.1f} mo, M2d = {km_medians['M2d']:.1f} mo.",
    "p-values: Kruskal-Wallis for continuous variables, chi-squared for categorical variables (3-group comparison).",
]

save_table_to_word(
    title    = "Table 1. Baseline characteristics of the primary analytic cohort",
    caption  = CAPTION,
    headers  = word_headers,
    rows     = word_rows,
    filename = "Table1_baseline_characteristics.docx",
    footnotes = FOOTNOTES,
)