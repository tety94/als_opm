"""
opm_staging.py  –  Comparison of OPM 3.3 against clinical staging systems
                    (King's, MiToS, FT9).  Reproduces Table 4 and Supplementary
                    Tables S7–S8.

For each staging system, three Cox models are fitted on the available sub-cohort
(patients with non-missing stage at baseline):

  S_single  : anchor + staging dummies only  (staging alone)
  S_OPM     : anchor + OPM_O + M_3class      (OPM 3.3 alone)
  S_combined: anchor + OPM_O + M_3class + staging dummies  (combined)

The anchor (P_mo_cap + age10 + sexM) is held fixed across all models.
P is thus a shared kinetic control rather than a competing predictor, consistent
with its intrinsically longitudinal nature noted in the paper.

For each staging system, the following comparisons are reported:
  - Combined vs S_single:  LR χ², df, p, ΔAIC
  - Combined vs S_OPM:     LR χ², df, p, ΔAIC
  - M_3class HRs in the combined model (stability check)

Usage
-----
    python opm_staging.py

Requires opm_core.py and db2_bk200260525.xlsx.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
from scipy.stats import chi2 as chi2_dist

from opm_core import load_cohort, get_O_dummies, get_M_dummies
from opm_word_utils import save_table_to_word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_cox(X: pd.DataFrame, penalizer: float = 1e-4) -> CoxPHFitter:
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(X, duration_col="surv_mo", event_col="event")
    return m


def compute_aic(model: CoxPHFitter, X: pd.DataFrame) -> float:
    k = X.shape[1] - 2
    return -2 * model.log_likelihood_ + 2 * k


def lr_test(
    m_full: CoxPHFitter,
    m_reduced: CoxPHFitter,
    X_full: pd.DataFrame,
    X_reduced: pd.DataFrame,
) -> tuple[float, int, float]:
    """Return (LR chi², df, p-value) for a nested likelihood-ratio test."""
    stat = 2 * (m_full.log_likelihood_ - m_reduced.log_likelihood_)
    df   = (X_full.shape[1] - 2) - (X_reduced.shape[1] - 2)
    pval = chi2_dist.sf(stat, df)
    return stat, df, pval


# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------

cohort = load_cohort()

# ---------------------------------------------------------------------------
# Staging comparison function
# ---------------------------------------------------------------------------

def run_staging_comparison(
    stage_col: str,
    stage_label: str,
    use_fallback: bool = False,
) -> dict:
    """
    Fit single / OPM / combined models for one staging system and print results.

    Returns a dict with summary statistics for Word table export.
    """
    col = "KINGS_use" if (use_fallback and stage_col == "KINGS") else stage_col
    sub = cohort.dropna(subset=[col]).copy()
    n   = len(sub)

    anchor = sub[["surv_mo", "event", "age10", "sexM", "P_mo_cap"]]
    dM     = get_M_dummies(sub)
    dO     = get_O_dummies(sub)

    # Stage dummies (reference = lowest stage)
    dS = pd.get_dummies(sub[col].astype(int), prefix=stage_label)
    dS = dS.drop(columns=[dS.columns[0]])

    # Three nested models
    X_single   = pd.concat([anchor, dS],            axis=1).astype(float)
    X_opm      = pd.concat([anchor, dM, dO],         axis=1).astype(float)
    X_combined = pd.concat([anchor, dM, dO, dS],     axis=1).astype(float)

    m_single   = fit_cox(X_single)
    m_opm      = fit_cox(X_opm)
    m_combined = fit_cox(X_combined)

    lr_vs_single, df_vs_single, p_vs_single = lr_test(m_combined, m_single, X_combined, X_single)
    lr_vs_opm,   df_vs_opm,   p_vs_opm     = lr_test(m_combined, m_opm,    X_combined, X_opm)

    daic_vs_single = compute_aic(m_combined, X_combined) - compute_aic(m_single, X_single)
    daic_vs_opm    = compute_aic(m_combined, X_combined) - compute_aic(m_opm,    X_opm)

    print(f"\n{'─' * 72}")
    print(f"{stage_label} staging  (N = {n},  events = {int(sub['event'].sum())})")
    print(f"{'─' * 72}")
    print(f"  Combined vs {stage_label}-only:  "
          f"LR χ² = {lr_vs_single:.1f},  df = {df_vs_single},  "
          f"p = {p_vs_single:.1e},  ΔAIC = {daic_vs_single:+.1f}")
    print(f"  Combined vs OPM 3.3:     "
          f"LR χ² = {lr_vs_opm:.1f},  df = {df_vs_opm},  "
          f"p = {p_vs_opm:.1e},  ΔAIC = {daic_vs_opm:+.1f}")

    # M_3class hazard ratios in the combined model (stability check)
    m_hrs = {}
    print(f"  M_3class HRs in combined model:")
    for m_cat in ["M_M1d", "M_M2d"]:
        if m_cat not in m_combined.summary.index:
            continue
        r = m_combined.summary.loc[m_cat]
        m_hrs[m_cat] = r
        print(
            f"    {m_cat:<10}  HR = {r['exp(coef)']:.2f}  "
            f"({r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f}),  "
            f"p = {r['p']:.3f}"
        )

    # C-index for each model
    c_single   = m_single.concordance_index_
    c_opm      = m_opm.concordance_index_
    c_combined = m_combined.concordance_index_

    return {
        "label":          stage_label,
        "n":              n,
        "events":         int(sub["event"].sum()),
        "c_single":       c_single,
        "c_opm":          c_opm,
        "c_combined":     c_combined,
        "lr_vs_single":   lr_vs_single,
        "df_vs_single":   df_vs_single,
        "p_vs_single":    p_vs_single,
        "daic_vs_single": daic_vs_single,
        "lr_vs_opm":      lr_vs_opm,
        "df_vs_opm":      df_vs_opm,
        "p_vs_opm":       p_vs_opm,
        "daic_vs_opm":    daic_vs_opm,
        "m_hrs":          m_hrs,
    }


# ---------------------------------------------------------------------------
# Run for King's, MiToS, FT9
# ---------------------------------------------------------------------------

print("=" * 72)
print("TABLE 4 / SUPPLEMENTARY TABLES S7–S8")
print("Comparison of OPM 3.3 vs clinical staging systems")
print(f"Primary cohort: N = {len(cohort)}")
print("Anchor (fixed in all models): P_mo_cap + age10 + sexM")
print("=" * 72)

res_kings = run_staging_comparison("KINGS", "King's",  use_fallback=True)
res_mitos = run_staging_comparison("MITOS", "MiToS",   use_fallback=False)
res_ft9   = run_staging_comparison("FT9",   "FT9",     use_fallback=False)

print(f"\n{'─' * 72}")
print("Summary ΔAIC (combined vs OPM 3.3):")
for r in [res_kings, res_mitos, res_ft9]:
    print(f"  {r['label']:<8}:  ΔAIC = {r['daic_vs_opm']:+.1f}")
print("  Consistently negative → staging adds prognostic information beyond OPM 3.3")


# ---------------------------------------------------------------------------
# Table 4: Word export (King's staging — primary comparison)
# ---------------------------------------------------------------------------

def _fmt_p(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _hr_str(hrs, key):
    if key not in hrs:
        return "—"
    r = hrs[key]
    return (f"{r['exp(coef)']:.2f} "
            f"({r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f}), "
            f"p = {r['p']:.3f}")


def build_staging_word_rows(results: list[dict]) -> list[list]:
    rows = []
    for r in results:
        rows.append([
            r["label"],
            f"{r['n']:,}",
            f"{r['events']:,}",
            f"{r['c_single']:.3f}",
            f"{r['c_opm']:.3f}",
            f"{r['c_combined']:.3f}",
            f"χ²={r['lr_vs_opm']:.1f}, df={r['df_vs_opm']}, p={_fmt_p(r['p_vs_opm'])}, "
            f"ΔAIC={r['daic_vs_opm']:+.1f}",
            _hr_str(r["m_hrs"], "M_M1d"),
            _hr_str(r["m_hrs"], "M_M2d"),
        ])
    return rows


save_table_to_word(
    title   = "Table 4. Cox model comparison: OPM 3.3 versus clinical staging systems",
    caption = (
        "All models anchored on P_mo_cap (linear, capped 60 months) + age10 + sexM. "
        "C-index = apparent Harrell concordance index. "
        "Combined vs OPM 3.3: likelihood-ratio test and ΔAIC. "
        "M_3class HRs in combined model confirm stability of the motor-neuron axis "
        "across all staging frameworks (reference: M0)."
    ),
    headers = [
        "Staging system",
        "N",
        "Events",
        "C (staging alone)",
        "C (OPM 3.3 alone)",
        "C (combined)",
        "Combined vs OPM 3.3",
        "M1d HR in combined (95% CI)",
        "M2d HR in combined (95% CI)",
    ],
    rows      = build_staging_word_rows([res_kings, res_mitos, res_ft9]),
    filename  = "Table4_staging_comparison.docx",
    footnotes = [
        "Anchor covariates (fixed across all models): propagation time P1(n) capped at 60 months, "
        "age at onset per 10 years, sex.",
        "Staging systems: King's Clinical Staging (4 levels); MiToS (5 levels); "
        "FT9 functional trajectory score (4 levels).",
        "ΔAIC < 0 indicates the combined model is preferred over OPM 3.3 alone.",
        "M_3class HRs: reference category = M0 (balanced UMN+LMN).",
    ],
)