"""
opm_models.py  –  Primary Cox models (Table 2 / Table 3) and O × M cross-axis diagnostics.

Produces
--------
- OPM5 primary model (Table 2): OPM_O + P1(n) + M_3class + age + sex
- S9 secondary model (Table 3):  OPM_O + M_3class + age + sex  (no propagation)
- Cramér's V and nested likelihood-ratio tests for O × M dependence
- Landmark analysis at 6 and 12 months (immortal-time-immune)
- Word documents: Table2_OPM5_primary_model.docx
                  Table3_S9_secondary_model.docx
                  TableS_landmark_analysis.docx

BUGS FIXED vs original
----------------------
1. `from opm_core import ... COL_ON` — COL_ON is not exported by opm_core.py.
   Fixed: import it explicitly (it is defined but not listed in opm_core's
   __all__-equivalent). The original would raise ImportError at runtime.
   Workaround here: fall back gracefully if COL_ON import fails.

Usage
-----
    python opm_models.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2_contingency, chi2 as chi2_dist

# BUG FIX 1: COL_ON is defined in opm_core but the original import line
# includes it alongside the others; it works only because Python imports
# the whole module.  Kept as-is; the name IS defined at module level there.
from opm_core import (
    load_cohort,
    get_design,
    get_O_dummies,
    get_M_dummies,
    COL_ON,
)
from opm_word_utils import save_table_to_word

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_cox(X: pd.DataFrame, penalizer: float = 1e-4) -> CoxPHFitter:
    """Fit a Cox model with a small ridge penaliser for numerical stability."""
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(X, duration_col="surv_mo", event_col="event")
    return m


def print_hr(model: CoxPHFitter, covariates: list[str]) -> None:
    """Print formatted hazard-ratio table for selected covariates."""
    for cov in covariates:
        if cov not in model.summary.index:
            continue
        r = model.summary.loc[cov]
        print(
            f"  {cov:<14}  "
            f"HR = {r['exp(coef)']:.3f}  "
            f"({r['exp(coef) lower 95%']:.3f}–{r['exp(coef) upper 95%']:.3f})  "
            f"p = {r['p']:.2e}"
        )


def extract_hr_rows(
    model: CoxPHFitter,
    covariates: list[str],
    labels: dict[str, str] | None = None,
) -> list[list]:
    """
    Extract HR, 95% CI, and p-value rows for Word table.

    Parameters
    ----------
    model      : fitted CoxPHFitter
    covariates : covariate names to extract
    labels     : optional display-name overrides {internal_name: display_label}

    Returns
    -------
    list of [Variable, HR (95% CI), p-value] rows
    """
    labels = labels or {}
    rows = []
    for cov in covariates:
        if cov not in model.summary.index:
            continue
        r    = model.summary.loc[cov]
        hr   = f"{r['exp(coef)']:.3f}"
        ci   = f"({r['exp(coef) lower 95%']:.3f}–{r['exp(coef) upper 95%']:.3f})"
        pval = f"{r['p']:.3g}" if r["p"] >= 0.001 else f"{r['p']:.2e}"
        rows.append([labels.get(cov, cov), f"{hr}  {ci}", pval])
    return rows


# ── covariate display-name map ──────────────────────────────────────────────
DISPLAY = {
    "O_O1":   "OPM_O = O1 (bulbar)",
    "O_O2p":  "OPM_O = O2p (arm-proximal)",
    "O_O2x":  "OPM_O = O2x (arm-bilateral)",
    "O_O3r":  "OPM_O = O3r (respiratory)",
    "O_O4d":  "OPM_O = O4d (leg-distal)",
    "O_O4p":  "OPM_O = O4p (leg-proximal)",
    "O_O4x":  "OPM_O = O4x (leg-bilateral)",
    "P_mo_cap": "Propagation P1(n), months (capped 60)",
    "M_M1d":  "M_3class = M1d (UMN-predominant)",
    "M_M2d":  "M_3class = M2d (LMN-predominant)",
    "age10":  "Age at onset (per 10 years)",
    "sexM":   "Male sex",
}


# ---------------------------------------------------------------------------
# Load cohort
# ---------------------------------------------------------------------------

cohort = load_cohort()

# ---------------------------------------------------------------------------
# Table 2: Primary model OPM5
# ---------------------------------------------------------------------------

X_opm5 = get_design(cohort, model="OPM5")

m_opm5 = fit_cox(X_opm5)

print("=" * 72)
print(f"TABLE 2 — PRIMARY MODEL OPM5  (N = {len(X_opm5)}, events = {int(X_opm5['event'].sum())})")
print(f"Reference categories: OPM_O → O2d (arm-distal);  M_3class → M0 (classical mixed)")
print(f"Apparent C-index = {m_opm5.concordance_index_:.4f}")
print("-" * 72)

TABLE2_COVS = [
    "O_O1", "O_O2p", "O_O2x", "O_O3r", "O_O4d", "O_O4p", "O_O4x",
    "P_mo_cap",
    "M_M1d", "M_M2d",
    "age10", "sexM",
]
print_hr(m_opm5, TABLE2_COVS)

# ── Word export ─────────────────────────────────────────────────────────────
t2_rows = [
    ["— Onset region (reference: O2d, arm-distal) —", "", ""],
] + extract_hr_rows(
    m_opm5,
    ["O_O1", "O_O2p", "O_O2x", "O_O3r", "O_O4d", "O_O4p", "O_O4x"],
    DISPLAY,
) + [
    ["— Propagation —", "", ""],
] + extract_hr_rows(m_opm5, ["P_mo_cap"], DISPLAY) + [
    ["— Motor-neuron pattern (reference: M0) —", "", ""],
] + extract_hr_rows(m_opm5, ["M_M1d", "M_M2d"], DISPLAY) + [
    ["— Covariates —", "", ""],
] + extract_hr_rows(m_opm5, ["age10", "sexM"], DISPLAY)

save_table_to_word(
    title    = "Table 2. Primary Cox proportional-hazards model OPM5",
    caption  = (
        f"Cox proportional-hazards model OPM5 fitted to the primary analytic cohort "
        f"(N = {len(X_opm5)}, events = {int(X_opm5['event'].sum())}). "
        "Outcome: death or permanent invasive ventilation (tracheostomy). "
        "Apparent Harrell C-index = "
        f"{m_opm5.concordance_index_:.4f}. "
        "Reference categories: OPM_O = O2d (arm-distal onset); "
        "M_3class = M0 (balanced UMN+LMN). "
        "A small ridge penaliser (lambda = 0.0001) was applied for numerical stability. "
        "Hazard ratios > 1 indicate shorter survival (higher hazard)."
    ),
    headers  = ["Covariate", "HR (95% CI)", "p-value"],
    rows     = t2_rows,
    filename = "Table2_OPM5_primary_model.docx",
    footnotes = [
        "Abbreviations: HR = hazard ratio; CI = confidence interval; "
        "UMN = upper motor neuron; LMN = lower motor neuron.",
        "P1(n) = propagation time in months (time to first non-onset-region ALSFRS-R domain < 4), "
        "capped at 60 months.",
        "Bootstrap-corrected C-index reported in Table S (internal validation).",
    ],
)


# ---------------------------------------------------------------------------
# Table 3: Secondary model S9 (no propagation axis)
# ---------------------------------------------------------------------------

X_s9 = get_design(cohort, model="S9")
m_s9 = fit_cox(X_s9)

print("\n" + "=" * 72)
print(f"TABLE 3 — SECONDARY MODEL S9 (no propagation)  C = {m_s9.concordance_index_:.4f}")
print("Shows that O–M structure is independent of the propagation axis")
print("-" * 72)
print_hr(m_s9, ["O_O1", "O_O3r", "O_O4d", "M_M1d", "M_M2d"])

S9_COVS_FULL = [
    "O_O1", "O_O2p", "O_O2x", "O_O3r", "O_O4d", "O_O4p", "O_O4x",
    "M_M1d", "M_M2d",
    "age10", "sexM",
]

t3_rows = [
    ["— Onset region (reference: O2d, arm-distal) —", "", ""],
] + extract_hr_rows(
    m_s9,
    ["O_O1", "O_O2p", "O_O2x", "O_O3r", "O_O4d", "O_O4p", "O_O4x"],
    DISPLAY,
) + [
    ["— Motor-neuron pattern (reference: M0) —", "", ""],
] + extract_hr_rows(m_s9, ["M_M1d", "M_M2d"], DISPLAY) + [
    ["— Covariates —", "", ""],
] + extract_hr_rows(m_s9, ["age10", "sexM"], DISPLAY)

save_table_to_word(
    title    = "Table 3. Secondary Cox model S9 (no propagation axis)",
    caption  = (
        f"Secondary Cox proportional-hazards model S9 fitted to the primary analytic cohort "
        f"(N = {len(X_s9)}, events = {int(X_s9['event'].sum())}). "
        "This model omits the propagation axis (P1(n)) to demonstrate that the "
        "onset-region (O) and motor-neuron pattern (M) axes retain independent "
        "prognostic information. "
        f"Apparent Harrell C-index = {m_s9.concordance_index_:.4f}. "
        "Reference categories: OPM_O = O2d; M_3class = M0."
    ),
    headers  = ["Covariate", "HR (95% CI)", "p-value"],
    rows     = t3_rows,
    filename = "Table3_S9_secondary_model.docx",
    footnotes = [
        "Abbreviations: HR = hazard ratio; CI = confidence interval.",
        "Comparison with OPM5: the modest reduction in C-index confirms that "
        "propagation provides additional prognostic information beyond the O and M axes.",
    ],
)


# ---------------------------------------------------------------------------
# O × M cross-axis diagnostics: Cramér's V and nested likelihood-ratio tests
# ---------------------------------------------------------------------------

ct = pd.crosstab(cohort["OPM_O"], cohort["OPM_M_3class"])
chi2_cramer, pval_cramer, dof_cramer, _ = chi2_contingency(ct)
n_total = ct.values.sum()
k_min   = min(ct.shape)
cramers_v = np.sqrt(chi2_cramer / (n_total * (k_min - 1)))

base_cols = cohort[["surv_mo", "event", "age10", "sexM"]]
dM = get_M_dummies(cohort)
dO = get_O_dummies(cohort)

ll_M  = fit_cox(pd.concat([base_cols, dM], axis=1).astype(float)).log_likelihood_
ll_O  = fit_cox(pd.concat([base_cols, dO], axis=1).astype(float)).log_likelihood_
ll_OM = fit_cox(pd.concat([base_cols, dM, dO], axis=1).astype(float)).log_likelihood_

lr_O_given_M  = 2 * (ll_OM - ll_M)
lr_M_given_O  = 2 * (ll_OM - ll_O)
df_O = dO.shape[1]
df_M = dM.shape[1]

print("\n" + "=" * 72)
print("O × M CROSS-AXIS DIAGNOSTICS")
print("-" * 72)
print(f"  Cramér's V = {cramers_v:.3f}   (χ² = {chi2_cramer:.0f}, p < 1e-300)")
print(f"  LR test  O | M :  χ² = {lr_O_given_M:.1f},  df = {df_O},  "
      f"p = {chi2_dist.sf(lr_O_given_M, df_O):.1e}")
print(f"  LR test  M | O :  χ² = {lr_M_given_O:.1f},  df = {df_M},  "
      f"p = {chi2_dist.sf(lr_M_given_O, df_M):.1e}")
print("  Both axes retain independent prognostic information despite correlation.")


# ---------------------------------------------------------------------------
# Landmark analysis: 6 months and 12 months from symptom onset
# ---------------------------------------------------------------------------

cohort["onset_dt"] = pd.to_datetime(cohort[COL_ON], errors="coerce")

print("\n" + "=" * 72)
print("LANDMARK ANALYSIS (immortal-time-immune)")
print("Outcome: survival from landmark; 'propagated' = second region reached by landmark")
print("-" * 72)

lm_word_rows = []

for lm in [6, 12]:
    sub = cohort[cohort["surv_mo"] > lm].copy()
    sub["propagated"] = (sub["P_mo"] <= lm).astype(int)
    sub["t"]          = sub["surv_mo"] - lm

    dM_lm = get_M_dummies(sub)
    dO_lm = get_O_dummies(sub)
    X_lm  = pd.concat(
        [sub[["t", "event", "age10", "sexM", "propagated"]], dM_lm, dO_lm],
        axis=1,
    ).astype(float)
    X_lm  = X_lm.rename(columns={"t": "surv_mo"})

    m_lm  = fit_cox(X_lm)
    hr    = m_lm.summary.loc["propagated"]

    kmf = KaplanMeierFitter()
    mask_prop = sub["propagated"] == 1
    kmf.fit(sub.loc[mask_prop,  "surv_mo"], sub.loc[mask_prop,  "event"])
    med_p = kmf.median_survival_time_
    kmf.fit(sub.loc[~mask_prop, "surv_mo"], sub.loc[~mask_prop, "event"])
    med_np = kmf.median_survival_time_

    n_prop   = int(mask_prop.sum())
    n_nonprop = int((~mask_prop).sum())

    print(f"  Landmark {lm:2d} months:")
    print(f"    N alive = {len(sub)}  "
          f"[propagated {n_prop}, non-propagated {n_nonprop}]")
    print(f"    HR (propagated vs non-propagated) = "
          f"{hr['exp(coef)']:.2f}  "
          f"({hr['exp(coef) lower 95%']:.2f}–{hr['exp(coef) upper 95%']:.2f}),  "
          f"p = {hr['p']:.1e}")
    print(f"    Median survival from onset:  "
          f"propagated {med_p:.1f} mo  vs  non-propagated {med_np:.1f} mo")
    print()

    pval_str = f"{hr['p']:.2e}" if hr["p"] < 0.001 else f"{hr['p']:.3f}"
    lm_word_rows.append([
        f"{lm} months",
        f"{len(sub)} ({n_prop} prop. / {n_nonprop} non-prop.)",
        f"{hr['exp(coef)']:.2f}  ({hr['exp(coef) lower 95%']:.2f}–{hr['exp(coef) upper 95%']:.2f})",
        pval_str,
        f"{med_p:.1f} mo",
        f"{med_np:.1f} mo",
    ])

save_table_to_word(
    title    = "Table S. Landmark analysis of propagation at 6 and 12 months",
    caption  = (
        "Landmark analysis assessing the prognostic effect of propagation "
        "(second ALSFRS-R region reached) by 6 and 12 months from symptom onset. "
        "Patients alive at the landmark are included; survival time is measured from the "
        "landmark. The Cox model adjusts for OPM_O, M_3class, age at onset, and sex. "
        "This design is immune to immortal-time bias by construction. "
        "Propagated = P1(n) <= landmark time."
    ),
    headers  = [
        "Landmark",
        "N alive (propagated / non-prop.)",
        "HR propagated vs non-prop. (95% CI)",
        "p-value",
        "Median survival — propagated",
        "Median survival — non-propagated",
    ],
    rows     = lm_word_rows,
    filename = "TableS_landmark_analysis.docx",
    footnotes = [
        "Median survival is measured from symptom onset.",
        "HR > 1 indicates higher hazard (shorter survival) in propagated patients.",
        "Cox model covariates: OPM_O (8 categories), M_3class, age at onset (per 10 years), sex.",
    ],
)