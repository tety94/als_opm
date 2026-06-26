"""
opm_taxonomies.py  –  Prognostic comparison of OPM 3.3 against alternative phenotypic
                       taxonomies (Supplementary Table S6).

Taxonomies compared
-------------------
  A  Bulbar / Spinal binary (1 df)
  B  El Escorial diagnostic certainty (ordinal, 3–4 levels)
  C  Classical phenotype (7 categories)
  E  OPM 3.3  (OPM_O 8-cat + M_3class + P1(n); k = 12)   ← primary
  I  OPM 3.3 + El Escorial  (combined)

All models share a common anchor: P_mo_cap (linear, capped at 60 mo) + age10 + sexM.
This ensures that differences in C / AIC reflect taxonomy performance net of
propagation kinetics and demographics, consistent with the paper's main claim.

Outputs
-------
- AIC, apparent C-index, and bootstrap-corrected C-index for each taxonomy
- ΔAIC (OPM vs Bulbar/Spinal; OPM vs Classic 7-cat)
- LR test statistics and p-values

Usage
-----
    python opm_taxonomies.py

Requires opm_core.py and db2_bk200260525.xlsx.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter
from sksurv.metrics import concordance_index_censored
from scipy.stats import chi2 as chi2_dist

from opm_core import load_cohort, get_O_dummies, get_M_dummies

np.random.seed(42)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_cox(X: pd.DataFrame, penalizer: float = 1e-4) -> CoxPHFitter:
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(X, duration_col="surv_mo", event_col="event")
    return m


def compute_aic(model: CoxPHFitter, X: pd.DataFrame) -> float:
    """AIC = −2 × log-likelihood + 2 × (number of free parameters)."""
    k = X.shape[1] - 2   # subtract 'surv_mo' and 'event'
    return -2 * model.log_likelihood_ + 2 * k


def c_sksurv(time: pd.Series, event: pd.Series, risk: pd.Series) -> float:
    return concordance_index_censored(
        event.astype(bool).values, time.values, risk.values
    )[0]


def bootstrap_optimism_c(
    X: pd.DataFrame,
    B: int = 120,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (c_apparent, c_corrected) using Harrell–Steyerberg optimism correction."""
    rng       = np.random.default_rng(seed)
    m_full    = fit_cox(X)
    c_app     = c_sksurv(X["surv_mo"], X["event"], m_full.predict_partial_hazard(X))
    n         = len(X)
    optimism  = []
    for _ in range(B):
        bi = rng.choice(n, size=n, replace=True)
        Xb = X.iloc[bi]
        try:
            mb  = fit_cox(Xb)
            c_b = c_sksurv(Xb["surv_mo"], Xb["event"], mb.predict_partial_hazard(Xb))
            c_o = c_sksurv(X["surv_mo"],  X["event"],  mb.predict_partial_hazard(X))
            optimism.append(c_b - c_o)
        except Exception:
            pass
    return c_app, c_app - float(np.mean(optimism))


# ---------------------------------------------------------------------------
# Phenotype classification helpers
# ---------------------------------------------------------------------------

def classify_classic7(site: str) -> str:
    """
    Map the 'site of onset1' free-text field to one of 7 classical phenotype categories.

    Categories
    ----------
    bulbar_pyr    : bulbar with pyramidal signs
    bulbar        : pure bulbar
    flail_arm     : flail-arm (scapulohumeral)
    flail_leg     : flail-leg
    respiratory   : respiratory onset
    pyramidal     : pure pyramidal / UMN-predominant spinal
    classic_spinal: all remaining spinal-onset presentations (reference)
    """
    s = str(site).upper()
    if s.startswith("B") and "PY" in s:
        return "bulbar_pyr"
    if s.startswith("B"):
        return "bulbar"
    if "FA" in s:
        return "flail_arm"
    if "FL" in s:
        return "flail_leg"
    if "RE" in s:
        return "respiratory"
    if "PY" in s:
        return "pyramidal"
    return "classic_spinal"   # reference category


# ---------------------------------------------------------------------------
# Load cohort and derive taxonomy variables
# ---------------------------------------------------------------------------

cohort = load_cohort()

cohort["classic"]  = cohort["site of onset1"].apply(classify_classic7)
cohort["bulbar"]   = (
    cohort["site of onset2"].astype(str).str.upper().str.startswith("B")
).astype(int)
# When 'site of onset2' is missing, fall back to OPM_O == 'O1'
cohort["bulbar"]   = np.where(
    cohort["site of onset2"].isna(),
    (cohort["OPM_O"].astype(str) == "O1").astype(int),
    cohort["bulbar"],
)
cohort["EL"]       = cohort["EL1"].astype(str).str.upper()

# Common anchor covariates (same for every taxonomy comparison)
anchor = cohort[["surv_mo", "event", "age10", "sexM", "P_mo_cap"]]


def build_design(model_id: str) -> pd.DataFrame:
    """
    Build the design matrix for taxonomy model *model_id*.

    All models include the common anchor (P_mo_cap + age10 + sexM).
    The taxonomy variable(s) are added on top.
    """
    if model_id == "A":
        # Bulbar/Spinal binary
        return pd.concat([anchor, cohort[["bulbar"]]], axis=1).astype(float)

    if model_id == "B":
        # El Escorial certainty (ordinal; reference = lowest category)
        dEL = pd.get_dummies(cohort["EL"], prefix="EL")
        dEL = dEL.drop(columns=[dEL.columns[0]])   # drop reference level
        return pd.concat([anchor, dEL], axis=1).astype(float)

    if model_id == "C":
        # Classical phenotype (7 categories; reference = classic_spinal)
        dCL = pd.get_dummies(cohort["classic"], prefix="cl")
        dCL = dCL.drop(columns=["cl_classic_spinal"])
        return pd.concat([anchor, dCL], axis=1).astype(float)

    if model_id == "E":
        # OPM 3.3: OPM_O (8-cat) + M_3class  (+ anchor already has P)
        dM = get_M_dummies(cohort)
        dO = get_O_dummies(cohort)
        return pd.concat([anchor, dM, dO], axis=1).astype(float)

    if model_id == "I":
        # OPM 3.3 + El Escorial
        dM  = get_M_dummies(cohort)
        dO  = get_O_dummies(cohort)
        dEL = pd.get_dummies(cohort["EL"], prefix="EL")
        dEL = dEL.drop(columns=[dEL.columns[0]])
        return pd.concat([anchor, dM, dO, dEL], axis=1).astype(float)

    raise ValueError(f"Unknown taxonomy model ID: '{model_id}'")


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

MODELS = [
    ("A", "Bulbar / Spinal"),
    ("B", "El Escorial"),
    ("C", "Classic 7-cat"),
    ("E", "OPM 3.3 (primary)"),
    ("I", "OPM 3.3 + El Escorial"),
]

print("=" * 80)
print(f"SUPPLEMENTARY TABLE S6 — Taxonomy comparison  (N = {len(cohort)}, B = 120 bootstrap)")
print(f"Anchor: P_mo_cap (linear) + age10 + sexM  [consistent with OPM5 specification]")
print("=" * 80)
print(f"{'ID':<4}  {'Taxonomy':<26}  {'k':>3}  {'AIC':>10}  {'C-app':>8}  {'C-corr':>8}")
print("-" * 80)

fitted   = {}
aics     = {}
log_liks = {}

for mid, label in MODELS:
    X = build_design(mid)
    m = fit_cox(X)
    a = compute_aic(m, X)
    c_app, c_corr = bootstrap_optimism_c(X, B=120)
    fitted[mid]    = m
    aics[mid]      = a
    log_liks[mid]  = m.log_likelihood_
    k = X.shape[1] - 2
    print(f"{mid:<4}  {label:<26}  {k:>3}  {a:>10.0f}  {c_app:>8.4f}  {c_corr:>8.4f}")

print("-" * 80)
print(f"\nΔAIC (OPM 3.3  vs  Bulbar/Spinal):  {aics['E'] - aics['A']:+.1f}  "
      f"(paper: −89; negative = OPM better)")
print(f"ΔAIC (OPM 3.3  vs  Classic 7-cat):  {aics['E'] - aics['C']:+.1f}  "
      f"(paper: +1.5; ~equivalent)")

# LR test OPM vs Bulbar/Spinal
lr_OPM_vs_BS = 2 * (log_liks["E"] - log_liks["A"])
df_OPM_vs_BS = (build_design("E").shape[1] - build_design("A").shape[1])
print(f"\nLR test OPM 3.3 vs Bulbar/Spinal:  "
      f"χ² = {lr_OPM_vs_BS:.1f},  df = {df_OPM_vs_BS},  "
      f"p = {chi2_dist.sf(lr_OPM_vs_BS, df_OPM_vs_BS):.1e}")

# LR test OPM vs Classic 7-cat
lr_OPM_vs_CL = 2 * (log_liks["E"] - log_liks["C"])
df_OPM_vs_CL = (build_design("E").shape[1] - build_design("C").shape[1])
print(f"LR test OPM 3.3 vs Classic 7-cat:  "
      f"χ² = {lr_OPM_vs_CL:.1f},  df = {df_OPM_vs_CL},  "
      f"p = {chi2_dist.sf(lr_OPM_vs_CL, df_OPM_vs_CL):.2f}")
