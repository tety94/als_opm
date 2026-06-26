"""
opm_core.py  –  Canonical data loader and design-matrix factory for the PARALS ALS-OPM 3.3 analysis.

Single source of truth for the primary analytic cohort (N = 2,738).
All analysis modules import load_cohort() and get_design() from here.

Cohort cascade (data freeze 22 June 2026):
  2,829 unique patients
  → 2,808  valid OPM_O
  → 2,802  complete outcome
  → 2,738  M_3class ∈ {M0, M1d, M2d}  [M1p n = 64 excluded from primary cohort]
  → 2,738  complete age  (no additional drop)
  → 2,738  complete sex  (no additional drop)
  = PRIMARY COHORT, 2,561 events (death or invasive ventilation)

Database conventions (sheet LONGITUD_diag2000plus):
  STATUS  : 1 = event (death / tracheostomy), 2 = censored
  CODICE  : primary key (patient identifier)
  SURVIVAL_FINAL [USA QUESTA]  : survival time in years from symptom onset
  AGE_ONSET_FINAL [USA QUESTA] : age at symptom onset in years
  OPM_P   : string format 'P1(n)' where n is months to first non-onset-region ALSFRS-R < 4
  OPM_M_3class : {M0, M1d, M2d, M1p, None}
  KINGS_calc   : King's stage derived from ALSFRS-R (fallback when KINGS is missing)
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Column names (contain embedded newlines as in the original database file)
# ---------------------------------------------------------------------------
COL_SURV = "SURVIVAL_FINAL\n[USA QUESTA]"
COL_AGE  = "AGE_ONSET_FINAL\n[USA QUESTA]"
COL_ON   = "onset_date_FINAL\n[USA QUESTA]"
COL_DIAG = "diag_date_FINAL\n[USA QUESTA]"

DEFAULT_DB = "db2.xlsx"
SHEET      = "LONGITUD_diag2000plus"

# Columns aggregated per patient (first observation)
_AGG_COLS = [
    "OPM_O", "OPM_P", "OPM_M_3class", "OPM_M",
    "STATUS", "Sex",
    "KINGS", "MITOS", "FT9", "KINGS_calc", "EL1",
    "site of onset1", "site of onset2",
    "P_new_mo", "P_new_source",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_patient_level(db: str = DEFAULT_DB) -> pd.DataFrame:
    """Load the longitudinal database and aggregate to one row per patient."""
    df = pd.read_excel(db, sheet_name=SHEET)
    agg = {c: "first" for c in _AGG_COLS if c in df.columns}
    for col in (COL_SURV, COL_AGE, COL_ON, COL_DIAG):
        agg[col] = "first"
    return df.groupby("CODICE").agg(agg)


def _derive(d: pd.DataFrame) -> pd.DataFrame:
    """
    Derive analysis variables. Does NOT filter rows.

    New columns
    -----------
    surv_mo   : survival time in months (from symptom onset)
    event     : 1 = death/tracheostomy, 0 = censored
    age       : age at symptom onset (years)
    age10     : age / 10  (used as continuous covariate in Cox models)
    sexU      : sex as uppercase single character: 'M' or 'F'
    sexM      : 1 if male, 0 if female
    P_mo      : propagation time in months (extracted from 'P1(n)' string)
    P_mo_cap  : P_mo capped at 60 months (pre-specified cap for Cox models)
    KINGS_use : King's stage with fallback to KINGS_calc when KINGS is missing
    """
    d = d.copy()
    d["surv_mo"]  = pd.to_numeric(d[COL_SURV], errors="coerce") * 12
    d["event"]    = (pd.to_numeric(d["STATUS"], errors="coerce") == 1).astype(int)
    d["age"]      = pd.to_numeric(d[COL_AGE], errors="coerce")
    d["age10"]    = d["age"] / 10
    d["sexU"]     = d["Sex"].astype(str).str.upper().str[0]
    d["sexM"]     = (d["sexU"] == "M").astype(int)
    P_old = (
        d["OPM_P"].astype(str)
        .str.extract(r"P1\((\d+\.?\d*)\)")[0]
        .astype(float)
    )
    P_new = pd.to_numeric(d.get("P_new_mo", pd.Series(dtype=float)), errors="coerce")
    d["P_mo"] = P_new.fillna(P_old)
    d["P_mo_cap"] = d["P_mo"].clip(upper=60)
    d["KINGS_use"] = d["KINGS"].fillna(d["KINGS_calc"])
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_cohort(db: str = DEFAULT_DB, verbose: bool = False) -> pd.DataFrame:
    """
    Return the PRIMARY ANALYTIC COHORT (N = 2,738) as a per-patient DataFrame.

    The five exclusion filters are applied in the canonical order:
      1. Valid OPM_O  (not NaN, not 'CO', not empty)
      2. Complete outcome  (surv_mo > 0, STATUS not NaN)
      3. M_3class ∈ {M0, M1d, M2d}  (excludes M1p PLS-spectrum and other rare classes)
      4. Non-missing age
      5. Non-missing sex (M or F)

    Parameters
    ----------
    db      : path to the Excel database (default: 'db2_bk200260525.xlsx')
    verbose : if True, print the cascade counts and event total

    Returns
    -------
    pd.DataFrame with one row per patient and all derived analysis columns.

    Raises
    ------
    AssertionError if the resulting cohort size differs from 2,738 — indicates
    a database change that requires review of all reported numbers.
    """
    d = _derive(_load_patient_level(db))

    n0 = len(d)

    # 1. Valid OPM_O
    valid_O = d["OPM_O"].notna() & ~d["OPM_O"].astype(str).str.upper().isin(["NAN", "", "CO"])
    d = d[valid_O]
    n1 = len(d)

    # 2. Complete outcome
    d = d[d["surv_mo"].notna() & d["STATUS"].notna() & (d["surv_mo"] > 0)]
    n2 = len(d)

    # 3. M_3class restricted to primary three classes
    d = d[d["OPM_M_3class"].isin(["M0", "M1d", "M2d"])]
    n3 = len(d)

    # 4. Non-missing age
    d = d[d["age"].notna()]
    n4 = len(d)

    # 5. Non-missing sex
    d = d[d["sexU"].isin(["M", "F"])]
    n5 = len(d)

    if verbose:
        print(
            f"Cohort cascade: {n0} unique patients"
            f" → valid OPM_O {n1}"
            f" → complete outcome {n2}"
            f" → M_3class {n3}"
            f" → complete age {n4}"
            f" → complete sex {n5}"
        )
        print(f"Events: {int(d['event'].sum())}")

    assert n5 == 2738, (
        f"Expected cohort size 2,738 but obtained {n5}. "
        "The database has changed — all reported numbers require re-verification."
    )
    return d


def get_O_dummies(
    d: pd.DataFrame,
    min_count: int = 10,
    ref: str = "O_O2d",
) -> pd.DataFrame:
    """
    One-hot encode OPM_O.

    Categories with fewer than *min_count* patients are collapsed into 'other'.
    The reference category (default: O2d, arm-distal onset) is dropped.

    Parameters
    ----------
    d         : cohort DataFrame (output of load_cohort)
    min_count : minimum cell size to retain a category (default: 10)
    ref       : dummy column name to drop as reference (default: 'O_O2d')

    Returns
    -------
    pd.DataFrame of dummy columns prefixed with 'O_'
    """
    counts = d["OPM_O"].value_counts()
    keep   = counts[counts >= min_count].index
    O_safe = d["OPM_O"].where(d["OPM_O"].isin(keep), other="other")
    dummies = pd.get_dummies(O_safe, prefix="O")
    if ref in dummies.columns:
        dummies = dummies.drop(columns=[ref])
    return dummies


def get_M_dummies(d: pd.DataFrame, ref: str = "M_M0") -> pd.DataFrame:
    """
    One-hot encode OPM_M_3class.

    The reference category (default: M0, classical balanced UMN+LMN) is dropped.

    Parameters
    ----------
    d   : cohort DataFrame (output of load_cohort)
    ref : dummy column name to drop as reference (default: 'M_M0')

    Returns
    -------
    pd.DataFrame of dummy columns prefixed with 'M_'
    """
    dummies = pd.get_dummies(d["OPM_M_3class"], prefix="M")
    if ref in dummies.columns:
        dummies = dummies.drop(columns=[ref])
    return dummies


def get_design(d: pd.DataFrame, model: str = "OPM5") -> pd.DataFrame:
    """
    Build the complete design matrix (including surv_mo and event) for Cox fitting.

    Model specifications
    --------------------
    'OPM5'  : OPM_O (8-cat) + P1(n) linear capped at 60 mo + M_3class + age10 + sexM
              Primary specification (Table 2); apparent C ≈ 0.692.
    'S9'    : OPM_O (8-cat) + M_3class + age10 + sexM
              Secondary specification without propagation (Table 3).
    'base'  : age10 + sexM
              Null model (demographics only).

    Notes
    -----
    All columns are cast to float. The returned DataFrame contains 'surv_mo' and
    'event' as the first two columns, followed by covariates.

    Parameters
    ----------
    d     : primary cohort DataFrame (output of load_cohort)
    model : one of 'OPM5', 'S9', 'base'

    Returns
    -------
    pd.DataFrame suitable for CoxPHFitter.fit(X, 'surv_mo', 'event')

    Raises
    ------
    ValueError for unrecognised model names.
    """
    base = d[["surv_mo", "event", "age10", "sexM"]].copy()

    if model == "base":
        return base.astype(float)

    dM = get_M_dummies(d)
    dO = get_O_dummies(d)

    if model == "OPM5":
        X = pd.concat([base, d[["P_mo_cap"]], dM, dO], axis=1)
    elif model == "S9":
        X = pd.concat([base, dM, dO], axis=1)
    else:
        raise ValueError(
            f"Unknown model '{model}'. Choose from: 'OPM5', 'S9', 'base'."
        )

    return X.astype(float)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cohort = load_cohort(verbose=True)
    print(f"\nPrimary cohort: N = {len(cohort)}, events = {int(cohort['event'].sum())}")
    print("M_3class distribution:", dict(cohort["OPM_M_3class"].value_counts()))
    print("Sex distribution:      ", dict(cohort["sexU"].value_counts()))
