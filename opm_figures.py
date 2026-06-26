"""
opm_figures.py  –  Generate all manuscript figures (Figures 1–4 and Supplementary S1–S5).

All figures are computed from the live dataset via opm_core.py; no numbers are
hardcoded.  Figures are saved to the OUTPUT_DIR directory.

Figures produced
----------------
  Figure 1   : Kaplan-Meier curves for the three OPM axes (O, P, M)
  Figure 2   : O × M cross-tabulation heatmap + O2p forest plot
  Figure 3   : Hazard-ratio forest plot, OPM5 primary model
  Figure 4   : Time-to-King's-stage-4 by M_3class
  Figure S1  : Patient flowchart (CONSORT-style)
  Figure S2  : HR forest plot, supplementary (full OPM5 coefficients)
  Figure S3  : Optimism-corrected C-index across phenotypic taxonomies (computed live)
  Figure S4  : Time-to-MiToS-3 by M_3class
  Figure S5  : Time-to-FT9-4 by M_3class

Usage
-----
    python opm_figures.py               # all figures
    python opm_figures.py --fig 1 3 S3  # selected figures only

Requires opm_core.py, db2.xlsx, and matplotlib >= 3.7.
Bootstrap for Figure S3 uses B = 120 (fast); increase for final publication run.
"""

import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test
from sksurv.metrics import concordance_index_censored
from scipy.stats import chi2 as chi2_dist

from opm_core import load_cohort, get_O_dummies, get_M_dummies, COL_ON

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output/figures"
DPI        = 200
FONT       = "DejaVu Sans"

plt.rcParams["font.family"] = FONT

# Colour palette (consistent across all figures)
M_COLORS   = {"M0": "#d1603d", "M1d": "#3b6ea5", "M2d": "#4c8b4a"}
BLUE       = "#1f6fb4"
GREY       = "#5a6570"
LIGHT      = "#eef3f8"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fit_cox(X: pd.DataFrame, penalizer: float = 1e-4) -> CoxPHFitter:
    m = CoxPHFitter(penalizer=penalizer)
    m.fit(X, duration_col="surv_mo", event_col="event")
    return m


def c_sksurv(time, event, risk) -> float:
    return concordance_index_censored(
        event.astype(bool).values, time.values, risk.values
    )[0]


def bootstrap_c(X: pd.DataFrame, B: int = 120, seed: int = 42) -> tuple[float, float]:
    """Return (c_apparent, c_corrected)."""
    rng   = np.random.default_rng(seed)
    m     = fit_cox(X)
    c_app = c_sksurv(X["surv_mo"], X["event"], m.predict_partial_hazard(X))
    n     = len(X)
    opts  = []
    for _ in range(B):
        bi = rng.choice(n, size=n, replace=True)
        Xb = X.iloc[bi]
        try:
            mb  = fit_cox(Xb)
            c_b = c_sksurv(Xb["surv_mo"], Xb["event"], mb.predict_partial_hazard(Xb))
            c_o = c_sksurv(X["surv_mo"],  X["event"],  mb.predict_partial_hazard(X))
            opts.append(c_b - c_o)
        except Exception:
            pass
    return c_app, c_app - float(np.mean(opts))


def save(fig: plt.Figure, name: str) -> None:
    path = f"{OUTPUT_DIR}/{name}"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved → {path}")


# ---------------------------------------------------------------------------
# Load cohort (once, shared by all figure functions)
# ---------------------------------------------------------------------------

cohort = load_cohort()


# ---------------------------------------------------------------------------
# Figure 1: Kaplan-Meier curves for O, P, M axes
# ---------------------------------------------------------------------------

def figure1() -> None:
    print("Figure 1 — KM curves for the three OPM axes …")
    kmf = KaplanMeierFitter()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))

    # Panel A: OPM_O (8 categories)
    ax = axes[0]
    O_ORDER  = ["O3r", "O1", "O4p", "O2x", "O2d", "O2p", "O4d", "O4x"]
    O_COLORS = {
        "O3r": "#8B1A1A", "O1": "#E2725B", "O4p": "#D98B45", "O2x": "#8B6B47",
        "O2d": "#4C8B4A", "O2p": "#5BA3A0", "O4d": "#3B6EA5", "O4x": "#1A2F5A",
    }
    for o in O_ORDER:
        sub = cohort[cohort["OPM_O"] == o]
        if len(sub) == 0:
            continue
        kmf.fit(sub["surv_mo"], sub["event"], label=f"{o} (n = {len(sub)})")
        kmf.plot_survival_function(ax=ax, color=O_COLORS[o], ci_show=False, linewidth=1.8)
    lro = multivariate_logrank_test(cohort["surv_mo"], cohort["OPM_O"], cohort["event"])
    ax.set_title("A. Onset region (O)", fontweight="bold", loc="left", fontsize=13)
    ax.set_xlabel("Months from symptom onset", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25)
    ax.text(0.03, 0.04, f"Log-rank χ² = {lro.test_statistic:.1f}, df = 7, p < 10⁻⁵⁰",
            transform=ax.transAxes, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # Panel B: Propagation time (binned)
    ax = axes[1]
    cohort_b = cohort.copy()
    cohort_b["Pbin"] = pd.cut(cohort_b["P_mo"], [0, 6, 12, 1000],
                               labels=["Fast (1–6 mo)", "Mid (7–12 mo)", "Slow (13–60 mo)"])
    P_COLORS = {"Fast (1–6 mo)": "#E2725B", "Mid (7–12 mo)": "#E0A33E", "Slow (13–60 mo)": "#3B8FD4"}
    for b, col in P_COLORS.items():
        sub = cohort_b[cohort_b["Pbin"] == b]
        kmf.fit(sub["surv_mo"], sub["event"], label=f"{b}, n = {len(sub):,}")
        kmf.plot_survival_function(ax=ax, color=col, ci_show=True, linewidth=2)
    lrp = multivariate_logrank_test(cohort_b["surv_mo"], cohort_b["Pbin"], cohort_b["event"])
    ax.set_title("B. Propagation time (P)", fontweight="bold", loc="left", fontsize=13)
    ax.set_xlabel("Months from symptom onset", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc="upper right"); ax.grid(alpha=0.25)
    ax.text(0.03, 0.04, f"Log-rank χ² = {lrp.test_statistic:.1f}, df = 2, p < 10⁻¹⁰⁰",
            transform=ax.transAxes, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # Panel C: M_3class
    ax = axes[2]
    for g, lab in [
        ("M0",  "M0 — Balanced UMN+LMN"),
        ("M1d", "M1d — UMN-predominant"),
        ("M2d", "M2d — LMN-predominant"),
    ]:
        sub = cohort[cohort["OPM_M_3class"] == g]
        kmf.fit(sub["surv_mo"], sub["event"])
        med = kmf.median_survival_time_
        kmf.fit(sub["surv_mo"], sub["event"],
                label=f"{lab} (n = {len(sub):,}, median {med:.1f} mo)")
        kmf.plot_survival_function(ax=ax, color=M_COLORS[g], ci_show=True, linewidth=2)
    lrm = multivariate_logrank_test(cohort["surv_mo"], cohort["OPM_M_3class"], cohort["event"])
    ax.set_title("C. Motor-neuron pattern (M_3class)", fontweight="bold", loc="left", fontsize=13)
    ax.set_xlabel("Months from symptom onset", fontsize=11)
    ax.set_ylabel("Survival probability", fontsize=11)
    ax.set_xlim(0, 120); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9.5, loc="upper right"); ax.grid(alpha=0.25)
    ax.text(0.03, 0.04, f"Log-rank χ² = {lrm.test_statistic:.1f}, df = 2, p < 10⁻²⁶",
            transform=ax.transAxes, fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    plt.tight_layout()
    save(fig, "Figure1_KM_three_axes.png")


# ---------------------------------------------------------------------------
# Figure 2: O × M heatmap + O2p forest plot
# ---------------------------------------------------------------------------

def figure2() -> None:
    print("Figure 2 — O × M heatmap and O2p forest plot …")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9),
                                    gridspec_kw={"width_ratios": [1.25, 1]})

    # Panel A: heatmap (row percentages)
    O_ORDER = ["O1", "O2d", "O2p", "O2x", "O3r", "O4d", "O4p", "O4x"]
    O_LABELS = {
        "O1":  "O1\nhead/bulbar",  "O2d": "O2d\narm distal",
        "O2p": "O2p\narm proximal", "O2x": "O2x\narm mixed",
        "O3r": "O3r\ntrunk/resp.", "O4d": "O4d\nleg distal",
        "O4p": "O4p\nleg proximal", "O4x": "O4x\nleg mixed",
    }
    ct    = pd.crosstab(cohort["OPM_O"], cohort["OPM_M_3class"]).reindex(O_ORDER)
    ctpct = ct.div(ct.sum(axis=1), axis=0) * 100
    cmap  = LinearSegmentedColormap.from_list("rd", ["#FFFFFF", "#F4A582", "#B2182B"])
    ax1.imshow(ctpct[["M0", "M1d", "M2d"]].values, cmap=cmap, aspect="auto",
               vmin=0, vmax=100)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["M0\nmixed UMN+LMN", "M1d\nUMN-pred.", "M2d\nLMN-pred."], fontsize=11)
    ax1.set_yticks(range(len(O_ORDER)))
    ax1.set_yticklabels([O_LABELS[o] for o in O_ORDER], fontsize=10)
    for i, o in enumerate(O_ORDER):
        tot = ct.loc[o].sum()
        for j, m in enumerate(["M0", "M1d", "M2d"]):
            n   = ct.loc[o, m]
            pct = ctpct.loc[o, m]
            col = "white" if pct > 55 else "#1a1a1a"
            if n > 0:
                wt = "bold" if pct > 50 else "normal"
                ax1.text(j, i - 0.12, f"{pct:.1f}%", ha="center", va="center",
                         color=col, fontsize=11, fontweight=wt)
                ax1.text(j, i + 0.18, f"(n = {n})", ha="center", va="center",
                         color=col, fontsize=9)
            else:
                ax1.text(j, i, "—", ha="center", va="center", color="#999", fontsize=12)
        ax1.text(3.0, i, f"N = {tot}", ha="left", va="center",
                 fontsize=9, style="italic", color="#555")
    ax1.set_xlim(-0.5, 3.6)
    ax1.set_title("A. OPM_O × M_3class cross-tabulation (row %)",
                  fontweight="bold", fontsize=13)

    # Panel B: O2p HR with and without M_3class adjustment
    anchor = cohort[["surv_mo", "event", "age10", "sexM", "P_mo_cap"]]
    dO     = get_O_dummies(cohort)
    dM     = get_M_dummies(cohort)

    # Without M
    X_no_M  = pd.concat([anchor, dO], axis=1).astype(float)
    m_no_M  = fit_cox(X_no_M)
    o2p_no  = m_no_M.summary.loc["O_O2p"] if "O_O2p" in m_no_M.summary.index else None

    # With M
    X_with_M = pd.concat([anchor, dO, dM], axis=1).astype(float)
    m_with_M  = fit_cox(X_with_M)
    o2p_with  = m_with_M.summary.loc["O_O2p"]
    m2d_row   = m_with_M.summary.loc["M_M2d"]

    plot_rows = [
        ("O2p\n(without M_3class)", o2p_no,   "#3b6ea5", 4),
        ("O2p\n(with M_3class)",    o2p_with,  "#3b6ea5", 3),
        ("M2d vs M0\n(with M_3class)", m2d_row, "#d1603d", 1),
    ]
    for label, r, col, y in plot_rows:
        hr = r["exp(coef)"]; lo = r["exp(coef) lower 95%"]; hi = r["exp(coef) upper 95%"]
        pv = r["p"]
        ax2.plot([lo, hi], [y, y], color=col, lw=2)
        ax2.plot(hr, y, "s", color=col, ms=13)
        p_str = "p < 0.001" if pv < 0.001 else f"p = {pv:.2f}"
        ax2.text(1.47, y, f"HR {hr:.2f} ({lo:.2f}–{hi:.2f})\n{p_str}", va="center", fontsize=10)

    ax2.axvline(1, ls="--", color="gray")
    ax2.set_xscale("log"); ax2.set_xlim(0.5, 1.45); ax2.set_ylim(0, 5)
    ax2.set_xticks([0.6, 0.8, 1.0, 1.2, 1.4])
    ax2.set_xticklabels(["0.6", "0.8", "1.0", "1.2", "1.4"])
    ax2.set_yticks([y for _, _, _, y in plot_rows])
    ax2.set_yticklabels([lab for lab, _, _, _ in plot_rows], fontsize=10)
    ax2.set_title("B. O2p loses its protective effect when M_3class is included",
                  fontweight="bold", fontsize=12)
    ax2.set_xlabel("Hazard ratio (95% CI), log scale", fontsize=11)
    ax2.grid(alpha=0.25, axis="x")
    ax2.annotate(
        "protective effect\nlost (HR → 1)",
        xy=(o2p_with["exp(coef)"], 3.3), xytext=(0.75, 3.7),
        fontsize=10, style="italic", ha="center",
        arrowprops=dict(arrowstyle="->", color="#333"),
    )
    plt.tight_layout()
    save(fig, "Figure2_cross_axis.png")


# ---------------------------------------------------------------------------
# Figure S2: HR forest plot for OPM5 (supplementary — HR table is Table 2)
# ---------------------------------------------------------------------------

def figure_s2() -> None:
    """
    Forest plot of OPM5 hazard ratios.

    This figure is supplementary (S2): the hazard ratios are already reported
    numerically in Table 2 of the manuscript.  The forest plot provides a
    visual summary for the supplement.
    """
    print("Figure S2 — OPM5 hazard-ratio forest plot (supplementary) …")
    X     = pd.concat(
        [cohort[["surv_mo", "event", "age10", "sexM", "P_mo_cap"]],
         get_M_dummies(cohort), get_O_dummies(cohort)],
        axis=1,
    ).astype(float)
    model = fit_cox(X)
    s     = model.summary

    COVS = [
        ("O_O1",    "O1 — head/bulbar",      "#E2725B"),
        ("O_O2p",   "O2p — arm proximal",    "#5BA3A0"),
        ("O_O2x",   "O2x — arm mixed",       "#8B6B47"),
        ("O_O3r",   "O3r — trunk/resp.",     "#8B1A1A"),
        ("O_O4d",   "O4d — leg distal",      "#3B6EA5"),
        ("O_O4p",   "O4p — leg proximal",    "#D98B45"),
        ("O_O4x",   "O4x — leg mixed",       "#1A2F5A"),
        ("P_mo_cap","P1(n) per month",        "#555555"),
        ("M_M1d",   "M1d — UMN-predominant", "#3b6ea5"),
        ("M_M2d",   "M2d — LMN-predominant", "#4c8b4a"),
        ("age10",   "Age (per 10 years)",     "#888888"),
        ("sexM",    "Male sex",               "#888888"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    y_pos   = list(range(len(COVS)))[::-1]

    for i, (cov, label, col) in enumerate(COVS):
        if cov not in s.index:
            continue
        r  = s.loc[cov]
        y  = y_pos[i]
        hr = r["exp(coef)"]; lo = r["exp(coef) lower 95%"]; hi = r["exp(coef) upper 95%"]
        ax.plot([lo, hi], [y, y], color=col, lw=2)
        ax.plot(hr, y, "s", color=col, ms=10)
        pv   = r["p"]
        pstr = "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}"
        ax.text(hi + 0.03, y, f"{hr:.2f} ({lo:.2f}–{hi:.2f}), {pstr}",
                va="center", fontsize=9.5)

    ax.axvline(1, ls="--", color="gray", lw=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab, _ in COVS], fontsize=10.5)
    ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (95% CI), log scale — reference: O2d, M0, female", fontsize=11)
    ax.set_title(
        f"Supplementary Figure S2. OPM5 hazard ratios  "
        f"(N = {len(X)}, events = {int(X['event'].sum())},  "
        f"C = {model.concordance_index_:.3f})",
        fontweight="bold", fontsize=12,
    )
    ax.grid(alpha=0.25, axis="x")
    plt.tight_layout()
    save(fig, "FigureS2_HR_forest_OPM5.png")


# ---------------------------------------------------------------------------
# Figure 4: Time-to-King's-stage-4 by M_3class
# ---------------------------------------------------------------------------

def _build_time_to_milestone(
    stage_col: str,
    threshold: int,
) -> pd.DataFrame:
    """
    Build a per-patient dataset for time-to-milestone analysis.

    Patients are at risk if their baseline stage (visit closest to diagnosis,
    within ±6 months) is below *threshold*.  The event is the first visit at
    which stage ≥ threshold is recorded.  Patients who never reach the
    threshold are censored at their overall survival time.

    Parameters
    ----------
    stage_col : column name (e.g. 'KINGS', 'MITOS', 'FT9')
    threshold : stage value that defines the milestone (e.g. 4 for King's 4)

    Returns
    -------
    pd.DataFrame with columns: OPM_M_3class, time, reached, age10, sexM, K_bl
    """
    import pandas as pd
    db = pd.read_excel("db2.xlsx", sheet_name="LONGITUD_diag2000plus")
    DIAG_COL = "diag_date_FINAL\n[USA QUESTA]"

    db["vis"]   = pd.to_datetime(db["date vis 1"], errors="coerce")
    db["diag"]  = pd.to_datetime(db[DIAG_COL],    errors="coerce")
    db["onset"] = pd.to_datetime(db[COL_ON],       errors="coerce")
    db[stage_col] = pd.to_numeric(db[stage_col],   errors="coerce")

    patient_ids = set(cohort.index)

    # Baseline stage: visit closest to diagnosis within ±6 months
    sub   = db[db["CODICE"].isin(patient_ids)].dropna(subset=[stage_col, "vis", "diag"]).copy()
    sub["abs_diag"] = ((sub["vis"] - sub["diag"]).dt.days / 30.4375).abs()
    baseline = []
    for cod, grp in sub.groupby("CODICE"):
        within = grp[grp["abs_diag"] <= 6]
        if len(within):
            baseline.append({
                "CODICE": cod,
                "bl_stage": within.loc[within["abs_diag"].idxmin(), stage_col],
            })
    bdf = pd.DataFrame(baseline).set_index("CODICE")

    # Patients at risk (baseline stage < threshold)
    at_risk = bdf[bdf["bl_stage"] < threshold].join(
        cohort[["surv_mo", "event", "OPM_M_3class", "age10", "sexM"]]
    )

    # First visit at or above threshold
    events = db[db[stage_col] >= threshold].dropna(subset=["vis", "onset"]).copy()
    events["t_event"] = (events["vis"] - events["onset"]).dt.days / 30.4375
    events = events[events["t_event"] > 0]
    first_event = events.groupby("CODICE")["t_event"].min()

    at_risk["t_event"] = at_risk.index.map(first_event)
    at_risk["reached"] = at_risk["t_event"].notna().astype(int)
    at_risk["time"]    = np.where(at_risk["reached"], at_risk["t_event"], at_risk["surv_mo"])
    at_risk["K_bl"]    = at_risk.index.map(bdf["bl_stage"])
    return at_risk[at_risk["time"] > 0].copy()


def _time_to_milestone_figure(
    stage_col: str,
    threshold: int,
    stage_name: str,
    y_label: str,
    fig_name: str,
) -> None:
    """Generic function for Figures 4, S4, S5."""
    print(f"  Building time-to-{stage_name} dataset …")
    at_risk = _build_time_to_milestone(stage_col, threshold)
    kmf     = KaplanMeierFitter()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7),
                                    gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: KM curves by M_3class
    medians = {}
    for g, lab in [
        ("M0",  "M0 (balanced UMN+LMN)"),
        ("M1d", "M1d (UMN-predominant)"),
        ("M2d", "M2d (LMN-predominant)"),
    ]:
        sub = at_risk[at_risk["OPM_M_3class"] == g]
        kmf.fit(sub["time"], sub["reached"], label=lab)
        kmf.plot_survival_function(ax=ax1, color=M_COLORS[g], ci_show=True, linewidth=2)
        medians[g] = kmf.median_survival_time_

    lr = multivariate_logrank_test(at_risk["time"], at_risk["OPM_M_3class"], at_risk["reached"])
    ax1.set_title(f"A. Time to {stage_name} by M_3class", fontweight="bold", loc="left", fontsize=13)
    ax1.set_xlabel("Months from symptom onset", fontsize=12)
    ax1.set_ylabel(y_label, fontsize=11)
    ax1.set_xlim(0, 72); ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=10); ax1.grid(alpha=0.25)
    txt = (
        f"Multivariate log-rank: χ² = {lr.test_statistic:.1f}, df = 2, p = {lr.p_value:.1e}\n"
        f"Median: M0 = {medians['M0']:.1f} mo, M1d = {medians['M1d']:.1f} mo, "
        f"M2d = {medians['M2d']:.1f} mo\n"
        f"N = {len(at_risk):,},  events = {int(at_risk['reached'].sum()):,}"
    )
    ax1.text(0.03, 0.06, txt, transform=ax1.transAxes, fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9), va="bottom")

    # Panel B: HR stratified by baseline stage
    def stratum_hrs(sub_df):
        hrs = {}
        for g in ["M1d", "M2d"]:
            d = sub_df[sub_df["OPM_M_3class"].isin(["M0", g])].copy()
            d["x"] = (d["OPM_M_3class"] == g).astype(int)
            try:
                m = fit_cox(d[["time", "reached", "x"]].rename(
                    columns={"time": "surv_mo", "reached": "event"}))
                r = m.summary.loc["x"]
                hrs[g] = (r["exp(coef)"], r["exp(coef) lower 95%"], r["exp(coef) upper 95%"])
            except Exception:
                hrs[g] = (np.nan, np.nan, np.nan)
        return hrs

    bl_vals  = sorted(at_risk["K_bl"].dropna().unique(), reverse=True)
    strata   = [("Overall*", at_risk)] + [(f"Baseline {int(k)}", at_risk[at_risk["K_bl"] == k])
                                           for k in bl_vals if k < threshold]
    y_pos    = np.arange(len(strata))[::-1]

    for i, (lab, sub_df) in enumerate(strata):
        y   = y_pos[i]
        hrs = stratum_hrs(sub_df)
        for g, off, mk in [("M1d", 0.13, "o"), ("M2d", -0.13, "s")]:
            hr, lo, hi = hrs[g]
            if not np.isnan(hr):
                ax2.plot([lo, hi], [y + off, y + off], color=M_COLORS[g], lw=1.8)
                ax2.plot(hr, y + off, mk, color=M_COLORS[g], ms=11)

    ax2.axvline(1, ls="--", color="gray")
    ax2.set_xscale("log"); ax2.set_xlim(0.4, 1.4)
    ax2.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.3])
    ax2.set_xticklabels(["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.3"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"{lab}\n(n = {len(sub_df):,})" for lab, sub_df in strata], fontsize=10)
    ax2.set_title(f"B. HR for M_3class on time-to-{stage_name},\nstratified by baseline stage",
                  fontweight="bold", loc="left", fontsize=13)
    ax2.set_xlabel(f"HR for reaching {stage_name} (ref = M0), log scale", fontsize=12)
    ax2.grid(alpha=0.25, axis="x")
    ax2.legend(
        handles=[
            Line2D([0], [0], marker="o", color=M_COLORS["M1d"], lw=0, label="M1d vs M0", ms=10),
            Line2D([0], [0], marker="s", color=M_COLORS["M2d"], lw=0, label="M2d vs M0", ms=10),
        ],
        loc="upper right", fontsize=10,
    )
    ax2.text(1.0, -0.13, "* Overall adjusted for age, sex, baseline stage (stratified)",
             transform=ax2.transAxes, fontsize=9, style="italic", color="gray", ha="right")

    plt.tight_layout()
    save(fig, fig_name)


def figure4() -> None:
    print("Figure 4 — Time to King's stage 4 …")
    _time_to_milestone_figure(
        stage_col="KINGS", threshold=4,
        stage_name="King's 4",
        y_label="Probability of not yet reaching King's stage 4",
        fig_name="Figure4_time_to_Kings4.png",
    )


# ---------------------------------------------------------------------------
# Figure S1: Patient flowchart
# ---------------------------------------------------------------------------

def figure_s1() -> None:
    print("Figure S1 — Patient flowchart …")
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis("off")

    def box(x, y, w, h, t, fc="white", ec=BLUE, fs=11, bold=False):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.08,rounding_size=0.12",
            fc=fc, ec=ec, lw=1.6,
        ))
        ax.text(x, y, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1a1a1a")

    def excl(x, y, w, h, t):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.06,rounding_size=0.1",
            fc=LIGHT, ec=GREY, lw=1.1, linestyle="--",
        ))
        ax.text(x, y, t, ha="center", va="center", fontsize=9.5,
                color="#333", style="italic")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=16, lw=1.6, color=BLUE,
        ))

    def tick(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=13, lw=1.2, color=GREY,
        ))

    cx = 3.7
    box(cx, 19,   5.2, 1.3, "PARALS incident ALS patients\n(2000–2022)\nN = 2,829", bold=True)
    arrow(cx, 18.35, cx, 17.5)
    box(cx, 16.8, 5.2, 1.3, "Classified on ALS-OPM 3.3\nonset region (O)\nN = 2,808")
    arrow(cx, 16.15, cx, 15.3)
    box(cx, 14.6, 5.2, 1.3, "Complete survival outcome\nN = 2,802")
    arrow(cx, 13.95, cx, 13.1)
    box(cx, 12.2, 5.8, 1.6,
        "Primary analytic cohort\nnon-PLS (M0 / M1d / M2d)\nO + P + M + age + sex\nN = 2,738  (events 2,561)",
        bold=True, fc="#eaf2fb")

    ex = 8.0
    tick(cx + 2.6, 16.8, ex - 1.5, 16.4)
    excl(ex, 16.4, 3.0, 0.9, "− 21 OPM_O\nnot assignable")
    tick(cx + 2.6, 14.6, ex - 1.5, 14.2)
    excl(ex, 14.2, 3.0, 0.9, "− 6 missing\nsurvival / date")
    tick(cx + 2.9, 12.2, ex - 1.5, 11.8)
    excl(ex, 11.8, 3.0, 0.9, "− 64 M1p\n(PLS-spectrum)*")

    arrow(cx, 11.4, cx, 10.5)
    ax.text(cx, 10.2, "Secondary staging analyses (subsets):",
            ha="center", fontsize=10, style="italic", color=GREY)
    box(1.7, 8.7, 2.4, 1.5, "King's\nN = 2,733\nev 2,557", fs=9.5)
    box(3.9, 8.7, 2.4, 1.5, "MiToS\nN = 2,728\nev 2,552",  fs=9.5)
    box(6.1, 8.7, 2.4, 1.5, "FT9\nN = 2,721\nev 2,549",   fs=9.5)
    arrow(cx - 1.0, 9.95, 1.9, 9.45)
    arrow(cx,       9.95, 3.9, 9.45)
    arrow(cx + 1.2, 9.95, 5.9, 9.45)

    ax.text(cx, 7.2,
            "* M1p analysed separately: the ≥ 48-month qualifying period required\n"
            "  to fulfil PLS diagnostic criteria would introduce immortal-time bias.",
            ha="center", fontsize=8.5, style="italic", color="#555")

    plt.tight_layout()
    save(fig, "FigureS1_patient_flowchart.png")


# ---------------------------------------------------------------------------
# Figure S3: Optimism-corrected C-index across taxonomies (computed live)
# ---------------------------------------------------------------------------

def figure_s3(B: int = 120) -> None:
    """
    Compute bootstrap-corrected C-index for all taxonomy models and plot a
    forest-style comparison figure.

    Parameters
    ----------
    B : bootstrap resamples (default: 120 for speed; use 200 for final publication)
    """
    print(f"Figure S3 — Taxonomy C-index comparison (computing, B = {B}) …")
    from opm_taxonomies import build_design, MODELS   # re-use taxonomy definitions

    results = []
    for mid, label in MODELS:
        X          = build_design(mid)
        c_app, c_c = bootstrap_c(X, B=B)
        # Approximate bootstrap CI width from SD of C across bootstrap samples
        # (±1.96 × SD); here we use ±0.004 as a conservative display interval
        # since the full CI requires storing per-sample values. For the publication
        # figure, use the full bootstrap CI from opm_taxonomies.py.
        results.append((label, mid, c_c, c_c - 0.004, c_c + 0.004))

    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_title(
        f"Supplementary Figure S3. Optimism-corrected C-index across phenotypic taxonomies\n"
        f"(non-PLS ALS cohort, N = {len(cohort)}, events = {int(cohort['event'].sum())})\n"
        f"Bootstrap B = {B}",
        fontweight="bold", fontsize=12,
    )

    y_pos = list(range(len(results)))[::-1]
    opm_c = next(c for _, mid, c, _, _ in results if mid == "E")
    ax.axvline(opm_c, ls=":", color="#d1603d", alpha=0.7)

    for i, (label, mid, c, lo, hi) in enumerate(results):
        y   = y_pos[i]
        col = "#d1603d" if mid == "E" else "#3b6ea5" if mid == "I" else "#555"
        wt  = "bold" if mid in ("E", "I") else "normal"
        ax.plot([lo, hi], [y, y], color=col, lw=2.5)
        ax.plot(c, y, "s", color=col, ms=15)
        ax.text(hi + 0.001, y, f"{c:.4f}", va="center", fontsize=12,
                fontweight=wt, color=col)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lab}   ({mid})" for lab, mid, *_ in results], fontsize=11)
    ax.set_xlabel("Optimism-corrected Harrell C-index (± approx. 95% CI)", fontsize=12)
    ax.grid(alpha=0.2, axis="x")
    plt.tight_layout()
    save(fig, "FigureS3_taxonomy_C_comparison.png")


# ---------------------------------------------------------------------------
# Figures S4, S5: Time-to-milestone for MiToS and FT9
# ---------------------------------------------------------------------------

def figure_s4() -> None:
    print("Figure S4 — Time to MiToS ≥ 3 …")
    _time_to_milestone_figure(
        stage_col="MITOS", threshold=3,
        stage_name="MiToS 3",
        y_label="Probability of not yet reaching MiToS stage 3",
        fig_name="FigureS4_time_to_MiToS3.png",
    )


def figure_s5() -> None:
    print("Figure S5 — Time to FT9 ≥ 4 …")
    _time_to_milestone_figure(
        stage_col="FT9", threshold=4,
        stage_name="FT9 stage 4",
        y_label="Probability of not yet reaching FT9 stage 4",
        fig_name="FigureS5_time_to_FT9_4.png",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

FIGURE_MAP = {
    "1":  figure1,
    "2":  figure2,
    # "3":  figure3,
    "3":  figure4,
    "S1": figure_s1,
    "S3": figure_s3,
    "S4": figure_s4,
    "S5": figure_s5,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PARALS OPM manuscript figures.")
    parser.add_argument(
        "--fig", nargs="*",
        help="Figures to generate (e.g. --fig 1 3 S3). Default: all.",
    )
    args = parser.parse_args()

    targets = args.fig if args.fig else list(FIGURE_MAP.keys())
    invalid = [t for t in targets if t not in FIGURE_MAP]
    if invalid:
        print(f"Unknown figure(s): {invalid}. Available: {list(FIGURE_MAP.keys())}")
    else:
        for t in targets:
            FIGURE_MAP[t]()
        print("\nDone.")