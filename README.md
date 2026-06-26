# PARALS ALS-OPM 3.3 — Unified Analysis Pipeline

Rewritten and unified version of the reproducibility package for the manuscript
*"Anatomy of onset and motor-neuron phenotype provide complementary prognostic
information in ALS: population-based evidence from the OPM 3.3 framework"*
(PARALS registry, Piedmont and Aosta Valley, 2000–2022).

---

## File overview

| File | Purpose |
|------|---------|
| `opm_core.py` | Single source of truth: data loading, cohort cascade, design-matrix factory |
| `opm_models.py` | Primary Cox models (Table 2 / Table 3), O × M diagnostics, landmark analysis |
| `opm_bootstrap.py` | TRIPOD-compliant internal validation (bootstrap + 5-fold CV) |
| `opm_taxonomies.py` | Taxonomy comparison (Suppl. Table S6): Bulbar/Spinal, El Escorial, Classic, OPM |
| `opm_staging.py` | Staging comparison (Table 4, Suppl. S7–S8): King's, MiToS, FT9 |
| `opm_table1.py` | Table 1: baseline characteristics stratified by M_3class |
| `opm_figures.py` | All manuscript figures (Figures 1–4 and Supplementary S1, S3–S5) |

---

## Requirements

```
python >= 3.11
numpy >= 1.26
pandas >= 2.2
lifelines >= 0.27
scikit-survival >= 0.22
scikit-learn >= 1.4
scipy >= 1.12
patsy >= 0.5          # for OPM6 natural cubic spline (opm_bootstrap.py only)
matplotlib >= 3.7     # for opm_figures.py
```

Install with:
```bash
pip install lifelines scikit-survival scikit-learn patsy
```

---

## Usage

All scripts assume `db2.xlsx` is in the working directory.
Run each module independently; they all import from `opm_core.py`.

```bash
python opm_table1.py        # Table 1
python opm_models.py        # Table 2, Table 3, O×M diagnostics
python opm_bootstrap.py     # Bootstrap + CV validation (slow: B=200, 10×5 CV)
python opm_taxonomies.py    # Suppl. Table S6
python opm_staging.py       # Table 4, Suppl. S7–S8
python opm_figures.py               # all figures
python opm_figures.py --fig 1 3 S3  # selected figures only
```

---

## Key design decisions

### Common anchor
All taxonomy and staging comparisons include `P_mo_cap + age10 + sexM` as a fixed
anchor.  This means every ΔC / ΔAIC comparison is net of propagation kinetics and
demographics — consistent with the paper's claim that O and M carry independent
phenotypic information.

### Concordance convention
C-index is computed throughout with `sksurv.metrics.concordance_index_censored`,
where **high partial hazard = high risk = early event**.  This is explicit and
avoids the sign-ambiguity when reconstructing concordance from
`lifelines.predict_partial_hazard`.

### Bootstrap optimism (Harrell–Steyerberg)
Optimism = mean(C_boot_on_boot − C_boot_on_original) across B = 200 bootstrap
samples.  C_corrected = C_apparent − optimism.

### M1p exclusion
The PLS-spectrum subgroup (M1p, n = 64) is excluded from the primary Cox model
anchored on symptom onset, to avoid immortal-time bias (the ≥ 48-month qualifying
period is event-free by construction).  A sensitivity analysis including M1p as a
fourth M class is straightforward by calling `load_cohort` without the M_3class
filter and passing `OPM_M_3class` as a 4-level dummy.

---

## Known issues in the original pipeline (fixed here)

1. **Language inconsistency** — originals mixed Italian comments with English code.
   All comments, docstrings, and print output are now in English.

2. **Bootstrap sign convention** — `opm_taxonomies.py` original used
   `lifelines.utils.concordance_index(time, -ph, event)` which correctly
   negates the partial hazard, but `bootstrap_fixed.py` initially did not,
   inflating optimism estimates.  Fixed in `tableS6_final.py` and reproduced
   here via `sksurv` throughout.

3. **Model E specification** — `opm_taxonomies.py` original defined model E
   as OPM_O + M_3class **without** P, whereas the paper's Table S6 reports
   OPM5 (O + M + P + age + sex, k = 12) as the OPM comparator.  All taxonomy
   comparisons here use the full OPM5 anchor.

4. **Scattered scripts** — the original package split logically related analyses
   across `opm_taxonomies.py`, `staging_full.py`, `staging_dAIC_exact.py`, and
   `bootstrap_fixed.py` with duplicated cohort-loading and fitting code.
   The unified pipeline uses a single `opm_core.py` for all shared logic.

6. **Hardcoded numbers in Figure S3** — `regen_figS3.py` had C-index values
   hardcoded manually (e.g. `c=0.7174, lo=0.7155`). If the bootstrap results
   changed, the figure would silently disagree with the table. `opm_figures.py`
   computes all values live from the data.

7. **Inconsistent output paths** — original scripts saved to a mix of
   `/mnt/user-data/outputs/` and `/home/claude/`. All outputs now go to
   `OUTPUT_DIR` (default: current directory), configurable at the top of
   `opm_figures.py`.

8. **Five separate figure files replaced by one** — `opm_figures.py` imports
   from `opm_core.py` and `opm_taxonomies.py` and exposes a `--fig` CLI
   argument for selective regeneration.
