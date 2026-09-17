# Charmonium spectrum from a local reduction of the Breit equation

Code and data behind the master's thesis *A local reduction of the Dirac
equation for the study of mesons* / *Una reducción local de la ecuación de
Dirac para el estudio de mesones* (Universidad Nacional de Colombia, advisor
M. De Sanctis).

The model solves a local, energy-dependent three-dimensional reduction
`K†(D₁ + D₂ + W)K` of the two-body Dirac equation, in which the reduction
operator `K` depends on the interaction itself and does not factorise into a
product of one-particle operators. The reduced Hamiltonian is projected onto
the coupled basis `|n,L,S,J⟩`, evaluated in a harmonic-oscillator basis through
closed-form radial integrals, and the resulting nonlinear eigenvalue problem
`λₙ[H(E_T)] = E_T` is solved self-consistently.

## Requirements

Python 3, NumPy and SciPy. Nothing else is imported outside the standard
library; `potentials.py` alone also uses Matplotlib.

```sh
git clone https://github.com/avillacobh/A-local-reduction-of-the-Dirac-equation-applied-to-the-study-of-mesons.git
cd A-local-reduction-of-the-Dirac-equation-applied-to-the-study-of-mesons
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy matplotlib
```

The reported fits were run under Python 3.13.5, NumPy 2.3.2 and SciPy 1.16.0 on
arm64 macOS. Of these only the SciPy version affects exact reproducibility, for
the reason given under [Cost and reproducibility](#cost-and-reproducibility).

## Layout

**Core** — the model itself:

| file | contents |
|---|---|
| `ho_primitives.py` | closed-form harmonic-oscillator radial integrals and the spin–angular factors |
| `H_full_matrix.py` | assembly of the reduced Hamiltonian `H(E_T)` in the HO basis |
| `salpeter_solver.py` | the self-consistent solver for `λₙ[H(E_T)] = E_T`, with the per-channel variational scale `b*` |
| `meson_potential.py` | interaction variant **v1** (6 parameters) |
| `meson_potential_v2.py` | interaction variant **v2** (5 parameters) — the best fit |
| `meson_potential_v3.py` | interaction variant **v3** (4 parameters, balance relation imposed) |

**Fitting** — produces the reports the thesis tables are generated from:

| file | contents |
|---|---|
| `fit_meson.py` | the three-stage fit: differential evolution → Levenberg–Marquardt → variational polish |
| `run_all_potentials_de.sh` | runs the nine reported fits (3 variants × 3 data sets) |
| `run_all_potentials_lm.sh` | the LM-only stage, kept for comparison |

**Analysis and figures** — everything downstream of the fits:

| file | produces |
|---|---|
| `make_thesis_tables.py` | the spectrum, parameter, splitting and χ² tables |
| `make_thesis_figures.py` | the level, zoom and residual figures |
| `make_potential_figure.py` | the fitted interactions (Fig. 6.1) |
| `make_hf_figure.py` | HF(1S) against the regulator width (Fig. 8.1) |
| `make_chi2map_figure.py` | the χ² map |
| `results_extras.py` | radii, the `W_s` on/off study, and `b*` recomputation |
| `discussion_studies.py` | the operator-by-operator decomposition of the `h_c` inversion |
| `grid_scan_production.py`, `closed_vs_product.py`, `profile_scan.py` | numerical validation (Appendix F) |

**Tests and verification**:

`verify_spin_factors.py`, `verify_LL_ordering.py`,
`test_ho_basis_numerical_quality.py`, `test_sym_quad_regression.py`,
`test_basis_convergence_v1_csv2.py`, `test_variational_continuous.py`.

**Notes**: `H_construction.md` documents the projected form of the Hamiltonian
term by term; `V_T_matrix_elements.md` the tensor matrix elements.

The factorised reduction of De Sanctis is **not** implemented here. The
comparison in §8.7 of the thesis uses his published spectrum, not a
re-computation, so nothing in this repository reproduces it. One consequence is
worth stating: `fit_meson.py` exposes `--ws-style {full,desanctis}` for the
space-vector term and its default is `desanctis`, but that path lazily imports
a module this repository does not ship. **Pass `--ws-style full` explicitly**,
as `run_all_potentials_de.sh` does and as every fit reported in the thesis did.

**Not used by the thesis.** `meson_potential_coulomb.py`, `potentials.py`,
`diagnose_grid.py`, `eval_v1_csv2_fixed_b.py` and `test.py` are exploratory
and are kept for the record.

## Data

`charmonium_states_1.csv` (set A, the eight states below the open-charm
threshold) and `charmonium_states_2.csv` (set B, the eight above it), with
columns `n,J,L,S,Experimental_value,uncertainty` and masses in MeV.

To fit a different system, write CSVs with the same columns and pass
`--csv-prefix` and `--m-q` (for bottomonium: `--csv-prefix bottomonium_states
--m-q 4.18`). Every file matching `<prefix>*.csv` in the working directory is
picked up, one fit per file, plus the joint fit when `--combined` is given.

## Usage

Everything is driven by `fit_meson.py`. It reads the CSVs from the working
directory, fits one interaction variant, and writes a plain-text report per
data set into `--out-dir`.

### Quick start

```sh
./run_all_potentials_de.sh          # the nine reported fits -> fits/*.txt  (hours)
python make_thesis_tables.py        # tables, from those reports
python make_thesis_figures.py       # figures
```

### A single fit

The settings used for every fit reported in the thesis, written out in full —
the script above passes exactly these:

```sh
python fit_meson.py \
    --potential v2 --mode de --combined \
    --with-ws --ws-style full --ws-sign -1 \
    --b 2.0 --n-grid 8000 --n-states 30 \
    --popsize 20 --maxiter 80 --workers -1 --max-nfev 200 \
    --polish-variational --variational-method continuous \
    --strict-bounds --sigma-floor 0.020 \
    --out-dir fits
```

**The built-in defaults are not those settings.** `fit_meson.py` defaults to
`--potential v3`, `--n-states 25`, `--n-grid 4000`, `--max-nfev 1000`,
`--popsize 15` and `--ws-style desanctis`, which are development values. Pass
the flags above, or use the shell script, to reproduce the thesis.

For a first look that finishes in minutes rather than hours, drop the basis and
the optimiser to development sizes:

```sh
python fit_meson.py --potential v3 --mode lm --ws-style full \
    --n-states 15 --n-grid 4000 --out-dir /tmp/try
```

The spectrum will be a few MeV off the converged one — that is the basis
truncation measured in Appendix F of the thesis, not a bug.

### What a run produces

Reports go to `--out-dir` as

```
fit_{variant}_floor20MeV_{stage}_{dataset}.txt
```

with `stage` ∈ {`DE`, `LM`} and `dataset` ∈ {`charmonium_states_1`,
`charmonium_states_2`, `combined`}; console logs go to `logs/` under the same
directory. Each report opens with the run settings, so a report is
self-describing:

```
Run settings
------------------------------------------------------------------------------
  Quark mass m              = 1275.00  MeV
  HO oscillator scale b     = VARIATIONAL per channel (see table below)
  Basis size n_states       = 30
  Theory uncertainty floor  = 20.00  MeV
  Optimisation              = differential evolution + LM polish + variational polish
  Spatial-vector W_s        = ENABLED  (ws_sign=-1.0, ws_style='full')
```

and continues with the parameterisation, the best-fit parameters with their 1σ
errors from the covariance matrix, the computed spectrum against experiment,
the splittings, the per-channel `b*`, and the wallclock of each stage. The
`make_thesis_*.py` scripts parse these reports; they never recompute anything,
so the tables and the figures cannot drift from the fits they describe.

### Command-line reference

**Model and data**

| flag | default | meaning |
|---|---|---|
| `--potential {v1,v2,v3,coulomb}` | `v3` | interaction variant; the thesis reports v1, v2 and v3 |
| `--m-q GEV` | `1.275` | quark mass in GeV; `4.18` for bottomonium |
| `--csv-prefix STR` | `charmonium_states_` | glob prefix for the input CSVs |
| `--combined` | off | also fit all CSVs jointly to one parameter set |
| `--only-combined` | off | skip the per-CSV fits; implies `--combined` |

**Space-vector term `W_s`**

| flag | default | meaning |
|---|---|---|
| `--with-ws` / `--no-ws` | on | include the spatial part of one-gluon exchange |
| `--ws-sign {+1,-1}` | `-1` | sign of `V_v^s` relative to `V_v`; `-1` is the OGE-consistent choice |
| `--ws-style {full,desanctis}` | `desanctis` | reduction style. **Use `full`** — the default path needs a module this repository does not ship |

**Basis and grid**

| flag | default | meaning |
|---|---|---|
| `--n-states N` | `25` | HO basis size; the thesis uses 30 |
| `--b GEV^-1` | `2.0` | fixed oscillator scale, unless `--variational` |
| `--variational` | off | per-channel variational `b*(L,S,J)` instead of the fixed `--b` |
| `--variational-method {continuous,grid}` | `continuous` | bounded Brent search, or a scan over `--b-grid` |
| `--b-grid B [B ...]` | 10 points in (0.5, 4.0) | grid for the variational search, with `grid` |
| `--polish-variational` | off | after the main fit, re-polish the parameters in the variational basis |
| `--n-grid N` | `4000` | radial grid points; the thesis uses 8000. Raise to 16000–32000 when `r_s` is small |
| `--r-max GEV^-1` | auto | outer radius; set explicitly only for convergence studies |

**Optimiser**

| flag | default | meaning |
|---|---|---|
| `--mode {de,lm}` | `de` | differential evolution + LM polish, or LM only |
| `--popsize N` | `15` | DE population |
| `--maxiter N` | `80` | DE generations |
| `--max-nfev N` | `1000` | evaluation cap for the LM stage |
| `--workers N` | `-1` | DE parallelism; `-1` all cores, `1` serial |
| `--strict-bounds` | off | tighten the lower bounds on the range parameters, so the fit cannot collapse into sub-grid widths |

**Weighting**

| flag | default | meaning |
|---|---|---|
| `--sigma-floor GEV` | `0.020` | theory uncertainty floor added to every state |
| `--hf-sigma-floor GEV` | — | a separate floor for the hyperfine-defining states `1¹S₀` and `1³S₁` |
| `--sigma-floor-overrides STR` | — | per-state floors, as `'n,L,S,J:floor_MeV'` separated by semicolons |

**Output**

| flag | default | meaning |
|---|---|---|
| `--out-dir DIR` | script directory | where the `fit_*.txt` reports go |
| `--out-tag STR` | — | tag appended to the report filenames, to keep parallel runs from overwriting each other |

### Verification

Each of these is standalone and takes seconds to minutes:

```sh
python verify_spin_factors.py              # spin-angular factors against closed forms
python verify_LL_ordering.py               # the symmetric-average ordering of |Phi_LL>
python test_ho_basis_numerical_quality.py  # Gram matrix of the truncated basis
python test_sym_quad_regression.py         # closed forms vs direct quadrature
python test_basis_convergence_v1_csv2.py   # spectrum against n_states
python test_variational_continuous.py      # continuous b* against a brute-force scan
python closed_vs_product.py                # closed forms vs products of truncated matrices
```

They are the numerical claims of Appendix F of the thesis, in runnable form.
Start with `verify_spin_factors.py` and `test_sym_quad_regression.py`: between
them they cover the algebra and the radial integrals, and they finish in
seconds.

### Cost and reproducibility

A single fit at the settings above takes on the order of two hours; the
analysis scripts run in seconds once `fits/` exists. Cost scales roughly as
`n_states²` × `n_grid` per Hamiltonian build, and linearly in
`popsize × maxiter`, so a quick exploration is best done by lowering
`--n-states` first.

The differential evolution is seeded deterministically, but it runs with
deferred population updating under parallel workers, so the trial sequence —
and hence the last digits of the parameters — is not bit-reproducible across
machines, and changes with the SciPy version. Pass `--workers 1` for a strictly
serial run. The converged minima are stable well beyond the precision quoted.

### If something goes wrong

| symptom | cause |
|---|---|
| `ImportError` naming a `desanctis` module | `--ws-style` left at its default; pass `--ws-style full` |
| `maximum number of function evaluations exceeded` in the log | raise `--max-nfev` |
| a fitted width collapses to a value below the grid spacing | add `--strict-bounds`, or raise `--n-grid` |
| no fit reports appear | no `<prefix>*.csv` in the working directory; check `--csv-prefix` |
| the spectrum sits a few MeV off the published one | basis truncation at low `--n-states`; the thesis uses 30 |

## Data provenance

**The fits were minimised against the PDG 2024 edition** (Navas *et al.*, Phys.
Rev. D **110**, 030001), while the CSVs here hold the **PDG 2026** values
(Takahashi *et al.*, Int. J. Mod. Phys. A **41**, 2630011), which is what the
thesis quotes deviations and χ² against. The parameters were not re-optimised
after the update. The effect is at most 0.7 in χ²; §6.3 of the thesis gives the
detail. Re-running the fits on the current CSVs will therefore not reproduce
the reported parameters exactly.

## Licence

MIT — see `LICENSE`.
