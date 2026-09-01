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
| `run_all_potentials_de.sh` | runs the nine production fits (3 variants × 3 data sets) |
| `run_all_potentials_lm.sh` | the LM-only stage, kept for comparison |

**Analysis and figures** — everything downstream of the fits:

| file | produces |
|---|---|
| `make_thesis_tables.py` | the spectrum, parameter, splitting and χ² tables |
| `make_thesis_figures.py` | the level, zoom and residual figures |
| `make_potential_figure.py` | the fitted interactions (Fig. 7.1) |
| `make_hf_figure.py` | HF(1S) against the regulator width (Fig. 9.1) |
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
comparison in §9.7 of the thesis uses his published spectrum, not a
re-computation, so nothing in this repository reproduces it. Accordingly the
only reduction style available for the space-vector term is `--ws-style full`,
which is what every fit reported in the thesis used.

**Not used by the thesis.** `meson_potential_coulomb.py`, `potentials.py`,
`diagnose_grid.py`, `eval_v1_csv2_fixed_b.py` and `test.py` are exploratory
and are kept for the record.

## Data

`charmonium_states_1.csv` (set A, the eight states below the open-charm
threshold) and `charmonium_states_2.csv` (set B, the eight above it), with
columns `n,J,L,S,Experimental_value,uncertainty` and masses in MeV.

One point of provenance: **the fits were minimised against the PDG 2024
edition** (Navas *et al.*, Phys. Rev. D **110**, 030001), while the CSVs here
hold the **PDG 2026** values (Takahashi *et al.*, Int. J. Mod. Phys. A **41**,
2630011), which is what the thesis quotes deviations and χ² against. The
parameters were not re-optimised after the update. The effect is at most 0.7 in
χ²; §7.4 of the thesis gives the detail. Re-running the fits on the current
CSVs will therefore not reproduce the reported parameters exactly.

## Reproducing the results

```sh
./run_all_potentials_de.sh          # nine fits -> fits/*.txt  (hours)
python make_thesis_tables.py        # tables from those reports
python make_thesis_figures.py       # figures
```

Production settings are `n_states = 30`, `N_grid = 8000`, a theory floor
`σ_floor = 20 MeV`, `m_c = 1275 MeV`, and the space-vector term `W_s` enabled
with the one-gluon-exchange sign `s = −1`. The differential evolution is
seeded deterministically, but it runs with deferred population updating under
parallel workers, so the trial sequence — and hence the last digits of the
parameters — is not bit-reproducible across machines. The converged minima are
stable well beyond the precision quoted.

A single fit at production settings takes on the order of two hours; the
analysis scripts run in seconds once `fits/` exists.
