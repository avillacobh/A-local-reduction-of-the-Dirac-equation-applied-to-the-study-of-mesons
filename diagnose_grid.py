"""
Grid-convergence diagnostic for the v3 best-fit charmonium parameters.

Background
----------
The v3 fit collapsed the scalar Gaussian width to  r_s ~ 0.024 GeV^-1
(~ 0.0047 fm).  The uniform radial grid built by `ho_primitives.build_grid`
has  dr ~ 0.005-0.010 GeV^-1  at the default N_grid=4000, which means
~2-5 grid points across the FWHM of V_s.  Inside H_full_matrix every
spin-dependent kernel goes through `deriv5` (5-point central difference)
TWO OR THREE TIMES, so the high-order derivatives of A_L^2 A_U' / r near
the origin -- which are exactly what generates the hyperfine splitting in
this parametrisation -- can be numerically aliased.

This script evaluates the 8 fitted charmonium states at the FIXED best-fit
parameters but varying N_grid in {4000, 8000, 16000, 32000}.  If the
hyperfine splitting (J/psi - eta_c) survives a 8x grid refinement within,
say, ~ 1 MeV, the small-r_s solution is numerically robust and we can
defend it physically as a regularised contact term.  If it drifts, the
fit was sitting on grid noise.

Run:
    python diagnose_grid.py
    python diagnose_grid.py --n-grids 4000 32000 65536
    python diagnose_grid.py --r-s 0.5      # override to test physical r_s
"""

import argparse
import csv
import os
import time

import numpy as np

from salpeter_solver import MesonHamiltonianSolver
import meson_potential_v3 as _pot_v3

HERE       = os.path.dirname(os.path.abspath(__file__))
SPEC_LABEL = ["S", "P", "D", "F", "G", "H"]


# v3 best-fit values from the LM-polished DE on charmonium_states_1.csv
BEST_FIT_V3 = dict(
    alpha   = 2.33193,
    d       = 1.64180,
    bar_V_s = 0.93553,
    r_s     = 0.02385,
)

DEFAULT_M_Q  = 1.275
DEFAULT_B    = 2.0
DEFAULT_NSTA = 15


def read_states(csv_path):
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out.append(dict(
                n = int(row["n"]), J = int(row["J"]),
                L = int(row["L"]), S = int(row["S"]),
                E_exp = float(row["Experimental_value"]) / 1000.0,
                sigma = float(row["uncertainty"])         / 1000.0,
            ))
    return out


def term_label(s):
    return f"{s['n']} ^{2*s['S']+1}{SPEC_LABEL[s['L']]}_{s['J']}"


def predict_at_grid(params_v3, states, m, b, n_states, N_grid, r_max=None,
                    E_T_lo=2.0, E_T_hi=5.5, verbose=False):
    """Predict E_T (GeV) for each state at the given N_grid / r_max."""
    funcs = _pot_v3.make_potential_funcs(m_q=m, **params_v3)
    unique = sorted({(s["L"], s["S"], s["J"]) for s in states})
    solvers = {
        (L, S, J): MesonHamiltonianSolver(
            L=L, S=S, J=J,
            V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
            V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
            V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
            m=m, b=b, n_states=n_states,
            N_grid=N_grid, r_max=r_max)
        for (L, S, J) in unique
    }
    pred = np.full(len(states), np.nan)
    for k, s in enumerate(states):
        try:
            pred[k] = solvers[(s["L"], s["S"], s["J"])].self_consistent_E_T(
                n_level=s["n"] - 1, E_T_lo=E_T_lo, E_T_hi=E_T_hi)
        except Exception as exc:
            if verbose:
                print(f"  {term_label(s)}  FAILED: {exc}")
    return pred


def describe_grid(N_grid, m, b, n_states, L=0, r_max=None):
    """Compute dr and grid coverage."""
    import math
    N = n_states + 20
    if r_max is None:
        r_max_eff = b * (4.0 + 1.5 * math.sqrt(2 * (N - 1) + L + 1.5))
    else:
        r_max_eff = r_max
    dr = r_max_eff / N_grid
    return r_max_eff, dr


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="charmonium_states_1.csv",
                   help="State CSV (default charmonium_states_1.csv).")
    p.add_argument("--n-grids", type=int, nargs="+",
                   default=[4000, 8000, 16000, 32000],
                   help="N_grid values to scan (default 4000 8000 16000 32000).")
    p.add_argument("--r-max", type=float, default=None,
                   help="Outer radius override (GeV^-1). Default: auto.")
    p.add_argument("--m-q", type=float, default=DEFAULT_M_Q)
    p.add_argument("--b",   type=float, default=DEFAULT_B)
    p.add_argument("--n-states", type=int, default=DEFAULT_NSTA)
    p.add_argument("--alpha",   type=float, default=BEST_FIT_V3["alpha"])
    p.add_argument("--d",       type=float, default=BEST_FIT_V3["d"])
    p.add_argument("--bar-V-s", type=float, default=BEST_FIT_V3["bar_V_s"])
    p.add_argument("--r-s",     type=float, default=BEST_FIT_V3["r_s"])
    args = p.parse_args()

    states = read_states(os.path.join(HERE, args.csv))
    params = dict(alpha=args.alpha, d=args.d,
                  bar_V_s=args.bar_V_s, r_s=args.r_s)

    print("=" * 90)
    print("Grid-convergence diagnostic  (v3 charmonium)")
    print("=" * 90)
    print(f"  Quark mass m_q       = {args.m_q:.4f} GeV")
    print(f"  HO scale b           = {args.b:.4f} GeV^-1   (~{args.b*0.1973:.3f} fm)")
    print(f"  HO basis n_states    = {args.n_states}")
    print(f"  Parameters:")
    for k, v in params.items():
        print(f"    {k:10s} = {v:+.6f}")
    print(f"  States:              {args.csv}  ({len(states)} states)")
    print(f"  Grid scan N_grid in: {args.n_grids}")
    print()

    print(f"  {'N_grid':>8s}  {'r_max':>10s}  {'dr':>10s}  "
          f"{'dr / r_s':>10s}  {'pts in r_s':>10s}")
    print("  " + "-" * 60)
    for N in args.n_grids:
        r_max_eff, dr = describe_grid(N, args.m_q, args.b, args.n_states,
                                       L=0, r_max=args.r_max)
        print(f"  {N:>8d}  {r_max_eff:>10.4f}  {dr:>10.5f}  "
              f"{dr/args.r_s:>10.3f}  {args.r_s/dr:>10.2f}")
    print()

    # Predict at every requested N_grid
    pred_table = {}
    for N in args.n_grids:
        t0 = time.time()
        pred_table[N] = predict_at_grid(
            params, states,
            m=args.m_q, b=args.b, n_states=args.n_states,
            N_grid=N, r_max=args.r_max,
            E_T_lo=2.0, E_T_hi=5.5)
        dt = time.time() - t0
        print(f"  N_grid={N:>6d}   wallclock = {dt:6.1f} s")
    print()

    # Per-state convergence table (MeV)
    print("Per-state convergence  (energies in MeV)")
    print("-" * 90)
    headers = "  ".join(f"{n:>10d}" for n in args.n_grids)
    print(f"  {'state':<10s}  {headers}   {'exp':>9s}   {'drift_max':>10s}")
    for k, s in enumerate(states):
        row = [pred_table[N][k] * 1000.0 for N in args.n_grids]
        finite_row = [x for x in row if np.isfinite(x)]
        drift = (max(finite_row) - min(finite_row)) if finite_row else float("nan")
        cells = "  ".join(f"{v:10.2f}" if np.isfinite(v) else f"{'NaN':>10s}"
                          for v in row)
        print(f"  {term_label(s):<10s}  {cells}   "
              f"{s['E_exp']*1000:9.2f}   {drift:10.2f}")
    print()

    # Hyperfine splittings
    def find_state(states, n, L, S, J):
        for k, s in enumerate(states):
            if (s["n"] == n and s["L"] == L
                    and s["S"] == S and s["J"] == J):
                return k
        return None

    print("Key splittings  (MeV)")
    print("-" * 90)
    pairs = [
        ("1^3S_1 - 1^1S_0 (J/psi - eta_c)",     (1, 0, 1, 1), (1, 0, 0, 0)),
        ("2^3S_1 - 2^1S_0 (psi' - eta_c')",      (2, 0, 1, 1), (2, 0, 0, 0)),
        ("1^3P_2 - 1^3P_0 (chi_c2 - chi_c0)",    (1, 1, 1, 2), (1, 1, 1, 0)),
    ]
    for name, hi, lo in pairs:
        ihi, ilo = find_state(states, *hi), find_state(states, *lo)
        if ihi is None or ilo is None:
            continue
        line = f"  {name:<35s}"
        for N in args.n_grids:
            d = (pred_table[N][ihi] - pred_table[N][ilo]) * 1000.0
            line += f"  {d:>10.2f}"
        d_exp = (states[ihi]["E_exp"] - states[ilo]["E_exp"]) * 1000.0
        line += f"   exp={d_exp:8.2f}"
        print(line)
    print()
    print("Interpretation")
    print("-" * 90)
    print("  drift_max < ~1 MeV per state         -> fit is grid-converged "
          "(physical, if r_s makes sense).")
    print("  drift_max ~ several to >10 MeV       -> small-r_s minimum is "
          "partly a grid-resolution artefact.")
    print("=" * 90)


if __name__ == "__main__":
    main()
