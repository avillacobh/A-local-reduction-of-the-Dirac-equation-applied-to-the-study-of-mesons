"""
Grid-sensitivity of the spectrum at the PRODUCTION best-fit parameters.

Unlike diagnose_grid.py, which was written for the collapsed-width v3 solution
(r_s ~ 0.024 GeV^-1), this scans N_grid at the v2 combined DE parameters that
Chapter 8 actually reports, where r_s = 1.437 GeV^-1.

Everything except N_grid is held fixed (b = 2.0, n_states = 30, ws_sign = -1,
ws_style = 'full'), so the spread across the scan is the grid error alone.
"""
import sys, time
import numpy as np
from salpeter_solver import MesonHamiltonianSolver
import meson_potential_v2 as pot_v2

M_Q, B, NSTATES = 1.275, 2.0, 30
WS_SIGN = -1.0

# v2, combined data set, DE stage -- fits/fit_v2_floor20MeV_DE_combined.txt
PAR = dict(bar_V_v=1.92410, alpha=4.11051, d=3.28334,
           bar_V_s=0.86668, r_s=1.43734)

# (label, n, L, S, J)
STATES = [("1 1S0", 1, 0, 0, 0), ("1 3S1", 1, 0, 1, 1),
          ("1 3P0", 1, 1, 1, 0), ("1 3P1", 1, 1, 1, 1),
          ("1 1P1", 1, 1, 0, 1), ("1 3P2", 1, 1, 1, 2),
          ("2 1S0", 2, 0, 0, 0), ("2 3S1", 2, 0, 1, 1)]


class Scaled:
    def __init__(self, f, s): self.f, self.s = f, s
    def __call__(self, r): return self.s * np.asarray(self.f(r), float)


def spectrum(N_grid):
    f = pot_v2.make_potential_funcs(**PAR)
    out = {}
    for (lab, n, L, S, J) in STATES:
        sol = MesonHamiltonianSolver(
            L=L, S=S, J=J,
            V_L_func=f["V_L"], V_U_func=f["V_U"], m=M_Q, b=B,
            n_states=NSTATES, N_grid=N_grid,
            V_L_prime=f["V_L_prime"], V_L_pp=f["V_L_pp"],
            V_U_prime=f["V_U_prime"], V_U_pp=f["V_U_pp"],
            V_v_func=Scaled(f["V_v"], WS_SIGN),
            V_v_prime=Scaled(f["V_v_prime"], WS_SIGN),
            V_v_pp=Scaled(f["V_v_pp"], WS_SIGN),
            ws_style="full")
        out[lab] = 1000.0 * sol.self_consistent_E_T(n - 1, 2.0, 5.5)
    return out


if __name__ == "__main__":
    grids = [int(x) for x in (sys.argv[1:] or [4000, 8000, 16000, 32000])]
    res = {}
    for N in grids:
        t = time.time()
        res[N] = spectrum(N)
        print(f"N_grid = {N:6d}   ({time.time()-t:5.1f} s)", flush=True)

    print("\nstate      " + "".join(f"{N:>14d}" for N in grids) + "     spread")
    for (lab, *_ ) in STATES:
        v = [res[N][lab] for N in grids]
        print(f"{lab:10s}" + "".join(f"{x:14.6f}" for x in v)
              + f"{max(v)-min(v):13.6f}")

    print("\nsplitting  " + "".join(f"{N:>14d}" for N in grids) + "     spread")
    for name, a, b in [("HF(1S)", "1 3S1", "1 1S0"),
                       ("HF(2S)", "2 3S1", "2 1S0"),
                       ("P2-P0",  "1 3P2", "1 3P0"),
                       ("P1-P0",  "1 3P1", "1 3P0"),
                       ("hc-cog", "1 1P1", "1 3P1")]:
        v = [res[N][a] - res[N][b] for N in grids]
        print(f"{name:10s}" + "".join(f"{x:14.6f}" for x in v)
              + f"{max(v)-min(v):13.6f}")
