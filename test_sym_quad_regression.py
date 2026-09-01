"""
Regression test for the symmetric-quadrature optimisation in ho_primitives.py.

Two checks:

  (1) Math equivalence.  Generate random u (shape N=30 x Ng=8000) and a
      mixed-sign weight w (shape Ng).  Verify that
        _sym_quad(u, w)  ==  u @ (u * w[None,:]).T
      to floating-point tolerance, and that the result is exactly symmetric.

  (2) Physics regression.  Re-evaluate the v1 / charmonium_states_1.csv
      spectrum at the saved LM-polished best-fit params and at n_states=20
      (matching the saved report fit_v1_floor20MeV_charmonium_states_1.txt).
      Confirm:
        - chi^2 reproduces 6.0575 from the saved report (no broken physics)
        - per-state predictions match the saved table to << 1 MeV
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))   # the project root
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np

# ====================================================================
# (1) Math equivalence
# ====================================================================
print("=" * 78)
print(" TEST 1: _sym_quad vs A @ (A * w).T  (mixed-sign random weight)")
print("=" * 78)
from ho_primitives import _sym_quad

rng = np.random.default_rng(20260528)
N, Ng = 30, 8000
A = rng.standard_normal((N, Ng))
w = rng.standard_normal(Ng)        # arbitrary mixed-sign weight

t0 = time.perf_counter()
M_ref = A @ (A * w[None, :]).T
t_ref = time.perf_counter() - t0

t0 = time.perf_counter()
M_sym = _sym_quad(A, w)
t_sym = time.perf_counter() - t0

err_max = np.max(np.abs(M_sym - M_ref))
asym    = np.max(np.abs(M_sym - M_sym.T))
print(f"  shape (N, Ng)              = ({N}, {Ng})")
print(f"  max |M_sym - M_ref|        = {err_max:.3e}")
print(f"  max |M_sym - M_sym.T|      = {asym:.3e}   (should be 0 exactly)")
print(f"  reference  time             = {t_ref*1000:7.2f} ms")
print(f"  symmetric  time             = {t_sym*1000:7.2f} ms")
print(f"  speed-up                    = {t_ref/max(t_sym,1e-9):5.2f} x")

# Loop to average out timing jitter
NREP = 50
t0 = time.perf_counter();
for _ in range(NREP): _ = A @ (A * w[None, :]).T
t_ref_avg = (time.perf_counter() - t0) / NREP
t0 = time.perf_counter()
for _ in range(NREP): _ = _sym_quad(A, w)
t_sym_avg = (time.perf_counter() - t0) / NREP
print(f"  reference avg ({NREP} reps)  = {t_ref_avg*1000:7.2f} ms")
print(f"  symmetric avg ({NREP} reps)  = {t_sym_avg*1000:7.2f} ms")
print(f"  speed-up                    = {t_ref_avg/max(t_sym_avg,1e-9):5.2f} x")

ok_math = err_max < 1e-10 and asym == 0.0
print(f"  --> {'PASS' if ok_math else 'FAIL'}")

# ====================================================================
# (2) Physics regression: v1 / csv1, n_states=20
# ====================================================================
print()
print("=" * 78)
print(" TEST 2: v1 / charmonium_states_1.csv regression at saved best-fit params")
print("=" * 78)

from fit_meson import POTENTIALS, read_states, predict_energies, _compute_sigma_eff

# saved best-fit (from fit_v1_floor20MeV_charmonium_states_1.txt)
params = np.array([
    1.41434,   # bar_V_v
    1.87915,   # alpha
    1.91274,   # d_v
    0.89347,   # bar_V_s
    1.14165,   # r_s
    3.24047,   # d_s
])
pot = POTENTIALS["v1"]
m_q = 1.275
sigma_floor_GeV = 0.020
states = read_states("charmonium_states_1.csv")

# saved predictions for cross-check (MeV)
saved_pred_MeV = np.array([2979.63, 3103.29, 3431.25, 3501.74,
                            3494.38, 3583.31, 3646.30, 3670.66])
saved_chi2 = 6.0575

t0 = time.perf_counter()
pred = predict_energies(params, states, m=m_q, b=2.0, pot=pot,
                        N_grid=8000, n_states=20,
                        variational=True, variational_method="continuous",
                        with_ws=True, ws_sign=-1.0, ws_style="full",
                        E_T_lo=2.0, E_T_hi=5.5)
t_eval = time.perf_counter() - t0
pred_MeV = pred * 1000.0
sig_eff  = _compute_sigma_eff(states, sigma_floor_GeV)
chi2 = float(np.sum(((pred - np.array([s["E_exp"] for s in states])) / sig_eff) ** 2))

print(f"  spectrum eval time         = {t_eval:.2f} s")
print()
print(f"  {'state':<10s} {'pred new':>10s} {'pred saved':>11s} {'delta [MeV]':>12s}")
print("  " + "-" * 50)
SPEC = ["S","P","D","F"]
def lab(s): return f"{s['n']} ^{2*s['S']+1}{SPEC[s['L']]}_{s['J']}"
worst = 0.0
for i, s in enumerate(states):
    d = pred_MeV[i] - saved_pred_MeV[i]
    worst = max(worst, abs(d))
    print(f"  {lab(s):<10s} {pred_MeV[i]:>10.2f} {saved_pred_MeV[i]:>11.2f} "
          f"{d:>+12.4f}")
print()
print(f"  chi^2 new                  = {chi2:.4f}")
print(f"  chi^2 saved (report)       = {saved_chi2:.4f}")
print(f"  |chi^2 new - chi^2 saved|  = {abs(chi2 - saved_chi2):.4e}")
print(f"  worst per-state |delta|    = {worst:.4e} MeV")

ok_chi2 = abs(chi2 - saved_chi2) < 1e-2
ok_pred = worst < 1e-2
print(f"  --> {'PASS' if (ok_chi2 and ok_pred) else 'FAIL'}")

print()
print("=" * 78)
print(f"Overall: {'PASS' if (ok_math and ok_chi2 and ok_pred) else 'FAIL'}")
print("=" * 78)
