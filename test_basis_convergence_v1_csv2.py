"""
Basis-convergence test for v1 / charmonium_states_2.csv.

Idea
----
The fit reported chi^2 = 26.35 (chi^2/dof = 13.2) with n_states = 20 and
per-channel variational b*.  Three states drive that residual:
    4 ^3S_1   pull = +2.42  (pred 4271.32 vs exp 4222.50)
    4 ^3P_1   pull = +3.54  (pred 4363.67 vs exp 4286.00)
    1 ^3D_1   pull = -1.57  (pred 3742.29 vs exp 3773.70)

The (0,1,1) channel carries 4 radial states (n_r = 0..3).  A single
variational b* in that channel cannot be optimal for both 1^3S_1 and
4^3S_1, so the highly excited radial states are most vulnerable to
basis truncation.

This script re-evaluates the SAME LM-polished parameters at
n_states = 20, 30, 40 and re-optimises b* per channel each time.
If the high-n_r pulls collapse, the chi^2 = 26 is basis-truncation
artefact.  If they barely move, it is a real model deficiency.

Runtime: a few minutes per n_states value on a modern Mac
(20 -> 30 -> 40 grows roughly as n^3 in the eigensolve).
"""
import sys, time, os
# Run from this directory so `import fit_meson` etc. work
HERE = os.path.dirname(os.path.abspath(__file__))   # the project root
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np

from fit_meson import (POTENTIALS, read_states, predict_energies,
                       _compute_sigma_eff)

# ---------- v1 / csv2 LM-polished best-fit params (from saved report) ----
params = np.array([
    1.81636,   # bar_V_v
    3.18366,   # alpha
    2.29752,   # d_v
    1.89914,   # bar_V_s
    2.25298,   # r_s
    1.16425,   # d_s
])
m_q = 1.275                       # charm quark mass [GeV]
sigma_floor_GeV = 0.020           # same floor as the fit

pot = POTENTIALS["v1"]
states = read_states("charmonium_states_2.csv")

# Spectroscopic labels for the report
SPEC = ["S", "P", "D", "F"]
def label(s):
    return f"{s['n']} ^{2*s['S']+1}{SPEC[s['L']]}_{s['J']}"

# Common kwargs to predict_energies (full W_s, variational, continuous b*)
common = dict(
    m=m_q, b=2.0, pot=pot,
    N_grid=4000,
    variational=True, variational_method="continuous",
    with_ws=True, ws_sign=-1.0, ws_style="full",
    E_T_lo=2.0, E_T_hi=5.5,
)

N_VALUES = [20, 30, 40]
table = {}
sigma_eff = _compute_sigma_eff(states, sigma_floor_GeV)  # GeV
exp_MeV = np.array([s["E_exp"] for s in states]) * 1000.0
sig_MeV = sigma_eff * 1000.0

print("Evaluating spectrum at n_states =", N_VALUES, "...")
for n_basis in N_VALUES:
    t0 = time.time()
    pred = predict_energies(params, states, n_states=n_basis, **common)
    t1 = time.time()
    b_per_ch = predict_energies._last_b_per_channel.copy()
    pred_MeV = pred * 1000.0
    chi2 = float(np.sum(((pred - np.array([s["E_exp"] for s in states])) / sigma_eff) ** 2))
    table[n_basis] = dict(pred_MeV=pred_MeV, chi2=chi2, b_star=b_per_ch,
                          time_s=t1 - t0)
    print(f"  n_states={n_basis}: chi^2={chi2:.3f}  time={t1-t0:.1f}s")

# ---------- Report ------------------------------------------------------
print()
print("=" * 92)
print(" Basis-convergence sweep: v1 / charmonium_states_2.csv (LM-polished params)")
print("=" * 92)
print()
print(" Variational b* per channel (GeV^-1)")
print(" " + "-" * 60)
all_ch = sorted({(s["L"], s["S"], s["J"]) for s in states})
header = "    (L,S,J)    " + "  ".join(f"n={n:>3d}" for n in N_VALUES)
print(header)
for ch in all_ch:
    row = f"   ({ch[0]},{ch[1]},{ch[2]})     " + "   ".join(
        f"{table[n]['b_star'][ch]:6.3f}" for n in N_VALUES)
    print(row)

print()
print(" Per-state predictions [MeV]")
print(" " + "-" * 92)
hdr = (f" {'state':<10s} {'exp':>9s} {'sigma':>7s}  " +
       "  ".join(f"{'n='+str(n):>9s}" for n in N_VALUES) +
       "   " + "  ".join(f"{'pull(n='+str(n)+')':>11s}" for n in N_VALUES))
print(hdr)
print(" " + "-" * 92)
for i, s in enumerate(states):
    lab = label(s)
    row = f" {lab:<10s} {exp_MeV[i]:9.2f} {sig_MeV[i]:7.2f}  "
    for n in N_VALUES:
        row += f"{table[n]['pred_MeV'][i]:9.2f}  "
    row += " "
    for n in N_VALUES:
        pull = (table[n]['pred_MeV'][i] - exp_MeV[i]) / sig_MeV[i]
        row += f"{pull:+11.2f}  "
    print(row)

print()
print(" Total chi^2 / dof  (dof = 8 - 6 = 2)")
print(" " + "-" * 60)
for n in N_VALUES:
    print(f"   n_states = {n:>3d}:  chi^2 = {table[n]['chi2']:7.3f}   "
          f"chi^2/dof = {table[n]['chi2']/2:7.3f}   "
          f"wallclock = {table[n]['time_s']:5.1f} s")

print()
print(" Movements from n_states=20 to n_states=40 [MeV]")
print(" " + "-" * 60)
for i, s in enumerate(states):
    d = table[40]['pred_MeV'][i] - table[20]['pred_MeV'][i]
    flag = "<-- excited radial (sensitive)" if s["n"] >= 3 else ""
    print(f"   {label(s):<10s}  delta = {d:+7.2f} MeV   {flag}")

print()
print("=" * 92)
print(" Interpretation")
print("=" * 92)
print("  * If high-n_r predictions move >> a few MeV between n=20 and n=40, then")
print("    n_states=20 was not converged for those states and re-fitting with a")
print("    larger basis will lower chi^2.  Worth re-running --n-states 40.")
print()
print("  * If high-n_r predictions move <~ 1 MeV between n=20 and n=40, then the")
print("    fit is already basis-converged; the residual chi^2 = 26 is model-")
print("    intrinsic (likely threshold / open-channel effects above DD-bar).")
print("    Re-fitting with bigger basis will not help meaningfully.")
print("=" * 92)
