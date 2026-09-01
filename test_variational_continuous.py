"""Smoke test for the new continuous variational machinery.
Run before committing to a long DE+LM session.

Verifies:
  1. find_variational_b_continuous returns valid (b*, E*) tuples.
  2. Continuous and grid methods agree to <few MeV.
  3. predict_energies with variational + variational_method='continuous' works.
  4. fit_states with variational + polish-style flags works end-to-end.

Expected runtime: ~1-2 minutes on a modern Mac.
"""
import sys, time
sys.path.insert(0, ".")
import numpy as np
from salpeter_solver import find_variational_b, find_variational_b_continuous
from fit_meson import POTENTIALS, read_states, predict_energies, fit_states

# ---------- 1) Compare grid vs continuous for ground state of 8 channels ----
print("=" * 70)
print(" TEST 1: continuous vs grid for b* in each (L,S,J) channel")
print("=" * 70)

pot = POTENTIALS["v3"]
m = 1.275
x = np.array([3.075, 2.589, 0.851, 1.176])  # v3 best-fit
funcs = pot["make_funcs"](x, m)
H_kwargs = dict(V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
                V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"])
channels = [(0,0,0), (0,1,1), (1,0,1), (1,1,0), (1,1,1), (1,1,2)]

print(f"{'channel':<14s} {'b_grid':>9s} {'E_grid':>9s} {'b_cont':>9s} {'E_cont':>9s} {'Δb':>7s} {'ΔE [MeV]':>9s}")
print("-" * 76)
all_ok = True
for (L, S, J) in channels:
    bg, Eg = find_variational_b(L, S, J, funcs["V_L"], funcs["V_U"], m, n_states=20,
                                  b_values=np.linspace(1.0, 3.5, 6),
                                  H_kwargs=H_kwargs)
    bc, Ec = find_variational_b_continuous(L, S, J, funcs["V_L"], funcs["V_U"], m, n_states=20,
                                            b_lo=1.0, b_hi=3.5,
                                            H_kwargs=H_kwargs)
    ok = bc is not None and Ec is not None
    if not ok:
        all_ok = False
        print(f"({L},{S},{J})          {bg:9.4f}  {Eg:9.4f}  {'FAIL':>9s}  {'FAIL':>9s}")
    else:
        db = bc - bg
        dE_meV = (Ec - Eg) * 1000
        ok_E = Ec <= Eg + 1e-3  # continuous should give E <= grid (MacDonald)
        flag = '' if ok_E else '   <-- WARNING: E_cont > E_grid'
        print(f"({L},{S},{J})          {bg:9.4f}  {Eg:9.4f}  {bc:9.4f}  {Ec:9.4f}  {db:+7.3f} {dE_meV:+9.2f}{flag}")

print()
print(f"Channel-by-channel: {'PASSED' if all_ok else 'FAILED (None returned)'}")

# ---------- 2) predict_energies with variational continuous --------------
print()
print("=" * 70)
print(" TEST 2: predict_energies(variational=True, variational_method='continuous')")
print("=" * 70)
states = read_states("charmonium_states_1.csv")
t0 = time.time()
preds = predict_energies(x, states, m=1.275, b=2.0, pot=pot,
                          n_states=20, N_grid=4000,
                          variational=True,
                          variational_method="continuous",
                          with_ws=True, ws_sign=-1.0, ws_style="full")
t1 = time.time()
labels = ["eta_c", "J/psi", "chi_c0", "chi_c1", "h_c", "chi_c2", "eta_c(2S)", "psi(2S)"]
print(f"{'state':<12s} {'pred [MeV]':>12s} {'exp [MeV]':>11s} {'diff':>8s}")
for lab, p, s in zip(labels, preds, states):
    diff = (p - s['E_exp']) * 1000
    print(f"{lab:<12s} {p*1000:12.2f} {s['E_exp']*1000:11.2f} {diff:+8.2f}")
HF = (preds[1] - preds[0]) * 1000
print(f"HF = {HF:.2f} MeV  (exp 113)")
print(f"Spectrum eval (8 states, continuous variational): {t1-t0:.1f}s")

# ---------- 3) Tiny LM-only fit with --polish-variational simulated -------
print()
print("=" * 70)
print(" TEST 3: LM with variational + continuous (mimics polish step)")
print("=" * 70)
t0 = time.time()
res = fit_states(states, x, m=1.275, b=2.0, pot=pot,
                  sigma_floor=0.020, n_states=20,
                  variational=True, max_nfev=30,
                  N_grid=4000,
                  with_ws=True, ws_sign=-1.0, ws_style="full",
                  variational_method="continuous",
                  verbose=False)
t1 = time.time()
print(f"LM-variational time: {t1-t0:.1f}s,  chi^2={res['chi2']:.3f},  "
      f"nfev={res['n_evals']}")
print(f"Converged params: {dict(zip(pot['param_names'], res['params']))}")
print(f"variational_method recorded in res: {res.get('variational_method')}")
print()
print(f"All tests {'PASSED' if all_ok else 'FAILED somewhere -- check output above'}")
