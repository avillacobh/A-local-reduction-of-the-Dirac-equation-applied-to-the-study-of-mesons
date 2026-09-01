"""
Evaluate v1 on charmonium_states_2.csv with per-STATE variational b*.

Instead of one b*(L,S,J) per channel (which minimises the channel's
ground-state energy, n_radial=0), we pick a separate b*(state) for each
physical state by minimising THAT state's predicted energy:

    b*_k  =  argmin_b  E_T(L_k, S_k, J_k, n_radial=k; b)

By MacDonald's theorem this gives the lowest upper bound on EACH state
individually, so the predictions are basis-converged at finite n_states
in a per-state sense.  Useful because csv2 contains highly excited radials
(3S, 3P, 4S, 4P) that would be poorly served by a channel-ground-state b*.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))   # the project root
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
from scipy.optimize import minimize_scalar

from fit_meson import POTENTIALS, read_states, _compute_sigma_eff, _ScaledFunc
from salpeter_solver import MesonHamiltonianSolver

# ----- given best-fit params (v1) -----
params = np.array([
    1.74590,   # bar_V_v
    2.44654,   # alpha
    2.39059,   # d_v
    0.35121,   # bar_V_s
    10.13233,  # r_s
    1.34180,   # d_s
])
m_q = 1.275
sigma_floor_GeV = 0.020
ws_sign  = -1.0
ws_style = "full"

# Basis / grid settings
N_STATES_BASIS = 30
N_GRID = 8000
B_LO, B_HI = 0.5, 4.0       # search interval for b*
E_T_LO, E_T_HI = 2.0, 5.5   # self-consistency bracket

pot = POTENTIALS["v1"]
states = read_states("charmonium_states_2.csv")

# ----- Pre-build potential callables -----
funcs = pot["make_funcs"](params, m_q)
Vv  = funcs["V_v"]
Vvp = funcs.get("V_v_prime")
Vvpp = funcs.get("V_v_pp")
if ws_sign == +1.0:
    ws_kwargs = dict(V_v_func=Vv, V_v_prime=Vvp, V_v_pp=Vvpp)
else:
    ws_kwargs = dict(
        V_v_func  = _ScaledFunc(Vv,   ws_sign),
        V_v_prime = _ScaledFunc(Vvp,  ws_sign) if Vvp  is not None else None,
        V_v_pp    = _ScaledFunc(Vvpp, ws_sign) if Vvpp is not None else None,
    )

H_kwargs_common = dict(
    V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
    V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
    N_grid=N_GRID,
    ws_style=ws_style,
    **ws_kwargs,
)


# ----- Per-state variational: pick b* that minimises E_T(n_radial; b) -----
PENALTY = 1e6
def find_b_for_state(L, S, J, n_radial, n_states_basis=N_STATES_BASIS):
    """Return (b_star, E_T_at_b_star_in_GeV)."""
    def fobj(b):
        try:
            solver = MesonHamiltonianSolver(
                L=L, S=S, J=J,
                V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
                m=m_q, b=float(b), n_states=n_states_basis,
                **H_kwargs_common)
            return solver.self_consistent_E_T(n_radial, E_T_LO, E_T_HI)
        except (ValueError, RuntimeError):
            return PENALTY
    res = minimize_scalar(fobj, bounds=(B_LO, B_HI), method="bounded",
                          options=dict(xatol=0.02, maxiter=30))
    return float(res.x), float(res.fun)


# ----- Loop over states; build prediction table -----
SPEC = ["S", "P", "D", "F"]
def lab(s): return f"{s['n']} ^{2*s['S']+1}{SPEC[s['L']]}_{s['J']}"

print("=" * 80)
print(" v1 on charmonium_states_2.csv with PER-STATE variational b*")
print(f" n_states (basis) = {N_STATES_BASIS}, N_grid = {N_GRID}, W_s full, "
      f"ws_sign = {ws_sign:+.0f}")
print("=" * 80)
print()
print(" Params:")
for nm, v in zip(pot["param_names"], params):
    print(f"   {nm:10s} = {v:+10.5f}")
print()

# Also do reference at b=2.0 (no variational) for direct comparison
print(" Running per-state variational searches...")
print()

n_states_csv = len(states)
b_star    = np.zeros(n_states_csv)
pred_var  = np.zeros(n_states_csv)
pred_b2   = np.zeros(n_states_csv)
t0 = time.time()
for i, s in enumerate(states):
    L, S, J = s["L"], s["S"], s["J"]
    n_radial = s["n"] - 1
    # per-state variational
    b_s, E_var = find_b_for_state(L, S, J, n_radial)
    b_star[i] = b_s
    pred_var[i] = E_var
    # reference at fixed b = 2.0
    solver_b2 = MesonHamiltonianSolver(
        L=L, S=S, J=J,
        V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
        m=m_q, b=2.0, n_states=N_STATES_BASIS, **H_kwargs_common)
    try:
        pred_b2[i] = solver_b2.self_consistent_E_T(n_radial, E_T_LO, E_T_HI)
    except (ValueError, RuntimeError):
        pred_b2[i] = float("nan")
    print(f"   {lab(s):<10s}  b*={b_s:.4f}   "
          f"E_var={E_var*1000:.2f} MeV   E_b=2={pred_b2[i]*1000:.2f} MeV")
print()
print(f" Wallclock: {time.time()-t0:.1f}s")

# ----- chi^2 with per-state variational predictions -----
sig_eff = _compute_sigma_eff(states, sigma_floor_GeV)
exp_GeV = np.array([s["E_exp"] for s in states])
res_var = (pred_var - exp_GeV) / sig_eff
res_b2  = (pred_b2  - exp_GeV) / sig_eff
chi2_var = float(np.sum(res_var ** 2))
chi2_b2  = float(np.sum(res_b2  ** 2))
dof = n_states_csv - len(params)

# ----- Comparison table -----
print()
print(" Per-state comparison")
print(" " + "-" * 90)
print(f"  {'state':<10s} {'b*':>7s} {'pred(b*)':>9s} {'pred(b=2)':>10s} "
      f"{'exp':>9s} {'sigma':>7s} "
      f"{'(b*-exp)':>10s} {'(b=2-exp)':>11s} "
      f"{'pull(b*)':>9s} {'pull(b=2)':>10s}")
print(" " + "-" * 90)
for i, s in enumerate(states):
    print(f"  {lab(s):<10s} {b_star[i]:7.3f} "
          f"{pred_var[i]*1000:9.2f} {pred_b2[i]*1000:10.2f} "
          f"{exp_GeV[i]*1000:9.2f} {sig_eff[i]*1000:7.2f} "
          f"{(pred_var[i]-exp_GeV[i])*1000:+10.2f} "
          f"{(pred_b2[i] -exp_GeV[i])*1000:+11.2f} "
          f"{res_var[i]:+9.2f} {res_b2[i]:+10.2f}")
print(" " + "-" * 90)
print()
print(f"  chi^2 (per-state variational) = {chi2_var:.4f}    "
      f"chi^2/dof = {chi2_var/dof:.4f}")
print(f"  chi^2 (fixed b = 2.0)         = {chi2_b2:.4f}    "
      f"chi^2/dof = {chi2_b2/dof:.4f}")
print(f"  dof = {dof}  ({n_states_csv} states - {len(params)} params)")
print()
print("=" * 80)
print(" Interpretation")
print("=" * 80)
print("  Per-state variational b* gives the lowest upper bound on EACH state's")
print("  energy (MacDonald applied per-eigenvalue).  Compare with fixed b=2.0:")
print()
print("  - If pred(b*) << pred(b=2) for the high-n_radial states (3S, 4S, 4P,")
print("    3P), then fixed-b=2 was artificially HIGH and per-state variational")
print("    is the honest, basis-converged prediction.")
print("  - If pred(b*) ~ pred(b=2), b=2 was already near optimal per-state.")
print("=" * 80)
