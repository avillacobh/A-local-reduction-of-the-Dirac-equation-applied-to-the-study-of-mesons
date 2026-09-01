"""
Diagnostic: is the Simpson integration faithful for the matrix elements that
build the hyperfine splitting?

Test A:  full Salpeter splitting with Simpson only (already known: ~7 MeV).
Test B:  the diagnostic that actually tells us if integration is to blame.
         Compute <0L0| nabla^2 V_v |0L0> THREE ways:
            1. Simpson on the standard build_grid (what your code uses)
            2. scipy.quad over (0, inf) with breakpoint hints     <-- ground truth
            3. Closed-form analytic
                nabla^2 V_v(r) = (16 alpha / 3 d^3 sqrt(pi)) exp(-r^2/d^2)
                <0|exp(-r^2/d^2)|0>_HO = d^3 / (b^2 + d^2)^(3/2)
         If all three agree, integration is exonerated and the missing
         splitting is in the operator structure (or its coefficients).

Skipping the full scipy_quad Salpeter pipeline:  it builds a NxN matrix
for every one of ~20 multiplicative functions per H(E_T), calling
scipy.quad once per matrix element  ->  tens of thousands of calls
per channel.  Not the right tool here.
"""
import os, sys, time, math, warnings
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
from scipy.integrate import quad as sp_quad
from ho_primitives import ho_radial_u, build_grid, simpson_weights
from meson_potential_v3 import make_potential_funcs
from salpeter_solver import MesonHamiltonianSolver

# v3 best-fit parameters from fit_v3_charmonium_states_1.txt
m_q = 1.275
b   = 2.0
n_states = 15
params = dict(alpha=2.04495, d=1.22118,
              bar_V_s=1.00708, r_s=0.10000, m_q=m_q)
funcs  = make_potential_funcs(**params)

print(f"V_L(0)  = {float(funcs['V_L'](np.array([1e-6]))[0]):+.4f} GeV")
print(f"V_U(0)  = {float(funcs['V_U'](np.array([1e-6]))[0]):+.4f} GeV")
print(f"V_L(2)  = {float(funcs['V_L'](np.array([2.0]))[0]):+.4f} GeV")
print(f"V_U(2)  = {float(funcs['V_U'](np.array([2.0]))[0]):+.4f} GeV\n")


# ----------------------------------------------------------------------
# Test A: full Salpeter splitting with Simpson (baseline)
# ----------------------------------------------------------------------
print("=" * 60)
print("Test A:  full Salpeter splitting (Simpson)")
print("=" * 60)
t0 = time.time()
E = {}
for (S, J, lab) in [(0, 0, '1S0'), (1, 1, '3S1')]:
    slv = MesonHamiltonianSolver(
        L=0, S=S, J=J,
        V_L_func=funcs['V_L'], V_U_func=funcs['V_U'],
        V_L_prime=funcs['V_L_prime'], V_L_pp=funcs['V_L_pp'],
        V_U_prime=funcs['V_U_prime'], V_U_pp=funcs['V_U_pp'],
        m=m_q, b=b, n_states=n_states, quadrature='simpson')
    E[lab] = slv.self_consistent_E_T(n_level=0, E_T_lo=2.0, E_T_hi=5.5)
t1 = time.time()
split = (E['3S1'] - E['1S0']) * 1000
print(f"  1S0   = {E['1S0']*1000:8.2f} MeV")
print(f"  3S1   = {E['3S1']*1000:8.2f} MeV")
print(f"  split = {split:+8.2f} MeV   (experiment: +113.00)")
print(f"  wall  = {t1-t0:.1f} s\n")


# ----------------------------------------------------------------------
# Test B: matrix element of nabla^2 V_v three ways
# ----------------------------------------------------------------------
print("=" * 60)
print("Test B:  <0L0| nabla^2 V_v |0L0>  three ways")
print("=" * 60)

alpha = params['alpha']
d     = params['d']
analytic_amplitude = (16.0 * alpha) / (3.0 * d**3 * math.sqrt(math.pi))
print(f"  prefactor (16 alpha / 3 d^3 sqrt(pi)) = "
      f"{analytic_amplitude:.6f}  GeV^3\n")

# (1) Simpson on the standard build_grid
r_grid, dr, _, _ = build_grid(0, 20, b, N_grid=4000)
weights = simpson_weights(len(r_grid), dr)
u0_grid = ho_radial_u(0, 0, r_grid, b)
nab2_Vv_grid = analytic_amplitude * np.exp(-(r_grid / d) ** 2)
me_simpson = float(np.sum(u0_grid * u0_grid * nab2_Vv_grid * weights))
print(f"  (1) Simpson on default grid       = {me_simpson:.6f}  GeV")
print(f"      r_grid: r_min={r_grid[0]:.4e}, r_max={r_grid[-1]:.2f}, "
      f"dr={dr:.4e},  N={len(r_grid)}")

# (2) scipy.quad with breakpoint hints (TRUE 0-to-inf, ground truth)
def integrand_scalar(r):
    u = float(ho_radial_u(0, 0, np.array([r]), b)[0])
    return u * u * analytic_amplitude * math.exp(-(r / d) ** 2)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    v1, _ = sp_quad(integrand_scalar, 0.0, 10.0,
                    epsabs=1e-14, epsrel=1e-12, limit=2000,
                    points=[params['r_s'], d, b])
    v2, _ = sp_quad(integrand_scalar, 10.0, np.inf,
                    epsabs=1e-14, epsrel=1e-12, limit=2000)
me_quad = v1 + v2
print(f"  (2) scipy.quad (true 0-inf)       = {me_quad:.6f}  GeV")

# (3) Closed-form analytic
# HO ground state with L=0:  u_0(r) = N r exp(-r^2/2 b^2),  N^2 = 4/(sqrt(pi) b^3)
#   <0|exp(-r^2/d^2)|0>_HO = d^3 / (b^2 + d^2)^(3/2)
gauss_overlap = (d ** 3) / ((b * b + d * d) ** 1.5)
me_analytic   = analytic_amplitude * gauss_overlap
print(f"  (3) Closed-form analytic          = {me_analytic:.6f}  GeV")

print()
print(f"  Simpson    - analytic  = {me_simpson - me_analytic:+.3e}  GeV   "
      f"(rel {(me_simpson - me_analytic)/me_analytic:+.2e})")
print(f"  scipy.quad - analytic  = {me_quad    - me_analytic:+.3e}  GeV   "
      f"(rel {(me_quad    - me_analytic)/me_analytic:+.2e})")
print()
print("  Interpretation:")
print("    If Simpson agrees with analytic to <= 1e-4 relative, then the")
print("    integration is faithful and the small splitting is NOT a numerical")
print("    artifact.  In that case the operator structure (or its coefficient)")
print("    is what's missing the bulk of the 113 MeV.")
