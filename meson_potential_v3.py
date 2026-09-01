"""
Variant 3 of the meson Salpeter potentials (4 free parameters):

    V_v(r) = bar_V_v - (4 alpha / 3) * erf(r/d) / r,      bar_V_v = 2 m_q - bar_V_s
    V_s(r) = - bar_V_s * exp(-r^2 / r_s^2)
    V_L(r) = V_v(r) - V_s(r)
    V_U(r) = V_v(r) + V_s(r)

The constraint  bar_V_v = 2 m_q - bar_V_s  fixes the constant offset of V_v
in terms of the scalar depth and the (input) charm mass m_q, so bar_V_v is
NOT a free parameter.  Free parameters: alpha, d, bar_V_s, r_s.

Asymptotics:
    V_v(r -> inf) = bar_V_v        = 2 m_q - bar_V_s
    V_s(r -> inf) = 0
    V_L(r -> inf) = bar_V_v        = 2 m_q - bar_V_s
    V_U(r -> inf) = bar_V_v        = 2 m_q - bar_V_s
    A_L(r -> inf) = 1 / (m_q + E_T/2 - bar_V_v/2) = 2 / (E_T + bar_V_s)
        (using bar_V_v = 2 m_q - bar_V_s, so m_q + E_T/2 - bar_V_v/2
         = (E_T + bar_V_s)/2 -- m_q drops out)

Analytic derivatives are identical to v2 (since bar_V_v is constant in r);
this module just bakes in the 2 m_q - bar_V_s relation.
"""

import math
import numpy as np

from meson_potential_v2 import (
    V_v, V_v_prime, V_v_double_prime,
    V_s, V_s_prime, V_s_double_prime,
)


# ============================================================================
#  V_L = V_v - V_s,  V_U = V_v + V_s   with  bar_V_v = 2 m_q - bar_V_s
# ============================================================================
def make_potential_funcs(alpha, d, bar_V_s, r_s, m_q):
    """
    Returns dict { V_L, V_L_prime, V_L_pp, V_U, V_U_prime, V_U_pp,
                   V_v, V_v_prime, V_v_pp } with the constraint
    bar_V_v = 2 m_q - bar_V_s  baked in.  V_v keys expose the bare vector
    potential and its first/second derivatives, so the W_s (spatial vector)
    contribution can be built with V_v^s = V_v (OGE consistency).

    Parameters (4 free):
        alpha    : effective coupling
        d        : vector range          [GeV^-1]
        bar_V_s  : scalar Gaussian depth [GeV]
        r_s      : scalar Gaussian width [GeV^-1]
    Hyperparameter (fixed):
        m_q      : (constituent) quark mass [GeV] -- e.g. 1.275 for charm
    """
    bar_V_v = 2.0 * m_q - bar_V_s

    def V_L_(r):   return V_v(r, bar_V_v, alpha, d) - V_s(r, bar_V_s, r_s)
    def V_U_(r):   return V_v(r, bar_V_v, alpha, d) + V_s(r, bar_V_s, r_s)
    def V_L_p(r):  return V_v_prime(r, bar_V_v, alpha, d) - V_s_prime(r, bar_V_s, r_s)
    def V_U_p(r):  return V_v_prime(r, bar_V_v, alpha, d) + V_s_prime(r, bar_V_s, r_s)
    def V_L_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d) \
                        - V_s_double_prime(r, bar_V_s, r_s)
    def V_U_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d) \
                        + V_s_double_prime(r, bar_V_s, r_s)
    def V_v_(r):   return V_v(r, bar_V_v, alpha, d)
    def V_v_p(r):  return V_v_prime(r, bar_V_v, alpha, d)
    def V_v_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d)

    return {
        "V_L": V_L_, "V_L_prime": V_L_p, "V_L_pp": V_L_pp,
        "V_U": V_U_, "V_U_prime": V_U_p, "V_U_pp": V_U_pp,
        "V_v": V_v_, "V_v_prime": V_v_p, "V_v_pp": V_v_pp,
        "bar_V_v_derived": bar_V_v,                # for reporting / diagnostics
    }


if __name__ == "__main__":
    p = dict(alpha=0.4, d=0.7, bar_V_s=0.5, r_s=1.5, m_q=1.275)
    f = make_potential_funcs(**p)
    print(f"Derived bar_V_v = 2 m_q - bar_V_s = "
          f"{2*p['m_q'] - p['bar_V_s']:.4f} GeV")
    print(f"Stored in dict   = {f['bar_V_v_derived']:.4f} GeV")

    r = np.linspace(0.001, 5.0, 50)
    h = 1e-5
    for name in ("V_L", "V_U"):
        fp_fd  = (f[name](r + h) - f[name](r - h)) / (2 * h)
        fpp_fd = (f[name + "_prime"](r + h) - f[name + "_prime"](r - h)) / (2 * h)
        print(f"{name}: max |V'  - FD| = {np.max(np.abs(f[name + '_prime'](r) - fp_fd)):.2e}")
        print(f"{name}: max |V'' - FD| = {np.max(np.abs(f[name + '_pp'](r) - fpp_fd)):.2e}")

    # Check asymptotics
    r_inf = np.array([20.0])
    print(f"V_L(r=20) = {f['V_L'](r_inf)[0]:.4f}   "
          f"theory = bar_V_v = {f['bar_V_v_derived']:.4f}")
    print(f"V_U(r=20) = {f['V_U'](r_inf)[0]:.4f}   "
          f"theory = bar_V_v = {f['bar_V_v_derived']:.4f}")
