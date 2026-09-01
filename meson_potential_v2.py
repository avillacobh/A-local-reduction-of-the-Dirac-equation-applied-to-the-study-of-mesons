"""
Variant 2 of the meson Salpeter potentials (5 free parameters):

    V_v(r) = bar_V_v - (4 alpha / 3) * erf(r/d) / r            (with offset)
    V_s(r) = - bar_V_s * exp(-r^2 / r_s^2)                     (Gaussian)
    V_L(r) = V_v(r) - V_s(r)
    V_U(r) = V_v(r) + V_s(r)

Free parameters: bar_V_v, alpha, d, bar_V_s, r_s.

Analytic first and second derivatives.  V_v has 1/r^k apparent singularities
that cancel exactly at r = 0; Taylor fallback used near origin for safety.
V_s is a smooth Gaussian, no fallback needed.

Note: bar_V_v only enters V_v in value (its derivatives are zero), so the
analytic V_v', V_v'' do not depend on bar_V_v.

Natural units (hbar = c = 1); we work in GeV throughout the fit module.
"""

import math
import numpy as np

SQRT_PI = math.sqrt(math.pi)

try:
    from scipy.special import erf as _erf_v
except ImportError:                                            # pragma: no cover
    _erf_v = np.vectorize(math.erf, otypes=[float])


# ============================================================================
#  Vector  V_v(r) = bar_V_v - (4 alpha / 3) * erf(r/d) / r
# ============================================================================
def V_v(r, bar_V_v, alpha, d, small=1e-6):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d
    big = r > small * d
    out = np.empty_like(r)
    if np.any(big):
        out[big] = bar_V_v - (4.0 * alpha / 3.0) * _erf_v(x[big]) / r[big]
    if np.any(~big):
        # erf(x)/x = (2/sqrt(pi)) (1 - x^2/3 + x^4/10 - x^6/42 + ...)
        xs = x[~big]
        out[~big] = bar_V_v - (4.0 * alpha / 3.0) * (2.0 / (SQRT_PI * d)) \
                    * (1.0 - xs**2 / 3.0 + xs**4 / 10.0 - xs**6 / 42.0)
    return out


def V_v_prime(r, bar_V_v, alpha, d, small=1e-2):
    """V_v'(r) does NOT depend on bar_V_v (constant has zero derivative)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d
    big = x > small
    out = np.empty_like(r)
    if np.any(big):
        rv = r[big]; xv = x[big]
        out[big] = (4.0 * alpha / (3.0 * rv * rv)) * (
            _erf_v(xv) - (2.0 / SQRT_PI) * xv * np.exp(-xv * xv))
    if np.any(~big):
        rv = r[~big]; xs = x[~big]
        out[~big] = (16.0 * alpha * rv / (9.0 * SQRT_PI * d**3)) \
                    * (1.0 - (3.0 / 5.0) * xs**2 + (3.0 / 14.0) * xs**4)
    return out


def V_v_double_prime(r, bar_V_v, alpha, d, small=1e-2):
    """V_v''(r) also independent of bar_V_v."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d
    big = x > small
    out = np.empty_like(r)
    if np.any(big):
        rv = r[big]; xv = x[big]
        out[big] = -(8.0 * alpha / 3.0) * _erf_v(xv) / rv**3 \
                   + (16.0 * alpha / (3.0 * SQRT_PI * d)) \
                     * np.exp(-xv * xv) * (1.0 / rv**2 + 1.0 / d**2)
    if np.any(~big):
        xs = x[~big]
        out[~big] = (1.0 / (SQRT_PI * d**3)) * (
            16.0 * alpha / 9.0
            - (16.0 * alpha / 5.0) * xs**2
            + (40.0 * alpha / 21.0) * xs**4)
    return out


# ============================================================================
#  Scalar Gaussian  V_s(r) = - bar_V_s * exp(-r^2 / r_s^2)
# ============================================================================
def V_s(r, bar_V_s, r_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return -bar_V_s * np.exp(-(r / r_s)**2)


def V_s_prime(r, bar_V_s, r_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return (2.0 * bar_V_s * r / r_s**2) * np.exp(-(r / r_s)**2)


def V_s_double_prime(r, bar_V_s, r_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    e = np.exp(-(r / r_s)**2)
    return (2.0 * bar_V_s / r_s**2) * e * (1.0 - 2.0 * (r / r_s)**2)


# ============================================================================
#  V_L = V_v - V_s,    V_U = V_v + V_s
# ============================================================================
def make_potential_funcs(bar_V_v, alpha, d, bar_V_s, r_s):
    """
    Returns dict { V_L, V_L_prime, V_L_pp, V_U, V_U_prime, V_U_pp } with the
    5 parameters baked in.  Plug into H_matrix(...) or MesonHamiltonianSolver.
    """
    def V_L_(r):  return V_v(r, bar_V_v, alpha, d) - V_s(r, bar_V_s, r_s)
    def V_U_(r):  return V_v(r, bar_V_v, alpha, d) + V_s(r, bar_V_s, r_s)
    def V_L_p(r): return V_v_prime(r, bar_V_v, alpha, d) - V_s_prime(r, bar_V_s, r_s)
    def V_U_p(r): return V_v_prime(r, bar_V_v, alpha, d) + V_s_prime(r, bar_V_s, r_s)
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
    }


# ============================================================================
#  Sanity check
# ============================================================================
if __name__ == "__main__":
    p = dict(bar_V_v=0.30, alpha=0.4, d=0.7, bar_V_s=0.5, r_s=1.5)
    f = make_potential_funcs(**p)
    r = np.linspace(0.001, 5.0, 50)
    h = 1e-5
    for name in ("V_L", "V_U"):
        fp_fd  = (f[name](r + h) - f[name](r - h)) / (2*h)
        fpp_fd = (f[name + "_prime"](r + h) - f[name + "_prime"](r - h)) / (2*h)
        print(f"{name}: |V'  - FD| max = {np.max(np.abs(f[name + '_prime'](r) - fp_fd)):.2e}")
        print(f"{name}: |V'' - FD| max = {np.max(np.abs(f[name + '_pp'](r) - fpp_fd)):.2e}")
    print(f"V_v(0) = {V_v(np.array([0.0]), 0.30, 0.4, 0.7)[0]:.4f}")
    print(f"V_v(0) theory = bar_V_v - 8 alpha / (3 sqrt(pi) d) "
          f"= {0.30 - 8*0.4/(3*SQRT_PI*0.7):.4f}")
    print(f"V_s(0) = {V_s(np.array([0.0]), 0.5, 1.5)[0]:.4f}   (theory: -0.5)")
