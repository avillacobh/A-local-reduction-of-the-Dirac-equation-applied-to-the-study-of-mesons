"""
Meson Salpeter potentials.

Definitions
-----------
    V_v(r) = bar_V_v  -  (4 alpha / 3) * erf(r / d_v) / r
    V_s(r) = (1/2) bar_V_s * [ erf((r - r_s) / d_s)  -  1 ]
    V_L(r) = V_v(r) - V_s(r)
    V_U(r) = V_v(r) + V_s(r)

Analytic first and second derivatives in r.  Natural units (hbar = c = 1).

V_v(r), V_v'(r), V_v''(r) have apparent 1/r^k singularities that cancel
exactly at r = 0.  For numerical stability we switch to a Taylor series
when r/d_v drops below a small threshold (default 0.01).

This module is scipy-free; it uses math.erf vectorised over numpy arrays.
"""

import math
import numpy as np

SQRT_PI = math.sqrt(math.pi)

# Use scipy.special.erf when available (vectorised, fast); fall back to a
# numpy-vectorised math.erf otherwise.
try:
    from scipy.special import erf as _erf_v                    # noqa: N813
except ImportError:                                            # pragma: no cover
    _erf_v = np.vectorize(math.erf, otypes=[float])


# ============================================================================
#  Vector part  V_v(r) = bar_V_v - (4 alpha / 3) erf(r/d_v) / r
# ============================================================================
def V_v(r, bar_V_v, alpha, d_v, small=1e-6):
    """V_v(r), array-safe and smooth through r = 0.

    Taylor:  erf(x)/x = (2/sqrt(pi)) [1 - x^2/3 + x^4/10 - x^6/42 + ...]
    so V_v(0) = bar_V_v - (8 alpha) / (3 sqrt(pi) d_v).
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d_v
    big = r > small * d_v
    out = np.empty_like(r)
    if np.any(big):
        out[big] = bar_V_v - (4.0 * alpha / 3.0) * _erf_v(x[big]) / r[big]
    if np.any(~big):
        xs = x[~big]
        out[~big] = bar_V_v - (4.0 * alpha / 3.0) \
                    * (2.0 / (SQRT_PI * d_v)) \
                    * (1.0 - xs**2 / 3.0 + xs**4 / 10.0 - xs**6 / 42.0)
    return out


def V_v_prime(r, bar_V_v, alpha, d_v, small=1e-2):
    """
    V_v'(r) = (4 alpha) / (3 r^2)
              * [ erf(r/d_v) - (2 r / (sqrt(pi) d_v)) e^{-r^2/d_v^2} ].

    Near r = 0 the bracket cancels to order r^3, so we use Taylor:
        V_v'(r) = (16 alpha r) / (9 sqrt(pi) d_v^3)
                  * [ 1 - (3/5) x^2 + (3/14) x^4 - ... ],   x = r/d_v.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d_v
    big = x > small
    out = np.empty_like(r)
    if np.any(big):
        rv = r[big]; xv = x[big]
        out[big] = (4.0 * alpha / (3.0 * rv * rv)) * (
            _erf_v(xv) - (2.0 / SQRT_PI) * xv * np.exp(-xv * xv))
    if np.any(~big):
        rv = r[~big]; xs = x[~big]
        out[~big] = (16.0 * alpha * rv / (9.0 * SQRT_PI * d_v**3)) \
                    * (1.0 - (3.0 / 5.0) * xs**2 + (3.0 / 14.0) * xs**4)
    return out


def V_v_double_prime(r, bar_V_v, alpha, d_v, small=1e-2):
    """
    V_v''(r) = - (8 alpha) erf(r/d_v) / (3 r^3)
               + (16 alpha) / (3 sqrt(pi) d_v) * e^{-r^2/d_v^2}
                 * ( 1/r^2 + 1/d_v^2 ).

    Near r = 0:
        V_v''(r) = (1 / (sqrt(pi) d_v^3)) *
                   [ 16 alpha / 9  -  (16 alpha / 5) x^2
                     + (40 alpha / 21) x^4 - ... ].
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    x = r / d_v
    big = x > small
    out = np.empty_like(r)
    if np.any(big):
        rv = r[big]; xv = x[big]
        out[big] = -(8.0 * alpha / 3.0) * _erf_v(xv) / rv**3 \
                   + (16.0 * alpha / (3.0 * SQRT_PI * d_v)) \
                     * np.exp(-xv * xv) * (1.0 / rv**2 + 1.0 / d_v**2)
    if np.any(~big):
        xs = x[~big]
        out[~big] = (1.0 / (SQRT_PI * d_v**3)) * (
            16.0 * alpha / 9.0
            - (16.0 * alpha / 5.0) * xs**2
            + (40.0 * alpha / 21.0) * xs**4)
    return out


# ============================================================================
#  Scalar part  V_s(r) = (1/2) bar_V_s [ erf((r - r_s) / d_s) - 1 ]
# ============================================================================
def V_s(r, bar_V_s, r_s, d_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return 0.5 * bar_V_s * (_erf_v((r - r_s) / d_s) - 1.0)


def V_s_prime(r, bar_V_s, r_s, d_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return (bar_V_s / (SQRT_PI * d_s)) * np.exp(-((r - r_s) / d_s)**2)


def V_s_double_prime(r, bar_V_s, r_s, d_s):
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return -(2.0 * bar_V_s / (SQRT_PI * d_s**3)) * (r - r_s) * \
           np.exp(-((r - r_s) / d_s)**2)


# ============================================================================
#  Combined V_L = V_v - V_s,  V_U = V_v + V_s   (plus derivatives)
# ============================================================================
def make_potential_funcs(bar_V_v, alpha, d_v, bar_V_s, r_s, d_s):
    """
    Returns a dict of callables  V_L, V_L_prime, V_L_pp, V_U, V_U_prime, V_U_pp
    each taking r (array-like) -> ndarray, with the given parameters baked in.
    Plug these directly into H_matrix(..., V_L_func=..., V_L_prime=..., etc.).
    """
    def V_L_(r):  return V_v(r, bar_V_v, alpha, d_v) - V_s(r, bar_V_s, r_s, d_s)
    def V_U_(r):  return V_v(r, bar_V_v, alpha, d_v) + V_s(r, bar_V_s, r_s, d_s)

    def V_L_p(r): return V_v_prime(r, bar_V_v, alpha, d_v) \
                       - V_s_prime(r, bar_V_s, r_s, d_s)
    def V_U_p(r): return V_v_prime(r, bar_V_v, alpha, d_v) \
                       + V_s_prime(r, bar_V_s, r_s, d_s)

    def V_L_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d_v) \
                        - V_s_double_prime(r, bar_V_s, r_s, d_s)
    def V_U_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d_v) \
                        + V_s_double_prime(r, bar_V_s, r_s, d_s)

    def V_v_(r):   return V_v(r, bar_V_v, alpha, d_v)
    def V_v_p(r):  return V_v_prime(r, bar_V_v, alpha, d_v)
    def V_v_pp(r): return V_v_double_prime(r, bar_V_v, alpha, d_v)

    return {
        "V_L": V_L_, "V_L_prime": V_L_p, "V_L_pp": V_L_pp,
        "V_U": V_U_, "V_U_prime": V_U_p, "V_U_pp": V_U_pp,
        "V_v": V_v_, "V_v_prime": V_v_p, "V_v_pp": V_v_pp,
    }


# ============================================================================
#  Quick sanity check: derivatives via finite differences should match analytic
# ============================================================================
if __name__ == "__main__":
    p = dict(bar_V_v=0.5, alpha=0.4, d_v=0.7,
             bar_V_s=-0.3, r_s=1.2, d_s=0.5)
    funcs = make_potential_funcs(**p)

    r = np.linspace(0.001, 5.0, 50)
    h = 1e-5
    for name in ("V_L", "V_U"):
        f   = funcs[name](r)
        fp  = funcs[name + "_prime"](r)
        fpp = funcs[name + "_pp"](r)
        fp_fd  = (funcs[name](r + h) - funcs[name](r - h)) / (2*h)
        fpp_fd = (funcs[name + "_prime"](r + h)
                  - funcs[name + "_prime"](r - h)) / (2*h)
        print(f"{name}:")
        print(f"  max |analytic V'  - finite-diff|: {np.max(np.abs(fp  - fp_fd)):.2e}")
        print(f"  max |analytic V'' - finite-diff|: {np.max(np.abs(fpp - fpp_fd)):.2e}")

    print("V_v at r=0:", V_v(np.array([0.0]), 0.4, 0.4, 0.7))
    print("V_v''(0) theory:", 16*0.4/(9*SQRT_PI*0.7**3))
