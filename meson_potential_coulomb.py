"""
Variant "coulomb" of the meson Salpeter potentials (4 free parameters):

    V_v(r) = bar_V_v - (4 alpha / 3) / r    (pure Coulomb + constant offset)
    V_s(r) = - bar_V_s * exp(-r^2 / r_s^2)  (Gaussian)
    V_L = V_v - V_s,    V_U = V_v + V_s

Free parameters: bar_V_v, alpha, bar_V_s, r_s   (4 params).

Why is bar_V_v needed even though it looks like a zero-of-energy shift?
In the non-relativistic limit it WOULD be degenerate with 2 m_q (a global
shift of V_v just shifts the spectrum).  But Salpeter is relativistic:
the kinetic factor

    A_X(r) = 1 / (m + E_T/2 - V_X(r) / 2)

is NON-LINEAR in V_X, so a shift bar_V_v -> bar_V_v + c is NOT absorbed
by m_q -> m_q + c/2 -- the spectrum changes.  Empirically, dropping
bar_V_v makes the self-consistent root finder fail for most channels:
V_L(infinity) = 0 leaves no asymptotic "wall" and the eigenvalue
lambda_n[H(E_T)] - E_T has no sign change in any reasonable bracket.

The lesson: my (Claude's) earlier near-degeneracy argument was right only
in the non-relativistic limit, and we are doing the *relativistic*
problem.  bar_V_v stays free.

Asymptotics
-----------
    V_v(r -> inf) = bar_V_v
    V_s(r -> inf) = 0
    V_L(r -> inf) = bar_V_v               (vector dominates the long range)
    V_U(r -> inf) = bar_V_v
    A_L(r -> inf) = 1 / (m_q + E_T/2 - bar_V_v/2)

CAVEAT
------
V_v(r), V_v'(r), V_v''(r) are SINGULAR at r = 0 (1/r, 1/r^2, 1/r^3).
For the simple <i|V_v|j> matrix elements the integrand u_i u_j (V_v) goes
like r^{2L+2} / r  =  r^{2L+1}, which is integrable for L >= 0.  But for
the Salpeter K_LL^dag(...) K_LL blocks the intermediate functions
A_L'(r) ~ 1/r^2 and (A_L')^2 A_U / r^2 ~ 1/r^6 diverge faster than the
wavefunctions can tame -- giving formally divergent integrals for L=0
and L=1.  In practice the cutoff is set by the grid's r_min (uniform
grid: r_min ~ b/N_grid ~ 4e-3 in default units), so the integrals are
finite but quadrature-cutoff dependent.

In short: the "coulomb" variant is included for completeness and for
comparison against the regularised (erf-smoothed) v1/v2/v3 forms, but
the latter are strongly recommended for production fits.

The (modest) cost of the singularities is that:
  (a) `optimize_b_variational` may struggle to bracket E_T^(0)(b) for some
      channels because of the cutoff-dependent eigenvalues -- you may need
      a wider --b-bracket or to set b explicitly;
  (b) The fit residuals may be biased relative to the smoothed potentials.

Natural units (hbar = c = 1); we work in GeV throughout the fit module.
"""

import numpy as np

# Reuse the Gaussian scalar from v2 (same convention as v3)
from meson_potential_v2 import V_s, V_s_prime, V_s_double_prime


# ============================================================================
#  Pure Coulomb vector  V_v(r) = bar_V_v - (4 alpha / 3) / r
# ============================================================================
def V_v(r, bar_V_v, alpha):
    """V_v(r) = bar_V_v - (4 alpha / 3) / r.  SINGULAR at r = 0."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return bar_V_v - (4.0 * alpha / 3.0) / r


def V_v_prime(r, bar_V_v, alpha):
    """V_v'(r) = (4 alpha / 3) / r^2.  bar_V_v drops out (constant)."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return (4.0 * alpha / 3.0) / (r * r)


def V_v_double_prime(r, bar_V_v, alpha):
    """V_v''(r) = -(8 alpha / 3) / r^3.  bar_V_v drops out."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    return -(8.0 * alpha / 3.0) / (r ** 3)


# ============================================================================
#  V_L = V_v - V_s,    V_U = V_v + V_s   (4 free parameters)
# ============================================================================
def make_potential_funcs(bar_V_v, alpha, bar_V_s, r_s):
    """4 free params (bar_V_v, alpha, bar_V_s, r_s).  No m_q is needed
    because nothing in this potential references the quark mass."""
    def V_L_(r):   return V_v(r, bar_V_v, alpha)        - V_s(r, bar_V_s, r_s)
    def V_U_(r):   return V_v(r, bar_V_v, alpha)        + V_s(r, bar_V_s, r_s)
    def V_L_p(r):  return V_v_prime(r, bar_V_v, alpha)  - V_s_prime(r, bar_V_s, r_s)
    def V_U_p(r):  return V_v_prime(r, bar_V_v, alpha)  + V_s_prime(r, bar_V_s, r_s)
    def V_L_pp(r): return V_v_double_prime(r, bar_V_v, alpha) \
                          - V_s_double_prime(r, bar_V_s, r_s)
    def V_U_pp(r): return V_v_double_prime(r, bar_V_v, alpha) \
                          + V_s_double_prime(r, bar_V_s, r_s)

    def V_v_(r):   return V_v(r, bar_V_v, alpha)
    def V_v_p(r):  return V_v_prime(r, bar_V_v, alpha)
    def V_v_pp(r): return V_v_double_prime(r, bar_V_v, alpha)

    return {
        "V_L": V_L_, "V_L_prime": V_L_p, "V_L_pp": V_L_pp,
        "V_U": V_U_, "V_U_prime": V_U_p, "V_U_pp": V_U_pp,
        "V_v": V_v_, "V_v_prime": V_v_p, "V_v_pp": V_v_pp,
    }


# ============================================================================
#  Quick smoke test: derivatives via FD vs analytic.  Stay away from r=0.
# ============================================================================
if __name__ == "__main__":
    p = dict(bar_V_v=1.0, alpha=0.4, bar_V_s=0.5, r_s=1.5)
    f = make_potential_funcs(**p)

    r = np.linspace(0.1, 5.0, 50)            # avoid the singularity at r=0
    h = 1e-5
    for name in ("V_L", "V_U"):
        fp_fd  = (f[name](r + h)            - f[name](r - h))            / (2 * h)
        fpp_fd = (f[name + "_prime"](r + h) - f[name + "_prime"](r - h)) / (2 * h)
        print(f"{name}: max |V'  - FD| = "
              f"{np.max(np.abs(f[name + '_prime'](r) - fp_fd)):.2e}")
        print(f"{name}: max |V'' - FD| = "
              f"{np.max(np.abs(f[name + '_pp'](r) - fpp_fd)):.2e}")
    r_inf = np.array([20.0])
    print(f"V_L(r=20) = {f['V_L'](r_inf)[0]:.4f}   "
          f"theory = bar_V_v - (4 alpha/3)/r = "
          f"{p['bar_V_v'] - (4*p['alpha']/3) / r_inf[0]:.4f}")
