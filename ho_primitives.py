"""
Closed-form primitives for HO-basis matrix elements of operators of the
form (kinetic) * (multiplicative function) * (kinetic).

Two spin-1/2 particles, definite L, total spin S in {0, 1}, J in {|L-S|,...,L+S}.
Natural units hbar = c = 1.  The HO has length scale b = bar_r.

KEY IDEA.  We use the HO Hamiltonian H_HO = p^2/2 + r^2/(2 b^4) and
p^2 = 2 H_HO - r^2 / b^4 to express every  {p^2,f}, {p^4,f}, p^2 f p^2
as a closed combination of the multiplicative-function matrices

    F[i,j]    = int u_i u_j  f(r)  dr,
    R2F[i,j]  = int u_i u_j  r^2 f(r)  dr,
    R4F[i,j]  = int u_i u_j  r^4 f(r)  dr,
    Rfp[i,j]  = int u_i u_j  r f'(r)  dr,

with  E_n = (2n + L + 3/2) / b^2  the HO eigenvalues.  No matrix products
of operators (P2 @ F etc.) appear anywhere.  The single sandwich
(p.r) f (r.p) is reduced to the radial integral
    int (r u_i' - u_i)(r u_j' - u_j) f(r) dr,
using analytic u and u' from the Laguerre derivative recurrence.

Pure numpy (no scipy).
"""

import math
import numpy as np

# Prefer scipy (fast, vectorised, numerically stable for high n).  Fall back
# to hand-written numpy/math implementations if scipy is not installed.
try:
    from scipy.special import (eval_genlaguerre as _sp_genlaguerre,
                                gammaln as _sp_gammaln)
    from scipy.integrate import simpson as _sp_simpson, quad as _sp_quad
    HAS_SCIPY = True
except ImportError:                                               # pragma: no cover
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
#  Generalised Laguerre polynomial  L_n^alpha(x)
# ---------------------------------------------------------------------------
def gen_laguerre(n, alpha, x):
    """L_n^{alpha}(x).  Uses scipy.special.eval_genlaguerre when available;
    otherwise the stable forward recurrence
       (k+1) L_{k+1}^a = (2k+1+a-x) L_k^a  -  (k+a) L_{k-1}^a."""
    x = np.asarray(x, dtype=float)
    if n < 0:
        return np.zeros_like(x)
    if HAS_SCIPY:
        return _sp_genlaguerre(n, alpha, x)
    if n == 0:
        return np.ones_like(x)
    L0 = np.ones_like(x)
    L1 = 1.0 + alpha - x
    if n == 1:
        return L1
    for k in range(1, n):
        L0, L1 = L1, ((2 * k + 1 + alpha - x) * L1 - (k + alpha) * L0) / (k + 1)
    return L1


def _lgamma_scalar(x):
    """log Gamma(x) for scalar argument."""
    return _sp_gammaln(x) if HAS_SCIPY else math.lgamma(x)


# ---------------------------------------------------------------------------
#  3D isotropic HO radial wavefunction and its analytic r-derivative
# ---------------------------------------------------------------------------
def ho_radial_u(n, L, r, b):
    """u_{nL}(r) = r R_{nL}(r), normalised by int |u|^2 dr = 1.

       R_{nL}(r) = sqrt(2 n! / (b^3 Gamma(n+L+3/2)))
                   * (r/b)^L * exp(-r^2/2b^2) * L_n^{L+1/2}(r^2/b^2)
    """
    log_N = 0.5 * (math.log(2.0) + _lgamma_scalar(n + 1)
                                  - _lgamma_scalar(n + L + 1.5)) - 1.5 * math.log(b)
    x = r / b
    return math.exp(log_N) * r * x**L * np.exp(-0.5 * x * x) \
           * gen_laguerre(n, L + 0.5, x * x)


def ho_radial_du(n, L, r, b):
    """Analytic du_{nL}/dr via  d/dx L_n^a(x) = -L_{n-1}^{a+1}(x):

         u' = N (r/b)^L exp(-r^2/2b^2) [ (1+L-r^2/b^2) L_n^{L+1/2}(r^2/b^2)
                                         - 2 r^2/b^2 * L_{n-1}^{L+3/2}(r^2/b^2) ].
    """
    log_N = 0.5 * (math.log(2.0) + _lgamma_scalar(n + 1)
                                  - _lgamma_scalar(n + L + 1.5)) - 1.5 * math.log(b)
    x = r / b
    x2 = x * x
    Ln   = gen_laguerre(n,     L + 0.5, x2)
    Lnm1 = gen_laguerre(n - 1, L + 1.5, x2)        # zero for n = 0
    return math.exp(log_N) * x**L * np.exp(-0.5 * x2) \
           * ((1.0 + L - x2) * Ln - 2.0 * x2 * Lnm1)


def ho_E(L, N, b):
    """HO eigenvalues  E_n = (2n + L + 3/2) / b^2,  n = 0..N-1."""
    return np.array([(2 * n + L + 1.5) / b**2 for n in range(N)])


def build_grid(L, N, b, r_max=None, N_grid=4000):
    """Returns (r, dr, u, du), all numpy arrays.  u, du have shape (N, N_grid)."""
    if r_max is None:
        r_max = b * (4.0 + 1.5 * math.sqrt(2 * (N - 1) + L + 1.5))
    r  = np.linspace(r_max / N_grid, r_max, N_grid)
    dr = r[1] - r[0]
    u  = np.vstack([ho_radial_u(n,  L, r, b) for n in range(N)])
    du = np.vstack([ho_radial_du(n, L, r, b) for n in range(N)])
    return r, dr, u, du


# ---------------------------------------------------------------------------
#  scipy.integrate.quad-based matrix element for true (0, infinity) limits
# ---------------------------------------------------------------------------
def func_matrix_quad(u_callables, f_func, n_states, epsabs=1e-12, epsrel=1e-10,
                     limit=200):
    """Matrix  F[i,j] = int_0^infty u_i(r) u_j(r) f(r) dr  via
    `scipy.integrate.quad` with infinite upper limit (no truncation).

    Parameters
    ----------
    u_callables : list of callables of length >= n_states.  Each
                  `u_callables[n](r)` returns u_{nL}(r) at scalar r.
    f_func      : callable f(r) returning the multiplicative function value.
    n_states    : int.  Build the n_states x n_states upper triangle of F
                  and mirror.
    epsabs, epsrel, limit : forwarded to scipy.integrate.quad.

    Notes
    -----
    Adaptive QUADPACK quadrature on (0, inf).  Each <i|f|j> is one call (so
    ~n_states*(n_states+1)/2 calls per multiplicative function).  Accuracy is
    typically 1e-10 .. 1e-12, no truncation at finite r_max — but it is
    significantly slower than Simpson on a precomputed grid.  Use mainly for
    accuracy spot-checks or for one-shot calculations where exactness matters
    more than speed.
    """
    if not HAS_SCIPY:
        raise RuntimeError("func_matrix_quad needs scipy.integrate.quad")
    F = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i, n_states):
            val, _ = _sp_quad(
                lambda r, i=i, j=j: u_callables[i](r) * u_callables[j](r) * f_func(r),
                0.0, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit)
            F[i, j] = val
            if i != j:
                F[j, i] = val
    return F


# ---------------------------------------------------------------------------
#  Composite Simpson weights (vectorised)
# ---------------------------------------------------------------------------
def simpson_weights(n_grid, dr):
    """Return weights w of length n_grid such that sum(w * y) = int y dr
    by composite Simpson's 1/3 rule (with trapezoid on a trailing odd
    interval, if any)."""
    if n_grid < 2:
        return np.zeros(n_grid)
    if n_grid == 2:
        return 0.5 * dr * np.ones(2)
    even_end = n_grid if (n_grid - 1) % 2 == 0 else n_grid - 1
    w = np.zeros(n_grid)
    w[0] = w[even_end - 1] = dr / 3.0
    w[1:even_end - 1:2] = 4.0 * dr / 3.0
    w[2:even_end - 2:2] = 2.0 * dr / 3.0
    if even_end != n_grid:
        w[even_end - 1] += 0.5 * dr
        w[even_end]      = 0.5 * dr
    return w


def simpson(y, dr):
    """1-D Simpson quadrature: sum(w * y).  Convenience wrapper."""
    return float(np.dot(simpson_weights(len(y), dr), y))


# ---------------------------------------------------------------------------
#  Multiplicative-function matrix  F[i,j] = int u_i u_j f(r) dr
# ---------------------------------------------------------------------------
def func_matrix(u, f_grid, weights):
    """Symmetric NxN matrix  F[i,j] = int u_i(r) u_j(r) f(r) dr.

    `weights` is a 1-D array of length len(f_grid): the quadrature weights
    matched to whatever sampling grid was used for `u` and `f_grid`.  For a
    uniform Simpson grid pass `simpson_weights(N_grid, dr)`.

    Vectorised: one matmul (a numerical integration sum, NOT an operator
    product in the truncated basis)."""
    weighted = u * (f_grid * weights)[None, :]
    return u @ weighted.T


# ---------------------------------------------------------------------------
#  Closed-form combinators for {p^2, f}, p^2 f p^2, {p^4, f}
#  (no matrix products of operators -- exact via H_HO algebra)
# ---------------------------------------------------------------------------
def acomm_p2_f_closed(F, R2F, E, b):
    """
    <i|{p^2, f}|j> = 2(E_i + E_j) F_ij  -  (2/b^4) (R2F)_ij,
    using p^2 = 2 H_HO - r^2/b^4 and  <i|H_HO = E_i <i|.
    """
    return 2.0 * (E[:, None] + E[None, :]) * F  -  (2.0 / b**4) * R2F


def p2_f_p2_closed(F, R2F, R4F, E, b):
    """
    <i|p^2 f p^2|j> = 4 E_i E_j F_ij - (2/b^4)(E_i+E_j) R2F_ij + (1/b^8) R4F_ij.
    """
    return (4.0 * np.outer(E, E) * F
            - (2.0 / b**4) * (E[:, None] + E[None, :]) * R2F
            + (1.0 / b**8) * R4F)


def acomm_p4_f_closed(F, R2F, R4F, Rfprime, E, b):
    """
    <i|{p^4, f}|j> = 4 (E_i^2 + E_j^2) F_ij
                   - (4/b^4)(E_i + E_j) R2F_ij
                   + (2/b^8) R4F_ij
                   - (4/b^4) Rfprime_ij,
    derived from p^4 = (2 H_HO - r^2/b^4)^2 and  [f, [H_HO, r^2]] = 2 r f'.
    """
    return (4.0 * (E[:, None]**2 + E[None, :]**2) * F
            - (4.0 / b**4) * (E[:, None] + E[None, :]) * R2F
            + (2.0 / b**8) * R4F
            - (4.0 / b**4) * Rfprime)


# ---------------------------------------------------------------------------
#  (p.r) f (r.p) reduced to a single radial integral via analytic u'
# ---------------------------------------------------------------------------
def pdotr_f_rdotp_radial(u, du, f_grid, r, weights):
    """
    <i|(p.r) f (r.p)|j> = int (r u_i' - u_i)(r u_j' - u_j) f(r) dr.

    Derivation:  r.p psi_n = -i r d/dr (u_n/r) Y = -i (u_n' - u_n/r) Y, so
        <(r.p) i| f |(r.p) j> = int (u_i'-u_i/r)(u_j'-u_j/r) f r^2 dr
                              = int (r u_i' - u_i)(r u_j' - u_j) f dr.

    Vectorised: one quadrature sum, no matrix products of operators.
    """
    phi = r * du - u
    weighted = phi * (f_grid * weights)[None, :]
    return phi @ weighted.T


# ---------------------------------------------------------------------------
#  Spin / J factors on |L S J> states
# ---------------------------------------------------------------------------
def spin_factors(L, S, J):
    """Returns dict with c-numbers
        kappa_LS  = J(J+1) - L(L+1) - S(S+1)         (= 2 l.S = l.sigma_T)
        kappa_SS  = 2 S(S+1) - 3                     (= sigma_1.sigma_2)
        kappa_LL12 = kappa_LS^2 + kappa_LS - 2 L(L+1) (= {(l.sig1),(l.sig2)})
        T_LL      = <L S J | S_12(hat r) | L S J>
                  = 0          (singlet, or triangle/parity violation)
                  = +2         L = J  >= 1
                  = -2(J-1)/(2J+1)  L = J - 1
                  = -2(J+2)/(2J+1)  L = J + 1
    """
    if S not in (0, 1):
        raise ValueError("S must be 0 or 1")
    if not (abs(L - S) <= J <= L + S):
        raise ValueError(f"J={J} not allowed for L={L}, S={S}")
    LL = L * (L + 1)
    SS = S * (S + 1)
    JJ = J * (J + 1)
    kappa_LS  = JJ - LL - SS
    kappa_SS  = 2 * SS - 3
    kappa_LL12 = kappa_LS * kappa_LS + kappa_LS - 2 * LL
    T_LL = 0.0
    if S == 1 and abs(L - 1) <= J <= L + 1:
        if L == J and L >= 1:
            T_LL = 2.0
        elif L == J - 1:
            T_LL = -2.0 * (J - 1) / (2 * J + 1)
        elif L == J + 1:
            T_LL = -2.0 * (J + 2) / (2 * J + 1)
    return {"kappa_LS": kappa_LS, "kappa_SS": kappa_SS,
            "kappa_LL12": kappa_LL12, "T_LL": T_LL}


# ---------------------------------------------------------------------------
#  Tensor radial m.e. of  2 T^{ab} p^a f p^b  in |n L S J>, L = L'
# ---------------------------------------------------------------------------
def K_K_tensor_radial(L, J, S, f_grid, fp_grid, u, du, r, weights):
    """
    <2 T^{ab} p^a f p^b>_ij  =  (2/3) T_LL^(J) * int [ f u_i' u_j' + A_L u_i u_j ] dr,
    A_L(r) = f * L(L+1) / r^2  -  f' / (2 r),
    using analytic u and u'.  Singlet (or vanishing T_LL) returns zeros.

    Derived via direct integration of the rank-2 spherical component
    O^{(2)}_0 of O^{ij} = p^i f p^j on the |L, M_L=L> state, then
    Wigner-Eckart (see modelo_eleccion derivation, 2026-05-26).

    Verified against the operator identity  T^{ab} p^a p^b = (1/3) S_12(p^) p^2
    for f = const across channels  L=1 (J=0,1,2)  and  L=2 (J=2).

    Previous (incorrect) version used  A_L = f*[L(L+1)-3]/r^2 - (2/3) f'/r
    and prefactor T_LL  (vs the correct prefactor (2/3) T_LL).
    """
    sf = spin_factors(L, S, J)
    T_LL = sf["T_LL"]
    N = u.shape[0]
    if T_LL == 0.0:
        return np.zeros((N, N))
    A_L = f_grid * L * (L + 1) / (r * r)  -  0.5 * fp_grid / r
    M_kin = du @ (du * (f_grid * weights)[None, :]).T
    M_AL  = u  @ (u  * (A_L    * weights)[None, :]).T
    return (2.0 / 3.0) * T_LL * (M_kin + M_AL)
