"""
Self-consistent solver for the Salpeter eigenvalue problem.

Because A_X(r) = 1 / (m + E_T/2 - V_X(r)/2) carries E_T, H depends on E_T.
The physical condition for the n-th eigenstate is

    f_n(E_T)  =  eigval_n[ H(E_T) ]  -  E_T  =  0,

with eigenvalues ordered ascending (n = 0 ground state).  We solve by Brent
(`scipy.optimize.brentq` if available, bisection fallback otherwise).

This module exposes BOTH a class API (`MesonHamiltonianSolver`, recommended:
caches the static parts of H so a sweep over E_T only rebuilds the small
E_T-dependent multiplicative-function matrices) and module-level helpers
that build a fresh MesonHamiltonian every call (slower; kept for one-off use).

Hermiticity is exploited at every level:
  - `numpy.linalg.eigvalsh` / `numpy.linalg.eigh` use LAPACK routines
    specialised for real symmetric / Hermitian matrices.
  - When scipy is available, `scipy.linalg.eigh` with `subset_by_index`
    can compute only the needed lowest few eigenvalues.
"""

import numpy as np

from H_full_matrix import MesonHamiltonian, H_matrix

try:
    from scipy.optimize import brentq as _brentq
    HAS_SCIPY = True
except ImportError:                                                # pragma: no cover
    HAS_SCIPY = False

    def _brentq(f, a, b, xtol=1e-10, rtol=1e-12, maxiter=100, args=()):
        """Bisection fallback when scipy is not installed."""
        fa, fb = f(a, *args), f(b, *args)
        if fa * fb > 0:
            raise ValueError(f"No sign change in bracket [{a}, {b}]: "
                             f"f(a)={fa}, f(b)={fb}.")
        for _ in range(maxiter):
            c = 0.5 * (a + b)
            fc = f(c, *args)
            if fc == 0.0 or 0.5 * (b - a) < xtol:
                return c
            if fa * fc < 0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return 0.5 * (a + b)


# ---------------------------------------------------------------------------
#  Optional: scipy.linalg.eigh with subset_by_index for partial eigvals
# ---------------------------------------------------------------------------
try:
    from scipy.linalg import eigh as _sp_eigh                     # noqa: F401

    def _eigvals_lowest(H, n_keep):
        """Lowest `n_keep` eigenvalues of symmetric H (ascending)."""
        return _sp_eigh(H, eigvals_only=True,
                        subset_by_index=[0, n_keep - 1])
except ImportError:                                                # pragma: no cover
    def _eigvals_lowest(H, n_keep):
        return np.linalg.eigvalsh(H)[:n_keep]


# ===========================================================================
#  Class API: MesonHamiltonianSolver  (cached, recommended for sweeps)
# ===========================================================================
class MesonHamiltonianSolver(MesonHamiltonian):
    """
    MesonHamiltonian + Hermitian-aware diagonalisation + Brent root-finder.

    Inherits all the caching of MesonHamiltonian (grid, u, du, V's, V_U matrix,
    E_arr, etc.); adds methods to find the self-consistent E_T(s).
    """

    # ----- partial diag (uses scipy.linalg.eigh subset_by_index if available)
    def lowest_eigvals(self, E_T, n_keep=None):
        """Lowest `n_keep` eigenvalues (default: all n_states)."""
        if n_keep is None:
            n_keep = self.n_states
        return _eigvals_lowest(self.matrix(E_T), n_keep)

    # ----- self-consistent root for one level ---------------------------
    def self_consistent_E_T(self, n_level, E_T_lo, E_T_hi,
                            xtol=1e-8, maxiter=200, verbose=False):
        if n_level < 0 or n_level >= self.n_states:
            raise ValueError(f"n_level={n_level} out of range "
                             f"for n_states={self.n_states}")
        # only the lowest (n_level + 1) eigenvalues are needed
        n_keep = n_level + 1

        def residual(E_T):
            eigs = self.lowest_eigvals(E_T, n_keep=n_keep)
            r = eigs[n_level] - E_T
            if verbose:
                print(f"  E_T = {E_T:+.8f}  ->  lambda_{n_level} = "
                      f"{eigs[n_level]:+.8f}   r = {r:+.4e}")
            return r

        return _brentq(residual, E_T_lo, E_T_hi, xtol=xtol, maxiter=maxiter)

    # ----- find multiple roots by sweep + bracketing --------------------
    def find_levels(self, levels, E_T_grid,
                    xtol=1e-8, maxiter=200, verbose=False):
        """Sweep E_T over `E_T_grid`, locate sign changes of
        eigval_n(H(E_T)) - E_T  for each n in `levels`, refine with Brent.
        Returns dict { n : list of E_T_star }."""
        E_grid = np.asarray(E_T_grid, dtype=float)
        n_grid = len(E_grid)
        n_keep = max(levels) + 1

        # one pass over the grid -- compute only the necessary eigvals
        eigs_arr = np.empty((n_grid, n_keep))
        for k, E_T in enumerate(E_grid):
            eigs_arr[k] = self.lowest_eigvals(E_T, n_keep=n_keep)

        out = {}
        for n in levels:
            residuals = eigs_arr[:, n] - E_grid
            sign_change = np.where(np.sign(residuals[:-1]) *
                                   np.sign(residuals[1:]) < 0)[0]
            roots = [self.self_consistent_E_T(n, E_grid[i], E_grid[i + 1],
                                               xtol=xtol, maxiter=maxiter,
                                               verbose=verbose)
                     for i in sign_change]
            out[n] = roots
        return out


# ===========================================================================
#  Module-level shortcuts (one-off use; rebuild MesonHamiltonian each call)
# ===========================================================================
def H_eigenvalues(L, S, J, V_L_func, V_U_func, m, E_T, b,
                  n_states=10, **H_kwargs):
    """Sorted-ascending eigenvalues of H(E_T).  One-off, no caching."""
    H = H_matrix(L=L, S=S, J=J, V_L_func=V_L_func, V_U_func=V_U_func,
                 m=m, E_T=E_T, b=b, n_states=n_states, **H_kwargs)
    return np.linalg.eigvalsh(H)


def self_consistent_E_T(L, S, J, V_L_func, V_U_func, m, b,
                        n_level, E_T_lo, E_T_hi,
                        n_states=10, xtol=1e-8, maxiter=200,
                        verbose=False, **H_kwargs):
    """One-shot self-consistent search (no caching). Use the class for sweeps."""
    solver = MesonHamiltonianSolver(L, S, J, V_L_func, V_U_func, m, b,
                                     n_states=n_states, **H_kwargs)
    return solver.self_consistent_E_T(n_level, E_T_lo, E_T_hi,
                                       xtol=xtol, maxiter=maxiter,
                                       verbose=verbose)


def find_levels(L, S, J, V_L_func, V_U_func, m, b,
                levels, E_T_grid,
                n_states=10, xtol=1e-8, maxiter=200, verbose=False,
                **H_kwargs):
    """One-shot multi-level search (no caching). Use the class for sweeps."""
    solver = MesonHamiltonianSolver(L, S, J, V_L_func, V_U_func, m, b,
                                     n_states=n_states, **H_kwargs)
    return solver.find_levels(levels, E_T_grid,
                               xtol=xtol, maxiter=maxiter, verbose=verbose)


# ===========================================================================
#  OPTIONAL: per-channel variational HO scale b* (opt-in via fit_meson flag)
# ===========================================================================
#
# Rationale.  By MacDonald's theorem, the n-th eigenvalue of any truncated
# orthonormal basis is an upper bound on the n-th exact eigenvalue.  Choosing
# the HO scale b that minimises lambda_0[H(E_T*; b)] therefore gives the best
# variational ground-state estimate in that channel.
#
# Implementation note.  We use a simple GRID SCAN (no scipy minimiser) so the
# search is deterministic and robust: pick the b in `b_values` that gives the
# smallest self-consistent E_T^(0).  If every b in the grid fails the inner
# self-consistency (no sign change in the brent bracket), the channel is
# reported as missing and the caller can fall back to fixed-b or skip.
# ===========================================================================
def find_variational_b(L, S, J, V_L_func, V_U_func, m, n_states,
                       b_values=None, E_T_lo=2.0, E_T_hi=10.0,
                       H_kwargs=None):
    """Scan b over `b_values`, return (b_star, E_T_star_for_ground_state).

    Returns (None, None) if no b in the grid gives a converged self-
    consistency for the ground state of this channel.
    """
    if b_values is None:
        b_values = np.linspace(0.5, 4.0, 10)
    if H_kwargs is None:
        H_kwargs = {}
    best_b, best_E = None, np.inf
    for b in b_values:
        try:
            solver = MesonHamiltonianSolver(
                L=L, S=S, J=J,
                V_L_func=V_L_func, V_U_func=V_U_func,
                m=m, b=float(b), n_states=n_states, **H_kwargs)
            E_T = solver.self_consistent_E_T(0, E_T_lo, E_T_hi)
            if E_T < best_E:
                best_b, best_E = float(b), E_T
        except (ValueError, RuntimeError):
            continue
    if best_b is None:
        return None, None
    return best_b, best_E


# ===========================================================================
#  Continuous variational b*: scipy.optimize.minimize_scalar (bounded Brent)
#  Same result as find_variational_b in the limit of dense grid, but uses
#  ~10-15 function evaluations adaptively to land on the true minimum
#  (sub-percent precision on b*) rather than the coarsest grid point.
# ===========================================================================
def find_variational_b_continuous(L, S, J, V_L_func, V_U_func, m, n_states,
                                   b_lo=0.5, b_hi=4.0,
                                   E_T_lo=2.0, E_T_hi=10.0,
                                   xatol=0.02, maxiter=30,
                                   H_kwargs=None):
    """Continuous version of find_variational_b using scipy.optimize.
    minimize_scalar (bounded Brent).  Returns (b_star, E_T_star) for the
    ground state of channel (L,S,J).  Returns (None, None) if scipy is
    unavailable OR if the minimizer fails to land on a finite minimum.

    Parameters
    ----------
    b_lo, b_hi : float
        Search bounds on b (GeV^-1).  Default (0.5, 4.0).
    xatol : float
        Tolerance on b*.  Default 0.02 GeV^-1 ~ 0.4% of typical b ~ 2.
    maxiter : int
        Hard cap on minimizer iterations.  Default 30.
    """
    try:
        from scipy.optimize import minimize_scalar
    except ImportError:
        return None, None
    if H_kwargs is None:
        H_kwargs = {}
    PENALTY = 1e6

    def fobj(b):
        try:
            solver = MesonHamiltonianSolver(
                L=L, S=S, J=J,
                V_L_func=V_L_func, V_U_func=V_U_func,
                m=m, b=float(b), n_states=n_states, **H_kwargs)
            return solver.self_consistent_E_T(0, E_T_lo, E_T_hi)
        except (ValueError, RuntimeError):
            return PENALTY

    res = minimize_scalar(fobj, bounds=(b_lo, b_hi), method="bounded",
                           options=dict(xatol=xatol, maxiter=maxiter))
    if not res.success or res.fun >= PENALTY * 0.5:
        return None, None
    return float(res.x), float(res.fun)


def build_variational_solvers(funcs, m, channels, n_states=10,
                              b_values=None,
                              E_T_lo=2.0, E_T_hi=10.0,
                              method="continuous",
                              b_lo=0.5, b_hi=4.0,
                              H_kwargs=None, verbose=False):
    """For each (L,S,J) channel, find b*(L,S,J) and build a cached
    MesonHamiltonianSolver at that b*.

    Parameters
    ----------
    method : {"continuous", "grid"}
        Which 1D minimizer to use for the variational b* per channel.
        - "continuous" (default): scipy.optimize.minimize_scalar with
          bounded Brent.  ~10-15 evals per channel, sub-grid precision.
        - "grid": coarse grid scan over `b_values` (default
          np.linspace(0.5, 4.0, 10)).  Robust, deterministic.
    b_lo, b_hi : float
        Bounds on b for the "continuous" method.  Ignored if method="grid".
    b_values : array-like or None
        Grid of b values for the "grid" method.  Ignored if method="continuous".

    Returns dict { (L,S,J) : {"b_star": float,
                              "E_T_0_at_b_star": float,
                              "solver": MesonHamiltonianSolver} }.
    Channels for which the variational search fails are simply omitted from
    the dict (the caller will see them as missing and report NaN or fall
    back).
    """
    if method not in ("continuous", "grid"):
        raise ValueError(f"method must be 'continuous' or 'grid', got {method!r}")
    if H_kwargs is None:
        H_kwargs = {}
    H_pot_kwargs = dict(
        V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
        V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
        **H_kwargs,
    )
    out = {}
    for (L, S, J) in channels:
        if method == "continuous":
            b_star, E_T_star = find_variational_b_continuous(
                L, S, J, funcs["V_L"], funcs["V_U"], m, n_states,
                b_lo=b_lo, b_hi=b_hi,
                E_T_lo=E_T_lo, E_T_hi=E_T_hi,
                H_kwargs=H_pot_kwargs)
        else:  # grid
            b_star, E_T_star = find_variational_b(
                L, S, J, funcs["V_L"], funcs["V_U"], m, n_states,
                b_values=b_values, E_T_lo=E_T_lo, E_T_hi=E_T_hi,
                H_kwargs=H_pot_kwargs)
        if b_star is None:
            if verbose:
                print(f"  ({L},{S},{J}): variational scan FAILED on all b "
                      f"in the grid.")
            continue
        solver = MesonHamiltonianSolver(
            L=L, S=S, J=J,
            V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
            m=m, b=b_star, n_states=n_states, **H_pot_kwargs)
        out[(L, S, J)] = {"b_star": b_star,
                          "E_T_0_at_b_star": E_T_star,
                          "solver": solver}
        if verbose:
            print(f"  ({L},{S},{J}): b* = {b_star:.4f}  "
                  f"E_T^(0) = {E_T_star:.4f}")
    return out


# ---------------------------------------------------------------------------
#  Example
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    from meson_potential import make_potential_funcs

    funcs = make_potential_funcs(bar_V_v=0.5, alpha=0.4, d_v=0.7,
                                  bar_V_s=-0.3, r_s=1.2, d_s=0.5)
    common = dict(L=0, S=1, J=1,
                  V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
                  V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
                  V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
                  m=1.27, b=1.0, n_states=10)

    # Recommended workflow: instantiate once, reuse for many E_T
    t0 = time.time()
    solver = MesonHamiltonianSolver(**common)
    t1 = time.time()
    print(f"Solver setup (caches grid, u, du, V's, F_VU): {1e3*(t1-t0):.1f} ms")

    t0 = time.time()
    E0 = solver.self_consistent_E_T(n_level=0, E_T_lo=2.0, E_T_hi=8.0)
    t1 = time.time()
    print(f"Ground-state self-consistency: E_T = {E0:.8f}   "
          f"({1e3*(t1-t0):.1f} ms)")

    t0 = time.time()
    spec = solver.find_levels(levels=[0, 1, 2, 3, 4],
                               E_T_grid=np.linspace(2, 30, 60))
    t1 = time.time()
    print(f"\nMulti-level sweep (60-point E_T grid x 5 levels): "
          f"{1e3*(t1-t0):.1f} ms")
    for n, roots in spec.items():
        print(f"  level {n}: {[f'{r:.4f}' for r in roots]}")
