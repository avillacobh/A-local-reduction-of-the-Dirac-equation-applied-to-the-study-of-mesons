"""
Numerical-quality diagnostic for the HO basis at high n.

Concern
-------
At n_r = 29 the radial HO wavefunction u_{29, L}(r) carries 29 oscillations.
Two things can fail:

  (1) Evaluation of L_n^{L+1/2}(r^2/b^2).  Forward recurrence accumulates
      round-off; scipy.special.eval_genlaguerre is stable but worth checking
      independently.

  (2) Simpson integration on the grid of products u_i u_j f(r).  The
      product oscillates at twice the frequency of u_i, and grid density
      that "looks fine" for u_i alone may not integrate u_i^2 cleanly.

Single decisive metric
----------------------
Gram matrix  G_{ij} = int_0^infty u_i u_j dr  must equal delta_{ij}.

We report  max_{i,j} |G_{ij} - delta_{ij}|  and also the worst row, so we
can see whether the failures are concentrated at the highest n (Laguerre
issue) or scattered across all n (grid-resolution issue).

Also: integrate <u_n | 1/r | u_m> against a scipy.quad reference (no
truncation, true (0, infinity)).  That isolates integration accuracy.

Run from the project root.
"""
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))   # the project root
sys.path.insert(0, HERE)
os.chdir(HERE)

import numpy as np
from ho_primitives import build_grid, simpson_weights, ho_radial_u

try:
    from scipy.integrate import quad as _quad
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def gram_test(L, N, b, N_grid):
    r, dr, u, du = build_grid(L, N, b, N_grid=N_grid)
    w = simpson_weights(N_grid, dr)
    # G_{ij} = int u_i u_j dr  -- via Simpson
    G = u @ (u * w[None, :]).T
    I = np.eye(N)
    D = G - I
    abs_max = np.max(np.abs(D))
    # per-row worst element (to see if it concentrates at high n)
    row_max = np.max(np.abs(D), axis=1)
    return G, abs_max, row_max, r, u


def integration_test_one_over_r(L, N, b, N_grid):
    """Reference: <u_n | 1/r | u_n> via scipy.quad on (0, inf), no truncation.
    Compare with Simpson on the build_grid mesh.
    """
    if not HAS_SCIPY:
        return None
    r, dr, u, du = build_grid(L, N, b, N_grid=N_grid)
    w = simpson_weights(N_grid, dr)
    simp = (u * u * (1.0 / r)[None, :]) @ w  # diag of <u_n | 1/r | u_n>

    # Reference via quad for the worst (highest n) state only -- avoids paying
    # the price for 30 quad calls each taking ~1 s.
    n_check = N - 1
    def integrand(rr):
        if rr <= 0:
            return 0.0
        un = ho_radial_u(n_check, L, np.asarray([rr]), b)[0]
        return float(un * un / rr)
    ref, abserr = _quad(integrand, 0.0, np.inf, epsabs=1e-14, epsrel=1e-12, limit=400)
    return simp[n_check], ref, abserr


# ---- Test at the realistic fit params --------------------------------
# Variational b* from the convergence sweep are smallest for L=0
# channels (~1.2 at n=30, ~1.7 at n=20).  We test L=0, 1, 2 at b's that
# span the realistic range.
print("=" * 78)
print(" HO basis numerical-quality diagnostic")
print("=" * 78)

CASES = [
    # (label, L, N, b, N_grid)
    ("L=0, b=1.21, N=30, Ng=8000",  0, 30, 1.21, 8000),
    ("L=0, b=1.21, N=40, Ng=8000",  0, 40, 1.21, 8000),
    ("L=0, b=1.21, N=30, Ng=16000", 0, 30, 1.21, 16000),
    ("L=0, b=1.70, N=30, Ng=8000",  0, 30, 1.70, 8000),
    ("L=1, b=1.58, N=30, Ng=8000",  1, 30, 1.58, 8000),
    ("L=2, b=2.14, N=30, Ng=8000",  2, 30, 2.14, 8000),
    # Stress test: push to n=50 to see when the basis breaks.
    ("L=0, b=1.21, N=50, Ng=8000",  0, 50, 1.21, 8000),
    ("L=0, b=1.21, N=50, Ng=16000", 0, 50, 1.21, 16000),
]

print()
print(f" {'case':<34s} {'max|G-I|':>12s} {'row of max':>11s} {'r_max [GeV-1]':>14s}")
print(" " + "-" * 74)
gram_results = []
for (lab, L, N, b, Ng) in CASES:
    G, mx, row, r, u = gram_test(L, N, b, Ng)
    irow = int(np.argmax(row))
    gram_results.append((lab, L, N, b, Ng, mx, irow, r.max(), G, u, r))
    print(f" {lab:<34s} {mx:12.3e} {irow:>11d} {r.max():>14.2f}")

print()
print(" Interpretation of |G - I|:")
print("    < 1e-8 :  basis is numerically exact for the fit")
print("    1e-8 .. 1e-5 : small grid error; matrix elements OK to ~ keV")
print("    > 1e-5 : grid is undersampling; HIGH-n functions are no longer")
print("             orthonormal -> matrix elements at those n unreliable")

# ---- Look at where the failure concentrates --------------------------
print()
print("=" * 78)
print(" Diagonal departure  G_{nn} - 1  vs  n   (should all be tiny)")
print("=" * 78)
for entry in gram_results:
    lab, L, N, b, Ng, mx, irow, rmax, G, u, r = entry
    diag_err = np.abs(np.diag(G) - 1.0)
    worst_n = np.argsort(diag_err)[-5:][::-1]
    worst_str = ", ".join(f"n={int(n)}:{diag_err[n]:.2e}" for n in worst_n)
    print(f" {lab}")
    print(f"   worst 5 diag errors: {worst_str}")
    print(f"   median diag error  : {np.median(diag_err):.2e}")
    print()

# ---- Stress test: integral of 1/r at highest n vs quad reference ------
if HAS_SCIPY:
    print("=" * 78)
    print(" <u_{N-1} | 1/r | u_{N-1}> : Simpson on build_grid vs scipy.quad")
    print("=" * 78)
    print()
    print(f" {'case':<34s} {'Simpson':>14s} {'quad (ref)':>14s} {'|delta|':>12s}")
    print(" " + "-" * 74)
    for (lab, L, N, b, Ng) in CASES[:6]:  # skip the n=50 stress tests
        result = integration_test_one_over_r(L, N, b, Ng)
        if result is None:
            continue
        simp_val, ref_val, abserr = result
        delta = abs(simp_val - ref_val)
        print(f" {lab:<34s} {simp_val:14.10f} {ref_val:14.10f} {delta:12.3e}")
    print()
    print(" If |delta| < 1e-6, Simpson is faithful at the highest n.")

print()
print("=" * 78)
print(" Verdict")
print("=" * 78)
print("  Use the first 4 numbers in the Gram table:")
print("   - L=0, N=30, Ng=8000 : production setting")
print("   - L=0, N=40, Ng=8000 : effect of basis growth")
print("   - L=0, N=30, Ng=16000: effect of grid refinement (double)")
print("   - L=0, N=30, b=1.70  : effect of larger b (less compressed)")
print()
print("  If Ng=8000 and Ng=16000 give nearly the same |G-I|, the grid is")
print("  resolving the oscillations cleanly and N_grid=8000 is fine for")
print("  the production fit at N=30.  If Ng=16000 is dramatically better,")
print("  bump --n-grid in run_all_potentials_de.sh.")
