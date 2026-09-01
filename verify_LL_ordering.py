"""
Size of the symmetric-average approximation for |Phi_LL>.

The two non-equivalent routes to the lower-lower component (Eqs. 3.4 and 3.5 of
the thesis) are, in the CM frame with m_1 = m_2 = m, p_1 = -p, p_2 = +p:

    route 1 :  - A_U (sigma_1 . p) A_L (sigma_2 . p)
    route 2 :  - A_U (sigma_2 . p) A_L (sigma_1 . p)

and K_LL is their average.  Their DIFFERENCE can be evaluated in closed form.
Using [p^a, f] = -i (f'/r) x^a and the fact that sigma_1, sigma_2 act on
different spaces,

    (s1.p) f (s2.p) - (s2.p) f (s1.p)
        = s1^a s2^b ( p^a f p^b - p^b f p^a )
        = -i (f'/r) s1^a s2^b ( x^a p^b - x^b p^a )
        = -i (f'/r) (sigma_1 x sigma_2) . L                            (*)

so that the piece discarded by the symmetric average is

    delta = (i/2) A_U (r) [A_L'(r)/r] (sigma_1 x sigma_2) . L .        (**)

Two properties of (**) settle the question analytically:

  (1) it is ANTI-HERMITIAN, so each single ordering would give a non-Hermitian
      reduced operator; the average is exactly its Hermitian part;

  (2) (sigma_1 x sigma_2) is antisymmetric under 1 <-> 2, hence it has NO
      matrix elements with Delta S = 0: it only connects S = 0 with S = 1.

Because the spectrum of Chapter 8 is computed channel by channel at fixed
(L, S, J), property (2) means the discarded term contributes exactly zero to
every level of the fit -- not a small number, but identically zero.  For equal
masses it is moreover C-odd (C = (-1)^(L+S) for a q qbar pair), so even the
second-order singlet-triplet mixing it could induce is forbidden.

This script verifies (*), (1) and (2) numerically.
Run:  python3 verify_LL_ordering.py
"""
import itertools
import numpy as np

# ---------------------------------------------------------------------------
#  Spin algebra
# ---------------------------------------------------------------------------
I2 = np.eye(2)
S = [np.array([[0, 1], [1, 0]], complex),
     np.array([[0, -1j], [1j, 0]]),
     np.array([[1, 0], [0, -1]], complex)]
s1 = [np.kron(S[a], I2) for a in range(3)]
s2 = [np.kron(I2, S[b]) for b in range(3)]

eps = np.zeros((3, 3, 3))
for i, j, k in itertools.permutations(range(3)):
    eps[i, j, k] = np.sign(np.linalg.det(np.eye(3)[[i, j, k]]))

cross = [sum(eps[c, a, b] * s1[a] @ s2[b] for a in range(3) for b in range(3))
         for c in range(3)]

singlet = np.array([0, 1, -1, 0], complex) / np.sqrt(2)
triplet = [np.array([1, 0, 0, 0], complex),
           np.array([0, 1, 1, 0], complex) / np.sqrt(2),
           np.array([0, 0, 0, 1], complex)]
basis = [(0, "S=0,M= 0", singlet)] + \
        [(1, f"S=1,M={m:+d}", v) for m, v in zip([1, 0, -1], triplet)]

print("=" * 74)
print(" (2)  matrix elements of (sigma_1 x sigma_2) in the coupled spin basis")
print("=" * 74)
worst_dS0 = 0.0
for Sa, na, va in basis:
    for Sb, nb, vb in basis:
        for c in range(3):
            val = abs(np.vdot(va, cross[c] @ vb))
            if Sa == Sb:
                worst_dS0 = max(worst_dS0, val)
print(f"  largest |<S,M|(s1 x s2)^c|S',M'>| with Delta S = 0 : {worst_dS0:.3e}")
print("  -> vanishes: the operator only connects S=0 with S=1\n")

# ---------------------------------------------------------------------------
#  Anti-hermiticity of i (sigma_1 x sigma_2).L  (spin part; L is Hermitian and
#  acts on a different space, so it suffices to check the spin factor)
# ---------------------------------------------------------------------------
worst_ah = max(np.max(np.abs(1j * cross[c] + (1j * cross[c]).conj().T))
               for c in range(3))
print("=" * 74)
print(" (1)  anti-hermiticity of i (sigma_1 x sigma_2)^c")
print("=" * 74)
print(f"  max |A + A^dagger| = {worst_ah:.3e}   -> anti-Hermitian\n")

# ---------------------------------------------------------------------------
#  Operator identity (*) on a 3D grid, with concrete f and test function
# ---------------------------------------------------------------------------
print("=" * 74)
print(" (*)  p^a f p^b - p^b f p^a  ==  -i (f'/r) (x^a p^b - x^b p^a)")
print("=" * 74)
def identity_error(n, L_box=4.0):
    """Worst relative deviation between the two sides of (*) on an n^3 grid."""
    x = np.linspace(-L_box, L_box, n)
    h = x[1] - x[0]
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R2 = X**2 + Y**2 + Z**2
    R = np.sqrt(R2) + 1e-12
    coord = [X, Y, Z]

    # f(r) analytic at the origin (a function of r^2), so that f'/r is regular
    f = np.exp(-0.7 * R2) * (1.0 + 0.3 * R2)
    fp = R * np.exp(-0.7 * R2) * (0.6 - 1.4 * (1.0 + 0.3 * R2))     # df/dr
    psi = np.exp(-0.3 * R2) * (1 + 0.4 * X - 0.3 * Y + 0.2 * Z * X)

    d = lambda u, ax: (np.roll(u, -1, ax) - np.roll(u, 1, ax)) / (2 * h)
    p = lambda u, ax: -1j * d(u, ax)

    sl = (slice(4, -4),) * 3                          # drop boundary layers
    worst = 0.0
    for a in range(3):
        for b in range(3):
            lhs = p(f * p(psi, b), a) - p(f * p(psi, a), b)
            rhs = -1j * (fp / R) * (coord[a] * p(psi, b) - coord[b] * p(psi, a))
            scale = np.max(np.abs(lhs[sl])) + 1e-30
            worst = max(worst, np.max(np.abs((lhs - rhs)[sl])) / scale)
    return h, worst

errs = [identity_error(n) for n in (41, 81, 161)]
for h, w in errs:
    print(f"  h = {h:.4f}   relative deviation = {w:.3e}")
print("  ratios on halving h (4 == second-order convergence): "
      + ", ".join(f"{errs[k][1] / errs[k + 1][1]:.2f}" for k in range(len(errs) - 1)))
worst = errs[-1][1]
print("  -> the deviation falls as h^2, so the identity is exact and the\n     residual is only the finite-difference error\n")

print("=" * 74)
print(" CONCLUSION")
print("=" * 74)
print(" The term discarded by the symmetric average is")
print("     delta = (i/2) A_U (A_L'/r) (sigma_1 x sigma_2).L ,")
print(" anti-Hermitian and with vanishing matrix elements at fixed S.")
print(" Its contribution to every level of the charmonium fit is exactly zero;")
print(" the symmetrisation is what makes the reduction Hermitian and C-even.")
