"""
Appendix E: how much does evaluating {p^2,f}_+ as a product of truncated
matrices cost, compared with the closed form?

P^2 is built in the same truncated basis from p^2 = 2 H_HO - r^2/b^4, so the
comparison isolates the truncation of the intermediate sum, not the quality of
the momentum operator.
"""
import numpy as np, ho_primitives as HO
import meson_potential_v2 as pv2

B, NGRID, L = 2.0, 8000, 0
P2 = dict(bar_V_v=1.92410, alpha=4.11051, d=3.28334, bar_V_s=0.86668, r_s=1.43734)
ET, MQ = 3.08513, 1.275
f2 = pv2.make_potential_funcs(**P2)

print(f"{'n_states':>9s}{'max|closed|':>14s}{'max|diff|':>13s}"
      f"{'rel.':>10s}{'diag rel.':>11s}")
for N in (10, 15, 20, 30, 40, 50):
    r, dr, u, du = HO.build_grid(L, N, B, N_grid=NGRID)
    w = HO.simpson_weights(len(r), dr)
    E = np.array([(2*n + L + 1.5) / B**2 for n in range(N)])
    VL = np.asarray(f2["V_L"](r), float)
    AL2 = (1.0 / (MQ + 0.5*ET - 0.5*VL))**2          # the block A_L^2
    F = u @ (u * (AL2 * w)[None, :]).T
    R2F = u @ (u * (AL2 * r*r * w)[None, :]).T
    closed = HO.acomm_p2_f_closed(F, R2F, E, B)
    # same operator as a product of truncated matrices
    one = u @ (u * w[None, :]).T
    r2m = u @ (u * (r*r * w)[None, :]).T
    P2m = 2*np.diag(E) @ one - r2m / B**4
    prod = P2m @ F + F @ P2m
    d = np.abs(closed - prod)
    print(f"{N:9d}{np.abs(closed).max():14.5f}{d.max():13.2e}"
          f"{d.max()/np.abs(closed).max():10.2e}"
          f"{np.abs(np.diag(closed-prod)).max()/np.abs(np.diag(closed)).max():11.2e}")

print()
print("lowest eigenvalues of the kinetic block 2m {p^2, A_L^2}_+  [MeV]")
print(f"{'n_states':>9s}{'closed':>12s}{'product':>12s}{'diff':>10s}")
for N in (10, 20, 30, 50):
    r, dr, u, du = HO.build_grid(L, N, B, N_grid=NGRID)
    w = HO.simpson_weights(len(r), dr)
    E = np.array([(2*n + L + 1.5) / B**2 for n in range(N)])
    VL = np.asarray(f2["V_L"](r), float)
    AL2 = (1.0 / (MQ + 0.5*ET - 0.5*VL))**2
    F = u @ (u * (AL2 * w)[None, :]).T
    R2F = u @ (u * (AL2 * r*r * w)[None, :]).T
    one = u @ (u * w[None, :]).T
    r2m = u @ (u * (r*r * w)[None, :]).T
    P2m = 2*np.diag(E) @ one - r2m / B**4
    a = 2*MQ*1000*np.linalg.eigvalsh(HO.acomm_p2_f_closed(F, R2F, E, B))[0]
    b = 2*MQ*1000*np.linalg.eigvalsh(P2m @ F + F @ P2m)[0]
    print(f"{N:9d}{a:12.2f}{b:12.2f}{b-a:10.2f}")
