"""
Verify spin_factors(L, S, J) against hand-derived analytical values
for all charmonium channels, with special attention to the h_c (1^1P_1).

Hand derivations (notation: <O> = matrix element of O on |LSJ>):

  kappa_LS  = <2 L.S> = <L.sigma_total> = J(J+1) - L(L+1) - S(S+1)

  kappa_SS  = <sigma_1.sigma_2> = (1/2)<(sigma_total)^2 - sigma_1^2 - sigma_2^2>
                                = (1/2)[4 S(S+1) - 6] = 2 S(S+1) - 3
            (since sigma_i^2 = 3 on a single spin-1/2)

  kappa_LL12 = <{l.sigma_1, l.sigma_2}_+>
             Using (l.sigma_total)^2 = (l.sigma_1)^2 + (l.sigma_2)^2 + {l.sig1,l.sig2},
             and (l.sigma_i)^2 = l^2 - l.sigma_i (Pauli identity):
               = <(2 l.S)^2 - 2 l^2 + 2 l.S>
               = kappa_LS^2 - 2 L(L+1) + kappa_LS

  T_LL     = <S_12(rhat)>  with  S_12 = 3(sigma_1.rhat)(sigma_2.rhat) - sigma_1.sigma_2
           S_12 is a rank-2 spherical tensor in spin space => vanishes on S=0
           For S=1, standard results (Lucha & Schoeberl):
             L = J:       +2          (e.g. 3P1, 3D2)
             L = J - 1:   -2(J-1)/(2J+1)   (e.g. 3S1 with J=1 gives 0; 3P2 with J=2 gives -2/5)
             L = J + 1:   -2(J+2)/(2J+1)   (e.g. 3P0 with J=0 gives -4; 3D1 with J=1 gives -2)
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))   # the project root
sys.path.insert(0, HERE)
os.chdir(HERE)

from ho_primitives import spin_factors


def expected(L, S, J):
    """Hand-derived expected values."""
    kLS  = J * (J + 1) - L * (L + 1) - S * (S + 1)
    kSS  = 2 * S * (S + 1) - 3
    kLL12 = kLS * kLS + kLS - 2 * L * (L + 1)
    # T_LL: zero for S=0 (singlet), or J outside triangle |L-1|..L+1
    T_LL = 0.0
    if S == 1 and abs(L - 1) <= J <= L + 1:
        if L == J and L >= 1:
            T_LL = 2.0
        elif L == J - 1:
            T_LL = -2.0 * (J - 1) / (2 * J + 1)
        elif L == J + 1:
            T_LL = -2.0 * (J + 2) / (2 * J + 1)
    return dict(kappa_LS=kLS, kappa_SS=kSS, kappa_LL12=kLL12, T_LL=T_LL)


SPEC = ["S", "P", "D", "F"]
def label(n, L, S, J):
    return f"{n} ^{2*S+1}{SPEC[L]}_{J}"


# Charmonium channels we actually fit (csv1 + csv2)
CHANNELS = [
    # (n, L, S, J, common label)
    (1, 0, 0, 0, "1 ^1S_0 (eta_c)"),
    (1, 0, 1, 1, "1 ^3S_1 (J/psi)"),
    (1, 1, 1, 0, "1 ^3P_0 (chi_c0)"),
    (1, 1, 1, 1, "1 ^3P_1 (chi_c1)"),
    (1, 1, 0, 1, "1 ^1P_1 (h_c)  <-- THE INTERESTING ONE"),
    (1, 1, 1, 2, "1 ^3P_2 (chi_c2)"),
    (1, 2, 1, 1, "1 ^3D_1 (psi(3770))"),
    (1, 2, 1, 2, "1 ^3D_2 (psi_2(3823))"),
]

print("=" * 100)
print(" Verification of spin_factors(L, S, J) against hand-derived analytical values")
print("=" * 100)
print()
print(f"  {'channel':<32s}  {'(L,S,J)':<10s}  "
      f"{'kappa_LS':>10s} {'kappa_SS':>10s} {'kappa_LL12':>12s} {'T_LL':>10s}")
print(" " + "-" * 96)
all_ok = True
for (n, L, S, J, lab) in CHANNELS:
    code = spin_factors(L, S, J)
    expc = expected(L, S, J)
    same = all(abs(code[k] - expc[k]) < 1e-12 for k in code)
    flag = "" if same else " <-- MISMATCH"
    print(f"  CODE     {lab:<25s}  ({L},{S},{J})       "
          f"{code['kappa_LS']:>10.4f} {code['kappa_SS']:>10.4f} "
          f"{code['kappa_LL12']:>12.4f} {code['T_LL']:>10.4f}{flag}")
    print(f"  EXPECTED {' '*25}  ({L},{S},{J})       "
          f"{expc['kappa_LS']:>10.4f} {expc['kappa_SS']:>10.4f} "
          f"{expc['kappa_LL12']:>12.4f} {expc['T_LL']:>10.4f}")
    if not same:
        all_ok = False
    print()

print(" " + "-" * 96)
print(f"  Verdict: {'ALL spin factors match analytical' if all_ok else 'DISCREPANCIES FOUND'}")
print()

# ---- Special focus: h_c entry analysis ----------------------------------
print("=" * 100)
print(" Special: how the h_c (L=1, S=0, J=1) enters W_0 (V_v + beta*beta V_s)")
print("=" * 100)
print()
print("  In H_full_matrix.py, the relevant H assembly lines are:")
print()
print("    line45 = 0.25 * kappa_LS * bracket_LS         # spin-orbit  (l.S)")
print("    line6  = 0.5  * kappa_LL12 * F('Y8')          # {l.sig1, l.sig2}")
print("    line7  = 0.5  * kappa_SS * (...)              # sigma_1.sigma_2")
print("    line8_scalar = -(kappa_SS/6) * (...)          # part of (sig.p) f (sig.p)")
print("    line8_tensor = -0.5 * K_K_tensor_radial(...)  # tensor piece, vanishes for S=0")
print()

# Effective coefficients per channel
print(f"  {'channel':<30s}  {'line45':>9s} {'line6':>9s} {'line7':>9s} "
      f"{'line8_scalar':>13s} {'line8_tensor':>13s}")
print(" " + "-" * 96)
def effective(sf):
    return (
        0.25 * sf["kappa_LS"],
        0.5  * sf["kappa_LL12"],
        0.5  * sf["kappa_SS"],
       -sf["kappa_SS"] / 6.0,
        sf["T_LL"],  # times -0.5 in actual code, but here just show T_LL itself
    )
for (n, L, S, J, lab) in CHANNELS:
    sf = spin_factors(L, S, J)
    e = effective(sf)
    print(f"  {lab:<30s}  {e[0]:>9.4f} {e[1]:>9.4f} {e[2]:>9.4f} "
          f"{e[3]:>13.4f} {e[4]:>13.4f}")
print()

print("=" * 100)
print(" Diagnosis: comparing h_c (1^1P_1) vs chi_c1 (1^3P_1)")
print("=" * 100)
sf_hc  = spin_factors(1, 0, 1)
sf_chi = spin_factors(1, 1, 1)
print(f"  Effective coefficients (h_c minus chi_c1):")
e_hc  = effective(sf_hc)
e_chi = effective(sf_chi)
names = ["line45 (SO)", "line6 (kLL12)", "line7 (sig1.sig2)", "line8_scalar", "T_LL"]
for nm, a, b in zip(names, e_hc, e_chi):
    d = a - b
    print(f"    {nm:<24s}  h_c = {a:+8.4f}   chi_c1 = {b:+8.4f}   diff = {d:+8.4f}")
print()
print("  -> The differences in line6 and line7 are what shift h_c vs chi_c1.")
print("     If h_c is consistently below experiment by ~30-65 MeV across all 9 fits,")
print("     the most likely culprits are factor errors in the COEFFICIENTS of")
print("     line6 (the 0.5) or line7 (the 0.5), since those are the only places")
print("     where h_c differs from chi_c1 at the spin-factor level.")
