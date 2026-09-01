"""
Four fixed-parameter studies for Chapter 8.  No refits: the potential
parameters are held at their combined best-fit values throughout, so every
difference reported is the effect of the one thing being varied.

  radii   r.m.s. radius and |Phi(0)|^2 of every fitted state        (Sect. 8.6)
  ws      spectrum with W_s off / s=+1 / s=-1                       (Sect. 8.7)
  tensor  spectrum with the corrected vs the earlier A_L(r)         (Sect. 8.7)
  mc      spectrum at m_c = 1.20, 1.275, 1.35 GeV                   (Sect. 8.8)
"""
import csv, sys
import numpy as np

import ho_primitives as HO
from salpeter_solver import MesonHamiltonianSolver
import meson_potential_v2 as pv2
import meson_potential_v3 as pv3

B, NSTATES, NGRID = 2.0, 30, 8000

# Per-channel variational scales b*(L,S,J).  The production spectrum of
# Chapter 8 is computed in this basis, so anything compared against it must use
# it too -- for the four highest states the difference from a fixed b = 2.0
# reaches 7 MeV.  The values below are those printed by
# fit_v2_floor20MeV_DE_combined.txt; `recompute_bstar()` derives them from
# scratch and is used by emit_tables(), so a refit cannot leave them stale.
BSTAR_REPORTED = {(0,0,0):1.8435, (0,1,1):1.8820, (1,0,1):2.0308,
                  (1,1,0):2.0704, (1,1,1):2.0645, (1,1,2):2.1006,
                  (2,1,1):2.2779, (2,1,2):2.3542}
BSTAR = dict(BSTAR_REPORTED)


def recompute_bstar(funcs, ws_sign=-1.0, verbose=True):
    """Redo the per-channel variational search and install the result in
    BSTAR.  Warns if any channel disagrees with the value in the fit report."""
    from salpeter_solver import find_variational_b_continuous
    kw = dict(V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
              V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
              V_v_func=Scaled(funcs["V_v"], ws_sign),
              V_v_prime=Scaled(funcs["V_v_prime"], ws_sign),
              V_v_pp=Scaled(funcs["V_v_pp"], ws_sign),
              ws_style="full", N_grid=NGRID)
    channels = sorted({(L, S, J) for (_, L, S, J, _) in states()})
    for (L, S, J) in channels:
        b, _ = find_variational_b_continuous(L, S, J, funcs["V_L"], funcs["V_U"],
                                             M_Q, NSTATES, H_kwargs=kw)
        if b is None:
            print(f"  WARNING variational search failed for ({L},{S},{J}); "
                  f"keeping {BSTAR.get((L, S, J))}")
            continue
        ref = BSTAR_REPORTED.get((L, S, J))
        if verbose and ref is not None and abs(b - ref) > 5e-3:
            print(f"  WARNING b*({L},{S},{J}) = {b:.4f} "
                  f"but the report says {ref:.4f}")
        BSTAR[(L, S, J)] = b
    return BSTAR
M_Q = 1.275

P2 = dict(bar_V_v=1.92410, alpha=4.11051, d=3.28334,
          bar_V_s=0.86668, r_s=1.43734)          # v2, combined
P3 = dict(alpha=1.90818, d=1.89164, bar_V_s=0.81805, r_s=9.28005)  # v3, combined

LSYM = "SPDFG"
NAME = {(1,0,0,0):r"\eta_c(1S)", (1,0,1,1):r"J/\psi",
        (1,1,1,0):r"\chi_{c0}(1P)", (1,1,1,1):r"\chi_{c1}(1P)",
        (1,1,0,1):r"h_c(1P)", (1,1,1,2):r"\chi_{c2}(1P)",
        (2,0,0,0):r"\eta_c(2S)", (2,0,1,1):r"\psi(2S)",
        (1,2,1,1):r"\psi(3770)", (1,2,1,2):r"\psi_2(3823)",
        (2,1,1,1):r"\chi_{c1}(3872)", (2,1,1,2):r"\chi_{c2}(3930)",
        (3,0,1,1):r"\psi(4040)", (3,1,1,1):r"\chi_{c1}(4140)",
        (4,0,1,1):r"\psi(4230)", (4,1,1,1):r"\chi_{c1}(4274)"}


def states():
    out = []
    for f in ("charmonium_states_1.csv", "charmonium_states_2.csv"):
        for row in csv.DictReader(open(f)):
            out.append((int(row["n"]), int(row["L"]), int(row["S"]),
                        int(row["J"]), float(row["Experimental_value"])))
    return out


def label(n, L, S, J):
    return f"{n} {2*S+1}{LSYM[L]}{J}"


class Scaled:
    def __init__(self, f, s): self.f, self.s = f, s
    def __call__(self, r): return self.s * np.asarray(self.f(r), float)


def solver(n, L, S, J, funcs, ws_sign=-1.0, with_ws=True, m=M_Q, b=None):
    b = BSTAR.get((L, S, J), B) if b is None else b
    kw = {}
    if with_ws:
        kw = dict(V_v_func=Scaled(funcs["V_v"], ws_sign),
                  V_v_prime=Scaled(funcs["V_v_prime"], ws_sign),
                  V_v_pp=Scaled(funcs["V_v_pp"], ws_sign))
    return MesonHamiltonianSolver(
        L=L, S=S, J=J, V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
        m=m, b=b, n_states=NSTATES, N_grid=NGRID,
        V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
        V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
        ws_style="full", **kw)


def spectrum(funcs, **kw):
    out = {}
    for (n, L, S, J, _) in states():
        s = solver(n, L, S, J, funcs, **kw)
        out[label(n, L, S, J)] = 1000.0 * s.self_consistent_E_T(n - 1, 2.0, 5.5)
    return out


def show(cols, headers, ref=None):
    keys = list(cols[0].keys())
    print(f"{'state':12s}" + "".join(f"{h:>12s}" for h in headers))
    for k in keys:
        row = "".join(f"{c[k]:12.2f}" for c in cols)
        print(f"{k:12s}{row}")


if __name__ == "__main__":
    what = sys.argv[1]
    f2 = pv2.make_potential_funcs(**P2)

    if what == "radii":
        print("v2 combined -- r.m.s. radius and S-wave contact density")
        print(f"{'state':12s}{'E_T [MeV]':>12s}{'<r^2>^1/2':>12s}"
              f"{'[fm]':>10s}{'|Phi(0)|^2':>14s}")
        for (n, L, S, J, _) in states():
            s = solver(n, L, S, J, f2)
            ET = s.self_consistent_E_T(n - 1, 2.0, 5.5)
            vals, vecs = s.eigh(ET)
            c = vecs[:, n - 1]
            u = s.u[:s.n_states]
            w = s.weights
            psi = c @ u                       # radial u(r) of the state
            nrm = np.sum(w * psi * psi)
            r2 = np.sum(w * psi * psi * s.r ** 2) / nrm
            # |Phi(0)|^2 = |R(0)|^2/4pi ; R = u/r, only L=0 is nonzero
            if L == 0:
                R0 = np.polyfit(s.r[:40], psi[:40] / s.r[:40], 2)[-1]
                phi0 = R0 ** 2 / (4 * np.pi) / nrm
            else:
                phi0 = 0.0
            print(f"{label(n,L,S,J):12s}{1000*ET:12.2f}{np.sqrt(r2):12.3f}"
                  f"{0.1973*np.sqrt(r2):10.3f}"
                  + (f"{phi0:14.4f}" if L == 0 else f"{'--':>14s}"))

    elif what == "ws":
        off = spectrum(f2, with_ws=False)
        plus = spectrum(f2, ws_sign=+1.0)
        minus = spectrum(f2, ws_sign=-1.0)
        show([off, plus, minus], ["W_s off", "s = +1", "s = -1"])

    elif what == "tensor":
        new = spectrum(f2)
        orig = HO.K_K_tensor_radial

        def buggy(L, J, S, f_grid, fp_grid, u, du, r, weights):
            sf = HO.spin_factors(L, S, J)
            T_LL = sf["T_LL"]
            N = u.shape[0]
            if T_LL == 0.0:
                return np.zeros((N, N))
            A_L = f_grid * (L * (L + 1) - 3) / (r * r) - (2.0/3.0) * fp_grid / r
            M_kin = du @ (du * (f_grid * weights)[None, :]).T
            M_AL = u @ (u * (A_L * weights)[None, :]).T
            return T_LL * (M_kin + M_AL)

        import H_full_matrix as HFM
        HO.K_K_tensor_radial = buggy
        HFM.K_K_tensor_radial = buggy
        old = spectrum(f2)
        HO.K_K_tensor_radial = orig
        HFM.K_K_tensor_radial = orig
        show([new, old], ["corrected", "earlier"])

    elif what == "mc":
        cols, heads = [], []
        for m in (1.20, 1.275, 1.35):
            f3 = pv3.make_potential_funcs(m_q=m, **P3)
            cols.append(spectrum(f3, m=m))
            heads.append(f"m_c={m}")
        show(cols, heads)


# ---------------------------------------------------------------------------
def emit_tables(outdir=".", recompute=True):
    import os
    f2 = pv2.make_potential_funcs(**P2)
    if recompute:
        recompute_bstar(f2)
    LAB = {}
    for (n, L, S, J, _) in states():
        LAB[label(n, L, S, J)] = (rf"\state{{{n}}}{{{2*S+1}}}{{{LSYM[L]}}}{{{J}}}",
                                  NAME[(n, L, S, J)])
    EXP = {label(n, L, S, J): e for (n, L, S, J, e) in states()}
    hdr = ("% Generated by results_extras.py -- do not edit by hand.\n")

    # ---- W_s effect -------------------------------------------------------
    off, on = spectrum(f2, with_ws=False), spectrum(f2)
    L_ = [hdr, r"\begin{table}[H]\centering",
          r"\caption{Effect of the space-vector term at fixed parameters "
          r"(variant~v2, combined best fit). Masses in \si{\MeV}; no refit is "
          r"performed, so every difference is the effect of $\opr{W}_s$ alone.}",
          r"\label{tab:ws-effect}", r"\small",
          r"\begin{tabular}{llccccc}", r"\toprule",
          r"State & $\state{n}{2S+1}{L}{J}$ & $E^{\text{exp}}$ & "
          r"without $\opr{W}_s$ & with $\opr{W}_s$ & shift\\", r"\midrule"]
    for k in off:
        sp, nm = LAB[k]
        L_.append(rf"${nm}$ & ${sp}$ & {EXP[k]:.1f} & {off[k]:.1f} & "
                  rf"{on[k]:.1f} & ${on[k]-off[k]:+.1f}$\\")
    L_ += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(outdir, "ws_effect.tex"), "w").write("\n".join(L_) + "\n")

    # ---- radii ------------------------------------------------------------
    rows = []
    for (n, L, S, J, e) in states():
        s = solver(n, L, S, J, f2)
        ET = s.self_consistent_E_T(n - 1, 2.0, 5.5)
        vals, vecs = s.eigh(ET)
        c = vecs[:, n - 1]
        psi = c @ s.u[:s.n_states]
        w = s.weights
        nrm = np.sum(w * psi * psi)
        rms = np.sqrt(np.sum(w * psi * psi * s.r ** 2) / nrm)
        rows.append((label(n, L, S, J), LAB[label(n, L, S, J)], rms))
    L_ = [hdr, r"\begin{table}[H]\centering",
          r"\caption{Root-mean-square quark--antiquark separation of the "
          r"fitted states (variant~v2, combined fit), computed in the same "
          r"basis as the spectrum, $\nst=30$ with the per-channel $\bstar$ of "
          r"Table~\ref{tab:bstar}. The reduced wave function is normalised by "
          r"$\braket{\Phi|\Phi}=1$; see the caveat on $\opr{Q}$ in the text.}",
          r"\label{tab:radii}", r"\small", r"\begin{tabular}{llcc}", r"\toprule",
          r"State & $\state{n}{2S+1}{L}{J}$ & "
          r"$\langle r^{2}\rangle^{1/2}$ [\si{\GeV^{-1}}] & [\si{fm}]\\",
          r"\midrule"]
    for k, (sp, nm), rms in rows:
        L_.append(rf"${nm}$ & ${sp}$ & {rms:.2f} & {0.1973*rms:.3f}\\")
    L_ += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(outdir, "radii.tex"), "w").write("\n".join(L_) + "\n")
    print("wrote ws_effect.tex, radii.tex")
