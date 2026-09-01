"""
Numerical inputs for Chapter 9.  All at fixed parameters (v2, combined fit)
unless a parameter is explicitly scanned, and always in the production basis.

  hfd     HF(1S) as a function of the regulator width d          (Sect. 9.4)
  hc      which operator produces the h_c inversion              (Sect. 9.3)
  hcvs    h_c - <chi_cJ> as the scalar depth is switched off      (Sect. 9.3)
"""
import sys
import numpy as np

import ho_primitives as HO
import results_extras as R

P2 = dict(R.P2)
CHI = [("1 3P0", 1, 1, 1, 0), ("1 3P1", 1, 1, 1, 1),
       ("1 1P1", 1, 1, 0, 1), ("1 3P2", 1, 1, 1, 2)]
S1 = [("1 1S0", 1, 0, 0, 0), ("1 3S1", 1, 0, 1, 1)]

_true_spin_factors = HO.spin_factors


def patched_spin_factors(zero):
    def f(L, S, J):
        d = dict(_true_spin_factors(L, S, J))
        for k in zero:
            d[k] = 0.0
        return d
    return f


def install(zero):
    import H_full_matrix as HFM
    f = patched_spin_factors(zero) if zero else _true_spin_factors
    HO.spin_factors = f
    HFM.spin_factors = f


def levels(states, par=None, with_ws=True):
    funcs = R.pv2.make_potential_funcs(**(par or P2))
    out = {}
    for (lab, n, L, S, J) in states:
        s = R.solver(n, L, S, J, funcs, with_ws=with_ws)
        out[lab] = 1000.0 * s.self_consistent_E_T(n - 1, 2.0, 5.5)
    return out


def cog(d):
    return (d["1 3P0"] + 3 * d["1 3P1"] + 5 * d["1 3P2"]) / 9.0


if __name__ == "__main__":
    what = sys.argv[1]

    if what == "hfd":
        print("HF(1S) vs the regulator width d  (v2 combined, other par. fixed)")
        print(f"{'d [GeV^-1]':>12s}{'eta_c':>11s}{'J/psi':>11s}{'HF':>9s}")
        rows = []
        for d in (1.0, 1.5, 2.0, 2.5, 3.0, 3.28334, 3.75, 4.5, 5.5, 7.0):
            p = dict(P2); p["d"] = d
            try:
                lv = levels(S1, p)
            except Exception as e:
                print(f"{d:12.3f}   failed: {e}"); continue
            hf = lv["1 3S1"] - lv["1 1S0"]
            rows.append((d, hf))
            print(f"{d:12.3f}{lv['1 1S0']:11.2f}{lv['1 3S1']:11.2f}{hf:9.2f}")
        print("\ncoordinates " + " ".join(f"({d:.4f},{h:.3f})" for d, h in rows))

    elif what == "hc":
        print("1P multiplet with individual spin structures switched off")
        cfgs = [("full", [], True),
                ("kappa_LL = 0", ["kappa_LL12"], True),
                ("kappa_SS = 0", ["kappa_SS"], True),
                ("T_LL = 0", ["T_LL"], True),
                ("kappa_LS = 0", ["kappa_LS"], True),
                ("W_s off", [], False)]
        print(f"{'configuration':16s}{'chi_c0':>10s}{'chi_c1':>10s}"
              f"{'h_c':>10s}{'chi_c2':>10s}{'cog':>10s}{'h_c-cog':>10s}")
        for name, zero, ws in cfgs:
            install(zero)
            try:
                lv = levels(CHI, with_ws=ws)
            finally:
                install([])
            c = cog(lv)
            print(f"{name:16s}{lv['1 3P0']:10.1f}{lv['1 3P1']:10.1f}"
                  f"{lv['1 1P1']:10.1f}{lv['1 3P2']:10.1f}{c:10.1f}"
                  f"{lv['1 1P1']-c:10.1f}")

    elif what == "hcvs":
        print("h_c - <chi_cJ> as the scalar depth is switched off")
        print(f"{'bar_V_s':>10s}{'h_c':>10s}{'cog':>10s}{'h_c-cog':>10s}")
        rows = []
        for vs in (0.86668, 0.70, 0.55, 0.40, 0.25, 0.10, 0.02):
            p = dict(P2); p["bar_V_s"] = vs
            try:
                lv = levels(CHI, p)
            except Exception as e:
                print(f"{vs:10.3f}   failed: {e}"); continue
            c = cog(lv)
            rows.append((vs, lv["1 1P1"] - c))
            print(f"{vs:10.3f}{lv['1 1P1']:10.1f}{c:10.1f}{lv['1 1P1']-c:10.1f}")
        print("\ncoordinates " + " ".join(f"({v:.4f},{h:.2f})" for v, h in rows))

    elif what == "hcd":
        print("h_c - <chi_cJ> and HF(1S) as functions of the regulator width d")
        print(f"{'d':>9s}{'HF(1S)':>10s}{'h_c-cog':>10s}{'<r>_1P [fm]':>13s}")
        rows = []
        for d in (2.5, 3.0, 3.28334, 3.75, 4.5, 5.5, 7.0):
            p = dict(P2); p["d"] = d
            try:
                lp, ls = levels(CHI, p), levels(S1, p)
            except Exception as e:
                print(f"{d:9.3f}   failed"); continue
            funcs = R.pv2.make_potential_funcs(**p)
            s = R.solver(1, 1, 1, 1, funcs)
            ET = s.self_consistent_E_T(0, 2.0, 5.5)
            psi = s.eigh(ET)[1][:, 0] @ s.u[:s.n_states]
            w = s.weights
            rms = 0.1973 * np.sqrt(np.sum(w*psi*psi*s.r**2)/np.sum(w*psi*psi))
            hf, hc = ls["1 3S1"] - ls["1 1S0"], lp["1 1P1"] - cog(lp)
            rows.append((d, hf, hc))
            print(f"{d:9.3f}{hf:10.2f}{hc:10.2f}{rms:13.3f}")
        print("\nHF  " + " ".join(f"({d:.4f},{h:.2f})" for d, h, _ in rows))
        print("hc  " + " ".join(f"({d:.4f},{c:.2f})" for d, _, c in rows))
