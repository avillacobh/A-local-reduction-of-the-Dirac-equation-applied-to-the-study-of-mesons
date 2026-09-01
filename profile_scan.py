"""
Profile-likelihood scans of chi^2 for the v1 interaction on data set A.

For each parameter theta_i, chi^2 is minimised over the remaining five at a
sequence of fixed values of theta_i (warm-started by continuation outwards
from the minimum).  The result is the profile chi^2, whose Delta chi^2 = 1
crossing is the honest 1-sigma interval -- as opposed to the covariance
estimate, which assumes a quadratic surface.

Also produces a 2-D map of chi^2 over (r_s, d_s), the pair that the
covariance matrix reports as degenerate.

Usage:
    python3 profile_scan.py time      # cost of one chi^2
    python3 profile_scan.py min       # local re-minimisation at scan settings
    python3 profile_scan.py profile   # 1-D profiles  -> profiles.json
    python3 profile_scan.py map       # 2-D (r_s,d_s) -> map.json
"""
import csv, json, os, sys, time
import numpy as np
from scipy.optimize import minimize

from salpeter_solver import MesonHamiltonianSolver
import meson_potential as pot_v1

# --- scan settings (stated in the text; see the note on b below) ------------
M_Q, B, NSTATES, NGRID = 1.275, 2.0, 20, 2000
WS_SIGN = -1.0
SIGMA_FLOOR = 0.020

NAMES = ["bar_V_v", "alpha", "d_v", "bar_V_s", "r_s", "d_s"]
# v1 / set A, DE report
X0 = np.array([1.39242, 1.69159, 1.78928, 1.00602, 1.27097, 4.08790])
BOUNDS = [(-1.0, 2.0), (0.05, 5.0), (0.2, 5.0),
          (0.05, 10.0), (0.3, 20.0), (0.2, 10.0)]


class Scaled:
    def __init__(self, f, s): self.f, self.s = f, s
    def __call__(self, r): return self.s * np.asarray(self.f(r), float)


def read_states(path):
    out = []
    for row in csv.DictReader(open(path)):
        out.append(dict(n=int(row["n"]), J=int(row["J"]), L=int(row["L"]),
                        S=int(row["S"]),
                        E=float(row["Experimental_value"]) / 1000.0,
                        sig=float(row["uncertainty"]) / 1000.0))
    return out


STATES = read_states("charmonium_states_1.csv")
SIG_EFF = np.array([np.hypot(s["sig"], SIGMA_FLOOR) for s in STATES])
E_EXP = np.array([s["E"] for s in STATES])


def chi2(x):
    if np.any(x < [b[0] for b in BOUNDS]) or np.any(x > [b[1] for b in BOUNDS]):
        return 1e6
    try:
        f = pot_v1.make_potential_funcs(*x)
    except Exception:
        return 1e6
    pred = np.empty(len(STATES))
    for k, s in enumerate(STATES):
        try:
            sol = MesonHamiltonianSolver(
                L=s["L"], S=s["S"], J=s["J"],
                V_L_func=f["V_L"], V_U_func=f["V_U"], m=M_Q, b=B,
                n_states=NSTATES, N_grid=NGRID,
                V_L_prime=f["V_L_prime"], V_L_pp=f["V_L_pp"],
                V_U_prime=f["V_U_prime"], V_U_pp=f["V_U_pp"],
                V_v_func=Scaled(f["V_v"], WS_SIGN),
                V_v_prime=Scaled(f["V_v_prime"], WS_SIGN),
                V_v_pp=Scaled(f["V_v_pp"], WS_SIGN),
                ws_style="full")
            pred[k] = sol.self_consistent_E_T(s["n"] - 1, 2.0, 5.5)
        except Exception:
            return 1e6
    return float(np.sum(((pred - E_EXP) / SIG_EFF) ** 2))


def minimise_free(x_start, frozen=None, maxfev=90):
    """Nelder-Mead over the parameters not in `frozen` (dict idx->value)."""
    frozen = frozen or {}
    free = [i for i in range(len(NAMES)) if i not in frozen]
    if not free:
        return chi2(np.array(x_start)), np.array(x_start)

    def wrap(y):
        x = np.array(x_start, float)
        for i, v in zip(free, y):
            x[i] = v
        for i, v in frozen.items():
            x[i] = v
        return chi2(x)

    res = minimize(wrap, np.array(x_start)[free], method="Nelder-Mead",
                   options=dict(maxfev=maxfev, xatol=1e-2, fatol=1e-2))
    x = np.array(x_start, float)
    for i, v in zip(free, res.x):
        x[i] = v
    for i, v in frozen.items():
        x[i] = v
    return float(res.fun), x


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "time"

    if what == "time":
        t = time.time(); c = chi2(X0)
        print(f"chi2(X0) = {c:.4f}   in {time.time()-t:.2f} s")

    elif what == "min":
        t = time.time()
        c, x = minimise_free(X0)
        print(f"chi2_min = {c:.4f}  in {time.time()-t:.1f} s")
        print("  " + "  ".join(f"{n}={v:.5f}" for n, v in zip(NAMES, x)))
        json.dump(dict(chi2=c, x=list(x)), open("min.json", "w"), indent=1)

    elif what == "profile":
        import os
        best = json.load(open("min.json"))
        xb, cb = np.array(best["x"]), best["chi2"]
        name = sys.argv[2]
        budget = float(sys.argv[3]) if len(sys.argv) > 3 else 450.0
        i = NAMES.index(name)
        lo, hi = BOUNDS[i]
        centre = xb[i]
        span = min(centre - lo, hi - centre, max(2.0 * abs(centre), 1.0))
        fn = f"prof_{name}.json"
        st = json.load(open(fn)) if os.path.exists(fn) else {
            "rows": [[float(centre), float(cb)]],
            "warm": {"+1": list(xb), "-1": list(xb)}, "k": {"+1": 0, "-1": 0}}
        t0 = time.time()
        for d in ("+1", "-1"):
            sgn = int(d)
            while st["k"][d] < 4 and time.time() - t0 < budget:
                k = st["k"][d] + 1
                g = centre + sgn * span * k / 4.0
                c, xw = minimise_free(np.array(st["warm"][d]),
                                      frozen={i: g}, maxfev=80)
                st["rows"].append([float(g), float(c)])
                st["warm"][d] = list(xw); st["k"][d] = k
                print(f"  {name}={g:9.4f}  chi2={c:9.4f}  "
                      f"({time.time()-t0:.0f} s)", flush=True)
                json.dump(st, open(fn, "w"), indent=1)
        st["rows"].sort()
        json.dump(st, open(fn, "w"), indent=1)
        print(f"{name}: {st['k']['+1']+st['k']['-1']}/8 points done")

    elif what == "map":
        best = json.load(open("min.json"))
        xb = np.array(best["x"])
        rs = np.linspace(0.3, 12.0, 22)
        ds = np.linspace(0.2, 9.0, 22)
        Z = np.zeros((len(ds), len(rs)))
        t0 = time.time()
        for a, dv in enumerate(ds):
            for b_, rv in enumerate(rs):
                x = xb.copy(); x[4] = rv; x[5] = dv
                Z[a, b_] = chi2(x)
            print(f"  row {a+1}/{len(ds)}  ({time.time()-t0:.0f} s)", flush=True)
        json.dump(dict(r_s=list(rs), d_s=list(ds), chi2=Z.tolist(),
                       x=list(xb), chi2_min=best["chi2"]),
                  open("map.json", "w"), indent=1)
        print("done")
