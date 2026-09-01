"""
Unified Salpeter fit driver for charmonium / bottomonium meson masses.

A single CLI consolidates what used to be three separate scripts
(fit_potential.py, fit_potential_v2.py, fit_potential_v3.py).  Plus a 4th
purely-Coulomb option.  All variants share the same self-consistent solver,
fixed HO scale b, LM / DE fitters, and report format.

Choose the potential at the command line:

    python fit_meson.py --potential v1
    python fit_meson.py --potential v2
    python fit_meson.py --potential v3
    python fit_meson.py --potential coulomb
    python fit_meson.py --potential v3 --mode lm --combined --workers -1

Potential summary
-----------------
    v1       6 free  : bar_V_v, alpha, d_v, bar_V_s, r_s, d_s
                       V_v = bar_V_v - (4 alpha / 3) erf(r/d_v) / r
                       V_s = (1/2) bar_V_s [ erf((r - r_s)/d_s) - 1 ]
    v2       5 free  : bar_V_v, alpha, d, bar_V_s, r_s
                       V_v = bar_V_v - (4 alpha / 3) erf(r/d) / r
                       V_s = -bar_V_s exp(-r^2/r_s^2)
    v3       4 free  : alpha, d, bar_V_s, r_s
                       Same as v2 but with bar_V_v = 2 m_q - bar_V_s.
    coulomb  4 free  : bar_V_v, alpha, bar_V_s, r_s
                       V_v = bar_V_v - (4 alpha / 3) / r    (pure Coulomb)
                       V_s = -bar_V_s exp(-r^2/r_s^2)

In every case  V_L = V_v - V_s,  V_U = V_v + V_s.
The HO scale b is fixed (passed via --b, default 2.0 GeV^-1).
"""

import csv
import math
import time
import numpy as np

from H_full_matrix import MesonHamiltonian
from salpeter_solver import MesonHamiltonianSolver, build_variational_solvers

import meson_potential          as _pot_v1
import meson_potential_v2       as _pot_v2
import meson_potential_v3       as _pot_v3
import meson_potential_coulomb  as _pot_coulomb

try:
    from scipy.optimize import least_squares, differential_evolution
    HAS_SCIPY = True
except ImportError:                                          # pragma: no cover
    HAS_SCIPY = False


SPEC_LABEL = ["S", "P", "D", "F", "G", "H"]


class _ScaledFunc:
    """Pickleable wrapper:  r -> sign * fn(r).  Used to flip the global sign of
    V_v^s (the spatial vector potential entering W_s) without touching V_L/V_U
    (which remain governed by V_v through V_L = V_v - V_s, V_U = V_v + V_s).

    DE workers (workers > 1) pickle the cost across processes; the closures
    produced by `make_potential_funcs` pickle fine, but a stacked lambda on top
    of them is more fragile.  This class is a safe, picklable composition.
    """
    __slots__ = ("fn", "sign")

    def __init__(self, fn, sign):
        self.fn   = fn
        self.sign = float(sign)

    def __call__(self, r):
        return self.sign * self.fn(r)


# ===========================================================================
#  Potential registry: one entry per (--potential) choice
# ===========================================================================
# Each entry is a dict with:
#   make_funcs(params, m)  -> dict of V_L, V_L_prime, V_L_pp, V_U, V_U_prime,
#                              V_U_pp callables ready for MesonHamiltonian.
#   param_names            : list[str]   names in the order of `params`
#   param_units            : dict[str, str]
#   param_descr            : dict[str, str]
#   default_x0             : np.ndarray   sensible starting point
#   default_bounds         : list[(lo, hi)]   physical search bounds
#   title                  : str   one-line description for the .txt report
#   formulae               : list[str]  potential formulae lines for the report
#   bar_V_v_extra          : callable(params, m) -> float or None
#                              if not None, this is the DERIVED bar_V_v that
#                              should be reported alongside the free params
#                              (used by v3 only).
# ===========================================================================
# Every callable referenced from POTENTIALS must be a TOP-LEVEL named
# function (no lambdas, no nested defs) so that scipy.optimize.differential_
# evolution can pickle the cost object when workers > 1.

def _funcs_v1(params, m):
    bar_V_v, alpha, d_v, bar_V_s, r_s, d_s = params
    return _pot_v1.make_potential_funcs(bar_V_v=bar_V_v, alpha=alpha, d_v=d_v,
                                        bar_V_s=bar_V_s, r_s=r_s, d_s=d_s)

def _funcs_v2(params, m):
    bar_V_v, alpha, d, bar_V_s, r_s = params
    return _pot_v2.make_potential_funcs(bar_V_v=bar_V_v, alpha=alpha, d=d,
                                        bar_V_s=bar_V_s, r_s=r_s)

def _funcs_v3(params, m):
    alpha, d, bar_V_s, r_s = params
    return _pot_v3.make_potential_funcs(alpha=alpha, d=d,
                                        bar_V_s=bar_V_s, r_s=r_s, m_q=m)

def _funcs_coulomb(params, m):
    bar_V_v, alpha, bar_V_s, r_s = params
    return _pot_coulomb.make_potential_funcs(bar_V_v=bar_V_v, alpha=alpha,
                                              bar_V_s=bar_V_s, r_s=r_s)


def _bar_V_v_from_v3(params, m):
    return 2.0 * m - params[2]


POTENTIALS = {
    "v1": {
        "make_funcs":      _funcs_v1,
        "param_names":     ["bar_V_v", "alpha", "d_v", "bar_V_s", "r_s", "d_s"],
        "param_units":     {"bar_V_v": "GeV",  "alpha": "",  "d_v": "GeV^-1",
                            "bar_V_s": "GeV",  "r_s":  "GeV^-1", "d_s": "GeV^-1"},
        "param_descr":     {
            "bar_V_v": "vector potential constant offset",
            "alpha":   "effective coupling (4 alpha / 3 in V_v)",
            "d_v":     "vector potential range",
            "bar_V_s": "scalar potential depth",
            "r_s":     "scalar potential transition radius",
            "d_s":     "scalar potential transition width",
        },
        # Anchored on v3 LM solution (alpha=3.08, d=2.59, bar_V_s=0.85, r_s=1.18,
        # bar_V_v_derived=1.70).  d_s=0.5 sets a soft erf-step in V_s.
        "default_x0":      np.array([1.70, 3.08, 2.59, 0.85, 1.18, 0.50]),
        "default_bounds":  [(-1.0, 2.0), (0.05, 5.0), (0.05, 5.0),
                            ( 0.05, 10.0),(0.0, 10.0),(0.005, 10.0)],
        "title":           "variant 1, 6 params",
        "formulae":        [
            "V_v(r) = bar_V_v - (4 alpha / 3) erf(r/d_v) / r",
            "V_s(r) = (1/2) bar_V_s [ erf((r - r_s) / d_s) - 1 ]",
            "V_L = V_v - V_s,    V_U = V_v + V_s",
        ],
        "bar_V_v_extra":   None,
    },
    "v2": {
        "make_funcs":      _funcs_v2,
        "param_names":     ["bar_V_v", "alpha", "d", "bar_V_s", "r_s"],
        "param_units":     {"bar_V_v": "GeV",  "alpha": "",  "d": "GeV^-1",
                            "bar_V_s": "GeV",  "r_s":  "GeV^-1"},
        "param_descr":     {
            "bar_V_v": "vector potential constant offset",
            "alpha":   "effective coupling (4 alpha / 3 in V_v)",
            "d":       "vector potential range",
            "bar_V_s": "scalar Gaussian depth   V_s(0) = -bar_V_s",
            "r_s":     "scalar Gaussian width    V_s(r) = -bar_V_s exp(-r^2/r_s^2)",
        },
        # Anchored on v3 LM solution; bar_V_v free (no constraint here),
        # initialised to 2*m_q - bar_V_s = 1.70.
        "default_x0":      np.array([1.70, 3.08, 2.59, 0.85, 1.18]),
        # Bounds tightened around the v3 anchor so DE wastes fewer trial
        # evaluations on unphysical regions (alpha=10 is well beyond OGE,
        # bar_V_v=5 makes 2 m_q - bar_V_v negative, etc.).  ~3-5x breathing
        # room around the anchor in each direction.
        "default_bounds":  [(-1.0, 4.0), (0.05, 6.0), (0.05, 6.0),
                            ( 0.05, 5.0), (0.10, 5.0)],
        "title":           "variant 2, 5 params",
        "formulae":        [
            "V_v(r) = bar_V_v - (4 alpha / 3) erf(r/d) / r",
            "V_s(r) = -bar_V_s exp(-r^2 / r_s^2)",
            "V_L = V_v - V_s,    V_U = V_v + V_s",
        ],
        "bar_V_v_extra":   None,
    },
    "v3": {
        "make_funcs":      _funcs_v3,
        "param_names":     ["alpha", "d", "bar_V_s", "r_s"],
        "param_units":     {"alpha": "",   "d": "GeV^-1",
                            "bar_V_s": "GeV", "r_s": "GeV^-1"},
        "param_descr":     {
            "alpha":   "effective coupling (4 alpha / 3 in V_v)",
            "d":       "vector potential range",
            "bar_V_s": "scalar Gaussian depth   V_s(0) = -bar_V_s",
            "r_s":     "scalar Gaussian width    V_s(r) = -bar_V_s exp(-r^2/r_s^2)",
        },
        # LM-converged values from a fit of charmonium_states_1 with
        # --ws-style full --ws-sign -1 --with-ws --b 2.0 --n-grid 8000.
        # chi^2 ~ 7.85, HF ~ 109 MeV, both 1S to <3 MeV.
        "default_x0":      np.array([3.08, 2.59, 0.85, 1.18]),
        "default_bounds":  [(0.05, 10.0), (0.05, 10.0), (0.05, 10.0), (0.10, 10.0)],
        "title":           "variant 3, 4 params (bar_V_v = 2 m_q - bar_V_s)",
        "formulae":        [
            "V_v(r) = bar_V_v - (4 alpha / 3) erf(r/d) / r",
            "V_s(r) = -bar_V_s exp(-r^2 / r_s^2)",
            "V_L = V_v - V_s,    V_U = V_v + V_s",
            "Constraint:  bar_V_v = 2 m_q - bar_V_s   (NOT a free parameter)",
        ],
        "bar_V_v_extra":   _bar_V_v_from_v3,
    },
    "coulomb": {
        "make_funcs":      _funcs_coulomb,
        "param_names":     ["bar_V_v", "alpha", "bar_V_s", "r_s"],
        "param_units":     {"bar_V_v": "GeV", "alpha": "",
                            "bar_V_s": "GeV", "r_s": "GeV^-1"},
        "param_descr":     {
            "bar_V_v": "vector potential constant offset (asymptotic wall)",
            "alpha":   "effective coupling (4 alpha / 3 in V_v)",
            "bar_V_s": "scalar Gaussian depth   V_s(0) = -bar_V_s",
            "r_s":     "scalar Gaussian width    V_s(r) = -bar_V_s exp(-r^2/r_s^2)",
        },
        # Pure 1/r vector: alpha is the bare OGE coupling (~ 0.4 in Cornell).
        # bar_V_v slightly lower than the regularised case since 1/r is more
        # attractive at small r; bar_V_s, r_s carried from v3.
        "default_x0":      np.array([1.50, 0.45, 0.85, 1.18]),
        "default_bounds":  [(-2.0, 5.0), (0.05, 10.0),
                            ( 0.05, 10.0), (0.10, 10.0)],
        "title":           "variant 'coulomb', 4 params (pure 1/r vector)",
        "formulae":        [
            "V_v(r) = bar_V_v - (4 alpha / 3) / r          (pure Coulomb)",
            "V_s(r) = -bar_V_s exp(-r^2 / r_s^2)",
            "V_L = V_v - V_s,    V_U = V_v + V_s",
            "CAVEAT: V_v, V_v', V_v'' singular at r=0; intermediate Salpeter",
            "        blocks are cutoff-dependent (see meson_potential_coulomb.py).",
        ],
        "bar_V_v_extra":   None,
    },
}


# ===========================================================================
#  Common pipeline (potential-agnostic, fixed b)
# ===========================================================================
def read_states(csv_path):
    states = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            states.append(dict(
                n = int(row["n"]), J = int(row["J"]),
                L = int(row["L"]), S = int(row["S"]),
                E_exp = float(row["Experimental_value"]) / 1000.0,
                sigma = float(row["uncertainty"])         / 1000.0,
            ))
    return states


def build_solvers(params, m, b, unique_LSJ, pot, n_states=10,
                  with_ws=False, ws_sign=+1.0, ws_style='full', **kwargs):
    """Build per-channel MesonHamiltonianSolver at fixed b for each (L,S,J).

    Parameters
    ----------
    with_ws : bool
        If True and the potential exposes ``V_v``/``V_v_prime``/``V_v_pp``,
        the spatial vector contribution
            Ŵ_s = K^dag (alpha_1.alpha_2 V_v^s) K
        is added to the reduced Hamiltonian.  Default False.
    ws_sign : +1.0 or -1.0
        Sign convention for V_v^s relative to V_v.  Choices:
          * ``+1`` (default):  V_v^s =  V_v   (literal, as in the appendix
                               of the model with V_v^s = V_v).
          * ``-1``:            V_v^s = -V_v   (OGE-consistent sign from
                               gamma^mu otimes gamma_mu = gamma^0 gamma_0
                               - gamma . gamma, so the spatial piece enters
                               with the opposite sign of the time piece).
        Only affects the W_s contribution; V_L and V_U are unchanged.
    """
    funcs = pot["make_funcs"](params, m)
    ws_kwargs = {}
    if with_ws and "V_v" in funcs:
        s = float(ws_sign)
        if abs(abs(s) - 1.0) > 1e-12:
            raise ValueError(f"ws_sign must be +1 or -1, got {s}")
        Vv   = funcs["V_v"]
        Vvp  = funcs.get("V_v_prime")
        Vvpp = funcs.get("V_v_pp")
        # Apply sign uniformly via picklable wrappers (needed for DE workers > 1).
        if s == +1.0:
            ws_kwargs = dict(V_v_func=Vv, V_v_prime=Vvp, V_v_pp=Vvpp)
        else:
            ws_kwargs = dict(
                V_v_func  = _ScaledFunc(Vv,   s),
                V_v_prime = _ScaledFunc(Vvp,  s) if Vvp  is not None else None,
                V_v_pp    = _ScaledFunc(Vvpp, s) if Vvpp is not None else None,
            )
    return {(L, S, J): MesonHamiltonianSolver(
                L=L, S=S, J=J,
                V_L_func=funcs["V_L"], V_U_func=funcs["V_U"],
                V_L_prime=funcs["V_L_prime"], V_L_pp=funcs["V_L_pp"],
                V_U_prime=funcs["V_U_prime"], V_U_pp=funcs["V_U_pp"],
                m=m, b=b, n_states=n_states, ws_style=ws_style,
                **ws_kwargs, **kwargs)
            for (L, S, J) in unique_LSJ}


def predict_energies(params, states, m, b, pot,
                     E_T_lo=2.0, E_T_hi=5.5,
                     n_states=10, variational=False, b_values=None,
                     variational_method="continuous",
                     verbose=False, **kwargs):
    """Predict E_T for each state.  Two routing modes:

    * variational=False (default): all channels use the same fixed `b`.
                                    Original behaviour, unchanged.
    * variational=True:  for each channel (L,S,J), find b*(L,S,J) via a
                         grid scan over `b_values` (default 0.5..4.0 in
                         10 steps) and use a solver at that b*.  The `b`
                         argument is ignored.  Per-channel b* is stashed
                         on predict_energies._last_b_per_channel for the
                         report.
    """
    unique = sorted({(s["L"], s["S"], s["J"]) for s in states})
    if variational:
        funcs = pot["make_funcs"](params, m)
        # Translate (with_ws, ws_sign, ws_style) kwargs into V_v_func/V_v_prime/V_v_pp
        # + ws_style, exactly like build_solvers does -- MesonHamiltonianSolver does
        # not accept the high-level flags.
        ws_kwargs = {}
        rest_kwargs = dict(kwargs)
        with_ws  = rest_kwargs.pop("with_ws", False)
        ws_sign  = rest_kwargs.pop("ws_sign", +1.0)
        ws_style = rest_kwargs.pop("ws_style", "full")
        if with_ws and "V_v" in funcs:
            s = float(ws_sign)
            if abs(abs(s) - 1.0) > 1e-12:
                raise ValueError(f"ws_sign must be +1 or -1, got {s}")
            Vv, Vvp, Vvpp = funcs["V_v"], funcs.get("V_v_prime"), funcs.get("V_v_pp")
            if s == +1.0:
                ws_kwargs = dict(V_v_func=Vv, V_v_prime=Vvp, V_v_pp=Vvpp)
            else:
                ws_kwargs = dict(
                    V_v_func  = _ScaledFunc(Vv,   s),
                    V_v_prime = _ScaledFunc(Vvp,  s) if Vvp  is not None else None,
                    V_v_pp    = _ScaledFunc(Vvpp, s) if Vvpp is not None else None,
                )
        entries = build_variational_solvers(
            funcs, m, unique, n_states=n_states,
            b_values=b_values, E_T_lo=E_T_lo, E_T_hi=max(E_T_hi, 10.0),
            method=variational_method,
            H_kwargs=dict(ws_style=ws_style, **ws_kwargs, **rest_kwargs))
        solvers = {ch: e["solver"] for ch, e in entries.items()}
        predict_energies._last_b_per_channel = {ch: e["b_star"]
                                                 for ch, e in entries.items()}
    else:
        solvers = build_solvers(params, m, b, unique, pot,
                                 n_states=n_states, **kwargs)
        predict_energies._last_b_per_channel = {ch: b for ch in unique}

    pred = np.empty(len(states))
    for k, s in enumerate(states):
        ch = (s["L"], s["S"], s["J"])
        if ch not in solvers:
            pred[k] = np.nan
            continue
        try:
            pred[k] = solvers[ch].self_consistent_E_T(
                n_level=s["n"] - 1, E_T_lo=E_T_lo, E_T_hi=E_T_hi)
        except (ValueError, RuntimeError):
            pred[k] = np.nan
        if verbose:
            tag = f"{s['n']} ^{2*s['S']+1}{SPEC_LABEL[s['L']]}_{s['J']}"
            print(f"  {tag:10s}  pred = {pred[k]*1000:8.2f} MeV  "
                  f"exp = {s['E_exp']*1000:8.2f} +/- {s['sigma']*1000:.2f}")
    return pred


def _compute_sigma_eff(states, sigma_floor, hf_sigma_floor=None,
                       sigma_floor_overrides=None):
    """Per-state effective sigma in quadrature.

    Floor priority (highest first):
      1. `sigma_floor_overrides[(n, L, S, J)]` if key matches the state.
      2. `hf_sigma_floor` for hyperfine-defining S-wave states
         (n=1, L=0).  Backwards-compatible with prior CLI.
      3. `sigma_floor` (the global default).

    `sigma_floor_overrides` is an optional dict mapping (n, L, S, J)
    -> floor in GeV.  Use to attribute extra theoretical uncertainty
    to channels with known model limitations (e.g. h_c).

    Returns a numpy array of length len(states).
    """
    sig = np.array([s["sigma"] for s in states])
    floor_v = np.empty(len(states), dtype=float)
    overrides = sigma_floor_overrides or {}
    for i, s in enumerate(states):
        key = (s["n"], s["L"], s["S"], s["J"])
        if key in overrides:
            floor_v[i] = overrides[key]
        elif hf_sigma_floor is not None and s["n"] == 1 and s["L"] == 0:
            floor_v[i] = hf_sigma_floor
        else:
            floor_v[i] = sigma_floor
    return np.sqrt(sig * sig + floor_v * floor_v)


def residuals(params, states, m, b, pot, sigma_floor=0.020,
              hf_sigma_floor=None, sigma_floor_overrides=None, **kwargs):
    pred  = predict_energies(params, states, m, b, pot, **kwargs)
    exp_v = np.array([s["E_exp"] for s in states])
    sig_eff = _compute_sigma_eff(states, sigma_floor, hf_sigma_floor,
                                  sigma_floor_overrides)
    res = (pred - exp_v) / sig_eff
    return np.where(np.isfinite(res), res, 1e6)


def chi_squared(params, states, m, b, pot, sigma_floor=0.020,
                hf_sigma_floor=None, sigma_floor_overrides=None, **kwargs):
    return float(np.sum(residuals(params, states, m, b, pot,
                                   sigma_floor=sigma_floor,
                                   hf_sigma_floor=hf_sigma_floor,
                                   sigma_floor_overrides=sigma_floor_overrides,
                                   **kwargs) ** 2))


# ===========================================================================
#  Fitters (LM and DE + LM polish)
# ===========================================================================
def fit_states(states, x0, m, b, pot, bounds=None, sigma_floor=0.020,
               hf_sigma_floor=None, sigma_floor_overrides=None,
               n_states=15, variational=False,
               b_values=None, verbose=True, max_nfev=1000, **kwargs):
    if not HAS_SCIPY:
        raise RuntimeError("scipy.optimize.least_squares required")
    if bounds is None:
        bounds = pot["default_bounds"]
    lo, hi = zip(*bounds)
    t0 = time.time()
    result = least_squares(
        residuals, x0, bounds=(lo, hi),
        args=(states, m, b, pot),
        kwargs={"sigma_floor": sigma_floor,
                "hf_sigma_floor": hf_sigma_floor,
                "sigma_floor_overrides": sigma_floor_overrides,
                "n_states": n_states,
                "variational": variational, "b_values": b_values, **kwargs},
        method="trf", verbose=2 if verbose else 0, max_nfev=max_nfev)
    t1 = time.time()
    chi2 = float(np.sum(result.fun ** 2))
    dof  = max(len(states) - len(x0), 1)
    try:
        cov    = np.linalg.pinv(result.jac.T @ result.jac) \
                  * (chi2 / dof if dof > 0 else 1.0)
        errors = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        cov, errors = None, np.full(len(result.x), np.nan)
    bar_V_v_extra = pot["bar_V_v_extra"]
    bar_V_v_der   = (bar_V_v_extra(result.x, m)
                     if bar_V_v_extra is not None else None)
    # capture the per-channel b (variational: from solver, fixed: just `b`)
    b_per_channel = dict(getattr(predict_energies, "_last_b_per_channel", {}))
    if verbose:
        print(f"\nLM done in {t1-t0:.1f}s  ({result.nfev} evals)")
        print(f"chi^2 = {chi2:.3f}   dof = {dof}   chi^2/dof = {chi2/dof:.3f}")
        for nm, v, er in zip(pot["param_names"], result.x, errors):
            print(f"  {nm:10s} = {v:+.5f} +/- {er:.5f}")
        if bar_V_v_der is not None:
            print(f"  derived bar_V_v = 2 m_q - bar_V_s = {bar_V_v_der:+.5f} GeV")
        if variational:
            print(f"  variational b* per channel:")
            for ch, bs in sorted(b_per_channel.items()):
                print(f"    (L,S,J)={ch}:  b* = {bs:.4f}")
    return dict(params=result.x, errors=errors, cov=cov,
                chi2=chi2, dof=dof, m_q=m,
                params_named=dict(zip(pot["param_names"], result.x)),
                bar_V_v_derived=bar_V_v_der,
                variational=variational,
                b_per_channel=b_per_channel,
                N_grid=kwargs.get("N_grid", 4000),
                r_max=kwargs.get("r_max", None),
                with_ws=kwargs.get("with_ws", False),
                ws_sign=kwargs.get("ws_sign", +1.0),
                ws_style=kwargs.get("ws_style", "full"),
                variational_method=kwargs.get("variational_method", "continuous"),
                wallclock_s=t1 - t0, n_evals=result.nfev,
                lsq_result=result)


class _DECost:
    """Pickleable DE cost (for scipy workers > 1).  Carries variational
    flag + grid so the cost is reproducible across worker processes."""
    def __init__(self, states, m, b, pot, sigma_floor, hf_sigma_floor,
                 sigma_floor_overrides,
                 n_states, variational, b_values, kwargs):
        self.states, self.m, self.b, self.pot = states, m, b, pot
        self.sigma_floor, self.n_states = sigma_floor, n_states
        self.hf_sigma_floor = hf_sigma_floor
        self.sigma_floor_overrides = sigma_floor_overrides
        self.variational = variational
        self.b_values    = b_values
        self.kwargs = kwargs
    def __call__(self, x):
        return chi_squared(x, self.states, self.m, self.b, self.pot,
                           sigma_floor=self.sigma_floor,
                           hf_sigma_floor=self.hf_sigma_floor,
                           sigma_floor_overrides=self.sigma_floor_overrides,
                           n_states=self.n_states,
                           variational=self.variational,
                           b_values=self.b_values, **self.kwargs)


def fit_states_global(states, m, b, pot, bounds=None, sigma_floor=0.020,
                      hf_sigma_floor=None, sigma_floor_overrides=None,
                      n_states=15, variational=False,
                      b_values=None, popsize=15, maxiter=80, seed=0,
                      workers=-1, polish_with_lm=True, verbose=True,
                      max_nfev=1000, **kwargs):
    """Differential evolution + optional LM polish.
    `max_nfev` is consumed locally (passed only to the LM polish step) and
    NOT forwarded into the DE cost kwargs, otherwise it leaks through
    predict_energies -> build_solvers -> MesonHamiltonianSolver which does
    not accept it."""
    if not HAS_SCIPY:
        raise RuntimeError("scipy.optimize.differential_evolution required")
    if bounds is None:
        bounds = pot["default_bounds"]
    cost = _DECost(states, m, b, pot, sigma_floor, hf_sigma_floor,
                   sigma_floor_overrides,
                   n_states, variational, b_values, kwargs)
    t0 = time.time()
    try:
        de = differential_evolution(cost, bounds, popsize=popsize,
                                     maxiter=maxiter, seed=seed,
                                     polish=False, tol=1e-6, workers=workers,
                                     disp=verbose, updating="deferred")
    except (AttributeError, RuntimeError) as exc:
        if workers != 1:
            print(f"  [DE] multiprocessing failed ({type(exc).__name__}); "
                  f"retrying serial...")
            de = differential_evolution(cost, bounds, popsize=popsize,
                                         maxiter=maxiter, seed=seed,
                                         polish=False, tol=1e-6, workers=1,
                                         disp=verbose, updating="immediate")
        else:
            raise
    t1 = time.time()
    if verbose:
        print(f"\nDE done in {t1-t0:.1f}s  ({de.nfev} evals)   chi^2 = {de.fun:.3f}")
    if polish_with_lm:
        if verbose: print("LM polish...")
        polished = fit_states(states, de.x, m, b, pot, bounds=bounds,
                               sigma_floor=sigma_floor,
                               hf_sigma_floor=hf_sigma_floor,
                               sigma_floor_overrides=sigma_floor_overrides,
                               n_states=n_states,
                               variational=variational, b_values=b_values,
                               max_nfev=max_nfev,
                               verbose=verbose, **kwargs)
        # Carry DE-stage params + timing through so the report can show
        # both stages.  LM-stage timing is already in polished["wallclock_s"];
        # rename it so total wallclock can be reconstructed without ambiguity.
        polished.setdefault("de_params", np.asarray(de.x, dtype=float))
        polished.setdefault("de_chi2",   float(de.fun))
        polished["de_wallclock_s"] = float(t1 - t0)
        polished["de_n_evals"]     = int(de.nfev)
        polished["lm_wallclock_s"] = polished.get("wallclock_s", float("nan"))
        polished["lm_n_evals"]     = polished.get("n_evals", 0)
        return polished
    bar_V_v_extra = pot["bar_V_v_extra"]
    bar_V_v_der   = (bar_V_v_extra(de.x, m)
                     if bar_V_v_extra is not None else None)
    return dict(params=de.x, chi2=float(de.fun),
                de_params=np.asarray(de.x, dtype=float),
                de_chi2=float(de.fun),
                de_wallclock_s=float(t1 - t0),
                de_n_evals=int(de.nfev),
                dof=max(len(states) - len(pot["param_names"]), 1), m_q=m,
                params_named=dict(zip(pot["param_names"], de.x)),
                bar_V_v_derived=bar_V_v_der,
                variational=variational,
                b_per_channel=dict(getattr(predict_energies,
                                            "_last_b_per_channel", {})),
                N_grid=kwargs.get("N_grid", 4000),
                r_max=kwargs.get("r_max", None),
                with_ws=kwargs.get("with_ws", False),
                ws_sign=kwargs.get("ws_sign", +1.0),
                ws_style=kwargs.get("ws_style", "full"),
                variational_method=kwargs.get("variational_method", "continuous"),
                wallclock_s=t1 - t0, n_evals=de.nfev, de_result=de)


# ===========================================================================
#  Pretty report
# ===========================================================================
def _state_term(s):
    return f"{s['n']} ^{2*s['S']+1}{SPEC_LABEL[s['L']]}_{s['J']}"


def format_report(label, fit_res, states, m, b, pot,
                  n_states, sigma_floor, mode):
    p     = fit_res["params"]
    e     = fit_res.get("errors", np.full(len(p), np.nan))
    chi2  = fit_res["chi2"]; dof = fit_res["dof"]
    bar_V_v_der = fit_res.get("bar_V_v_derived")
    variational = fit_res.get("variational", False)
    b_per_ch    = fit_res.get("b_per_channel", {})

    L = []
    L.append("=" * 78)
    L.append(f"Meson Salpeter fit ({pot['title']}) -- {label}")
    L.append("=" * 78); L.append("")
    L.append("Run settings"); L.append("-" * 78)
    L.append(f"  Quark mass m              = {m * 1000:.2f}  MeV")
    if variational:
        L.append(f"  HO oscillator scale b     = VARIATIONAL per channel "
                 f"(see table below)")
    else:
        L.append(f"  HO oscillator scale b     = {b:.4f}  GeV^-1   "
                 f"(~ {b*0.1973:.3f} fm, FIXED)")
    L.append(f"  Basis size n_states       = {n_states}")
    L.append(f"  Theory uncertainty floor  = {sigma_floor * 1000:.2f}  MeV")
    L.append(f"  Optimisation              = {mode}")
    # W_s reduction settings (critical for reproducibility)
    with_ws  = fit_res.get("with_ws", False)
    ws_sign  = fit_res.get("ws_sign", +1.0)
    ws_style = fit_res.get("ws_style", "full")
    if with_ws:
        L.append(f"  Spatial-vector W_s        = ENABLED  "
                 f"(ws_sign={ws_sign:+.1f}, ws_style='{ws_style}')")
    else:
        L.append(f"  Spatial-vector W_s        = DISABLED")
    # ----- Multi-stage wallclock breakdown ---------------------------
    de_wc   = fit_res.get("de_wallclock_s")
    de_nf   = fit_res.get("de_n_evals")
    lm_wc   = fit_res.get("lm_wallclock_s")
    lm_nf   = fit_res.get("lm_n_evals")
    var_wc  = fit_res.get("var_wallclock_s")
    var_nf  = fit_res.get("var_n_evals")
    tot_wc  = fit_res.get("total_wallclock_s")
    if de_wc is not None:
        L.append(f"  Wallclock (DE)            = "
                 f"{de_wc:8.1f} s   (nfev = {de_nf})")
    if lm_wc is not None:
        L.append(f"  Wallclock (LM)            = "
                 f"{lm_wc:8.1f} s   (nfev = {lm_nf})")
    if var_wc is not None:
        L.append(f"  Wallclock (variational)   = "
                 f"{var_wc:8.1f} s   (nfev = {var_nf})")
    if tot_wc is not None:
        L.append(f"  Wallclock (total)         = "
                 f"{tot_wc:8.1f} s")
    elif de_wc is None and lm_wc is None and var_wc is None:
        # Fallback for old-style res dicts (shouldn't happen in new code)
        L.append(f"  Wallclock                 = "
                 f"{fit_res.get('wallclock_s', float('nan')):.1f} s   "
                 f"(nfev = {fit_res.get('n_evals', '?')})")
    L.append("")
    L.append("Potential parameterisation"); L.append("-" * 78)
    for line in pot["formulae"]:
        L.append("  " + line)
    L.append("")
    L.append("Best-fit potential parameters"); L.append("-" * 78)
    L.append(f"  {'param':10s}  {'value':>14s}   +/-   {'error':>14s}   "
             f"{'units':<10s}  description")
    for nm, v, er in zip(pot["param_names"], p, e):
        L.append(f"  {nm:10s}  {v:+14.5f}   +/-   {er:14.5f}   "
                 f"{pot['param_units'][nm]:<10s}  {pot['param_descr'][nm]}")
    if bar_V_v_der is not None:
        L.append(f"  {'bar_V_v':10s}  {bar_V_v_der:+14.5f}   "
                 f"(derived = 2*m_q - bar_V_s, not optimised)        GeV         "
                 f"vector potential constant offset")
    L.append("")
    # ----- Stage 1 (DE) parameters before any LM polish ----------------------
    de_params = fit_res.get("de_params")
    de_chi2   = fit_res.get("de_chi2")
    if de_params is not None:
        L.append("Stage 1: DE-only parameters (before LM polish)")
        L.append("-" * 78)
        L.append(f"  {'param':10s}  {'value':>14s}    units       description")
        for nm, v in zip(pot["param_names"], de_params):
            L.append(f"  {nm:10s}  {v:+14.5f}    "
                     f"{pot['param_units'][nm]:<10s}  {pot['param_descr'][nm]}")
        if de_chi2 is not None:
            L.append(f"  DE chi^2                  = {de_chi2:.4f}    "
                     f"(reduced {de_chi2/dof:.4f})")
        L.append("")
    L.append("Fit quality"); L.append("-" * 78)
    L.append(f"  chi^2                     = {chi2:.4f}")
    L.append(f"  degrees of freedom        = {dof}    "
             f"({len(states)} states - {len(p)} parameters)")
    L.append(f"  reduced chi^2  (chi^2/dof)= {chi2/dof:.4f}")
    L.append("")
    # Re-evaluate the spectrum at the converged parameters FIRST.  This
    # also refreshes predict_energies._last_b_per_channel with the b*
    # corresponding to those exact parameters (the value previously saved in
    # fit_res may be from the optimizer's last trial step, not the converged
    # point).  We use the freshly-refreshed value for the b* table below.
    pred = predict_energies(p, states, m, b, pot, n_states=n_states,
                            variational=variational,
                            variational_method=fit_res.get("variational_method", "continuous"),
                            N_grid=fit_res.get("N_grid", 4000),
                            r_max=fit_res.get("r_max", None),
                            with_ws=fit_res.get("with_ws", False),
                            ws_sign=fit_res.get("ws_sign", +1.0),
                            ws_style=fit_res.get("ws_style", "full"))
    if variational:
        # Prefer the freshly-evaluated b* (at converged params) over the
        # potentially-stale value saved in fit_res.
        fresh = dict(getattr(predict_energies, "_last_b_per_channel", {}))
        if fresh:
            b_per_ch = fresh
    if variational and b_per_ch:
        L.append("Variational HO scale b* per channel"); L.append("-" * 78)
        L.append(f"  {'(L,S,J)':>10s}      {'b*  [GeV^-1]':>14s}   "
                 f"{'~ fm':>8s}")
        for ch in sorted(b_per_ch):
            bs = b_per_ch[ch]
            L.append(f"  {str(ch):>10s}      {bs:14.4f}   "
                     f"{bs * 0.1973:8.3f}")
        L.append("")
    L.append("Per-state predictions (energies in MeV)"); L.append("-" * 78)
    L.append(f"  {'state':<10s}  {'pred':>10s}   {'exp':>10s}   "
             f"{'sigma_exp':>10s}   {'(pred-exp)':>10s}   {'pull':>7s}")
    for s, pv in zip(states, pred):
        E_pred = pv * 1000; E_exp = s["E_exp"] * 1000
        sig    = s["sigma"] * 1000
        sig_eff = math.sqrt(sig * sig + (sigma_floor * 1000) ** 2)
        delta = E_pred - E_exp
        L.append(f"  {_state_term(s):<10s}  {E_pred:10.2f}   {E_exp:10.2f}   "
                 f"{sig:10.3f}   {delta:+10.2f}   {delta/sig_eff:+7.2f}")
    L.append("")
    if fit_res.get("cov") is not None:
        L.append("Parameter correlation matrix"); L.append("-" * 78)
        cov = fit_res["cov"]
        sd = np.sqrt(np.maximum(np.diag(cov), 1e-30))
        corr = cov / np.outer(sd, sd)
        L.append("          " + "  ".join(f"{n:>10s}" for n in pot["param_names"]))
        for i, n in enumerate(pot["param_names"]):
            row = "  ".join(f"{corr[i, j]:+10.3f}" for j in range(len(pot["param_names"])))
            L.append(f"  {n:8s}  {row}")
        L.append("")
    L.append("=" * 78)
    return "\n".join(L)


def write_report(path, *args, **kwargs):
    txt = format_report(*args, **kwargs)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(txt + "\n")
    return txt


# ===========================================================================
#  CLI
# ===========================================================================
if __name__ == "__main__":
    import os, argparse
    HERE = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Unified Salpeter meson-mass fit driver "
                    "(charmonium / bottomonium).  Pick the potential with "
                    "--potential; everything else is shared.")
    p.add_argument("--potential", choices=sorted(POTENTIALS.keys()),
                   default="v3",
                   help="Which V_v / V_s parametrisation to fit "
                        "(default: v3 -- 4 free params, "
                        "bar_V_v = 2 m_q - bar_V_s).")
    p.add_argument("--mode", choices=["lm", "de"], default="de",
                   help="lm: Levenberg-Marquardt only.  "
                        "de: differential evolution + LM polish (default).")
    p.add_argument("--sigma-floor", type=float, default=0.020,
                   help="Theory uncertainty floor in GeV (default 20 MeV).")
    p.add_argument("--hf-sigma-floor", type=float, default=None,
                   help="Theory uncertainty floor in GeV applied ONLY to the "
                        "hyperfine-defining states (n=1, L=0 — i.e. 1^1S_0 "
                        "and 1^3S_1).  If unset, --sigma-floor applies to "
                        "all states.  Lower this (e.g. 0.001 = 1 MeV) to "
                        "force the fit to engage with the hyperfine splitting.")
    p.add_argument("--strict-bounds", action="store_true",
                   help="Tighten the lower bounds on the range parameters "
                        "(d_v, d, r_s, d_s) to ~0.2-0.3 GeV^-1 to prevent the "
                        "fit from collapsing into sub-grid widths.  Cutoff "
                        "(~0.04-0.06 fm) is well below the physical charmonium "
                        "scale (~0.4 fm) but well above the radial grid "
                        "resolution (dr ~ 0.0025 GeV^-1 at N_grid=4000).  "
                        "Use for datasets where the variational LM polish "
                        "tends to drive widths to zero (e.g. charmonium_states_2).")
    p.add_argument("--sigma-floor-overrides", default=None,
                   help="Per-state sigma_floor overrides as semicolon-"
                        "separated triples 'n,L,S,J:floor_MeV'.  "
                        "Example:  '1,1,0,1:5;1,1,1,1:10'  sets the floor "
                        "for 1^1P_1 (h_c) to 5 MeV and 1^3P_1 to 10 MeV.  "
                        "Lower the floor on a state to FORCE the optimizer "
                        "to fit it tighter.  Raise it to attribute model "
                        "uncertainty (relax the constraint).")
    p.add_argument("--b", type=float, default=2.0,
                   help="HO oscillator scale b in GeV^-1 (default 2.0 "
                        "~ 0.4 fm).  Used as the FIXED b unless "
                        "--variational is set.")
    p.add_argument("--variational", action="store_true",
                   help="Use a per-channel variational b*(L,S,J) instead "
                        "of the fixed --b.  For each channel, scans b over "
                        "--b-grid and picks the b that minimises the "
                        "ground-state self-consistent E_T (MacDonald "
                        "variational principle).  Slower per evaluation "
                        "but in principle a tighter upper bound.  Default "
                        "is the fixed-b path (faster, more robust).")
    p.add_argument("--b-grid", type=float, nargs="+", default=None,
                   metavar="B",
                   help="Grid of b values (GeV^-1) for the variational "
                        "search.  Default: 10 points linspace(0.5, 4.0).  "
                        "Only used when --variational is set.")
    p.add_argument("--variational-method", choices=["continuous", "grid"],
                   default="continuous",
                   help="When --variational is active, choose how to find b* "
                        "per channel.  'continuous' (default) uses "
                        "scipy.optimize.minimize_scalar (bounded Brent, ~10-15 "
                        "evals/channel, sub-grid precision).  'grid' uses the "
                        "coarse scan over --b-grid.")
    p.add_argument("--polish-variational", action="store_true",
                   help="After the main fit (LM or DE+LM at fixed b), run an "
                        "additional LM polish step with --variational on.  This "
                        "re-tunes the parameters for the variational basis, "
                        "giving a consistent fit AND variational-quality "
                        "predictions in the report.  Cost: ~3-8 min extra per "
                        "fit (one LM run with ~50 variational evals).")
    p.add_argument("--n-states", type=int, default=25,
                   help="HO basis size (default 15).")
    p.add_argument("--m-q", type=float, default=1.275,
                   help="Quark mass in GeV.  Default 1.275 (charm).  Set "
                        "to 4.18 for bottom.")
    p.add_argument("--csv-prefix", default="charmonium_states_",
                   help="Glob prefix for CSV files in the working directory "
                        "(default 'charmonium_states_').  Use "
                        "'bottomonium_states' for bottomonium.")
    p.add_argument("--combined", action="store_true",
                   help="Also fit all CSVs jointly to one parameter set.")
    p.add_argument("--only-combined", action="store_true",
                   help="Skip the per-CSV fits and ONLY do the combined "
                        "joint fit.  Implies --combined.")
    p.add_argument("--out-dir", default=HERE,
                   help="Directory for the fit_<csv>.txt report(s).")
    p.add_argument("--out-tag", default="",
                   help="Optional tag appended to the report filenames "
                        "(e.g. 'LM' produces fit_v1_floor20MeV_LM_csv1.txt). "
                        "Used to keep parallel runs from overwriting each "
                        "other (e.g. LM-only vs DE+LM).")
    p.add_argument("--workers", type=int, default=-1,
                   help="DE parallelism: -1 all cores (default), 1 serial.")
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--maxiter", type=int, default=80,
                   help="Max DE generations (default 80).")
    p.add_argument("--max-nfev", type=int, default=1000,
                   help="Max function evaluations for the LM polish "
                        "after DE (or for the LM-only fit).  Default "
                        "1000.  Bump higher if you see 'maximum number "
                        "of function evaluations exceeded' in the log.")
    p.add_argument("--n-grid", type=int, default=4000,
                   help="Number of points in the uniform radial grid used by "
                        "the Simpson quadrature and finite-difference "
                        "derivative stencils inside H_full_matrix (default "
                        "4000). Bump to 16000/32000 when r_s is small.")
    p.add_argument("--r-max", type=float, default=None,
                   help="Outer radius of the uniform grid in GeV^-1.  Default "
                        "is auto.  Pass explicit value for convergence study.")
    # W_s defaults to ON with the OGE sign convention (V_v^s = -V_v).
    # Use --no-ws to disable, --ws-sign +1 to use the literal V_v^s = +V_v.
    p.add_argument("--with-ws", dest="with_ws", action="store_true", default=True,
                   help="Activate the spatial vector contribution Ŵ_s = "
                        "K^dag (alpha_1.alpha_2 V_v^s) K -- adds the "
                        "appendix Eq. `spatial` to the reduced Hamiltonian.  "
                        "Default: ON (the new corrected derivation is "
                        "expected to dominate over the legacy reduction "
                        "alone).  Use --no-ws to disable.")
    p.add_argument("--no-ws", dest="with_ws", action="store_false",
                   help="Disable W_s.  Equivalent to the old behaviour before "
                        "the spatial-vector contribution was added.")
    p.add_argument("--ws-sign", type=int, choices=[+1, -1], default=-1,
                   help="Sign convention for V_v^s relative to V_v in W_s.  "
                        "Default -1 (OGE-consistent: V_v^s = -V_v, from "
                        "gamma^mu otimes gamma_mu = gamma^0 gamma_0 - "
                        "gamma.gamma -- the spatial piece carries the "
                        "opposite sign of the time piece).  +1: V_v^s = V_v "
                        "(literal model, only useful for sanity comparison). "
                        "Only affects W_s; V_L and V_U are unchanged.")
    p.add_argument("--ws-style", choices=["full"],
                   default="full",
                   help="Reduction style for the spatial-vector term W_s. "
                        "'full' (legacy): user's full K^dag W_s K with K "
                        "depending on V_L, V_U -- hyperfine source ∝ V_v · V_s "
                        "(V_s factor suppresses HF). "
                        "This is the setting used for every fit reported in "
                        "the thesis. The alternative hybrid style, which used "
                        "a factorised K for W_s, required a module that is not "
                        "part of this repository and has been removed.")
    args = p.parse_args()

    # Per-variant strict bounds (active only with --strict-bounds).
    # Cutoff ~ 0.2-0.3 GeV^-1 keeps the radial scales above the grid resolution
    # (see diagnose_grid.py).  Upper bounds and other params kept identical
    # to the defaults so we don't bias the search globally.
    STRICT_BOUNDS = {
        # v1:  [bar_V_v, alpha, d_v, bar_V_s, r_s, d_s]
        # 2026-05-28: r_s upper 10 -> 20 after v1/csv2 LM-only fit pegged at 9.94.
        "v1": [(-1.0, 2.0), (0.05, 5.0), (0.2, 5.0),
               ( 0.05, 10.0),(0.3, 20.0),(0.2, 10.0)],
        # v2:  [bar_V_v, alpha, d, bar_V_s, r_s]
        # r_s upper 5 -> 20 for consistency (Gaussian width).
        "v2": [(-1.0, 4.0), (0.05, 6.0), (0.2, 6.0),
               ( 0.05, 5.0), (0.3, 20.0)],
        # v3:  [alpha, d, bar_V_s, r_s]
        # r_s upper 10 -> 20.
        "v3": [(0.05, 10.0), (0.2, 10.0), (0.05, 10.0), (0.3, 20.0)],
    }
    if args.strict_bounds and args.potential in STRICT_BOUNDS:
        POTENTIALS[args.potential]["default_bounds"] = STRICT_BOUNDS[args.potential]
        print(f"Strict bounds applied for {args.potential}: "
              f"{STRICT_BOUNDS[args.potential]}")

    # Parse --sigma-floor-overrides into a dict { (n, L, S, J) : floor_GeV }
    sigma_floor_overrides = None
    if args.sigma_floor_overrides:
        sigma_floor_overrides = {}
        for entry in args.sigma_floor_overrides.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            key_part, val_part = entry.split(":")
            n, L, S, J = (int(x) for x in key_part.split(","))
            sigma_floor_overrides[(n, L, S, J)] = float(val_part) / 1000.0
        print(f"sigma_floor overrides (GeV): {sigma_floor_overrides}")

    pot       = POTENTIALS[args.potential]
    m         = args.m_q
    b         = args.b
    n_states  = args.n_states
    variational = args.variational
    b_grid    = (np.asarray(args.b_grid, dtype=float)
                 if args.b_grid is not None else None)
    csvs      = sorted(f for f in os.listdir(HERE)
                       if f.startswith(args.csv_prefix) and f.endswith(".csv"))
    if not csvs:
        raise SystemExit(f"No CSV files matching '{args.csv_prefix}*.csv' "
                         f"in {HERE}")
    mode_label = ("LM" if args.mode == "lm"
                  else "differential evolution + LM polish")

    def run_fit(states, label, out_path):
        print(f"\n========== {label}   "
              f"(potential={args.potential}, {len(states)} states) ==========")
        if not HAS_SCIPY:
            print("  scipy not installed -- skipping fit.")
            return None
        t_run_start = time.time()
        if args.mode == "lm":
            res = fit_states(states, pot["default_x0"], m=m, b=b, pot=pot,
                              sigma_floor=args.sigma_floor,
                              hf_sigma_floor=args.hf_sigma_floor,
                              sigma_floor_overrides=sigma_floor_overrides,
                              n_states=n_states,
                              variational=variational, b_values=b_grid,
                              N_grid=args.n_grid, r_max=args.r_max,
                              max_nfev=args.max_nfev,
                              with_ws=args.with_ws, ws_sign=float(args.ws_sign),
                              ws_style=args.ws_style,
                              variational_method=args.variational_method)
        else:
            # When --polish-variational is set we will re-polish with variational
            # b* anyway; skip the redundant fixed-b LM polish inside DE.
            res = fit_states_global(states, m=m, b=b, pot=pot,
                                     sigma_floor=args.sigma_floor,
                                     hf_sigma_floor=args.hf_sigma_floor,
                                     sigma_floor_overrides=sigma_floor_overrides,
                                     n_states=n_states,
                                     variational=variational, b_values=b_grid,
                                     popsize=args.popsize, maxiter=args.maxiter,
                                     workers=args.workers,
                                     N_grid=args.n_grid, r_max=args.r_max,
                                     max_nfev=args.max_nfev,
                                     polish_with_lm=not args.polish_variational,
                                     with_ws=args.with_ws,
                                     ws_sign=float(args.ws_sign),
                                     ws_style=args.ws_style,
                                     variational_method=args.variational_method)
        # Optional: an extra LM polish with variational=True after the
        # main fit, starting from the fixed-b converged parameters.  Re-tunes
        # the parameters for the variational basis so the reported predictions
        # remain self-consistent with what the optimizer minimized.
        if args.polish_variational and res is not None and not variational:
            print("\n=== Variational LM polish (continuous b*) ===")
            t0 = time.time()
            try:
                res_var = fit_states(
                    states, res["params"], m=m, b=b, pot=pot,
                    sigma_floor=args.sigma_floor,
                    hf_sigma_floor=args.hf_sigma_floor,
                    sigma_floor_overrides=sigma_floor_overrides,
                    n_states=n_states,
                    variational=True, b_values=b_grid,
                    N_grid=args.n_grid, r_max=args.r_max,
                    max_nfev=max(100, args.max_nfev // 4),
                    with_ws=args.with_ws, ws_sign=float(args.ws_sign),
                    ws_style=args.ws_style,
                    variational_method=args.variational_method)
                t1 = time.time()
                print(f"Variational polish: {t1-t0:.1f}s, "
                      f"chi^2 {res['chi2']:.3f} -> {res_var['chi2']:.3f}")
                # Promote the polished result and tag the variational mode.
                # Forward the DE-stage and LM-stage timing/params so they
                # survive the swap (res_var is from fit_states which doesn't
                # know about DE or about the pre-polish LM run).
                de_params_saved   = res.get("de_params")
                de_chi2_saved     = res.get("de_chi2")
                de_wc_saved       = res.get("de_wallclock_s")
                de_nfev_saved     = res.get("de_n_evals")
                lm_wc_saved       = res.get("lm_wallclock_s",
                                            res.get("wallclock_s"))
                lm_nfev_saved     = res.get("lm_n_evals",
                                            res.get("n_evals"))
                res = res_var
                # Re-tag the wallclock fields cleanly:
                #   var_*  : the variational polish stage we just finished
                #   lm_*   : the LM-fixed-b stage that produced res_var's seed
                #   de_*   : the DE stage (only present in DE mode)
                res["var_wallclock_s"] = float(t1 - t0)
                res["var_n_evals"]     = int(res.get("n_evals", 0))
                res["lm_wallclock_s"]  = lm_wc_saved
                res["lm_n_evals"]      = lm_nfev_saved
                if de_params_saved is not None:
                    res["de_params"]       = de_params_saved
                    res["de_chi2"]         = de_chi2_saved
                    res["de_wallclock_s"]  = de_wc_saved
                    res["de_n_evals"]      = de_nfev_saved
                mode_label_used = mode_label + " + variational polish"
            except (ValueError, RuntimeError) as exc:
                print(f"Variational polish FAILED ({type(exc).__name__}: {exc}); "
                      f"keeping fixed-b fit.")
                mode_label_used = mode_label
        else:
            mode_label_used = mode_label
        # Stamp total wallclock (from start of run_fit to here) and a copy
        # of the LM stage's clock for the no-variational-polish branch, so
        # the report can show all stages uniformly.
        if res is not None:
            res["total_wallclock_s"] = float(time.time() - t_run_start)
            if "var_wallclock_s" not in res:
                # No variational polish ran -- the "current stage" IS LM
                # (or DE+LM); make sure lm_wallclock_s is present for the
                # report formatter.
                res.setdefault("lm_wallclock_s", res.get("wallclock_s"))
                res.setdefault("lm_n_evals",     res.get("n_evals"))
        report = write_report(out_path, label, res, states, m, b, pot,
                               n_states, args.sigma_floor, mode_label_used)
        print(f"\nReport written to: {out_path}\n")
        print(report)
        return res

    # Build a "floor tag" string for use in output filenames, so that
    # different sigma-floor runs do not overwrite each other.  Examples:
    #   --sigma-floor 0.020                       -> "_floor20MeV"
    #   --sigma-floor 0.005                       -> "_floor5MeV"
    #   --sigma-floor 0.020 --hf-sigma-floor 0.001 -> "_floor20MeV_hf1MeV"
    def _fmt_floor_mev(x_gev):
        mev = x_gev * 1000.0
        return f"{int(round(mev))}MeV" if abs(mev - round(mev)) < 1e-6 \
               else f"{mev:.2f}MeV"
    floor_tag = f"_floor{_fmt_floor_mev(args.sigma_floor)}"
    if args.hf_sigma_floor is not None:
        floor_tag += f"_hf{_fmt_floor_mev(args.hf_sigma_floor)}"
    if args.out_tag:
        floor_tag += f"_{args.out_tag}"

    # Per-CSV fits (skipped if --only-combined)
    if not args.only_combined:
        for c in csvs:
            out = os.path.join(
                args.out_dir,
                f"fit_{args.potential}{floor_tag}_{c.replace('.csv', '.txt')}")
            run_fit(read_states(os.path.join(HERE, c)), c, out)

    # Combined fit (implied by --only-combined)
    if args.combined or args.only_combined:
        all_states = []
        for c in csvs:
            all_states += read_states(os.path.join(HERE, c))
        out = os.path.join(args.out_dir,
                           f"fit_{args.potential}{floor_tag}_combined.txt")
        run_fit(all_states, f"COMBINED ({', '.join(csvs)})", out)
