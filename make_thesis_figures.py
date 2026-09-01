#!/usr/bin/env python3
"""
Generate the spectrum figures of the thesis from the fit reports.

    python3 make_thesis_figures.py             # DE reports (default)
    python3 make_thesis_figures.py --stage LM

Writes into thesis/figures/:
    levels.tex        level diagram by J^PC, with the D Dbar threshold
    levels_zoom.tex   zoom on the 1P multiplet (where the h_c inversion lives)
    residuals.tex     E_th - E_exp per state, with the +-20 MeV theory band

Each file is a bare tikzpicture, to be \\input inside a figure environment.
"""
import argparse, glob, math, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "fits")
OUT  = os.path.join(HERE, "thesis", "figures")
VARIANTS = ["v1", "v2", "v3"]
STYLE = {"v1": ("blue!70!black", "densely dashed", r"v1"),
         "v2": ("red!70!black",  "densely dotted", r"v2"),
         "v3": ("green!45!black", "dashdotted",    r"v3")}
MARK  = {"v1": "*", "v2": "square*", "v3": "triangle*"}

# spectroscopic label -> (J^PC column, short name)
INFO = {
    "1 ^1S_0": (r"$0^{-+}$", r"$\eta_c$"),        "2 ^1S_0": (r"$0^{-+}$", r"$\eta_c(2S)$"),
    "1 ^3S_1": (r"$1^{--}$", r"$J/\psi$"),        "2 ^3S_1": (r"$1^{--}$", r"$\psi(2S)$"),
    "3 ^3S_1": (r"$1^{--}$", r"$\psi(4040)$"),    "4 ^3S_1": (r"$1^{--}$", r"$Y(4230)$"),
    "1 ^3D_1": (r"$1^{--}$", r"$\psi(3770)$"),
    "1 ^1P_1": (r"$1^{+-}$", r"$h_c$"),
    "1 ^3P_0": (r"$0^{++}$", r"$\chi_{c0}$"),
    "1 ^3P_1": (r"$1^{++}$", r"$\chi_{c1}$"),     "2 ^3P_1": (r"$1^{++}$", r"$X(3872)$"),
    "3 ^3P_1": (r"$1^{++}$", r"$X(4140)$"),       "4 ^3P_1": (r"$1^{++}$", r"$X(4274)$"),
    "1 ^3P_2": (r"$2^{++}$", r"$\chi_{c2}$"),     "2 ^3P_2": (r"$2^{++}$", r"$\chi_{c2}(3930)$"),
    "1 ^3D_2": (r"$2^{--}$", r"$\psi_2(3823)$"),
}
COLUMNS = [r"$0^{-+}$", r"$1^{--}$", r"$1^{+-}$", r"$0^{++}$", r"$1^{++}$",
           r"$2^{++}$", r"$2^{--}$"]
THRESHOLD = 3730.0          # D Dbar

def parse(path):
    txt = open(path).read()
    blk = re.search(r"Per-state predictions.*?\n-+\n.*?\n(.*?)\n\n", txt, re.S)
    st = []
    if blk:
        for line in blk.group(1).splitlines():
            m = re.match(r"\s+(\d \^\d[SPDF]_\d)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)"
                         r"\s+([+-][\d.]+)\s+([+-][\d.]+)", line)
            if m:
                st.append(dict(label=m.group(1), pred=float(m.group(2)),
                               exp=float(m.group(3)), sig=float(m.group(4)),
                               delta=float(m.group(5)), pull=float(m.group(6))))
    return st

def write(name, body):
    open(os.path.join(OUT, name), "w").write(body)
    print("  wrote figures/" + name)

def levels(data, present, fname, emin, emax, columns, height, seglen=0.42,
           label_states=True, threshold=True, dx=2.4, minor=50, major=100,
           label_side="above"):
    """Raw-TikZ level diagram.  data[v] = {label: pred}; data['exp'] = {label: E}."""
    yscale = height / (emax - emin)
    def Y(e): return (e - emin) * yscale
    series = ["exp"] + present
    slot = {s: (i - (len(series) - 1) / 2) * (seglen + 0.10)
            for i, s in enumerate(series)}
    L = [r"\begin{tikzpicture}[x=1cm,y=1cm]"]
    # axis
    L.append(r"\draw[->] (-0.55,0) -- (-0.55,%.2f);" % (height + 0.35))
    for e in range(int(emin // minor * minor) + minor, int(emax) + 1, minor):
        if e < emin: continue
        big = (e % major == 0)
        L.append(r"\draw (-0.55,%.3f) -- (%s,%.3f);" % (Y(e), "-0.68" if big else "-0.62", Y(e)))
        if big:
            L.append(r"\node[left,font=\scriptsize] at (-0.72,%.3f) {%d};" % (Y(e), e))
    L.append(r"\node[rotate=90,font=\small] at (-1.55,%.2f) {$M$ [\si{\MeV}]};"
             % (height / 2))
    if threshold:
        L.append(r"\draw[gray,densely dashed] (-0.55,%.3f) -- (%.2f,%.3f);"
                 % (Y(THRESHOLD), (len(columns) - 1) * dx + 1.0, Y(THRESHOLD)))
        L.append(r"\node[right,font=\scriptsize,gray] at (%.2f,%.3f) {$D\bar D$};"
                 % ((len(columns) - 1) * dx + 1.02, Y(THRESHOLD)))
    for ci, col in enumerate(columns):
        x = ci * dx
        L.append(r"\node[font=\small] at (%.2f,-0.55) {%s};" % (x, col))
        for lab, (c, name) in INFO.items():
            if c != col: continue
            for s in series:
                e = data.get(s, {}).get(lab)
                if e is None or not (emin <= e <= emax): continue
                if s == "exp":
                    sty = "black,thick"
                else:
                    col_, dash, _ = STYLE[s]
                    sty = col_ + "," + dash + ",thick"
                x0 = x + slot[s] - seglen / 2
                L.append(r"\draw[%s] (%.3f,%.3f) -- (%.3f,%.3f);"
                         % (sty, x0, Y(e), x0 + seglen, Y(e)))
            if label_states and data.get("exp", {}).get(lab) is not None:
                vals = [data.get(t, {}).get(lab) for t in series]
                vals = [v for v in vals if v is not None and emin <= v <= emax]
                if vals:
                    L.append(r"\node[font=\tiny,anchor=south] at (%.2f,%.3f) {%s};"
                             % (x, Y(max(vals)) + 0.14, name))

    # legend
    xl, yl = 0.0, height + 0.85
    for i, s in enumerate(series):
        xx = xl + i * 2.9
        sty = "black,thick" if s == "exp" else STYLE[s][0] + "," + STYLE[s][1] + ",thick"
        name = "experiment" if s == "exp" else STYLE[s][2]
        L.append(r"\draw[%s] (%.2f,%.2f) -- (%.2f,%.2f);" % (sty, xx, yl, xx + 0.5, yl))
        L.append(r"\node[right,font=\scriptsize] at (%.2f,%.2f) {%s};" % (xx + 0.55, yl, name))
    L.append(r"\end{tikzpicture}")
    write(fname, "\n".join(L) + "\n")

def residuals(states, preds, present, fname):
    n = len(states)
    labs = ",".join(r"{$\state{%s}{%s}{%s}{%s}$}" %
                    re.match(r"(\d) \^(\d)([SPDF])_(\d)", s["label"]).groups()
                    for s in states)
    L = [r"\begin{tikzpicture}",
         r"\begin{axis}[",
         r"  width=\textwidth, height=7.2cm,",
         r"  xmin=0.4, xmax=%.1f," % (n + 0.6),
         r"  xtick={%s}," % ",".join(str(i + 1) for i in range(n)),
         r"  xticklabels={%s}," % labs,
         r"  xticklabel style={rotate=60, anchor=east, font=\scriptsize},",
         r"  ylabel={$E^{\text{th}}-E^{\text{exp}}$ [\si{\MeV}]},",
         r"  ylabel style={font=\small}, yticklabel style={font=\scriptsize},",
         r"  grid=major, grid style={black!10},",
         r"  legend style={font=\scriptsize, at={(0.02,0.03)}, anchor=south west,",
         r"                draw=black!30, fill=white, fill opacity=0.9, text opacity=1},",
         r"]",
         r"\addplot[draw=none, fill=black!8, forget plot] coordinates "
         r"{(0.4,-20) (%.1f,-20) (%.1f,20) (0.4,20)} \closedcycle;" % (n + 0.6, n + 0.6),
         r"\addplot[black, thin, forget plot] coordinates {(0.4,0) (%.1f,0)};" % (n + 0.6)]
    for v in present:
        col, _, name = STYLE[v]
        pts = " ".join("(%d,%.1f)" % (i + 1, preds[v][s["label"]] - s["exp"])
                       for i, s in enumerate(states) if s["label"] in preds[v])
        L.append(r"\addplot[only marks, mark=%s, mark size=1.9pt, %s] coordinates {%s};"
                 % (MARK[v], col, pts))
        L.append(r"\addlegendentry{%s}" % name)
    L += [r"\end{axis}", r"\end{tikzpicture}"]
    write(fname, "\n".join(L) + "\n")

# ---------------------------------------------------------------------------
#  Experimental values are taken from the CSVs, not from the fit reports, so
#  that updating the data (e.g. a new PDG edition) does not require refitting.
#  Predictions come from the reports; Delta, pull and chi^2 are recomputed.
# ---------------------------------------------------------------------------
SIGMA_FLOOR = 20.0        # MeV, must match --sigma-floor of the fits
_LMAP = {"S": 0, "P": 1, "D": 2, "F": 3}

def load_csv_exp():
    """(n, L, S, J) -> (exp, sigma) from charmonium_states_*.csv"""
    out = {}
    for fn in sorted(glob.glob(os.path.join(HERE, "charmonium_states_?.csv"))):
        with open(fn) as fh:
            next(fh)
            for line in fh:
                if not line.strip():
                    continue
                n, J, L, S, E, sig = line.split(",")
                out[(int(n), int(L), int(S), int(J))] = (float(E), float(sig))
    return out

def label_key(label):
    m = re.match(r"(\d) \^(\d)([SPDF])_(\d)", label)
    n, mult, Ls, J = int(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
    return (n, _LMAP[Ls], (mult - 1) // 2, J)

def apply_csv_exp(rep, exp_table):
    """Overwrite exp/sigma with the CSV values and recompute Delta, pull, chi^2."""
    chi2 = 0.0
    for st in rep["states"]:
        key = label_key(st["label"])
        if key not in exp_table:
            continue
        E, sig = exp_table[key]
        st["exp"], st["sig"] = E, sig
        st["delta"] = st["pred"] - E
        st["pull"] = st["delta"] / math.sqrt(sig * sig + SIGMA_FLOOR ** 2)
        chi2 += st["pull"] ** 2
    if rep["states"]:
        rep["chi2"] = chi2
        if rep.get("dof"):
            rep["chi2dof"] = chi2 / rep["dof"]
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="DE", choices=["DE", "LM"])
    ap.add_argument("--dataset", default="combined")
    a = ap.parse_args()

    present, data, preds, states = [], {}, {}, None
    EXP = load_csv_exp()
    for v in VARIANTS:
        p = os.path.join(FITS, f"fit_{v}_floor20MeV_{a.stage}_{a.dataset}.txt")
        if not os.path.exists(p):
            print(f"  MISSING {os.path.basename(p)} -- {v} omitted from the figures")
            continue
        st = apply_csv_exp({"states": parse(p)}, EXP)["states"]
        present.append(v)
        preds[v] = {s["label"]: s["pred"] for s in st}
        data[v] = preds[v]
        if states is None:
            states = st
    if not present:
        raise SystemExit("no reports found")
    data["exp"] = {s["label"]: s["exp"] for s in states}

    levels(data, present, "levels.tex", 2900, 4400, COLUMNS, 12.0,
           minor=50, major=200, dx=2.4)
    levels(data, present, "levels_zoom.tex", 3390, 3625,
           [r"$0^{++}$", r"$1^{++}$", r"$1^{+-}$", r"$2^{++}$"], 6.5,
           seglen=0.55, threshold=False, minor=25, major=50, dx=2.9)
    residuals(states, preds, present, "residuals.tex")
    print("done.")

if __name__ == "__main__":
    main()
