"""Zoomed (r_s, d_s) chi^2 map for v1 / set A, other parameters fixed.

Companion of profile_scan.py: uses its chi2() and scan settings, and the
re-minimised point of min.json (python3 profile_scan.py min).  The lower
limit of r_s is extended below the fit interval (r_s >= 0.3 GeV^-1) so that
the whole valley can be seen; for that the interval check of profile_scan.py
is relaxed for r_s only.  The scan is resumable cell by cell: values already
present in map_zoom.json are kept, so the grid can be extended or the run
split into several short sessions.

Usage:
    python3 chi2map_zoom_scan.py [budget_seconds]     -> map_zoom.json
    python3 make_chi2map_figure.py                    -> chi2map.tex
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
src = open("profile_scan.py").read().split('if __name__ == "__main__":')[0]
exec(src)
BOUNDS[4] = (-5.0, 20.0)          # r_s may go below the fit interval here

best = json.load(open("min.json")); xb = np.array(best["x"])
rs = np.round(np.linspace(-0.3, 3.0, 45), 6)   # step 0.075 GeV^-1
ds = np.round(np.linspace(1.0, 9.0, 33), 6)    # step 0.25  GeV^-1
fn = "map_zoom.json"
old = json.load(open(fn)) if os.path.exists(fn) else None
known = {}
if old:
    for a, dv in enumerate(old["d_s"]):
        row = old["chi2"][a]
        if row is None: continue
        for b, rv in enumerate(old["r_s"]):
            if row[b] is not None:
                known[(round(rv, 6), round(dv, 6))] = row[b]
Z = [[known.get((float(rv), float(dv))) for rv in rs] for dv in ds]

def save():
    json.dump(dict(r_s=[float(v) for v in rs], d_s=[float(v) for v in ds],
                   chi2=Z, x=list(xb), chi2_min=best["chi2"]), open(fn, "w"))

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
t0 = time.time(); n = 0
for a, dv in enumerate(ds):
    for b, rv in enumerate(rs):
        if Z[a][b] is not None: continue
        if time.time() - t0 > budget: break
        x = xb.copy(); x[4] = rv; x[5] = dv
        Z[a][b] = chi2(x); n += 1
    save()
missing = sum(v is None for row in Z for v in row)
print(f"computed {n} cells in {time.time()-t0:.0f} s; missing {missing}")
