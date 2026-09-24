"""Zoomed (r_s, d_s) chi^2 map for v1 / set A, other parameters fixed.

Companion of profile_scan.py: uses its chi2() and scan settings, and the
re-minimised point of min.json (python3 profile_scan.py min).  The scan is
resumable: rows already present in map_zoom.json are skipped, so it can be
run in several short sessions.

Usage:
    python3 chi2map_zoom_scan.py [budget_seconds]     -> map_zoom.json
    python3 make_chi2map_figure.py                    -> chi2map.tex
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, ".")
src = open("profile_scan.py").read().split('if __name__ == "__main__":')[0]
exec(src)

best = json.load(open("min.json")); xb = np.array(best["x"])
rs = np.linspace(0.3, 3.0, 37)     # step 0.075 GeV^-1
ds = np.linspace(1.0, 9.0, 33)     # step 0.25  GeV^-1
fn = "map_zoom.json"
st = json.load(open(fn)) if os.path.exists(fn) else dict(
    r_s=list(rs), d_s=list(ds), chi2=[None] * len(ds), x=list(xb),
    chi2_min=best["chi2"])
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 1e9
t0 = time.time()
for a, dv in enumerate(ds):
    if st["chi2"][a] is not None:
        continue
    if time.time() - t0 > budget:
        break
    row = []
    for rv in rs:
        x = xb.copy(); x[4] = rv; x[5] = dv
        row.append(chi2(x))
    st["chi2"][a] = row
    json.dump(st, open(fn, "w"))
    print(f"row {a+1}/{len(ds)} d_s={dv:.2f} min={min(row):.2f} "
          f"({time.time()-t0:.0f} s)", flush=True)
done = sum(r is not None for r in st["chi2"])
print(f"DONE {done}/{len(ds)} rows")
