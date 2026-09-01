#!/usr/bin/env bash
# LM + variational polish for v1, v2, v3 -- per-CSV + combined, sigma_floor=20 MeV.
# All outputs go to ./fits/ with the "_LM" tag so they DO NOT overwrite the
# DE+LM results (if any) sitting alongside them.
#
# Filename pattern:  fit_{pot}_floor20MeV_LM_{csv-or-combined}.txt
#
# 3 potentials x 1 floor = 3 invocations.  Each invocation produces:
#   fit_{pot}_floor20MeV_LM_charmonium_states_1.txt
#   fit_{pot}_floor20MeV_LM_charmonium_states_2.txt
#   fit_{pot}_floor20MeV_LM_combined.txt
# -> 9 reports total.

set -euo pipefail
cd "$(dirname "$0")"

# Prevent macOS from sleeping while the long LM+variational runs proceed.
# Two layers of protection:
#   (1) `sudo pmset -a disablesleep 1` -- the only thing that lets you
#       CLOSE THE LID without the system sleeping (clamshell sleep is
#       hardware-sensor-driven; caffeinate cannot override it).  Requires
#       sudo; the script asks once and restores the previous value on exit.
#   (2) `caffeinate -i -s -w $$` -- prevents idle sleep and system sleep
#       on AC power.  Sufficient if you keep the lid open OR have an
#       external display attached (clamshell mode).
# Linux (no caffeinate / no pmset) just skips these silently.

# ----- (1) pmset disablesleep (optional, allows lid close) ----------------
if command -v pmset >/dev/null 2>&1; then
    if sudo -n true 2>/dev/null || sudo -v 2>/dev/null; then
        PREV_DISABLESLEEP=$(pmset -g | awk '/SleepDisabled/ {print $NF; exit}')
        [ -z "$PREV_DISABLESLEEP" ] && PREV_DISABLESLEEP=0
        if sudo pmset -a disablesleep 1 2>/dev/null; then
            echo "(lid-close OK -- sleep fully disabled via pmset)"
            trap "sudo pmset -a disablesleep ${PREV_DISABLESLEEP} 2>/dev/null; \
                  echo \"(sleep settings restored: disablesleep=${PREV_DISABLESLEEP})\"" EXIT
        fi
    else
        echo "(sudo not available -- lid close will still sleep the Mac"
        echo " unless you have an external monitor.  Falling back to"
        echo " caffeinate-only.  Re-run with sudo cached if you want lid-close.)"
    fi
fi

# ----- (2) caffeinate fallback (always, harmless if pmset already ran) ----
if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -s -w $$ &
    echo "(caffeinate idle+system sleep prevention active, PID=$!)"
fi

mkdir -p ./fits ./fits/logs

COMMON_FLAGS=(
    --mode lm               # LM-only (no differential evolution)
    --out-tag LM            # 2026-05-29: stamps "_LM" in output filenames so
                            # they don't collide with DE+LM results.
    --combined
    --ws-style full
    --ws-sign -1
    --with-ws
    --b 2.0
    --n-grid 8000
    --n-states 30           # 2026-05-28: bumped 20 -> 30 after basis-convergence
                            # test on v1/csv2 showed chi^2 drops 29 -> 17 from n=20
                            # to n=40 at fixed params.  n=30 vs n=40 differ by
                            # chi^2 = 0.32 -> autovalor converged.  See
                            # test_basis_convergence_v1_csv2.py.
    --max-nfev 500          # LM-only needs more LM iterations than DE+LM (DE seeds
                            # near optimum).
    --polish-variational
    --variational-method continuous
    --strict-bounds         # 2026-05-28: lower bounds protect against sub-grid
                            # collapse.  Upper r_s bumped 10 -> 20 on 2026-05-29
                            # (was binding for v1/csv2).
    --out-dir ./fits
)

run_one () {
    local pot="$1"
    local floor_gev="$2"
    local floor_label="$3"
    local log="./fits/logs/${pot}_floor${floor_label}_LM.log"
    echo "=============================================================="
    echo " RUN: potential=${pot}  sigma_floor=${floor_gev} GeV  (mode=LM)"
    echo " -> log=${log}"
    echo "=============================================================="
    python3 -u fit_meson.py \
        "${COMMON_FLAGS[@]}" \
        --potential "${pot}" \
        --sigma-floor "${floor_gev}" \
        2>&1 | tee "${log}"
    echo
}

for pot in v1 v2 v3; do
    run_one "$pot" 0.020 20MeV
done

echo "=============================================================="
echo "ALL DONE.  LM reports in ./fits/:"
echo "=============================================================="
ls -la ./fits/*_LM_*.txt 2>/dev/null
