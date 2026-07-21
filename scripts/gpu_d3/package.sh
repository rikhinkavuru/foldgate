#!/usr/bin/env bash
# Bundle the small artifacts to copy back off the GPU box: the scored table plus every
# native confidence file (Boltz confidence JSON, Chai scores NPZ). Excludes the large
# structures and PAE/pLDDT tensors. Result is a few MB.
set -euo pipefail
cd "$(dirname "$0")"
RES="results/gpu_d3"
OUT="$RES/d3_package.tar.gz"

[ -f "$RES/scored.csv" ] || { echo "no scored.csv; run bash score.sh first"; exit 1; }

find "$RES/boltz2" -name 'confidence_*.json' > "$RES/.pkg_list" 2>/dev/null || true
find "$RES/chai"   -name 'scores.model_idx_*.npz' >> "$RES/.pkg_list" 2>/dev/null || true

tar -czf "$OUT" -C . "$RES/scored.csv" -T "$RES/.pkg_list"
rm -f "$RES/.pkg_list"
echo ">> wrote $OUT ($(du -h "$OUT" | cut -f1))"
echo ">> copy back: scp <box>:$(pwd)/$OUT ."
