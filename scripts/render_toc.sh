#!/bin/bash
# Render the TOC graphic panels with ChimeraX, then composite the ACS layout.
# Run from the repo root after installing ChimeraX:  bash scripts/render_toc.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Locate the ChimeraX executable across the usual macOS/Linux install spots.
CX=""
for c in \
  /Applications/ChimeraX*.app/Contents/MacOS/ChimeraX \
  "$HOME"/Applications/ChimeraX*.app/Contents/MacOS/ChimeraX \
  "$(command -v chimerax 2>/dev/null || true)" \
  "$(command -v ChimeraX 2>/dev/null || true)"; do
  if [ -n "$c" ] && [ -x "$c" ]; then CX="$c"; break; fi
done
if [ -z "$CX" ]; then
  echo "ChimeraX not found. Install from https://www.cgl.ucsf.edu/chimerax/download.html"
  echo "then re-run: bash scripts/render_toc.sh"
  exit 1
fi
echo "using ChimeraX: $CX"

# The superposed PDBs must exist (and pass their validation gate) before rendering.
if [ ! -f results/toc_render/5sku_pred.pdb ]; then
  echo "superposed PDBs missing; run scripts/export_toc_superposed.py first"
  exit 1
fi

"$CX" --nogui --offscreen --exit --script scripts/toc_chimerax.cxc
echo "panels rendered; compositing ACS layout"
.venv/bin/python scripts/make_toc_graphic_f.py
echo "done -> results/figures/toc_option_f.png"
