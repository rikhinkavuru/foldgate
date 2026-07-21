#!/usr/bin/env bash
# D3 GPU box setup: install Boltz-2, Chai-1, and the torch-free scorer on a fresh
# CUDA Ubuntu instance (e.g. Lambda H100). Boltz and Chai each pin their own torch,
# so they get ISOLATED venvs; the scorer is torch-free in a third venv (mirrors the
# repo convention of keeping inference separate from analysis). Idempotent.
set -euo pipefail
cd "$(dirname "$0")"

BOLTZ_VER="2.2.1"     # Boltz-2 (MIT)
CHAI_VER="0.6.1"      # Chai-1 (Apache-2.0)
PY="${PYTHON:-python3.11}"

echo ">> checking GPU"
nvidia-smi -L || { echo "no GPU visible; run on a CUDA box"; exit 1; }

# uv for fast, reproducible venvs
if ! command -v uv >/dev/null 2>&1; then
  echo ">> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

mk() {  # mk <venv_dir> <pyver>
  [ -d "$1" ] || uv venv --python "$2" "$1"
}

echo ">> venv_boltz (boltz==${BOLTZ_VER})"
mk venv_boltz 3.11
VIRTUAL_ENV=venv_boltz uv pip install --python venv_boltz/bin/python "boltz==${BOLTZ_VER}"

echo ">> venv_chai (chai_lab==${CHAI_VER})"
mk venv_chai 3.11
VIRTUAL_ENV=venv_chai uv pip install --python venv_chai/bin/python "chai_lab==${CHAI_VER}"

echo ">> venv_score (torch-free: gemmi, spyrmsd, pandas)"
mk venv_score 3.11
# spyrmsd's symmrmsd (graph isomorphism) REQUIRES a graph backend at runtime; networkx is
# the default and rustworkx is the fast one. Without one, every RMSD score fails after the
# full run. scipy is a spyrmsd core dep. rdkit is NOT needed (we use the low-level API).
VIRTUAL_ENV=venv_score uv pip install --python venv_score/bin/python \
  "gemmi>=0.7" "spyrmsd>=0.8" "networkx>=3" "rustworkx>=0.14" "scipy>=1.11" \
  "pandas>=2.0" "numpy>=1.26" "pyarrow>=15"
echo ">> verify spyrmsd graph backend"
venv_score/bin/python -c "
import numpy as np
from spyrmsd import rmsd as r
a=np.array([[0,0,0],[1.5,0,0]]); n=np.array([6,8]); adj=np.array([[0,1],[1,0]])
print('  spyrmsd symmrmsd OK:', round(float(r.symmrmsd(a,a,n,n,adj,adj,minimize=False)),3))
"

echo ">> versions"
venv_boltz/bin/boltz --help >/dev/null 2>&1 && echo "  boltz OK: $(venv_boltz/bin/python -c 'import boltz;print(boltz.__version__)' 2>/dev/null || echo installed)"
venv_chai/bin/chai-lab --help >/dev/null 2>&1 && echo "  chai-lab OK: $(venv_chai/bin/python -c 'import chai_lab;print(chai_lab.__version__)')"
venv_score/bin/python -c "import gemmi,spyrmsd,pandas; print('  score env OK')"

echo ">> setup complete. Next: bash run.sh --smoke"
