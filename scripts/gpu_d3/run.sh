#!/usr/bin/env bash
# D3 inference driver: fold every prepared target with Boltz-2 and Chai-1, writing
# structures + confidence to results/gpu_d3/{boltz2,chai}/<dataset>/. Resumable
# (skips targets already predicted). Wall-clock is dominated by the ColabFold MSA
# server on the first pass (MSAs are cached per unique sequence).
#
# Usage:
#   bash run.sh --smoke                 # first 2 targets of each dataset x model (validate)
#   bash run.sh                         # everything
#   bash run.sh --datasets posebusters_v2   # one dataset only
#   bash run.sh --models boltz2         # one model only
#   MSA_SERVER_URL=http://localhost:8888 bash run.sh   # point at a local MMseqs2 server
set -uo pipefail
cd "$(dirname "$0")"

SMOKE=0
DATASETS="posebusters_v2 plinder_test"
MODELS="boltz2 chai"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done
N_SMOKE=2

RES="results/gpu_d3"
BOLTZ_MSA=(--use_msa_server); CHAI_MSA=(--use-msa-server)
if [ -n "${MSA_SERVER_URL:-}" ]; then
  BOLTZ_MSA+=(--msa_server_url "$MSA_SERVER_URL")
  CHAI_MSA+=(--msa-server-url "$MSA_SERVER_URL")
fi

run_boltz() {  # run_boltz <dataset>
  local ds="$1" out="$RES/boltz2/$1"
  mkdir -p "$out"
  local files; files=$(ls boltz_inputs/"$ds"/*.yaml 2>/dev/null)
  [ "$SMOKE" = 1 ] && files=$(echo "$files" | head -n "$N_SMOKE")
  for y in $files; do
    local name; name=$(basename "$y" .yaml)
    if [ -d "$out/boltz_results_$name/predictions/$name" ]; then
      echo "  [boltz2/$ds] skip $name (done)"; continue
    fi
    echo "  [boltz2/$ds] predict $name"
    venv_boltz/bin/boltz predict "$y" --out_dir "$out" \
      --diffusion_samples 5 --output_format mmcif --override "${BOLTZ_MSA[@]}" \
      || echo "  [boltz2/$ds] FAILED $name"
  done
}

run_chai() {  # run_chai <dataset>
  local ds="$1" out="$RES/chai/$1"
  mkdir -p "$out"
  local files; files=$(ls chai_inputs/"$ds"/*.fasta 2>/dev/null)
  [ "$SMOKE" = 1 ] && files=$(echo "$files" | head -n "$N_SMOKE")
  for fa in $files; do
    local name; name=$(basename "$fa" .fasta)
    local tdir="$out/$name"
    if ls "$tdir"/pred.model_idx_*.cif >/dev/null 2>&1; then
      echo "  [chai/$ds] skip $name (done)"; continue
    fi
    rm -rf "$tdir"    # chai requires a fresh output dir
    echo "  [chai/$ds] fold $name"
    venv_chai/bin/chai-lab fold "${CHAI_MSA[@]}" "$fa" "$tdir" \
      || echo "  [chai/$ds] FAILED $name"
  done
}

for ds in $DATASETS; do
  for m in $MODELS; do
    echo ">> $m / $ds"
    case "$m" in
      boltz2) run_boltz "$ds" ;;
      chai)   run_chai  "$ds" ;;
      *) echo "unknown model $m" ;;
    esac
  done
done

echo ">> inference pass complete. Next: bash score.sh  (then bash package.sh)"
