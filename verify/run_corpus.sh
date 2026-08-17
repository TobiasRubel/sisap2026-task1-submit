#!/usr/bin/env bash
# Run a neighbors-pipnn binary over a corpus under a fixed, controlled environment.
#
#   ./verify/run_corpus.sh <binary> <corpus_dir> <out_dir> [threads] [cpuset]
#
# The binary is bind-mounted into the runtime image rather than baked in, so the
# committed binary and a rebuilt one execute against byte-identical libc,
# libhdf5 and libomp. That is what makes an old-vs-new comparison meaningful.
#
# <corpus_dir> must contain the dataset .h5 and a config.json in the layout TIRA
# provides. The command below is exactly the one TIRA runs, verbatim.
set -euo pipefail

BIN=$(realpath "${1:?usage: run_corpus.sh <binary> <corpus_dir> <out_dir> [threads] [cpuset]}")
CORPUS=$(realpath "${2:?missing corpus_dir}")
OUT=$(realpath -m "${3:?missing out_dir}")
THREADS=${4:-8}
CPUSET=${5:-0-7}
IMAGE=${PIPNN_RUNTIME_IMAGE:-pipnn:submit}
# 24g matches the challenge budget and is right for the dev corpora. The full
# 6.35M corpus needs more headroom than the graded run had, because we are not
# trying to fit the budget here, just to reproduce the graph.
MEMORY=${PIPNN_MEMORY:-24g}

# Quantizer flags. The default is the VERBATIM spelling used in the graded TIRA
# run, so this script stays a faithful reproduction of it. PIPNN_FLAGS=modern
# exercises the current spellings instead; both must produce the same graph,
# which is what keeps the legacy aliases honest.
case "${PIPNN_FLAGS:-legacy}" in
  legacy) QFLAGS="-turboquant -tq_bits 8 -build_tq1_panel" ;;
  modern) QFLAGS="-rsq -rsq_bits 8 -build_rsq1_panel" ;;
  *) echo "PIPNN_FLAGS must be 'legacy' or 'modern'" >&2; exit 2 ;;
esac
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

mkdir -p "$OUT"

docker run --rm \
  --cpuset-cpus="$CPUSET" \
  --memory="$MEMORY" \
  -e PARLAY_NUM_THREADS="$THREADS" \
  -v "$BIN":/app/neighbors-pipnn:ro \
  -v "$REPO/portfolio.task1.txt":/app/portfolio.task1.txt:ro \
  -v "$CORPUS":/tira-data/input:ro \
  -v "$OUT":/tira-data/output \
  "$IMAGE" sh -c "/app/neighbors-pipnn -input /tira-data/input/*.h5 \
      -task_description /tira-data/input/config.json -output /tira-data/output \
      -data_type float -dist_func mips $QFLAGS -stream_quantize \
      -query_configs /app/portfolio.task1.txt" \
  > "$OUT/stdout.log" 2>&1

echo "wrote $(find "$OUT" -name '*.h5' | wc -l) result files to $OUT"
