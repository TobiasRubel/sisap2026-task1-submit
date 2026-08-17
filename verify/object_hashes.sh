#!/usr/bin/env bash
# Print the content hash of every object the submission binary is built from.
#
#   ./verify/object_hashes.sh [container-name]
#
# This is the primary check when removing supposedly-dead code: if an object's
# hash is unchanged, nothing in it changed, full stop. It needs no normalization
# and cannot give a false positive, unlike comparing disassembly (which shifts
# whenever anything moves in the link).
#
# Expect exactly the objects you edited to change, and no others.
set -euo pipefail
C=${1:-pipnn-dev}
docker exec "$C" bash -c '
  cd /app/external/PipNN/build/algorithms/PipNN
  sha256sum \
    CMakeFiles/neighbors-pipnn_FLOAT_T_MIPS.dir/__/bench/neighborsTime.C.o \
    CMakeFiles/neighbors-pipnn_FLOAT_T_MIPS.dir/cmake_pch.hxx.pch \
    librsq_kernels.a 2>/dev/null' \
| awk '{n=split($2,p,"/"); printf "%s  %s\n", substr($1,1,16), p[n]}'
