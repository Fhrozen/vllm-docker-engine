#!/usr/bin/env bash
# Spike: measure what a selective PyTorch build can actually save, and prove the
# distributed gate empirically rather than by reading source.
#
# This is a spike, not a build path. It clones PyTorch and compiles it, which is
# expensive; run it deliberately, once, and record the numbers in memory.md.
#
# Variants:
#   control        - default build, the baseline
#   no-distributed - USE_DISTRIBUTED=0, expected to break vLLM at import time
#   lean-dist      - distributed kept, everything else optional dropped
#
# Usage: bash tools/spike_torch_distributed.sh [control] [lean-dist]

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLONE="${VDE_TORCH_CLONE:-/tmp/pytorch-spike}"
TORCH_REF="${VDE_TORCH_REF:-v2.13.0}"
OUT="${VDE_SPIKE_OUT:-$REPO/out/spike}"
JOBS="${VDE_JOBS:-$(nproc)}"

mkdir -p "$OUT"

if [ ! -d "$CLONE" ]; then
  git clone --depth 1 --branch "$TORCH_REF" https://github.com/pytorch/pytorch.git "$CLONE"
fi

# Flags that are safe to drop. USE_DISTRIBUTED, USE_C10D_NCCL and USE_GLOO are
# NOT in this list: vLLM imports torch.distributed at module load and uses gloo
# in the serving path even at tp_size=1. See Plan.md D1.
LEAN_FLAGS=(
  -DBUILD_PYTHON=ON
  -DBUILD_TEST=OFF
  -DBUILD_CAFFE2_OPS=OFF
  -DUSE_CUDA=ON
  -DUSE_DISTRIBUTED=ON
  -DUSE_C10D_NCCL=ON
  -DUSE_GLOO=ON
  -DUSE_NCCL=ON
  -DUSE_NUMPY=ON
  -DUSE_KINETO=OFF
  -DUSE_OPENMP=ON
  -DUSE_NUMA=ON
  -DUSE_VULKAN=OFF
  -DUSE_RT=OFF
  -DUSE_ONNX=OFF
  -DUSE_FBGEMM=OFF
  -DUSE_CUDNN=OFF
  -DUSE_MKLDNN=OFF
  -DUSE_MPI=OFF
  -DUSE_UCC=OFF
  -DUSE_LLVM=OFF
  -DUSE_NVTX=OFF
  -DUSE_CUSPARSELT=OFF
  -DUSE_XNNPACK=OFF
  -DUSE_KLEIDIAI=OFF
  -DUSE_ITT=OFF
  -DUSE_PERFETTO=OFF
)

build_variant() {
  local name="$1"
  shift
  local build="$OUT/$name-build"
  local install="$OUT/$name-install"
  mkdir -p "$build" "$install"

  echo "== $name"
  python3 -m pip install --no-cache-dir --target "$install" cmake ninja >/dev/null 2>&1 || true

  PYTHONPATH="$install" python3 "$CLONE/tools/build_libtorch.py" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$install/torch" \
    "${@}" >"$OUT/$name-config.log" 2>&1 || {
      echo "configure failed, see $OUT/$name-config.log"
      return 1
    }

  cmake --build "$build" -j "$JOBS" >"$OUT/$name-build.log" 2>&1 || {
    echo "build failed, see $OUT/$name-build.log"
    return 1
  }

  for lib in libtorch_cpu.so libtorch_cuda.so libtorch_python.so; do
    local f
    f="$(find "$build" -name "$lib" -type f -printf '%s\t%p\n' | head -1)"
    printf '  %-18s %s bytes\n' "$lib" "${f%%$'\t'*}"
  done

  local c10d
  c10d="$(find "$build" -path '*c10d*' -name '*.o' -printf '%s\n' | awk '{s+=$1} END {print s+0}')"
  printf '  %-18s %s bytes of objects\n' "c10d" "$c10d"
}

gate_variant() {
  local name="$1"
  local install="$OUT/$name-install"
  echo "-- gate: $name"

  PYTHONPATH="$install" python3 - <<'PY'
import torch.distributed as dist
print(f"  is_available: {dist.is_available()}")
PY

  PYTHONPATH="$install:$REPO/vllm" python3 - <<'PY'
try:
    import vllm  # noqa: F401
    print("  import vllm: ok")
except Exception as exc:  # noqa: BLE001
    print(f"  import vllm: FAILED - {type(exc).__name__}: {exc}")
PY

  for tp in 1 2; do
    PYTHONPATH="$install:$REPO/vllm" python3 - <<PY
import os
os.environ.setdefault("VLLM_TP", "${tp}")
try:
    from vllm.distributed import parallel_state as ps
    print("  tp=${tp}: parallel_state import ok")
except Exception as exc:  # noqa: BLE001
    print("  tp=${tp}: FAILED -", type(exc).__name__, exc)
PY
  done
}

for variant in "${@}"; do
  case "$variant" in
    control)     build_variant control ;;
    no-distributed) build_variant no-distributed -DBUILD_PYTHON=ON -DUSE_CUDA=ON -DUSE_DISTRIBUTED=OFF ;;
    lean-dist)   build_variant lean-dist "${LEAN_FLAGS[@]}" ;;
    *) echo "unknown variant: $variant (control | no-distributed | lean-dist)"; exit 2 ;;
  esac
done

echo
echo "Record results in memory.md. No claim without a measurement."
