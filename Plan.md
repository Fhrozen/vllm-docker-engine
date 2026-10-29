# Plan of record — slim vLLM docker images via a packaging engine

Read `wiki.md` first for the machine-facing rules. Progress and measurements go in `memory.md`
and `docs/SIZES.md`. Never state a size, a speed, or a reproducibility claim without a measured
number from this repo's own harness.

## Target

- Deliverable: a **vLLM slim GPU image** (+ optional `vllm-omni` overlay image).
- Baseline: official `vllm/vllm-openai`. Gate for the omni phase: **<= 4 GB AND >= 60% smaller**
  than the baseline, measured on the same machine.
- A literal tenth is not reachable for a GPU image: torch + the `nvidia-*` runtime wheels are a
  ~2.5–3.5 GB floor. The honest ceiling is ~60–70% reduction. Say so when reporting.
- `extra_repos/Strata` and `extra_repos/vllm.cpp` are **read-only reference clones**. Their
  behaviour is reimplemented in `packaging/`; nothing imports them, and no file from them enters
  an image.

## Confirmed constraints

- GPU runtime image only. No CPU-only lane.
- Single CUDA arch, detected at build time by command (`packaging/arch.py`), `VDE_CUDA_ARCH` override.
- Artifacts (locks, packages, archives) are generated at build time into `out/` (gitignored).
  Rules files are committed.
- Measure **both** size and speed. Harness model: `Qwen/Qwen3.5-0.8B`.
- Both packaging variants are implemented (engine + thin) and compared; the comparison harness
  picks the default.
- Both bytecode strategies are implemented (host-built archive vs `.pyc`-only ship) and measured.
- Base image: minimal `ubuntu:24.04` + pip `nvidia-*` wheels, driver from the host via
  nvidia-container-toolkit. CUDA base image only if the Phase 2 risk check fails.

## Verified findings that shape the plan

1. **libtorch-only is not viable.** Autograd, Dynamo/FX/Inductor and `torch.distributed` are hard
   runtime deps of vLLM, not training extras. Only `torch.optim`, `LRScheduler`, `torch.testing`,
   `torch.utils.benchmark` are genuinely droppable, and they live inside `libtorch_python.so`, so
   there is no win without a custom PyTorch build.
2. **`USE_DISTRIBUTED=OFF` breaks multi-GPU and single-GPU alike.** `torch.distributed.is_available()`
   is `hasattr(torch._C, "_c10d_init")`; with the flag off only `GroupName` and `ProcessGroup` are
   stubbed, while `vllm/distributed/parallel_state.py` imports `Backend`, `ProcessGroup`, `Store`,
   `_functional_collectives` and `_symmetric_memory` unconditionally at module load. `USE_C10D_NCCL`
   and `USE_GLOO` are `cmake_dependent_option` gated on `USE_DISTRIBUTED`. Distributed stays ON.
   Only `USE_TENSORPIPE`, `USE_MPI`/`USE_C10D_MPI`, `USE_UCC`/`USE_C10D_UCC` and autograd-RPC are
   safely droppable. Gloo must be measured, not assumed — it is used in the serving path
   (`dp_utils.py` default all-reduce under async scheduling, `stateless_coordinator.py`,
   `gpu_worker_sentinel.py`, EPLB `torch_gloo`, `kv_offload` barrier).
   Test: `tools/check_dist_gate.py` (no build) then `tools/spike_torch_distributed.sh` (real build).
3. **uv is not required.** No uv-only feature is used by vLLM's Dockerfile; uv lands in the final
   image (~15–20 MB + `/opt/uv`). The expected winner is "install in a build stage, copy only
   site-packages + the interpreter" with no pip, no uv, no caches, no headers.
4. **vllm.cpp is not the engine** (from-scratch reimplementation, prerelease, no run gate for the
   target model, no distributed, hardcoded block count). Its build/size/gate techniques are
   harvested — see `docs/PACKAGING.md`.

## Phases

- **Phase 1 — research & audit.** Baseline measurement, base-image matrix, dependency
  classification into `packaging/rules/vllm.json`, arch detection, JIT-toolchain decision,
  torch-subset spike (documents the negative result).
- **Phase 2 — docker build spikes.** Control / pruned / lock-only spikes; confirm torch resolves
  `nvidia-*` libs without the CUDA base's ldconfig entries; prune pass (`strip`, `.a`, `pkgconfig`,
  headers, `tests/`); bytecode variants side by side; `tools/compare_packaging.py` picks the default.
- **Phase 3 — vLLM slim image.** `packaging/package.py` + `package_thin.py`, `docker/Dockerfile.vllm`
  (pkg → verify → runtime), entrypoint/healthcheck/smoke-test, compose.
- **Phase 3b — harness.** `tools/measure.sh`: image size, compressed download, cold start,
  tokens/s, peak GPU memory. Results appended to `docs/SIZES.md`.
- **Phase 4 — vllm-omni overlay.** Runs only if the Phase 3 gate is met.
- **Phase 5 — vllm.cpp.** Analysis verdict only; techniques harvested, engine not adopted.

## Excluded

CPU-only builds, ROCm/XPU/NPU/TPU, distroless, alpine (musl vs manylinux wheels), bundled
standalone CPython, signed feeds/installers, model weights baked into the image, CI wiring,
Strata runtime, adopting vllm.cpp as the engine.
