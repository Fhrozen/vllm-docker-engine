# Plan of record — slim vLLM docker images via a packaging engine

Read `wiki.md` first for the machine-facing rules. Progress and measurements go in `memory.md`
and `docs/SIZES.md`. Never state a size, a speed, or a reproducibility claim without a measured
number from this repo's own harness.

## Decision rule

This is an experimental research repo. **When a decision is not clear, evaluate
both options and select from the results.** A decision stays open until both
branches have been run, and each branch records the command that produced its
number. Findings so far are collected in `docs/findings.md`.

## Target

- Deliverable: a **vLLM slim GPU image** (+ optional `vllm-omni` overlay image).
- Baseline: official `vllm/vllm-openai`. Gate for the omni phase: **under 6 GB**
  AND **>= 60% smaller** than the baseline, measured on the same machine. 4 GB is
  kept as the **stretch goal**, so a stricter run (`--budget 4GB`) is still valid
  and is reported separately from the gate.
- A literal tenth is not reachable for a GPU image: torch + the `nvidia-*` runtime wheels are a
  ~2.5–3.5 GB floor. The honest ceiling is ~60–70% reduction. Say so when reporting.
- `extra_repos/Strata` and `extra_repos/vllm.cpp` are **read-only reference clones**. Their
  behaviour is reimplemented in `packaging/`; nothing imports them, and no file from them enters
  an image.

## Confirmed constraints

- GPU runtime image only. No CPU-only lane.
- Single CUDA arch, detected at build time by command (`packaging/arch.py`), `VDE_CUDA_ARCH` override.
- **Personal/local images.** The CUDA version is decided from the installed environment by
  command, not by a default in a Dockerfile: driver version and installed CUDA are read, stored
  in a variables file (`out/cuda.env`), and used to select and generate the image. A shared
  image is possible, but the plan targets images that work on a specific machine.
  **A CUDA version the user sets wins over detection** (`--cuda` or `VDE_CUDA`).
  The only condition on it is the installed driver: if the driver cannot run it,
  the script raises an error naming the driver as the cause and says to use a
  different CUDA version or update the driver.
- Artifacts (locks, packages, archives) are generated at build time into `out/` (gitignored).
  Rules files are committed.
- Measure **both** size and speed. Harness model: `Qwen/Qwen3.5-0.8B`.
- **A 401 from a registry or package index is reported, not retried.** Source and
  reason are printed, and the user is asked to check availability or download the
  file directly. Nothing retries, changes credentials, or guesses an alternate
  URL.
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
5. **R01 and triton are in conflict unless the cache is pre-warmed.** triton raises
   `Failed to find C compiler` on a cache miss, and the cache is checked before the compiler,
   so a runtime image with no compiler can only run kernels the build stage compiled. gcc
   alone is not enough (compiling needs `libc6-dev` headers). The cache key carries no GPU
   arch, so the build stage must warm it on the target arch and ship it. Minimum measured warm
   cache: 10 files / 0.07 MB. **Open decision:** ship a build-stage-warmed cache (R01 stands),
   or relax R01 for the runtime image. Numbers in `docs/SIZES.md`.
6. **The CUDA base image is not required at runtime.** Its value is the dynamic
   linker configuration, not the libraries — the libraries come from the pip
   wheels. Measured with no `LD_LIBRARY_PATH` and no cache entry for the tree:
   `ctypes.CDLL(vllm/_C.abi3.so)` before torch fails on `libtorch.so`; `import
   vllm` on a plain `ubuntu:24.04` works because torch is imported first and every
   `torch/lib/*.so` carries a `$ORIGIN/../../nvidia/*/lib` RUNPATH; a
   `ld.so.conf.d` entry (3.2 KB config, 13.2 KB cache) or a RUNPATH rewrite makes
   a direct load work. `vllm/_C.abi3.so` has no RUNPATH of its own. Of 367 ELF
   files with a dynamic section, 65 have RUNPATH/RPATH, 54 use `$ORIGIN`. The devel
   image is a **build-stage toolchain only**; the runtime stage needs the driver
   (never bundled) plus the wheel libraries. Details in `docs/findings.md`.
7. **The CUDA tag is capped by the driver.** Host driver 580.95.05 reports CUDA
   13.0. `13.0.0-base-ubuntu24.04` starts; `13.1.0` and `13.3.0` fail with
   `unsatisfied condition: cuda>=13.1` / `>=13.3`, even though those tags exist on
   the registry. The default `12.8.1-devel-ubuntu24.04` is chosen because it
   matches the cu12 wheelhouse (torch 2.10.0 pins every `nvidia-*` dependency to
   cu12), not because 13.3 is unavailable.
8. **The CUDA version comes from commands, and there are two different values.**
   `packaging/cuda.py` reads the driver (`nvidia-smi`), the installed toolchain
   (`nvcc --version`), the wheelhouse (the floor: a cu12 wheel cannot be relabelled
   cu13) and the installed tree, then writes `out/cuda.env`, which `docker/build.sh`
   sources. The **declared series** is what the package ships (12 here, from the
   wheelhouse); the **copy that survives** is what the environment resolves (13 here,
   from `nvidia_cutlass_dsl`). Both branches were run against the real tree: keeping
   cu12 prunes 43.7 MB, keeping cu13 prunes 43.6 MB, keeping both prunes nothing
   (506.3 MB of cutlass). The choice is not a size decision — it is which copy the
   loader actually uses, so the surviving copy is the one the environment resolves.
   The variables file is sourced by shell, so every value must be quoted: an
   unquoted `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a command.
   Details in `docs/findings.md`.
9. **The 401 wording has one source.** `packaging/report.py` holds the sentence;
   `docker/pull.sh` is the shell copy (a shell script cannot import the module),
   and `tests/test_report.py` fails if the two drift. Measured in the container:
   a fabricated 401 detail prints the sentence, an already-local image returns
   without pulling, and a tag the registry refuses prints the failure without the
   401 sentence and exits 1 (`nvidia/cuda:12.4.1-devel-ubuntu24.04`). A real
   network 401 has not been observed since the rule was written.

## Phases

- **Phase 1 — research & audit.** Baseline measurement, base-image matrix, dependency
  classification into `packaging/rules/vllm.json`, arch detection, JIT-toolchain decision,
  torch-subset spike (documents the negative result).
- **Phase 2 — docker build spikes.** Control / pruned / lock-only spikes; the loader
  check is already confirmed (finding 6: torch's `$ORIGIN` RUNPATH resolves the
  `nvidia-*` libs on a plain base, so no CUDA base image is needed at runtime);
  prune pass (`strip`, `.a`, `pkgconfig`, headers, `tests/`); bytecode variants side
  by side; `tools/compare_packaging.py` picks the default.
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
