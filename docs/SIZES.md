# SIZES.md — measured sizes and speed

Every number in the first table must come from a run in this repo. An estimate
is not a measurement, and a claim that cannot be reproduced is not written here.

## Measured

| Image | On disk | Download | Cold start | tokens/s | Peak GPU MiB |
| --- | --- | --- | --- | --- | --- |
| `vllm/vllm-openai:latest` (baseline) | — | — | — | — | — |
| `vllm-slim` (engine variant) | — | — | — | — | — |
| `vllm-slim` (thin variant) | — | — | — | — | — |
| `vllm-slim` (.pyc-only ship) | — | — | — | — | — |
| `vllm-omni` overlay | — | — | — | — | — |

Fill from `bash docker/run.sh measure <image>`, which records unpacked bytes,
compressed bytes, container start to healthcheck green, tokens/s, and peak GPU
MiB.

## Phase 4 gate

The gate is an engine image **under 6 GB** and **≥ 60% smaller** than the
baseline. 4 GB is kept as the **stretch goal**, so a `--budget 4GB` run is still
a valid stricter measurement; `tools/check_image_size.py` reports the stretch
separately from the gate. Both conditions must be measured numbers.

The omni overlay is only worth building once the engine image passes the gate.

"A tenth of the baseline" is not reachable for a GPU image: torch plus the
NVIDIA runtime libraries are roughly a 2.5–3.5 GB floor before any vLLM code.

## Selection measured

Run inside the container:

```bash
bash docker/run.sh script tools/evaluate_deny.py \
  --wheelhouse out/wheelhouse --venv out/eval-venv \
  --rule vllm --arch sm_86 --view out/eval-view-5
```

Target GPU, read from the container: `NVIDIA GeForce RTX 3060`, compute cap
8.6, so `sm_86`.

| Measurement | Number |
| --- | --- |
| Selection, rule as previously committed | 51,514 files, 11,021.6 MB |
| Selection, rule after the decisions below | 44,441 files, 9,986.6 MB |
| Selection, rule as it now stands (`sm_86`) | 47,917 files, 10,505.7 MB |
| Wheelhouse (compressed download) | 5,364.3 MB, 189 wheels |
| Compressed share of the unpacked selection | 48.7 % |

These are selection bytes, not image bytes: no image has been built from this
rule yet, so the table above stays empty.

Largest trees in the selection, after the decisions:

| Tree | MB | What it is |
| --- | --- | --- |
| `nvidia` (cublas) | 4572.0 | linked by `libtorch_cuda.so` |
| `torch` | 1759.7 | executed at import time |
| `vllm` | 1195.6 | executed at import time |
| `triton` | 585.2 | executed at import time |
| `tokenspeed_triton` | 294.0 | imported by `tokenspeed-mla` |
| `nvidia_cutlass_dsl` | 269.6 | no module-level importer |
| `tilelang` | 147.2 | imported by itself |
| `llvmlite` | 133.6 | imported by `numba` |
| `transformers` | 107.9 | executed at import time |
| `cuda` (bindings) | 106.3 | imported by 11 distributions |

The 18 largest files are 62.9 % of the selection: `libtorch_cuda.so` 1043.8,
`libcublasLt.so.12` 751.8, `libcudnn_engines_precompiled.so.9` 547.4,
`vllm/_C.abi3.so` 465.6, `libcusparseLt.so.0` 452.0, `libtorch_cpu.so` 448.2,
`libnccl.so.2` 429.6, `libtriton.so` 415.6, `libcusparse.so.12` 388.0,
`libcudnn_adv.so.9` 285.2, `libcufft.so.11` 278.9, `_vllm_fa2_C.abi3.so` 263.6,
`_moe_C.abi3.so` 254.7, `libcusolver.so.11` 243.3, `_cutlass_ir` 191.6,
`libcusolverMg.so.11` 161.5, `libnvshmem_host.so.3` 160.3,
`_vllm_fa3_C.abi3.so` 146.7 MB. None of these is prunable by a glob: they are
the libraries libtorch links.

## What the process actually loads

The selection is what the rule ships. What a running process needs is a
different measurement, taken by reading `/proc/self/maps` after a workload
(`tools/measure_runtime_libs.py`, run in a container with a GPU attached):

| Workload | Files mapped | Mapped MB | Not mapped MB |
| --- | --- | --- | --- |
| blas + conv + triton + sparse + distributed | 38 | 5,798.9 | 3,878.1 |
| triton only, warm cache | 34 | 5,165.3 | — |

Mapped trees for the first run: nvidia 25 files (3,807.4 MB), torch 8
(1,537.2 MB), triton 1 (`libtriton.so`, 415.6 MB), numpy 2, numpy.libs 3.

The `DT_NEEDED` closure is 20 libraries / 2,904.2 MB, which badly
underestimates the process: `libcudnn_graph`, `libcudnn_heuristic`,
`libcudnn_engines_precompiled`, `libcudnn_engines_runtime_compiled` and
`libtriton.so` are reached through `dlopen` or `ctypes` from Python and appear
in no ELF `DT_NEEDED`. A "nothing links it" list is therefore a candidate
list, not a verdict.

Libraries the workload did not map, kept as candidates only:

| Library | MB | Why it is not pruned |
| --- | --- | --- |
| `libcudnn_adv.so.9` | 285.2 | a conv shape or vision encoder can reach it |
| `libcusolverMg.so.11` | 161.5 | read by `cuda/pathfinder` through `ctypes` |
| `libnvrtc.alt.so.12` | 105.0 | alternative nvrtc, selected at runtime |
| `libtriton_amd_codegen.so` | 96.0 | AMD codegen, dead on NVIDIA by design |
| `libcute_dsl_runtime.so` | 87.5 | loaded by 4 cutlass source files |
| `libnvperf_host.so` | 51.7 | profiling path, guarded import |
| `libtilelang.so` | 28.6 | loaded by 2 tilelang cache files |
| `libcudnn_cnn.so.9` | 13.1 | conv path not exercised by this shape |
| `libnvperf_target.so` | 10.6 | profiling path, guarded import |

Runtime import closure from the entry path: 90 distributions / 4,161.8 MB —
the set of packages a shipped file imports at module level.

### cupy and ray, measured

Correction to an earlier reading: the serving workload does **not** map cupy.
Cupy appears only in the never-mapped list, as three distributions installed by
one package (`cupy` 221.0 MB, `cupy_backends` 34.4 MB, `cupyx` 9.9 MB =
**265.3 MB**), next to `ray` 53.7 MB.

Unpacked on the real install (`out/run2/vllm.venv`, 196 lock entries):

| Distribution | Files | Unpacked MB | Wheel MB |
| --- | --- | --- | --- |
| `ray` | 5,317 | 230.4 | 78.9 |
| `cupy` | 2,330 | 240.3 | 136.1 (cupy side, one wheel) |
| `cupyx` | 418 | 16.5 | |
| `cupy_backends` | 23 | 34.4 | |
| **cupy side** | **2,771** | **291.2** | **136.1** |

Cupy cannot ship: the cutensor extension in
`cupy_backends/cuda/libs/` (`cutensor.cpython-312-x86_64-linux-gnu.so`) needs
`libcutensor.so.2` and `libcutensorMg.so.2`, which are in no wheel of the lock
and in no folder of the tree. Its other `.so` files are satisfiable
(`libcusparse`, `libcusolver`, `libnccl`), so this one file is the whole
blocker. The deny stands, and it is enforced through the RECORD owner map,
because `cupy-cuda12x` owns three folders.

Both names are in the lock: `ray==2.59.0` and `cupy-cuda12x==14.2.0`. vllm's
requirements files name neither, so they enter through `ray[cgraph]`, a hard
`Requires-Dist` of the vllm wheel. A lock-based build therefore installs cupy
even though the rule denies it — the deny is a policy choice, not a statement
about what the lock resolves.

Runtime branches on the GPU, warm cache, same model and seed:

| Branch | Selection | Result |
| --- | --- | --- |
| cupy pruned, importer pruned | 66,796 files / 10,911.4 MB | OK, 41.6 s |
| cupy pruned, `ray` kept whole | 66,797 files / 10,911.4 MB | OK, 41.8 s |
| `ray` and cupy pruned | 61,480 files / 10,681.1 MB | OK, 40.7 s |

The second row is the point: the cupy import is never reached, so the deny
conflict is static. The gate is stricter than the served path.

### Imports that point at a pruned file, measured

The pruned-file gate found 10 findings from 9 importers (132.5 KB of importers)
pointing at 7 targets (76.3 KB):

Targets, by size:

- `numba/tests/support.py` — 45.4 KB
- `jsonschema/tests/_suite.py` — 8.4 KB
- `numpy/_core/tests/_natype.py` — 6.3 KB
- `torch/testing/_internal/distributed/rpc/examples/`
  `reinforcement_learning_rpc_test.py` — 9.3 KB
- `torch/testing/_internal/distributed/rpc/examples/`
  `parameter_server_test.py` — 4.6 KB
- `torch/_export/db/examples/__init__.py` — 1.6 KB
- `numba/tests/__init__.py` — 0.8 KB
- **total — 76.3 KB**

Importing the importers in the materialised view fails on other names that are
also missing (`pyperf`, `numba.tests`, `hypothesis`, `expecttest`), so neither
side of these imports is reached at runtime. Decision: keep the 7 targets —
76.3 KB is cheaper than breaking an import. After the decision the selection is
47,921 files / 10,505.7 MB and both import gates report 0 findings.

### Triton cache, measured

The runtime image has no compiler, so triton can only run kernels the build
stage compiled. Measured in the container:

- cache hit → runs; cache miss + no compiler → `Failed to find C compiler`.
  The cache is checked **before** the compiler is looked up.
- `gcc` alone cannot compile; `libc6-dev` headers are needed too.
- Cache key is `sha256(src + machine, system, architecture)` — no GPU arch, so
  a stage-1 cache is read in stage 3 on the same x86_64 Linux + Python. The
  kernel artifacts themselves are arch-specific.
- Minimum warm cache for one kernel: **10 files / 0.07 MB**. A serving run
  compiles many kernels, so the shipped cache is small but not fixed in size.
  Measured for the target model on the GPU: **1033 files / 48.6 MB** eager,
  **1710 files / 58.5 MB** compiled.
- JIT files that cannot be pruned even on a full cache hit: `libtriton.so`
  415.6 MB, `libnvJitLink.so.12` 94.1 MB, `ptxas` 31.9 MB. Prunable for this
  model: cupti 110.8 MB, `ptxas-blackwell` 34.9 MB, `cuda_nvrtc` 222.1 MB,
  `tilelang` 176.3 MB, `flashinfer` 53.2 MB — **597.3 MB** in total.
- A C toolchain in the runtime image costs **111.6 MB** installed (`gcc` +
  `libc6-dev`) and forces nvrtc and cupti to stay. Pre-warming the cache is
  about **660 MB** cheaper and keeps the runtime image toolchain-free.

The decision this produces is recorded in `Plan.md` finding 5.

### The loader path, measured

Files being present in the tree is not the same as the loader finding them.
`libdeps.uncovered()` checks that a `DT_NEEDED` name exists somewhere by file
name; `tools/check_loader_paths.py` counts edges as *cache-or-env*, which is what
a real loader does. Measured against the installed tree, with no
`LD_LIBRARY_PATH` and no cache entry for the tree:

- 305 cache-or-env edges, **14 missing**, 1,590 system, 7 driver.
- RUNPATH: 367 ELF files with a dynamic section, 65 with RUNPATH/RPATH, **54 use
  `$ORIGIN`**. Every `torch/lib/*.so` carries
  `$ORIGIN/../../nvidia/*/lib`; `vllm/_C.abi3.so` has none.
- `ctypes.CDLL(vllm/_C.abi3.so)` before torch → fails on `libtorch.so`.
- `import vllm` on a plain `ubuntu:24.04` → works, `vllm._C` loaded, because
  torch is imported first and its RUNPATH covers the wheel folders.
- `ctypes.CDLL` after torch → fails only on `libcuda.so.1`, the driver.
- `ldconfig` with a `ld.so.conf.d` entry for the 41 lib/`.libs` folders: cache
  6.8 KB → 20.0 KB (+13.2 KB, conf 3,169 bytes); direct load then fails only on
  the driver.
- `patchelf --set-rpath` on a copy of `vllm/_C.abi3.so`: load then fails only on
  the driver. patchelf footprint measured as tiny (690 bytes by the probe).

So the CUDA base image is not required at runtime: its value is the
`ld.so.conf.d` cache, not the libraries. The runtime stage needs the driver plus
the wheel libraries and one of three loader fixes. The verdict is in `Plan.md`
finding 6 and `docs/findings.md`.

## Base images measured

On disk / compressed download, from `docker image inspect` and
`docker manifest inspect`:

| Image | On disk | Download |
| --- | --- | --- |
| `ubuntu:24.04` | 0.07 GB | 0.18 GB |
| `nvidia/cuda:12.8.1-base-ubuntu24.04` | 0.28 GB | 0.19 GB |
| `nvidia/cuda:12.8.1-runtime-ubuntu24.04` | 3.17 GB | 3.97 GB |
| `nvidia/cuda:13.3.0-base-ubuntu24.04` | 0.53 GB | 0.51 GB |
| `nvidia/cuda:13.3.0-runtime-ubuntu24.04` | 2.22 GB | 3.03 GB |
| `nvidia/cuda:12.8.1-devel-ubuntu24.04` | 8.75 GB | 6.89 GB |
| `python:3.12-slim` | — | 0.34 GB |

A CUDA runtime base is over the 4 GB stretch goal on its own, which is the strongest
argument for a plain base plus pip wheels. Largest files in
`12.8.1-runtime`: libcublasLt 716.9 MB, libcusparse 368.7 MB, libnccl 365.8 MB.

ld.so cache entries: `ubuntu:24.04` 83 (`libc.conf`, `x86_64-linux-gnu.conf`);
`12.8.1-base` 91 (`000_cuda.conf`, `988_cuda-12.conf`, `nvidia.conf`);
`13.3.0-base` 91 (`000_cuda.conf`, `987_cuda-13.conf`, `nvidia.conf`). A plain
base has no CUDA entries — which is exactly the gap the loader measurements above
fill another way.

Driver ceiling: 580.95.05, reported CUDA 13.0. `13.0.0-base` starts on this host;
`13.1.0-base` and `13.3.0-base` are refused by nvidia-container-toolkit. The
wheelhouse is cu12, so the build-stage toolchain stays on 12.8.1.

Duplicate tree: `nvidia_cutlass_dsl` ships `cu12/lib/libcute_dsl_runtime.so`
(41.6 MB) and `cu13/lib/libcute_dsl_runtime.so` (41.6 MB). `CUDA_VERSION.major`
resolves to 13 in this environment, so the cu12 copy is dead weight — open prune,
pending the CUDA-version decision.

Both branches of that prune were run against the real tree (rule `vllm`, arch
`sm_86`):

| Keep | Files | Tree MB | cutlass MB | Pruned MB |
| --- | --- | --- | --- | --- |
| cu12 | 48,672 | 10,736.2 | 462.6 | 43.7 |
| cu13 | 48,672 | 10,736.3 | 462.7 | 43.6 |
| no series | 48,673 | 10,779.9 | 506.3 | 0.0 |

The branches are the same size, so the decision is which copy the loader
resolves, not which is smaller. The mechanism is settled; the value it uses is
still open, and it comes from the detection variables file (`out/cuda.env`).
The rule's `system_libs` names are `.12` sonames, so they must change if the
declared series becomes 13.

## What the measurements changed

- **flashinfer-cubin is dead weight for this GPU.** 1,114.8 MB unpacked
  (267.7 MB compressed), no module-level importer, reached only through
  `importlib.util.find_spec`. Its kernels are `sm_100` 937.0 MB and `sm_103`
  127.3 MB, with 50.5 MB unmarked: **zero bytes usable on `sm_86`**. Denied.
  flashinfer itself (30.0 MB selected, 53.5 MB tree) is imported at module
  level by core files (`vllm/ir/op.py`, `vllm/v1/worker/gpu_worker.py`), so it
  cannot be pruned. Without cubins it falls back to JIT, which needs `nvcc` at
  runtime and breaks rule R01 — so on this GPU flashinfer ships as inert code
  unless the backend is never selected.
- **`pyc-only` is the smaller build, not the larger one.** The tree already
  carries 24,079 `.pyc` files (394.9 MB) before any build; the selection keeps
  21,064 of them (338.2 MB). A `bytecode: source` build compiles the `.py`
  files it keeps and ships **both** forms — 21,061 `.py` (304.7 MB) plus the
  338.2 MB of bytecode. A `pyc-only` build removes the `.py` mirrors and ships
  bytecode only. Measured on the same rule and arch: engine 10,102.7 MB from
  source against 9,798.0 MB from `.pyc` (−304.7 MB), thin 10,020.8 against
  9,755.0 (−265.8 MB). The earlier reading that `.pyc-only` costs more
  compared the tree's pre-existing bytecode (258.0 MB) with the `.py` read set
  (230.3 MB); that is a selection detail, not what a build delivers.
- **Two prune patterns never fired.** `nvidia/cusparseLt/**` matches nothing
  because the real folder is `nvidia/cusparselt` (452.0 MB shipped);
  `nvidia/cupti/**` matches nothing because the real folder is
  `nvidia/cuda_cupti` (41.3 MB shipped). Both are `DT_NEEDED` by libtorch
  (`libcusparseLt.so.0` by `libtorch_cuda.so` and `libtorch_python.so`,
  `libcupti.so.12` by `libtorch_cpu.so`), so they cannot be pruned at all —
  only dropped by a torch rebuild with `USE_CUSPARSELT=OFF` / `USE_KINETO=OFF`.
  `nvidia/nvvm/**`, `nvidia/*/bin/**` and `nvidia/*/nsight/**` match no folder
  in the tree.
- **The libtorch-only estimate was too low.** Measured as libraries, the
  droppable torch features are cusparselt 452.0 MB, nvshmem 196.2 MB, cupti
  41.3 MB, cufile 3.3 MB — about 692.8 MB, not 100–250 MB. The pip wheel ships
  them enabled, so the saving only exists for a source build.
- **JIT source inside the selection: 104.8 MB** (`.cpp` 87.6 MB, mostly
  cuda-bindings; `.cc` 14.3 MB, mostly tilelang). R01 forbids a compiler at
  runtime, so that source can never be used by the shipped image.
- **Prune patterns that do fire**, measured against the full tree (72,495
  files): `**/include/**` 137.6 MB, `**/*.a` 84.4 MB, `**/tests/**` 70.6 MB,
  `**/*.pyi` 5.6, `**/*.pyx` 5.2, `**/examples/**` 2.6, `**/test/**` 0.6,
  `**/*.cmake` 0.3. `**/benchmarks/**` (2.9 MB) was removed from the rule
  because the entry path reaches `vllm.benchmarks` through the CLI.
- **Two of those patterns break imports.** `**/tests/**` and `**/test/**` prune
  files that shipped modules import at module level — that is why the
  pruned-file import gate exists. Seven targets (76.3 KB) are now exceptions in
  the rule, because shipping them is cheaper than breaking an import.
- **Deny entries are distribution names, not folder names.** `cupy-cuda12x` owns
  `cupy`, `cupyx` and `cupy_backends`, so a deny is only enforced through the
  RECORD owner map. `package.build()` was calling `select()` without it, which
  let a denied distribution ship and made the uncovered gate fire.
- **Names the entry path executes, so they cannot be pruned:**
  `huggingface-hub` 6.8 MB and `setuptools` 7.5 MB. Both were in the deny list
  and are now shipped.

## Reference sizes (not measured here)

From the reference clone's documentation, kept as a shape check only:

| Variant | On disk |
| --- | --- |
| cuda13 | 2.2 GB |
| cuda12 | 2.9 GB |
| rocm-gfx1151 | 1.5 GB |

Download size is about a third of the on-disk size. These are the reference
clone's numbers, not this repo's, and they are not evidence that this repo
reaches them.

## What is expected to move the needle

Ordered by expected effect, each one still needs a run:

- copy-only site-packages plus the interpreter, no pip, no venv
- stdlib pruning: `ensurepip`, `test/`, `tkinter`, `idlelib`, `pydoc_data`,
  `lib2to3`, `distutils`
- `.pyc`-only ship: measured as a cost, not a saving (see above)
- single CUDA arch, chosen by detection rather than the full list — this is
  what made flashinfer-cubin prunable, and it only works where the wheel names
  the arch; the cubin files are hash-named, so `arch_filter` cannot see them
- `strip --strip-debug` as an explicit step, now a real step because binutils
  is installed in the pkg stage
- dropping JIT source, which R01 makes unusable at runtime (104.8 MB)
- a torch source build with the droppable features off: about 692.8 MB, the
  only route to that saving, since the pip wheel ships them enabled

Glob pruning cannot reach the libraries libtorch links. The measured floor for
this rule is the NVIDIA runtime closure plus libtorch plus the vLLM extensions.
