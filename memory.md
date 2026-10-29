# Progress log

Continuity mechanism. Update it as work happens: measurements, decisions, dead ends, blockers,
next action. Every number here must come from a run in this repo, not from an estimate.

## Status

| Phase | State | Notes |
| --- | --- | --- |
| 0 — files & plan | done | `Plan.md`, `memory.md`, `wiki.md`, `docs/`, agents |
| 1 — package research | complete (research + measurement) | findings folded into `Plan.md`, numbers in `docs/SIZES.md` |
| 2 — docker build spikes | not started | needs a runnable base image (Docker Hub pulls previously failed with 401; local `vllm/vllm-openai:latest` works) |
| 3 — vLLM slim image | tooling written, gates + tests green in docker | `packaging/` + `docker/` are in place |
| 3b — harness | tooling written, no measurements yet | `tools/measure.sh` |
| 4 — vllm-omni overlay | blocked on the Phase 3 gate | gate: <= 4 GB AND >= 60% smaller than baseline |
| 5 — vllm.cpp | verdict delivered | engine rejected, techniques harvested |

## Measurements

Nothing measured yet. `docs/SIZES.md` holds the table; fill it from `tools/measure.sh` output.

### Gate and test runs

Everything below ran inside the container (`bash docker/run.sh ...`), not on the host.

| Run | Result |
| --- | --- |
| `gate` on 27 explicit files | clean after fixes; first run failed on two files with two final newlines |
| `test` (63 tests) | exit 0, all passed |
| `check_files.py` with no files | exit 1, refuses — a gate with no arguments is not a pass |
| `dist --scan` | 41 unguarded / 0 guarded in `vllm/vllm/distributed`, 4 / 0 in `v1/worker` |
| determinism (two builds, two out folders) | byte-identical archives |
| `test` (70 tests) | exit 0, all passed |
| `test` (71 tests) | exit 0, all passed |
| `test` (73 tests) | exit 0, all passed |
| `test` (74 tests) | exit 1, 2 failed — `runtime_conflicts` resolved a file's owner from its first path segment instead of the `RECORD` map, so a package importing its own submodules counted as a conflict, and the first fixture had no `RECORD` for the denied dist |
| `test` (76 tests) | exit 0, all passed after the ownership fix |
| `test` (77 tests) | exit 0, all passed after the selection fix |
| `test` (88 tests) | exit 0, all passed after the evaluation decisions |
| `test` (91 tests) | exit 0, all passed after the 15 uncovered findings were decided |
| `test` (92 tests) | exit 0, all passed after the maps-reader test |
| `gate` on `tools/measure_runtime_libs.py` `tests/test_packaging.py` `docker/run.sh` | clean |
| `gate` on `docs/SIZES.md` `docs/PACKAGING.md` `docs/DOCKER.md` `packaging/rules/vllm.json` `docker/Dockerfile.vllm` `docker/run.sh` | clean |
| `gate` on `docs/SIZES.md` `docs/PACKAGING.md` `docs/DOCKER.md` `docker/run.sh` `wiki.md` after the docs pass | clean; markdownlint 0 issues in 3 gated files, `wiki.md` skipped as outside the gated pattern. Re-gated `docs/SIZES.md` alone after the triton cache subsection: clean |
| `test` re-run after the docs pass | 92 passed, exit 0 |
| GPU read through the harness (`bash docker/run.sh gpu`) | `NVIDIA GeForce RTX 3060`, compute cap 8.6 → `sm_86` |
| evaluation run after the decisions | entry import completed in round 1, gate clean |
| first build, hard gate | printed nothing: `bytecode: pyc-only` removes the `.py` files from the shipped tree, so `pkg.rglob("*.py")` found no files and the gate passed on an empty set. The scan now takes the selection (`package.select`) and reads the source tree, so the chosen files are scanned whichever form they ship in |
| `gate` on `packaging/importscan.py` `packaging/package_thin.py` `tools/check_dist_gate.py` `tests/test_packaging.py` | clean |
| `gate` on `docker/build.sh` | clean |
| module-level import scan of the clone | `huggingface_hub` 12, `torchvision` 10, `flashinfer` 7, `torchaudio` 3, `ninja` 0, `sentencepiece` 0, `setuptools` 0, `model-hosting-container-standards` 0 |
| `build --variant engine` | failed at stage 1: `ModuleNotFoundError: No module named 'importscan'` — the context is built from `git ls-files`, which lists tracked files only, so the new `packaging/importscan.py` was not in the context |

### Build runs

First real build through `bash docker/run.sh build --variant engine`. Stage 1 resolves
the requirement set inside the target builder container; the wheelhouse is then hashed
into the lock and checked against the declared target.

What the runs showed:

- `pip download --platform manylinux_2_28_x86_64` does not widen across manylinux
  revisions: `blake3` publishes only `manylinux_2_17_x86_64` wheels (99 files on PyPI)
  and is rejected. `pyyaml` matched only because its filename already carried 2_28.
- `--abi cp312` hides `abi3` wheels, so `triton` (cp312-abi3) is invisible to the
  resolver; `flashinfer-python`, `tilelang` and `xgrammar` hit
  `ResolutionImpossible` under explicit cross-target flags.
- Decision: resolve inside the target environment (the builder container), and enforce
  the target afterwards by reading tags out of every wheel filename — accepted python,
  accepted abi (`cp312`/`abi3`/`none`), and a glibc floor of 2.39 for `ubuntu:24.04`,
  with the legacy aliases `manylinux1/2010/2014` mapped to their floors. musllinux is
  rejected because the runtime base is glibc. `--cross` keeps the explicit-flag path for
  a target the machine cannot run, and is documented as best-effort.
- The first native run downloaded 165 wheels and wrote the lock; `requirements: 9 denied
  name(s) not downloaded` is the deny list applied to the download set.
- `--arch auto` reached `normalise()` and stopped the build ("cannot read a CUDA
  architecture from 'auto'"). `arch.resolve()` now reads `auto` as detection.
- The thin install stage failed its own diff gate because the builder environment
  ships `pip`, which is not in the lock. `lock_diff` now takes the rule and reports
  rule-denied packages as ignored — they are pruned from the payload by design.
- `--find-links` alone did not keep the install local: pip still fetched from the
  index. `--no-index` makes the wheelhouse the only source, so a missing wheel fails
  loudly instead of quietly reaching the network.
- The requirement set covered only dependencies, so the package being packaged was
  never installed and `resolve_version` had nothing to read:
  `version: auto but no dist-info for vllm`. `requirements_for` now includes the rule's
  own package, and refuses a rule that denies its own name. Measured inside the
  container that both resolve to binary wheels for the target:
  `vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl`,
  `vllm_omni-0.32.0-py3-none-any.whl`.
- `pip download` resolves the whole transitive graph, so a rule-denied name re-enters
  the lock through a dependency: `setuptools` is denied by the rule yet was downloaded
  as a dependency of `model-hosting-container-standards`. New gate
  `denied_dependencies()` reports a denied name required at runtime by a shipped
  package — unmarked requirements only, since extras are not installed and an
  environment marker cannot be evaluated without the `packaging` library.
- `denied_dependencies()` is the weak half of the gate: a declared requirement is
  not an import, so it reports and does not fail. `importscan.runtime_conflicts`
  is the half that can fail, by reading the shipped files. `package_thin.build()`
  now prints the declared half and fails on the hard half (a shipped file that
  imports a denied package at module scope); imports inside a function, `try`, or
  `if` are not executed at import time, so a denied name reached only through
  those stays optional as the rule intends.
- Ownership is read from each distribution's `RECORD`, not guessed from a package
  name: `flashinfer-python` owns `flashinfer/`, `opencv-python-headless` owns
  `cv2/`. A distribution is not a conflict with itself, and a file's owner comes
  from the same map — that was the bug in the first version of the gate.
- `build.sh` lists the context with `git ls-files --cached --others
  --exclude-standard`. Tracked files alone silently drop a new module from the
  image build, which is how the first build failed.
- The deny list contradicts the clone's own requirements. Measured by module-level
  import scan of the clone: `huggingface_hub` 12 (including `engine/arg_utils.py`,
  the serve path), `torchvision` 10, `flashinfer` 7, `torchaudio` 3, and
  `ninja`, `sentencepiece`, `setuptools`,
  `model-hosting-container-standards` 0. So `huggingface_hub` must ship, the three
  feature modules need their package denied-or-shipped decided, and the
  zero-import names can stay denied.
- Name-only requirements record what pip resolved, not what the clone pins: the
  clone declares `torch==2.13.0`, `torchaudio==2.11.0`, `torchvision==0.28.0`,
  `flashinfer-python==0.7.0.post1`, while the wheelhouse held `torch 2.10.0`,
  `torchaudio 2.10.0`, `torchvision 0.25.0`, `flashinfer_python 0.6.6`. The pins
  are dropped by `read_requirements()`; whether the resolved versions are
  acceptable is not yet measured.
- First full build reached the uncovered gate and failed with 14 entries:
  `numba/np/ufunc/omppool...so: libgomp.so.1.0.0`,
  `numba/np/ufunc/tbbpool...so: libtbb.so.12`,
  `nvidia/cufile/lib/libcufile_rdma.so.1: libmlx5.so.1 / librdmacm.so.1 /
  libibverbs.so.1`, and the nvshmem plugin/bootloader set
  (`nvshmem_bootstrap_mpi.so.3: libmpi.so.40`, `..._pmix.so.3: libpmix.so.2`,
  `..._shmem.so.3: liboshmem.so.40`, `..._ibdevx/ibgda.so.3: libmlx5.so.1`,
  `..._libfabric.so.3: libfabric.so.1`, `..._ucx.so.3: libucs.so.0 / libucp.so.0`).
  These are optional runtime plugins, so the choice is to prune them or to declare
  their libraries as system libraries. `libgomp` is installed by `runtime_apt` but
  is not in the `SYSTEM_LIBS` glob, and `nvidia/cufile/**` is not in the prune list.
- `strip: tool unavailable, skipping` in the pkg stage: the builder base has no
  binutils, so the strip step in the pipeline does nothing there. `strip` exists in
  the tools image only. Decide whether to add binutils to the pkg stage or drop
  strip from the pipeline.
- Anomaly to measure: the declared half printed nothing on the first full build,
  even though shipped packages declare denied names. No claim until a run shows why.
  Part of it is now explained by ordering: `build()` fails on the hard half before it
  prints the declared half, so a failing hard half never prints the declared one.
- Second build (scan reads the selection): the hard gate fired with **957 findings**,
  counted from the log by denied name —
  `huggingface-hub` 769, `torchvision` 157, `setuptools` 14,
  `flashinfer-python` 8, `sentencepiece` 5, `torchaudio` 4;
  by owning package — `transformers` 906, `vllm` 34, `numba` 6,
  `flashinfer` 5, `torch` 2, `llguidance` 1, `compressed_tensors` 1,
  top-level `build_backend.py` 1.
  The deny list contradicts the shipped tree at scale: `transformers` imports
  `huggingface_hub` at module level in 769 files and `torchvision` in 157, so both
  must ship. The earlier clone-only scan (12 / 10) counted only vLLM's own files and
  understated this by an order of magnitude.
  `setuptools` (14) is reached only through build-time paths —
  `torch/utils/cpp_extension.py`, `numba/pycc/**`, `flashinfer/data/**` setup
  scripts, `build_backend.py` — so those files are prune candidates rather than a
  reason to ship setuptools.

### Evaluation (the measurement phase)

Run inside the container: `bash docker/run.sh script tools/evaluate_deny.py
--wheelhouse out/wheelhouse --venv out/eval-venv --rule vllm --view out/eval-view-5`.
The view must be a path that does not exist yet — `out/eval-view*` already held
views from earlier rounds, and a reused view reports a package as reachable
because an earlier round put it back.

What the runs measured:

- Target GPU, read through the harness: `NVIDIA GeForce RTX 3060`, compute cap
  8.6, so the arch is `sm_86`. The tools container has no GPU, so `arch.detect()`
  there always returns the fallback full list; the `gpu` task in `docker/run.sh`
  is what makes detection possible.
- Selection: **51,514 files / 11,021.6 MB before decisions → 44,441 files /
  9,986.6 MB after**. Wheelhouse 5,364.3 MB / 189 wheels = 48.7% of unpacked.
  Top 18 files are 62.9% of the selection; the numbers are in `docs/SIZES.md`.
- `flashinfer-cubin`: 1,114.8 MB unpacked / 267.7 MB compressed, 9,762 files, no
  module-level importer, reached only through `importlib.util.find_spec`. Arch
  split measured by hand (the cubin files are hash-named, so `arch_of()` cannot
  see them): sm_100 937.0 MB, sm_103 127.3 MB, unmarked 50.5 MB — **zero sm_86
  kernels**, so it is dead weight for this GPU and is now denied.
- flashinfer itself cannot be pruned: imported at module level by 8 core files,
  and its own tree has sm_80 0.9 MB but no sm_86 kernels plus 24.0 MB of JIT
  source. On sm_86 it needs nvcc, which R01 forbids — so the wheel ships and the
  JIT path is a runtime dependency, not a build-time one.
- `.pyc` 258.0 MB vs `.py` 230.3 MB inside the selection: `bytecode: pyc-only`
  costs **+27.7 MB**, kept for startup latency rather than for size.
- JIT source inside the selection: 104.8 MB (`.cpp` 87.6, `.cc` 14.3). Unusable at
  runtime under R01, so it is a candidate prune — not yet applied.
- Dead prune patterns, checked against the **full tree** (72,495 files):
  `nvidia/cusparseLt/**` and `nvidia/cupti/**` never fired. The real folders are
  `nvidia/cusparselt` (452.0 MB) and `nvidia/cuda_cupti` (41.3 MB), and both are
  `DT_NEEDED` by libtorch, so they cannot be pruned at all. Patterns removed.
- Corrected libtorch win: **~692.8 MB** (cusparselt 452.0 + nvshmem 196.2 +
  cupti 41.3 + cufile 3.3), not the earlier 100–250 MB — and only for a source
  build, since the pip wheel ships those features enabled.
- Entry-path names that must ship: `huggingface-hub` 6.8 MB and `setuptools`
  7.5 MB. The declared half can execute — `transformers` requires
  `huggingface-hub>=1.31.0,<3.0` and the entry path reads it through
  `importlib.metadata`, so a denied name is not automatically optional.
- Decisions applied to `packaging/rules/vllm.json`: denied `torchcodec`,
  `pynvvideocodec`, `flashinfer-cubin`; shipped `torchvision`, `torchaudio`,
  `sentencepiece`, `setuptools`, `huggingface-hub`, `flashinfer-python`; prune
  `**/include/**`, `**/*.a`, `**/tests/**`, `**/*.pyi`, `**/*.pyx`,
  `**/examples/**`, `nvidia/*/bin/**`, `nvidia/nvvm/**`; removed
  `**/benchmarks/**` (the entry path reaches `vllm.benchmarks` through the CLI).
- binutils added to the pkg stage of `docker/Dockerfile.vllm`, so `strip` is a
  real step instead of printing "strip: tool unavailable, skipping". binutils
  stays in stage 1; the runtime image is checked for `strip` by the R01 gate.

Runtime library set, measured on the GPU (`tools/measure_runtime_libs.py`, which reads
`/proc/self/maps` after a workload runs):

- Workload `blas, conv, triton, sparse, distributed` on `sm_86`: **38 files mapped,
  5,798.9 MB** of the 9,677.0 MB of libraries in the tree. Per tree: nvidia 25 files
  (3,807.4 MB), torch 8 (1,537.2 MB), triton 1 (415.6 MB), numpy 2 + numpy.libs 3.
- The `DT_NEEDED` closure alone (20 libs / 2,904.2 MB) badly **underestimates** what
  the process loads: `libcudnn_graph`, `libcudnn_heuristic`,
  `libcudnn_engines_precompiled`, `libcudnn_engines_runtime_compiled` and
  `libtriton.so` are reached through `dlopen`/`ctypes` from Python and appear in no
  ELF's `DT_NEEDED`. A "nothing links it" list is a candidate list, not a verdict.
- `libcudnn_adv` (285.2 MB) and `libcudnn_cnn` (13.1 MB) were **not** mapped by this
  conv; a vision encoder or a different conv shape can still reach them, so the cudnn
  tree is not prunable without a model-specific run.
- `libcusolverMg.so.11` (161.5 MB), `libnvrtc.alt` (105.0 MB), `libnvperf_host` (51.7
  MB), `libtilelang.so` (28.6 MB), `libcute_dsl_runtime.so` (87.5 MB) and
  `libtriton_amd_codegen.so` (96.0 MB) were not mapped. Only
  `libtriton_amd_codegen.so` is dead on NVIDIA by design (AMD codegen); the others are
  reached through guarded Python `ctypes`/`find_spec` paths.
- Runtime import closure from the entry path: **90 packages / 4,161.8 MB** — the set of
  distributions a shipped file imports at module level.

Triton and the no-compiler rule (R01), measured:

- `triton/runtime/build.py` looks for `clang`/`gcc` (or `CC`) on first use and raises
  `RuntimeError: Failed to find C compiler` when neither is present. The tools
  container has no compiler, so the first triton kernel launch failed there.
- `compile_module_from_src` checks the cache **before** the compiler, so a warm cache
  is enough: the same kernel ran in a compiler-free container once the cache was
  warmed by a run that had `gcc` + `libc6-dev` (gcc alone is not enough — `cuda.h`
  includes `<stdlib.h>` and the compile failed on it).
- A kernel the cache had never seen still failed in the compiler-free container. So
  the runtime image can only run kernels that were compiled in the build stage: the
  build stage must pre-compile the kernels the target model uses, which is the same
  conclusion as the vendored-cubin approach in the reference clone.
- Cache key is `sha256(src + platform_key())` where `platform_key()` is
  `machine, system, architecture` — no GPU arch in it, so a cache built in stage 1 is
  valid in stage 3 as long as both are x86_64 Linux with the same Python. The kernel
  artifacts themselves are arch-specific.
- Minimum warm cache for this test: 10 files, 0.07 MB (two C extensions plus one
  kernel). A real serving run compiles many kernels.

Two false readings this phase, both corrected by re-running:

- `rules_mod.matches(rel, pattern)` called with a bare string iterates
  characters, so every pattern matched all 51,514 files and the dead-pattern
  conclusion was wrong. Passing `[pat]` gave the correct split.
- Checking prune patterns against the selection is circular (`select()` already
  excludes pruned files, so every pattern trivially matches nothing). The check
  must run against the full tree.

Runs after the decisions: entry import completes in round 1 (gate clean),
`test` 88 passed, `gate` clean on the 6 changed files.

Other defects found and fixed by running:

- pip spells the target flags with hyphens; `--python_version` is not an option.
- `HEALTHCHECK` requires `CMD`; the Dockerfile failed to parse without it.
- git inside the bind mount refuses the index as "dubious ownership" — configured
  through `GIT_CONFIG_COUNT`/`safe.directory` in `docker/run.sh`, not by writing
  config into the workspace.
- The docker CLI resolves paths in its own filesystem. Running inside the tools
  container means the build context path is `/workspace/out/context`; the daemon
  receives a tar, never a host path.
- `lock.py` needed the clone, which is never in the build context. Rules now carry a
  `requirements.names` snapshot generated from the clone, and the build uses it.

Bugs the container run exposed:

- `check_files` reported an extra final newline but did not fix it.
- The trailing-whitespace fix duplicated the newline (`rstrip(" \t")` leaves `\n`
  in place, so adding one produced a blank line).
- `main()` read `sys.argv` directly, so it could not be called from a test.
- `gates.sh` had no bucket for files with no extension, so `.dockerignore` was
  never checked. Added a catch-all bucket.
- A shellcheck directive inside a `\` continuation is a parse error; it must come
  before the command.

| Image | Size (on disk) | Compressed download | Cold start | tokens/s | Peak GPU MiB |
| --- | --- | --- | --- | --- | --- |
| `vllm/vllm-openai:latest` (baseline) | — | — | — | — | — |
| `vllm-slim` (engine variant) | — | — | — | — | — |
| `vllm-slim` (thin variant) | — | — | — | — | — |
| `vllm-slim` (.pyc-only ship) | — | — | — | — | — |
| `vllm-omni` overlay | — | — | — | — | — |

## Decisions

- Strata is reference only; behaviour reimplemented in `packaging/`. No Strata path in any image.
- Read-only clones; build-context copies allowed (`cp -al`, no second copy of GBs).
- Locks and packages generated at build time into `out/`; rules files committed.
- Single CUDA arch detected by command; `VDE_CUDA_ARCH` overrides.
- `USE_DISTRIBUTED` stays ON (see `Plan.md` finding 2). Only MPI/UCC/TensorPipe are droppable.
- No compiler, no pip, no uv, no git, no downloader in the runtime image (gate `R01`).
- `libcuda.so.1` and `libnvidia-*` are never bundled — the host driver provides them.
- Nothing executes on the host. `docker/run.sh` is the only host-side entry point;
  it issues `docker build` / `docker run` and every gate, test and measurement runs
  inside the pinned tools container (`docker/Dockerfile.tools`). Content moves
  between images, code does not run on the host.

## Dead ends

- libtorch-only: rejected (autograd/Dynamo/distributed are hard deps).
- vllm.cpp as the engine: rejected (see `Plan.md` Phase 5).
- `USE_DISTRIBUTED=0`: rejected — breaks `import vllm` even at `tp_size=1`.
- Alpine: rejected (musl, no manylinux wheels).
- Op-level selective torch build: ExecuTorch feature, not libtorch.

## Next action

1. `bash docker/run.sh build --variant engine` with the decisions applied — the
   selection is now 44,441 files / 9,986.6 MB, so the build has to be re-run to
   fill the size table in `docs/SIZES.md`.
2. Baseline row: `bash docker/run.sh measure vllm/vllm-openai:latest`.
3. Decide the JIT source (104.8 MB): R01 forbids a compiler in the runtime image,
   so the `.cpp`/`.cc` trees in the selection cannot be used there. Pruning them
   is only safe if nothing on the entry path compiles at import time — flashinfer
   is the case to check on the GPU, because on sm_86 it has no prebuilt kernels.
4. Uncovered gate: **0 findings** after the decisions (was 14, then 13, then 3). The
   `libgomp.so.1` glob was widened to `libgomp.so.*` because apt installs
   `libgomp.so.1.0.0` plus a symlink, and foreign-arch ELF files are now refused by
   `select()` (the build run failed at `strip` on cutlass-dsl aarch64 stubs).
5. Decide R01 vs triton (mechanism measured, decision recorded as `Plan.md` finding 5):
   either the build stage warms and ships the triton cache for the target model and
   arch, or R01 is relaxed for the runtime image. The choice changes what the build
   stage must run and what the package must carry, so it is decided before the build
   is re-run.
6. Spike A (control) vs Spike B (pruned): `bash docker/run.sh spike`, then
   `bash docker/run.sh script tools/compare_packaging.py`.
