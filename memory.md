# Progress log

Continuity mechanism. Update it as work happens: measurements, decisions, dead ends, blockers,
next action. Every number here must come from a run in this repo, not from an estimate.

## Status

| Phase | State | Notes |
| --- | --- | --- |
| 0 — files & plan | done | `Plan.md`, `memory.md`, `wiki.md`, `docs/`, agents |
| 1 — package research | complete (research), pending measurement | findings folded into `Plan.md` |
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

1. `bash docker/run.sh build` — first real image build, then the size and runtime gates.
2. Baseline row: `bash docker/run.sh measure vllm/vllm-openai:latest`.
3. Spike A (control) vs Spike B (pruned): `bash docker/run.sh spike`, then
   `bash docker/run.sh script tools/compare_packaging.py`.
4. Write `docs/PACKAGING.md`, `docs/DOCKER.md`, `docs/SIZES.md` and the agents.
