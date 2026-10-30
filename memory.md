# Progress log

Continuity mechanism. Update it as work happens: measurements, decisions, dead ends, blockers,
next action. Every number here must come from a run in this repo, not from an estimate.

## Status

| Phase | State | Notes |
| --- | --- | --- |
| 0 — files & plan | done | `Plan.md`, `memory.md`, `wiki.md`, `docs/`, agents |
| 1 — package research | complete (research + measurement) | findings folded into `Plan.md`, numbers in `docs/SIZES.md` |
| 2 — docker build spikes | ready, not started | the three rules are implemented, tested and documented, so the next step is the first build. Needs a runnable base image: `nvidia/cuda:12.8.1-devel-ubuntu24.04` is already local, and `bash docker/run.sh pull <image>` now reports a 401 instead of retrying. Docker Hub pulls previously failed with 401; local `vllm/vllm-openai:latest` works. Both import gates now report 0 findings on the real install after the cupy/ray decision (finding 10 in `Plan.md`) |
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
| `gate` on `docs/findings.md` `docs/SIZES.md` `docs/DOCKER.md` `README.md` `Plan.md` `memory.md` `wiki.md` `.github/instructions/packaging.instructions.md` `tools/check_loader_paths.py` | first run FAILED: 8 MD013 line-length issues (5 in `docs/findings.md` table rows, 1 in `docs/DOCKER.md`, 1 in `docs/SIZES.md`, plus the same file re-checked after rewrap). After rewrapping: clean. markdownlint skipped `Plan.md`, `memory.md`, `wiki.md` and the instructions file as outside the gated pattern |
| `gate` on `README.md` `docs/findings.md` `docs/PACKAGING.md` `docs/DOCKER.md` `docs/SIZES.md` after the CUDA-by-command docs pass | first run FAILED: 3 MD013 issues (2 in `docs/DOCKER.md`, 1 in `docs/PACKAGING.md`). After rewrapping prose at word boundaries: clean, 0 issues in 5 files |
| MD013 behaviour, measured with a probe file | prose lines and table rows are checked, fenced code blocks are not: a prose line of 82 and a table row of 82 were reported, an 85-character line inside a fence was not. Command examples in fenced blocks may therefore exceed 80 |
| the CUDA docs pass, after an auto-reflow script damaged it | the four `docs/` files were reverted to HEAD by the user; the CUDA content was re-added by hand (prose wrapped at ≤80) and `README.md` was rewritten without its duplicated paragraphs. Gate re-run on the 5 files explicitly: clean, 0 issues. Lesson: never reflow markdown with a script — wrap prose with explicit edits, one paragraph at a time |
| `test` re-run after the loader docs pass | 92 passed, exit 0 |
| module-level import scan of the clone | `huggingface_hub` 12, `torchvision` 10, `flashinfer` 7, `torchaudio` 3, `ninja` 0, `sentencepiece` 0, `setuptools` 0, `model-hosting-container-standards` 0 |
| `build --variant engine` | failed at stage 1: `ModuleNotFoundError: No module named 'importscan'` — the context is built from `git ls-files`, which lists tracked files only, so the new `packaging/importscan.py` was not in the context |
| `gate` on `packaging/cuda.py` `packaging/rules.py` `packaging/package.py` `packaging/package_thin.py` `packaging/rules/vllm.json` `docker/Dockerfile.vllm` `docker/build.sh` `docker/run.sh` `tests/test_cuda.py` | first run FAILED, not on a lint issue: `bash docker/run.sh` sources `out/cuda.env`, and the unquoted line `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a command (`nvcc fatal : Don't know what to do with 'PATH'`). After quoting every value in `write_env`: clean, 3 files reformatted |
| `test` (114 tests) | exit 0, all passed, run twice (before and after the gate reformatted files) |
| `cuda --verify` measured run on the GPU | driver 580.95.05, driver CUDA 13.0, no nvcc in the tools container, wheelhouse series 12 (16 of 17 wheels), tree series 13, declared series 12, surviving copy 13, builder `nvidia/cuda:12.8.1-devel-ubuntu24.04` verified by `docker manifest inspect`, runtime base `ubuntu:24.04` |
| both branches of the duplicate-copy prune, run against the real tree | keep cu12 → 48,672 files / 10,736.2 MB (cutlass 462.6 MB, 43.7 MB pruned); keep cu13 → 48,672 files / 10,736.3 MB (cutlass 462.7 MB, 43.6 MB pruned); no series → 48,673 files / 10,779.9 MB (nothing pruned). The two branches are the same size, so the decision is which copy the loader resolves, not which is smaller |
| user-set CUDA against the driver ceiling, asserted by `tests/test_cuda.py` | a user value above the ceiling raises an error naming the driver as the cause and says to use another CUDA version or update it; a user value equal to the ceiling is accepted; a user value the wheelhouse disagrees with is accepted and records that the wheelhouse copy must be fetched; a user value below the ceiling needs no wheelhouse warning |
| 401 reporting, measured in the container | a fabricated 401 detail prints the source, the sentence asking the user to check availability or download directly, and the detail itself; `check_image` on an image that is already local returns without pulling; `nvidia/cuda:12.4.1-devel-ubuntu24.04` prints the registry failure without the 401 sentence and exits 1. A real network 401 has not been observed since the rule was written |
| `test` (127 tests) | exit 0, all passed. Six failures were fixed first: the driver-error wording needed the "would not start on this machine" clause, `check_image_size` needed the 6 GB gate threshold with 4 GB as the stretch goal, and the report/shell-copy parity test reads `docker/pull.sh` from the repo root |
| `gate` on the 19 files changed for the three new rules | clean: ruff check and format (6 python files unchanged), `check_files`, shellcheck, `bash -n`, typos, and markdownlint 0 issues in the 5 gated files. `wiki.md`, `memory.md`, `Plan.md` and the `.github/` files are outside the markdownlint pattern and were skipped. Re-gated after wrapping the added prose at 80 columns: clean, and `test` re-run at 127 passed |
| `test` (131 tests) | exit 0, all passed. Four new tests cover the cupy/ray decision: the pruned-file import gate, the `keep` exception, a deny enforced through the `RECORD` owner map, and `select()` called with that map |
| `gate` on the 7 files changed for the cupy/ray decision (`packaging/importscan.py`, `packaging/rules.py`, `packaging/package.py`, `packaging/package_thin.py`, `packaging/rules/vllm.json`, `tools/evaluate_deny.py`, `tests/test_packaging.py`) | clean. Ruff reformatted 2 of them; re-run needed before the docs gate rows below |
| import gate on the real install, before the decision | 10 findings from 9 importers (132.5 KB) pointing at 7 targets (76.3 KB); deny gate 1 finding (`ray/util/collective/collective_group/cuda_stream.py`, 2,975 bytes, imports the denied `cupy-cuda12x`) |
| import gate on the real install, after the `keep` decision | 0 findings on both halves, selection 47,921 files / 10,505.7 MB (47,917 before, +7 keep −3 ray prune entries) |
| cupy / ray runtime branches on the GPU (warm cache, same model and seed) | cupy pruned + importer pruned → OK 41.6 s; cupy pruned + ray kept whole → OK 41.8 s; ray and cupy pruned → OK 40.7 s. The cupy import is never reached, so the deny conflict is static |
| importing the 9 pruned-file importers in the materialised view | all 9 fail on other missing names (`pyperf` ×2, `numba.tests` ×3, `hypothesis`, `numpy._core.tests`, `torch._export.db.examples`, `expecttest`) — neither side of the import is reached at runtime |
| cupy against the lock, checked after the decision | `out/locks/vllm-linux-x86_64.txt` contains `ray==2.59.0` and `cupy-cuda12x==14.2.0`. vllm's `common.txt` and `cuda.txt` name neither, so both come from the `ray[cgraph]` extra in the wheel metadata: the thin variant installs cupy, which is why the deny gate fires on the real install. Denying cupy is a policy choice, not a claim that the lock resolves without it |
| `compare_packaging.py` re-run after the owner-map fix (`out/compare-cupy2.txt`) | all four combinations clean, no build failure: engine/source 47,928 files / 10,102.7 MB, thin/source 45,829 / 10,020.8 MB (archive 5,155.1 MB), engine/pyc-only 26,867 / 9,798.0 MB, thin/pyc-only 25,803 / 9,755.0 MB (archive 5,099.9 MB). Manifest True, gates clean, usability ok on all four |
| the `bytecode` modes, measured on the builds | the tree carries 24,079 `.pyc` (394.9 MB) before any build; the selection keeps 21,064 (338.2 MB). `source` compiles the `.py` it keeps and ships both forms (21,061 `.py` = 304.7 MB + 338.2 MB bytecode); `pyc-only` removes the `.py` mirrors and ships bytecode only. So `pyc-only` is the smaller build: engine 10,102.7 → 9,798.0 MB, thin 10,020.8 → 9,755.0 MB. The earlier "pyc-only costs more" reading was a selection comparison (258.0 vs 230.3 MB), not a build measurement — corrected in `docs/SIZES.md` |
| `gate` on `docs/SIZES.md` after the build numbers were added | clean, 0 issues |
| `gate` on `docs/PACKAGING.md` `docs/SIZES.md` after the bytecode correction | first run FAILED on one 81-character prose line in `docs/PACKAGING.md` (MD013). After rewrapping: clean, 0 issues in 2 files |
| pyc validity as delivered (`out/compare-cupy2/source/engine`) | `{'timestamp/valid': 21054, 'checked-hash/valid': 7, 'timestamp/sourceless': 3}`. The 7 hash pycs are exactly the 7 `keep` targets; the 3 sourceless are the 3 ray files the cupy/ray decision pruned |
| the same cache after the archive's forced mtime | `{'timestamp/invalid': 21054, 'checked-hash/valid': 7, 'timestamp/sourceless': 3}` — `write_archive()` sets `mtime = SOURCE_DATE_EPOCH` on every member, so every timestamp pyc it carries is invalidated. Hash pycs are validated by hashing the source (blake2b, 8 bytes) and survive |
| `docker COPY` preserves mtimes | probe: source mtime 1600000000, pyc field 1 1600000000, valid True. So the runtime image keeps whatever the verify stage produced |
| can the runtime user write `__pycache__` | no: uid 1000 on a root-owned tree with members at `0o555` imports fine and writes nothing. The user's "cache is built at launch" hypothesis does not hold for this image |
| startup cost on a read-only tree | fresh cache 5.4 s, invalid cache 6.9 s (both rc 0), 1,665 modules executed by the entry import → about +1.5 s per start |
| forced checked-hash cache | 21,061 valid after the forced mtime, 337.0 MB, 70.1 s compile. `compileall` ignores the string form of `invalidation_mode` and writes flags 1 (unchecked-hash); the enum `py_compile.PycInvalidationMode.CHECKED_HASH` is required (flags 3) |
| sourceless layouts | a pyc in `__pycache__` with the source deleted is not importable at all (both timestamp and hash pycs). The legacy layout — pyc at the module's own location — imports fine |
| the delivered pyc-only build | 0 `.py`, 21,064 `.pyc`; `import vllm` yields `vllm.__file__ is None` (namespace package, no code executed). Mirroring pycs to the module location makes the tree importable, and then `torch/fx/experimental/_config.py` raises `OSError: could not get source code` from `inspect.getsource()` inside `install_config_module()` |
| source-read users in the selection | 68 files use `inspect.getsource`/`getsourcelines`/`getsourcefile`; 1 runs at module level (`torch/_dynamo/eval_frame.py`). A static scan cannot find calls made from a function body at import time, which is the torch config case |
| the size side of the bytecode decision | source 10,102.7 MB unpacked / 5,155.1 MB archive; pyc-only 9,798.0 / 5,099.9. The download difference is 55.2 MB, below the "a few MB, ship it" threshold → source is kept |
| `gate` on `tools/evaluate_bytecode.py` `tools/compare_packaging.py` | first run FAILED on F841 (unused `views` in the entry-time block). After removing it: clean, 2 files |
| `compileall` and a cache the source tree already carries | a pre-existing timestamp pyc is NOT rewritten by `CHECKED_HASH` — flags stayed 0 on a small probe; a fresh build writes flags 3. So `bytecode_tree()` now drops `__pycache__` pycs before compiling whenever the invalidation is hash-based, otherwise a build ships a mix of both kinds |
| enum values vs header flags | `PycInvalidationMode`: TIMESTAMP=1, CHECKED_HASH=2, UNCHECKED_HASH=3. The header stores mode bits, not the value — hash modes write `value | 1`, so checked-hash reads 3 and timestamp reads 0. A test that compares the header to `.value` fails; compare the bits |
| tests after the invalidation change | 132 passed (131 + the new `test_hash_invalidation_rewrites_a_cache_the_source_tree_carries`); `gate` on the 5 changed files clean |
| `compare_packaging.py` re-run with the corrected import check (`--modes pyc-only`, `out/compare-bc.txt`) | both variants report **"namespace package: no code ran"** — the earlier "usability ok" was a false pass. Sizes unchanged: engine 26,867 files / 9,798.0 MB, thin 25,803 / 9,755.0 MB (archive 5,099.9 MB), gates clean on both |
| `compare_packaging.py` source modes after the invalidation change (`out/compare-bc2`) | engine 47,925 files / 10,101.4 MB, thin 45,826 / 10,020.7 MB (archive 5,155.1 MB). Three fewer files than before on each variant: the 3 sourceless ray pycs are now dropped instead of shipped. Gates clean, import check ok on both |
| the delivered checked-hash build as it is read at runtime | all 21,061 pycs `checked-hash/valid`, entry import as uid 1000 in 2.6 s, 1,665 modules executed. Repeated 3 times: median 2.75 s |
| the same tool on the timestamp build, same tree before and after forcing the mtime | valid 4.9 s, invalid 6.3 s (+1.4 s); 3 runs on the invalid tree → median 6.19 s. The +1.5 s penalty stands. The gap between 6.19 s and the 2.75 s hash build is not explained by validation cost (hash vs stat) and is not claimed as a reason to prefer hash pycs |
| pyc-only build through the same tool | `namespace package: no code ran`, 0.019 s — an import that proves nothing. `timed_import` now checks `vllm.__file__` instead of only the exit code |
| new tool `tools/measure_startup.py` | prints pyc validity as delivered, then the entry import as uid 1000 and the module count; reuses `tools/evaluate_bytecode.py`. Gate clean on it |
| `.pyi` stubs as import targets, evaluated on both sides | counting stubs gave 324 findings on the real install (`torch/_C/*.pyi`, `triton/_C/libtriton/*.pyi`, `tvm_ffi/core.pyi`); the importers tested in the materialised view (`triton.tools`, `torch.profiler`, `torch.utils.data`, `tvm_ffi.dataclasses`, `vllm.compilation.wrapper`) all import fine with the stub folders gone, because the runtime resolves through the compiled `.so`. The change was reverted and the gate re-run: 0 findings |
| `gate` on `packaging/importscan.py` `tests/test_packaging.py` `docs/PACKAGING.md` `memory.md` | first run FAILED on one 82-character prose line in `docs/PACKAGING.md` (MD013). After rewrapping: clean, 0 issues. `test` on `tests/test_packaging.py`: 106 passed (105 + the new `test_stub_files_are_not_import_targets`) |

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

**Build #3 — CUDA decided by command, first end-to-end three-stage run.**

- `bash docker/run.sh cuda` writes `out/cuda.env`. Measured on this host: driver
  580.95.05, driver CUDA ceiling 13.0, no `nvcc` on PATH (the tools image has no
  CUDA toolkit), wheelhouse series 12, tree series 13 — the cutlass tree holds both
  `cu12/` and `cu13/`. No user override, so the wheelhouse wins over the tree:
  `VDE_CUDA_SERIES=12` (declared), `VDE_CUDA_KEEP=13` (surviving), builder
  `nvidia/cuda:12.8.1-devel-ubuntu24.04`, runtime base
  `nvidia/cuda:12.8.1-base-ubuntu24.04`.
- Build #1 failed at the pkg stage with `/bin/sh: 1: python3: not found` (exit 127).
  Measured: `nvidia/cuda:12.8.1-devel-ubuntu24.04` has **no python3 and no pip3**.
  Fix: the pkg stage base is `VDE_PKG_BASE` (default `python:3.12-slim`); the CUDA
  devel image is the toolchain image for spike/compile tasks, not the packaging stage.
- Loader path implemented in the runtime stage (option 2, `ld.so.conf.d` + `ldconfig`)
  after confirming `/usr/bin/find` and `/usr/sbin/ldconfig` exist in `ubuntu:24.04`.
  `LD_LIBRARY_PATH` is deliberately not set, so the config file is the only mechanism.
- Build #2 failed at the pkg stage with `unrecognized arguments: --cuda-keep 13`.
  Fix: `cuda.resolve_keep()` plus `--cuda-keep` and `--bytecode` on both packaging
  CLIs, and `cuda_keep` threaded through `select()`, `write_manifest()` and `build()`.
  `--cuda-keep auto` resolves from the tree, so the build stage and the packaging CLI
  agree without a hand-written value.
- Build #3 (`bash docker/run.sh build --variant thin`, log `out/build-run.log`):
  lock 195 entries, install ok, `launchers dropped: 0`,
  `cuda: declared cu12, tree resolves cu13` — the two-series split is live in the
  build, and the prune follows the surviving series.
  The pkg stage ran clean end to end: `files: 47925 bytes: 10058080767`,
  archive 5160192182 bytes, `gate: no uncovered imports`, ninja-only notes.
- Build #3 then failed in the **verify** stage: `FileNotFoundError: /opt/vllm/manifest.json`.
  Cause: `--out` is the parent directory, and the package lands in a subdirectory
  named `<version>-<platform>-<backend>-<archs>`, so `COPY --from=pkg /pkg/stage`
  put the parent in the image and the manifest was one level deeper.
  Fix: `package_dir(..., name)` + `--name` on both CLIs, and the Dockerfiles pin
  `--name package` with `COPY --from=pkg /pkg/stage/package`. A COPY instruction
  cannot name a path derived from the version and the detected arch list, and the
  manifest already records version, platform, backend and archs separately, so
  pinning the folder loses nothing. Same fix applied to `Dockerfile.omni`.
  Gate clean on the five changed files; tests 136 passed.

**Build #4 — the check gate ran in a stage that does not hold its inputs.**

- Failed in the verify stage: `FileNotFoundError: /pkg/locks/vllm-linux-x86_64.txt`.
  The verify stage is a fresh container that has the package but not the lock or the
  wheelhouse, so a check there reads paths that do not exist.
- Fix: the check gate moved to the **pkg stage**, after `--download` and before the
  install — the stage that holds both artifacts. Copying the 5.3 GB wheelhouse into
  the verify stage would cost build time for a gate unrelated to the package. In this
  path the hash half is a re-read; the half that still matters is the tag gate, and
  both halves are real for a cached lock.
  Gate clean on `docker/Dockerfile.vllm`.

**Build #5 — the tag gate was wrong, not the data.**

- The gate failed with 19 python-tag findings on a wheelhouse that is legitimately
  for this target. `python_tags()` returned only `cp312`, `py312`, `py3`, so every
  `cpXY-abi3` wheel was rejected — 19 of 195 wheels, including vllm itself,
  tokenizers, safetensors, protobuf, psutil, opencv, tilelang, hf_xet, llguidance,
  watchfiles, cryptography, humming_kernels, openai_harmony, oss_harmony.
  `linux-x86_64` is a platform, not a python version, and the gate was comparing a
  name rather than reading the claim.
- Fix: `python_ok()` and `abi_ok()` read the fields as claims about versions.
  `cpXY-abi3` is usable on 3.12 when XY <= 3.12; `py37-none-any` is usable because
  `py37` claims 3.7 upward; `py2-none-any` is not, because `py2` claims python 2
  only; `cp313-abi3` is not, because a stable-ABI wheel built for a newer
  interpreter cannot load on 3.12; a concrete `cp311-cp311` is still rejected.
  `python_tags()` is gone (it had no other caller).
- Re-run on the real data: `lock check: clean (out/locks/vllm-linux-x86_64.txt)`,
  with the unchanged `requirements: 3 denied name(s) not downloaded` note.
  Gates clean on `packaging/lock.py` and `tests/test_packaging.py`; tests **137 passed**
  (new test `test_tag_rules_read_python_and_abi_as_claims`).

**Build #6 — the first build that reached the end of all three stages.**

- `bash docker/run.sh build --packaging thin` → `Successfully built 05198e62551e`,
  tagged `vllm-slim:dev`. Every stage ran, including the gates that stopped the
  previous four runs.
- pkg stage: `lock: /pkg/locks/vllm-linux-x86_64.txt (195 entries)`, check gate clean,
  install ok, `launchers dropped: 0`, `cuda: declared cu12, tree resolves cu13`,
  selection **47,925 files / 10,058,113,757 B**, archive **5,160,222,411 B**
  (51.3% of the tree), uncovered-import gate clean, deny gate 0 findings with the
  two known ninja notes.
- verify stage: `verify: clean (/opt/vllm)` and the import smoke printed
  `0.17.0`.
- Image: **9.45 GiB unpacked** (the size tool reports GiB; 9.45 GiB = 10.16 GB
  decimal, which is the pkg-stage tree 9.37 GiB plus the base layers), baseline
  21.80 GiB, **reduction 56.6%** → gate **FAIL** on both halves (needs < 6.00 GiB
  and >= 60%). Compressed download is UNVERIFIED: the size tool cannot read a local
  image's compressed layers without a registry push.
- Runtime gate (R01) on the image: **PASS** — no forbidden tool present, user
  `1000:1000`, tini entrypoint, healthcheck present.
- Two tool bugs found by running the gates on the real image, both fixed:
  `check_runtime_image.py` read a clean result as a docker failure, because the last
  `command -v` in the check chain exits 1 when the tool is absent — which is the pass
  condition — so the shell now ends with `true`; and `docker/run.sh` passed the image
  as a positional argument to gates that require `--image`, so the documented
  positional form is now translated to the flag (a call that already uses flags is
  passed through unchanged).
- ptxas is present in the image (`triton/backends/nvidia/bin/ptxas` 30.4 MB,
  `tokenspeed_triton/.../ptxas` 49.6 MB), so the JIT path is not broken by the prune:
  `nvidia/*/bin/**` matches nothing in this tree — the only `bin` files are under
  `torch/bin` and `*/backends/nvidia/bin` (8 files, 135.9 MB), and the pattern needs
  a segment between `nvidia` and `bin`. `ptxas-blackwell` (33.3 MB) is still shipped
  and is not needed for `sm_86`.

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

JIT runs on the GPU, measured (Phase 2, `tools/evaluate_jit.py`):

- Cache key for the target tree is `3.6.00355c3e3452`, and it is **the same with and
  without `triton/backends/amd`** — pruning compiler-side `.py` files does not change
  the key, so a cache built on the full tree is valid on the pruned tree.
- `get_cache_invalidating_env_vars()` returns `{}` with and without
  `TRITON_BACKENDS_IN_TREE=1`: env vars do not change the key.
- Warm cache, no compiler on PATH, serving run: **OK, 42.4 s, hits=114 misses=0
  puts=0**. Only `libtriton.so` (415.6 MB) and `libnvJitLink.so.12` (94.1 MB) load.
- Cold cache, no compiler: **FAILED** (`Failed to find C compiler`). With `gcc` alone:
  **FAILED on `stdlib.h`**. With `gcc + libc6-dev`: **OK, 97.0 s**, cache = 1033 files
  / 48.6 MB. Compiled path (`--compiled`), cold: **OK, 119.4 s**, cache = 1710 files /
  58.5 MB.
- `ptxas` is required **even on a full cache hit** (the backend hash runs
  `ptxas --version`): removing `bin/ptxas` failed with `Cannot find ptxas`.
  `ptxas-blackwell` (34.9 MB) is not required.
- `nvidia/cuda_nvrtc` (222.1 MB) is only used when a kernel is compiled: runs serve
  with it removed. `nvidia/nvjitlink` (94.1 MB) cannot be removed — `libtriton.so`
  needs it at load time. cupti (110.8 MB) is not used for serving.
- `tilelang` (176.3 MB) and `flashinfer` (53.2 MB) are not used by this model in eager
  mode; removing them entirely still serves. They are optional kernel backends, so the
  decision is prune-by-default / keep-as-flavor, not prune-always.
- Compiler cost measured in the container: `gcc + libc6-dev` = **111.6 MB installed**.
- Decision from the two branches: **ship a pre-warmed cache and prune the compile-side
  tree**. Warm cache (48.6 MB eager, 58.5 MB compiled) + 597.3 MB pruned beats a
  compiler (111.6 MB, plus nvrtc and cupti must stay) by about **660 MB**, and keeps
  R01 for the runtime image. `ptxas` stays: it is an assembler the cache key needs,
  not a C compiler.
- The compiled path also runs on the pruned view against the cache it produced:
  **OK, 53.2 s, hits=110**, no compiler, no nvrtc.

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

### The dynamic linker path, measured (`tools/check_loader_paths.py`)

Counts edges as *cache-or-env*: a `DT_NEEDED` name is covered if it is in the
ld.so cache, in `LD_LIBRARY_PATH`, in the tree by file name, system, or driver.
That is what a real loader does, so it is stricter than `uncovered()`.

- 305 cache-or-env edges, **14 missing**, 1,590 system, 7 driver.
- RUNPATH: 367 ELF files with a dynamic section, 65 with RUNPATH/RPATH, **54 use
  `$ORIGIN`**. `torch/lib/*.so` all carry `$ORIGIN/../../nvidia/*/lib`. `vllm/_C.abi3.so`
  has **no RUNPATH**.
- `ctypes.CDLL(vllm/_C.abi3.so)` before torch → fails on `libtorch.so`.
- `import vllm` on a plain `ubuntu:24.04` → OK (0.17.0), `vllm._C` loaded, because
  torch is imported first and its RUNPATH covers the wheel folders.
- `ctypes.CDLL` after torch → fails only on `libcuda.so.1` (the driver, never bundled).
- `ldconfig` with a `ld.so.conf.d/vde-tree.conf` for the 41 lib/`.libs` folders:
  cache 6.8 KB → 20.0 KB (+13.2 KB, conf 3,169 bytes); direct load then fails only
  on the driver.
- `patchelf --set-rpath` on a copy of `vllm/_C.abi3.so`: load then fails only on the
  driver. patchelf footprint measured as tiny (690 bytes reported by the probe).
- Verdict: the CUDA base image is **not** required at runtime. Its value is the
  `ld.so.conf.d` cache, not the libraries. Runtime needs the driver plus the wheel
  libraries, with one of: import torch first, a cache entry + `ldconfig`, or a
  RUNPATH rewrite. The devel image is a build-stage toolchain only.
- Base images (on disk / download): `ubuntu:24.04` 0.07/0.18 GB, `12.8.1-base`
  0.28/0.19, `12.8.1-runtime` 3.17/3.97, `13.3.0-base` 0.53/0.51, `13.3.0-runtime`
  2.22/3.03, `12.8.1-devel` 8.75/6.89 (over budget alone), `python:3.12-slim` 0.34.
- ld.so cache entries: `ubuntu:24.04` 83, `12.8.1-base` 91 (`000_cuda.conf`,
  `988_cuda-12.conf`, `nvidia.conf`), `13.3.0-base` 91 (`987_cuda-13.conf`). A plain
  base has no CUDA entries.
- CUDA tag vs driver: driver 580.95.05 reports CUDA 13.0. `13.0.0-base` starts;
  `13.1.0-base` and `13.3.0-base` fail with `unsatisfied condition: cuda>=13.1` /
  `>=13.3`. `13.3.0-base/runtime/devel-ubuntu24.04` exist on the registry. The default
  `12.8.1-devel-ubuntu24.04` matches the cu12 wheelhouse, not the driver ceiling.
- cutlass-dsl ships **both** `cu12/lib/libcute_dsl_runtime.so` (41.6 MB) and
  `cu13/lib/…` (41.6 MB); `CUDA_VERSION.major` resolves to 13 here, so the cu12 copy is
  dead weight. Open prune.

| Image | Size (on disk) | Compressed download | Cold start | tokens/s | Peak GPU MiB |
| --- | --- | --- | --- | --- | --- |
| `vllm/vllm-openai:latest` (baseline) | — | — | — | — | — |
| `vllm-slim` (engine variant) | — | — | — | — | — |
| `vllm-slim` (thin variant) | — | — | — | — | — |
| `vllm-slim` (.pyc-only ship) | — | — | — | — | — |
| `vllm-omni` overlay | — | — | — | — | — |

### Interim reading, not a verdict

No image has been built from this rule yet, so the table above is still empty and
this is a reading of the selection numbers, not a measured image.

What the measurements do support: the reduction has to come from the layout
(plain base, copy-only site-packages, single arch, no toolchain), not from
pruning files. The 18 largest files are 62.9% of the selection and none of them
is reachable by a glob — they are the libraries libtorch links. A workload maps
38 files / 5,798.9 MB, so the process floor is about 5.8 GB of libraries before
any vLLM code, and the 3,878.1 MB that a run never loads is still shipped unless
a torch source build turns those features off (~692.8 MB, the only route to it).

Against the local baseline image (23.4 GB as reported by `docker images`), the
gap is mostly the CUDA base/devel toolchain, pip, and caches — that part is
removable. The library floor is not. So the honest range is ~60–70% smaller, not
a tenth.

## Decisions

- Strata is reference only; behaviour reimplemented in `packaging/`. No Strata path in any image.
- Read-only clones; build-context copies allowed (`cp -al`, no second copy of GBs).
- Locks and packages generated at build time into `out/`; rules files committed.
- Single CUDA arch detected by command; `VDE_CUDA_ARCH` overrides.
- `USE_DISTRIBUTED` stays ON (see `Plan.md` finding 2). Only MPI/UCC/TensorPipe are droppable.
- No compiler, no pip, no uv, no git, no downloader in the runtime image (gate `R01`).
- `libcuda.so.1` and `libnvidia-*` are never bundled — the host driver provides them.
- Nothing executes on the host. `docker/run.sh` is the only host-side entry point;
  it issues docker daemon commands (`build`, `run`, `pull`, `images`) and every
  gate, test and measurement runs inside the pinned tools container
  (`docker/Dockerfile.tools`). Content moves between images, code does not run on
  the host.
- Experimental research repo: when a decision is not clear, evaluate both options
  and select from the results. Open decisions are listed in `docs/findings.md`.
- CUDA is decided by command, not by a default: driver version and installed CUDA are
  read, stored in `out/cuda.env`, and used to select the builder tag. Values are
  single-quoted because `docker/run.sh` sources the file. Two values are recorded —
  the declared series (wheelhouse, 12) and the surviving copy (tree, 13) — and the
  prune follows the surviving copy. Runtime base is `ubuntu:24.04` (finding 6).
- A CUDA version the user sets wins over detection. The only condition is the
  installed driver, and the error names the driver as the cause and says to use
  another CUDA version or update it. A user value that disagrees with the
  wheelhouse is a warning, not an error.
- A 401 is reported to the user, not retried: the source and the reason are
  printed and the user is asked to check availability or download the file
  directly. Nothing retries, changes credentials, or guesses an alternate URL.
  The wording has one source (`packaging/report.py`), with `docker/pull.sh` as its
  shell copy guarded by a parity test. An image that is already local is used
  without pulling.
- Size target: an image under 6 GB, with 4 GB kept as the stretch goal and
  reported separately. Reduction against the baseline (≥ 60%) is the second
  condition.

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
3. JIT source: **decided by measurement** (see the GPU runs above). The build stage
   warms the cache for the target model and arch, the runtime image ships that cache
   and drops the compile-side tree except `ptxas` and `nvjitlink`. The `.cpp`/`.cc`
   trees in the selection (104.8 MB) are build-stage material only, which is what R01
   requires. flashinfer is still worth a GPU check for models that select it, because
   on sm_86 it has no prebuilt kernels.
4. Uncovered gate: **0 findings** after the decisions (was 14, then 13, then 3). The
   `libgomp.so.1` glob was widened to `libgomp.so.*` because apt installs
   `libgomp.so.1.0.0` plus a symlink, and foreign-arch ELF files are now refused by
   `select()` (the build run failed at `strip` on cutlass-dsl aarch64 stubs).
   **Correction:** the "0 findings" reading was taken on a view built from a tree
   without `ray`. On the real install the gate fails on
   `cupy_backends/cuda/libs/cutensor.cpython-312-x86_64-linux-gnu.so`
   (`libcutensor.so.2`, `libcutensorMg.so.2` — in no wheel of the lock and in no
   folder of the tree) once the `RECORD` owner map is passed to `select()`. Cupy
   cannot ship, so the deny stands. The pruned-file half of the import gate found
   10 findings (9 importers, 132.5 KB, 7 targets, 76.3 KB); decision: keep the 7
   targets, because shipping 76.3 KB is cheaper than breaking an import. After the
   decision both halves report 0 findings on 47,921 files / 10,505.7 MB.
5. R01 vs triton: **decided** — pre-warmed cache, no compiler in the runtime image.
   The build stage must run the serving workload once to produce the cache, and the
   package must carry it. What the build stage runs is now fixed by that decision.
6. Spike A (control) vs Spike B (pruned): `bash docker/run.sh spike`, then
   `bash docker/run.sh script tools/compare_packaging.py`.
7. Decide the loader path for the runtime image (mechanism measured above): import
   order alone, a `ld.so.conf.d` entry + `ldconfig`, or a build-time RUNPATH rewrite.
   Import order is the cheapest and works for the serving path; the cache entry is the
   only option that survives a direct `ctypes.CDLL`. Evaluate both and pick from the
   run, per the repo rule.
8. Decide the CUDA version the image declares: driver ceiling (13.0) vs wheelhouse
   (cu12). Both branches have now been run against the real tree — 43.7 MB pruned for
   keep=12, 43.6 MB for keep=13, nothing pruned when no series is given — so the
   branches are the same size and the choice is which copy the loader resolves. The
   `system_libs` names in the rule are `.12` sonames and must change if the declared
   series becomes 13.
9. Check the base image through the helper before building:
   `bash docker/run.sh pull nvidia/cuda:12.8.1-devel-ubuntu24.04`. That image is
   already local on this host, so `check_image` returns without pulling. If a pull
   is needed and the registry answers 401, the user is asked to check availability
   or download the image directly; the build does not retry.
10. **Bytecode decision, measured: source is kept** (finding 11). Closed:
    `pyc-only` is not importable at all (a sourceless pyc in `__pycache__` is not
    a module), the importable legacy layout breaks on `inspect.getsource()`, and
    the download saving is 55.2 MB — below the ship-it threshold.
    `bytecode_tree()` now forces checked-hash and drops the caches a source tree
    already carries, so every shipped pyc survives the archive's forced mtime:
    measured, the delivered build reads as all-valid (21,061 checked-hash pycs)
    and the entry import as uid 1000 measures 2.75 s median against 6.19 s median
    for the invalid-cache case on the same tool. Nothing is left to fix in the
    runtime stage: the caches the archive ships stay valid, so the launch-time
    penalty is gone.
11. **Partial deny: closed, no measured case requires it.** Evaluated both
    branches on cupy, the only distribution the rule denies: all-or-nothing
    keeps the whole 291.2 MB cupy-side tree out, partial deny would ship 289.6
    MB back for a 1.6 MB cutensor-linked `.so` that still cannot load, because
    `libcutensor.so.2` is a shared library outside the denied package. More
    bytes, same broken import, so all-or-nothing stands. The deny gate reports 0
    findings, so no shipped file needs a denied distribution today; if one
    appears, the gate reports it. Unimplemented, as the user said it is not
    mandatory.
    Gate re-run on the files edited after the previous gate run
    (`docs/PACKAGING.md docs/findings.md tools/evaluate_bytecode.py
    tools/measure_startup.py Plan.md memory.md`): clean, 0 issues, and 132 tests
    passed.
13. Bytecode is settled (finding 11 in `Plan.md`): source is kept, and the
    invalidation change it asked for is implemented in
    `packaging/package.py::bytecode_tree()` — `py_compile.PycInvalidationMode.CHECKED_HASH`
    as the enum (the string form writes flags=1), and the caches a source tree
    already carries are dropped first, because `compileall` skips a cache it
    considers valid and would otherwise ship timestamp pycs. Two facts to keep:
    enum values are 1/2/3 while the header writes `value | 1`, so a header reads
    3 for checked-hash and 0 for timestamp — compare the header, not `.value`;
    and the parity test must force source mtimes before compiling, otherwise the
    tree's own valid caches make the two builds byte-identical for the wrong
    reason.
12. `importscan.imports_pointing_at_dropped()` resolves a dotted import against
    the tree, so it must be given the same relative paths as the selection.
    `package_thin.build()` passes `str(rel)`, `compare_packaging.py` passes a
    `pathlib.Path`; both work, and the gate's docstring says paths must be
    relative to the tree root.
13. The gate resolves imports to `.py` files only, and that is a measured choice:
    adding `.pyi` to the target set produced 324 findings on the real install, all
    of them stubs for compiled extensions. Tested in the materialised view with the
    stub folders pruned, `triton._C.libtriton.linear_layout`, `torch._C._profiler`
    and `tvm_ffi.core` resolve to the `.so` and import fine, as do the importers
    (`triton.tools`, `torch.profiler`, `torch.utils.data`, `tvm_ffi.dataclasses`,
    `vllm.compilation.wrapper`). Gate stays at 0 findings; the test
    `test_stub_files_are_not_import_targets` keeps it that way.14. **JIT cache, closed by measurement.** Two things the harness exposed:
    - `strip` changes the cache key, because the key hashes `triton/_C/libtriton.so`
      (415,617,000 B → 186,568,112 B, prefix `28daaa70a904e4f8` → `31952e683c13e95a`).
      The warm step now runs on the tree that ships, with the C headers put back for
      the run (`tools/warm_cache.py::warm_view`, headers are `.h` so they are not key
      inputs). End-to-end run `out/warm-ship2.txt`: keys agree True, gate PASS.
    - `relocate()` must keep each group in its own kernel-hash folder. A flattened
      rewrite was measured on the shipped tree at 92.4 s with puts=872; the
      folder-preserving form serves in 48.3 s with `hits=219 misses=0 puts=0`
      (`out/jit-runtime-hit2.txt`). `verify()` now compares against the group's
      folder under the target, not the target itself — `os.path.exists()` alone
      failed all 212 groups after a rewrite, because the paths only resolve inside
      the image. Tests: `test_relocate_keeps_the_group_own_folder`,
      `test_verify_resolves_a_relocated_cache_against_the_target`.
    Gates on the changed files (`packaging/cache.py tools/warm_cache.py
    docs/findings.md docker/Dockerfile.vllm memory.md tests/test_packaging.py`):
    clean; suite 141 → 143 passed.