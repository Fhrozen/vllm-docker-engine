# Findings so far

A compact record of what has been measured, what it decided, and what is still
open. Every number here came from a run in this repo; nothing is estimated.

## How decisions are made

This is an experimental research repo. When a decision is not clear from first
principles, both options are built and measured, and the winner is selected from
the results. That is the default rule, and it is recorded in `Plan.md` and
`wiki.md`. Two conditions follow from it:

- A decision stays open until both branches have been run. Writing a decision
  down before the comparison is a claim, not a finding.
- The comparison must be reproducible, so each branch records the command that
  produced its number.

Gates enforce the rule mechanically: `tools/check_files.py` refuses to run with
no files, and `tools/compare_packaging.py` only picks a default after both
variants have been measured.

Two rules came from the user during this phase and are now enforced in code:

- A CUDA version the user sets wins over detection. The only condition on it is
  the installed driver, and the error names the driver as the cause.
- A 401 is reported, not retried. Nothing retries, changes credentials, or
  guesses an alternate URL; the user is asked to check availability or download
  the file directly.

## Is the CUDA base image required?

Short answer: **no, not at runtime.** The value of `nvidia/cuda:*-base` is its
dynamic-linker configuration, not its libraries — the libraries come from the pip
wheels. What is required is one of three ways to make the loader find them.

Measured in the tools container against the installed tree, with no
`LD_LIBRARY_PATH` and no ld.so cache entry for the tree:

| Condition | Result |
| --- | --- |
| `ctypes.CDLL(vllm/_C.abi3.so)` before torch | fails: `libtorch.so` not found |
| `import vllm` on a plain `ubuntu:24.04` | works, `vllm._C` loads |
| `ctypes.CDLL` after torch | fails only on `libcuda.so.1` (the driver) |
| `ldconfig` + a `ld.so.conf.d` entry | works, driver only |
| `patchelf --set-rpath` on the .so | works, driver only |

The reason `import vllm` works on a plain base is that torch is imported first,
and every `torch/lib/*.so` carries a `$ORIGIN/../../nvidia/*/lib` RUNPATH that
covers the wheel folders. `vllm/_C.abi3.so` has no RUNPATH of its own, so it can
only be found through torch. Of 367 ELF files with a dynamic section, 65 have a
RUNPATH/RPATH, 54 of which use `$ORIGIN`.

So the runtime image needs exactly one of:

1. nothing, if torch is always imported first (true for the serving path);
2. a `ld.so.conf.d` file plus `ldconfig` — costs 3.2 KB of config and 13.2 KB of
   cache, and is the only option that does not depend on import order;
3. a RUNPATH rewrite at build time — costs nothing at runtime but requires
   patchelf in the build stage.

Option 2 is the safe default because it survives a direct `ctypes.CDLL` call. The
devel image remains useful only as a **build-stage toolchain** (nvcc, compiler),
never as a runtime base.

## Which CUDA tag

The tag is capped by the driver, not by the registry. The host driver is
580.95.05, which reports CUDA 13.0. Running the images confirms the ceiling:

| Tag (ubuntu24.04) | Starts on this host |
| --- | --- |
| `nvidia/cuda:13.0.0-base` | yes, driver 580.95.05 |
| `nvidia/cuda:13.1.0-base` | no, `cuda>=13.1` unsatisfied |
| `nvidia/cuda:13.3.0-base` | no, `cuda>=13.3` unsatisfied |

CUDA 13.3 tags exist on the registry, so the current default of
`nvidia/cuda:12.8.1-devel-ubuntu24.04` is not chosen because 13.3 is unavailable
— it is chosen because it matches the CUDA 12.8 libraries that the rule actually
ships. The wheelhouse is cu12: torch 2.10.0 pins every `nvidia-*` dependency to
cu12 (`nvidia-cuda-runtime-cu12==12.8.90`, `nvidia-cudnn-cu12==9.10.2.21`,
`nvidia-nccl-cu12==2.27.5`), and `cuda-bindings==12.9.4`.

Matching the base image to the wheelhouse matters for one reason: the driver must
be new enough for the runtime libraries, and a base image that declares a newer
CUDA than the driver can run is a build that will not start. A cu12 wheelhouse on
a driver that reports CUDA 13.0 is fine; the reverse is not.

If the build stage is changed to a CUDA image at all, use the smallest one that
has nvcc, and never the runtime or base variant for the final stage — see the
size table in `docs/SIZES.md`.

### The tag is read by command, not chosen by a default

This package releases code for a personal/local image, so the CUDA version comes
from the machine: driver version and installed CUDA are read by command, stored
in `out/cuda.env`, and used to select and generate the builder tag. Detection
order is override, then command, then fallback, and each value records how it was
obtained.

The order is override first, so a user value wins over the driver ceiling and
over the wheelhouse, with one condition: the driver must be able to run it. Tests
assert this: a user value above the ceiling raises an error that names the driver
as the cause and says to use another CUDA version or update the driver. A value
at or below the ceiling is accepted, and a value that disagrees with the
wheelhouse is accepted with a note that the wheelhouse copy must be fetched.

The reading has to happen in a CUDA devel container. `nvcc` is on the host but
not in the tools container, so `packaging/cuda.py` reports `no nvcc on PATH`
there; `bash docker/run.sh spike-run` is where the reading is done.

The variables file is sourced by shell, so every value must be single-quoted.
Measured failure: an unquoted `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was run as
a command and the gate failed with `nvcc fatal : Don't know what to do with
'PATH'`.

### Two CUDA values, and they are not the same

| Value | What it names | Measured here |
| --- | --- | --- |
| `VDE_CUDA_SERIES` | the series the wheelhouse declares | 12 |
| `VDE_CUDA_KEEP` | the copy the environment resolves | 13 |

The prune follows the latter. Both branches were run against the real tree (rule
`vllm`, arch `sm_86`):

| Keep | Files | Tree MB | cutlass MB | Pruned MB |
| --- | --- | --- | --- | --- |
| cu12 | 48,672 | 10,736.2 | 462.6 | 43.7 |
| cu13 | 48,672 | 10,736.3 | 462.7 | 43.6 |
| no series | 48,673 | 10,779.9 | 506.3 | 0.0 |

The branches are the same size, so the decision is which copy the loader
resolves, not which is smaller. With no series nothing is pruned.

## The duplicate cu12/cu13 tree

`nvidia-cutlass-dsl` ships both `cu12/lib/libcute_dsl_runtime.so` and
`cu13/lib/libcute_dsl_runtime.so`, 41.6 MB each. The runtime selects one folder
by `CUDA_VERSION.major`, which resolves to 13 in this environment, so the cu12
copy is dead weight. This is an open prune, pending the decision on which CUDA
version the image declares.

Also open: the four stub `.so` files in that package are not ELF, which makes
`strip` print "Unable to recognise the format". `strip` now skips non-ELF and
foreign-arch files, so this is no longer a build failure.

## JIT: what the image actually needs, measured

JIT is not one thing in this stack, so it was measured as three: the
engine that compiles, the per-kernel cache, and the packages that carry
their own compiler. The tool is `tools/evaluate_jit.py`, and every run
below happened inside a container with the GPU attached, on the target
model.

The cache key is the digest of the compiler source files, so pruning
them changes the key. Each tree was therefore measured in its own
process, against the cache the full tree produced.

| Run | Tree | Cache | Compiler | Result |
| --- | --- | --- | --- | --- |
| G | full | cold | none | FAILED: no C compiler |
| M | full | cold | gcc + libc6-dev | OK 97.0 s, 1033 puts |
| N | full | cold | gcc + libc6-dev, compiled | OK 119.4 s, 1710 puts |
| J | full - cupti, ptxas-blk, 3rdparty, data | warm | none | 42.4 s, hits=114 |
| K | J - `nvidia/cuda_nvrtc` | warm | none | 42.3 s, hits=114 |
| L | K - whole `tilelang`, `flashinfer` | warm | none | 42.0 s, hits=114 |
| K | K | warm from run N | none | 53.2 s, hits=110 |
| H | full - cupti, `bin/ptxas` | warm | none | FAILED: no ptxas |
| I | full - cupti, nvrtc, nvjitlink | warm | none | FAILED: libtriton load |

Runs that pass load 509.7 MB of JIT files; runs that had to compile load
620.5 MB. The difference is `cuda_nvrtc`, which only appears when a
kernel is built.

What the runs say:

- **The engine itself cannot be pruned.** `libtriton.so` (415.6 MB) and
  `libnvJitLink.so.12` (94.1 MB) load even when every kernel is a cache hit.
- **`ptxas` is required even on a full cache hit.** The backend hash calls
  `ptxas --version`, so run H fails although nothing needed compiling. `ptxas`
  (31.9 MB) stays; `ptxas-blackwell` (34.9 MB) goes.
- **`cuda_nvrtc` (222.1 MB) is only used when a kernel is compiled.**
  Runs J, K and L serve with it removed, so a pre-warmed image does not
  need it.
- **cupti (110.8 MB) is not used for serving.** It is profiling.
- **A cold cache needs a C compiler *and* C standard-library headers.** Run M first
  failed on `stdlib.h` from `triton/backends/nvidia/include/cuda.h`, so installing
  `gcc` alone is not enough — `libc6-dev` is also required.
- **The cache key does not change when compiler/backends `.py` files
  are pruned**: `3.6.00355c3e3452` with and without
  `triton/backends/amd`. A cache produced on the full tree is valid on
  the pruned tree.
- **Environment variables do not change the key**:
  `get_cache_invalidating_env_vars()` returns `{}`, with and without
  `TRITON_BACKENDS_IN_TREE=1`.

### The tradeoff, in numbers

| Option | What ships | What it costs |
| --- | --- | --- |
| Cache pre-warmed, JIT pruned | 48.6 eager / 58.5 compiled MB | 597.3 MB cut |
| Compiler in the image | `gcc` + `libc6-dev` = 111.6 MB | nvrtc + cupti stay |

Pre-warming wins by roughly **660 MB**, and it keeps the no-toolchain
rule for the runtime image: the rule forbids a C toolchain and headers,
which a warm cache removes, while `ptxas` is an assembler that the cache
key needs, not a C compiler.

Caveat to record: run L drops `tilelang` (176.3 MB) and `flashinfer`
(53.2 MB) entirely. Nothing breaks for this model in eager mode, but
both are optional kernel backends, so a different model or a compiled
path that selects them would fail. The decision is therefore *prune by
default, keep as an opt-in flavor*, not prune always.

## cupy and ray: a deny read through the record owner

`cupy-cuda12x` is denied by the rule, and the gate fired on it. Two facts made
that fire possible, and both are now in the code:

- A `deny` entry names a **distribution**, not a folder. `cupy-cuda12x` installs
  `cupy/`, `cupyx/` and `cupy_backends/`, so `norm("cupy_backends")` is
  `cupy-backends`, which is not the denied name. The `RECORD` owner map is what
  makes the deny apply. Measured before the fix, the engine variant selected the
  cupy folders and the uncovered gate failed on
  `cupy_backends/cuda/libs/cutensor.cpython-312-x86_64-linux-gnu.so:
  libcutensor.so.2 / libcutensorMg.so.2`. Neither library is in the tree or in
  the wheelhouse, so cupy cannot ship without adding a package: the gate cannot
  be satisfied for it.
- The importer came from `ray`, which the rule does not name. vllm's own
  requirements files (`common.txt`, `cuda.txt`) name neither ray nor cupy; both
  enter through `ray[cgraph]>=2.48.0`, a hard `Requires-Dist` of the vllm wheel.
  Cupy is only an extra of ray (`extra == "all-cpp"`), so the rule intends it to
  be optional — but the lock as generated resolves that extra and contains
  `ray==2.59.0` and `cupy-cuda12x==14.2.0`, so the lock-based thin variant does
  install cupy, and that is why the deny gate fires on the real install.
  Exactly one file outside cupy imports it at module scope:
  `ray/util/collective/collective_group/cuda_stream.py`, 2,975 bytes, unguarded.
  No vllm file imports `ray.util.collective`.

Both branches were run on the GPU, same model, seed, prompt and warm cache:

| Branch | Tree | Result |
| --- | --- | --- |
| cupy pruned, ray's importer pruned | 66,796 files / 10,911.4 MB | OK 41.6 s |
| cupy pruned, ray kept whole | 66,797 files / 10,911.4 MB | OK 41.8 s |
| ray and cupy both pruned | 61,480 files / 10,681.1 MB | OK 40.7 s |

The second row is the point: the cupy import is never reached by the serving
path, so the conflict is static only. The gate is stricter than the served path
— it fails on a file that ships but is never imported. The rule therefore denies
cupy and prunes the three ray files that reach it (`cuda_stream.py`,
`nccl_collective_group.py`, `ray/experimental/channel/nccl_group.py`), which is
the cheaper side: 291.2 MB unpacked (136.1 MB wheel) against a few kilobytes of
ray code.

Correction to record: the serving workload does **not** map cupy. cupy appears in
the never-mapped list — `cupy` 221.0 MB, `cupy_backends` 34.4 MB, `cupyx` 9.9 MB
(265.3 MB), `ray` 53.7 MB.

### Imports that point at a pruned file

The deny gate looked at top-level folders only, so a shipped file that imports a
pruned file *inside a kept distribution* passed the gate. Measured on the real
tree: 9 importers (132.5 KB) pointing at 7 targets (76.3 KB).

Targets, and the shipped file that imports each:

- `numba/tests/support.py` — 45.4 KB — `numba/cuda/testing.py`,
  `numba/misc/help/inspector.py`
- `jsonschema/tests/_suite.py` — 8.4 KB — `jsonschema/benchmarks/issue232.py`,
  `jsonschema/benchmarks/json_schema_test_suite.py`
- `numpy/_core/tests/_natype.py` — 6.3 KB — `numpy/conftest.py`,
  `numpy/testing/_private/utils.py`
- `torch/testing/_internal/distributed/rpc/examples/`
  `reinforcement_learning_rpc_test.py` — 9.3 KB —
  `torch/testing/_internal/distributed/rpc_utils.py`
- `torch/testing/_internal/distributed/rpc/examples/`
  `parameter_server_test.py` — 4.6 KB — the same file
- `torch/_export/db/examples/__init__.py` — 1.6 KB —
  `torch/_export/db/gen_example.py`
- `numba/tests/__init__.py` — 0.8 KB — `numba/misc/POST.py`

Decision: **keep the 7 targets**. The rule's size condition says to ship what
costs a few kilobytes rather than break an import, so `prune` now has a `keep`
exception list, and `select()` applies `prune` only when `keep` does not match.
Gate re-run on the real tree after the decision: 47,921 files / 10,505.7 MB,
0 findings on both halves (it reported 10 before the `keep` entries).

Caveat: neither side is reached at runtime. Importing the 9 importers in the
materialised view fails on `pyperf`, `hypothesis` and `expecttest`, so those
importers are themselves dev files that the entry path never executes. The gate
is a static check, and the decision is what the static check should allow.

## When a download answers 401

The wording lives in `packaging/report.py`, and `docker/pull.sh` is its shell copy
because a shell script cannot import the module. `tests/test_report.py` reads the
shell file and fails if the two copies drift.

Measured runs in the container:

| Run | Result |
| --- | --- |
| fabricated 401 detail | prints the source and asks the user |
| `check_image` on a local image | returns without pulling |
| `nvidia/cuda:12.4.1-devel-ubuntu24.04` | refused, no 401 sentence, exit 1 |

A real network 401 has not been observed since the rule was written, so the
reporting path is proven by the fabricated case and the local-image case only.

## Size facts that hold so far

- The gate threshold is an image under 6 GB on disk, with 4 GB kept as the
  stretch goal. `tools/check_image_size.py` reports the stretch separately from
  the gate, so a `--budget 4GB` run is still valid.
- The wheelhouse is 48.7% of the selected tree (5,364.3 MB of 10,505.7 MB).
- The libraries the serving workload actually maps are 38 files / 5,798.9 MB,
  against 44,441 files / 9,986.6 MB selected — so most of the tree is never
  loaded, and the static `DT_NEEDED` closure (20 libs / 2,904.2 MB)
  underestimates what a run needs.
- `ubuntu:24.04` is 0.07 GB on disk. A base CUDA image is 0.28 GB (cu12) or
  0.53 GB (cu13); a runtime CUDA image is 2.22–3.17 GB, which is over budget on
  its own. This is the strongest argument for the plain-base + wheels layout.
- The honest floor for a GPU image is ~2.5–3.5 GB. A tenth of the baseline is
  not reachable; the achievable reduction is ~60–70%.

## Open decisions

Each of these is undecided by design and will be settled by running
both sides. The JIT decision is already settled by measurement, so it is
recorded in the section above rather than here.

| Decision | What decides it |
| --- | --- |
| Loader path | import order, `ld.so.conf.d` + `ldconfig`, or RUNPATH rewrite |
| CUDA version the image declares | read by command: ceiling vs wheelhouse |
| `.pyc`-only or source ship | startup latency vs +27.7 MB |
| Packaging variant | `tools/compare_packaging.py` after both are measured |
| cutlass cu12 vs cu13 prune | which copy the loader resolves, both measured |
| tilelang / flashinfer | pruned by default, kept as an opt-in flavor |

The cutlass decision is settled in mechanism and open in value: the choice is
which copy the environment resolves, and both branches have been measured above.
