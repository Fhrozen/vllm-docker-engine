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

Each of these is undecided by design and will be settled by running both sides.

| Decision | What decides it |
| --- | --- |
| Triton cache vs relaxing R01 | warmed cache (0.07 MB) or a compiler |
| Loader path | import order, `ld.so.conf.d` + `ldconfig`, or RUNPATH rewrite |
| CUDA version the image declares | driver ceiling (13.0) vs wheelhouse (cu12) |
| `.pyc`-only or source ship | startup latency vs +27.7 MB |
| Packaging variant | `tools/compare_packaging.py` after both are measured |
| cutlass cu12 vs cu13 prune | which copy the loader resolves, both measured |
| CUDA version the image declares | read by command, ceiling vs wheelhouse |

The cutlass decision is settled in mechanism and open in value: the choice is
which copy the environment resolves, and both branches have been measured above.
