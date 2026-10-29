# DOCKER.md — image layout, gates, build and run

## Nothing runs on the host

`docker/run.sh` is the only host-side entry point. It issues `docker build` and
`docker run`. Gates, tests, measurements and project scripts run inside the
pinned tools container built from `docker/Dockerfile.tools`.

```bash
bash docker/run.sh image                       # build the tools container
bash docker/run.sh gate <changed files...>     # pinned gates, files explicit
bash docker/run.sh test                        # behaviour tests
bash docker/run.sh build --variant engine      # build an image
bash docker/run.sh size <image> [baseline]     # size + budget gate
bash docker/run.sh runtime <image>             # R01 config gate
bash docker/run.sh measure <image>             # cold start, tokens/s, peak GPU
bash docker/run.sh gpu "<command>"             # a command with a GPU attached
bash docker/run.sh py "<snippet>"              # run a snippet in the container
bash docker/run.sh cuda                        # writes out/cuda.env
bash docker/run.sh pull <image>                # pull through the helper
bash docker/run.sh script <file> [args]        # run a repo script in the container
bash docker/run.sh spike-run "<command>"       # a command that needs a compiler
bash docker/run.sh compile "<command>"         # a command with gcc + libc6-dev
```

The `gpu` task is what makes arch detection possible: the tools container has no
GPU, so `arch.detect()` there always falls back to the full list. Measured with
the GPU attached, the target is `sm_86`.

The `spike` and `spike-run` tasks use `nvidia/cuda:12.8.1-devel-ubuntu24.04`.
That default was corrected after a run: `12.4.1-devel-ubuntu24.04` does not exist
in the registry (manifest unknown), and `*-devel-ubuntu24.04` images have no
`python3`, so a measurement in those images must not assume an interpreter.

The `compile` task exists because the pinned tools image has no C compiler by
design. triton looks up its cache **before** looking up a compiler, so a cache
miss in the runtime image is a hard failure under R01. The build stage must warm
triton's cache for the shapes the target model uses and ship it; the cache key
contains no GPU architecture, so warming must happen on the target arch.

The `cuda` task writes `out/cuda.env`. `run.sh` and `build.sh` source that
file, so every value in it is single-quoted. This was measured: an unquoted
`VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a command and the gate
failed with `nvcc fatal : Don't know what to do with 'PATH'`. `build.sh` takes
`--cuda SERIES|auto` and `--keep SERIES|auto`, falling back to the variables
file. The reading must happen in a CUDA devel container, because `nvcc` is not
in the tools image — the reading is done in a `spike-run` container instead.

A CUDA version the user sets (`--cuda` or `VDE_CUDA`) wins over detection. The
only condition on it is the installed driver: if the driver cannot run the
selected version, the script raises an error naming the driver as the cause and
says to use another CUDA version or update the driver.

The `pull` task and the `spike` tasks go through `docker/pull.sh`, the shell copy
of `packaging/report.py`. `check_image` returns straight away when an image is
already local, so no pull is attempted for it. On a 401 the source and the reason
are printed and the user is asked to check availability or download the file
directly; nothing retries, changes credentials, or guesses an alternate URL.
Measured in the container: a fabricated 401 detail prints the sentence, an
already-local image returns without pulling, and
`nvidia/cuda:12.4.1-devel-ubuntu24.04` prints the refusal without the sentence
and exits 1. A test fails if the shell copy drifts from the module.

The tools image holds the pinned gate versions from the reference clone's
pre-commit config: ruff 0.14.10, typos 1.49.0, markdownlint-cli2 0.23.2,
shellcheck, plus binutils for `strip`, git, docker-cli, pyyaml and pytest.
It is never part of a shipped image.

## Image stages

Three stages, legacy-builder-safe (no heredocs, no `--mount`, no `# syntax=`):

1. `pkg` — resolve the lock, download wheels, run the packaging engine
2. `verify` — manifest re-hash, `DT_NEEDED` gate, lock check, import smoke
3. `runtime` — minimal base, apt packages, the verified tree, tini, healthcheck

`strip` is a pipeline step, so the pkg stage installs binutils. Without the tool
`strip_tree()` prints `strip: tool unavailable, skipping` and the pipeline
silently ships unstripped binaries. binutils stays in stage 1: the runtime image
is checked for `strip` by the R01 gate, so it must not appear there.

The runtime stage installs nothing from pip, has no compiler, and downloads
nothing. `USER 1000:1000`, `tini` as PID 1, script `HEALTHCHECK`, `VOLUME` for
the model cache, `EXPOSE 8000`, OCI labels.

## Gates read from image metadata

`tools/check_runtime_image.py` inspects the built image, not the Dockerfile text:
user, healthcheck, entrypoint, exposed ports, labels, and the presence of each
forbidden tool. Anything it cannot check is reported as `UNVERIFIED`, never as
`OK`. Layout, boot and hub reachability are always `UNVERIFIED` from inside a
build.

## Build context

`docker/build.sh` copies only `packaging/`, `docker/` and `tools/` into the
build context, from the git index, and normalises line endings. The clones are
never in the context. `--ctx-only` writes the context without building, which
is what makes a context auditable.

## Running

The entrypoint runs the module chosen by `VLLM_ENTRY_MODULE`, so the same
entrypoint serves the engine image and the omni overlay. The healthcheck is a
stdlib-only HTTP GET on `/v1/models`; it exits 0 only on HTTP 200 with a
non-empty `data`.

`docker/smoke-test.sh` uses a fixed seed and `temperature 0.0`, so a tokens/s
number is comparable between runs. It fails if the response has no completion
tokens.

## The base image question, answered

Two questions were asked, and both were answered by running rather than by
assuming.

**Why `nvidia/cuda:12.8.1-devel-ubuntu24.04` and not 13.3.** The tag is capped by
the driver, not by the registry. The host driver is 580.95.05 and reports CUDA
13.0. Running the images: `13.0.0-base-ubuntu24.04` starts, `13.1.0-base` and
`13.3.0-base` fail with `nvidia-container-cli: requirement error: unsatisfied
condition: cuda>=13.1` / `>=13.3`. The 13.3 tags do exist on the registry, so the
default is not chosen because 13.3 is unavailable — it is chosen because it
matches the CUDA 12.8 libraries the rule actually ships. torch 2.10.0 pins every
`nvidia-*` dependency to cu12 (`nvidia-cuda-runtime-cu12==12.8.90`,
`nvidia-cudnn-cu12==9.10.2.21`, `nvidia-nccl-cu12==2.27.5`), and
`cuda-bindings==12.9.4`. A base image that declares a newer CUDA than the driver
can run is a build that will not start. The devel variant is used only as a
**build-stage toolchain** (nvcc, compiler); it is never a runtime base, and it is
8.75 GB on disk, which is over the budget on its own.

**Is the CUDA base image required at all, since pip installs the nvidia
libraries?** No, not at runtime. What a CUDA base image contributes is its
`ld.so.conf.d` cache, not its libraries. Measured against the installed tree with
no `LD_LIBRARY_PATH` and no cache entry: `ctypes.CDLL` on `vllm/_C.abi3.so` before
torch fails on `libtorch.so`, while `import vllm` on a plain `ubuntu:24.04` works,
because torch is imported first and every `torch/lib/*.so` carries a
`$ORIGIN/../../nvidia/*/lib` RUNPATH covering the wheel folders. `vllm/_C.abi3.so`
has no RUNPATH of its own. So the runtime image needs one of three things: torch
imported first (true for the serving path), a `ld.so.conf.d` entry plus `ldconfig`
(3.2 KB config, 13.2 KB cache), or a RUNPATH rewrite at build time. Only the
driver — never bundled — is still required, and after torch is loaded the only
missing library is `libcuda.so.1`.

This is why the runtime stage stays on a plain base: `ubuntu:24.04` is 0.07 GB on
disk, a CUDA base is 0.28 GB (cu12) or 0.53 GB (cu13), and a CUDA runtime image
is 2.22–3.17 GB — over the 6 GB target before anything is added. The loader path
is an open decision, recorded in `docs/findings.md`.

## Verified

Recorded from runs inside the container:

- gates: clean on 6 explicitly passed files (the evaluation changes), and clean
  on 27 files in the earlier run
- gates: clean on the 19 files changed for the three rules added in this phase
  (the driver condition, the 401 reporting rule, the 6 GB target). markdownlint
  reported 0 issues in the 5 gated files
- gates: clean on the 9 CUDA-by-command files. That run first FAILED, not on a
  lint issue: `run.sh` sources `out/cuda.env`, and the unquoted line
  `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a command
  (`nvcc fatal : Don't know what to do with 'PATH'`). After quoting every value
  in `write_env`: clean, 3 files reformatted
- tests: 127 passed, exit 0. The earlier run was 114; the added tests cover the
  driver condition, the 401 reporting rule, and the shell-copy parity check
- GPU read through the harness: `NVIDIA GeForce RTX 3060`, compute cap 8.6, so
  the target arch is `sm_86`
- `check_files.py` with no arguments: exit 1, refuses
- shellcheck: clean, with one documented exception — `--gpus` must not be
  quoted, and a shellcheck directive must precede the command rather than sit
  inside a `\` continuation
