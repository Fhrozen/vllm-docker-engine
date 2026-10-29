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

## Verified

Recorded from runs inside the container:

- gates: clean on 6 explicitly passed files (the evaluation changes), and clean
  on 27 files in the earlier run
- tests: 92 passed, exit 0
- GPU read through the harness: `NVIDIA GeForce RTX 3060`, compute cap 8.6, so
  the target arch is `sm_86`
- `check_files.py` with no arguments: exit 1, refuses
- shellcheck: clean, with one documented exception — `--gpus` must not be
  quoted, and a shellcheck directive must precede the command rather than sit
  inside a `\` continuation
