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
bash docker/run.sh py "<snippet>"              # run a snippet in the container
```

The tools image holds the pinned gate versions from the reference clone's
pre-commit config: ruff 0.14.10, typos 1.49.0, markdownlint-cli2 0.23.2,
shellcheck, plus binutils for `strip`, git, docker-cli, pyyaml and pytest.
It is never part of a shipped image.

## Image stages

Three stages, legacy-builder-safe (no heredocs, no `--mount`, no `# syntax=`):

1. `pkg` — resolve the lock, download wheels, run the packaging engine
2. `verify` — manifest re-hash, `DT_NEEDED` gate, lock check, import smoke
3. `runtime` — minimal base, apt packages, the verified tree, tini, healthcheck

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

- gates: clean on 27 explicitly passed files
- tests: 63 passed, exit 0
- `check_files.py` with no arguments: exit 1, refuses
- shellcheck: clean, with one documented exception — `--gpus` must not be
  quoted, and a shellcheck directive must precede the command rather than sit
  inside a `\` continuation
