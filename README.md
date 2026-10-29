# vllm-docker-engine

Packaging and Dockerfiles only. The engine is built by this repo's own
`packaging/` tooling; the reference clones under `extra_repos/` are read-only
and nothing from them ships in an image.

## Where to read

| File | What it is |
| --- | --- |
| `Plan.md` | plan of record, phases, decision rule |
| `memory.md` | measurements, gate results, decisions |
| `wiki.md` | hard rules and conventions for machine assistants |
| `docs/findings.md` | findings so far, with the command behind each number |
| `docs/PACKAGING.md` | how a package is built |
| `docs/SIZES.md` | sizes, only from runs |
| `docs/DOCKER.md` | how to build, test and run |

## Layout

```text
docker/       run.sh (the only host-side entry point), build.sh, Dockerfiles
packaging/    this repo's packaging engine
tools/        gates and measurement scripts
tests/        unit tests
docs/         findings, sizes, packaging, docker
extra_repos/  read-only reference clones, never shipped
out/          build output and machine-specific artifacts, not committed
```

## Build, test and run

Everything runs in docker; nothing executes on the host.

```bash
bash docker/run.sh gate <changed files>  # lint gates, files must be named
bash docker/run.sh test                  # unit tests
bash docker/run.sh py "<snippet>"        # run python in the tools container
bash docker/run.sh gpu "<command>"       # same, with the GPU attached
bash docker/run.sh cuda                  # writes out/cuda.env
bash docker/run.sh pull <image>          # pull through the helper, 401 reported
bash docker/run.sh build                 # build an image from the selection
bash docker/run.sh size                  # measure an image
bash docker/run.sh serve-smoke           # parity smoke test
```

CUDA is decided by command, not by a default in a Dockerfile. Driver version and
installed CUDA are read, stored in `out/cuda.env`, and used to select the builder
tag for a personal/local image. The run script sources that file, so every value
in it is single-quoted.

A CUDA version the user sets wins over detection (`--cuda` or `VDE_CUDA`). The
only condition on it is the installed driver: if the driver cannot run it, the
script raises an error naming the driver as the cause and says to use another
CUDA version or update the driver.

A 401 from a registry or package index is reported, not retried. The source and
the reason are printed, and the user is asked to check availability or download
the file directly. Nothing retries, changes credentials, or guesses an alternate
URL.

The size target is an image under 6 GB; 4 GB is kept as the stretch goal.
