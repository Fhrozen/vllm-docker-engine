# vllm-docker-engine

vLLM engine built for docker: a slim GPU image, plus an overlay for vLLM-Omni.

This repo holds only the packaging and docker layer. `vllm/`, `vllm-omni/` and
`extra_repos/` are read-only clones that are not maintained here.

## Where to read

| File | What it is |
| --- | --- |
| `wiki.md` | Rulebook for machine-based assistants: hard rules, conventions |
| `Plan.md` | Plan of record: confirmed constraints, phases, exclusions |
| `memory.md` | Progress log, decisions, dead ends, measurement table |
| `docs/PACKAGING.md` | How a package is built, verified and archived |
| `docs/DOCKER.md` | Image layout, gates, build and run |
| `docs/SIZES.md` | Measured sizes and speed, per variant |
| `docs/findings.md` | Findings so far, open decisions, how decisions are made |

## Layout

```text
packaging/    the packaging engine (rules, archs, ELF deps, archives)
docker/       Dockerfiles, run.sh (container entry point), gates, build script,
              entrypoint, healthcheck, compose
tools/        size gate, runtime gate, file checks, packaging comparison,
              measurement, distributed spike
tests/        behaviour tests for the packaging engine
docs/         human and machine documentation
out/          generated artifacts, not committed
```

## Build, test and run

Everything runs inside a container. `docker/run.sh` is the only host-side entry
point; it issues `docker build` and `docker run`, and nothing executes in the
local shell.

```bash
bash docker/run.sh gate <changed files>   # pinned gates, files passed explicitly
bash docker/run.sh test                   # behaviour tests for the engine
bash docker/run.sh build --variant engine --tag vllm-slim:dev
bash docker/run.sh size vllm-slim:dev
bash docker/run.sh runtime vllm-slim:dev
bash docker/run.sh measure vllm-slim:dev
docker compose up vllm-slim
```

Nothing in the runtime image may compile, install, or download. Gates are read
from image metadata, and no size or speed claim is written without a run.
