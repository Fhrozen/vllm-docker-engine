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

## Layout

```text
packaging/    the packaging engine (rules, archs, ELF deps, archives)
docker/       Dockerfiles, build script, entrypoint, healthcheck, compose
tools/        size gate, packaging comparison, measurement, distributed spike
docs/         human and machine documentation
out/          generated artifacts, not committed
```

## Build and run

```bash
bash docker/build.sh --variant engine --tag vllm-slim:dev
bash docker/build.sh --variant omni --tag vllm-omni-slim:dev
docker compose up vllm-slim
```

Nothing in the runtime image may compile, install, or download. Gates are read
from image metadata, and no size or speed claim is written without a run.
