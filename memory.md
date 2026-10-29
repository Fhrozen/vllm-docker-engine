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
