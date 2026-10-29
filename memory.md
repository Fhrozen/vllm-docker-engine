# Progress log

Continuity mechanism. Update it as work happens: measurements, decisions, dead ends, blockers,
next action. Every number here must come from a run in this repo, not from an estimate.

## Status

| Phase | State | Notes |
| --- | --- | --- |
| 0 — files & plan | done | `Plan.md`, `memory.md`, `wiki.md`, `docs/`, agents |
| 1 — package research | complete (research), pending measurement | findings folded into `Plan.md` |
| 2 — docker build spikes | not started | needs a runnable base image (Docker Hub pulls previously failed with 401; local `vllm/vllm-openai:latest` works) |
| 3 — vLLM slim image | tooling written, not built | `packaging/` + `docker/` are in place |
| 3b — harness | tooling written, no results yet | `tools/measure.sh` |
| 4 — vllm-omni overlay | blocked on the Phase 3 gate | gate: <= 4 GB AND >= 60% smaller than baseline |
| 5 — vllm.cpp | verdict delivered | engine rejected, techniques harvested |

## Measurements

Nothing measured yet. `docs/SIZES.md` holds the table; fill it from `tools/measure.sh` output.

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

## Dead ends

- libtorch-only: rejected (autograd/Dynamo/distributed are hard deps).
- vllm.cpp as the engine: rejected (see `Plan.md` Phase 5).
- `USE_DISTRIBUTED=0`: rejected — breaks `import vllm` even at `tp_size=1`.
- Alpine: rejected (musl, no manylinux wheels).
- Op-level selective torch build: ExecuTorch feature, not libtorch.

## Next action

1. Run the baseline row: `bash tools/measure.sh vllm/vllm-openai:latest`.
2. Run the no-build distributed gate: `python3 tools/check_dist_gate.py --simulate-off`.
3. Spike A (control) vs Spike B (pruned) in docker, then `python3 tools/compare_packaging.py`.
