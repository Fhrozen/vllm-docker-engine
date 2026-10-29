# SIZES.md — measured sizes and speed

Every number in the first table must come from a run in this repo. An estimate
is not a measurement, and a claim that cannot be reproduced is not written here.

## Measured

| Image | On disk | Download | Cold start | tokens/s | Peak GPU MiB |
| --- | --- | --- | --- | --- | --- |
| `vllm/vllm-openai:latest` (baseline) | — | — | — | — | — |
| `vllm-slim` (engine variant) | — | — | — | — | — |
| `vllm-slim` (thin variant) | — | — | — | — | — |
| `vllm-slim` (.pyc-only ship) | — | — | — | — | — |
| `vllm-omni` overlay | — | — | — | — | — |

Fill from `bash docker/run.sh measure <image>`, which records unpacked bytes,
compressed bytes, container start to healthcheck green, tokens/s, and peak GPU
MiB.

## Phase 4 gate

The omni overlay is only worth building if the engine image is **≤ 4 GB** and
**≥ 60% smaller** than the baseline. Both conditions are checked by
`tools/check_image_size.py`, and both must be measured numbers.

"A tenth of the baseline" is not reachable for a GPU image: torch plus the
NVIDIA runtime libraries are roughly a 2.5–3.5 GB floor before any vLLM code.

## Reference sizes (not measured here)

From the reference clone's documentation, kept as a shape check only:

| Variant | On disk |
| --- | --- |
| cuda13 | 2.2 GB |
| cuda12 | 2.9 GB |
| rocm-gfx1151 | 1.5 GB |

Download size is about a third of the on-disk size. These are the reference
clone's numbers, not this repo's, and they are not evidence that this repo
reaches them.

## What is expected to move the needle

Ordered by expected effect, each one still needs a run:

- copy-only site-packages plus the interpreter, no pip, no venv
- stdlib pruning: `ensurepip`, `test/`, `tkinter`, `idlelib`, `pydoc_data`,
  `lib2to3`, `distutils`
- `.pyc`-only ship, which removes source files after bytecode
- single CUDA arch, chosen by detection rather than the full list
- `strip --strip-debug` as an explicit step
- dropping `nvidia/*/bin`, `cupti`, `nsight`, `nvvm`, dev headers, `pkgconfig`

The libtorch-only idea is worth only 100–250 MB, because the pip `torch` wheels
are unaffected by a source build, and `BUILD_PYTHON` is mandatory for
`torch._C`.
