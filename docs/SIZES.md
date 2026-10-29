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

## Selection measured

Run inside the container:

```bash
bash docker/run.sh script tools/evaluate_deny.py \
  --wheelhouse out/wheelhouse --venv out/eval-venv \
  --rule vllm --arch sm_86 --view out/eval-view-5
```

Target GPU, read from the container: `NVIDIA GeForce RTX 3060`, compute cap
8.6, so `sm_86`.

| Measurement | Number |
| --- | --- |
| Selection, rule as previously committed | 51,514 files, 11,021.6 MB |
| Selection, rule after the decisions below | 44,441 files, 9,986.6 MB |
| Wheelhouse (compressed download) | 5,364.3 MB, 189 wheels |
| Compressed share of the unpacked selection | 48.7 % |

These are selection bytes, not image bytes: no image has been built from this
rule yet, so the table above stays empty.

Largest trees in the selection, after the decisions:

| Tree | MB | What it is |
| --- | --- | --- |
| `nvidia` (cublas) | 4572.0 | linked by `libtorch_cuda.so` |
| `torch` | 1759.7 | executed at import time |
| `vllm` | 1195.6 | executed at import time |
| `triton` | 585.2 | executed at import time |
| `tokenspeed_triton` | 294.0 | imported by `tokenspeed-mla` |
| `nvidia_cutlass_dsl` | 269.6 | no module-level importer |
| `tilelang` | 147.2 | imported by itself |
| `llvmlite` | 133.6 | imported by `numba` |
| `transformers` | 107.9 | executed at import time |
| `cuda` (bindings) | 106.3 | imported by 11 distributions |

The 18 largest files are 62.9 % of the selection: `libtorch_cuda.so` 1043.8,
`libcublasLt.so.12` 751.8, `libcudnn_engines_precompiled.so.9` 547.4,
`vllm/_C.abi3.so` 465.6, `libcusparseLt.so.0` 452.0, `libtorch_cpu.so` 448.2,
`libnccl.so.2` 429.6, `libtriton.so` 415.6, `libcusparse.so.12` 388.0,
`libcudnn_adv.so.9` 285.2, `libcufft.so.11` 278.9, `_vllm_fa2_C.abi3.so` 263.6,
`_moe_C.abi3.so` 254.7, `libcusolver.so.11` 243.3, `_cutlass_ir` 191.6,
`libcusolverMg.so.11` 161.5, `libnvshmem_host.so.3` 160.3,
`_vllm_fa3_C.abi3.so` 146.7 MB. None of these is prunable by a glob: they are
the libraries libtorch links.

## What the measurements changed

- **flashinfer-cubin is dead weight for this GPU.** 1,114.8 MB unpacked
  (267.7 MB compressed), no module-level importer, reached only through
  `importlib.util.find_spec`. Its kernels are `sm_100` 937.0 MB and `sm_103`
  127.3 MB, with 50.5 MB unmarked: **zero bytes usable on `sm_86`**. Denied.
  flashinfer itself (30.0 MB selected, 53.5 MB tree) is imported at module
  level by core files (`vllm/ir/op.py`, `vllm/v1/worker/gpu_worker.py`), so it
  cannot be pruned. Without cubins it falls back to JIT, which needs `nvcc` at
  runtime and breaks rule R01 — so on this GPU flashinfer ships as inert code
  unless the backend is never selected.
- **`.pyc-only` costs more than source.** Inside the selection, `.pyc` is
  258.0 MB and `.py` is 230.3 MB: `bytecode: pyc-only` is a startup-latency
  choice worth −27.7 MB, not a saving.
- **Two prune patterns never fired.** `nvidia/cusparseLt/**` matches nothing
  because the real folder is `nvidia/cusparselt` (452.0 MB shipped);
  `nvidia/cupti/**` matches nothing because the real folder is
  `nvidia/cuda_cupti` (41.3 MB shipped). Both are `DT_NEEDED` by libtorch
  (`libcusparseLt.so.0` by `libtorch_cuda.so` and `libtorch_python.so`,
  `libcupti.so.12` by `libtorch_cpu.so`), so they cannot be pruned at all —
  only dropped by a torch rebuild with `USE_CUSPARSELT=OFF` / `USE_KINETO=OFF`.
  `nvidia/nvvm/**`, `nvidia/*/bin/**` and `nvidia/*/nsight/**` match no folder
  in the tree.
- **The libtorch-only estimate was too low.** Measured as libraries, the
  droppable torch features are cusparselt 452.0 MB, nvshmem 196.2 MB, cupti
  41.3 MB, cufile 3.3 MB — about 692.8 MB, not 100–250 MB. The pip wheel ships
  them enabled, so the saving only exists for a source build.
- **JIT source inside the selection: 104.8 MB** (`.cpp` 87.6 MB, mostly
  cuda-bindings; `.cc` 14.3 MB, mostly tilelang). R01 forbids a compiler at
  runtime, so that source can never be used by the shipped image.
- **Prune patterns that do fire**, measured against the full tree (72,495
  files): `**/include/**` 137.6 MB, `**/*.a` 84.4 MB, `**/tests/**` 70.6 MB,
  `**/*.pyi` 5.6, `**/*.pyx` 5.2, `**/examples/**` 2.6, `**/test/**` 0.6,
  `**/*.cmake` 0.3. `**/benchmarks/**` (2.9 MB) was removed from the rule
  because the entry path reaches `vllm.benchmarks` through the CLI.
- **Names the entry path executes, so they cannot be pruned:**
  `huggingface-hub` 6.8 MB and `setuptools` 7.5 MB. Both were in the deny list
  and are now shipped.

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
- `.pyc`-only ship: measured as a cost, not a saving (see above)
- single CUDA arch, chosen by detection rather than the full list — this is
  what made flashinfer-cubin prunable, and it only works where the wheel names
  the arch; the cubin files are hash-named, so `arch_filter` cannot see them
- `strip --strip-debug` as an explicit step, now a real step because binutils
  is installed in the pkg stage
- dropping JIT source, which R01 makes unusable at runtime (104.8 MB)
- a torch source build with the droppable features off: about 692.8 MB, the
  only route to that saving, since the pip wheel ships them enabled

Glob pruning cannot reach the libraries libtorch links. The measured floor for
this rule is the NVIDIA runtime closure plus libtorch plus the vLLM extensions.
