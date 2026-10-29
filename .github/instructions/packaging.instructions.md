---
description: Applies to packaging/, docker/ and tools/ — the engine that builds slim images.
include: ["packaging/**", "docker/**", "tools/**", "tests/**"]
---

# Packaging and image instructions

- Nothing executes on the host. Every run goes through `bash docker/run.sh ...`,
  which issues only docker daemon commands (`build`, `run`, `pull`, `images`).
- `extra_repos/`, `vllm/` and `vllm-omni/` are read-only reference clones. Never
  edit them, never import from them, never copy their files into an image.
- Nothing from Strata or vllm.cpp ships in a produced image. Patterns may be
  reimplemented in `packaging/`; imports are not allowed.
- Runtime images contain no toolchain: no pip, no compiler, no git, no
  downloader. The runtime stage copies a package tree and an apt list only.
- Driver libraries are never bundled. The host driver is the boundary.
- Verify before unpacking: sha256 and size first, then refuse symlinks, absolute
  paths, and members that escape the target directory.
- The cache key is the digest of bytes. Paths, mtimes, and build order are not
  part of a build identity.
- Gates must receive the changed files explicitly. A gate with no arguments
  renders nothing and exits 0, which looks like a pass and is not one.
- No claim without a measurement.
- This is an experimental research repo: if a decision is not clear, evaluate both
  options and select from the results. A decision stays open until both branches
  have been run, and each branch records the command that produced its number.
  Reproducibility claims require a comparison run, not a reasoning step.
- The size target is an image under 6 GB. 4 GB is the stretch goal, so a
  stricter run is still valid and is reported separately from the gate.
- GPU only. CPU-only paths are out of scope.
- Hashing happens after strip and bytecode, so the manifest describes the
  package as delivered. `SYSTEM-LIBS.txt` is written after hashing and is
  skipped by `verify`.
- When a shellcheck directive is needed, it must precede the command. A directive
  inside a `\` line continuation is a parse error.
- `--gpus` must not be quoted: `--gpus ${VDE_GPUS:-device=0}`.
- CUDA and arch values come from commands and a variables file (`out/cuda.env`),
  not from hardcoded defaults. Detection order is override, then command, then
  fallback, and each value records how it was decided.
- A CUDA version the user sets (`--cuda` or `VDE_CUDA`) wins over detection.
  The only condition on it is the installed driver: if the driver cannot run it,
  raise an error naming the driver as the cause and say to use another CUDA
  version or update the driver.
- A 401 from a registry or package index is reported to the user, not retried.
  Nothing retries, changes credentials, or guesses an alternate URL. The wording
  lives in `packaging/report.py`; `docker/pull.sh` is its shell copy, and a test
  fails if the two drift. An image that is already local is used without
  pulling.
- The variables file is sourced by shell, so every value must be single-quoted. An
  unquoted value is executed as a command — this was measured.
