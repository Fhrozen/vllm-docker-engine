---
description: Applies to packaging/, docker/ and tools/ — the engine that builds slim images.
include: ["packaging/**", "docker/**", "tools/**", "tests/**"]
---

# Packaging and image instructions

- Nothing executes on the host. Every run goes through `bash docker/run.sh ...`,
  which issues only `docker build` and `docker run`.
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
  have been run, and each branch records the command that produced its number. Reproducibility claims require a comparison
  run, not a reasoning step.
- GPU only. CPU-only paths are out of scope.
- Hashing happens after strip and bytecode, so the manifest describes the
  package as delivered. `SYSTEM-LIBS.txt` is written after hashing and is
  skipped by `verify`.
- When a shellcheck directive is needed, it must precede the command. A directive
  inside a `\` line continuation is a parse error.
- `--gpus` must not be quoted: `--gpus ${VDE_GPUS:-device=0}`.
