# wiki.md — machine-assistant reference

Read this before touching the repo. It is the rulebook for automated assistants; `Plan.md` is the
plan of record and `memory.md` is the running progress log.

## What this repo is

A packaging + docker build system for a **slim GPU vLLM image** (and an optional `vllm-omni`
overlay). The engines themselves (`vllm/`, `vllm-omni/`) and the reference clones
(`extra_repos/Strata`, `extra_repos/vllm.cpp`) are **not maintained here** — they are cloned.
This repo owns only: `packaging/`, `docker/`, `tools/`, `docs/`, and the root files.

## Hard rules

1. **Read-only clones.** Never edit `vllm/`, `vllm-omni/`, `extra_repos/`. Copying files *into* a
   build context is allowed (`cp -al`, hard links, no second copy of gigabytes).
2. **Nothing from Strata or vllm.cpp ships.** Their behaviour is reimplemented in `packaging/`.
   No file from `extra_repos/` may be copied into an image, and nothing at build time imports them.
3. **No toolchain in the runtime image** (gate `R01`): no `pip`, `uv`, `gcc`, `g++`, `cc`, `cmake`,
   `ninja`, `nvcc`, `ld`, `git`, `curl`, `wget`. Build stages may have them; the final stage may not.
4. **Driver boundary.** `libcuda.so.1` and `libnvidia-*` are never bundled — the host driver
   provides them through nvidia-container-toolkit. Only non-stub `libcudart`/`libcublas*`/
   `libnvJitLink` travel in the image.
5. **Verify before unpacking** (gate `P11`): every shipped file's SHA-256 and size is checked
   against the manifest before extraction; symlinks, absolute paths, and members that escape the
   package root are refused.
6. **Cache key = digest of bytes.** A build step is keyed on the content hash of its inputs, never
   on mtime or on a file-name list.
7. **Gates run on the changed files, passed explicitly.** A gate that receives no arguments
   renders nothing and exits 0 — that is not a pass.
8. **No claim without a measurement.** Never write "bit-identical", "deterministic", "X MB", or
   "faster" without the run that produced it. Record numbers in `docs/SIZES.md` and `memory.md`.
9. **GPU only.** No CPU-only lane. A CPU dual-run is not a deliverable.
10. **Nothing executes on the host.** `docker/run.sh` is the only host-side entry point and it
    issues only `docker build` / `docker run`. Every gate, test, measurement and script runs inside
    the pinned tools container (`docker/Dockerfile.tools`). Content may move between images; project
    code never runs in the local shell.

## Layout

```text
packaging/     this repo's own engine (stdlib only, no third-party imports)
  rules/       committed rule files (allow/deny/prune/bytecode/strip/arch_filter)
  package.py   engine variant: select + manifest + verify + atomic rename
  package_thin.py  thin variant: hash-locked install, prune, deterministic archive
  libdeps.py   ELF DT_NEEDED closure, system/driver classification, uncovered gate
  lock.py      hash-pinned lock generation + --check (generated into out/, not committed)
  arch.py      CUDA arch detection by command
docker/        Dockerfile.vllm, Dockerfile.omni, Dockerfile.tools (pinned gate/test
               container), run.sh (the only host-side entry point), gates.sh,
               build.sh, entrypoint.sh, healthcheck.sh, smoke-test.sh,
               docker-compose.yaml
tools/         check_image_size.py, check_runtime_image.py, check_files.py,
               compare_packaging.py, measure.sh, check_dist_gate.py,
               spike_torch_distributed.sh
tests/         conftest.py, test_packaging.py — behaviour tests for the engine
docs/          PACKAGING.md, DOCKER.md, SIZES.md
out/           generated packages, wheelhouse, locks, harness results (gitignored)
```

## Packaging conventions

- Manifest schema 2: `schema_version, id, name, version, platform, backend, archs, min_driver,
  build_id, entrypoint, files[{path, sha256, bytes}], signature`. `signature` stays empty —
  nothing here signs or publishes.
- Staging writes into `<name>.partial`, then `verify(full=True)`, then an atomic rename. An
  existing output folder is refused, never overwritten.
- Lock format (same as the reference clone, reimplemented): `name==version \` then an
  indented `--hash=sha256:<hex>`, one entry per package, sorted.
- Install command for the thin variant:
  `pip install --isolated --no-cache-dir --no-warn-script-location --require-hashes --no-deps
  --only-binary=:all: -r <lock>`, followed by `pip check` and a diff against the lock names.
- Deterministic archive: `SOURCE_DATE_EPOCH`, sorted members, no symlinks, no hard links,
  `compresslevel=9`.
- `strip --strip-debug` is an explicit pipeline step, not a side effect of the build.
- Forbidden suffixes in the shipped tree: `.a`, `.o`, `.cmake`, plus `pkgconfig/` and dev headers —
  except when the `.pyc`-only mode is *not* chosen, in which case `.py` is allowed by rule.

## Docker conventions

- Legacy-builder-safe Dockerfiles only: no heredocs, no `--mount`, no `RUN --mount=type=cache`.
- Three stages: `pkg` (build) → `verify` (manifest + DT_NEEDED gate + import smoke) → runtime.
- Runtime: minimal base, `apt-get install -y --no-install-recommends`, then
  `rm -rf /var/lib/apt/lists/*`. `tini` as PID 1. Script-based `HEALTHCHECK`. Non-root
  `USER 1000:1000`. `VOLUME` for the cache. Digest-pinned base where possible.
- Gates are read from **image metadata**, not from the Dockerfile text: entrypoint, no `CMD`, uid,
  `EXPOSE`, `VOLUME`, `HEALTHCHECK`, OCI labels. Report `UNVERIFIED` rather than `OK` when a gate
  cannot be checked.
- The build context is assembled from an allow-list (`packaging/`, `docker/`, `tools/`) through
  `git ls-files --cached --others --exclude-standard`. Tracked files alone are not enough: a new
  file is silently absent from the context, and the build fails at its first import.

## Environment facts (this host)

- Docker Hub pulls work (`python:3.12-slim`, `node:22-slim`). Local `vllm/vllm-openai:latest` works
  too. Its entrypoint is `vllm serve`, so `--entrypoint bash` is required for anything else.
- The tools container has `bash`, `ruff`, `typos`, `markdownlint-cli2`, `shellcheck`, `git`,
  `docker-cli`, `pyyaml`, `pytest`, `strip` (binutils) and PyPI access. `apt` works after
  `apt-get update`.
- `--gpus` must not be quoted: `--gpus '"device=0"'` breaks the daemon. Pass it unquoted
  (`--gpus ${VDE_GPUS:-device=0}`), with a shellcheck disable since a directive inside a `\`
  continuation is a parse error.
- Container-created `.venv` is not usable from the host. Host has no `pip`, no `python` (only
  `python3`), no `pre-commit`, no `shellcheck`, no `git` inside the container.
- Gate tools run pinned: `uvx ruff@0.14.10 --no-cache`, `uvx typos@1.49.0`,
  `npx --yes markdownlint-cli2@0.23.2`, `bash -n` for shell files. Use the pre-commit pin — a
  different ruff version reformats differently than CI.
- Ruff parses `.sh` as Python and reports invalid syntax. Never run ruff over shell files.
- Project lint selects `UP` and `I` with `target-version = "py310"`.
- `grep -E '[ \t]+$'` is not a tab check (in ERE, `\t` is a literal backslash + `t`). Use `grep -P`
  or a Python byte check. `last=$(tail -c 1 f)` is always empty — check the final newline with
  `tail -c 1 f | od -An -c`.
- `tail -N` on a long pytest run hides failure detail. Redirect to a log file and grep for the
  short test summary plus the exception lines.
- `pip download --platform` does not widen a request across manylinux revisions, and
  `--abi cp312` hides `abi3` wheels. A single pinned tag cannot resolve a real requirement
  set (`blake3` publishes only `manylinux_2_17`; `triton` is `cp312-abi3`). Resolve inside
  the target environment and enforce the target by reading tags back out of the wheel
  filenames.
- pip spells the target flags with hyphens: `--python_version` is not an option.
- git refuses a bind-mounted repo as "dubious ownership". Configure it through
  `GIT_CONFIG_COUNT`/`safe.directory` rather than writing config into the workspace.
- The docker CLI resolves paths in its own filesystem. Inside the tools container the build
  context is `/workspace/out/context`; the daemon receives a tar, never a host path.
- `HEALTHCHECK` requires `CMD`.
- `--arch auto` is what the rules declare, so the CLI must read it as detection instead of
  handing it to `normalise()`.

## Tool-usage notes

- `grep_search` always requires `isRegexp`.
- `grep_search` returns empty for `vllm/**` and `vllm-omni/**` in this workspace; fall back to
  `read_file`, or `github_text_search` with `scope` exactly `owner/repo` (a path-scoped scope is
  rejected).
- `fetch_webpage` on raw.githubusercontent returns garbled or omitted lines; prefer
  `github_text_search`.
- `read_file` requires both `startLine` and `endLine`. `create_file` cannot overwrite an existing
  file. `multi_replace_string_in_file` needs `filePath` inside every replacement item.

## Testing

- Run tests through the container: `bash docker/run.sh test`. The suite writes only to `tmp_path`
  and never reaches the network; ELF tests copy a real system `.so` into `tmp_path` and skip when
  none is found, because a text fixture cannot stand in for an ELF file.
- `tests/conftest.py` inserts `packaging/` and `tools/` into `sys.path`. Never `sys.path.append(".")`
  from the repo root — `import tokenizers` then resolves to `vllm_omni/tokenizers`.
- Offline pytest on this host (no `vllm` installed) needs fake package parents built with
  `types.ModuleType` + `__path__`, leaf modules loaded via
  `importlib.util.spec_from_file_location` (registered in `sys.modules` **before** `exec_module`),
  then `pytest.main([files, "-q", "-c", "/dev/null", "--confcutdir=...", "-p", "no:cacheprovider",
  "-m", "not gpu and not cuda"])`.
- Inside the container use `python -m pytest` after activating `.venv`; plain `pytest` resolves to
  the system pytest and cannot see packages installed into `/workspace/.venv`.
- Standalone scripts under this repo need `PYTHONPATH=/workspace` in the container.
- Marker collection is measurable without GPUs: `pytest -q --collect-only <file> -m '<expr>'`.
