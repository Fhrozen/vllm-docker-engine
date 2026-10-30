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
10. **Nothing executes on the host.** `docker/run.sh` is the only host-side entry
    point and it issues only docker daemon commands (`build`, `run`, `pull`,
    `images`). Every gate, test, measurement and script runs inside the pinned
    tools container (`docker/Dockerfile.tools`). Content may move between images;
    project code never runs in the local shell.
11. **This is an experimental research repo.** If a decision is not clear, evaluate
    both options and select from the results. A decision stays open until both
    branches have been run, and each branch records the command that produced its
    number. Open decisions are listed in `docs/findings.md`.
12. **CUDA is decided by command, not by a default in a Dockerfile.** Driver version
    and installed CUDA are read (`nvidia-smi`, `nvcc`), stored in a variables file
    (`out/cuda.env`), and used to select and generate the image. Every value in that
    file is single-quoted because `docker/run.sh` sources it: an unquoted line was
    executed as a command (`VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` → `nvcc fatal :
    Don't know what to do with 'PATH'`). The images built here are personal/local, so
    the detected environment is the input — not a guess.
    **A CUDA version the user sets wins over detection** (`--cuda`, `VDE_CUDA`);
    the only condition on it is the installed driver. When the driver cannot run
    the selected version, the script raises an error naming the driver as the
    cause and tells the user to use a different CUDA version or update the driver.
13. **A 401 is reported to the user, not retried.** A 401 from an image registry
    or package index means the source may not be available, which a build cannot
    fix by trying again. Nothing retries, changes credentials, or guesses an
    alternate URL: the source and reason are printed, and the user is asked to
    check availability or download the file directly. The wording lives in
    `packaging/report.py`; `docker/pull.sh` is its shell copy, and a test fails
    if the two drift.
14. **The size target is under 6 GB.** 4 GB stays as the stretch goal, so a
    stricter run is still valid. Reduction against the baseline is the secondary
    condition.

## Layout

```text
packaging/     this repo's own engine (stdlib only, no third-party imports)
  rules/       committed rule files (allow/deny/prune/bytecode/strip/arch_filter)
  package.py   engine variant: select + manifest + verify + atomic rename
  package_thin.py  thin variant: hash-locked install, prune, deterministic archive
  libdeps.py   ELF DT_NEEDED closure, system/driver classification, uncovered gate
  lock.py      hash-pinned lock generation + --check (generated into out/, not committed)
  arch.py      CUDA arch detection by command
  report.py    the 401 rule in one place, so the same wording reaches the
               download, the registry check and the run script
docker/        Dockerfile.vllm, Dockerfile.omni, Dockerfile.tools (pinned gate/test
               container), run.sh (the only host-side entry point), pull.sh (the
               shell copy of report.py), gates.sh,
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
- `strip --strip-debug` is an explicit pipeline step, not a side effect of the build, so the pkg
  stage installs binutils. Without the tool `strip_tree()` prints "strip: tool unavailable,
  skipping" and the pipeline ships unstripped binaries.
- A prune pattern is checked against the **full tree**, never against the selection: `select()`
  already excludes pruned files, so checking the selection is circular and every pattern looks
  dead. `fnmatch` is case-sensitive, so `nvidia/cusparseLt/**` never fired — the real folder is
  `nvidia/cusparselt`.
- `rules.matches()` takes a **list** of patterns. A bare string iterates characters and makes
  every pattern match every file, which produced a false "no dead prune patterns" reading.
- A prune can break the image when libtorch `DT_NEEDED`s the target. That is why
  `nvidia/cusparselt` (452.0 MB) and `nvidia/cuda_cupti` (41.3 MB) cannot be
  pruned at all.
- A `deny` entry names a **distribution**, not a folder, so it is enforced through
  the `RECORD` owner map: `cupy-cuda12x` owns `cupy/`, `cupyx/` and
  `cupy_backends/`, and `norm("cupy_backends")` is `cupy-backends`. `select()`
  must be given that map, or a denied distribution still ships and the uncovered
  gate fires on it.
- A deny can fire on a lock-based build even when the requirements files do not
  name the package. vllm's `common.txt` and `cuda.txt` name neither `ray` nor
  `cupy-cuda12x`, but the lock contains both because the vllm wheel requires
  `ray[cgraph]`. Denying a name is a policy choice; it does not mean the lock
  resolves without it.
- The import gate has two halves: a shipped file must not import a denied
  distribution, and it must not import a file the rule prunes. Both halves are
  static, so a finding can name a file that no workload ever imports — that is
  what happened with `ray`'s cupy importer. `keep` in the rule is the exception
  list for the second half, used when shipping the target is cheaper than
  breaking an import.
- `arch_of()` requires the arch as a distinct token. Files whose arch is embedded inside the
  name (hash-named cubins) cannot be arch-filtered, so that split has to be measured by hand.
- The evaluation view must start clean. A reused view reports a package as reachable because an
  earlier round put it back.
- `package.select()` returns **relative posix path strings**, not `Path` objects. `install_locked()`
  needs `Path`, so passing selection strings raises `AttributeError`.
- `build()` runs the uncovered gate on the staged folder **before** the atomic rename, so a failing
  build leaves no package folder behind.
- `strip` must be gated by `is_elf()` **and** by ELF machine: `elf_soname()` raises `IndexError` on a
  non-ELF file, and `select()` already refuses a foreign-arch ELF, so a `.so` that fails `strip` is
  a machine mismatch, not a corrupt file.
- The wheelhouse is the source of the lock and must start empty. It is a directory, so a second
  download leaves the older wheel beside the newer one, and a lock listing both cannot install.
- `rules.load()` returns a **dict**, so rule fields are `rule["system_libs"]`, not attributes.
- Triton needs a C compiler at runtime unless its cache is pre-warmed, and the cache is checked
  **before** the compiler is looked up. `cc` alone is not enough — compiling also needs `libc6-dev`
  headers. Triton's cache key contains no GPU architecture, so the build stage must warm the cache
  on the target arch and ship it.
- Even on a full cache hit, `ptxas` is called (the backend hash runs
  `ptxas --version`), so it stays in the runtime image;
  `ptxas-blackwell` does not. `libtriton.so` needs `nvjitlink` at load
  time, so that tree stays too. `cuda_nvrtc` and cupti are only used
  when a kernel is compiled or profiled, so a warm-cache image can drop
  them.
- The cache key is the digest of the compiler source files, so pruning
  them changes the key. Pruning `.py` files under `compiler/` or
  `backends/` does **not** change it (measured: `3.6.00355c3e3452`
  with and without `triton/backends/amd`), and env vars do not either
  (`get_cache_invalidating_env_vars()` returns `{}`). Measure each tree
  in its own process, because `triton_key()` reads files from disk.
- Forbidden suffixes in the shipped tree: `.a`, `.o`, `.cmake`, plus `pkgconfig/` and dev headers —
  except when the `.pyc`-only mode is *not* chosen, in which case `.py` is allowed by rule.
- Files being present in the tree is not the same as the loader finding them. `uncovered()`
  checks that a `DT_NEEDED` name exists somewhere by file name; it does not check that
  `ld.so` can resolve it. The value of a CUDA base image is its `ld.so.conf.d` cache, not
  its libraries. Either torch is imported first (its `$ORIGIN/../../nvidia/*/lib` RUNPATH
  covers the wheel folders), or the image needs a `ld.so.conf.d` entry plus `ldconfig`, or
  a RUNPATH rewrite at build time. `vllm/_C.abi3.so` has no RUNPATH of its own, so a direct
  `ctypes.CDLL` on it fails before torch is imported.
- CUDA and arch values come from commands and are stored in a variables file
  (`out/cuda.env`), never hardcoded as defaults. The file is **sourced by shell**, so every
  value must be single-quoted — `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a
  command before quoting was added. Detection order is override → command → fallback, and
  each value records *how* it was decided, the same pattern `arch.py` uses.
- There are **two** CUDA values and they are not the same: the series the wheelhouse
  declares (12 for this rule) and the copy the environment resolves (13, from
  `nvidia_cutlass_dsl/cuNN` and `CUDA_VERSION.major`). Both are recorded — `VDE_CUDA_SERIES`
  for the declared series, `VDE_CUDA_KEEP` for the surviving copy — and the prune follows the
  latter. `rules.cuda_variant()` prunes a folder's copies of other series; it returns False
  when the rule has no `cuda_variant` key or no series is given.
- `nvcc` is on the host but **not** in the tools container, so the installed-toolchain reading
  must happen in a CUDA devel container (`spike-run`), not in the tools image. Inside the tools
  container `packaging/cuda.py` reports `no nvcc on PATH`, which is the honest answer there.

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
- An image is checked before it is used: `docker/pull.sh` (`check_image`) returns straight away
  when the image is already local, so a pull is not attempted for it. This matters because the
  tools image is built on first use, and a base that cannot be pulled would fail every task.
  `docker/pull.sh` is sourced by `docker/run.sh` and `docker/build.sh`; it is the shell copy of
  `packaging/report.py`, and `tests/test_report.py` fails if the wording drifts.

## Running things

`docker/run.sh` is the only host-side entry point. Tasks: `image`, `build`, `gate`, `test`,
`script`, `py`, `dist`, `size`, `runtime`, `measure`, `serve-smoke`, `spike`, `spike-run`,
`compile`, `gpu`, `cuda`, `pull`, `shell`. Everything runs in the pinned tools
container; `gpu` runs a command in
a GPU container because the tools container has no GPU, so arch detection there always falls back
to the full list. `spike-run` and `compile` exist because the pinned tools image has no C
compiler on purpose — triton builds a C extension at runtime, so a measurement that needs a
compiler must run in the CUDA devel container or in a one-run container that installs gcc and
`libc6-dev`.

Two mechanisms answer "what must ship", and they must both be used:

- **Static**: `packaging/libdeps.py` reads `DT_NEEDED` from ELF files. This proves what the loader
  is asked to map, and it is what the uncovered gate enforces.
- **Dynamic**: `tools/measure_runtime_libs.py` reads `/proc/self/maps` after a workload runs. This
  proves what a workload actually uses, and it catches libraries reached through `dlopen()` or
  `ctypes.CDLL()`, which `DT_NEEDED` cannot see.

The dynamic view is the one that can prune; the static view is the one that can fail a build.

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
- MD013 (max 80) checks prose lines and table rows but **not fenced code blocks**.
  Measured with a probe file: a prose line of 82 and a table row of 82 were
  reported, an 85-character line inside a fenced block was not. So command
  examples in fenced blocks may run long, and prose paragraphs must be wrapped
  at word boundaries.
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
- Download and registry failures are tested without a network. `test_report.py`
  checks the 401 wording, that a non-auth failure prints only the source and the
  detail, and that the shell copy in `docker/pull.sh` still contains the same
  sentence: the test reads the file and fails if the two drift.
- Offline pytest on this host (no `vllm` installed) needs fake package parents built with
  `types.ModuleType` + `__path__`, leaf modules loaded via
  `importlib.util.spec_from_file_location` (registered in `sys.modules` **before** `exec_module`),
  then `pytest.main([files, "-q", "-c", "/dev/null", "--confcutdir=...", "-p", "no:cacheprovider",
  "-m", "not gpu and not cuda"])`.
- Inside the container use `python -m pytest` after activating `.venv`; plain `pytest` resolves to
  the system pytest and cannot see packages installed into `/workspace/.venv`.
- Standalone scripts under this repo need `PYTHONPATH=/workspace` in the container.
- Marker collection is measurable without GPUs: `pytest -q --collect-only <file> -m '<expr>'`.
