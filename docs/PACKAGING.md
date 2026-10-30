# PACKAGING.md — how a package is built

`packaging/` is this repo's own engine. It is stdlib only and imports nothing
from `extra_repos/`. The reference clones show the shape of the pipeline; the
code here is a reimplementation, not an import.

## Pipeline order

The order matters, so it is fixed:

1. load the rule (`rules.py`), resolve the version
2. select files (`select()`): deny, prune, static artifacts, symlinks, arch filter
3. stage into `<target>.partial` — an existing target is refused, never overwritten
4. `strip --strip-debug` — an explicit step, not a side effect
5. bytecode (`bytecode_tree`), honouring `SOURCE_DATE_EPOCH` and the rule's
   `pyc_invalidation`
6. hash the staged tree, in sorted order
7. write `manifest.json`
8. `verify(full=True)` — re-hash, refuse symlinks and unlisted files
9. `uncovered()` gate — every `DT_NEEDED` must be bundled or a system library
10. write `SYSTEM-LIBS.txt`
11. atomic rename of the `.partial` folder
12. write the archive, then `check_archive()` on it

Hashing happens **after** strip and bytecode, so the manifest describes the
package as delivered. `SYSTEM-LIBS.txt` is written after hashing, so `verify`
skips it — the manifest can only vouch for the files it names.

## Rule schema (version 2)

```json
{
  "schema_version": 2,
  "id": "vllm-slim",
  "name": "vllm",
  "version": "auto",
  "platform": "linux-x86_64",
  "backend": "cuda",
  "entrypoint": "vllm serve",
  "python": "3.12",
  "requirements": { "clone": "vllm", "files": ["requirements/common.txt"] },
  "allow": ["**"],
  "deny": ["pip"],
  "prune": ["**/tests/**", "**/*.a"],
  "bytecode": "source",
  "pyc_invalidation": "checked-hash",
  "strip": true,
  "arch_filter": "auto",
  "cuda_variant": "nvidia_cutlass_dsl",
  "system_libs": ["libcudart.so.12"],
  "driver_libs": ["libcuda.so.1"],
  "runtime_apt": ["tini"],
  "forbidden_tools": ["pip", "gcc"]
}
```

`version: "auto"` reads the version from `<name>-<version>.dist-info`, using PEP
503 normalisation, so `PyNvVideoCodec-1.2.3.dist-info` matches `pynvvideocodec`.

Patterns are fnmatch with `**` spanning path separators. `arch_filter: "auto"`
uses the detected archs; a file with no architecture in its name is shared data
and is kept.

`cuda_variant` is optional. It names a package that ships more than one CUDA
series, so the prune needs to know which copy to keep — see the section on CUDA
values below. `rules.cuda_variant()` returns False when the rule has no key or
when no series is given.

## Manifest schema (version 2)

```json
{
  "schema_version": 2,
  "id": "vllm-slim",
  "name": "vllm",
  "version": "1.0",
  "platform": "linux-x86_64",
  "backend": "cuda",
  "archs": ["sm_90a"],
  "min_driver": "",
  "entrypoint": "vllm serve",
  "build_id": "<digest of the sorted selection>",
  "files": [{ "path": "foo/a.py", "sha256": "<64 hex>", "bytes": 6 }],
  "signature": ""
}
```

`signature` stays empty: nothing here signs or publishes.

## Determinism

`SOURCE_DATE_EPOCH`, sorted members, `mtime` set, uid/gid 0, uname/gname root,
GNU format, `compresslevel=9`, no symlinks, no hard links.

For bytecode, `compileall` needs `stripdir` and `prependdir`; without them the
`.pyc` carries the build directory inside `co_filename`, so two builds of the
same source into two output folders are not the same package.

Verified: two builds of the same tree into two different `--out` folders produce
byte-identical archives.

### What each bytecode mode delivers

`source` compiles the `.py` files it keeps and does not drop the tree's existing
`__pycache__`, so a source build ships **both** forms. `pyc-only` removes the
`.py` mirrors of files that were compiled, so it ships bytecode only. Measured on
the same rule and arch, `pyc-only` is the smaller package: 9,798.0 MB against
10,102.7 MB for the engine variant, 9,755.0 against 10,020.8 for the thin one.

A tree that already carries bytecode is therefore not a neutral input: the
selection keeps the `.pyc` files it finds, and the mode decides whether the source
form is delivered alongside them.

### What the runtime does with it

Measured with `tools/evaluate_bytecode.py`, on the delivered engine package:

- `__pycache__` is not rebuilt at launch. The runtime stage runs as uid 1000 on
  a root-owned tree whose members are `0o555`, so the cache cannot be written:
  the import succeeds and nothing is added.
- `docker COPY` preserves mtimes, so the image keeps whatever the verify stage
  produced.
- The deterministic archive sets `mtime = SOURCE_DATE_EPOCH` on every member,
  which invalidates every timestamp pyc it carries — 21,054 of 21,064. A cache
  that fails validation is recompiled in memory on each start. Measured on the
  same tree before and after forcing the mtime: 4.9 s against 6.3 s for the
  entry import as uid 1000, 1,665 modules executed. Repeating each run three
  times on the same tree gives 6.19 s median for the invalid cache.
- Hash-based pycs are validated by hashing the source (blake2b, 8 bytes), so
  they survive the forced mtime, and the delivered checked-hash build imports in
  2.75 s median (3 runs). That is below the valid-timestamp reading above; the
  gap is not explained by validation cost, which is a hash of the source rather
  than a stat, so it is reported as a measurement and not as a reason to prefer
  hash pycs. The reason to prefer them is that they stay valid after extraction.
- `compileall` must be given the enum `py_compile.PycInvalidationMode.CHECKED_HASH`;
  the string form is ignored and writes unchecked-hash pycs instead. `compileall`
  also skips a cache it considers valid, so it does not rewrite a timestamp pyc
  the source tree already carries — measured on a probe, flags stayed 0 while a
  fresh build writes 3. `bytecode_tree()` drops the `__pycache__` pycs before
  compiling whenever the invalidation is hash-based, otherwise a build ships a
  mix of both kinds.
- `pyc-only` is not usable as delivered. A sourceless pyc in `__pycache__` is
  not importable at all, so the package resolves as a namespace package and no
  code runs. The layout that does import is the legacy one, with the pyc at the
  module's own location — and there the import fails anyway, because
  `torch/fx/experimental/_config.py` calls `inspect.getsource()` at import time
  and raises `OSError: could not get source code`.
- The timing helper reports this explicitly: it prints
  `namespace package: no code ran` when `vllm.__file__` is None, because a
  folder with no source imports successfully without executing anything. The
  pyc-only build is measured at 0.019 s that way — an import that proves nothing.
  `tools/measure_startup.py <package dir>` is the entry point for both readings:
  pyc validity as delivered, then the entry import as uid 1000.

So `pyc-only` is measured as the smaller package and rejected: 55.2 MB of
download is below the threshold for keeping something that breaks a run. The
mode that ships is `source`, with the cache compiled as checked-hash so the
archive's forced mtime does not invalidate it.

## Target tags

Two measured facts about `pip download` with an explicit target:

- `--platform` does not widen across manylinux revisions. A pinned
  `manylinux_2_28_x86_64` rejects `blake3`, which publishes only
  `manylinux_2_17_x86_64` wheels.
- `--abi cp312` hides `abi3` wheels, so `triton` (cp312-abi3) is invisible to
  the resolver, and packages with their own dependency pins hit
  `ResolutionImpossible` under explicit cross-target flags.

So resolution happens inside the target environment (the builder container),
and the target is enforced afterwards by reading the tags out of every wheel
filename: accepted python tag, accepted abi (`cp312`, `abi3`, `none`), and a
glibc floor equal to the runtime base (2.39 for `ubuntu:24.04`). Legacy
aliases `manylinux1`, `manylinux2010` and `manylinux2014` are mapped to their
own floors. musllinux is rejected because the runtime base is glibc.

`--cross` keeps the explicit-flag path for a target the build machine cannot
run. It is best-effort: pip cannot see the target's dependency graph, so a
conflict there is reported rather than solved.

## The two halves of the deny gate

A rule's `deny` list and a package's declared dependencies can contradict each
other: a name can be denied while a shipped package declares it as a runtime
requirement. That contradiction is split into two gates, because they prove
different things.

- **Declared** (`packaging/package_thin.denied_dependencies`) — reads
  `Requires-Dist` from each shipped distribution's `METADATA` and reports a
  denied name that a shipped package declares as a runtime requirement. This is
  a report, not a failure: a declared dependency is not necessarily imported,
  and the rule intends those names to be optional. Marker lines are skipped
  because extras are never installed here, and a marker cannot be evaluated
  without the `packaging` library.
- **Hard** (`packaging/importscan.runtime_conflicts`) — a denied name is not
  shipped, so if a shipped file imports it at module scope, the image breaks the
  moment that file is imported. This fails the build. Imports inside a function,
  a `try`, or an `if` guard are not executed at import time, so a denied name
  reached only through those stays optional.

Ownership is resolved through each distribution's `RECORD`, not by guessing a
package name into an import name: `flashinfer-python` owns `flashinfer/`,
`opencv-python-headless` owns `cv2/`. A distribution is not a conflict with
itself, and a file's owner comes from the same map.

A deny names a **distribution**, not a folder, so `select()` must be given the
same map. Without it a file's owner is guessed from its first path segment,
`cupy_backends/` does not match `cupy-cuda12x`, and the denied distribution
ships anyway — measured on the engine variant, where the uncovered gate then
fired on a cupy `.so`.

The hard half scans the selection in source form, before the package is built. A
`bytecode: pyc-only` rule removes the `.py` files the scan reads, so scanning the
built tree finds nothing — measured on the first real build, where this gate
printed no findings at all.

The gate prints the declared half and fails on the hard half. `huggingface_hub`
is the case that shows why the halves must be separate: it is denied as optional
and is imported at module level on the serve path (`engine/arg_utils.py`), so it
must ship — the rule has to change, not the build.

- `verify`: hash or size drift, symlinks, files not in the manifest
- `check_archive`: absolute paths, `..` members, non-regular members,
  hash drift, unlisted members
- `uncovered`: a `DT_NEEDED` entry that is neither bundled nor a system lib
- static gate: `.a`, `.o`, `.cmake`, `.prl` in the shipped tree
- lock gate: a wheel not in the lock, a hash mismatch, a lock entry with
  no wheel
- deny gate, declared half: a denied name that a shipped package declares as a
  runtime requirement — reported, not failed
- deny gate, hard half: a shipped file that imports a denied package at module
  level — fails the build
- runtime gate (`R01`): a tool in the final image, root user, no healthcheck

`check_archive` verifies bytes **inside** the archive before anything is
unpacked. That is the point of the gate: a member is trusted only after its
bytes match the manifest.

### Imports pointing at pruned files

A deny stops a distribution from shipping. A prune stops individual files, and a
shipped file can import one of them. `importscan.imports_pointing_at_dropped()`
resolves each module-level import against the tree and reports an importer whose
target is in the tree but not in the selection. Paths must be relative to the
tree root, the same form `select()` returns.

This gate fails the build, and a rule answers it with `keep`: a pattern that
re-selects a file the prune would drop. The decision is which side is cheaper to
ship. On the real install the gate found 7 targets totalling 76.3 KB behind 9
importers, and all 7 were kept — shipping 76.3 KB is cheaper than breaking an
import.

Both import gates are static, so a finding can name a file that no workload ever
imports. That is exactly what happened with `ray`: the importer is reachable, but
the serving path never reaches it. A finding is a call to evaluate both sides, not
proof that the image is broken.

## The evaluation

`tools/evaluate_deny.py` answers three questions for each denied name, and each
answer changes a rule:

1. **Size** — wheel bytes in the wheelhouse, and tree bytes once installed.
2. **Reach** — does a shipped file import it at module scope.
3. **Run** — is it executed at import time on the entry path.

Run inside the container, one interpreter per round, so a name that is executed
is not confused with a name that is only imported:

```bash
bash docker/run.sh script tools/evaluate_deny.py \
  --wheelhouse out/wheelhouse --venv out/eval-venv --rule vllm \
  --view out/eval-view-5
```

The view must start clean. Earlier rounds put folders back into a reused view,
so a reused view reports a package as reachable when the previous run restored
it. `view_is_clean()` refuses a non-empty view for this reason.

Two facts that only a run can give:

- The declared half can execute. `transformers` declares
  `huggingface-hub>=1.31.0,<3.0`, and the entry path reads that through
  `importlib.metadata`, so a denied name is not automatically optional.
- A prune pattern is checked against the real tree layout, not against the
  selection. Checking the selection is circular: `select()` already excludes
  pruned files, so every pattern trivially matches nothing. `fnmatch` is
  case-sensitive, so `nvidia/cusparseLt/**` never fired — the real folder is
  `nvidia/cusparselt`. Both that library and `nvidia/cuda_cupti` are
  `DT_NEEDED` by libtorch, so they cannot be pruned at all: the uncovered gate
  would fail.

`arch_filter` cannot prune a file whose arch is embedded inside its name rather
than a separate token. The flashinfer cubin files are hash-named, so the arch
split had to be measured by hand before the package could be denied.

The decisions the evaluation produced are in `packaging/rules/vllm.json`, and
the numbers behind them are in `docs/SIZES.md`.

### What the reverse closure cannot prove

`elf_needed()` reports what a binary asks the loader to map. It cannot see a
library reached through `dlopen()` or `ctypes.CDLL()`, so "nothing links it" is
a candidate list, not a verdict. `tools/measure_runtime_libs.py` records what a
workload actually maps, through `/proc/self/maps`, and lists what the selection
holds but the process never loaded. Both views are needed: the gate proves a
library is reachable, the run proves which ones are used.

The same gap runs the other way. `importscan` reads source files, so a package
whose shipped form is `.pyc` only cannot be scanned — `runtime_conflicts()`
takes the selection so it reports conflicts for files that will actually be in
the image.

### Triton and the no-compiler rule

R01 forbids a compiler in the runtime image, and triton needs one: a cache miss
raises `CompilationError`, and the compiler lookup happens after the cache is
checked. `cc` alone is not enough — compiling also needs `libc6-dev` headers.
So the build stage must warm triton's cache for the shapes the target model
uses, and the warm cache must travel in the package. The cache key contains no
GPU architecture, so a cache warmed for one arch is read on another; the cache
is arch-scoped only by the code it contains, which is why the build stage must
warm it on the target arch.

Minimum warm cache for one kernel: 10 files, 0.07 MB. A cache covering the
shapes a model uses is small enough to ship, so R01 can stand. The choice is
recorded in `Plan.md` as an open decision.

## CUDA values come from commands

The detection module (`packaging/cuda.py`) is stdlib only and imports nothing
from the reference clones. Values are resolved in this order:

1. an explicit override (`VDE_CUDA`, or a value passed on the command line);
2. the variables file written by detection (`out/cuda.env`);
3. the command itself: `nvidia-smi` for the driver, `nvcc --version` for the
   installed toolchain;
4. the wheelhouse as the floor — a cu12 wheel cannot be relabelled cu13.

Each value records how it was obtained, so a number in a report can be traced
back to the command that produced it. `run()` returns an empty string on
`OSError` or a timeout, so detection never fails a build.

The variables file is sourced by shell, so every value must be single-quoted.
Measured failure: `VDE_INSTALLED_CUDA_HOW=no nvcc on PATH` was executed as a
command and the gate failed with `nvcc fatal : Don't know what to do with
'PATH'`.

A value the user sets wins over detection. `select_series()` takes it first, and
the only condition on it is the installed driver: when the driver cannot run the
selected version, `driver_error()` raises an error that names the driver as the
cause and says to use another CUDA version or update the driver. A user value
that disagrees with the wheelhouse is accepted with a note that the wheelhouse
copy must be fetched — a warning, not an error, because the wheelhouse is a floor
for what can be downloaded, not a ceiling.

The reading must happen in a CUDA devel container: `nvcc` is on the host but not
in the tools container, so `packaging/cuda.py` reports `no nvcc on PATH` there.

Two values, and they are not the same:

| Value | What it names | Measured here |
| --- | --- | --- |
| `VDE_CUDA_SERIES` | the series the wheelhouse declares | 12 |
| `VDE_CUDA_KEEP` | the copy the environment resolves | 13 |

`--cuda` resolves a series in this order: explicit value, `VDE_CUDA_KEEP`,
`VDE_CUDA_SERIES`, then the wheelhouse. The prune follows the surviving copy,
not the declared series. `select_series()` raises when a value is above the
driver ceiling, because that image would not start. `tag_for()` uses a tag
confirmed to exist on the registry, otherwise it falls back to the known default
for the series.

Verified: `write_env()` and `read_env()` round trip a quoted value that contains
spaces; `rules.cuda_variant()` returns False when the rule has no key or no
series is given; both prune branches were measured against the real tree (rule
`vllm`, arch `sm_86`) — keep cu12 43.7 MB pruned, keep cu13 43.6 MB, no series
0.0 MB. Tests also assert that a user value above the driver ceiling raises, that
a value at the ceiling is accepted, and that the shell copy of the 401 sentence
matches the module.

## Download failures

`packaging/report.py` holds the wording for a failed download. A 401 means the
source may not be available, which a build cannot fix by trying again, so nothing
retries, changes credentials, or guesses an alternate URL: `notify()` prints the
source, the sentence asking the user to check availability or download the file
directly, and the reason itself, because that is what the user has to look at.

`docker/pull.sh` is the shell copy, because a shell script cannot import the
module, and `tests/test_report.py` reads that file and fails if the two drift. A
failure that is not an auth failure prints the source and the reason without the
sentence.

Measured runs in the container: a fabricated 401 detail prints the sentence;
`check_image` on an image that is already local returns without pulling;
`nvidia/cuda:12.4.1-devel-ubuntu24.04` prints the registry refusal without the
sentence and exits 1. `lock.py` `download()` reports through the same module, so
a 401 from the package index is reported the same way.

## Driver boundary

`libcuda.so.1` and `libnvidia-*` are never bundled. The host driver provides
them through nvidia-container-toolkit. Only non-stub runtime libraries such as
`libcudart`, `libcublas`, `libnvJitLink` travel in the image, and they come from
the pinned wheel packages, not from a CUDA image.

## Verified behaviour

Recorded from a run inside the tools container (`bash docker/run.sh test`), not
from an estimate:

- `select()` skips symlinks, denied packages, pruned paths, static artifacts,
  and files for an undeclared arch.
- `verify()` reports `foo/a.py: hash mismatch` for a same-length tamper and
  `foo/a.py: size 7 != 6` for a length change.
- `check_archive()` reports escaping members, symlink members, hash drift,
  a manifest entry with no member, and a member with no manifest entry.
- `uncovered()` follows the closure past the first hop: `libssl.so.3` needs
  `libcrypto.so.3`, which needs `libzstd.so.1`.
- `copy_closure()` copies under the soname and honours the arch filter.
- `lock.py --check` reports `foo==1.0: hash mismatch` for a tampered wheel and
  `new==0.1: wheel not in lock` for an extra wheel.
- `check_files.py` refuses being called with no files: a gate that receives no
  arguments renders nothing and exits 0, which is not a pass.
- `matches()` requires a list of patterns. Passing a bare string iterates
  characters, which makes every pattern match every file — that produced a
  false "no dead prune patterns" reading.
- `requirements_for()` refuses a rule that denies its own package, and prints
  how many requirement names the deny list removed from the download.
- `lock_diff()` reports a package installed outside the lock, and ignores the
  environment's own tooling when the rule denies it (`pip` is never shipped).
- `INSTALL_FLAGS` keeps the wheelhouse as the only install source
  (`--no-index`), so a missing wheel is an error rather than a network fetch.
- `requirements_for()` includes the package named by the rule. A lock of
  dependencies alone leaves no `*.dist-info` for `resolve_version` to read. Both
  packages resolve to binary wheels for the target, measured inside the
  container: `vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl` and
  `vllm_omni-0.32.0-py3-none-any.whl`.
- `pip download` resolves the whole transitive graph, so a denied name can
  re-enter the lock through a dependency — `setuptools` is denied by the rule yet
  was downloaded as a dependency of `model-hosting-container-standards`.
  `denied_dependencies()` reports a denied name that a shipped package still
  requires at runtime, counting unmarked requirements only: extras are not
  installed, and an environment marker cannot be evaluated without the
  `packaging` library.
