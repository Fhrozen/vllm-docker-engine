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
5. bytecode (`bytecode_tree`), honouring `SOURCE_DATE_EPOCH`
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
  "bytecode": "skip",
  "strip": true,
  "arch_filter": "auto",
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
