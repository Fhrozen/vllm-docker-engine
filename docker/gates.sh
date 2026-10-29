#!/bin/bash
# Gate runner. This runs INSIDE the tools container (see docker/run.sh); the host
# never runs a gate directly.
#
# Rule: a gate runs on the files passed explicitly. A gate with no arguments
# renders nothing and exits 0, which looks like a pass, so it is refused here.
set -eu

REPO="${VDE_WORKDIR:-/workspace}"
cd "$REPO"

FILES=("$@")
if [ ${#FILES[@]} -eq 0 ]; then
    echo "gates: no files given. Pass the changed files explicitly."
    exit 1
fi

PY=()
SH=()
MD=()
YAML=()
JSON=()
OTHER=()
for file in "${FILES[@]}"; do
    case "$file" in
        *.py) PY+=("$file") ;;
        *.sh) SH+=("$file") ;;
        *.md) MD+=("$file") ;;
        *.yaml|*.yml) YAML+=("$file") ;;
        *.json) JSON+=("$file") ;;
        # The text hygiene hooks apply to every file, not only the ones with a
        # tool of their own, so anything else still gets checked.
        *) OTHER+=("$file") ;;
    esac
done

echo "gates: ${#FILES[@]} file(s) passed explicitly"

status=0

# ruff: the project config selects UP and I with target py310, 88 columns.
if [ ${#PY[@]} -gt 0 ]; then
    echo "-- ruff check"
    ruff check --fix --output-format github "${PY[@]}" || status=1
    echo "-- ruff format"
    ruff format "${PY[@]}" || status=1
    echo "-- check_files (debug-statements, final newline, line endings)"
    python3 tools/check_files.py --fix "${PY[@]}" || status=1
fi

if [ ${#SH[@]} -gt 0 ]; then
    echo "-- shellcheck"
    shellcheck "${SH[@]}" || status=1
    echo "-- bash -n"
    for file in "${SH[@]}"; do
        bash -n "$file" || status=1
    done
    python3 tools/check_files.py --fix "${SH[@]}" || status=1
fi

# markdownlint-cli2 is gated on (^docs/|^recipes/|README.md$|CONTRIBUTING.md).
if [ ${#MD[@]} -gt 0 ]; then
    gated=()
    for file in "${MD[@]}"; do
        case "$file" in
            docs/*|recipes/*|*README.md|CONTRIBUTING.md) gated+=("$file") ;;
            *) echo "markdownlint: $file is outside the gated pattern, skipped" ;;
        esac
    done
    if [ ${#gated[@]} -gt 0 ]; then
        echo "-- markdownlint-cli2"
        markdownlint-cli2 --fix "${gated[@]}" || status=1
    fi
    for file in "${MD[@]}"; do
        python3 tools/check_files.py --fix "$file" || status=1
    done
fi

if [ ${#YAML[@]} -gt 0 ]; then
    echo "-- check-yaml"
    python3 tools/check_files.py --fix "${YAML[@]}" || status=1
fi

if [ ${#JSON[@]} -gt 0 ]; then
    echo "-- check-json"
    python3 tools/check_files.py --fix "${JSON[@]}" || status=1
fi

if [ ${#OTHER[@]} -gt 0 ]; then
    echo "-- text hygiene (files with no tool of their own)"
    python3 tools/check_files.py --fix "${OTHER[@]}" || status=1
fi

echo "-- typos"
typos "${FILES[@]}" || status=1

echo
if [ "$status" -eq 0 ]; then
    echo "gates: clean"
else
    echo "gates: FAILED (see above). No claim without a run."
fi
exit "$status"
