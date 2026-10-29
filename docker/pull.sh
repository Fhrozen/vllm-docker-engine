#!/bin/bash
# Pull helper, sourced by docker/run.sh and docker/build.sh. Both of those run
# inside the tools container, so this is the shell copy of packaging/report.py:
# the wording is the same, and tests/test_report.py fails if the two drift.
#
# The rule this implements: on a 401 the source may not be available, so nothing
# is retried, no credential is changed, and no alternate URL is guessed. The
# failure is reported with the source and the reason, and the user is asked to
# check availability or download the file directly.
#
# Nothing here runs project code on the host: it only issues docker commands.

AUTH_MARKERS='401|unauthorized|unauthorised|authentication required|authorization required'

report_failure() {
    local source="$1"
    local detail="$2"

    echo "download failed: $source" >&2
    if printf '%s' "$detail" | grep -Eiq "$AUTH_MARKERS"; then
        # Same sentence as packaging/report.py (report.AUTH_SENTENCE), split for
        # readability; tests/test_report.py compares the joined lines.
        echo "the source answered 401, so this repo does not retry it. Please check the" >&2
        echo "availability of the source, or download it directly and put the file where" >&2
        echo "the build expects it." >&2
    fi
    printf '%s\n' "$detail" >&2
}

check_image() {
    local image="$1"
    if [ -n "$(docker images -q "$image" 2>/dev/null)" ]; then
        return 0
    fi

    local detail
    if detail="$(docker pull "$image" 2>&1)"; then
        printf '%s\n' "$detail"
        return 0
    fi

    report_failure "docker pull $image" "$detail"
    return 1
}
