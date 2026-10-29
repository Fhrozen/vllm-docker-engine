#!/bin/bash
# Context assembly and build.
#
# Only committed files under the allow-list reach the docker context. The
# read-only clones (vllm/, vllm-omni/, extra_repos/) are never part of the
# context, so a build cannot accidentally depend on an untracked edit.
#
# Hard links are used when the context lives on the same filesystem, which
# keeps assembly cheap without changing the bytes.
set -eu

REPO="$(cd "$(dirname "$0")/.." && pwd)"
ALLOWED=("packaging" "docker" "tools")
CONTEXT="${VDE_CONTEXT:-$REPO/out/context}"

# The docker CLI reads the Dockerfile and the context from its own filesystem,
# then sends a tar to the daemon. Running inside the tools container means the
# correct path is the bind mount (/workspace), not the host path — the daemon
# never resolves a host path here, it receives the archive.

VARIANT="engine"
PACKAGING="thin"
TAG="vllm-slim:dev"
BASE="ubuntu:24.04"
BUILDER="python:3.12-slim"
FLAVOR="slim"
ARCH="auto"
TARGET_ARCH="linux-x86_64"
CTX_ONLY=0

usage() {
    cat <<USAGE
usage: bash docker/build.sh [options]

  --variant engine|omni        which Dockerfile to build (default: engine)
  --packaging engine|thin      packaging variant recorded as a build arg
  --tag NAME:TAG               image tag (default: vllm-slim:dev)
  --base IMAGE                 runtime base, digest-pinned when possible
  --builder IMAGE              build stage base
  --flavor NAME                label written into the image title
  --arch LIST|auto             CUDA architectures for the arch filter
  --target TRIPLET             wheel platform tag for lock generation
  --ctx-only                   assemble the context and stop, do not build
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --packaging) PACKAGING="$2"; shift 2 ;;
        --tag) TAG="$2"; shift 2 ;;
        --base) BASE="$2"; shift 2 ;;
        --builder) BUILDER="$2"; shift 2 ;;
        --flavor) FLAVOR="$2"; shift 2 ;;
        --arch) ARCH="$2"; shift 2 ;;
        --target) TARGET_ARCH="$2"; shift 2 ;;
        --ctx-only) CTX_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

case "$VARIANT" in
    engine) DOCKERFILE="docker/Dockerfile.vllm" ;;
    omni) DOCKERFILE="docker/Dockerfile.omni" ;;
    *) echo "unknown variant: $VARIANT" >&2; exit 1 ;;
esac

rm -rf "$CONTEXT"
mkdir -p "$CONTEXT"

if command -v git >/dev/null 2>&1; then
    (cd "$REPO" && git ls-files -z -- "${ALLOWED[@]}" | tar --null -T - -cf -) | (cd "$CONTEXT" && tar -xf -)
else
    for path in "${ALLOWED[@]}"; do
        [ -d "$REPO/$path" ] || continue
        cp -a "$REPO/$path" "$CONTEXT/$path" 2>/dev/null || cp -r "$REPO/$path" "$CONTEXT/$path"
    done
fi

# CRLF in a script is a runtime failure on a Linux image, not a style issue.
find "$CONTEXT" -name '*.sh' -type f -print | while read -r file; do
    tr -d '\r' < "$file" > "$file.tmp"
    mv "$file.tmp" "$file"
    chmod 755 "$file"
done

echo "context: $CONTEXT"
if [ "$CTX_ONLY" = "1" ]; then
    exit 0
fi

docker build \
    -f "$REPO/$DOCKERFILE" \
    --build-arg BASE="$BASE" \
    --build-arg BUILDER="$BUILDER" \
    --build-arg TARGET_ARCH="$TARGET_ARCH" \
    --build-arg CUDA_ARCH="$ARCH" \
    --build-arg PACKAGING="$PACKAGING" \
    --build-arg VERSION="${TAG#*:}" \
    --label "org.opencontainers.image.title=vllm-${FLAVOR}" \
    --label "org.opencontainers.image.ref.name=${TAG}" \
    --tag "$TAG" \
    "$CONTEXT"
