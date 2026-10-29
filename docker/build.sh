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

# Same 401 reporting as docker/run.sh: a base image that the registry will not
# serve is reported, not retried, so the user can check the source or download it.
# shellcheck disable=SC1090
. "$REPO/docker/pull.sh"

ALLOWED=("packaging" "docker" "tools")
CONTEXT="${VDE_CONTEXT:-$REPO/out/context}"

# The docker CLI reads the Dockerfile and the context from its own filesystem,
# then sends a tar to the daemon. Running inside the tools container means the
# correct path is the bind mount (/workspace), not the host path — the daemon
# never resolves a host path here, it receives the archive.

# The CUDA values come from the variables file written by the cuda task, which
# reads them by command from the machine the image will run on. This script is a
# consumer of that decision, not the place where the decision is made. When the
# file is absent nothing is assumed beyond the fallback the Dockerfile declares.
CUDA_ENV="${VDE_CUDA_ENV:-$REPO/out/cuda.env}"
if [ -f "$CUDA_ENV" ]; then
    # shellcheck disable=SC1090
    . "$CUDA_ENV"
fi

VARIANT="engine"
PACKAGING="thin"
TAG="vllm-slim:dev"
BASE="${VDE_RUNTIME_BASE:-ubuntu:24.04}"
BUILDER="${VDE_CUDA_BUILDER:-python:3.12-slim}"
FLAVOR="slim"
ARCH="auto"
CUDA_SERIES="${VDE_CUDA_SERIES:-auto}"
CUDA_KEEP="${VDE_CUDA_KEEP:-auto}"
DRIVER="${VDE_DRIVER_VERSION:-}"
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
  --cuda SERIES|auto           CUDA series, e.g. 12; default is out/cuda.env
  --keep SERIES|auto           which duplicate copy survives; default is out/cuda.env
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
        --cuda) CUDA_SERIES="$2"; shift 2 ;;
        --keep) CUDA_KEEP="$2"; shift 2 ;;
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
    # --cached alone lists tracked files only, so a new file is silently missing
    # from the context and the build fails on its first import. --others adds
    # untracked files, --exclude-standard keeps gitignored output out of it.
    (cd "$REPO" && git ls-files -z --cached --others --exclude-standard -- "${ALLOWED[@]}" | tar --null -T - -cf -) | (cd "$CONTEXT" && tar -xf -)
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

# Both stages pull from the registry, so a base that is not available is reported
# before the build starts rather than inside a build step.
for image in "$BUILDER" "$BASE"; do
    check_image "$image" || exit 1
done

docker build \
    -f "$REPO/$DOCKERFILE" \
    --build-arg BASE="$BASE" \
    --build-arg BUILDER="$BUILDER" \
    --build-arg TARGET_ARCH="$TARGET_ARCH" \
    --build-arg CUDA_ARCH="$ARCH" \
    --build-arg CUDA_SERIES="$CUDA_SERIES" \
    --build-arg CUDA_KEEP="$CUDA_KEEP" \
    --build-arg DRIVER="$DRIVER" \
    --build-arg PACKAGING="$PACKAGING" \
    --build-arg VERSION="${TAG#*:}" \
    --label "org.opencontainers.image.title=vllm-${FLAVOR}" \
    --label "org.opencontainers.image.ref.name=${TAG}" \
    --tag "$TAG" \
    "$CONTEXT"
