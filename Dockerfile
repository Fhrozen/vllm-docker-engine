# Pointer only — this file is not a build target.
#
# The build targets are:
#   docker/Dockerfile.vllm   the slim engine image
#   docker/Dockerfile.omni    the overlay on top of the engine image
#
# docker/build.sh always passes -f explicitly, and every build runs inside the
# tools container, so `docker build .` at the repo root is not the supported
# path. Use:
#
#   bash docker/run.sh build --variant engine
#   bash docker/run.sh build --variant omni
