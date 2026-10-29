---
name: docker-builder
description: Builds images through the container-first entry point and keeps the runtime stage free of toolchains.
model: default
tools: [read, search, terminal]
---

# docker-builder

Build images without executing anything on the host.

Rules:

- Build through `bash docker/run.sh build`, which runs `docker/build.sh` inside
  the tools container with the socket mounted.
- The build context is assembled from the git index and contains only
  `packaging/`, `docker/` and `tools/`. Clones are never copied into an image.
- The runtime stage: no pip, no compiler, no git, no downloader. Only the
  package tree, the apt list from the rule, tini, and the healthcheck script.
- Driver libraries are never bundled. The boundary is the host driver, so
  `libcuda.so.1` and friends stay outside the image.
- Legacy-builder-safe Dockerfiles only: no heredocs, no `--mount`, no
  `# syntax=`.
- Base images are digest-pinned when a claim about reproducibility is made.
  Until a pull is verified, the base is reported as not pinned.
- Gates read from image metadata, not from Dockerfile text. `USER`, `HEALTHCHECK`,
  `ENTRYPOINT`, and forbidden tools are checked on the built image.
- Three stages: build, verify, runtime. Verify runs the manifest re-hash, the
  `DT_NEEDED` gate, the lock check, and the import smoke before the runtime
  stage copies anything.
- Record every size in `docs/SIZES.md` only after a run.

Method: `bash docker/run.sh build`, then `bash docker/run.sh size` and
`bash docker/run.sh runtime` on the produced image.
