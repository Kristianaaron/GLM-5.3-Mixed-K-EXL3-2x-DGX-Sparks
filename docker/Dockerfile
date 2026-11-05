# Copyright 2026 Mia's AI Lab (MiaAI-Lab). Licensed under the Apache License, Version 2.0.
# Adapted from MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104 for the GLM-5.3 full engine build.
# Build context is a local TensorFold checkout; no engine patches are applied.

ARG CUDA_VERSION=13.3.1
ARG UBUNTU=ubuntu24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS build
ARG ZIG_VERSION=0.17.0
ARG ZIG_SHA256
ARG XGRAMMAR_TAG=v0.2.8
ARG XGRAMMAR_COMMIT=97787376faee5ed8466cfad57c99855e4ce2f6aa
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl git xz-utils cmake nasm libibverbs-dev \
 && rm -rf /var/lib/apt/lists/*
RUN test -n "${ZIG_SHA256}" || { echo "ZIG_SHA256 build arg is required" >&2; exit 1; } \
 && curl -fsSL --retry 3 -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" \
 && echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - \
 && mkdir -p /opt/zig \
 && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 \
 && ln -s /opt/zig/zig /usr/local/bin/zig \
 && test "$(zig version)" = "${ZIG_VERSION}"

WORKDIR /src/tensorfold
COPY . .
RUN test "$(cat .zig-version)" = "${ZIG_VERSION}" \
 && git clone -q --depth 1 --branch "${XGRAMMAR_TAG}" --recurse-submodules https://github.com/mlc-ai/xgrammar.git /src/xgrammar \
 && test "$(git -C /src/xgrammar rev-parse HEAD)" = "${XGRAMMAR_COMMIT}" \
 && test "$(git -C /src/xgrammar describe --tags --exact-match)" = "${XGRAMMAR_TAG}" \
 && zig build install native glm53-bringup fatbins -Dsm=121 -Dnvcc=/usr/local/cuda/bin/nvcc \
 && zig build grammar -Dxgrammar=/src/xgrammar

FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU} AS runtime
ENV DEBIAN_FRONTEND=noninteractive LC_ALL=C.UTF-8
RUN apt-get update \
 && apt-get install -y --no-install-recommends libibverbs1 ibverbs-providers rdma-core \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/tensorfold/zig-out/ /tfz/zig-out/
RUN ln -s /tfz/zig-out/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
WORKDIR /workspace
CMD ["tensorfold-native", "--help"]
