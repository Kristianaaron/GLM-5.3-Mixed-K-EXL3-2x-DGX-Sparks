#!/usr/bin/env bash
# Copyright 2026 Mia's AI Lab (MiaAI-Lab). Licensed under the Apache License, Version 2.0.
# Adapted from MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104; image and link discovery adapted for this engine.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/scripts/memavail.sh"
ENV_FILE="$SCRIPT_DIR/.env"
if [[ ! -r "$ENV_FILE" ]]; then
  printf 'Missing config file: %s\n' "$SCRIPT_DIR/.env" >&2
  exit 1
fi
# The adjacent .env is a trusted shell-style configuration file.
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${MODEL_DIR:?Set MODEL_DIR in .env}"
: "${DRAFTER_DIR:?Set DRAFTER_DIR in .env}"
: "${ENGINE_DIR:?Set ENGINE_DIR in .env}"
: "${IMAGE:?Set IMAGE in .env}"
: "${PEER_SSH:?Set PEER_SSH in .env}"
: "${RANK0_ADDR:?Set RANK0_ADDR in .env}"
: "${CACHE_DIR:?Set CACHE_DIR in .env}"
PORT=${PORT:-8888}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-110}
CONTEXT=${CONTEXT:-262144}
MAX_TOKENS=${MAX_TOKENS:-98304}
NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
NCCL_IB_HCA=${NCCL_IB_HCA:-}
NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-}
DRY_RUN=${DRY_RUN:-0}
ROCE=${ROCE:-1}
ROCE_HCA=${ROCE_HCA:-}
MEMGUARD_HOLD_S=${MEMGUARD_HOLD_S:-4}
MEMGUARD_FLOOR_GIB=${MEMGUARD_FLOOR_GIB:-3}
MEMGUARD_FREE_GIB=${MEMGUARD_FREE_GIB:-0}
ZIG_SHA256=${ZIG_SHA256:-}
ABLATE_FILE=${ABLATE_FILE:-}

RANK0_NAME=glm53-r0
RANK1_NAME=glm53-r1
LOG_DIR="$SCRIPT_DIR/logs"

# Docker runtime, mounts, and entrypoint from the release environment.
DOCKER_ARGS=(
  --gpus all
  --network host
  --ipc=host
  --device /dev/infiniband
  --ulimit memlock=-1
  -v "$MODEL_DIR:$MODEL_DIR:ro"
  -v "$DRAFTER_DIR:/drafter:ro"
  -v "$CACHE_DIR:/digest-cache"
  -v "$CACHE_DIR:$CACHE_DIR"
  -e HOME=/digest-cache
  --entrypoint /tfz/zig-out/native/bin/tensorfold-native
)

# NCCL transport settings are discovered from the route to the other rank.
DOCKER_ARGS+=(
  -e TF_GLM_WORLD=2
  -e NCCL_COMM_ID="$RANK0_ADDR:29600"
  -e NCCL_BUFFSIZE=1048576
  -e NCCL_MAX_NCHANNELS=2
  -e NCCL_MIN_NCHANNELS=1
  -e NCCL_NVLS_ENABLE=0
)

# GLM-5.3 runtime, cache, and serving environment from the release environment.
DOCKER_ARGS+=(
  -e TF_GLM53_PROMPT_MLA=2
  -e TF_GLM53_SMLA_P=1
  -e TF_GLM53_SIDE=1
  -e TF_GLM53_SCRATCH_OVERLAY=1
  -e TF_GLM53_INDEX_ROWS=16
  -e TF_GLM53_PIECE_PIPE=1
  -e TF_GLM53_CHUNK_ROWS=2048
  -e TF_GLM53_RUNTIME_RESERVE_MIB=2560
  -e TF_GLM53_GRAPH_MIB=32
  -e TF_GLM53_DRAFTER_SHARE=1
  -e TF_GLM53_RESIDENT=1
  -e TF_GLM53_MEM_TRACE=1
  -e TF_GLM53_DECODE_MLA=3
  -e TF_GLM53_MOE_PIPE=1
  -e TF_GLM53_INDEX_MERGE=2
  -e TF_GLM53_SMLA_P3=1
  -e TF_GLM53_EXPERT_ROWS=1024
  -e TF_GLM53_PROMPT_EXPERTS=10
  -e TF_GLM53_DECODE_UNION=1
  -e TF_GLM53_PRE_FUSE=1
  -e TF_GLM53_PREP_FUSE=1
  -e TF_GLM53_MOE_POST=1
  -e "TF_ROCE=$ROCE"
  -e TF_GLM53_EXPERT_ROWS_ADAPT=2048
  -e TF_GLM53_DRAFT_CHANCES=1
  -e TF_GLM53_GRAPH_BUDGET_MIB=1024
  -e TF_GLM53_SERVE_MARGIN_MIB=512
  -e TF_GLM53_MEM_FLOOR_GIB=3
  -e TF_GLM53_WARM_TURNS=1
  -e TF_GLM53_LONG_CTX_ROWS=200000
  -e TF_GLM53_MEMSLIM=1
  -e TF_GLM53_WEIGHTS_VMM=1
  -e TF_GLM53_LIVE_COSTS=1
  -e TF_GLM53_COPYD=1
  -e TF_GLM53_FINE_REUSE=1
  -e TF_GLM53_PREFIX_CACHE=1
  -e TF_GLM53_PREFIX_CACHE_DIR="$CACHE_DIR"
  -e TF_GLM53_PREFIX_CACHE_GIB=16
  -e TF_GLM53_PREFIX_CACHE_FREE_GIB=47
  -e TF_GLM53_PREFIX_CACHE_THREADS=4
  -e TF_GLM53_PREFIX_CACHE_MIN_FREE_GIB=3
  -e TF_GLM53_PREFIX_CACHE_MIN_AVAIL_GIB=4
  -e TF_GLM53_SIDE_ROWS=16384
  -e TF_GLM53_ATTN_PIPE=1
  -e TF_GLM53_COPY_ROWS=16
  -e TF_GLM53_DECODE_UNION16=1
  -e TF_GLM53_VERIFY_MAX=16
)

# Ablation hook is opt-in. When ABLATE_FILE is set, both ranks mount the same vectors file
# at the same host path, set TF_GLM53_ABLATE_FILE, and start with --allow-ablate.
# When unset, nothing changes (bit-identical default).
if [[ -n "$ABLATE_FILE" ]]; then
  DOCKER_ARGS+=(
    -v "$ABLATE_FILE:$ABLATE_FILE:ro"
    -e "TF_GLM53_ABLATE_FILE=$ABLATE_FILE"
  )
fi

serve_args=(
  serve "$MODEL_DIR"
  --context "$CONTEXT"
  --max-tokens "$MAX_TOKENS"
  --drafter /drafter
  --host 127.0.0.1
  --port "$PORT"
)
if [[ -n "$ABLATE_FILE" ]]; then
  serve_args+=(--allow-ablate)
fi

# Prints "address netdev hca gid_index" for this node. The route to the peer wins when it runs over RDMA.
# Sparks cabled directly often have only link-local addresses there, so otherwise take the first active RoCE v2 port.
# shellcheck disable=SC2016
LINK_PROBE='
peer=$1
gid_for() { # hca: first RoCE v2 GID that holds an IPv4 address, printed as "index a.b.c.d netdev"
  local f g t
  for f in /sys/class/infiniband/$1/ports/1/gids/*; do
    g=$(cat "$f" 2>/dev/null) || continue
    case "$g" in 0000:0000:0000:0000:0000:ffff:*) ;; *) continue;; esac
    t=$(cat "/sys/class/infiniband/$1/ports/1/gid_attrs/types/${f##*/}" 2>/dev/null) || continue
    case "$t" in *v2*) ;; *) continue;; esac
    g=${g#0000:0000:0000:0000:0000:ffff:}; g=${g//:/}
    printf "%s %d.%d.%d.%d %s\n" "${f##*/}" "0x${g:0:2}" "0x${g:2:2}" "0x${g:4:2}" "0x${g:6:2}" "$(cat "/sys/class/infiniband/$1/ports/1/gid_attrs/ndevs/${f##*/}" 2>/dev/null)"
    return 0
  done
  return 1
}
hca=
case "$peer" in 169.254.*) ;; *)  # a link-local peer says nothing about which rail: both ranks then take the same first port
  dev=$(ip -o -4 route get "$peer" 2>/dev/null | sed -n "s/.* dev \([^ ]*\).*/\1/p")
  hca=$(ls "/sys/class/net/$dev/device/infiniband" 2>/dev/null | head -1);;
esac
if [ -n "$hca" ] && r=$(gid_for "$hca"); then set -- $r; printf "%s %s %s %s\n" "$2" "$3" "$hca" "$1"; exit 0; fi
for d in $(ls -d /sys/class/infiniband/* | LC_ALL=C sort -r); do
  hca=${d##*/}
  case "$(cat "$d/ports/1/state" 2>/dev/null)" in *ACTIVE*) ;; *) continue;; esac
  if r=$(gid_for "$hca"); then set -- $r; printf "%s %s %s %s\n" "$2" "$3" "$hca" "$1"; exit 0; fi
done
exit 1
'

# Comma-joined active RDMA devices on this node, for the one-shot RoCE all-gather.
# shellcheck disable=SC2016
ACTIVE_HCAS='for d in /sys/class/infiniband/*; do grep -q ACTIVE "$d/ports/1/state" 2>/dev/null && echo "${d##*/}"; done | sort -f | paste -sd, -'

discover_link() {
  local remote_info remote_addr local_info local_addr local_dev local_hca local_gid
  remote_info=$(ssh_peer bash -s -- "$RANK0_ADDR" <<<"$LINK_PROBE") || { printf 'Rank 1 has no active RoCE v2 port. Set NCCL_* overrides in .env if needed.\n' >&2; return 1; }
  read -r remote_addr _ _ _ <<<"$remote_info"
  local_info=$(bash -c "$LINK_PROBE" _ "$remote_addr") || { printf 'Rank 0 has no active RoCE v2 port. Set NCCL_* overrides in .env if needed.\n' >&2; return 1; }
  read -r local_addr local_dev local_hca local_gid <<<"$local_info"
  read -r _ RANK1_IFACE RANK1_HCA RANK1_GID <<<"$remote_info"
  RANK0_IFACE=${NCCL_SOCKET_IFNAME:-$local_dev}; RANK1_IFACE=${NCCL_SOCKET_IFNAME:-$RANK1_IFACE}
  RANK0_HCA=${NCCL_IB_HCA:-$local_hca}; RANK1_HCA=${NCCL_IB_HCA:-$RANK1_HCA}
  RANK0_GID=${NCCL_IB_GID_INDEX:-$local_gid}; RANK1_GID=${NCCL_IB_GID_INDEX:-$RANK1_GID}
  if [[ "$RANK0_HCA" == - || "$RANK1_HCA" == - || "$RANK0_GID" == - || "$RANK1_GID" == - || -z "$RANK0_IFACE" || -z "$RANK1_IFACE" ]]; then
    printf 'Link discovery found no complete RDMA/RoCE v2 route (rank0 %s/%s gid %s; rank1 %s/%s gid %s). Set NCCL_SOCKET_IFNAME, NCCL_IB_HCA, and/or NCCL_IB_GID_INDEX in .env.\n' "$RANK0_IFACE" "$RANK0_HCA" "$RANK0_GID" "$RANK1_IFACE" "$RANK1_HCA" "$RANK1_GID" >&2
    return 1
  fi
  printf 'Link: rank0 %s (%s) HCA %s GID %s <-> rank1 %s (%s) HCA %s GID %s\n' "$local_addr" "$RANK0_IFACE" "$RANK0_HCA" "$RANK0_GID" "$remote_addr" "$RANK1_IFACE" "$RANK1_HCA" "$RANK1_GID"
}

ssh_peer() {
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$PEER_SSH" "$@"
}

container_exists_local() {
  docker container inspect "$1" >/dev/null 2>&1
}

assert_no_named_containers() {
  local node=$1 name
  shift
  for name in "$@"; do
    if docker container inspect "$name" >/dev/null 2>&1; then
      printf '%s already exists on %s\n' "$name" "$node" >&2
      return 1
    fi
  done
}

runtime_usable() {
  local runtimes
  runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null) || return 1
  [[ "$runtimes" == *'"nvidia"'* || -x /usr/bin/nvidia-container-runtime-hook || -x /usr/bin/nvidia-ctk ]]
}

preflight() {
  local mode=${1:-} system arch command available parked peer_mem peer_available peer_parked local_gpu peer_gpu
  system=$(uname -s 2>/dev/null) || system=
  arch=$(uname -m 2>/dev/null) || arch=
  if [[ "$system" != Linux || "$arch" != aarch64 ]]; then
    printf 'Rank 0 must run Linux on aarch64; use an aarch64 Linux host.\n' >&2
    return 1
  fi
  for command in docker curl python3 ssh ip awk sha256sum nvidia-smi; do
    if ! command -v "$command" >/dev/null 2>&1; then
      printf 'Rank 0 is missing %s; install it and retry.\n' "$command" >&2
      return 1
    fi
  done
  if ! docker info >/dev/null 2>&1; then
    printf 'Docker is unavailable on rank 0; start Docker and grant this user access.\n' >&2
    return 1
  fi
  if ! runtime_usable; then
    printf 'Rank 0 has no usable NVIDIA Docker runtime; install nvidia-container-toolkit and configure Docker.\n' >&2
    return 1
  fi
  if ! ssh_peer true >/dev/null 2>&1; then
    printf 'Key-based SSH to PEER_SSH failed; set up key login from rank 0 to PEER_SSH.\n' >&2
    return 1
  fi
  if ! ssh_peer bash -s <<'REMOTE' >/dev/null 2>&1
[[ "$(uname -m)" == aarch64 ]] && docker info >/dev/null 2>&1 && {
  runtimes=$(docker info --format '{{json .Runtimes}}' 2>/dev/null) || exit 1
  [[ "$runtimes" == *'"nvidia"'* || -x /usr/bin/nvidia-container-runtime-hook || -x /usr/bin/nvidia-ctk ]]
}
REMOTE
  then
    printf 'Rank 1 must run aarch64 Linux with Docker and NVIDIA GPU runtime access; install and configure Docker with nvidia-container-toolkit.\n' >&2
    return 1
  fi
  if ! local_gpu=$(nvidia-smi -L 2>&1); then
    printf 'nvidia-smi -L failed on rank 0; install or repair the NVIDIA driver.\n' >&2
    return 1
  fi
  if ! grep -qi GB10 <<<"$local_gpu"; then
    printf 'Warning: rank 0 nvidia-smi output does not identify a GB10 GPU.\n' >&2
  fi
  if ! peer_gpu=$(ssh_peer nvidia-smi -L 2>&1); then
    printf 'nvidia-smi -L failed on rank 1; install or repair the NVIDIA driver.\n' >&2
    return 1
  fi
  if ! grep -qi GB10 <<<"$peer_gpu"; then
    printf 'Warning: rank 1 nvidia-smi output does not identify a GB10 GPU.\n' >&2
  fi
  if [[ "$mode" == start ]]; then
    if [[ ! -f "$MODEL_DIR/config.json" || ! -d "$DRAFTER_DIR" ]]; then
      printf 'Rank 0 model config or drafter directory is missing; weights and drafter must be at the same path on both nodes.\n' >&2
      return 1
    fi
    if ! ssh_peer test -f "$MODEL_DIR/config.json" || ! ssh_peer test -d "$DRAFTER_DIR"; then
      printf 'Rank 1 model config or drafter directory is missing; weights and drafter must be at the same path on both nodes.\n' >&2
      return 1
    fi
    read -r available parked _ <<<"$(mem_avail_gb)"
    printf 'Rank 0 available memory: %s GiB (parked: %s GiB; minimum: %s GiB).\n' "$available" "$parked" "$MIN_AVAIL_GIB"
    if [[ -z "$available" ]] || ! awk -v a="$available" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a >= m)}'; then
      printf 'Rank 0 has less than MIN_AVAIL_GIB available; stop other work or set a smaller MIN_AVAIL_GIB in .env.\n' >&2
      return 1
    fi
    peer_mem=$(ssh_peer bash -s <<<"$(declare -f mem_counting mem_avail_gb); mem_avail_gb") || {
      printf 'Could not read available memory on rank 1; check /proc/meminfo and /proc/zoneinfo access.\n' >&2
      return 1
    }
    read -r peer_available peer_parked _ <<<"$peer_mem"
    printf 'Rank 1 available memory: %s GiB (parked: %s GiB; minimum: %s GiB).\n' "$peer_available" "$peer_parked" "$MIN_AVAIL_GIB"
    if [[ -z "$peer_available" ]] || ! awk -v a="$peer_available" -v m="$MIN_AVAIL_GIB" 'BEGIN {exit !(a >= m)}'; then
      printf 'Rank 1 has less than MIN_AVAIL_GIB available; stop other work or set a smaller MIN_AVAIL_GIB in .env.\n' >&2
      return 1
    fi
    if command -v ss >/dev/null 2>&1 && ss -ltn | awk -v p=":$PORT " 'index($0, p) {found=1} END {exit !found}'; then
      printf 'Port %s is already listening on rank 0; stop that listener or set a different PORT in .env.\n' "$PORT" >&2
      return 1
    fi
  fi
}

start_rank1() {
  local -a command
  local command_text
  command=(docker run -d --name "$RANK1_NAME" "${DOCKER_ARGS[@]}" "${rank1_link_args[@]}" -e TF_GLM_RANK=1 "$RUNTIME_IMAGE" "${serve_args[@]}")
  printf -v command_text '%q ' "${command[@]}"
  ssh_peer "$command_text" >/dev/null
}

start_rank0() {
  docker run -d --name "$RANK0_NAME" "${DOCKER_ARGS[@]}" "${rank0_link_args[@]}" -e TF_GLM_RANK=0 "$RUNTIME_IMAGE" "${serve_args[@]}" >/dev/null
}

container_status() {
  local node=$1 name=$2 status
  if [[ "$node" == rank0 ]]; then
    if ! container_exists_local "$name"; then printf '%s: absent\n' "$name"; return 0; fi
    status=$(docker inspect --format '{{.State.Status}}' "$name")
  else
    status=$(ssh_peer docker inspect --format '{{.State.Status}}' "$name" 2>/dev/null) || status=absent
  fi
  printf '%s: %s\n' "$name" "$status"
}

stop() {
  local stamp peer_present=false local_present=false
  stamp=$(date +%Y%m%d-%H%M%S)
  mkdir -p "$LOG_DIR"
  if ssh_peer docker container inspect "$RANK1_NAME" >/dev/null 2>&1; then peer_present=true; fi
  if container_exists_local "$RANK0_NAME"; then local_present=true; fi
  if [[ "$peer_present" == true ]]; then
    ssh_peer docker stop -t 60 "$RANK1_NAME" >/dev/null || true
  fi
  if [[ "$local_present" == true ]]; then
    docker stop -t 60 "$RANK0_NAME" >/dev/null || true
  fi
  if [[ "$peer_present" == true ]]; then
    ssh_peer docker logs "$RANK1_NAME" >"$LOG_DIR/rank1-$stamp.log" 2>&1 || true
  fi
  if [[ "$local_present" == true ]]; then
    docker logs "$RANK0_NAME" >"$LOG_DIR/rank0-$stamp.log" 2>&1 || true
  fi
  if [[ "$peer_present" == true ]]; then
    ssh_peer docker rm "$RANK1_NAME" >/dev/null || true
  fi
  if [[ "$local_present" == true ]]; then
    docker rm "$RANK0_NAME" >/dev/null || true
  fi
  printf 'Stopped containers. Logs: %s\n' "$LOG_DIR"
}

wait_ready() {
  local deadline stamp rank0_log rank1_log models model response reply
  deadline=$((SECONDS + 1200))
  while ((SECONDS < deadline)); do
    if ! container_exists_local "$RANK0_NAME" || [[ $(docker inspect --format '{{.State.Running}}' "$RANK0_NAME") != true ]]; then
      printf 'rank 0 exited before the server became ready\n' >&2
      stamp=$(date +%Y%m%d-%H%M%S)
      mkdir -p "$LOG_DIR"
      rank0_log="$LOG_DIR/rank0-$stamp.log"
      rank1_log="$LOG_DIR/rank1-$stamp.log"
      docker logs "$RANK0_NAME" >"$rank0_log" 2>&1 || true
      ssh_peer docker logs "$RANK1_NAME" >"$rank1_log" 2>&1 || true
      if grep -q MemoryBudget "$rank0_log" "$rank1_log"; then
        printf 'Not enough memory for CONTEXT=%s on this setup. Stop other work, or set a smaller CONTEXT in .env, for example 131072.\n' "$CONTEXT" >&2
      fi
      return 1
    fi
    if curl --fail --silent --show-error --max-time 5 "http://127.0.0.1:$PORT/v1/models"; then
      printf '\n'
      models=$(curl --fail --silent --show-error --max-time 10 "http://127.0.0.1:$PORT/v1/models") || {
        printf 'Could not read the model id from /v1/models for the readiness smoke test.\n' >&2
        return 1
      }
      model=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["data"][0]["id"])' "$models") || {
        printf 'Could not parse a model id from /v1/models for the readiness smoke test.\n' >&2
        return 1
      }
      response=$(curl --fail --silent --show-error --max-time 60 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        --data-binary "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Reply with exactly: ready"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":False}}))' "$model")") || {
        printf 'Readiness chat request failed.\n' >&2
        return 1
      }
      reply=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["choices"][0]["message"].get("content") or "")' "$response") || {
        printf 'Could not parse the readiness chat response.\n' >&2
        return 1
      }
      if [[ -z "$reply" ]] || ! python3 -c 'import sys; s=sys.argv[1]; raise SystemExit(1 if len(s)>1 and len(set(s))==1 else 0)' "$reply"; then
        printf 'Readiness smoke test returned empty or repeated-character content.\n' >&2
        return 1
      fi
      printf 'Readiness reply: %s\n' "$reply"
      return 0
    fi
    sleep 10
  done
  printf 'Timed out waiting for /v1/models after 20 minutes\n' >&2
  return 1
}

start() {
  if [[ "$DRY_RUN" != 1 ]]; then
    preflight start || return 1
  fi
  discover_link
  rank0_link_args=(-e "NCCL_SOCKET_IFNAME=$RANK0_IFACE" -e "NCCL_IB_HCA=$RANK0_HCA" -e "NCCL_IB_GID_INDEX=$RANK0_GID")
  rank1_link_args=(-e "NCCL_SOCKET_IFNAME=$RANK1_IFACE" -e "NCCL_IB_HCA=$RANK1_HCA" -e "NCCL_IB_GID_INDEX=$RANK1_GID")
  if [[ "$ROCE" == 1 ]]; then
    local hca0 hca1
    hca0=${ROCE_HCA:-$(bash -c "$ACTIVE_HCAS")}
    hca1=${ROCE_HCA:-$(ssh_peer bash -c "'$ACTIVE_HCAS'")}
    [[ -n "$hca0" && -n "$hca1" ]] || { printf 'No active RDMA device for RoCE. Set ROCE=0 in .env to use NCCL.\n' >&2; return 1; }
    rank0_link_args+=(-e TF_GLM53_COLLECTIVE=roce -e "TF_ROCE_HCA=$hca0")
    rank1_link_args+=(-e TF_GLM53_COLLECTIVE=roce -e "TF_ROCE_HCA=$hca1")
  fi
  RUNTIME_IMAGE=$(image_hash_tag)
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Plan: start rank 1 then rank 0; image %s; hold memory floor %.1f GiB for %s s.\n' "$RUNTIME_IMAGE" "$MEMGUARD_FLOOR_GIB" "$MEMGUARD_HOLD_S"
    printf 'rank 1: '; printf '%q ' docker run -d --name "$RANK1_NAME" "${DOCKER_ARGS[@]}" "${rank1_link_args[@]}" -e TF_GLM_RANK=1 "$RUNTIME_IMAGE" "${serve_args[@]}"; printf '\n'
    printf 'rank 0: '; printf '%q ' docker run -d --name "$RANK0_NAME" "${DOCKER_ARGS[@]}" "${rank0_link_args[@]}" -e TF_GLM_RANK=0 "$RUNTIME_IMAGE" "${serve_args[@]}"; printf '\n'
    return 0
  fi
  assert_no_named_containers rank0 "$RANK0_NAME" "$RANK1_NAME"
  if ssh_peer docker container inspect "$RANK0_NAME" >/dev/null 2>&1 || ssh_peer docker container inspect "$RANK1_NAME" >/dev/null 2>&1; then
    printf 'A named GLM-5.3 container already exists on rank 1\n' >&2
    return 1
  fi
  # The GPU shares DRAM with the page cache, so clear it first. Needs no root; fails closed.
  python3 "$SCRIPT_DIR/reclaim.py" --target-free-gib 100 --fadvise "$MODEL_DIR" "$DRAFTER_DIR" "$CACHE_DIR"
  ssh_peer python3 - --target-free-gib 100 --fadvise "$MODEL_DIR" "$DRAFTER_DIR" "$CACHE_DIR" < "$SCRIPT_DIR/reclaim.py"
  if ! ssh_peer docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
    printf 'Rank 1 does not have image %s; run "%s image" first.\n' "$RUNTIME_IMAGE" "$SCRIPT_DIR/run.sh" >&2
    return 1
  fi
  mkdir -p "$CACHE_DIR" "$LOG_DIR"
  start_rank1
  start_rank0
  start_pair_guard
  printf 'Started rank 1, then rank 0. Run "%s wait" to check readiness.\n' "$SCRIPT_DIR/run.sh"
}

image_hash_tag() {
  local image_id
  if ! image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null); then
    if [[ "$DRY_RUN" == 1 ]]; then printf '%s:<image-hash-after-build>\n' "$IMAGE"; return 0; fi
    printf 'Image %s is missing; run "%s image" first.\n' "$IMAGE" "$SCRIPT_DIR/run.sh" >&2
    return 1
  fi
  printf '%s:%s\n' "$IMAGE" "${image_id#sha256:}"
}

build_image() {
  local runtime_image image_id
  if [[ "$DRY_RUN" != 1 ]]; then
    preflight || return 1
  fi
  [[ -n "$ZIG_SHA256" ]] || { printf 'ZIG_SHA256 is required in .env for the pinned Zig 0.17.0 image build.\n' >&2; return 1; }
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Plan: docker build -f %q --build-arg ZIG_SHA256=<configured SHA-256> -t %q %q\n' "$SCRIPT_DIR/docker/Dockerfile" "$IMAGE" "$ENGINE_DIR"
    printf 'Plan: docker save <content-addressed image> | ssh %q docker load\n' "$PEER_SSH"
    return 0
  fi
  docker build -f "$SCRIPT_DIR/docker/Dockerfile" --build-arg "ZIG_SHA256=$ZIG_SHA256" -t "$IMAGE" "$ENGINE_DIR"
  image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
  runtime_image="$IMAGE:${image_id#sha256:}"
  docker tag "$IMAGE" "$runtime_image"
  docker save "$runtime_image" | ssh_peer docker load
  printf 'Built and transferred %s\n' "$runtime_image"
}

start_pair_guard() {
  local guard
  guard="$(declare -f mem_counting)
$(cat <<'GUARD'
peer=$1; r1=$2; r0=$3; floor=$4; free_floor=$5; hold=$6
since_a1=0; since_f1=0; since_a0=0; since_f0=0
while sleep 1; do
    docker container inspect "$r0" >/dev/null 2>&1 || exit 0
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$peer" docker container inspect "$r1" >/dev/null 2>&1 || exit 0
    read -r a0 p0 _ <<<"$(mem_counting)"
    f0=$(awk '$1 == "MemFree:" {printf "%.1f", $2/1048576}' /proc/meminfo)
    f0=$(awk -v f="$f0" -v p="$p0" 'BEGIN {printf "%.1f", f+p}')
    peer_values=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$peer" bash -s <<REMOTE
$(declare -f mem_counting)
read -r a p b <<<"\$(mem_counting)"
f=\$(awk '\$1 == "MemFree:" {printf "%.1f", \$2/1048576}' /proc/meminfo)
awk -v a="\$a" -v p="\$p" -v f="\$f" 'BEGIN {printf "%.1f %.1f\\n", a, f+p}'
REMOTE
) || exit 0
    read -r a1 f1 <<<"$peer_values"
    now=$(date +%s)
    if awk -v a="$a1" -v f="$floor" 'BEGIN {exit !(a < f)}'; then (( since_a1 == 0 )) && since_a1=$now; else since_a1=0; fi
    if awk -v a="$f1" -v f="$free_floor" 'BEGIN {exit !(a < f)}'; then (( since_f1 == 0 )) && since_f1=$now; else since_f1=0; fi
    if awk -v a="$a0" -v f="$floor" 'BEGIN {exit !(a < f)}'; then (( since_a0 == 0 )) && since_a0=$now; else since_a0=0; fi
    if awk -v a="$f0" -v f="$free_floor" 'BEGIN {exit !(a < f)}'; then (( since_f0 == 0 )) && since_f0=$now; else since_f0=0; fi
    if (( (since_a1 > 0 && now - since_a1 >= hold) || (since_f1 > 0 && now - since_f1 >= hold) || (since_a0 > 0 && now - since_a0 >= hold) || (since_f0 > 0 && now - since_f0 >= hold) )); then
      ssh -o BatchMode=yes -o ConnectTimeout=15 "$peer" docker stop -t 30 "$r1" >/dev/null 2>&1 || true
      docker stop -t 30 "$r0" >/dev/null 2>&1 || true
      exit 1
    fi
  done
GUARD
)"
  nohup bash -c "$guard" _ "$PEER_SSH" "$RANK1_NAME" "$RANK0_NAME" "$MEMGUARD_FLOOR_GIB" "$MEMGUARD_FREE_GIB" "$MEMGUARD_HOLD_S" >/dev/null 2>&1 </dev/null &
}

case "${1:-}" in
  check) preflight ;;
  image) build_image ;;
  start) start ;;
  stop) stop ;;
  wait) wait_ready ;;
  status)
    container_status rank1 "$RANK1_NAME"
    container_status rank0 "$RANK0_NAME"
    ;;
  *) printf 'Usage: %s {check|image|start|stop|wait|status}\n' "$SCRIPT_DIR/run.sh" >&2; exit 2 ;;
esac
