# GLM-5.3 on TensorFold

This repository provides a public run recipe for serving **GLM-5.3 full** across two NVIDIA DGX Sparks. This is the full model, not GLM-5.3-Flash. The Sparks are joined by their 200 GbE ports. Rank 0 serves the local OpenAI-compatible API on loopback; rank 1 joins the model-parallel run.

## Get the files

Download the GLM-5.3 weights into a local directory represented by `[Krishando/GLM-5.3-mixed-K-EXL3](https://huggingface.co/Krishando/GLM-5.3-mixed-K-EXL3)` and the `RedHatAI/GLM-5.3-speculator.dspark` drafter into a local directory. Check the weight and drafter repositories for their terms before downloading. Clone the public TensorFold engine into a local directory represented by `[Kristianaaron/TensorFold](https://github.com/Kristianaaron/TensorFold)`.

Create `.env` beside `run.sh` by copying `.env.example`, then set each path, the Docker image repository, the rank 1 SSH target, the rank 0 fabric address, and the Zig archive SHA-256. Keep the configured engine path and model paths valid on both nodes. The rank 1 path to the built binary must match the rank 0 path so `run.sh` can compare their hashes.

## Requirements

- Two DGX Sparks or GB10 boxes joined by the 200 GbE cable.
- Key-based SSH from rank 0 to rank 1.
- Docker with GPU access on both nodes.
- Weights and drafter at the same path on both nodes.
- At least 110 GiB of available memory on each node.

Tested on one DGX Spark Founders Edition with one ASUS Ascent GX10. Other setups are untested.

## Build the engine

Clone the engine on rank 0 and point `ENGINE_DIR` at it. Rank 1 only needs the image.

```sh
git clone -b glm-5.3-2x-sparks https://github.com/Kristianaaron/TensorFold
```

Build the container from that checkout and transfer it to rank 1:

```sh
./run.sh image
```

The image pins Zig 0.17.0 using the required `ZIG_SHA256` build argument, checks out xgrammar v0.2.8, and runs `zig build install native glm53-bringup fatbins -Dsm=121` followed by `zig build grammar -Dxgrammar=DIR`. It uses the unmodified local engine checkout and tags the resulting image with its content hash before transferring it. The engine checkout must be available locally at `ENGINE_DIR`.

## Run

On rank 0, after the image is built and both nodes are ready:

```sh
cp .env.example .env
# Edit .env with the paths and connection settings for your setup.
./run.sh image
./run.sh check
./run.sh start
./run.sh wait
./run.sh status
```

The server listens on `127.0.0.1` at the configured port, default `8888`. Connect a client on rank 0 or use a separately secured tunnel. `./run.sh stop` stops rank 1 and then rank 0, saves both logs under `logs/`, and removes the containers.

`run.sh start` refuses pre-existing containers with the recipe's rank names, checks that each node has at least 110 GiB of available memory, and checks that rank 1 has the same image. It discovers each node's link interface, RDMA device, and RoCE v2 GID from the route between the ranks. Set `NCCL_SOCKET_IFNAME`, `NCCL_IB_HCA`, or `NCCL_IB_GID_INDEX` in `.env` to override discovery. `DRY_RUN=1` prints the discovered link and both complete rank commands without changing anything. The memory guard stops both ranks if either node stays below `MEMGUARD_FLOOR_GIB` available GiB or `MEMGUARD_FREE_GIB` free GiB for `MEMGUARD_HOLD_S` seconds (defaults 3 GiB, 2 GiB, and 4 seconds).

## Client settings

For GLM-5.3 chat requests, use thinking enabled, maximum effort, temperature `1.0`, `top_p` `0.95`, and `clear_thinking: true`. Keep the effort setting fixed within a chat. Set the client or proxy's stale-request timeout to at least 600 seconds.

## Optional: ablation hook

The engine can apply a rank-1 residual projection per layer and site (`attn`, `ffn`): `r' = r - a * u (u^T r)`, where `u` is a unit direction and `a` is the strength. At `a=1` the component along `u` is removed; at `a=0` the weights are untouched. It is opt-in and lossy.

- Set `ABLATE_FILE` in `.env` to a safetensors file. Leave it empty to run with the hook fully off (default; bit-identical).
- When set, `run.sh` mounts the file read-only at the same host path on both ranks, sets `TF_GLM53_ABLATE_FILE`, and starts `serve` with `--allow-ablate`. The file must exist at the same path on both nodes.
- Vectors: tensors named `ablate.L{L}.attn.r` (`[6144]`, F32 or BF16) and `ablate.L{L}.attn.a` (`[1]`, F32), same under `.ffn`, `L` decimal `0..77`. A site is active only when both its tensors are present; a lone `r` or `a`, a wrong shape or dtype, any other tensor name, or a non-finite value makes the engine refuse to start.
- When vectors are loaded (`--allow-ablate`), the engine auto-ablates every request by default. Send `"ablate": false` per request to opt out. With no `ABLATE_FILE`, nothing is ablated and a request that asks for ablation is rejected.
- Verify with `python3 check.py --ablate`, which sends one auto-ablated chat plus one `ablate: false` opt-out and expects normal replies.

Credits: the container image build and link discovery adapt MiaAI-Lab's Apache-2.0 recipe.

## Known limits

The recipe configures a 262,144-token context window. Expect roughly 4 to 6 GiB of memory headroom at the intended configuration. Actual capacity depends on the model files, runtime, and other processes on both nodes; the launcher applies conservative startup and emergency memory checks.

## Speed on two DGX Sparks (single user)

Measured on the release build, served through the OpenAI-compatible API with a 262k context window. Prompt length is stated per row: short-prompt decode is not long-context decode.

These numbers were measured with `ROCE=0`. The recipe now defaults to `ROCE=1`, which measured +0.9% on decode and +0.3-0.6% on prefill with identical token counts.

Speculative decoding is exact against no drafter on the same weights: seeded replies were identical with the drafter on and off (6 of 6). This is a statement about the drafter only. It says nothing about closeness to the original.

| | What it means | Result |
|---|---|---|
| Decode, short prompts | writing speed with thinking on, T=1.0, top_p 0.95, prompts under 1k tokens | **prose 22.9 / new code 23.1 / code edits 24.2 tok/s** |
| Decode, 92k prompt | real agent turns, thinking on, max effort | **17.6 and 18.9 tok/s** |
| Decode, greedy with an empty thinking block | T=0, short prompts. The server closes `<think>` at once. The hosted API does not offer this mode, and its quality is not measured | prose 18.1 / new code 26.5 / code edits 40.4 tok/s |
| Prefill | how fast it reads your prompt | **629 tok/s** at 2k tokens, **539** at 16k, **about 480** at 92k, **452** at 134k, **411** at 205k |
| Time to first token, short chat | | **0.3 s** |
| Time to first token, cold prompt | nothing reused | **3.1 s** at 2k, **30.6 s** at 16k, **193 s** at 92k, **297 s** at 134k, **500 s** at 205k |
| Time to first token, same chat | only the new message is processed | **0.1-0.2 s**, **0.8 s** at 92k |
| Time to first token, new question on a shared 205k prefix | reused from memory | **1.2 s** |
| Time to first token, 205k prompt after a server restart | reloaded from the prefix cache on disk | **3.4 s** |
| Time to first token, new message after a tool loop | warm turns reread the loop in the background | **0.31 s** at 16k |
| Speculative decoding | DSpark drafter (Red Hat), exact against no drafter | on |

With thinking on, most of a reply is reasoning, so code edits run near prose speed.

## Prefill speed switches (since the first release)

`run.sh` now enables expert-parallel routed experts, 8-bit prefill matmuls and several same-output kernel changes (listed with comments in `run.sh`). Measured with the engine's own benchmark, not through the API: a 16k-token cold prompt, both ranks, mean of 2 runs per row.

| Build | Prefill, 16k cold prompt |
|---|---|
| First release switches | 626 tok/s |
| + expert-parallel routed experts, 8-bit dense prefill projections | 694 tok/s |
| + same-output kernel changes (ROT_W4, GU_MB, W8_M64) | 742 tok/s |
| + 8-bit activations for the routed experts (GU_W8S=2, DOWN_W8S=1), with the host setting below | 820 tok/s |
| + same-output kernel and transport changes (TOPK_FIT, MERGE2_FAST, PIECE_ROWS, DN_EPI, ROT_DD, IQ_FOLD, GATHER_SUM, INDEX_PIPE), NCCL on both RoCE ports | **899 tok/s** |

The served prefill and time-to-first-token rows in the table above were measured on the first release and have not been re-run.

The 899 row's switches do not change the model's output: each was checked bit-identical against the 820 build on the engine's quality gate. `INDEX_PIPE` uses about 1 GiB more GPU memory at 262k context.

**Both RoCE ports for NCCL (optional, +0.6%).** `run.sh` gives NCCL the one RDMA device it discovers. If both QSFP ports are cabled between the Sparks, list both devices in `.env`, for example `NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1` (check the names with `ibv_devices` on each node).

**Decode cost.** Expert-parallel adds about 4 ms to each decode step (about 5%). The decode rows above predate it. With the drafter, at max effort with thinking on, a 40-prompt check measured 22.7 to 22.8 tok/s.

**Quality.** The bit-changing switches were checked against the original model on 24 text windows: the original's top choice matches ours on 86.96% of tokens, against 87.10% for the first release (-0.14 points; tool-call text -0.27, prose -0.22, code and math unchanged). A served check of 40 prompts, 20 of them with tool definitions, found no extra loops, malformed tool calls or leaked tags, and the same tool use (19 of 20) as before.

**Host setting (optional, recommended).** The kernel's background memory compaction can stall one node for about a second at the start of a long prompt. Disabling it on both nodes removed that stall in our runs:

```sh
sudo sysctl -w vm.compaction_proactiveness=0 vm.compact_unevictable_allowed=0 vm.watermark_boost_factor=0
```

This lasts until reboot. To restore the defaults: `sudo sysctl -w vm.compaction_proactiveness=20 vm.compact_unevictable_allowed=1 vm.watermark_boost_factor=15000`.

Expect long turns on large builds. At max effort, GLM-5.3 can reason for 35,000 to 55,000 tokens before it writes a file, and the original does the same. In our end-to-end test, a 61 KB single-file page took 69 minutes through an agent harness.

## Features

- 262k context with decode graphs on.
- Prefix cache on memory and disk for repeat prefixes.
- Fine prompt reuse for the next turn of a chat.
- Sampling with temperature, top_p and seed.
- Strict tool calls and structured output through a grammar (xgrammar v0.2.8).
- Tool-call arguments stream as they are written, so long file writes do not look like a stalled connection.
- Warm turns: after a reply, the server reads the next turn's prompt in the background.
