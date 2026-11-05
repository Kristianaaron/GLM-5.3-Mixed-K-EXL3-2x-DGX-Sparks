# Credits

Ash (ashhart, TensorFold Labs), creator of TensorFold, and Matt Mastracci (mmastrac), EXLR8.

Also credited: turboderp (ExLlamaV3); Z.ai (GLM-5.3); Red Hat AI and the vllm-project (DSpark and speculators); MiaAI-Lab; and every TensorFold contributor.

Under MiaAI-Lab: image build and link discovery adapted from GLM-5.3-Flash-DGX-Sparks-1104, Apache-2.0.

## License notes

These license attributions follow `THIRD_PARTY_NOTICES.md` in the TensorFold engine repository:

- ExLlamaV3: MIT License, Copyright © 2025 Turboderp.
- The GLM-5.3 tokenizer configuration fixture: the model repository's MIT License. The notices describe RedHatAI's DSpark model as under the GLM-5.3 license, and vllm-project/speculators as Apache-2.0.
- MiaAI-Lab work: the notices identify the Flash Next vision patch as MIT and the GLM-5.3 draft-depth policy patches as Apache-2.0.
- TensorFold-derived components and other upstream work carry the component-specific licenses and notices described in the engine's `THIRD_PARTY_NOTICES.md`; those terms vary by component.

The engine notices do not assign a separate license to every individual credit above. Model weights and the DSpark drafter are downloaded separately and remain subject to their repository terms.

### The engine

| Who | Role in this release | Licence | Source |
|---|---|---|---|
| **Ash (ashhart) and the TensorFold contributors** | TensorFold, the engine our fork builds on | Apache-2.0 (code before 0.6.0 keeps its MIT notice) | [ashhart/TensorFold](https://github.com/ashhart/TensorFold), [credits](https://github.com/ashhart/TensorFold/blob/main/docs/credits.md) |
| **vcruz305** | TensorFold's EXL3 module ([#42](https://github.com/ashhart/TensorFold/pull/42)), the base of the format notes and decoders | Apache-2.0 | TensorFold |
| **jschmied** | Block-scaled FP8 weights on CUDA ([#126](https://github.com/ashhart/TensorFold/pull/126)) | Apache-2.0 | TensorFold |
| **drowzeys** | Full GLM-5.3 on CUDA ranks ([#159](https://github.com/ashhart/TensorFold/pull/159)), including the attention all-gather and merge design | Apache-2.0 | TensorFold |
| **BobClawblaw** | Prompt kernel structure ([#976](https://github.com/ashhart/TensorFold/pull/976)) | Apache-2.0 | TensorFold |
| **jayleaton, akol1** | DeepSeek-V4.1 CUDA sources, from which the GLM-5.3 index kernel is adapted | Apache-2.0 | TensorFold |
| **ajensenwaud** | Shared EXL3 resolver work ([#728](https://github.com/ashhart/TensorFold/pull/728)) | Apache-2.0 | TensorFold |
| **philip-pentatonic, jeidbugs404** | Structured output ([#80](https://github.com/ashhart/TensorFold/pull/80)), GLM tool-call parsing ([#35](https://github.com/ashhart/TensorFold/pull/35)) | Apache-2.0 | TensorFold |
| **bertholomus (Bertholomus AI)** | The full GLM-5.3 family on CUDA tensor-parallel ranks, the base of our port, and its decode side stream and index tile skip | Apache-2.0 | [bertholomus/TensorFold, glm-dsa-tp4](https://github.com/bertholomus/TensorFold/tree/glm-dsa-tp4) |
| **MiaAI-Lab** | Draft depth policy, disk prompt cache design, memory trace and prompt kernel patches, in independent reimplementation | Apache-2.0 | [GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-EXL3-3x-DGX-Sparks-TensorFold) |
| **Luke Alonso and the b12x contributors** | One-shot RoCE gather protocol, in the engine and off in this recipe | Apache-2.0 | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x), [flashinfer fork](https://github.com/local-inference-lab/flashinfer) |
| **NVIDIA** | CUTLASS, for the block-scaled FP8 GEMM host adapter. DGX Spark hardware | BSD-3-Clause | [NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) |
| **XGrammar contributors** | xgrammar v0.2.8, strict tool calls and structured output | Apache-2.0 | [mlc-ai/xgrammar](https://github.com/mlc-ai/xgrammar) |
| **gr8 engine contributors** | E4M3 paired conversion sequence | MIT | [kindlingai/gr8](https://github.com/kindlingai/gr8) |
| **DeepSeek AI, NVIDIA Flare** | Design references only, no code copied: DeepInfer, Flare | Apache-2.0 | [DeepInfer](https://github.com/deepseek-ai/DeepInfer), [Flare](https://github.com/NVIDIA/spark-local-inference) |

The engine's `THIRD_PARTY_NOTICES.md` holds the complete list, with commits and files.

### The drafter

| Who | Role in this release | Licence | Source |
|---|---|---|---|
| **Red Hat AI** | DSpark drafter, used as released and not tuned by us | GLM-5.3 License | [RedHatAI/GLM-5.3-speculator.dspark](https://huggingface.co/RedHatAI/GLM-5.3-speculator.dspark) |
| **vLLM project** | Speculators, the library that drafter was trained with | Apache-2.0 | [vllm-project/speculators](https://github.com/vllm-project/speculators) |
| **Xin Cheng et al., Peking University and DeepSeek-AI** | The DSpark paper | CC BY 4.0 | [arXiv:2607.05147](https://arxiv.org/abs/2607.05147) |
| **Z Lab** | DFlash, the architecture DSpark extends | MIT | [z-lab/dflash](https://github.com/z-lab/dflash) |

**No DFlash2 weights.** The served drafter is Red Hat's DSpark. This release contains no DFlash2 (CC BY-NC-ND) weights or derivatives of them, and no number on this card used them.

### Prior art

**Matt Suiche** published runtime GLP control vectors ([weightless](https://github.com/msuiche/weightless), MIT, and [a pull request](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/pull/35) to the MiaAI-Lab GLM-5.3-Flash recipe). He is prior art for the engine's runtime residual-direction hook. The hook is off by default, no direction files ship with this release, and it was off for every number on this card.
