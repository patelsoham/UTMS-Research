# Cosmos Policy — fine-tune a pretrained video model into a robot policy with no architectural changes

**Source:** [paper](https://arxiv.org/abs/2601.16163) · NVIDIA + Stanford (Kim, Finn, Liang, Song, Liu et al.) · 22 Jan 2026 · [Project](https://research.nvidia.com/labs/dir/cosmos-policy/).
**Code:** [research.nvidia.com/labs/dir/cosmos-policy](https://research.nvidia.com/labs/dir/cosmos-policy/) · code/models/data released · backbone **Cosmos-Predict2-2B-Video2World** (latent video diffusion, EDM-style noise), fully fine-tuned.

> **Note:** this is **Cosmos *Policy*** (an action-conditioned robot policy fine-tuned from Cosmos), *not* the base Cosmos world model (Agarwal 2025, Cosmos-Predict1) that V-JEPA 2 uses as a planning baseline. Different paper, different checkpoint.

## Motivations

Video models learn "temporal causality, implicit physics, and motion patterns" that VLMs (static image-text) lack. Prior video-for-robotics methods extract those priors but "require multiple training stages … and introduce new architectural components, such as separate action diffusers or inverse dynamics models." Unified video-action models avoid the staging but "do not leverage pretrained video models due to their custom design." Goal: adapt a *pretrained* video model into a policy in **one** post-training stage with **no architectural modifications**.

## Methodology

**Latent frame injection.** The base model (Cosmos-Predict2-2B-Video2World) takes an image + text and generates a short single-view video. Cosmos Policy encodes **new modalities — robot proprioception, action chunk, future-state value — as additional latent frames** interleaved into the model's latent-diffusion sequence (e.g. an 11-frame sequence mixing current/future proprio, multi-view images, action chunk, value). The model is **fully fine-tuned** (all weights) to **denoise** the noised new-modality latent frames conditioned on the clean image frames, via the standard video diffusion objective — no separate action head. It jointly predicts (1) an action chunk, (2) future state (proprio + images), (3) value (expected rewards-to-go).

**Planning by best-of-N.** Using the future-state + value predictions: sample N candidate action chunks, imagine each resulting future state, rank by predicted value, execute the highest-value action. Real-robot search uses N=8 on 8 parallel H100s. Ablations show predicting **future state is crucial** (removing it drops RoboCasa 67.1% → 44.4%).

## How it differs / Limitations

| vs prior practice | what they did | prior-work limitation it addresses |
|---|---|---|
| Multi-stage video-for-robotics (separate action diffuser / IDM) | video fine-tune → action module | **single stage**, **no new architecture** — reuse the diffusion objective itself |
| Unified video-action models (UVA, etc.) | custom from-scratch designs | **leverages a pretrained** video model's priors |
| Diffusion policies / VLAs fine-tuned on same demos | scratch or VLM backbone | SOTA: LIBERO **98.5%**, RoboCasa **67.1%**; highest real bimanual score |
| **Stated limitations** | — | planning inference **slow (~5 s/action chunk)** → hard for dynamic tasks; planning needs **substantial rollout data**; only **best-of-N, one search-tree layer** |

## Conclusions

A large pretrained video model can be turned into a SOTA policy by **fully fine-tuning it** and injecting action/state/value as latent frames — no architectural change, the diffusion objective alone captures action distributions. Direct-policy latency is 0.16 s (1 denoising step) to 0.95 s (10 steps) per action chunk on one H100; model-based planning adds cost (~5 s/chunk) but raises success on hard tasks. Predicting future state, not just actions, is what makes it work.

## Ties to the project

`contrast-baseline` · `scope-boundary`. The clearest "do the obvious thing" foil on **both** novelty axes: Cosmos Policy **fully fine-tunes** the base video model (not frozen) **and** **manufactures** an action interface by injecting action latent frames + training on demonstrations. So **Cosmos Policy weights are out-of-target** (action-conditioned → probing them is circular). The **valid probing target is the *base* Cosmos-Predict2-2B-Video2World checkpoint** — action-blind, latent-diffusion, exposes a structured noise/latent interface like Wan. This paper is the strongest evidence that the priors are *usable* for control (proof the dynamics are there), while we bet they are *readable* from the frozen base without the fine-tune. Its slow planning latency is another datapoint for the "generation-for-control is expensive" backdrop.

## Citations

- "fine-tuned from the NVIDIA Cosmos-Predict2-2B video foundation model" / "No architectural changes are made to the base video model" — Fig. 1 caption.
- "a single stage of post-training on the robot demonstration data … with no architectural modifications" — Abstract.
- "directly generate robot actions encoded as latent frames within the video model's latent diffusion process" — Abstract.
- "best-of-N sampling to plan by generating candidate actions, imagining their resulting future states, ranking these states by predicted value" — §Introduction.
- "98.5% and 67.1% average success rates" (LIBERO, RoboCasa) — Abstract.
- "Cosmos-Predict2-2B-Video2World … a latent video diffusion model" — §3.
- "The base Cosmos-Predict2 model is fully fine-tuned (all model weights)" — §Appendix (training).
- "around 5 seconds to produce one action chunk" (planning latency limitation) — §6 Limitations.
- "drop in performance from the final ablation … training the policy to also predict future state is crucial" — Table 5.

## References — pure-video & video-generation

- **Cosmos-Predict2 / NVIDIA et al., 2025** — the base latent video diffusion backbone. **[target]** (action-blind base checkpoint).
- **Wan2.1** (Wan et al., 2025) — named alongside Cosmos-Predict2 as the same model class. **[target]**
- **Policy Steering** (Wang et al., 2025) — world models w/ optical-flow action rep + separate value fn. **[background]**
- **UVA / unified video-action models** (Li et al., 2025a; Zhu et al., 2025) — from-scratch unified models that forgo pretrained priors. **[background]**
- **Genie Envisioner** (Liao et al., 2025) — video-model robot adaptation cited in related work. **[background]**
