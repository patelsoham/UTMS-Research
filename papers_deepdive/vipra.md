# ViPRA — turn a video-prediction model into a continuous robot policy from actionless videos

**Source:** [paper](https://arxiv.org/abs/2511.07732) · CMU / Skild AI / UC Irvine (Routray, Pan, Jain, Bahl, Pathak) · ICLR 2026 (v2 30 Mar 2026) · [Project](https://vipra-project.github.io).
**Code:** [vipra-project.github.io](https://vipra-project.github.io) · models + code released · base policy **LWM-Chat-1M** (Liu et al., 2024) video-language model + its VQ-VAE frame tokenizer.

## Motivations

Robot demos are expensive and embodiment-limited; actionless videos (human + teleop) are abundant but lack action labels. Video-prediction models grasp object dynamics and physics. Question: "Can a video prediction model be transformed into a control policy for physical robots?" Gap in prior latent-action work: methods either (a) "treat pretraining purely as policy learning in latent space, without leveraging video prediction," or (b) "rely on inverse dynamics models or pose tracking to translate predicted frames into actions." ViPRA wants to model **both what changes and how** in one model.

## Methodology

Three stages (Fig. 2):
1. **Latent-action learning.** A neural quantizer (**NSVQ — Noise Substitution in Vector Quantization**) extracts **motion-centric latent actions** $z_t$ between frames, supervised by **perceptual losses + optical-flow consistency** so the latents reflect physically grounded motion (not appearance).
2. **Multimodal pretraining.** A video-language model (LWM-Chat-1M) is co-trained to **jointly predict future observations $o_{t+H}$ and latent-action chunks $z_{t:t+H-1}$** from past frames + task text: $\mathcal{L}_\text{pretrain}=\mathcal{L}_\text{img}+\mathcal{L}_\text{act}$. Uses unlabeled human *and* robot videos → cross-embodiment.
3. **Continuous finetuning.** A **chunked flow-matching decoder** (Lipman 2022) maps latent actions → robot-specific continuous action chunks, using only **100–200 teleoperated demos**. Action chunking amortizes latency → smooth control **up to 22 Hz**.

## How it differs / Limitations

| vs prior practice | what they did | prior-work limitation it addresses |
|---|---|---|
| Latent-action pretraining as AR policy (LAPA, etc.) | autoregressive next-latent in latent space only | also predicts **future pixels** → models "what changes *and* how" |
| Video-gen-then-act (UniPi, AVDC) | generate frames → IDM / optical flow post-hoc | **end-to-end** latent actions, no separate post-hoc IDM |
| Discrete-token policies | VQ tokens + AR | **continuous** control via **flow-matching** decoder, smooth & high-freq (22 Hz) |
| **Stated limitations** | — | struggles with **dexterity / precise contact** (multi-fingered, dual-arm); **limited pretraining-data diversity**; **scaling behavior unknown**; precision grasps (cylindrical cans) fail |

## Conclusions

Motion-centric latent actions learned from large-scale actionless video + video-language pretraining + a flow-matching decoder yield a generalist continuous-control policy with only 100–200 labels: **+16% on SIMPLER**, **+13% on real-world** manipulation over the strongest continuous-control baselines, smooth control to 22 Hz. The latent-action decoder doubles as a predictive world model (sampling latents generates visual rollouts).

## Ties to the project

`contrast-baseline`. The **closest empirical neighbor** and a top baseline — and a clean illustration of the *manufacture* axis we differentiate from. ViPRA **manufactures** its latent action space with a trained **NSVQ quantizer + optical-flow/perceptual supervision**, and **fine-tunes** the LWM backbone (co-trained on the latent-action loss). So on both novelty axes (frozen-vs-trained, discover-vs-manufacture) it is what we are *not* doing: we keep Φ frozen and *discover* structure rather than training a flow-supervised quantizer to *create* it. Its key contrast for your pitch: ViPRA's latents are **flow-supervised intermediate representations**, whereas your claim is that the frozen video model's *own* representation already carries action-relevant structure — no quantizer, no optical-flow loss. ViPRA is also `proof-of-premise` evidence that actionless-video pretraining yields recoverable control signal. Data-efficiency target to beat: it needs 100–200 labels; the frozen-readout pitch is "fewer."

## Citations

- "train a video-language model to predict both future visual observations and motion-centric latent actions" — Abstract.
- "perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior" — Abstract.
- "chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations" — Abstract.
- "Unlike prior latent action works that treat pretraining as autoregressive policy learning, ViPRA explicitly models both what changes and how" — Abstract.
- "16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks" — Abstract.
- "smooth, high-frequency continuous control upto 22 Hz" — Abstract.
- "we build on the instruction-tuned LWM-Chat-1M (Liu et al., 2024) as our base policy" / "LWM's VQ-VAE encoder" — §3.
- "Noise Substitution in Vector Quantization (NSVQ)" — §Fig. 2 / Appendix.
- "struggles with precision grasps such as cylindrical cans" — §Results; "extending to more dexterous domains … introduces new challenges" — §7 Limitations.

## References — pure-video & video-generation

- **LWM-Chat-1M** (Liu et al., 2024) — the video-language backbone ViPRA fine-tunes. **[target]** (base; fine-tuned here, but the base is action-blind).
- **LAPA** (Ye et al., 2024) — latent-action pretraining as AR; the prior-art foil. **[background]**
- **UniPi / video-gen-then-act** (Du et al., 2023) — generate-then-extract lineage. **[background]**
- **Flow Matching** (Lipman et al., 2022) — the continuous decoder objective. **[target]** (machinery).
- **UniVLA** (Prismatic-7B + L1 decoder, DINOv2 latent) / **π0** (PaliGemma-3B + chunked flow) — continuous-control baselines. **[background]**
- **Cosmos / NVIDIA et al., 2025a** — cited among video-prediction backbones. **[target]**
