# Research Paper Index

Skimmable map of read papers for the **frozen, action-blind video-model dynamics** project. Each entry is the one-line + 6-section view; full extraction lives in the linked deep-dive. Canonical project statement: [phase1_priority.md](phase1_priority.md) · reading frame + rubric: [project_notes.md](project_notes.md).

**Rubric roles:** `proof-of-premise` · `contrast-baseline` · `valid-probing-target` · `scope-boundary` · `method-tool`.

---

### V-JEPA 2 — non-generative video world model that understands, predicts, and plans
[deep-dive](papers_deepdive/vjepa2.md) · arXiv 2506.09985 · [Code](https://github.com/facebookresearch/vjepa2) · **roles:** proof-of-premise · valid-probing-target (Stage-1 encoder) · scope-boundary (-AC)

- **Motivations:** Learn a world model by observation from action-free internet video; avoid the cost of planning by generating pixels.
- **Methodology:** Stage-1 action-free mask-denoising feature prediction (JEPA) → Stage-2 freeze encoder, train 300M action-conditioned predictor on <62 h Droid → Stage-3 energy-minimization planning (CEM/MPC), image goals.
- **How it differs:** Predicts in representation space, not pixels; frozen encoder + tiny action head plans real robots zero-shot.
- **Limitations:** ~16 s horizon; image goals only; scaled only to 1B params.
- **Conclusions:** Non-generative pretraining suffices for planning; 16 s/action vs Cosmos 4 min/action, higher success.
- **Ties to the project:** JEPA-world instantiation — probe the action-blind Stage-1 latent (or sample trajectories in it); the -AC head is out-of-target.

### Cosmos Policy — fine-tune a pretrained video model into a robot policy, no architectural changes
[deep-dive](papers_deepdive/cosmos_policy.md) · arXiv 2601.16163 · [Project](https://research.nvidia.com/labs/dir/cosmos-policy/) · **roles:** contrast-baseline · scope-boundary

- **Motivations:** Leverage video-model spatiotemporal priors for control without multi-stage post-training or new action modules.
- **Methodology:** Latent-frame injection — encode action chunk / proprio / value as extra latent frames in Cosmos-Predict2-2B's latent-diffusion sequence; fully fine-tune; denoise conditioned on clean image frames. Best-of-N value-ranked planning.
- **How it differs:** Single stage, no new architecture; the diffusion objective itself models actions.
- **Limitations:** Planning inference slow (~5 s/chunk); needs substantial rollout data; best-of-N single-layer search.
- **Conclusions:** SOTA LIBERO 98.5%, RoboCasa 67.1%; predicting future state (not just actions) is crucial.
- **Ties to the project:** The "fine-tune + manufacture action interface" foil on both axes; weights out-of-target, the **base Cosmos-Predict2** checkpoint is a valid probing target.

### ViPRA — turn a video-prediction model into a continuous robot policy from actionless videos
[deep-dive](papers_deepdive/vipra.md) · arXiv 2511.07732 · ICLR 2026 · [Project](https://vipra-project.github.io) · **roles:** contrast-baseline · proof-of-premise

- **Motivations:** Learn continuous control from abundant actionless human/robot video without action annotation.
- **Methodology:** NSVQ quantizer learns motion-centric latent actions (perceptual + optical-flow supervision) → co-train LWM to predict future frames + latent actions → chunked flow-matching decoder to continuous actions from 100–200 demos.
- **How it differs:** Models "what changes *and* how"; continuous control via flow matching at up to 22 Hz, not AR-in-latent.
- **Limitations:** Weak on dexterity / precise contact; limited pretraining-data diversity; scaling behavior unknown.
- **Conclusions:** +16% SIMPLER, +13% real-world over strongest continuous-control baselines.
- **Ties to the project:** Closest empirical neighbor; manufactures flow-supervised latents + fine-tunes the backbone — the discover-vs-manufacture contrast; data-efficiency target to beat.

### DreamZero — World Action Models are zero-shot policies *(prior deep-dive)*
[deep-dive](papers_deepdive/dreamzero_vjepa2.md) · arXiv 2602.15922 · [Code](https://github.com/dreamzero0/dreamzero) · **roles:** contrast-baseline · valid-probing-target (base Wan)

- **One-line:** Fine-tune Wan 2.2 I2V diffusion to jointly generate future frames + actions (joint FDM × IDM); 38× speedup → 7 Hz control.
- **Ties to the project:** Closest architectural neighbor and the "train an action head + fine-tune Φ" baseline; the base **Wan2.1-I2V-14B-480P** backbone is the sharpest action-blind probing target.

---

## Reference table

`#` = number of the read papers (of the 3 deep-dived here: V-JEPA 2, Cosmos Policy, ViPRA) that cite the reference. Advisory; counts are presence-based (grep-verified), not raw occurrence. Re-sort as the corpus grows.

| Reference | Cited by | # | Role | Read? |
|---|---|---|---|---|
| Cosmos (base video model) | VJEPA2, ViPRA | 2 | target | partial (Cosmos Policy deep-dive) |
| Genie | VJEPA2, Cosmos, ViPRA | 3 | background | no |
| Dreamer | VJEPA2, Cosmos, ViPRA | 3 | background | no |
| OpenVLA | VJEPA2, Cosmos, ViPRA | 3 | background | no |
| Flow Matching (Lipman 2022) | ViPRA | 1 | mechanism | no |
| LAPA | ViPRA | 1 | background | no |
| UniPi | ViPRA | 1 | background | no |
| LWM-Chat-1M | ViPRA | 1 | target | no |
| DINO-WM | VJEPA2 | 1 | target | no |
| Wan | Cosmos, ViPRA | 2 | target | no |

- Read-next priority (role=`target`, unread, high #): **base Cosmos-Predict2** and **Wan** (both action-blind probing candidates), then **DINO-WM** (frozen-features → planning, the strongest "frozen premise works" anchor).
