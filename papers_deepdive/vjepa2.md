# V-JEPA 2 — self-supervised, non-generative video world model that understands, predicts, and plans

**Source:** [paper](https://arxiv.org/abs/2506.09985) · FAIR at Meta (LeCun et al.) · 11 Jun 2025 · [Code](https://github.com/facebookresearch/vjepa2).
**Code:** [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) · MIT · ViT-g encoder ≤1B params; HF [facebook/v-jepa-2 collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6); PyTorch Hub `vjepa2_vit_giant` / action-conditioned `vjepa2_ac_vit_giant`.

## Motivations

Learn a world model "largely by observation" (the LeCun 2022 agenda) from abundant **action-free** internet video, adding only a small amount of action data at the end. Two specific complaints about prior work: (1) interaction-data world models (Dreamer-style) are limited by "the limited availability of real-world interaction data"; (2) action-conditioned **video generation** models "demonstrate limited results in robot execution," emphasizing prediction faithfulness over planning "due to the computational cost of planning by generating video." The JEPA bet: predict in a learned representation space so the model ignores unpredictable pixel detail ("each blade of grass in a field") and keeps only predictable structure.

## Methodology

Stage-wise. **Stage 1 (action-free pretraining):** a **mask-denoising feature-prediction** objective — predict masked video segments *in representation space*, with a stop-gradient EMA target encoder to prevent collapse. No pixels, no noise vector. Scaled to >1M hours video + 1M images, encoder up to 1B params.

**Stage 2 (V-JEPA 2-AC):** **freeze the encoder**; train a 300M-param **block-causal** transformer that autoregressively predicts the **representation of the next frame** conditioned on action + previous states, on <62 h of unlabeled Droid robot video (23k trajectories, successes and failures).

**Stage 3 (planning, no training/reward):** encode goal image to $z_g$; at each step minimize a goal-conditioned **energy** = L1 distance between the imagined future representation and $z_g$:
$$a^\star_{1:T} = \arg\min_{\hat a_{1:T}}\ \big\| \hat z_T(\hat a_{1:T}) - z_g \big\|_1$$
optimized with the **cross-entropy method**, execute first action, re-observe, re-plan (MPC). Deployed zero-shot on Franka arms in two labs, same weights, monocular RGB.

## How it differs / Limitations

| vs prior practice | what they did | prior-work limitation it addresses |
|---|---|---|
| Generative video world models (UniPi, GAIA, Cosmos) | plan by generating pixels | predicts in **representation space** → ignores unpredictable detail, far cheaper |
| Dreamer / TD-MPC | learn latent dynamics from scratch on interaction data | learns dynamics from **internet-scale passive video** first, adds only 62 h actions |
| Task-specific understanding models | one model per task | one frozen encoder → SOTA motion understanding (SSv2 **77.3**), anticipation (EK100 **39.7** R@5, +44% rel.), VQA at 8B (PerceptionTest **84.0**, TempCompass **76.9**) |
| **Stated limitations** (Future Work) | — | predictions only to ~**16 s** horizon (long-horizon needs sub-goals/hierarchy); goals must be **image goals** (no language); scaled only to **1B** params |

## Conclusions

Non-generative, representation-space video pretraining yields representations strong enough for understanding, prediction, and zero-shot planning — you need not generate pixels to get a useful world model. A *frozen* encoder + tiny (62 h) action head plans real robots with no reward and no in-lab data. Representation-space planning is dramatically cheaper than generation: **16 s/action** (V-JEPA 2-AC) vs **4 min/action** for the Cosmos generative baseline (both CEM, single RTX 4090), with higher success across all robot skills.

## Ties to the project

`proof-of-premise` · `valid-probing-target` · `scope-boundary`. The **non-generative (JEPA-world) instantiation** of the project: a *frozen* video encoder's representation space already carries enough dynamics structure that a small action-conditioned predictor reads it out — the JEPA-world analog of "freeze Φ, recover actions cheaply." **Valid probing target = the Stage-1 action-free encoder's latent space** (no actions in training); under the generalized framing you probe that subspace or **sample trajectories within it** (the -AC predictor literally rolls trajectories forward in embedding space). **Out-of-target = V-JEPA 2-AC** — its predictor is *trained on recorded actions*, so reading actions back is circular. The Cosmos 4 min vs 16 s contrast is your efficiency backdrop: pitch latent-space analysis as cheaper than generation-based control.

## Citations

- "action-free joint-embedding-predictive architecture" pretrained on "over 1 million hours of internet video" — Abstract.
- "freeze the video encoder and train a new action-conditioned predictor" — Fig. 1 caption.
- "300M-parameter transformer network employing a block-causal attention mechanism" / "less than 62 hours of unlabeled interaction data from the Droid dataset" — §Introduction.
- "the precise location of each blade of grass in a field" (anti-generation argument) — §Introduction.
- "it takes 4 minutes to compute a single action … with Cosmos" / "V-JEPA 2-AC world model requires only 16 seconds per action" — §4.2.
- "action-free Cosmos model (latent diffusion-7B with continuous tokenizer), which was trained on 20M hours of video" — §4.1.
- "77.3 top-1 accuracy on Something-Something v2" / "39.7 recall-at-5 on Epic-Kitchens-100" / "84.0 on PerceptionTest, 76.9 on TempCompass" — Abstract.
- "focused on tasks requiring predictions up to roughly 16 seconds" / "relies upon tasks specified as image goals" / "scaled V-JEPA 2 models up to a modest 1B parameters" — §Future work.

## References — pure-video & video-generation

- **Cosmos** (Agarwal et al., 2025) — latent-diffusion video world model; the generative planning baseline. **[target]** (its action-free checkpoint).
- **Octo** (Octo Model Team, 2024) — VLA/BC baseline with goal-image conditioning. **[background]**
- **V-JEPA / Revisiting Feature Prediction** (Bardes et al., 2024) — the Stage-1 objective V-JEPA 2 scales. **[target]**
- **Genie** (Bruce et al., 2024) — action-conditioned generative env; cited as the interaction-data generative line. **[background]**
- **Ha & Schmidhuber, 2018 — World Models** — origin of the term. **[background]**
- **DINO-WM** (Zhou et al., 2024) — frozen-features → zero-shot planning; closest cousin to the frozen premise. **[target]**
