# Paper Deep-Dives — DreamZero & V-JEPA 2

Concise extraction in the Main Idea / Prior-Work Limits / Methods / Results / This-Work Limits format.
Full versions with math and citations live in [project_notes.md](../../project_notes.md).

---

## Paper Title: DreamZero ([Paper](https://dreamzero0.github.io) · [Code](https://github.com/dreamzero0/dreamzero))

### Main Idea

- **Motivation**
- Because VLMs are trained on static image-text, the VLAs (which leverage these VLMs) can only generalize semantically (i.e. if "move" was in the data, it could understand "move the coke can to Lionel Messi") and not physically (i.e. if "untie" wasn't in the data, it couldn't understand "untie Lionel Messi's shoes").
- Given Video Models are trained on motion, it means that env dynamics are learned -> if action predictions depend on video predictions, you can leverage these dynamics.
- Note: the paper frames the "video priors give you dynamics" claim as a *hypothesis* validated by the 2× generalization gain, not proven before training.

### Limitations of Prior Work

- **VLAs** (RT-2, OpenVLA, π0, GR00T): BC on top of a VLM -> only semantic generalization, fail on unseen motions; lean on many repeated demos per task.
- **Video-for-robotics** (UniPi, AVDC, RoboDreamer): generate a video, then extract actions *post-hoc* with a separate IDM / optical flow -> loose video-action alignment, error-prone.
- **Earlier joint WAMs** (UVA, GR-2, Unified World Models): joint video+action but trained from scratch or with decoupled / bidirectional denoising -> drift, distorted native frame rate.
- Video diffusion is too slow for closed-loop control -> the standard reason people don't use video models as policies.

### Methods

- Model the *joint* distribution of future video and actions; it factorizes into video prediction (FDM) × inverse dynamics (IDM), learned by one end-to-end network (not two separate models).
- Backbone: autoregressive Diffusion Transformer (DiT) initialized from **Wan 2.2 image-to-video diffusion**, then **fine-tuned (not frozen)**. Inputs: VAE visual context, text encoder, proprio state encoder. Adds only state/action encoders + separate video/action decoders.
- Objective: **flow matching**, with a *shared* denoising timestep across video and action, teacher-forced chunk-wise denoising.
- Inference trick: after each action chunk executes, replace the *predicted* frames in the KV cache with the *real observed* frames -> kills autoregressive compounding error (only possible because a WAM acts in the world).
- **DreamZero-Flash** + system/kernel optimizations -> 38× speedup -> ~7 Hz closed-loop control on 2× GB200.

### Results

- >2× average task progress over SOTA VLAs (GR00T N1.6, π0.5) on task + environment generalization benchmarks.
- After task-specific post-training, +10% average task progress over VLAs.
- Cross-embodiment: 10-20 min of video-only data from another robot/human -> +42% relative on unseen tasks; 30 min of play data adapts to a brand-new robot.
- Data *diversity* beats data *repetition* at equal total hours.
- 38× inference speedup without measurable performance loss.

### Limitations of This Work

- No scaling laws for WAMs yet (size vs. data vs. compute).
- Still heavy: ~7 Hz on 2× GB200 vs. VLAs at 20+ Hz on consumer GPUs.
- Short-horizon memory (~6 s) -> reactive System 1; long-horizon tasks need a System 2 planner.
- Weak on high-precision (sub-cm / mm) tasks, inheriting BC's weakness.
- Implicit IDM accuracy is unquantified -> higher-DOF robots need more play data.

---

## Paper Title: VJEPA2 ([Paper](https://arxiv.org/abs/2506.09985) · [Code](https://github.com/facebookresearch/vjepa2))

### Main Idea

- **Motivation**
- Learn a world model largely from *passive observation* (the LeCun agenda): interaction data is scarce, but internet video is abundant though action-free (you see states, not the actions between them).
- Anti-generation argument: generating pixels wastes capacity on unpredictable detail ("the precise location of each blade of grass"); predict in a *learned representation space* instead, so the model can ignore noise and focus on predictable structure -> cheaper and better for control.

### Limitations of Prior Work

- **Generative video world models** (UniPi, GAIA, Cosmos): plan by *generating pixels* -> minutes per action, and worse control.
- **Dreamer / TD-MPC**: learn latent dynamics *from scratch* on scarce interaction data.
- **Task-specific understanding / anticipation models**: one bespoke model per task, no shared self-supervised backbone.

### Methods

- **Stage 1 (action-free):** mask-denoising in representation space. Encoder + predictor; EMA target encoder + stop-gradient prevent representation collapse. No pixel target, no noise vector. Scaled to 1B+ params on 22M videos (VideoMix22M).
- **Stage 2 (V-JEPA 2-AC):** freeze the encoder; train a 300M block-causal predictor on 62 h of *unlabeled* Droid robot video, conditioned on past frame-representations, actions, and end-effector state.
- **Stage 3 (planning):** encode a goal image to z_g; minimize the L1 energy between the imagined future representation and z_g via the cross-entropy method (CEM), execute first action, re-plan = MPC. Zero-shot on Franka arms, same weights across two labs, no reward.

### Results

- SOTA motion understanding (SSv2 77.3), action anticipation (EK100 39.7 R@5, +44% rel.), and (LLM-aligned) competitive video-QA at 8B scale.
- Planning: **16 s/action, 50-80% pick-&-place** vs. **Cosmos 4 min/action, 0-20%** -> ~15× faster *and* better.
- Zero-shot pick-&-place on real Franka arms in two labs with the same weights, no in-lab data, no reward.

### Limitations of This Work

- Camera-position sensitivity: with no calibration it must infer the action coordinate axis from monocular RGB; ill-posed when the robot base is out of frame.
- Long-horizon planning: autoregressive representation rollouts accumulate error; action search grows exponentially with horizon.
- Image goals only -> language goals need future LLM alignment.
- Doesn't fully solve EK100; verb/noun anticipation errors remain.
