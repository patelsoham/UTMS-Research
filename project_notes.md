# Project Notes — Paper Deep-Dives & Reading Map

> **Purpose of this file.** This is the *content* companion to [phase1_priority.md](phase1_priority.md). The priority file is the *plan* (what to study, in what order, why). This file is the *substance* — paper extractions, conceptual teaching, and curated follow-up reading. When you read a paper, its notes go here, not in the planning doc.
>
> **Reading frame.** Every note ties back to the project: *unsupervised recovery of inverse dynamics (action structure) from a **frozen, action-blind** pretrained video model, by analyzing — or sampling trajectories within — whatever **representation** that model exposes: the **conditioning noise** of a diffusion/flow model, the **latent embedding space** of a predictive model (e.g. V-JEPA 2), or an **autoregressive (token) rollout**.* The action signal need **not be directly stored** as a static subspace; the method may instead **sample/traverse trajectories through the representation** and read dynamics from how those trajectories move. Building a **toolbox to understand that representation** (so the mapping is findable at all) is a primary deliverable, not a means to an end. The three load-bearing claims are (a) the world model Φ stays **frozen**; (b) the action structure is **recoverable** from Φ's representation without retraining it — by probing a static subspace *or* sampling trajectories within it; and (c) **scope: Φ is a *pure* video model with no action interface**, *not* a video-model-**plus-action-conditioning** system (Genie's LAM, V-JEPA 2-**AC**, DreamZero's action head all *manufacture* an action space — we *discover* one). Diffusion-noise probing is the **sharpest single instantiation** (a continuous structured noise vector as a first-class input), not the whole project. Keep stress-testing all three claims as you read. *(Full canonical statement: the "Research Problem" block in [phase1_priority.md](phase1_priority.md).)*
>
> **How each paper ties to the project (classification rubric).** Tag every deep-dive below with the role(s) it plays, so notes connect to the thesis instead of floating free:
> - **Proof-of-premise** — demonstrates action structure *is* recoverable from video (validates the bet; *not* something we refute).
> - **Contrast / baseline** — *manufactures* a latent action space or *fine-tunes* Φ; what we differentiate against on the **discover-vs-manufacture** and **frozen-vs-trained** axes.
> - **Valid probing target** — an action-blind *frozen checkpoint* we could actually instrument (e.g. Wan2.1-I2V base → noise; base Cosmos → noise/latent; V-JEPA 2 **Stage-1** encoder → latent; a no-LAM AR video model → logits/hidden states).
> - **Scope boundary / out-of-target** — the action-conditioned *sibling* we must **not** probe (V-JEPA 2-**AC**, Cosmos **Policy**, Genie's co-trained **LAM**); probing these is circular.
> - **Method tool** — an interpretability/analysis technique we borrow for the toolbox (probing, SAEs, DINO-WM-style frozen-feature planning).
>
> A single paper can carry several tags — e.g. **V-JEPA 2** is proof-of-premise (-AC works on a frozen encoder) *and* a valid Stage-1 target *and* a scope boundary (its -AC head is out-of-target).

---

## Background Primer (assumes only rudimentary RL)

Definitions you need before the two papers make sense. Read once, refer back as needed.

- **Behavior Cloning (BC).** Supervised imitation: collect (observation, action) pairs from an expert, train a policy $\pi(a \mid o)$ to copy the action. Needs **action labels**. Fails outside the demonstrated distribution (compounding error).
- **VLA — Vision-Language-Action model.** A BC-style policy initialized from a pretrained **Vision-Language Model** (VLM, trained on static image+text). Maps (image, language instruction) → motor action. Inherits *semantic* priors ("what a banana is") but **not** *dynamics* priors ("how an arm moves to grasp it"), because VLMs never saw motion — only still images.
  - **A VLA is built *on top of* a VLM — they are not two separate systems.** Structurally: $\text{VLA} = \underbrace{\text{VLM}}_{\text{web image-text} \to \text{semantics, the }what} + \underbrace{\text{action head}}_{\text{robot demos} \to \text{motion, the }how}$. You take the pretrained VLM (the base "brain" that knows what a coke can is, who Taylor Swift is, what "leftmost" means) and **fine-tune it on robot data** to also emit actions. The "A" is the new part bolted on.
  - **Why the two halves matter.** Each capability is credited to a *different* training source. "A VLA can do *move the coke can to Taylor Swift* **because the VLM knows** who she is" means: the robot data never contained Taylor Swift, so that recognition comes purely from the **VLM half** (web knowledge); the *move* trajectory comes from the **action half** (robot demos). It succeeds only because the two combine.
  - **Where it breaks.** "Untie the shoelace" fails not because the VLM lacks knowledge (it knows what a shoelace is) but because **untying is a *motion***, and (a) it was never in the robot demos and (b) the VLM cannot supply it — the VLM only ever saw *static images*, never motion. The one thing missing (dynamics) is exactly the one thing web image-text pretraining can never provide. *This is the precise gap DreamZero's video backbone — and your frozen video-model dynamics project — aim to fill.*
- **Inverse Dynamics Model (IDM).** Given two consecutive observations $o_t, o_{t+1}$, predict the action $a_t$ that caused the transition: $a_t \approx \mathrm{IDM}(o_t, o_{t+1})$. This is the conceptual heart of LAPO/LAPA and is *exactly* what your project is trying to recover from a frozen model's **representation** — its conditioning noise, latent space, or autoregressive rollout — instead of training a separate module.
- **Forward Dynamics Model (FDM).** The opposite: given $o_t$ and $a_t$, predict $o_{t+1}$. A video-prediction model is a (very expensive, generative) FDM.
- **World Model.** Any model that predicts how the world evolves — future states given current state (and optionally action). A video generator is a *pixel-space* world model; a JEPA is a *representation-space* world model.
- **Diffusion / flow-matching model.** A generative model that turns Gaussian noise into data through iterative denoising. **Flow matching** (the objective both Wan and DreamZero use) is the modern, cleaner cousin of DDPM: it learns a **velocity field** that transports noise $z_0 \sim \mathcal{N}(0,I)$ to data $z_1$ along the straight-line path $z_t = t\,z_1 + (1-t)\,z_0$, with target velocity $v = z_1 - z_0$. *This noise $z_0$ is the object your project wants to probe.*
- **JEPA — Joint-Embedding Predictive Architecture.** A **non-generative** predictive model. Instead of predicting pixels, it predicts the *learned embedding* of a masked/future part of the input. Crucially: **no pixel reconstruction, no noise vector, no generation.** (So the *noise-probing* instantiation doesn't apply to V-JEPA 2 — but under the generalized framing its **latent space is itself a candidate representation** to probe, or to sample trajectories within; see 2.6.)
- **MPC — Model-Predictive Control.** A planning loop: imagine the consequences of candidate action sequences using a world model, pick the best, execute the first action, observe, re-plan. The "imagining" is where the world model is used.
- **CEM — Cross-Entropy Method.** A simple black-box optimizer used inside MPC: sample action sequences from a Gaussian, keep the best-scoring ("elite") few, refit the Gaussian to them, repeat. No gradients required.

---

## Paper 1 — DreamZero: *World Action Models are Zero-shot Policies*

**Source:** [Papers/Understanding_Noise/DreamZero.pdf](Papers/Understanding_Noise/DreamZero.pdf) · NVIDIA GEAR Lab, Feb 2026 (arXiv 2602.15922) · 14B-parameter model, code + weights open-sourced · [Code](https://github.com/dreamzero0/dreamzero).
**Code:** [github.com/dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero) (Apache-2.0). Weights on HF: [GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID) (inference) and [GEAR-Dreams/DreamZero-AgiBot](https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot) (post-train on new embodiments). **Backbone confirmed from the repo:** built on **Wan2.1-I2V-14B-480P** (also a smaller **Wan2.2-TI2V-5B** option for lower VRAM), umt5-xxl text tokenizer — i.e. the exact diffusion noise interface your project would probe is the public Wan checkpoint.

**One-line:** Take a pretrained **image-to-video diffusion** backbone (Wan 2.2), and fine-tune it to jointly generate *future video frames* **and** *robot actions* — a "World Action Model" (WAM) — so that action learning rides on top of web-scale video dynamics priors instead of being learned from scratch via behavior cloning.

### 1.1 Motivations

The complaint is precise and worth internalizing because it is the same gap your project targets:

- **VLAs generalize semantically but not physically.** A VLA can do "move the coke can to Taylor Swift" because the VLM knows who Taylor Swift is and the move-skill was in the robot data. It *fails* at "untie the shoelace" if that **motion** was never demonstrated. VLM priors encode *what* to do, not *how* to execute it with spatial/dynamic precision.
- **Root cause:** VLMs are pretrained on **static image-text**. There is no motion, no dynamics, no "what happens next" in their pretraining. So VLAs cannot inherit spatiotemporal priors.
- **The bet:** Video models *are* trained on motion. A model that predicts future frames has implicitly learned environment dynamics. If you make action prediction *depend on* video prediction, you inherit those dynamics priors "for free."
- **Practical payoff sought:** learn from **diverse, non-repetitive** robot data (not hundreds of repeated demos per task), get **zero-shot** generalization to new tasks/environments, and get **cross-embodiment** transfer (learn a skill from a *different* robot's or a *human's* video).

### 1.2 Methodology

**Core object.** DreamZero models the *joint* distribution of future video and actions, which factors exactly into video-prediction × inverse-dynamics:

$$\underbrace{\pi_\theta(o_{l:l+H},\, a_{l:l+H} \mid o_{0:l}, c, q_l)}_{\text{DreamZero (joint)}} = \underbrace{\pi_\theta(o_{l:l+H}\mid o_{0:l}, c, q_l)}_{\text{video prediction (FDM)}} \cdot \underbrace{\pi_\theta(a_{l:l+H}\mid o_{0:l+H}, q_l)}_{\text{inverse dynamics (IDM)}}$$

where $c$ = language instruction, $q_l$ = proprioceptive state, $o$ = observations, $a$ = actions, $H$ = horizon. **They do not train two separate models** — one end-to-end network denoises both modalities, so the action head is tightly coupled to the predicted visual future.

**Architecture (this is the part to read carefully for your project):**
- Backbone: an **autoregressive Diffusion Transformer (DiT)**, initialized from the **Wan 2.2 image-to-video diffusion** model. Frozen? **No — it is fine-tuned.** (Important contrast with your project; see 1.6.)
- Inputs: visual context via a **VAE**, language via a text encoder, proprioception via a **state encoder**.
- Added params are *minimal*: state encoder, action encoder, and **separate video/action decoders**. The backbone is reused.
- Training objective: **flow matching**. For chunk $k$ at timestep $t_k \in [0,1]$, noisy latents are linear interpolants $z_{t_k}^k = t_k z_1^k + (1-t_k)z_0^k$ (video) and $a_{t_k}^k = t_k a_1^k + (1-t_k)a_0^k$ (action), with $z_0^k, a_0^k \sim \mathcal{N}(0,I)$. The model predicts the **joint velocity** $v^k = [z_1^k, a_1^k] - [z_0^k, a_0^k]$, minimizing $\| u_\theta(\cdot) - v^k \|^2$.
- **Shared denoising timestep** across video and action (unlike other WAMs that decouple the two schedules) — speeds early convergence.
- **Teacher forcing, chunk-wise:** denoise the current chunk conditioned on *clean* previous chunks. Each chunk has a fixed number of latent frames matched to the action horizon — like LLMs training on variable-length token sequences.
- **Inference trick (key advantage):** because control is closed-loop, after each action chunk executes, the *predicted* frames in the KV cache are **replaced with the real observed frames**. This kills the compounding-error problem that plagues autoregressive video generation. This is something a *pure* video generator cannot do — it is unique to WAMs that act in the world.

**System engineering (why it's notable, not just a footnote).** Naive 14B diffusion is ~5.7 s per action chunk — useless for control. They stack (1) algorithmic (**DreamZero-Flash**: decoupled video/action denoising *at inference*, few-step action sampling), (2) system (parallelism, KV + DiT caching, CFG parallelism), and (3) low-level (quantization, CUDA kernels) optimizations for a **38× speedup → 7 Hz** closed-loop control on 2× GB200.

### 1.3 How the methodology differs from prior practice

| Prior approach | What it did | What DreamZero changes |
|---|---|---|
| **VLAs** (RT-2, OpenVLA, π0, GR00T) | BC policy on a VLM, static-image priors | Replaces image priors with **video/dynamics priors**; action prediction is conditioned on a predicted visual future |
| **Video-for-robotics** (UniPi, AVDC, RoboDreamer) | Generate a video, then extract actions *post-hoc* (separate IDM / optical flow) | **Single end-to-end** model; no separate post-hoc extraction → tighter video-action alignment |
| **Earlier joint WAMs** (UVA, GR-2, Unified World Models) | Joint video+action but trained *from scratch* or with *decoupled* denoising schedules, *bidirectional* diffusion | **Autoregressive** (KV-cache, native frame rate, less drift), **shared** timestep, large pretrained backbone, systematic data-diversity study |
| **Latent world models** (Dreamer, TD-MPC2, JEPA) | Learn dynamics from scratch in a compact latent space | Leverages a **pretrained pixel-space video model** that already encodes dynamics from internet video |

**Prior-work limitations it addresses:**
- VLA inability to generalize to unseen *motions* → 2× improvement on task/environment generalization benchmarks.
- The "you need many repeated demos per task" assumption → shows generalist policies can be learned from **diverse, non-repetitive** data of the *same total hours*.
- Slow video-diffusion inference (the standard reason people *don't* use video models for closed-loop control) → 38× speedup to 7 Hz.
- Data hunger of cross-embodiment transfer → 10–20 min of video-only data from another robot/human gives +42% relative on unseen tasks; **30 min** of play data adapts to a brand-new robot.

### 1.4 Limitations (as stated by the authors — §6)

- **No scaling laws yet.** They show "bigger backbone + more diverse data = better," but no principled scaling law for WAMs (size vs. data vs. compute).
- **Still computationally heavy.** 7 Hz on 2× GB200 vs. VLAs at 20+ Hz on consumer GPUs. The iterative denoising + 14B size is the cost. Only viable as a fast "System 1" once smaller backbones get good.
- **Short-horizon memory only (~6 s).** It is a reactive System 1 model; long-horizon tasks need a System 2 planner or much longer context.
- **High-precision tasks.** Inherits BC's weakness at sub-centimeter / millimeter assembly; the breadth-first data strategy underrepresents dense precision demos.
- **Implicit IDM accuracy is unquantified.** Higher-DOF robots need more play data because the visual-future → motor-command mapping grows combinatorially; "quantifying the accuracy of implicit IDMs remains a challenge."

### 1.5 Conclusions

- Policy quality is **fundamentally tied to video-generation quality** — better video prediction → better actions. This reframes "improve the robot" as "improve the video model."
- A pretrained video diffusion backbone + joint video-action objective beats VLAs on the dimensions VLAs are weakest: novel motions, novel environments, cross-embodiment.
- Autoregressive > bidirectional for WAMs (smoother motion, better modality alignment, KV-cache efficiency, closed-loop frame replacement).
- Data **diversity** beats data **repetition** at equal hours.

### 1.6 Ties to your project (read this twice)

> **Rubric role(s):** **Contrast / baseline** (fine-tunes Φ + adds an action head → the *frozen-vs-trained* and *discover-vs-manufacture* foil). Secondary: **valid probing target** via its *base* backbone — Wan2.1-I2V-14B **before** DreamZero's action fine-tuning — which is action-blind. The DreamZero weights themselves are **out-of-target** (action-conditioned).

DreamZero is the **closest architectural neighbor** to your idea, and also the clearest illustration of what you are *not* doing:

- **They fine-tune (un-freeze) the backbone and add explicit action decoders.** Your thesis is the opposite: keep Φ **frozen**, add **no** action module, and instead *read* the action structure out of the conditioning **noise** $z_0$. DreamZero is the "do the obvious thing (train an action head)" baseline you must beat *on novelty*, not necessarily on benchmark score.
- Notice the factorization in Eq. (1.2 above): the action term is literally an **inverse-dynamics** model. DreamZero *learns* that IDM as a decoder. Your claim is that this IDM is *already latent inside the noise subspace* of a frozen Wan-like model — i.e., the same $z_0$ that DreamZero overwrites with action-coupled noise already carries the transition signal before any fine-tuning.
- Their flow-matching noise $z_0^k \sim \mathcal{N}(0,I)$ is *exactly* the vector you want to probe. DreamZero treats it as a nuisance variable to be denoised; you treat it as the carrier of the action. **This contrast is your one-paragraph pitch.**
- **The "bet," and your stronger version of it.** DreamZero's central wager is that large-scale video pretraining *implicitly encodes environment dynamics*, so making action prediction *depend on* video prediction inherits those dynamics priors "for free" — it states this across the abstract ("video as a dense representation of how the world evolves"), the intro ("leverage rich spatiotemporal priors... shifts action learning... to inverse dynamics"), and the "Why WAMs" paragraph ("pretrained video representations that already encode physical dynamics... video prediction serving as an implicit visual planner"), culminating in *"improving robotic capabilities reduces to improving video generation."* **Crucially, DreamZero presents this as a hypothesis** ("We further hypothesize that this encourages better generalization...") and treats its 2× generalization gain as the *evidence* it paid off — it does **not** prove the priors are present a priori. **Your project makes the *stronger* version of the same bet:** not merely "the dynamics priors live somewhere in the video model" (which DreamZero accesses by fine-tuning the whole backbone), but "the *action-relevant slice* of those priors is already concentrated in the conditioning **noise** $z_0$, and is readable **without any fine-tuning** via latent-space analysis of a frozen Φ." DreamZero *pays* (gradient updates to 14B params) to surface the dynamics; you *bet you can read* them out of the noise for free. That is the precise distinction your contribution rests on.
- **Sharpest skeptical question DreamZero raises against you:** if the noise already encoded actions, why does DreamZero need an action decoder + fine-tuning at all? Your answer must be that the structure is *there but entangled*, and the contribution is the **latent-space toolbox** (probing, SVD/subspace alignment, a bottleneck) that disentangles it — not a claim that it is trivially readable.

> **Citations (DreamZero — for this section).** Backing passages in [Papers/Understanding_Noise/DreamZero.pdf](Papers/Understanding_Noise/DreamZero.pdf): Abstract — "video as a dense representation of how the world evolves" (p. 2). Intro — "leverage rich spatiotemporal priors... shifts action learning from dense state–action imitation to inverse dynamics" (p. 2); "static image-text datasets, which limits their ability to inherit spatiotemporal priors" (p. 4). §2.2 *Why WAMs* — "pretrained video representations that already encode physical dynamics... video prediction serving as an implicit visual planner"; "improving robotic capabilities reduces to improving video generation" (p. 5). §3.1 — hypothesis verb: "We further hypothesize that this encourages better generalization than the conventional practice of training VLA from VLM" (p. 6). IDM factorization: Eq. (1) (p. 6).

---

## Paper 2 — V-JEPA 2: *Self-Supervised Video Models Enable Understanding, Prediction and Planning*

**Source:** [Papers/Understanding_Noise/VJEPA2.pdf](Papers/Understanding_Noise/VJEPA2.pdf) · FAIR at Meta (LeCun et al.), Jun 2025 (arXiv 2506.09985) · [Code](https://github.com/facebookresearch/vjepa2).
**Code:** [github.com/facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) (MIT). Checkpoints on HF ([facebook/v-jepa-2 collection](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6)) and PyTorch Hub (`vjepa2_vit_giant`; action-conditioned via `vjepa2_ac_vit_giant`). Includes an energy-landscape planning notebook. Note a newer **V-JEPA 2.1** (Mar 2026) refines the dense-feature recipe — still non-generative, still no noise vector.

**One-line:** Pretrain a **non-generative** video world model that predicts in *representation space* (not pixels) on 1M+ hours of internet video, then post-train a small **action-conditioned** predictor on 62 h of unlabeled robot video, and plan with it zero-shot on real Franka arms. **This is the JEPA paradigm — the principal alternative to the diffusion/generation paradigm, and the key counterexample for your project.**

### 2.1 Motivations

- Learn a **world model largely from observation** (the LeCun 2022 agenda): humans build internal predictive models from passive sensory input, then use them to plan.
- **Interaction data is scarce**; internet video is abundant but **action-free** (you see states, not the actions between them). So learn dynamics from video, add a *tiny* amount of action data only at the end.
- **The anti-generation argument (central to the paper, and to your project):** Generative video models waste capacity modeling *unpredictable pixel detail* — "the precise location of each blade of grass." A JEPA predicts in a **learned representation space**, so it can *ignore* unpredictable detail and focus on *predictable structure* (object trajectories, motion). Cheaper, and arguably more useful for control.

### 2.2 Methodology

**Stage 1 — V-JEPA 2 pretraining (action-free, self-supervised).**
- **Mask-denoising in representation space.** Take a video, drop a subset of patch tokens. An encoder $E_\theta$ (a ViT) encodes the visible tokens; a predictor $P_\phi$ predicts the *representations* of the masked tokens. Objective:
$$\min_{\theta,\phi,\Delta_y}\ \big\| P_\phi(\Delta_y, E_\theta(x)) - \mathrm{sg}(E_{\bar\theta}(y)) \big\|_1$$
where $\mathrm{sg}$ = stop-gradient and $E_{\bar\theta}$ is an **EMA (exponential moving average)** copy of the encoder. The stop-grad + EMA target is what prevents **representation collapse** (the trivial solution where everything maps to the same vector). **There is no pixel target and no noise vector** — the loss is in embedding space.
- Scaling recipe (the empirical contribution): data 2M→22M videos (VideoMix22M), model 300M→1B+ params (ViT-L→ViT-g), longer training (90K→252K iters), higher resolution + longer clips via warmup-constant-decay. 3D-RoPE position encoding, 2×16×16 tubelets.

**Stage 2 — V-JEPA 2-AC (action-conditioned, post-trained).**
- **Freeze the encoder.** Train a new 300M-param **predictor** with **block-causal attention** that autoregressively predicts the *representation* of the next frame conditioned on past frame-representations, **actions**, and end-effector states.
- Trained on **62 hours of unlabeled Droid robot video** (23k trajectories, successes *and* failures). "Unlabeled" = no rewards/task labels; it does use the recorded end-effector actions.

**Stage 3 — Planning (no training, no reward).**
- Given a **goal image**, encode it to $z_g$. At each control step, optimize an action sequence $\hat a_{1:T}$ to **minimize the energy** = L1 distance between the world model's *imagined* future representation $T$ steps ahead and $z_g$:
$$a^\star_{1:T} = \arg\min_{\hat a_{1:T}}\ \big\| \hat z_T(\hat a_{1:T}) - z_g \big\|_1$$
- Optimized with the **cross-entropy method (CEM)**, execute the first action, re-observe, re-plan = **MPC**. Deployed zero-shot on Franka arms in two labs with the *same weights*, no lab-specific data, no reward.

### 2.3 How the methodology differs from prior practice — and the result that matters most for you

| Prior approach | V-JEPA 2's difference |
|---|---|
| **Generative video world models** (UniPi, GAIA, Cosmos) plan by *generating pixels* | Predicts in **representation space** → far cheaper, ignores unpredictable detail |
| **Dreamer/TD-MPC** learn latent dynamics *from scratch* on interaction data | Learns dynamics from **internet-scale passive video** first, then adds 62 h of action data |
| **Task-specific anticipation/understanding models** | One self-supervised encoder → SOTA on motion understanding (SSv2 77.3), action anticipation (EK100 39.7 R@5, +44% rel.), and (aligned with an LLM) video-QA at 8B scale |

**The number to remember.** In their planning table, they benchmark V-JEPA 2-AC directly against **Cosmos** (an action-conditioned **latent-diffusion video generation** world model):

| Planner | Time per action | Pick-&-Place success |
|---|---|---|
| Cosmos (diffusion, generative) | **4 minutes** / action | 0–20% |
| V-JEPA 2-AC (JEPA, representation-space) | **16 seconds** / action | 50–80% |

> Planning *by generating video* is ~15× slower **and** worse. This is the empirical core of the "don't generate pixels to plan" argument — and it is the exact tension DreamZero spent its entire systems section (38× speedup) trying to engineer around.

### 2.4 Limitations (as stated — §4.3)

- **Sensitivity to camera positioning.** With no camera calibration, the model must *infer the action coordinate axis* from monocular RGB. When the robot base isn't in frame this is ill-posed; they hand-tuned camera placement. (A real robustness hole.)
- **Long-horizon planning.** Autoregressive representation rollouts accumulate error; the action search space grows exponentially with horizon. Limits non-greedy tasks (e.g., pick-and-place *without* image sub-goals).
- **Image goals only.** Goals must be specified as a *goal image*; language goals require future LLM alignment.
- **Understanding limits.** Doesn't fully solve EK100; verb/noun anticipation errors remain.

### 2.5 Conclusions

- Self-supervised, **non-generative** video pretraining yields representations strong enough for **understanding, prediction, and zero-shot planning** — you do not need to generate pixels to get a useful world model.
- A *tiny* amount of action data (62 h) on top of a frozen encoder is enough for real-robot planning, with no reward and no in-lab data.
- Representation-space prediction is dramatically more efficient than generation for control.

### 2.6 Ties to your project (this is the important one)

> **Rubric role(s):** **Proof-of-premise** (a *frozen* encoder + tiny action head plans real robots → "freeze the world model, read actions cheaply" is viable) · **valid probing target** (the **Stage-1** action-blind ViT encoder's latent space) · **scope boundary / out-of-target** (the **Stage-2 -AC** predictor is action-trained → probing it is circular).

V-JEPA 2 is the **non-generative (JEPA-world) instantiation of your project** — it bounds the *noise* instantiation while opening a *latent-space* one — and you should be able to articulate this precisely:

- **JEPA has no noise vector — but it still has a representation.** The *noise-probing* instantiation of your method depends on a diffusion/flow model exposing a continuous structured noise $z_0$ as a first-class input; a JEPA does *not* generate, *does not* sample pixels, and has **no conditioning noise**, so that *specific* probe is inapplicable. **Under the generalized framing this is not a dead end:** V-JEPA 2's frozen **latent space** is itself a candidate representation to analyze — you can probe it for an action subspace, or **sample/traverse trajectories within it** (its action-conditioned predictor literally rolls trajectories forward *in embedding space*) and read dynamics from how those trajectories move. So the JEPA family is **out of scope for the noise instantiation but in scope for the general method**. Be precise about which claim you are making — and note this is exactly why the framing now says "video *models*," not "video *diffusion* models" only.
- **But V-JEPA 2-AC validates your underlying premise.** It demonstrates that a *frozen* video encoder's representation space already contains enough dynamics structure that a *small* action-conditioned predictor can be bolted on. That is the JEPA-world analog of your claim that a *frozen* diffusion model's noise space already contains action structure. Use V-JEPA 2-AC as **evidence that "freeze the world model, read out actions cheaply" is a viable paradigm** — you are proposing the diffusion-noise version of the same idea.
- **The efficiency war is your backdrop.** V-JEPA 2's 16 s vs. Cosmos's 4 min, and DreamZero's heroic 38× speedup, both say the same thing: *generating pixels to act is expensive.* Your pitch should pre-empt this — you are **not generating** during deployment; you are doing **representation-space analysis** of a single forward pass (or a short trajectory sampled within the representation), whether that representation is diffusion noise, a latent embedding, or an autoregressive rollout. Position your method as cheaper than generation-based control, closer in spirit to JEPA's representation-space efficiency, while retaining — in the diffusion instantiation — the model's explicit, structured, probe-able noise.

---

## DreamZero vs V-JEPA 2 — side by side

| Axis | DreamZero | V-JEPA 2 |
|---|---|---|
| Paradigm | **Generative** (video diffusion / flow matching) | **Non-generative** (JEPA, representation-space) |
| Predicts | Future **pixels** + actions (jointly) | Future **embeddings** (+ actions in Stage 2) |
| Backbone | Wan 2.2 image-to-video diffusion, **fine-tuned** | ViT-g encoder, **frozen** after pretraining |
| Has a noise vector? | **Yes** ($z_0 \sim \mathcal N(0,I)$, flow-matching) ← *target for the noise instantiation* | **No** — but its **latent space** is a target under the generalized framing |
| Action mechanism | End-to-end joint denoising, action decoder | Action-conditioned predictor + CEM/MPC planning |
| Uses actions? | Yes, co-trained (BC-like signal) | Stage 2 uses recorded actions (62 h) |
| Deployment cost | 7 Hz after 38× optimization (2× GB200) | 16 s/action planning (single RTX 4090) |
| Real-robot result | 2× over VLAs; cross-embodiment transfer | Zero-shot pick-&-place on Franka, image goals |
| Relevance to you | Closest neighbor; the "train an action head" baseline | Bounds your scope; validates "frozen world model" premise |

---

## Curated reference list — "pure video" & "video generation" follow-ups

Extracted from both bibliographies and **organized by the taxonomy question you actually care about** ("what *is* a video model today — diffusion vs flow vs autoregressive generation vs predictive-embedding vs world model?"). Start at the top of each bucket.

> Citation tags: **[DZ]** = cited by DreamZero · **[VJ]** = cited by V-JEPA 2 · **[both]** = cited by both.

### A. Foundations — the generative machinery (diffusion / flow)
*Read these to understand what the "noise vector" is and how flow matching differs from DDPM.*
- **Lipman et al., 2022 — Flow Matching for Generative Modeling** (arXiv 2210.02747) — **[DZ]**. The objective DreamZero/Wan use. **Start here.**
- **Liu et al., 2022 — Rectified Flow / "Flow Straight and Fast"** (arXiv 2209.03003) — **[DZ]**. Straight-line transport; intuition for the linear interpolant $z_t$.
- **Albergo et al., 2023 — Stochastic Interpolants** (arXiv 2303.08797) — **[DZ]**. Unifying framework for flows *and* diffusions — the cleanest "what is the noise carrying" mental model.
- **Ho & Salimans, 2022 — Classifier-Free Guidance** (arXiv 2207.12598) — **[DZ]**. How conditioning is injected into diffusion; relevant to how/where action or instruction conditioning could live.

### B. Video generation backbones (text/image-to-video diffusion)
*The class of model your frozen Φ would be drawn from.*
- **Team Wan, 2025 — Wan: Open large-scale video generative models** — **[DZ]**. DreamZero's actual backbone. Highest priority if you'll instrument a real model.
- **Teng et al., 2025 — MAGI-1: Autoregressive Video Generation at Scale** (arXiv 2505.13211) — **[DZ]**. Autoregressive video diffusion (DreamZero's architectural family).
- **Jin et al., 2024 — Pyramidal Flow Matching for Video** (arXiv 2410.05954) — **[DZ]**. Flow-matching video generation, efficiency-oriented.
- **Gao et al., 2024 — CA2-VDM: causal autoregressive video diffusion w/ cache sharing** (arXiv 2411.16375) — **[DZ]**.
- **Yin et al., 2025 — From Slow Bidirectional to Fast Autoregressive Video Diffusion** (CVPR 2025) — **[DZ]**. The bidirectional-vs-autoregressive question, concretely.
- **Huang et al., 2025 — Self Forcing: bridging train-test gap in autoregressive video diffusion** (arXiv 2506.08009) — **[DZ]**.

### C. Video-generation-as-policy (generate video → extract actions)
*Closest empirical neighbors to "use a video model for control." This is the lineage your project sits in.*
- **Du et al., 2023 — UniPi: Learning Universal Policies via Text-Guided Video Generation** (NeurIPS) — **[both]**. The canonical "generate a video plan, extract actions."
- **Du et al., 2024 — Video Language Planning** (ICLR) — **[both]**.
- **Ko et al., 2024 — Learning to Act from Actionless Videos through Dense Correspondences** (ICLR) — **[DZ]**. Optical-flow action extraction.
- **Zhou et al., 2024 — RoboDreamer: compositional world models for robot imagination** (arXiv 2404.12377) — **[DZ]**.
- **Wu et al., 2023/2024 — GR-1 / Unleashing Large-scale Video Generative Pre-training for Manipulation** (ICLR) — **[both]**.
- **Liang et al., 2025 — Video Generators are Robot Policies** (arXiv 2508.00795) — **[DZ]**.
- **Luo et al., 2025 — Solving New Tasks by Adapting Internet Video Knowledge** (ICLR) — **[DZ]**.
- **Jang et al., 2025 — DreamGen: Unlocking Generalization via Video World Models** (CoRL) — **[DZ]**. Synthetic robot data from video models. (NVIDIA, same group as DreamZero)

### D. Joint video+action World Action Models (WAMs) — the direct competitors
*These are exactly the "train a separate action mechanism" baselines your frozen action-blind probing idea aims to bypass.*
- **Zhu et al., 2025 — Unified World Models: Coupling Video and Action Diffusion** (arXiv 2504.02792) — **[both]**.
- **Li et al., 2025 — Unified Video Action Model (UVA)** (arXiv 2503.00200) — **[DZ]**.
- **Kim et al., 2026 — Cosmos Policy: Fine-tuning Video Models for Visuomotor Control** (arXiv 2601.16163) — **[DZ]**. NVIDIA Cosmos as a policy — relevant since Cosmos was your candidate Φ.
- **Liao et al., 2025 — Genie Envisioner: unified world foundation platform** (arXiv 2508.05635) — **[DZ]**.
- **Pai et al., 2025 — mimic-video: Video-Action Models beyond VLAs** (arXiv 2512.15692) — **[DZ]**.
- **Cheang et al., 2024 — GR-2: generative video-language-action model** (arXiv 2410.06158) — **[DZ]**.
- **Zheng et al., 2025 — FLARE: Robot Learning with Implicit World Modeling** (arXiv 2505.15659) — **[both]**.

### E. Predictive-embedding / JEPA world models (the non-generative alternative)
*The paradigm with no noise vector — its **latent space** is the representation you'd probe (or sample trajectories within) under the generalized framing: the JEPA-world instantiation of the project.*
- **Bardes et al., 2024 — V-JEPA / Revisiting Feature Prediction** (arXiv 2404.08471) — **[both]** (DZ cites the V-JEPA arXiv 2402.05065; VJ cites the Revisiting-Feature-Prediction version). The Stage-1 recipe V-JEPA 2 scales.
- **Zhou et al., 2024 — DINO-WM: World Models on Pretrained Visual Features → zero-shot planning** (arXiv 2411.04983) — **[VJ]**. **Very close conceptual cousin to your "frozen features → planning" idea, in the JEPA/feature world.**
- **Tomar et al., 2024 — Video Occupancy Models** (arXiv 2407.09533) — **[VJ]**. Representation-space prediction.
- **Gupta et al., 2022 — MaskViT: Masked Visual Pre-training for Video Prediction** (arXiv 2206.11894) — **[VJ]**.
- **Rajasegaran et al., 2025 — Empirical Study of Autoregressive Pre-training from Videos** (arXiv 2501.05453) — **[VJ]**.

### F. Latent / generative world models for control (broader context)
- **Ha & Schmidhuber, 2018 — World Models** (arXiv 1803.10122) — **[VJ]**. The origin of the term.
- **Bruce et al., 2024 — Genie: Generative Interactive Environments** (ICML) — **[VJ]**. Latent-action + generative; conceptual ancestor of LAPO-style latent actions. (DreamZero cites *Genie 3* and *Genie Envisioner*, not the original Genie.)
- **Hafner et al., 2023 — DreamerV3** (arXiv 2301.04104) — **[both]**. Latent imagination for control (learned from scratch).
- **Hu et al., 2023 / Russell et al., 2025 — GAIA-1 / GAIA-2: generative world models for driving** (arXiv 2309.17080 / 2503.20523) — **[VJ]**.
- **Finn et al., 2016 — Unsupervised Learning for Physical Interaction through Video Prediction** (NeurIPS) — **[VJ]**. The grandparent of video-prediction-for-control.

### Suggested next-read order (given your project)
1. **Lipman 2022 (flow matching)** + **Albergo 2023 (stochastic interpolants)** — so the noise vector is no longer a black box.
2. **DINO-WM (Zhou 2024)** — the frozen-features→planning idea, already done in the JEPA world; your strongest "someone proved the frozen premise works" citation.
3. **Wan 2.2** — your likely actual Φ; understand its noise interface.
4. **UniPi (Du 2023)** + **Cosmos Policy (Kim 2026)** — the generate-then-act lineage you are differentiating from.

---

## Open threads to revisit
- Confirm whether Wan 2.2 (or Cosmos) exposes initial noise $z_0$ *and* per-step denoising states as inspectable tensors — your probing target depends on this API surface.
- DINO-WM is the closest "frozen world model → zero-shot planning" prior. Read it before writing your related-work, and be ready to say why *noise-space* analysis of a *diffusion* model is different from *feature-space* analysis of a JEPA.
- The degeneracy risk (noise just "copies the next frame") still needs a concrete bottleneck design (low-rank / sparse / quantized) — note candidate mechanisms as you read the flow-matching papers.
