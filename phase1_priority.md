# Profile

**Current role:** Product Manager — Microsoft Foundry (Platform-level AI Infrastructure)
**Program:** MSCS, UT Austin
**Lab:** MIDI Lab, Professor Amy Zhang
**Career targets:** Research Engineer (frontier AI labs) · Big Tech AI Infra Engineer/PM · Quant Finance Dev/Researcher

## Research Context — MIDI Lab

**Lab:** MIDI Lab, Professor Amy Zhang · **Primary collaborator:** Max

### Research Problem (Current Project — Pivot)

Extract actionable control signals from pretrained **video generative models** (diffusion models, or any model that predicts next frames from prior ones) by mapping the **noise vector** used during forward prediction to meaningful robotic **low-level actions** (as one example target). We target video generative models because of the **dynamics information** likely encoded in them from large-scale pretraining. If this mapping is possible, we can show that **action-labeled datasets are not strictly required**.

**Core hypothesis:** Large-scale video pretraining implicitly encodes environment dynamics. The forward-prediction noise/latent is a candidate carrier of action-relevant structure that can be decoded into low-level control.

### Max's Framing (Direction Note)

> Yes, that's generally correct. I'd add that we want to develop a **toolbox of techniques for actually understanding the latent space** here so that we can actually find the low-level mapping. This is not trivial in and of itself.

**Takeaway:** The deliverable isn't only a noise→action decoder — it's a set of **latent-space analysis techniques** (probing, structure discovery, alignment) that make the mapping findable in the first place. Interpretability of the generative latent space is a first-class research goal, not just a means to an end.

### Open Questions
- Which latent/noise component (initial noise, intermediate denoising states, cross-frame deltas) carries action-relevant information?
- Is the mapping linear / low-dimensional, or does it require a learned nonlinear decoder?
- How few action labels are actually needed to ground the latent→action map?
- Does the recovered structure transfer across embodiments and tasks?

## Infrastructure Notes

- Cluster: MIDI servers (NVIDIA RTX A5500, driver 535.216.03, CUDA 12.2)
- Container: Singularity (`cvoelcker/ubuntu24_base_amd`), missing NVIDIA Vulkan ICD — Isaac Sim runs with `App ready` warnings on 5.1.0
- Setup script: `pyEnvs/Masters/RL-Research/setup_isaac_lab.sh`

## Research → Industry Practices

Every research activity should mirror how industry teams operate. Frame contributions accordingly.

| Research Activity | Industry Mirror | Transferable Framing |
|---|---|---|
| Isaac Sim on HPC, debugging Vulkan/driver compat | MLOps / Infra Engineering | "Built and maintained GPU-accelerated simulation infrastructure on shared HPC clusters" |
| Training baselines (RMA, FB/PSM) | ML Engineering — experiment design | "Designed and executed systematic RL training experiments with ablation studies" |
| Proprioceptive → pixel/lidar progression | Sensor fusion / Perception pipelines | "Extended state-based RL policies to multimodal observations (vision, lidar)" |
| Zero-shot generalization on complex terrains | Robustness / Evaluation — OOD testing | "Evaluated policy robustness under domain shift across terrain distributions" |
| Collecting data, managing experiment artifacts | Data Engineering — pipeline design | "Built data collection and experiment tracking pipelines for large-scale RL" |
| Debugging convergence issues (math vs. code) | Root cause analysis | "Diagnosed training failures across the math/implementation boundary" |

### Practices to Adopt Now

1. **Version your experiments** — use Weights & Biases or MLflow. Log configs, seeds, metrics. Industry expects this.
2. **Write reproducible scripts** — parameterize every experiment: `train_spot_flat.sh`, `train_spot_rough.sh` with all hyperparams explicit. (Started with `setup_isaac_lab.sh`.)
3. **Document decisions, not just results** — keep a research log of *why* you chose each baseline, *why* you changed a hyperparameter. This translates to design docs in industry.
4. **Own the pipeline end-to-end** — don't just run training scripts. Understand env configs, reward shaping, policy architecture. Be the person who can modify any layer.
5. **Profile and optimize** — measure GPU utilization, training throughput (steps/sec), wall-clock time. This is the language of ML infra teams.

### What I Need to Build (Research Engineer Identity)

- **Mathematical fluency:** Spectral methods, eigendecomposition, Markov chains — can't debug PSM convergence without it
- **Research judgment:** Know when a result is a bug vs. a finding, when to pivot baselines, when to push harder on a direction

---

# Phase 1 Priority — Reprioritized for the Video-Generative-Model Pivot

The new project is about **understanding the latent/noise space of pretrained video generative (diffusion) models and decoding it into low-level actions**. Re-scoring the Get_Good library against that goal: the math core (Linear Algebra, Probability) is *unchanged at the top* — it's actually more central now, because "developing a toolbox to understand the latent space" (Max) is fundamentally linear-algebraic and probabilistic. What drops out is the quant/systems track (finance books, learncpp, OS, networking) — those revert to career-background, not research-critical.

**New priority order:** Linear Algebra → Probability (Stats 110) → Diffusion/Generative foundations → RL/control grounding. Everything else is background.

## 1. Linear Algebra Done Right — First, Still the Bottleneck

The "toolbox for understanding the latent space" *is* linear algebra. Probing directions, PCA/SVD of activations, Jacobians of the denoiser w.r.t. its noise input, subspace alignment between latent deltas and action deltas — all of it lives in Ch 5: Eigenvalues and Eigenvectors, Ch 6: Inner Product Spaces, and Ch 7: Operators on Inner Product Spaces (Spectral Theorem, SVD). If you want to claim "this noise direction maps to this action," you need to reason about subspaces, projections, and singular vectors fluently — not just "read once."

**Start here:** Ch 5: Eigenvalues and Eigenvectors → Ch 6: Inner Product Spaces → Ch 7: Operators on Inner Product Spaces. Pull in **SVD** (Strang) explicitly — it's the workhorse for analyzing learned latent spaces.

## 2. Stats 110 — Second, Now Pointed at Diffusion

Reframed for the pivot: diffusion models *are* a stochastic process. The forward process is progressive Gaussian corruption; the reverse process is learned denoising / score estimation. You need the probability backbone — Gaussians, conditioning, change of variables, expectations/LOTUS, and the limit theorems — to read the diffusion literature and reason about what the "noise vector" even is.

**Fast path for this project:** Lec 13-14 (Normal, standardization, LOTUS) → Lec 16 (Exponential/memoryless) → Lec 19-21 (Joint, conditional, covariance) → Lec 26-29 (conditional expectation, Adam/Eve, CLT). The Markov-chain lectures (31-33) are still useful framing for dynamics, but they are no longer the headline — the Gaussian/score machinery is.

## 3. Diffusion & Generative Modeling Foundations — New, Project-Critical

This is the genuinely new requirement the pivot introduces; it is not a book in the original Get_Good list. Budget dedicated time here.

- **Score-based generative modeling / SDEs** (Song et al.) — the forward/reverse SDE view is the cleanest mental model for "what the noise carries."
- **DDPM / DDIM** — the discrete-time formulation you'll actually instrument; DDIM's deterministic sampler makes the noise↔output map a function you can probe.
- **Deep Learning Book (Goodfellow)** — representation learning, autoencoders, and the generative-model chapters for grounding. Use it as reference, not cover-to-cover.

## Baselines — Read Before Monday

All four live in `Papers/Understanding_Noise/`. Read in this order; the first two are the conceptual anchors, the third is the closest empirical comparison, the fourth is the map of the field.

1. **LAPO — *Learning to Act without Actions*** (ICLR 2024, Schmidt & Jiang / FAIR) — `Learn to Act without Actions.pdf`
   Latent Action Policies: recovers latent action information **purely from videos** via a coupled inverse-dynamics model (predict the latent action between consecutive frames) regularized by forward dynamics. First method to recover the **structure of the true action space** from observed dynamics alone; latent-action policies fine-tune to expert level with few labels or online RL.
   *Why it matters:* the canonical "actions are recoverable from dynamics" result and the strongest conceptual baseline. Your noise→action hypothesis is the **generative-model analog** of LAPO's inverse-dynamics latent — be ready to articulate that contrast.

2. **LAPA — *Latent Action Pretraining from Videos*** (ICLR 2025, KAIST/UW/Microsoft Research/NVIDIA) — `LAPA.pdf`
   First **unsupervised** VLA pretraining without ground-truth action labels. Stage 1: a **VQ-VAE** learns *discrete* latent actions between frames. Stage 2: a latent VLA predicts those latent actions from observation + task text. Stage 3: finetune on small robot data to map latent → real actions.
   *Why it matters:* direct precedent for "action labels aren't required," but it learns latents with a **separate quantizer**. Your angle: can the video model's **own forward-prediction noise** already contain this, skipping the VQ-VAE entirely?

3. **ViPRA — *Video Prediction for Robot Actions*** (ICLR 2026, CMU / Skild AI / UC Irvine) — `VIPRA.pdf`
   Turns a **video-prediction model** into a policy from actionless videos. Trains a video-language model to predict **both** future observations **and** motion-centric latent actions (perceptual + optical-flow-consistency losses → physically grounded latents); a chunked **flow-matching** decoder maps latents → continuous actions from only 100–200 demos (smooth control up to 22 Hz). Models "what changes *and* how."
   *Why it matters:* closest to your thesis and your top empirical baseline (SIMPLER +16%, real +13%). Key contrast: their latents are **flow-supervised**, not the raw generative noise — your claim is that the noise is *already* an action-relevant latent.

4. **World Action Models (WAMs)** (survey, Fudan / SII / NUS) — `World Action Models.pdf`
   Survey defining **WAMs**: embodied models unifying predictive state modeling + action generation (joint distribution over future states *and* actions). Taxonomy of **Cascaded vs Joint** WAMs by generation modality, conditioning, and action-decoding strategy; surveys the data ecosystem and evaluation.
   *Why it matters:* the field map. Use it to **position** the noise→action idea and pull adjacent baselines/citations fast. Read for taxonomy/framing, not method.

**Monday one-liner to have ready:** "LAPO/LAPA/ViPRA all *learn a separate latent action space* (inverse dynamics, VQ-VAE, or flow-supervised). My hypothesis is that a pretrained video diffusion model's **own noise/latent already encodes** that action structure — so the contribution is a *toolbox to read it out*, not another latent-action learner."

## Fifty Problems — Third, Passive

This is a **side dish**, not a main course. Do 3 problems per week as warmup before your math track. It keeps your probabilistic reasoning sharp for quant interviews without consuming real study time.

## learncpp — Parallel but Lower Urgency

You're not interviewing tomorrow. Your research needs math **today**. Start learncpp in your afternoon systems slot, but if you have to sacrifice one track in a crunch week, sacrifice this one — not LinAlg.

---

## TL;DR Execution for Week 1-2

```
Before Monday:  Read Understanding_Noise baselines (LAPO → LAPA → ViPRA → WAMs survey)
Week 1:  LinAlg Ch 5: Eigenvalues + Ch 6: Inner Product Spaces (+ SVD)  +  skim DDPM/DDIM
Week 2:  LinAlg Ch 7: Operators on Inner Product Spaces  +  Stats 110 Lec 13-14, 19-21  +  Song score-SDE paper
Week 3+: SVD/probing on a real diffusion latent  +  Stats 110 conditional-expectation/CLT block  +  reproduce one baseline's latent-action setup
```

**The one rule:** If your research code breaks and you can't diagnose whether it's a math problem or a code problem, that's a signal you haven't gone deep enough on LinAlg yet. Stay there longer.

---

## Resources & Links

### Linear Algebra Done Right (Axler, 4th Ed.)
- **Textbook (free PDF):** [linear.axler.net](https://linear.axler.net)
- **Axler's Video Lectures:** [linear.axler.net/LADRvideos4e.html](https://linear.axler.net/LADRvideos4e.html)
- **MIT 18.06 — Strang (geometric intuition supplement):** [MIT OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)
- **EarlyOrbit Math (exercise walkthroughs):** [YouTube Playlist](https://www.youtube.com/playlist?list=PLsaYR0CsKqCbbReVKzKWXyiS00A0wsMJ5)

### Stats 110 — Introduction to Probability (Blitzstein & Hwang, 2nd Ed.)
- **Textbook:** `Get_Good/Stats_110.pdf`
- **Harvard Stats 110 Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo)
- **Course page (problem sets, strategic practice):** [projects.iq.harvard.edu/stat110](https://projects.iq.harvard.edu/stat110)

### Fifty Challenging Problems in Probability (Mosteller)
- **Book:** [Archive.org](https://archive.org/details/fiftychallengingproblemsinprobabilitywithsolutions)

### learncpp
- **Site:** [learncpp.com](https://www.learncpp.com/)

### Computer Networking: A Top-Down Approach (Kurose & Ross, 8th Ed.)
- **Companion site:** [gaia.cs.umass.edu/kurose_ross/eighth.php](https://gaia.cs.umass.edu/kurose_ross/eighth.php)
- **Wireshark Labs (hands-on):** [gaia.cs.umass.edu/kurose_ross/wireshark.php](https://gaia.cs.umass.edu/kurose_ross/wireshark.php)

### Quant-Specific (if quant becomes a real target)
- **Joshi — C++ Design Patterns and Derivatives Pricing**
- **Shreve — Stochastic Calculus for Finance (Vol I & II)**

---

## Stats 110 — Lecture-to-Chapter Map

Based on Blitzstein & Hwang's *Introduction to Probability* textbook:

| Lec | Topic | Ch | Your Priority |
|-----|-------|----|---------------|
| 1 | Sample Spaces, Naive Probability, Counting | 1-2 | ⏭ Skip if strong |
| 2 | Bose-Einstein, Story Proofs, Vandermonde | 2 | ⏭ Skip if strong |
| 3 | Birthday Problem, Inclusion-Exclusion | 2-3 | ⏭ Skip if strong |
| 4 | Independence, Conditional Probability, Bayes | 2-3 | ⏭ Skip if strong |
| 5 | Law of Total Probability, Independence | 2-3 | ⏭ Skip if strong |
| 6 | Monty Hall, Simpson's Paradox | 3 | ⏭ Skip if strong |
| **7** | **Gambler's Ruin, First Step Analysis, Bernoulli, Binomial** | **3-4** | **🟡 START HERE** |
| 8 | Random Variables, CDFs, PMFs, Hypergeometric | 4-5 | 🟡 |
| 9 | Independence, Geometric, Expectation, Indicators | 4-5 | 🟡 |
| 10 | Linearity, Negative Binomial, St. Petersburg | 4-5 | 🟡 |
| 11 | Poisson Distribution, Poisson Approximation | 5-6 | 🟡 |
| 12 | Discrete vs Continuous, PDFs, Variance, Uniform | 6-7 | 🟡 |
| 13 | Standard Normal, Normalizing Constant | 7 | 🟡 |
| **14** | **Normal Distribution, Standardization, LOTUS** | **7** | **🔴 KEY** |
| 15 | Midterm Review | — | ⏭ Skip |
| 16 | Exponential, Memoryless Property | 6-7 | 🟡 |
| **17** | **MGFs, Hybrid Bayes', Laplace's Rule** | **7** | **🔴 Quant interview** |
| **18** | **MGFs for Exp/Normal, Sums of Poissons, Joint Dist** | **7-8** | **🔴 Quant interview** |
| 19 | Joint, Conditional, Marginal Distributions, 2D LOTUS | 8 | 🟡 |
| 20 | Expected Distance, Multinomial, Cauchy | 8-9 | 🟡 |
| **21** | **Covariance, Correlation, Variance of Sum** | **8-9** | **🔴 KEY** |
| 22 | Transformations, LogNormal, Convolutions | 9 | 🟡 |
| 23 | Beta Distribution, Bayes' Billiards | 10 | 🟡 |
| 24 | Gamma Distribution, Poisson Process | 10 | 🟡 |
| 25 | Beta-Gamma, Order Statistics, Conditional Expectation | 9-10 | 🟡 |
| **26** | **Conditional Expectation cont., Waiting for HT vs HH** | **10** | **🔴 KEY for RL** |
| **27** | **Adam's Law, Eve's Law** | **10** | **🔴 KEY for RL** |
| **28** | **Sum of Random # of RVs, Inequalities (Markov, Chebyshev, Jensen)** | **10-11** | **🔴 KEY** |
| **29** | **Law of Large Numbers, Central Limit Theorem** | **11** | **🔴 Quant interview** |
| 30 | Chi-square, Student-t, Multivariate Normal | 12 | 🟡 |
| **31** | **Markov Chains, Transition Matrix, Stationary Dist** | **12** | **🔴 CRITICAL for PSM** |
| **32** | **Markov Chains cont., Irreducibility, Reversibility** | **12** | **🔴 CRITICAL for PSM** |
| **33** | **Markov Chains cont., Google PageRank** | **12** | **🔴 CRITICAL for PSM** |
| 34 | A Look Ahead | — | ⏭ Skip |

**Fast path:** Lectures 7-14 → 17-18 → 21 → 26-29 → 31-33 (18 lectures instead of 34).

> **Blitzstein & Hwang Chapter Key:**
> Ch 1: Probability and Counting · Ch 2: Conditional Probability · Ch 3: Random Variables and Their Distributions · Ch 4: Expectation · Ch 5: Continuous Random Variables · Ch 6: Moments · Ch 7: Joint Distributions · Ch 8: Transformations · Ch 9: Conditional Expectation · Ch 10: Inequalities and Limit Theorems · Ch 11: Markov Chains · Ch 12: Markov Chain Monte Carlo · Ch 13: Poisson Processes

---

## Linear Algebra Done Right — Lecture Pairings

**Primary recommendation: Axler's own video lectures** — he recorded a full series for the 4th edition, mapped section-by-section to the book.

| Resource | Style | Link | Best For |
|----------|-------|------|----------|
| **Axler's Own Videos (Recommended)** | Abstract, proof-oriented, exact book match | [linear.axler.net/LADRvideos4e.html](https://linear.axler.net/LADRvideos4e.html) | Direct companion — watch before reading each chapter |
| **MIT 18.06 (Strang)** | Computational, geometric intuition | [MIT OCW](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | When Axler feels too abstract and you need to *see* what's happening |
| **EarlyOrbit Math (YouTube)** | Detailed walkthroughs, worked examples | [YouTube Playlist](https://www.youtube.com/playlist?list=PLsaYR0CsKqCbbReVKzKWXyiS00A0wsMJ5) | When you're stuck on a proof in the exercises |

**Recommendation:** Use **Axler's videos as primary**, and pull up **Strang** when you need geometric intuition for eigenvalues (Strang Lectures 21-24) or SVD. Axler gives you the abstract operator theory your PSM work needs; Strang gives you the computational instinct for debugging numerical code.

> **Axler Chapter Key (4th Edition):**
> Ch 1: Vector Spaces · Ch 2: Finite-Dimensional Vector Spaces · Ch 3: Linear Maps · Ch 4: Polynomials · Ch 5: Eigenvalues and Eigenvectors · Ch 6: Inner Product Spaces · Ch 7: Operators on Inner Product Spaces · Ch 8: Operators on Complex Vector Spaces · Ch 9: Multilinear Algebra and Determinants

---

# Appendix — Quant / Systems Career Track (Deprioritized for the Pivot)

These tracks (networking, OS, learncpp, quant finance) are **not** research-critical for the video-generative-model project. They are parked here as **career background** — pick them up on a separate cadence once the research track has momentum. Nothing below blocks Phase 1.

## Computer Networking: A Top-Down Approach — The 80/20 Cut

> **Appendix status (post-pivot):** Not part of the active research track. Retained here as **quant/systems + Security-PM career background** — reference it on its own cadence, not as Phase 1 research-critical study.

Kurose & Ross, 8th Edition. Given your three tracks (AI Research, Quant/HFT Infra, Big Tech AI Infra) **plus** Security PM with agents focus, most of this book is dead weight. Here's what actually moves the needle.

### 🔴 CRITICAL — Read Cover to Cover

**Ch 2: Application Layer**
HTTP, DNS, SMTP, socket programming. This is how agents talk to the world. Every API call, every webhook, every agent-to-agent communication protocol sits here. The socket programming sections (2.7) are hands-on practice you'll actually use.
- *Security PM angle:* Understanding DNS poisoning, HTTP request smuggling, and application-layer attacks requires knowing the clean version first.

**Ch 3: Transport Layer**
TCP and UDP internals, congestion control, flow control, reliable data transfer principles. This is the densest chapter and the highest-ROI one.
- *HFT angle:* This is where you learn why HFT firms use UDP multicast, why Nagle's algorithm is the enemy, and what `TCP_NODELAY` actually does.
- *Security PM angle:* TCP state machines are foundational for firewalls, IDS/IPS, and SYN flood attacks.
- *AI Infra angle:* Distributed training (AllReduce, parameter servers) is bottlenecked by TCP congestion control in datacenter networks.

**Ch 8: Security in Computer Networks**
TLS/SSL, public key infrastructure, authentication protocols, firewalls, IDS. This is your bread and butter as a Security PM. Non-negotiable.

### 🟡 SELECTIVE — Read Key Sections Only

**Ch 1: Computer Networks and the Internet**
- **Skim** 1.1–1.3 (what is the internet, network edge/core) for vocabulary.
- **Read carefully:** 1.4 (Delay, Loss, and Throughput) — this is the mental model for every performance conversation you'll ever have. Propagation vs. transmission vs. queuing delay.
- **Skip** 1.5–1.7 (protocol layers history) unless you need the OSI reference model for interviews.

**Ch 4: The Network Layer: Data Plane**
- **Read:** 4.1–4.3 (Router architecture, IP addressing, NAT). You need IP and subnetting cold for security work.
- **Read:** 4.4 (Generalized Forwarding and SDN) — SDN is how modern datacenter networks (your AI Infra track) are managed.
- **Skip** the detailed forwarding algorithm internals.

**Ch 5: The Network Layer: Control Plane**
- **Read:** 5.2–5.3 (Routing algorithms — link state and distance vector). Conceptual understanding only.
- **Read:** 5.4 (OSPF and BGP). BGP is critical — BGP hijacking is a real attack vector, and BGP misconfigurations cause major outages. As a Security PM, you need to know this.
- **Skip** 5.5–5.7 (SDN control plane details, ICMP internals) unless you're building network security products.

### ⏭ SKIP — Low ROI for Your Goals

**Ch 6: The Link Layer and LANs**
Ethernet frames, MAC addresses, ARP, switches. You'll absorb what you need from Chapters 1–5. ARP spoofing is worth 15 minutes of reading (6.4), not the whole chapter.

**Ch 7: Wireless and Mobile Networks**
WiFi, cellular, mobility management. Zero relevance to your three career tracks unless you pivot to IoT security.

### Execution Order

```
Phase 1:  Ch 2: Application Layer  +  Ch 8: Security in Computer Networks  (immediate PM relevance)
Phase 2:  Ch 3: Transport Layer  (densest technical content, needs dedicated focus)
Phase 3:  Ch 1 (selective)  +  Ch 4: The Network Layer: Data Plane (selective)  +  Ch 5: The Network Layer: Control Plane (selective)
```

**The acid test:** If someone describes a network attack and you can't mentally trace the packet path from application layer through transport to network layer, you haven't gone deep enough on Chapters 2–4.
