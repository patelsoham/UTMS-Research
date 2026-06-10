---
name: paper-deepdive
description: "Use when working on the frozen-video-model dynamics research project: recovering inverse-dynamics / low-level action structure from a frozen, action-blind pretrained video model's representation (diffusion/flow conditioning noise, predictive latent space e.g. V-JEPA 2, or autoregressive token rollout). Covers the project thesis, the claims to stress-test, positioning against LAPO/LAPA/ViPRA/Genie, paper deep-dives (DreamZero, V-JEPA 2), the paper-tie rubric, and the note-taking conventions for project_notes.md. Triggers: latent-action recovery, representation probing, noise/latent subspace, trajectory sampling, inverse dynamics, frozen world model, paper extraction, citations subsection."
---

# Frozen Video-Model Dynamics Research

## Project position

**Canonical statement.** The single source of truth for *what this project is* lives in the "Research Problem" block of [phase1_priority.md](phase1_priority.md); the reading frame + paper-tie rubric live at the top of [project_notes.md](project_notes.md). Keep this skill consistent with those.

**Research question.** Can a *frozen, action-blind* pretrained video model's **representation** be read — via a toolbox of latent-space analysis techniques — to recover **low-level action / inverse-dynamics** structure, **without training any new action module**?

**Setup.** Given a video model $\Phi$ trained at scale on internet video and a pair of adjacent frames $(o_t, o_{t+1})$, recover action structure from whatever **representation** $\Phi$ exposes — the **conditioning noise** $z_0$ of a diffusion/flow model, the **latent embedding space** of a predictive model (e.g. V-JEPA 2 Stage-1), or an **autoregressive (token) rollout** — such that the structure parameterizes the transition $o_t \to o_{t+1}$ and decodes, with a small action-labeled dataset, to embodiment-specific control. The signal need **not be directly stored** as a static subspace; the method may instead **sample/traverse trajectories within the representation** and read dynamics from how they move. Robot end-effector deltas are the headline target. Diffusion-noise probing (Wan) is the **sharpest single instantiation**, not the whole project.

**Premise.** Internet-scale predictive training must encode environment dynamics, or the model could not generate plausible futures. The question is not *whether* dynamics are encoded but *where* they are readable from: whether the representation exposes them as a clean subspace (or a traversable trajectory manifold), or whether action information is too entangled with nuisance variation (lighting, camera motion, stochastic background) to project out.

**Contribution.**
1. **A latent-analysis toolbox** — probing, subspace identification, Jacobian/SVD analysis, trajectory sampling, and amortized inversion — to characterize what does and does not live in $\Phi$'s representation. This is a primary deliverable, not a means to an end.
2. **An existence result and method.** If the structure exists, recover it without retraining $\Phi$ and without per-domain world-model training, and decode it to control with far smaller action-labeled datasets than LAPA or ViPRA need. If it does not exist, the toolbox stands, and the conclusive negative shows action-readability is not a free byproduct of internet-scale predictive training.

**Positioning (the two novelty axes).** (1) *Frozen vs. trained:* LAPO, LAPA, ViPRA, Genie, and DreamZero all train/co-train a world model and/or an action module; we keep $\Phi$ frozen and train only the extractor. (2) *Discover vs. manufacture:* those methods *manufacture* a latent action space by construction (VQ/IDM-FDM bottleneck, action head); we *discover* structure already latent in a frozen, **action-blind** $\Phi$ that never had an action interface. The defensible claim is the conjunction — drop "frozen" and you're Genie; drop "action-blind" and you're reading an exposed action port; drop "toolbox/discover" and you're just another latent-action learner.

## Claims to keep stress-testing

- $\Phi$ stays **frozen** — central novelty axis #1. If any step quietly fine-tunes $\Phi$, the contribution collapses into prior work.
- $\Phi$ is **action-blind** — central novelty axis #2 (*discover vs. manufacture*). The probing **target must be a checkpoint trained with no action interface**. "Action-blind" is a property of the *checkpoint*, not the model *family*: probe the action-blind sibling (Wan base, base Cosmos, V-JEPA 2 **Stage-1** encoder, a no-LAM AR video model), never the action-conditioned one (DreamZero weights, Cosmos **Policy**, V-JEPA 2-**AC**, Genie's co-trained **LAM**). Probing an action-conditioned component is **circular** and not a contribution.
- The method is **representation-general, not diffusion-only.** Diffusion/flow noise $z_0$ is the *sharpest* instantiation (a continuous structured vector as a first-class input), but the latent space of a predictive model (V-JEPA 2) and an AR token rollout are equally in scope. The action signal need not be a static stored subspace — **sampling trajectories within the representation** is a first-class method.
- A **bottleneck** (low-rank / sparse / quantized) is mandatory, or the extractor degenerates into copying the next frame.
- Must **beat a frozen-feature IDM / linear-probe baseline** decisively, or the contribution looks like reinvented probing.
- The sharpest diffusion checkpoint to probe is the public **Wan2.1-I2V-14B-480P** (DreamZero's backbone), loadable before any robot fine-tuning.

## Anti-hallucination rules (non-negotiable)

These are prescriptive because the failure mode is fabrication. Follow exactly.

- **"Not stated" over inference.** If the paper does not state something a section asks for, write `not stated`. Never infer a number, date, baseline, or result the text does not contain.
- **Quote-or-omit for claims.** Every non-obvious claim carries a verbatim quoted phrase (3–6 words) from the paper that a reader can grep back. If you cannot quote it, do not write it.
- **Grep-or-omit for counts.** No number enters the reference table unless a grep produced it. Record the command, e.g. `grep -ic "Genie" /tmp/DreamZero.pdf.txt`.
- **Section anchors, not page numbers.** Cite `§3.1`, not "page 7" — extracted page numbers are approximate; section names are reliable.

## Gotchas (environment-specific, will bite otherwise)

- `/tmp` is outside the workspace, so workspace search tools (grep_search/file_search) return nothing there. Use terminal `grep -n` or read the absolute `/tmp` path.
- PyMuPDF imports as `fitz`, not `pymupdf`. It is available system-wide; do not pip-install.
- The sharpest diffusion checkpoint to probe is **Wan2.1-I2V-14B-480P** (DreamZero's backbone), loadable before any robot fine-tuning. DreamZero fine-tunes from it; load the **base Wan**, not the DreamZero weights, to keep Φ frozen and action-blind.
- **Target the action-blind checkpoint, not the family.** V-JEPA 2 and Cosmos each ship a valid target *and* an out-of-target sibling: probe V-JEPA 2 **Stage-1** (action-free encoder) / **base Cosmos**, never V-JEPA 2-**AC** / **Cosmos Policy**. Genie is the awkward case — its LAM is co-trained, so it has no clean action-blind checkpoint; treat Genie as **contrast/proof-of-premise**, not a probing target.
- Watch for **soft action leakage**: a model trained on robot video (camera/proprioception correlated with action) is still action-blind if actions were never an *input/output*, but a skeptic will ask whether you recover dynamics or just a robot-data prior. Pre-empt, don't disqualify.

## File architecture

Split, not monolithic:

- One deep-dive per paper under `papers_deepdive/` (e.g. `papers_deepdive/dreamzero.md`). Full extraction lives here.
- One index in [research_paper_index.md](research_paper_index.md): shared background primer, a concise per-paper entry (one-line + 6 sections, see below) linking to its deep-dive, and the reference table. The skill is the single source of truth for structure.
- [project_notes.md](project_notes.md) is retained as-is for now — do not migrate it.
- The plan stays in [phase1_priority.md](phase1_priority.md).

Reader assumes only rudimentary RL — teach the background a paper needs, but only what it needs.

## Per-paper deep-dive template

Keep each file to ~one screen. Prefer tables over prose; tables cap verbosity. Use `not stated` for any empty field.

```markdown
# <Paper> — <one-line claim>

**Source:** [paper](arxiv-or-project-url) · lab · date · [Code](repo-url).
**Code:** [repo](url) · license · weights (extra detail: backbone, checkpoints, HF links)

The **[paper]** link must point to an online source (arXiv `arxiv.org/abs/<id>`, or the project page if the paper has no arXiv id) so it opens on any device once pushed. Do **not** add a local `[pdf]` link — the PDFs are gitignored and will not sync.

## Motivations
What gap or failure of prior work it targets.

## Methodology
How it works. KaTeX for key equations only.

## How it differs / Limitations  (one table)
| vs prior practice | what they did | prior-work limitation it addresses |
|---|---|---|
(add a "stated limitations" row group from the paper's own discussion + your skeptical read)

## Conclusions
What the results establish (one or two lines).

## Ties to the project
Explicit link back to the frozen, action-blind video-model dynamics thesis. **Open with a rubric role tag** (one or more): `proof-of-premise` / `contrast-baseline` / `valid-probing-target` / `scope-boundary` / `method-tool` (see the rubric at the top of [project_notes.md](project_notes.md)). State which checkpoint, if any, is an action-blind target vs. an out-of-target action-conditioned sibling.

## Citations
Each non-obvious claim → §section + verbatim phrase.

## References — pure-video & video-generation
Every reference about pure-video / video generation + strong supplementary reading.
Tag each [target] (an action-blind representation we could probe, or directly on the dynamics-recovery thesis) or [background]. Feeds the table below.
```

## Index entry (in research_paper_index.md)

Each paper gets a concise entry: a one-line claim plus the same 6 sections as the deep-dive, one or two lines each. This is the skimmable view; the deep-dive is the full read.

```markdown
### <Paper> — <one-line claim>
[deep-dive](papers_deepdive/<name>.md) · arXiv id · [Code](repo-url)

- **Motivations:** … (1 line)
- **Methodology:** … (1–2 lines)
- **How it differs:** … (1 line, vs prior practice)
- **Limitations:** … (1 line)
- **Conclusions:** … (1 line)
- **Ties to the project:** … (1 line)
```

## Reference table (in research_paper_index.md)

| Reference | Cited by | # | Role | Read? |
|---|---|---|---|---|

- **Cited by** = which read papers reference it; **#** = that count (grep-verified).
- **Role** is a 3-value enum, not free text: `target` / `mechanism` / `background`.
- Read-next priority = high **#** ∧ role=`target`, filtered to unread. Re-sort as the corpus grows.
- Advisory until the corpus has ≥ 3 papers — with 2 papers it is barely a graph.

## Procedure

1. Extract: `python3 -c "import fitz; d=fitz.open('f.pdf'); open('/tmp/f.txt','w').write(''.join(p.get_text() for p in d))"`
2. Fill the template. Apply the anti-hallucination rules at every field.
3. Pull pure-video / video-generation references; tag each `target`/`background`.
4. Update research_paper_index.md: add the one-line + 6-section entry, merge references into the table, increment grep-verified counts.
5. **Validate before finishing:** every count traces to a recorded grep; every claim has a quote or says `not stated`; the deep-dive is ~one screen. Fix and re-check until all three pass.

## Writing style

Concise and direct. No AI-isms: no "Concretely,", "twofold", "it is worth noting", "the field needs", padding adjectives, or em-dash pileups for emphasis. State the claim, then the evidence.
