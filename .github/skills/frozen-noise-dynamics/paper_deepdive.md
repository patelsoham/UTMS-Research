---
name: frozen-noise-dynamics
description: "Use when working on the frozen-noise video-diffusion research project: recovering inverse dynamics from a frozen pretrained video diffusion model's conditioning noise. Covers the project thesis, the claims to stress-test, positioning against LAPO/LAPA/ViPRA/Genie, paper deep-dives (DreamZero, V-JEPA 2), and the note-taking conventions for project_notes.md. Triggers: latent-action recovery, noise subspace, inverse dynamics, frozen world model, diffusion noise probing, paper extraction, citations subsection."
---

# Frozen-Noise Dynamics Research

## Project position

**Research question.** Does the conditioning noise of a *frozen* pretrained video diffusion model contain a recoverable subspace that encodes single-step inverse dynamics?

**Setup.** Given a video diffusion model $\Phi$ trained at scale on internet video and a pair of adjacent frames $(o_t, o_{t+1})$, find a structured low-dimensional region of $\Phi$'s sampling noise $\epsilon$ such that varying within it parameterizes the transition $o_t \to o_{t+1}$, and decodes — with a small action-labeled dataset — to embodiment-specific control. Robot end-effector deltas are the headline target.

**Premise.** Internet-scale predictive training must encode environment dynamics, or the model could not generate plausible futures. The question is not *whether* dynamics are encoded but *where* they are readable from: whether the noise vector exposes them as a clean subspace, or whether action information is too entangled with nuisance variation (lighting, camera motion, stochastic background) to project out.

**Contribution.**
1. **A latent-analysis toolbox** — probing, subspace identification, Jacobian/SVD analysis, and amortized noise inversion — to characterize what does and does not live in $\Phi$'s noise. This is a primary deliverable, not a means to an end.
2. **An existence result and method.** If the subspace exists, recover it without retraining $\Phi$ and without per-domain world-model training, and decode it to control with far smaller action-labeled datasets than LAPA or ViPRA need. If it does not exist, the toolbox stands, and the negative result shows that action-readability is not a free byproduct of internet-scale predictive training.

**Positioning.** LAPO, LAPA, ViPRA, and Genie all train a new latent-action module from scratch on domain-specific data. This work trains no new world model and no new latent-action tokenizer. The only trained component is the extractor that reads structure off a frozen $\Phi$.

## Claims to keep stress-testing

- $\Phi$ stays **frozen** — this is the central novelty axis. If any step quietly fine-tunes $\Phi$, the contribution collapses into prior work.
- The method is **diffusion-specific** because only diffusion/flow models expose a continuous structured noise vector $z_0$ as a first-class input. A JEPA (e.g. V-JEPA 2) has no such vector and is out of scope as a target — it only validates the "frozen world model" premise.
- A **bottleneck** (low-rank / sparse / quantized) is mandatory, or the extractor degenerates into copying the next frame.
- The base checkpoint to probe is the public **Wan2.1-I2V-14B-480P** (DreamZero's backbone), loadable before any robot fine-tuning.

## Anti-hallucination rules (non-negotiable)

These are prescriptive because the failure mode is fabrication. Follow exactly.

- **"Not stated" over inference.** If the paper does not state something a section asks for, write `not stated`. Never infer a number, date, baseline, or result the text does not contain.
- **Quote-or-omit for claims.** Every non-obvious claim carries a verbatim quoted phrase (3–6 words) from the paper that a reader can grep back. If you cannot quote it, do not write it.
- **Grep-or-omit for counts.** No number enters the reference table unless a grep produced it. Record the command, e.g. `grep -ic "Genie" /tmp/DreamZero.pdf.txt`.
- **Section anchors, not page numbers.** Cite `§3.1`, not "page 7" — extracted page numbers are approximate; section names are reliable.

## Gotchas (environment-specific, will bite otherwise)

- `/tmp` is outside the workspace, so workspace search tools (grep_search/file_search) return nothing there. Use terminal `grep -n` or read the absolute `/tmp` path.
- PyMuPDF imports as `fitz`, not `pymupdf`. It is available system-wide; do not pip-install.
- The diffusion checkpoint to probe is **Wan2.1-I2V-14B-480P** (DreamZero's backbone), loadable before any robot fine-tuning. DreamZero fine-tunes from it; load the base Wan, not the DreamZero weights, to test the frozen-Φ premise.
- V-JEPA 2 has **no noise vector** — it is a JEPA, not generative. It validates the "frozen world model" premise but is out of scope as a probing *target*.

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

**Source:** [pdf](path) · lab · date · arXiv id · [Code](repo-url).
**Code:** [repo](url) · license · weights (extra detail: backbone, checkpoints, HF links)

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
Explicit link back to frozen-noise dynamics.

## Citations
Each non-obvious claim → §section + verbatim phrase.

## References — pure-video & video-generation
Every reference about pure-video / video generation + strong supplementary reading.
Tag each [target] (directly on the noise/dynamics thesis) or [background]. Feeds the table below.
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
