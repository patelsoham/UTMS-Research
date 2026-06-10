# Research — Frozen Video-Model Dynamics

Notes and reading map for the MIDI Lab project: unsupervised recovery of inverse-dynamics /
low-level action structure from a **frozen, action-blind** pretrained video model, by analyzing
— or sampling trajectories within — whatever representation it exposes (a diffusion/flow model's
conditioning noise, a predictive model's latent space, or an autoregressive token rollout),
**without training any new action module**. Diffusion-noise probing is the sharpest single
instantiation, not the whole project.

## Contents

- [phase1_priority.md](phase1_priority.md) — study plan, prioritization, and the canonical project statement.
- [project_notes.md](project_notes.md) — conceptual primer, paper deep-dives, the paper-tie rubric, and curated reading list.
- [research_paper_index.md](research_paper_index.md) — skimmable per-paper index and reference table.
- `papers_deepdive/` — one full deep-dive per paper.
- `.github/skills/paper-deepdive/` — skill encoding the project position, extraction conventions, and per-paper templates.

## Notes

- Paper PDFs live under `Papers/` and are **gitignored** (large, third-party). They are
  not synced through this repo — keep them locally per device.
- This working copy lives inside iCloud Drive via a symlink. Avoid editing the same
  file on two devices simultaneously; pull/commit through git rather than relying on
  iCloud to merge.

- Paper PDFs live under `Papers/` and are **gitignored** (large, third-party). They are
  not synced through this repo — keep them locally per device.
- This working copy lives inside iCloud Drive via a symlink. Avoid editing the same
  file on two devices simultaneously; pull/commit through git rather than relying on
  iCloud to merge.
