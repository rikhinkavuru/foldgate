# Two papers, one project

This directory holds two distinct submissions built from the same results. They are not drafts of
each other; they target different venues at different lengths and are maintained separately.

## 1. Archival paper — Digital Discovery (RSC) / Regeneron STS

- **Source:** `foldgate_journal.tex`  →  **build:** `tectonic paper/foldgate_journal.tex`  (~8 pages)
- **Title:** *Know When to Fold: Risk-Controlled Abstention for Protein–Ligand Co-Folding under Novelty Shift*
- **Status:** archival, full-length, non-anonymous.
- **Contains, beyond the short paper:** the impossibility + achievability theorem with its proof
  appendix; validation on a synthetic known conditional and on public non-co-folding benchmarks
  (elec2, ACS Income); the per-stratum feasibility frontier (the theorem's empirical shadow); the
  certifier-matching result (exact binomial on the binary label, betting bound on the graded RMSD
  loss); explicit multiplicity control; the benchmark-integrity note on the Runs N' Poses
  protomer trap.
- **Deadline:** Digital Discovery is a rolling submission; STS is the ambition target. No fixed
  date drives this file, so it carries the complete record.
- **Format note:** the preamble is a self-contained single-column article. Swap in the Digital
  Discovery class at submission.

## 2. Non-archival short paper — MoML 2026 @ MIT

- **Source:** `moml2026_shortpaper.tex`  →  **build:** `tectonic paper/moml2026_shortpaper.tex`
  (4 body pages + references). This is the canonical submission file; it reproduces the MoML/LoG
  example style (title rules, running header, first-page venue footer, booktabs tables, a
  4-to-6-sentence single-paragraph abstract). `moml2026_shortpaper.md` is the earlier
  pandoc-rendered draft, kept for reference only; do not submit it.
- **Title:** *Know When to Fold: Distribution-Shift-Aware Selective Prediction for Protein–Ligand Co-Folding*
- **Status:** non-archival, non-anonymous. MoML has no proceedings, so this does not burn the
  journal's novelty and may be submitted alongside it.
- **Venue rules (verified 2026-07-16 at moml.mit.edu/submit) and compliance:** 2 to 4 page PDF
  (this paper: 4 body pages); references and appendices do not count toward the limit; optional
  LaTeX template via Overleaf (we reproduce the style in a self-contained preamble, no external
  `.sty` dependency). Abstract is a single 6-sentence paragraph; tables use booktabs with no
  vertical rules and captions above; figures (none here) would caption below. **Deadline
  2026-09-01 23:59 AOE**, decisions 2026-09-08, in-person poster 2026-10-14.
- **Scope:** the co-folding reliability story only — the gate, the exchangeability break, the
  group-conditional and weighted repair, the combined reliability score, the pose-agreement
  upgrade, and the non-circular downstream payoff. It deliberately omits the theorem's proof, the
  general-benchmark validation, the feasibility frontier, and the certifier-matching result; those
  are the archival paper's weight.

## What is deliberately in neither

The D1 label-free danger floor (`sec:danger` in git history) was cut from both after a strategy
review: its "label-free" claim rested on a crystal-derived frame a prospective caller lacks, and it
bounds the ensemble rather than the deployed model. The full D1/D2 work is preserved at git branch
`d1-d2-full` and is available for the interview record, not for the page.

## Build both

```
tectonic paper/foldgate_journal.tex -o paper/          # -> paper/foldgate_journal.pdf (~8pp)
python paper/build_pdf.py paper/moml2026_shortpaper.md # -> paper/moml2026_shortpaper.pdf (4pp)
```
