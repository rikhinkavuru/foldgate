# Submission guide — accounts, DOIs, venues (grounded 2026-07-12)

Web-verified against arXiv/bioRxiv/Zenodo help pages and venue sites. Author: Rikhin Kavuru,
Independent Researcher, `rikhin@virahacks.com`. Repo: github.com/rikhinkavuru/foldgate.

---

## 0. The one decision that constrains everything

You **cannot post the same manuscript to both arXiv and bioRxiv** — bioRxiv's rule is that a
paper already on another preprint server will not be posted (and can be withdrawn if caught).
Pick one server of record per paper.

- **arXiv** if you want the ML-community reach (`cs.LG` / `stat.ML` cross-lists). Costs you an
  endorsement step as a first-time, non-institutional author.
- **bioRxiv** if you want the lower barrier (no endorsement) and life-science discovery + the
  B2J direct-to-journal transfer. Loses the ML cross-listing.

Recommendation: for the applied co-folding paper, **bioRxiv** (no endorsement wall, matches the
Digital Discovery target). For a future methods-forward paper aimed at ICLR/AISTATS, **arXiv**
(cs.LG/stat.ML). One paper each, no conflict.

---

## 1. arXiv (q-bio.BM primary, cross-list cs.LG + stat.ML)

**Critical path = endorsement, not the upload.** A `virahacks.com` email is not institutional and
you have no prior arXiv paper, so neither auto-endorsement condition is met (rule updated
2026-01-21). Budget 1–2 weeks to line up a human endorser before you plan to submit.

1. Register at `arxiv.org/user/register` (username is permanent; password needs an underscore).
   Organization = `Independent`, email = `rikhin@virahacks.com`, verify email.
2. Link ORCID at `arxiv.org/user/confirm_orcid_id` **before** v1 so attribution is clean from the start.
3. Endorsement: start a submission with primary `q-bio.BM` → arXiv generates a 6-char endorsement
   code. Browse `arxiv.org/list/q-bio.BM/recent`, open a relevant paper, click "Which authors of
   this paper are endorsers?", email one your ORCID + title/abstract + the code. They enter it at
   `arxiv.org/auth/endorse`. Good asks: RNP / FoldBench / conformal-prediction authors.
4. `q-bio`, `cs`, `stat` are three separate endorsement domains — cross-lists may each prompt for
   endorsement. Clear `q-bio.BM` first; one endorser active in both bio and ML can cover all three.
5. Submit **LaTeX source, not the PDF** (arXiv rejects a TeX-generated PDF). Include the `.bbl`,
   check the compiled preview (that generated PDF is what posts). Filenames are case-sensitive.
6. Primary `q-bio.BM`; add cross-lists one at a time: `cs.LG`, then `stat.ML`. Stop at two.
7. License = **CC BY 4.0** (matches gold-OA journals; irrevocable per version, so choose deliberately).
8. Timing: first submissions sit on a moderation hold 1–4+ business days. Freeze 14:00 ET / announce
   20:00 ET, no announcements Fri/Sat. Do not promise a same-day live link.
9. After journal acceptance: do **not** upload the publisher PDF; instead fill the journal-ref + DOI
   fields on your user page (instant, no new version needed).

## 2. bioRxiv (lower barrier — recommended for the applied paper)

No endorsement, no established co-author, no institutional email required. bioRxiv screens the
manuscript, not your credentials; a solo unaffiliated author is eligible. ~5% rejected at screening.

1. Register at biorxiv.org (free, no APC). Get/link ORCID (orcid.org), contact = `rikhin@virahacks.com`.
2. Affiliation = **"Independent Researcher"** (standard accepted value; do not leave blank or invent one).
3. Frame the paper as **original research** (a benchmark/methods paper with results qualifies) —
   reviews/commentary get bounced at Stage-1 screening.
4. Manuscript: upload a **combined PDF** (convert LaTeX → PDF yourself; bioRxiv does not compile TeX).
   Optionally attach the `.tex` as supplementary.
5. Subject category = **Bioinformatics** (best fit) or Biophysics. There is no "Computational Biology".
   Lead with the protein–ligand biology angle; a pure ML/stats framing risks a scope rejection.
6. License = **CC BY 4.0** (compatible with Digital Discovery / J. Cheminformatics later).
7. Posting is 24–48 h; you get a permanent citable DOI. Versions v1, v2… all stay visible.
8. B2J: after posting, the Author Area offers "Submit to a Journal" → forwards PDF + metadata to
   240+ journals (optional; you can also just cite the preprint DOI in a manual submission).

## 3. Zenodo DOI (software + paper + data)

Pre-flight fix (load-bearing): **`.zenodo.json` overrides `CITATION.cff`** — Zenodo reads only
`.zenodo.json`. Both exist in the repo. **Add your real ORCID to `.zenodo.json`** (`{"name":
"Kavuru, Rikhin", "orcid": "..."}`) — it is currently missing there, and an ORCID only in
`CITATION.cff` will NOT reach the Zenodo record. Mirror it into `CITATION.cff` too. Confirm root
`LICENSE` exists and repo is public. Commit before tagging.

1. Sign up at zenodo.org **with GitHub** (and also link ORCID under profile → Linked accounts).
2. Enable the repo: profile → GitHub (`zenodo.org/account/settings/github/`) → Sync now → toggle
   **rikhinkavuru/foldgate ON**. Must be done **before** cutting the release (no retroactive archiving).
3. Cut the release: GitHub → Releases → Draft → tag `v0.1.0` → Publish
   (`gh release create v0.1.0 --title "foldgate v0.1.0" --notes "…"`). The webhook mints the DOI
   automatically from `.zenodo.json`. GitHub-flow metadata is **not** editable pre-mint — if wrong,
   fix the files and cut `v0.1.1`.
4. Two DOIs appear: a **Concept DOI** (always latest) and a **Version DOI** (pinned to v0.1.0).
   Cite the **Concept DOI** in the README badge, paper "software available at", and `CITATION.cff`;
   cite the **Version DOI** in the reproducibility appendix so a reader gets the exact figure-producing code.
5. Deposit the paper and data as **separate records** (Zenodo's own guidance): New upload →
   Publication/Preprint for `paper/moml2026_foldgate.pdf` (here you *can* Reserve DOI before
   publishing, so you can print the DOI inside the PDF); Dataset for `results/` JSON + figures
   (do **not** include un-committed `data/{raw,processed}`). Link them via Related identifiers
   (`isSupplementTo` / `isSupplementedBy`, `isDerivedFrom` RNP `10.5281/zenodo.14794785`).
6. Add the badge to `README.md` (Concept DOI) and a `@software` entry to `REFERENCES.bib`; cite it
   in a Code & Data Availability section. (Note: the Zenodo grounding referenced the old filename
   `moml2026_shortpaper.pdf`; the current paper is `paper/moml2026_foldgate.pdf`.)

---

## 4. Venue ranking (decision-oriented)

**Two-track plan off one non-archival signal push.** Same paper cannot go to both an archival
journal and an archival conference — split into an applied paper (journal) and a methods paper
(ML conference), sharing one codebase.

### Non-archival now (free signal, burns no novelty)
| Rank | Venue | Deadline | Fit |
|---|---|---|---|
| 1 | **MoML 2026** (MIT) | **Sept 1 2026, 11:59pm AoE** (verified); event Oct 14 | Direct hit; PDF already exists; dual-submission allowed. Submit. |
| 2 | **MLSB @ NeurIPS 2026** | CFP not posted; ~early Oct (approx) | The single most on-target audience (AF3/Boltz/Chai reviewers). Watch mlsb.io. |
| 3 | GenBio / AI4Science @ NeurIPS 2026 | ~Sept–Oct (approx) | Broader; secondary, only if bandwidth. |

### Archival journal (the paper of record)
| Rank | Venue | Decision | OA/APC | Competitive as-is? |
|---|---|---|---|---|
| 1 | **Digital Discovery (RSC)** — recommended | ~45 days | Gold OA, **£2,200** | Best fit; strong-but-borderline as-is (see review doc); solid accept with the 2 additions. |
| 2 | **J. Cheminformatics** | weeks | OA **£1,690** | Strong backup; software-tool framing; cheaper. |
| 3 | **JCIM (ACS)** | weeks | hybrid (free paywalled option) | Comp-chem brand; reviewers may push for more docking baselines. |
| 4 | Bioinformatics (Application Note) | ~23-day median | OA | Fast software citation, but 2–4pp undersells the science. |
| — | Nature Comms / Nature Methods (ceiling) | — | $7,350 | **No** as-is; opens only if prospective validation lands. |

### Methods track (reach, parallel)
| Rank | Venue | Deadline (approx) | Needs |
|---|---|---|---|
| 1 | **ICLR 2027** | paper ~Sept 24 2026 | A clean general theorem + benchmark, co-folding as flagship app. Confirm on iclr.cc. |
| 2 | AISTATS 2027 | abstract ~Jan 15 2027 | Native home for shift/CVaR/DRO conformal; more forgiving than ICLR. |
| 3 | UAI 2027 | ~Feb 2027 | UQ-friendly alternate. |

### Recommended sequence
1. Now → Sept 1: submit MoML (have it).
2. ~Oct: MLSB extended abstract when CFP posts.
3. Oct–Nov: applied paper → **Digital Discovery** (fallback J. Cheminf → JCIM).
4. Parallel: methods paper → ICLR 2027 (fallback AISTATS/UAI).
5. If Mac1 coords release or a prospective screen lands → escalate the combined story to Nature Comms.

Digital Discovery and ICLR are both archival → the same PDF cannot go to both. The journal paper
owns the biology/screening/tool; the conference paper owns the general statistical machinery, each
citing the other.
