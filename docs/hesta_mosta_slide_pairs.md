# HESTA ↔ MOSTA: which slides to align

This note lists the best slide pairs for two kinds of alignment: human (HESTA) against mouse (MOSTA) at each matched time point, and two different embryos of the same species and stage. Each pair is chosen to match in developmental stage and in mediolateral position. The within-species pairs give the baseline that cross-species differences have to be measured against.

**Sources.** Stage matching follows the table in notes.tex ("Developmental stage correspondence between HESTA and MOSTA"). Mediolateral position is the laterality score $L$ (0 = midline, 1 = far lateral) from `laterality_test/full_run.ipynb`, read from the full-run results in [`laterality/laterality_scores.csv`](../laterality/laterality_scores.csv) (77 HESTA + 52 MOSTA sections). The CSVs in `laterality_test/laterality/` are from the first run (27 HESTA sections, no juxta-axial class) and are out of date.

Slide names are file stems: `CS17_E1S4` is `CS17_E1S4_HESTA.h5ad`, and `E12.5_E1S1` is `E12.5_E1S1.MOSTA.h5ad`.

## Selection rules

1. **Exclusions.** Sections with < 3 % classified spots are dropped, because their score rests on too few spots: CS18_E1S6, CS20_E1S1–S3, CS23_E2S9, E12.5_E1S5, E14.5_E1S3, E14.5_E2S1, E16.5_E2S3, E16.5_E2S12 and E16.5_E2S13. Also dropped are E16.5_E2S1 (no classified spots) and the combined file `CS12-13_E2_3D_HESTA.h5ad`.
2. **Stage windows.** CS12–13 → E9.5, CS14–15 → E10.5, CS17 → E11.5 / E12.5, CS18 → E12.5, CS19 → E12.5 / E13.5, CS20 → E13.5 and CS23 → E14.5. As proposed in the notes' open issues, the windows are extended by **E11.5 for CS14–15** and **E14.5 for CS20**; these pairs are marked *ext.* E15.5 and E16.5 are mouse fetal stages and have no HESTA partner.
3. **Position.** Pairs are ranked by $|\Delta L|$, using the smoothed HESTA score `L_pair` and the raw MOSTA score `L`. Neighbouring sections of one series differ by about 0.05 (median), so $|\Delta L| < 0.05$ is at noise level, 0.05–0.1 is acceptable, and anything > 0.1 is flagged ⚠.
4. **Tie-breakers.** Prefer midline pairs (no mirror ambiguity), then better-labelled sections (higher fraction of classified spots), then the anchor stage over an ambiguous one.
5. **Orientation.** *as is*: both sections have $L < 0.1$, so reflection does not matter. *both*: run the alignment with and without reflecting one slide (the mirror ambiguity of Problem 2). HESTA sections of one embryo on the same relative side must get the same reflection choice.

★ marks the recommended pair for each row.

## Cross-species pairs (HESTA ↔ MOSTA)

| Stage match | HESTA | MOSTA | $L_H$ | $L_M$ | $\lvert\Delta L\rvert$ | Orientation | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CS12–13 ↔ E9.5 (good) | ★ CS12-13_E2S7 | E9.5_E2S1 | 0.476 | 0.476 | 0.000 | both | best-labelled HESTA slide (64 % classified) |
| | CS12-13_E2S8 | E9.5_E2S3 | 0.526 | 0.522 | 0.004 | both | |
| | CS12-13_E1S3 | E9.5_E2S1 | 0.451 | 0.476 | 0.024 | both | most medial section of embryo E1 |
| CS14–15 ↔ E10.5 (good) | ★ CS14-15_E2S8 | E10.5_E1S1 | 0.072 | 0.074 | 0.002 | as is | midline anchor pair |
| | CS14-15_E2S7 | E10.5_E1S1 | 0.046 | 0.074 | 0.027 | as is | midline |
| | CS14-15_E3S4 | E10.5_E2S1 | 0.502 | 0.503 | 0.002 | both | off-midline |
| CS14–15 ↔ E11.5 (*ext.*) | ★ CS14-15_E1S3 | E11.5_E1S2 | 0.733 | 0.734 | 0.001 | both | lateral; E10.5 has nothing above 0.50 |
| CS17 ↔ E11.5 (ambiguous) | ★ CS17_E1S4 | E11.5_E1S4 | 0.773 | 0.771 | 0.001 | both | CS17 has no midline section (min 0.47) |
| CS17 ↔ E12.5 (ambiguous) | ★ CS17_E1S4 | E12.5_E1S1 | 0.773 | 0.765 | 0.008 | both | E12.5 S1 is a representative section |
| | CS17_E1S8 | E12.5_E1S4 | 0.744 | 0.752 | 0.008 | both | CS17_E1S8 is the slide in `demos/hesta_align.ipynb` |
| CS18 ↔ E12.5 (good) | CS18_E1S3 | E12.5_E1S4 | 0.574 | 0.752 | 0.178 ⚠ | both | no usable E12.5 section has $L$ between 0.12 and 0.75 |
| | CS18_E1S1 | E13.5_E1S4 | 0.326 | 0.207 | 0.119 ⚠ | both | fallback, outside the E12.5 window |
| CS19 ↔ E13.5 (ambiguous) | ★ CS19_E2S4 | E13.5_E1S3 | 0.061 | 0.041 | 0.020 | as is | midline |
| | CS19_E2S3 | E13.5_E1S4 | 0.215 | 0.207 | 0.007 | both | |
| CS19 ↔ E12.5 (ambiguous) | CS19_E2S4 | E12.5_E1S3 | 0.061 | 0.103 | 0.042 | both | near-midline; E12.5_E1S3 is just above 0.1 |
| CS20 ↔ E13.5 (good) | ★ CS20_E2S4 | E13.5_E1S4 | 0.272 | 0.207 | 0.065 | both | only option: CS20 ≥ 0.27, E13.5 ≤ 0.21 |
| CS20 ↔ E14.5 (*ext.*) | ★ CS20_E2S3 | E14.5_E1S4 | 0.471 | 0.475 | 0.003 | both | |
| CS23 ↔ E14.5 (weak) | ★ CS23_E2S7 | E14.5_E1S1 | 0.565 | 0.595 | 0.030 | both | best-labelled close pair |
| | CS23_E2S8 | E14.5_E1S5 | 0.630 | 0.615 | 0.015 | both | |
| — ↔ E15.5, E16.5 | — | — | | | | | mouse fetal period, no HESTA counterpart |

## Within-species pairs (two embryos, same stage)

The relative side of a section is known only within one HESTA embryo, so every pair across embryos is run in *both* orientations unless both sections lie at the midline.

| Stage | Embryo 1 | Embryo 2 | $L_1$ | $L_2$ | $\lvert\Delta L\rvert$ | Note |
| --- | --- | --- | --- | --- | --- | --- |
| CS12–13 | ★ CS12-13_E1S3 | CS12-13_E2S4 | 0.451 | 0.451 | 0.000 | well labelled (43 % / 57 %) |
| CS14–15 | ★ CS14-15_E2S5 | CS14-15_E3S4 | 0.557 | 0.502 | 0.055 | alt. CS14-15_E1S3 ↔ CS14-15_E2S1 (0.057); E3 never goes below 0.33, so there is no midline pair |
| CS17 | — | — | | | | single embryo |
| CS18 | — | — | | | | single embryo |
| CS19 | CS19_E1S1 | CS19_E2S1 | 0.555 | 0.460 | 0.094 | weak: E1 has no medial section |
| CS20 | ★ CS20_E1S6 | CS20_E2S2 | 0.657 | 0.640 | 0.017 | |
| CS23 | ★ CS23_E1S2 | CS23_E2S7 | 0.570 | 0.565 | 0.005 | |
| E9.5 | E9.5_E1S1 | E9.5_E2S1 | 0.370 | 0.476 | 0.105 ⚠ | E1 has a single section |
| E10.5 | E10.5_E1S3 | E10.5_E2S1 | 0.347 | 0.503 | 0.156 ⚠ | E2 has a single section |
| E11.5 | — | — | | | | single embryo |
| E12.5 | ★ E12.5_E1S3 | E12.5_E2S1 | 0.103 | 0.115 | 0.012 | near-midline |
| E13.5 | — | — | | | | single embryo |
| E14.5 | E14.5_E1S2 | E14.5_E2S2 | 0.797 | 0.724 | 0.073 | both poorly labelled (< 5 % classified) |
| E15.5 | E15.5_E1S4 | E15.5_E2S1 | 0.830 | 0.868 | 0.038 | no cross-species use |
| E16.5 | ★ E16.5_E1S5 | E16.5_E2S6 | 0.144 | 0.114 | 0.030 | no cross-species use; E16.5_E2 is the ordered series (V at S7) |

## Caveats

- **Scores across datasets.** $L$ is more reliable for ordering sections within an embryo than for comparing HESTA with MOSTA. MOSTA E9.5 has no spinal-cord label, so its $L$ is overestimated, and MOSTA has no limb label, so far-lateral mouse sections look less lateral than they are.
- **Gaps.** CS18 ↔ E12.5 and CS20 ↔ E13.5 have no pair within noise level. E12.5 has only midline-near and lateral usable sections, and E13.5 has only midline-near ones. Most within-species MOSTA pairs are weak, because the second embryo usually has one section.
- **Matching baselines.** A within-species pair at a different depth than the cross-species pair is a weaker baseline. Example: CS14–15 is compared across species at the midline but within species at $L \approx 0.5$.
- **Stage dependence.** The same $L$ does not mean the same physical offset at different stages.
