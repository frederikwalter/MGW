# HESTA and MOSTA: data overview

This note lists, for every slice of the human (HESTA) and mouse (MOSTA) embryo datasets, the number of spots, the number of genes and the number of tissue labels. The tables come first per stage, then per slice.

**Source.** The numbers come from [`data_analysis/dataset_overview.py`](../data_analysis/dataset_overview.py), run on tueilnt-sim03 on 2026-10-29 over all files in `/nas/lnt/ga79jin/MGW/data/{hesta,mosta}` (78 HESTA, 53 MOSTA). The full per-slice output, including the list of label names per slice, is in [`data_analysis/dataset_overview.csv`](../data_analysis/dataset_overview.csv). To regenerate:

```bash
uv run python data_analysis/dataset_overview.py --data-dir /nas/lnt/ga79jin/MGW/data --jobs 6
```

The script reads only `X`'s sparse indices and the label column with h5py, so each worker needs at most about 1 GB.

**Columns.**

- **Spots**: rows of the slice (`n_obs`), i.e. Stereo-seq bins.
- **Genes (panel)**: columns of the slice (`n_vars`). The gene panel differs from slice to slice in both datasets.
- **Genes (detected)**: genes with at least one nonzero count in the slice.
- **Labels**: distinct tissue labels present in the slice. HESTA uses `obs['celltype']` and MOSTA uses `obs['annotation']`. Every spot has a label in every slice; neither dataset has an "unknown" class.

Slice names are file stems: `CS17_E1S4` is `CS17_E1S4_HESTA.h5ad`, and `E12.5_E1S1` is `E12.5_E1S1.MOSTA.h5ad`.

## Summary by stage

### HESTA (human)

| Stage | Embryos | Slices | Spots per slice (min–max) | Genes per slice (min–max) | Labels per slice (min–max) | Distinct labels in stage |
| --- | --- | --- | --- | --- | --- | --- |
| CS12-13 | 2 | 12 | 3,860–14,843 | 32,712–42,266 | 7–19 | 25 |
| CS14-15 | 3 | 18 | 16,470–37,379 | 39,623–43,051 | 12–20 | 28 |
| CS17 | 1 | 10 | 44,531–78,155 | 43,324–46,371 | 14–20 | 27 |
| CS18 | 1 | 6 | 77,312–161,900 | 45,743–47,028 | 9–18 | 25 |
| CS19 | 2 | 7 | 147,865–263,815 | 47,570–52,304 | 17–21 | 26 |
| CS20 | 2 | 11 | 175,487–289,486 | 46,473–54,056 | 14–24 | 29 |
| CS23 | 2 | 13 | 531,126–825,115 | 49,753–53,311 | 18–25 | 31 |
| **all** | | **77** | 3,860–825,115 (Σ 14.7 M) | 32,712–54,056 | 7–25 | 52 |

### MOSTA (mouse)

| Stage | Embryos | Slices | Spots per slice (min–max) | Genes per slice (min–max) | Labels per slice (min–max) | Distinct labels in stage |
| --- | --- | --- | --- | --- | --- | --- |
| E9.5 | 2 | 5 | 4,356–5,913 | 23,398–25,568 | 12–14 | 18 |
| E10.5 | 2 | 4 | 8,494–18,670 | 22,385–25,647 | 11–18 | 23 |
| E11.5 | 1 | 4 | 27,455–30,775 | 25,485–26,854 | 18–19 | 25 |
| E12.5 | 2 | 6 | 23,640–55,295 | 17,337–27,810 | 16–23 | 30 |
| E13.5 | 1 | 4 | 64,584–84,811 | 27,732–28,408 | 17–19 | 23 |
| E14.5 | 2 | 7 | 92,764–112,086 | 18,582–29,245 | 16–26 | 29 |
| E15.5 | 2 | 5 | 72,944–113,350 | 26,921–28,798 | 19–26 | 30 |
| E16.5 | 2 | 18 | 59,125–155,741 | 27,588–29,034 | 10–26 | 31 |
| **all** | | **53** | 4,356–155,741 (Σ 4.1 M) | 17,337–29,245 | 10–26 | 49 |

## Observations

- **Size.** HESTA is about 3.6× larger than MOSTA (14.7 M against 4.1 M spots). Slice size grows steeply with stage in both datasets: from about 4–15 k spots at CS12–13 / E9.5 to 0.5–0.8 M spots per HESTA slice at CS23. Every CS23 slice is larger than any MOSTA slice, so GW-type methods need subsampling or pooling there.
- **Gene panels.** HESTA slices have 33–54 k genes, MOSTA slices 17–29 k. HESTA panels include genes with no counts in that slice (median 576 per slice, up to 4,422 in CS12-13_E2S8). MOSTA panels were already filtered to detected genes; the only exceptions are E9.5_E2S3 (160 undetected) and E16.5_E1S3 (3 undetected). Because panels differ between slices, alignments within one dataset also have to intersect genes first.
- **Panel outliers.** E12.5_E1S3 (17,337 genes) and E14.5_E1S3 (18,582) have about a third fewer genes than their neighbours (26–29 k). E12.5_E1S3 is the recommended within-species E12.5 slice in [hesta_mosta_slide_pairs.md](hesta_mosta_slide_pairs.md), so check the shared gene count before using it.
- **Labels.** The two vocabularies are similar in size (52 HESTA, 49 MOSTA) but are not harmonized: they differ in capitalization ("Spinal Cord" against "Spinal cord") and in granularity ("Gut" / "Primitive Gut" against "GI tract" / "Primitive gut tube", "Ganglion" against "Dorsal root ganglion", "Skeletal Muscle" against "Muscle" / "Smooth muscle"). Labels found in only one dataset include Erythroblasts, Limb Bud, Mesonephron, Somite and Gonad (HESTA), and AGM, Olfactory epithelium, Adipose tissue, Sympathetic nerve and Ovary / Testis (MOSTA). A label map is needed before computing any cross-species label score.
- **Label count per slice.** 95 % of HESTA and 89 % of MOSTA slices have 12–24 labels. The sparse ones are CS12-13_E1S1 and CS12-13_E1S4 (7 each), CS18_E1S6 (9), E16.5_E2S1 (10) and E10.5_E1S2 (11). CS18_E1S6 and E16.5_E2S1 are also excluded in [hesta_mosta_slide_pairs.md](hesta_mosta_slide_pairs.md).
- **Combined 3D file.** `CS12-13_E2_3D_HESTA.h5ad` is not a single slice and is left out of the tables above: 106,063 spots, 23 labels and only 5 columns in `X` (not checked further).

## Per-slice tables

### HESTA

| Slice | Spots | Genes (panel) | Genes (detected) | Labels |
| --- | ---: | ---: | ---: | ---: |
| CS12-13_E1S1 | 5,819 | 32,712 | 32,017 | 7 |
| CS12-13_E1S2 | 7,405 | 35,421 | 34,236 | 12 |
| CS12-13_E1S3 | 6,824 | 36,612 | 35,534 | 12 |
| CS12-13_E1S4 | 3,860 | 33,970 | 32,821 | 7 |
| CS12-13_E2S1 | 12,138 | 40,249 | 37,584 | 18 |
| CS12-13_E2S2 | 11,123 | 40,086 | 37,580 | 15 |
| CS12-13_E2S3 | 13,257 | 42,266 | 40,595 | 19 |
| CS12-13_E2S4 | 12,865 | 38,786 | 37,509 | 18 |
| CS12-13_E2S5 | 11,259 | 37,481 | 35,308 | 18 |
| CS12-13_E2S6 | 14,843 | 38,026 | 36,828 | 15 |
| CS12-13_E2S7 | 9,603 | 37,220 | 35,213 | 16 |
| CS12-13_E2S8 | 8,906 | 38,410 | 33,988 | 13 |
| CS14-15_E1S1 | 18,374 | 41,231 | 40,670 | 13 |
| CS14-15_E1S2 | 21,014 | 40,752 | 40,156 | 15 |
| CS14-15_E1S3 | 20,637 | 41,490 | 40,912 | 15 |
| CS14-15_E1S4 | 22,476 | 40,846 | 40,370 | 19 |
| CS14-15_E2S1 | 16,470 | 42,580 | 42,069 | 18 |
| CS14-15_E2S2 | 17,782 | 41,775 | 40,821 | 16 |
| CS14-15_E2S3 | 19,800 | 43,042 | 42,436 | 20 |
| CS14-15_E2S4 | 16,872 | 41,287 | 40,536 | 17 |
| CS14-15_E2S5 | 21,129 | 41,613 | 40,924 | 16 |
| CS14-15_E2S6 | 19,662 | 40,893 | 39,931 | 16 |
| CS14-15_E2S7 | 23,195 | 41,385 | 40,911 | 16 |
| CS14-15_E2S8 | 19,803 | 40,804 | 39,683 | 14 |
| CS14-15_E2S9 | 18,797 | 40,069 | 38,790 | 12 |
| CS14-15_E2S10 | 22,203 | 39,623 | 38,467 | 15 |
| CS14-15_E3S1 | 25,535 | 43,051 | 42,022 | 20 |
| CS14-15_E3S2 | 31,576 | 42,396 | 41,628 | 20 |
| CS14-15_E3S3 | 32,816 | 41,539 | 40,963 | 19 |
| CS14-15_E3S4 | 37,379 | 41,356 | 40,605 | 15 |
| CS17_E1S1 | 44,531 | 43,559 | 43,139 | 15 |
| CS17_E1S2 | 47,670 | 44,532 | 43,900 | 18 |
| CS17_E1S3 | 52,366 | 45,339 | 44,784 | 14 |
| CS17_E1S4 | 59,423 | 45,516 | 45,134 | 16 |
| CS17_E1S5 | 62,655 | 44,351 | 43,459 | 15 |
| CS17_E1S6 | 56,279 | 46,371 | 45,246 | 18 |
| CS17_E1S7 | 65,443 | 44,092 | 43,406 | 20 |
| CS17_E1S8 | 66,586 | 44,254 | 43,566 | 17 |
| CS17_E1S9 | 65,607 | 46,144 | 45,635 | 18 |
| CS17_E1S10 | 78,155 | 43,324 | 42,913 | 17 |
| CS18_E1S1 | 77,312 | 46,688 | 46,003 | 15 |
| CS18_E1S2 | 112,705 | 46,711 | 46,079 | 18 |
| CS18_E1S3 | 132,277 | 47,028 | 46,326 | 13 |
| CS18_E1S4 | 152,371 | 46,814 | 46,451 | 14 |
| CS18_E1S5 | 161,900 | 45,872 | 45,567 | 14 |
| CS18_E1S6 | 108,726 | 45,743 | 45,316 | 9 |
| CS19_E1S1 | 147,865 | 51,924 | 51,096 | 18 |
| CS19_E1S2 | 155,811 | 51,245 | 50,695 | 18 |
| CS19_E1S3 | 160,662 | 52,304 | 51,784 | 17 |
| CS19_E2S1 | 247,960 | 51,116 | 50,834 | 21 |
| CS19_E2S2 | 247,783 | 51,257 | 51,030 | 19 |
| CS19_E2S3 | 263,202 | 47,570 | 47,242 | 17 |
| CS19_E2S4 | 263,815 | 48,476 | 48,290 | 17 |
| CS20_E1S1 | 175,487 | 47,415 | 46,935 | 15 |
| CS20_E1S2 | 197,466 | 47,736 | 47,111 | 14 |
| CS20_E1S3 | 208,433 | 47,415 | 46,919 | 17 |
| CS20_E1S4 | 253,104 | 48,540 | 47,995 | 19 |
| CS20_E1S5 | 263,538 | 46,473 | 46,100 | 21 |
| CS20_E1S6 | 248,690 | 51,890 | 51,486 | 21 |
| CS20_E1S7 | 265,988 | 52,248 | 51,775 | 21 |
| CS20_E2S1 | 265,120 | 53,727 | 53,076 | 20 |
| CS20_E2S2 | 289,486 | 52,809 | 52,193 | 21 |
| CS20_E2S3 | 286,933 | 53,786 | 53,280 | 19 |
| CS20_E2S4 | 269,429 | 54,056 | 53,552 | 24 |
| CS23_E1S1 | 554,306 | 52,983 | 52,678 | 20 |
| CS23_E1S2 | 539,831 | 53,185 | 52,818 | 25 |
| CS23_E1S3 | 531,126 | 52,125 | 51,904 | 20 |
| CS23_E2S1 | 662,769 | 49,848 | 49,589 | 19 |
| CS23_E2S2 | 655,242 | 49,753 | 49,753 | 19 |
| CS23_E2S3 | 637,053 | 51,169 | 50,572 | 19 |
| CS23_E2S4 | 633,615 | 50,065 | 50,065 | 18 |
| CS23_E2S5 | 671,089 | 50,470 | 50,045 | 20 |
| CS23_E2S6 | 727,736 | 51,014 | 50,600 | 21 |
| CS23_E2S7 | 712,097 | 50,813 | 50,347 | 20 |
| CS23_E2S8 | 825,115 | 53,311 | 53,094 | 22 |
| CS23_E2S9 | 747,172 | 52,509 | 52,180 | 21 |
| CS23_E2S10 | 733,507 | 52,365 | 51,984 | 23 |

### MOSTA

| Slice | Spots | Genes (panel) | Genes (detected) | Labels |
| --- | ---: | ---: | ---: | ---: |
| E9.5_E1S1 | 5,913 | 25,568 | 25,568 | 12 |
| E9.5_E2S1 | 5,292 | 23,756 | 23,756 | 14 |
| E9.5_E2S2 | 4,356 | 24,107 | 24,107 | 13 |
| E9.5_E2S3 | 5,059 | 24,238 | 24,078 | 13 |
| E9.5_E2S4 | 5,797 | 23,398 | 23,398 | 13 |
| E10.5_E1S1 | 18,408 | 25,201 | 25,201 | 13 |
| E10.5_E1S2 | 18,647 | 25,544 | 25,544 | 11 |
| E10.5_E1S3 | 18,670 | 25,647 | 25,647 | 16 |
| E10.5_E2S1 | 8,494 | 22,385 | 22,385 | 18 |
| E11.5_E1S1 | 30,124 | 26,854 | 26,854 | 19 |
| E11.5_E1S2 | 30,756 | 25,485 | 25,485 | 18 |
| E11.5_E1S3 | 27,455 | 25,741 | 25,741 | 19 |
| E11.5_E1S4 | 30,775 | 25,798 | 25,798 | 19 |
| E12.5_E1S1 | 51,365 | 27,810 | 27,810 | 23 |
| E12.5_E1S2 | 47,436 | 27,393 | 27,393 | 20 |
| E12.5_E1S3 | 49,908 | 17,337 | 17,337 | 17 |
| E12.5_E1S4 | 55,295 | 27,330 | 27,330 | 19 |
| E12.5_E1S5 | 45,669 | 26,837 | 26,837 | 16 |
| E12.5_E2S1 | 23,640 | 26,688 | 26,688 | 18 |
| E13.5_E1S1 | 77,369 | 28,408 | 28,408 | 19 |
| E13.5_E1S2 | 64,584 | 28,208 | 28,208 | 19 |
| E13.5_E1S3 | 84,811 | 27,841 | 27,841 | 18 |
| E13.5_E1S4 | 73,168 | 27,732 | 27,732 | 17 |
| E14.5_E1S1 | 102,519 | 28,463 | 28,463 | 26 |
| E14.5_E1S2 | 102,977 | 28,766 | 28,766 | 21 |
| E14.5_E1S3 | 92,928 | 18,582 | 18,582 | 16 |
| E14.5_E1S4 | 105,857 | 29,245 | 29,245 | 21 |
| E14.5_E1S5 | 101,353 | 28,150 | 28,150 | 21 |
| E14.5_E2S1 | 92,764 | 26,944 | 26,944 | 16 |
| E14.5_E2S2 | 112,086 | 28,193 | 28,193 | 17 |
| E15.5_E1S1 | 113,350 | 28,798 | 28,798 | 26 |
| E15.5_E1S2 | 109,811 | 28,154 | 28,154 | 22 |
| E15.5_E1S3 | 104,042 | 28,488 | 28,488 | 19 |
| E15.5_E1S4 | 104,845 | 28,635 | 28,635 | 19 |
| E15.5_E2S1 | 72,944 | 26,921 | 26,921 | 21 |
| E16.5_E1S1 | 121,767 | 28,204 | 28,204 | 25 |
| E16.5_E1S2 | 120,842 | 28,930 | 28,930 | 19 |
| E16.5_E1S3 | 155,741 | 28,579 | 28,576 | 23 |
| E16.5_E1S4 | 135,973 | 29,034 | 29,034 | 20 |
| E16.5_E1S5 | 152,559 | 28,291 | 28,291 | 24 |
| E16.5_E2S1 | 59,125 | 27,831 | 27,831 | 10 |
| E16.5_E2S2 | 73,922 | 27,762 | 27,762 | 13 |
| E16.5_E2S3 | 110,755 | 28,118 | 28,118 | 15 |
| E16.5_E2S4 | 99,212 | 27,782 | 27,782 | 16 |
| E16.5_E2S5 | 132,481 | 28,555 | 28,555 | 24 |
| E16.5_E2S6 | 137,209 | 28,144 | 28,144 | 21 |
| E16.5_E2S7 | 155,159 | 28,257 | 28,257 | 21 |
| E16.5_E2S8 | 120,854 | 28,701 | 28,701 | 26 |
| E16.5_E2S9 | 129,425 | 28,878 | 28,878 | 24 |
| E16.5_E2S10 | 114,026 | 28,160 | 28,160 | 18 |
| E16.5_E2S11 | 109,399 | 27,945 | 27,945 | 18 |
| E16.5_E2S12 | 94,733 | 27,588 | 27,588 | 15 |
| E16.5_E2S13 | 77,748 | 27,739 | 27,739 | 14 |
