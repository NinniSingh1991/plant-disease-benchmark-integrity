## plantdoc (clean test n = 222)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |
|---|---|---|---|---|---|
| effnet_ft | keep | 63.51 ± 2.38 | 61.90 ± 2.06 | 53.59 ± 7.09 | 0.193 |
| clip_l | keep | 67.57 ± 1.62 | 64.82 ± 2.15 | 66.27 ± 3.42 | 0.066 |
| dinov2_l | keep | 77.33 ± 0.52 | 75.96 ± 0.58 | 77.26 ± 1.57 | 0.061 |
| fusion2 | keep | 79.28 ± 1.19 | 76.99 ± 1.16 | 78.12 ± 1.62 | 0.051 |
| fusion2 | drop | 79.13 ± 0.26 | 76.64 ± 0.36 | 77.29 ± 0.65 | 0.060 |
| fusion2 | soft | 80.03 ± 0.69 | 77.67 ± 0.89 | 79.58 ± 2.39 | 0.061 |
| fusion3 | keep | 78.23 ± 1.38 | 76.04 ± 1.37 | 76.22 ± 2.57 | 0.113 |
| fusion3 | drop | 77.63 ± 1.04 | 75.31 ± 0.97 | 75.49 ± 1.92 | 0.107 |
| fusion3 | soft | 77.63 ± 1.45 | 75.41 ± 1.54 | 75.59 ± 1.01 | 0.106 |
| vit_s_ft | keep | 77.18 ± 0.69 | 74.85 ± 1.03 | 74.44 ± 1.40 | 0.093 |

H1, fusion vs best single model (dinov2_l), and vs fine-tuned EfficientNet:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/keep vs dinov2_l | +1.03 [-4.04, +6.29] | 12/20, 0.215 |
| fusion2/keep vs effnet_ft | +15.09 [+9.03, +21.44] | 13/42, 0.000 |
| fusion2/soft vs dinov2_l | +1.71 [-3.33, +7.20] | 12/20, 0.215 |
| fusion2/soft vs effnet_ft | +15.77 [+10.40, +22.32] | 13/42, 0.000 |
| fusion3/keep vs dinov2_l | +0.08 [-4.90, +5.27] | 14/22, 0.243 |
| fusion3/keep vs effnet_ft | +14.14 [+10.14, +18.97] | 3/32, 0.000 |
| fusion3/soft vs dinov2_l | -0.55 [-5.01, +4.82] | 14/22, 0.243 |
| fusion3/soft vs effnet_ft | +13.51 [+9.70, +18.15] | 3/32, 0.000 |

H2, label treatment:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2: keep → soft | +0.68 [-0.83, +2.21] | 1/1, 1.000 |
| fusion2: drop → soft | +1.03 [-1.00, +3.30] | 2/5, 0.453 |
| fusion2: keep → drop | -0.35 [-2.33, +1.30] | 4/1, 0.375 |
| fusion3: keep → soft | -0.63 [-2.25, +0.62] | 2/2, 1.000 |
| fusion3: drop → soft | +0.10 [-1.43, +1.84] | 1/1, 1.000 |
| fusion3: keep → drop | -0.73 [-2.36, +0.70] | 2/2, 1.000 |

H3, ranking by macro-F1 (clean n = 222 vs official n = 236):

| Rank | Clean test | Official test |
|---|---|---|
| 1 | fusion2/soft 77.67 | fusion2/soft 76.24 |
| 2 | fusion2/keep 76.99 | fusion2/drop 76.16 |
| 3 | fusion2/drop 76.64 | fusion2/keep 75.55 |
| 4 | fusion3/keep 76.04 | fusion3/keep 74.42 |
| 5 | dinov2_l/keep 75.96 | dinov2_l/keep 74.23 |
| 6 | fusion3/soft 75.41 | fusion3/drop 73.97 |
| 7 | fusion3/drop 75.31 | fusion3/soft 73.91 |
| 8 | clip_l/keep 64.82 | clip_l/keep 64.08 |
| 9 | effnet_ft/keep 61.90 | effnet_ft/keep 61.02 |

## paddydoctor (clean test n = 2086)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |
|---|---|---|---|---|---|
| effnet_ft | keep | 97.68 ± 0.06 | 97.28 ± 0.18 | 96.46 ± 0.42 | 0.012 |
| clip_l | keep | 86.67 ± 0.19 | 84.41 ± 0.22 | 77.73 ± 0.27 | 0.047 |
| dinov2_l | keep | 90.62 ± 0.11 | 89.11 ± 0.15 | 84.52 ± 0.28 | 0.058 |
| fusion2 | keep | 93.42 ± 0.17 | 92.77 ± 0.13 | 91.19 ± 0.65 | 0.030 |
| fusion2 | drop | 92.91 ± 0.17 | 91.99 ± 0.36 | 90.11 ± 1.82 | 0.034 |
| fusion2 | soft | 93.61 ± 0.20 | 92.90 ± 0.15 | 92.01 ± 0.52 | 0.033 |
| fusion3 | keep | 97.94 ± 0.25 | 97.67 ± 0.31 | 96.90 ± 0.78 | 0.010 |
| fusion3 | drop | 97.94 ± 0.25 | 97.71 ± 0.31 | 97.19 ± 0.95 | 0.008 |
| fusion3 | soft | 98.13 ± 0.14 | 97.93 ± 0.11 | 97.36 ± 0.76 | 0.009 |

H1, fusion vs best single model (effnet_ft), and vs fine-tuned EfficientNet:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/keep vs effnet_ft | -4.51 [-5.58, -3.42] | 105/13, 0.000 |
| fusion2/soft vs effnet_ft | -4.37 [-5.41, -3.41] | 103/15, 0.000 |
| fusion3/keep vs effnet_ft | +0.40 [-0.01, +0.79] | 6/7, 1.000 |
| fusion3/soft vs effnet_ft | +0.66 [+0.34, +1.01] | 4/8, 0.388 |

H2, label treatment:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2: keep → soft | +0.13 [-0.44, +0.68] | 12/16, 0.572 |
| fusion2: drop → soft | +0.92 [+0.45, +1.42] | 5/21, 0.002 |
| fusion2: keep → drop | -0.78 [-1.49, -0.07] | 26/14, 0.081 |
| fusion3: keep → soft | +0.26 [+0.06, +0.46] | 2/5, 0.453 |
| fusion3: drop → soft | +0.22 [+0.03, +0.43] | 3/1, 0.625 |
| fusion3: keep → drop | +0.03 [-0.22, +0.27] | 2/7, 0.180 |

## cassava2020 (clean test n = 4282)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |
|---|---|---|---|---|---|
| effnet_ft | keep | 85.59 ± 0.11 | 74.09 ± 0.58 | 55.20 ± 3.32 | 0.084 |
| clip_l | keep | 82.58 ± 0.21 | 70.40 ± 0.25 | 53.36 ± 0.26 | 0.020 |
| dinov2_l | keep | 84.67 ± 0.06 | 72.61 ± 0.14 | 52.29 ± 0.46 | 0.012 |
| fusion2 | keep | 85.65 ± 0.20 | 74.86 ± 0.28 | 57.65 ± 0.70 | 0.011 |
| fusion2 | drop | 85.59 ± 0.35 | 74.76 ± 0.57 | 57.65 ± 0.95 | 0.010 |
| fusion2 | soft | 85.63 ± 0.20 | 74.83 ± 0.29 | 57.65 ± 0.70 | 0.011 |
| fusion3 | keep | 84.62 ± 0.46 | 72.71 ± 0.83 | 53.98 ± 1.91 | 0.115 |
| fusion3 | drop | 84.58 ± 0.47 | 72.70 ± 0.80 | 54.28 ± 1.15 | 0.116 |
| fusion3 | soft | 84.62 ± 0.46 | 72.71 ± 0.83 | 53.98 ± 1.91 | 0.115 |

H1, fusion vs best single model (effnet_ft), and vs fine-tuned EfficientNet:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/keep vs effnet_ft | +0.78 [-0.80, +2.37] | 260/207, 0.016 |
| fusion2/soft vs effnet_ft | +0.75 [-0.75, +2.38] | 261/208, 0.016 |
| fusion3/keep vs effnet_ft | -1.37 [-2.01, -0.73] | 87/59, 0.025 |
| fusion3/soft vs effnet_ft | -1.37 [-2.05, -0.76] | 87/58, 0.020 |

H2, label treatment:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2: keep → soft | -0.03 [-0.07, -0.00] | 2/2, 1.000 |
| fusion2: drop → soft | +0.07 [-0.19, +0.30] | 8/9, 1.000 |
| fusion2: keep → drop | -0.10 [-0.34, +0.15] | 11/10, 1.000 |
| fusion3: keep → soft | +0.00 [+0.00, +0.00] | 1/0, 1.000 |
| fusion3: drop → soft | +0.01 [-0.19, +0.19] | 4/3, 1.000 |
| fusion3: keep → drop | -0.01 [-0.19, +0.17] | 4/4, 1.000 |

## rice_leaf (clean test n = 769)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |
|---|---|---|---|---|---|
| effnet_ft | keep | 98.70 ± 0.23 | 98.69 ± 0.23 | 98.94 ± 0.23 | 0.010 |
| clip_l | keep | 88.08 ± 0.42 | 88.01 ± 0.40 | 88.13 ± 0.40 | 0.056 |
| dinov2_l | keep | 90.81 ± 0.27 | 90.72 ± 0.27 | 92.74 ± 1.27 | 0.056 |
| fusion2 | keep | 93.71 ± 0.54 | 93.67 ± 0.54 | 95.91 ± 1.27 | 0.041 |
| fusion2 | drop | 93.54 ± 0.33 | 93.49 ± 0.33 | 96.31 ± 0.60 | 0.042 |
| fusion2 | soft | 93.50 ± 0.68 | 93.45 ± 0.67 | 95.51 ± 0.99 | 0.043 |
| fusion3 | keep | 98.70 ± 0.23 | 98.69 ± 0.23 | 99.21 ± 0.00 | 0.007 |
| fusion3 | drop | 98.66 ± 0.38 | 98.65 ± 0.38 | 99.21 ± 0.00 | 0.009 |
| fusion3 | soft | 98.70 ± 0.34 | 98.69 ± 0.35 | 99.21 ± 0.00 | 0.009 |

H1, fusion vs best single model (effnet_ft), and vs fine-tuned EfficientNet:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/keep vs effnet_ft | -5.03 [-6.74, -3.47] | 40/3, 0.000 |
| fusion2/soft vs effnet_ft | -5.24 [-6.95, -3.72] | 48/3, 0.000 |
| fusion3/keep vs effnet_ft | -0.00 [-0.38, +0.41] | 2/1, 1.000 |
| fusion3/soft vs effnet_ft | -0.00 [-0.38, +0.36] | 3/1, 0.625 |

H2, label treatment:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2: keep → soft | -0.22 [-0.62, +0.13] | 8/0, 0.008 |
| fusion2: drop → soft | -0.04 [-0.62, +0.57] | 8/0, 0.008 |
| fusion2: keep → drop | -0.18 [-0.74, +0.42] | 3/3, 1.000 |
| fusion3: keep → soft | +0.00 [-0.12, +0.13] | 1/0, 1.000 |
| fusion3: drop → soft | +0.04 [-0.01, +0.14] | 1/0, 1.000 |
| fusion3: keep → drop | -0.04 [-0.20, +0.09] | 1/1, 1.000 |

## plantwild (clean test n = 3744)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE |
|---|---|---|---|---|---|
| effnet_ft | keep | 70.24 ± 0.24 | 66.78 ± 0.35 | 55.35 ± 1.27 | 0.141 |
| clip_l | keep | 72.41 ± 0.24 | 69.53 ± 0.37 | 57.69 ± 1.01 | 0.051 |
| dinov2_l | keep | 78.13 ± 0.23 | 75.37 ± 0.25 | 64.24 ± 0.41 | 0.039 |
| fusion2 | keep | 79.65 ± 0.12 | 77.36 ± 0.16 | 67.71 ± 0.67 | 0.075 |
| fusion2 | drop | 80.09 ± 0.10 | 77.76 ± 0.22 | 68.50 ± 0.40 | 0.081 |
| fusion2 | soft | 79.86 ± 0.30 | 77.59 ± 0.36 | 68.42 ± 0.46 | 0.080 |
| fusion3 | keep | 77.56 ± 0.20 | 75.02 ± 0.40 | 66.59 ± 1.57 | 0.123 |
| fusion3 | drop | 77.96 ± 0.48 | 75.38 ± 0.54 | 66.78 ± 1.31 | 0.125 |
| fusion3 | soft | 77.73 ± 0.09 | 75.03 ± 0.20 | 66.27 ± 1.06 | 0.126 |

H1, fusion vs best single model (dinov2_l), and vs fine-tuned EfficientNet:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/keep vs dinov2_l | +1.98 [+0.90, +3.14] | 173/225, 0.010 |
| fusion2/keep vs effnet_ft | +10.58 [+9.17, +12.16] | 224/471, 0.000 |
| fusion2/soft vs dinov2_l | +2.22 [+1.12, +3.39] | 162/223, 0.002 |
| fusion2/soft vs effnet_ft | +10.81 [+9.48, +12.45] | 218/474, 0.000 |
| fusion3/keep vs dinov2_l | -0.35 [-1.70, +1.20] | 247/281, 0.151 |
| fusion3/keep vs effnet_ft | +8.25 [+7.36, +9.28] | 68/297, 0.000 |
| fusion3/soft vs dinov2_l | -0.34 [-1.67, +0.90] | 245/288, 0.069 |
| fusion3/soft vs effnet_ft | +8.26 [+7.41, +9.41] | 66/304, 0.000 |

H2, label treatment:

| Comparison | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2: keep → soft | +0.23 [-0.13, +0.61] | 25/34, 0.298 |
| fusion2: drop → soft | -0.17 [-0.72, +0.37] | 59/46, 0.241 |
| fusion2: keep → drop | +0.40 [-0.22, +1.03] | 49/71, 0.055 |
| fusion3: keep → soft | +0.01 [-0.37, +0.38] | 35/44, 0.368 |
| fusion3: drop → soft | -0.35 [-0.76, +0.06] | 36/30, 0.539 |
| fusion3: keep → drop | +0.36 [-0.10, +0.80] | 39/54, 0.146 |
