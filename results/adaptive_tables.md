## plantdoc (test n = 222, validation n = 219, train90 n = 2123, soft-label images 49)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |
|---|---|---|---|---|---|---|
| effnet_ft | keep | 63.81 ± 2.13 | 61.64 ± 3.26 | 52.87 ± 5.46 | 0.195 | 0.081 |
| clip_l | keep | 66.52 ± 1.13 | 63.79 ± 1.24 | 66.48 ± 3.54 | 0.060 | 0.066 |
| dinov2_l | keep | 77.33 ± 2.08 | 76.11 ± 2.32 | 76.63 ± 3.07 | 0.071 | 0.058 |
| fusion2 | keep | 78.83 ± 0.45 | 76.28 ± 0.38 | 81.35 ± 0.95 | 0.060 | 0.064 |
| fusion2 | soft | 78.98 ± 1.13 | 76.66 ± 1.23 | 81.35 ± 1.54 | 0.060 | 0.053 |
| fusion3 | keep | 77.48 ± 2.25 | 75.39 ± 2.08 | 77.14 ± 4.45 | 0.111 | 0.066 |
| fusion3 | soft | 77.93 ± 1.19 | 75.71 ± 1.12 | 78.08 ± 1.70 | 0.108 | 0.074 |
| adaptive | keep | 81.08 ± 0.78 | 78.62 ± 1.04 | 81.77 ± 2.53 | 0.078 | 0.082 |
| adaptive | soft | 80.33 ± 1.13 | 77.81 ± 1.43 | 81.77 ± 2.53 | 0.070 | 0.065 |

Gate (weight of the fine-tuned CNN branch, soft labels): 0.20, 0.20, 0.20, mean 0.20

| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/soft | +1.15 [-1.54, +3.65] | 5/9, 0.424 |
| fusion2/keep | +1.53 [-1.42, +4.36] | 6/10, 0.454 |
| fusion3/soft | +2.10 [-0.89, +5.45] | 6/9, 0.607 |
| effnet_ft/keep | +16.16 [+12.07, +21.26] | 7/37, 0.000 |
| dinov2_l/keep | +1.70 [-3.25, +6.85] | 11/19, 0.200 |
| adaptive/keep | -0.81 [-1.99, +0.42] | 1/1, 1.000 |

## rice_leaf (test n = 769, validation n = 303, train90 n = 2757, soft-label images 2)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |
|---|---|---|---|---|---|---|
| effnet_ft | keep | 98.70 ± 0.13 | 98.70 ± 0.13 | 99.08 ± 0.23 | 0.010 | 0.007 |
| clip_l | keep | 87.47 ± 0.65 | 87.40 ± 0.65 | 87.07 ± 0.23 | 0.053 | 0.044 |
| dinov2_l | keep | 90.68 ± 0.42 | 90.60 ± 0.41 | 93.13 ± 0.61 | 0.056 | 0.019 |
| fusion2 | keep | 93.45 ± 0.30 | 93.40 ± 0.31 | 95.64 ± 0.40 | 0.041 | 0.022 |
| fusion2 | soft | 93.24 ± 0.26 | 93.19 ± 0.25 | 95.51 ± 1.00 | 0.041 | 0.023 |
| fusion3 | keep | 98.57 ± 0.13 | 98.57 ± 0.13 | 99.08 ± 0.23 | 0.010 | 0.007 |
| fusion3 | soft | 98.48 ± 0.08 | 98.48 ± 0.07 | 99.08 ± 0.23 | 0.007 | 0.009 |
| adaptive | keep | 98.53 ± 0.15 | 98.52 ± 0.15 | 99.08 ± 0.23 | 0.008 | 0.011 |
| adaptive | soft | 98.44 ± 0.13 | 98.43 ± 0.13 | 99.08 ± 0.23 | 0.008 | 0.011 |

Gate (weight of the fine-tuned CNN branch, soft labels): 0.40, 0.45, 0.40, mean 0.42

| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/soft | +5.24 [+3.76, +6.93] | 2/42, 0.000 |
| fusion2/keep | +5.03 [+3.56, +6.54] | 2/38, 0.000 |
| fusion3/soft | -0.04 [-0.32, +0.22] | 1/1, 1.000 |
| effnet_ft/keep | -0.26 [-0.84, +0.26] | 6/1, 0.125 |
| dinov2_l/keep | +7.83 [+5.97, +9.88] | 2/60, 0.000 |
| adaptive/keep | -0.09 [-0.24, +0.00] | 0/0, 1.000 |

## paddydoctor (test n = 2086, validation n = 829, train90 n = 7492, soft-label images 130)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |
|---|---|---|---|---|---|---|
| effnet_ft | keep | 96.95 ± 0.24 | 96.73 ± 0.28 | 96.44 ± 0.58 | 0.017 | 0.011 |
| clip_l | keep | 86.07 ± 0.19 | 83.74 ± 0.39 | 76.36 ± 1.17 | 0.050 | 0.019 |
| dinov2_l | keep | 90.59 ± 0.03 | 88.92 ± 0.08 | 84.37 ± 0.22 | 0.066 | 0.017 |
| fusion2 | keep | 93.14 ± 0.08 | 92.53 ± 0.09 | 90.39 ± 0.45 | 0.033 | 0.013 |
| fusion2 | soft | 92.76 ± 0.13 | 92.06 ± 0.23 | 90.69 ± 0.73 | 0.033 | 0.013 |
| fusion3 | keep | 97.14 ± 0.26 | 96.87 ± 0.31 | 96.74 ± 0.69 | 0.013 | 0.009 |
| fusion3 | soft | 97.28 ± 0.19 | 97.04 ± 0.24 | 96.74 ± 0.57 | 0.012 | 0.010 |
| adaptive | keep | 97.44 ± 0.15 | 97.39 ± 0.14 | 97.07 ± 0.66 | 0.008 | 0.008 |
| adaptive | soft | 97.41 ± 0.21 | 97.31 ± 0.23 | 96.94 ± 0.47 | 0.010 | 0.009 |

Gate (weight of the fine-tuned CNN branch, soft labels): 0.45, 0.50, 0.45, mean 0.47

| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/soft | +5.24 [+4.24, +6.41] | 6/108, 0.000 |
| fusion2/keep | +4.77 [+3.69, +5.87] | 7/102, 0.000 |
| fusion3/soft | +0.27 [-0.04, +0.58] | 5/5, 1.000 |
| effnet_ft/keep | +0.58 [+0.26, +0.90] | 8/12, 0.503 |
| dinov2_l/keep | +8.39 [+6.91, +9.83] | 10/152, 0.000 |
| adaptive/keep | -0.09 [-0.25, +0.07] | 0/0, 1.000 |

## plantwild (test n = 3744, validation n = 1443, train90 n = 13355, soft-label images 851)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |
|---|---|---|---|---|---|---|
| effnet_ft | keep | 69.53 ± 0.27 | 66.11 ± 0.06 | 55.19 ± 0.29 | 0.141 | 0.022 |
| clip_l | keep | 72.43 ± 0.39 | 69.25 ± 0.42 | 56.52 ± 0.75 | 0.044 | 0.022 |
| dinov2_l | keep | 77.56 ± 0.31 | 74.39 ± 0.14 | 62.36 ± 0.54 | 0.040 | 0.020 |
| fusion2 | keep | 79.09 ± 0.13 | 76.54 ± 0.20 | 66.20 ± 0.53 | 0.075 | 0.020 |
| fusion2 | soft | 79.16 ± 0.23 | 76.57 ± 0.29 | 66.44 ± 0.33 | 0.080 | 0.022 |
| fusion3 | keep | 77.22 ± 0.23 | 74.29 ± 0.40 | 64.78 ± 1.58 | 0.123 | 0.035 |
| fusion3 | soft | 77.24 ± 0.20 | 74.36 ± 0.33 | 65.03 ± 1.64 | 0.129 | 0.036 |
| adaptive | keep | 80.70 ± 0.27 | 78.03 ± 0.30 | 67.13 ± 0.66 | 0.064 | 0.023 |
| adaptive | soft | 80.62 ± 0.24 | 77.99 ± 0.17 | 67.75 ± 0.17 | 0.069 | 0.020 |

Gate (weight of the fine-tuned CNN branch, soft labels): 0.30, 0.30, 0.25, mean 0.28

| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/soft | +1.42 [+0.68, +2.17] | 50/127, 0.000 |
| fusion2/keep | +1.45 [+0.77, +2.24] | 58/131, 0.000 |
| fusion3/soft | +3.63 [+2.84, +4.51] | 100/180, 0.000 |
| effnet_ft/keep | +11.88 [+10.84, +13.12] | 119/434, 0.000 |
| dinov2_l/keep | +3.60 [+2.49, +4.88] | 142/278, 0.000 |
| adaptive/keep | -0.04 [-0.38, +0.31] | 26/32, 0.512 |

## cassava2020 (test n = 4282, validation n = 1709, train90 n = 15406, soft-label images 0)

| Model | Labels | Accuracy | Macro-F1 | Minority recall | ECE raw | ECE after TS |
|---|---|---|---|---|---|---|
| effnet_ft | keep | 85.21 ± 0.22 | 73.38 ± 0.33 | 55.50 ± 2.00 | 0.093 | 0.013 |
| clip_l | keep | 82.56 ± 0.33 | 70.38 ± 0.35 | 53.06 ± 0.53 | 0.025 | 0.015 |
| dinov2_l | keep | 84.40 ± 0.25 | 72.29 ± 0.51 | 52.29 ± 2.29 | 0.014 | 0.013 |
| fusion2 | keep | 85.08 ± 0.18 | 73.90 ± 0.22 | 55.66 ± 1.91 | 0.009 | 0.018 |
| fusion2 | soft | 85.08 ± 0.18 | 73.90 ± 0.22 | 55.66 ± 1.91 | 0.009 | 0.018 |
| fusion3 | keep | 84.44 ± 0.56 | 72.60 ± 1.03 | 54.74 ± 1.61 | 0.117 | 0.017 |
| fusion3 | soft | 84.44 ± 0.56 | 72.60 ± 1.03 | 54.74 ± 1.61 | 0.117 | 0.017 |
| adaptive | keep | 87.29 ± 0.18 | 76.78 ± 0.44 | 57.34 ± 1.65 | 0.042 | 0.012 |
| adaptive | soft | 87.29 ± 0.18 | 76.78 ± 0.44 | 57.34 ± 1.65 | 0.042 | 0.012 |

Gate (weight of the fine-tuned CNN branch, soft labels): 0.35, 0.30, 0.30, mean 0.32

| Adaptive (soft) vs | Δ macro-F1 [95% CI] | McNemar b/c, p |
|---|---|---|
| fusion2/soft | +2.88 [+1.73, +4.03] | 83/187, 0.000 |
| fusion2/keep | +2.88 [+1.71, +4.10] | 83/187, 0.000 |
| fusion3/soft | +4.19 [+3.25, +5.20] | 64/136, 0.000 |
| effnet_ft/keep | +3.40 [+2.41, +4.48] | 96/135, 0.012 |
| dinov2_l/keep | +4.49 [+3.12, +5.72] | 88/219, 0.000 |
| adaptive/keep | +0.00 [+0.00, +0.00] | 0/0, 1.000 |
