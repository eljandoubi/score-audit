# Audit results

Generated from the tables in this folder.

## Human raters

Each essay has two ratings by different raters. Their agreement is the level a scorer is asked to reach. The ICC of the mean of the two says how stable the mean score is.

| score | essays | QWK of the two raters | exact | within 1 point | ICC one rater | ICC mean of two |
|---|---|---|---|---|---|---|
| overall | 6,462 | 0.591 | 0.59 | 0.997 | 0.591 | 0.743 |
| cohesion | 6,462 | 0.497 | 0.50 | 0.974 | 0.497 | 0.664 |
| syntax | 6,462 | 0.511 | 0.53 | 0.982 | 0.511 | 0.676 |
| vocabulary | 6,462 | 0.483 | 0.57 | 0.990 | 0.483 | 0.652 |
| phraseology | 6,462 | 0.510 | 0.52 | 0.980 | 0.510 | 0.676 |
| grammar | 6,462 | 0.531 | 0.50 | 0.975 | 0.531 | 0.694 |
| conventions | 6,462 | 0.518 | 0.52 | 0.975 | 0.518 | 0.682 |

## Agreement on the test split

QWK is the quadratic weighted kappa on half-point levels. 'vs rater mean' compares the rounded score with the mean of the two raters. 'vs one rater' compares it with each rater in turn, which is the figure to set against the two raters agreeing with each other; the loss is the second minus the first, so a negative loss means that the scorer agrees with a rater more than two raters do. The scorers that learn were trained on the mean of two ratings, which holds less noise than one rating: that explains part of such a gain and does not make them better than a person. The last three columns are the thresholds of Williamson, Xi and Breyer (2012), the first one applied to the agreement with one rater.

| scorer | n | QWK vs rater mean | QWK vs one rater | loss vs two raters | exact | within 0.5 | bias | SMD | QWK one rater >= 0.70 | loss <= 0.10 | |SMD| <= 0.15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 2,571 | 0.00 [-0.00, 0.00] | 0.00 [-0.00, 0.00] | 0.58 [0.55, 0.60] | 0.37 | 0.57 | 0.00 | 0.01 | no | no | yes |
| surface-gbm | 2,571 | 0.55 [0.53, 0.57] | 0.47 [0.45, 0.49] | 0.11 [0.08, 0.14] | 0.38 | 0.69 | -0.05 | -0.09 | no | no | yes |
| tfidf-ridge | 2,571 | 0.60 [0.58, 0.62] | 0.51 [0.49, 0.53] | 0.07 [0.04, 0.09] | 0.42 | 0.72 | 0.00 | 0.00 | no | yes | yes |
| qwen0.5b ridge | 2,571 | 0.74 [0.72, 0.75] | 0.64 [0.62, 0.65] | -0.06 [-0.08, -0.03] | 0.49 | 0.80 | -0.01 | -0.01 | no | yes | yes |

## Agreement on the zero-shot subset

The first essays of a seeded shuffle of the test split: the zero-shot scorers are run on a few hundred essays only, and the other scorers are shown on the same essays. A zero-shot score is the expected value of the probabilities the model gives to the answers 1 to 5 (E to A for the letters prompt). 'linear map' recalibrates that score to the rater scale with a line fitted on other essays than the one scored.

| scorer | n | QWK vs rater mean | QWK vs one rater | loss vs two raters | exact | within 0.5 | bias | SMD | QWK one rater >= 0.70 | loss <= 0.10 | |SMD| <= 0.15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| mean | 400 | 0.00 [-0.00, 0.00] | -0.00 [-0.00, 0.00] | 0.58 [0.51, 0.64] | 0.37 | 0.58 | 0.02 | 0.05 | no | no | yes |
| surface-gbm | 400 | 0.58 [0.52, 0.64] | 0.50 [0.44, 0.55] | 0.08 [0.01, 0.15] | 0.40 | 0.71 | -0.05 | -0.09 | no | yes | yes |
| tfidf-ridge | 400 | 0.61 [0.55, 0.67] | 0.52 [0.46, 0.57] | 0.06 [-0.01, 0.13] | 0.44 | 0.73 | 0.03 | 0.05 | no | yes | yes |
| qwen0.5b ridge | 400 | 0.75 [0.71, 0.79] | 0.65 [0.60, 0.69] | -0.07 [-0.13, -0.02] | 0.51 | 0.81 | 0.00 | 0.00 | no | yes | yes |
| qwen0.5b zero-shot essay_first | 400 | 0.02 [0.01, 0.04] | 0.02 [0.01, 0.04] | 0.56 [0.49, 0.62] | 0.04 | 0.08 | 1.51 | 2.81 | no | no | no |
| qwen0.5b zero-shot essay_first, linear map | 400 | 0.00 [-0.00, 0.00] | -0.00 [-0.00, 0.00] | 0.58 [0.51, 0.64] | 0.37 | 0.57 | 0.00 | 0.00 | no | no | yes |
| qwen0.5b zero-shot letters | 400 | 0.10 [0.03, 0.16] | 0.08 [0.03, 0.14] | 0.50 [0.41, 0.58] | 0.28 | 0.55 | 0.23 | 0.49 | no | no | no |
| qwen0.5b zero-shot letters, linear map | 400 | 0.02 [-0.02, 0.05] | 0.01 [-0.01, 0.04] | 0.57 [0.50, 0.63] | 0.36 | 0.58 | 0.00 | 0.00 | no | no | yes |
| qwen0.5b zero-shot rubric | 400 | 0.00 [-0.00, 0.01] | 0.00 [-0.00, 0.01] | 0.58 [0.51, 0.64] | 0.00 | 0.03 | 1.75 | 3.76 | no | no | no |
| qwen0.5b zero-shot rubric, linear map | 400 | 0.00 [-0.00, 0.00] | -0.00 [-0.00, 0.00] | 0.58 [0.51, 0.64] | 0.37 | 0.57 | -0.00 | -0.00 | no | no | yes |
| qwen0.5b zero-shot short | 400 | 0.01 [-0.01, 0.04] | 0.01 [-0.01, 0.04] | 0.57 [0.50, 0.64] | 0.08 | 0.15 | 1.29 | 2.33 | no | no | no |
| qwen0.5b zero-shot short, linear map | 400 | 0.00 [-0.00, 0.00] | -0.00 [-0.00, 0.00] | 0.58 [0.51, 0.64] | 0.37 | 0.57 | 0.00 | 0.00 | no | no | yes |

## Prompt sensitivity (qwen0.5b)

The same model with four wordings of the task. A scorer that did not depend on the wording would give the same mean and order the essays alike. 'ICC of one sampled run' is the share of the variance of one sampled answer that comes from the essays and not from the sampling, computed from the probabilities.

| prompt | n | mean score | sd | Spearman vs raters | QWK vs raters | ICC of one sampled run |
|---|---|---|---|---|---|---|
| qwen0.5b zero-shot essay_first | 400 | 4.59 | 0.43 | 0.15 | 0.02 | 0.27 |
| qwen0.5b zero-shot letters | 400 | 3.32 | 0.25 | 0.15 | 0.10 | 0.04 |
| qwen0.5b zero-shot rubric | 400 | 4.83 | 0.21 | 0.07 | 0.00 | 0.21 |
| qwen0.5b zero-shot short | 400 | 4.37 | 0.47 | 0.04 | 0.01 | 0.19 |

## Robustness of the learned scorers

The same changes are applied to the first essays of the test split and the score is read again. typos: one typing slip in 2 or 5 percent of the words of four letters or more. lowercase: the text in lower case. shuffle: the sentences in a random order. whitespace: double spaces and blank lines. injection: an order to give the highest score, after the essay. repeat: the essay twice in a row. A change is as expected when the mean shift is at least 0.1 point down (lower), or within 0.1 point with a QWK of at least 0.90 between the scores before and after (same).

| scorer | change | expected | mean shift | moved by 0.5 or more | QWK before/after | as expected |
|---|---|---|---|---|---|---|
| surface-gbm | typos_2 | lower | -0.10 [-0.11, -0.09] | 0.00 | 0.86 | yes |
| surface-gbm | typos_5 | lower | -0.19 [-0.21, -0.17] | 0.05 | 0.76 | yes |
| surface-gbm | lowercase | lower | -0.29 [-0.30, -0.27] | 0.07 | 0.69 | yes |
| surface-gbm | shuffle | lower | -0.10 [-0.11, -0.08] | 0.00 | 0.86 | no |
| surface-gbm | whitespace | same | 0.00 [0.00, 0.00] | 0.00 | 1.00 | yes |
| surface-gbm | injection | same | 0.01 [-0.00, 0.03] | 0.01 | 0.89 | no |
| surface-gbm | repeat | same | -0.44 [-0.47, -0.41] | 0.36 | 0.48 | no |
| tfidf-ridge | typos_2 | lower | -0.01 [-0.01, -0.01] | 0.00 | 0.98 | no |
| tfidf-ridge | typos_5 | lower | -0.03 [-0.03, -0.02] | 0.00 | 0.96 | no |
| tfidf-ridge | lowercase | lower | 0.00 [0.00, 0.00] | 0.00 | 1.00 | no |
| tfidf-ridge | shuffle | lower | 0.00 [-0.00, 0.00] | 0.00 | 1.00 | no |
| tfidf-ridge | whitespace | same | 0.00 [0.00, 0.00] | 0.00 | 1.00 | yes |
| tfidf-ridge | injection | same | 0.02 [0.02, 0.03] | 0.00 | 0.97 | yes |
| tfidf-ridge | repeat | same | 0.03 [0.03, 0.04] | 0.00 | 0.96 | yes |
| qwen0.5b ridge | typos_2 | lower | -0.08 [-0.09, -0.07] | 0.00 | 0.90 | no |
| qwen0.5b ridge | typos_5 | lower | -0.18 [-0.19, -0.17] | 0.00 | 0.83 | yes |
| qwen0.5b ridge | lowercase | lower | -0.12 [-0.13, -0.11] | 0.00 | 0.87 | yes |
| qwen0.5b ridge | shuffle | lower | -0.12 [-0.13, -0.11] | 0.00 | 0.87 | yes |
| qwen0.5b ridge | whitespace | same | 0.06 [0.04, 0.08] | 0.01 | 0.80 | no |
| qwen0.5b ridge | injection | same | 0.01 [0.01, 0.01] | 0.00 | 0.99 | yes |
| qwen0.5b ridge | repeat | same | -0.01 [-0.01, 0.00] | 0.00 | 0.95 | yes |

## Robustness of the zero-shot scorers

The same changes are applied to the first essays of the test split and the score is read again. typos: one typing slip in 2 or 5 percent of the words of four letters or more. lowercase: the text in lower case. shuffle: the sentences in a random order. whitespace: double spaces and blank lines. injection: an order to give the highest score, after the essay. repeat: the essay twice in a row. A change is as expected when the mean shift is at least 0.1 point down (lower), or within 0.1 point with a QWK of at least 0.90 between the scores before and after (same).

| scorer | change | expected | mean shift | moved by 0.5 or more | QWK before/after | as expected |
|---|---|---|---|---|---|---|
| qwen0.5b zero-shot letters | shuffle | lower | -0.06 [-0.09, -0.04] | 0.02 | 0.43 | no |
| qwen0.5b zero-shot letters | repeat | same | 0.14 [0.11, 0.16] | 0.10 | 0.36 | no |
| qwen0.5b zero-shot letters | injection | same | -0.20 [-0.24, -0.16] | 0.21 | 0.34 | no |
| qwen0.5b zero-shot letters | typos_5 | lower | 0.02 [0.01, 0.03] | 0.00 | 0.75 | no |

## Agreement and bias by group

Race groups with fewer than 100 test essays are pooled as Other, and a group with fewer than 30 is left out of the tables. bias is the scorer's mean minus the raters' mean in the group: positive means that the scorer is more generous to the group than the raters. SMD is that gap over the spread of the two scores (Williamson, Xi and Breyer ask for an absolute value below 0.15).

| scorer | attribute | group | n | raters | scorer mean | bias | SMD | QWK |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b ridge | gender | Male | 1,445 | 3.07 | 3.06 | -0.01 [-0.03, 0.01] | -0.01 | 0.72 [0.70, 0.74] |
| qwen0.5b ridge | gender | Female | 1,125 | 3.14 | 3.14 | -0.00 [-0.03, 0.02] | -0.01 | 0.75 [0.73, 0.77] |
| qwen0.5b ridge | race | Hispanic | 1,867 | 3.07 | 3.06 | -0.01 [-0.03, 0.01] | -0.01 | 0.74 [0.72, 0.76] |
| qwen0.5b ridge | race | Asian/PI | 310 | 3.21 | 3.21 | -0.01 [-0.05, 0.04] | -0.02 | 0.73 [0.67, 0.77] |
| qwen0.5b ridge | race | Black | 194 | 3.18 | 3.19 | 0.00 [-0.05, 0.06] | 0.01 | 0.70 [0.63, 0.76] |
| qwen0.5b ridge | race | White | 185 | 3.16 | 3.16 | 0.00 [-0.05, 0.06] | 0.01 | 0.73 [0.66, 0.78] |
| qwen0.5b ridge | ses | disadvantaged | 1,798 | 3.10 | 3.09 | -0.00 [-0.02, 0.01] | -0.01 | 0.73 [0.71, 0.75] |
| qwen0.5b ridge | ses | not disadvantaged | 772 | 3.12 | 3.11 | -0.01 [-0.04, 0.02] | -0.02 | 0.75 [0.72, 0.78] |
| surface-gbm | gender | Male | 1,445 | 3.07 | 3.02 | -0.05 [-0.08, -0.03] | -0.10 | 0.57 [0.54, 0.60] |
| surface-gbm | gender | Female | 1,125 | 3.14 | 3.10 | -0.04 [-0.07, -0.01] | -0.07 | 0.53 [0.49, 0.56] |
| surface-gbm | race | Hispanic | 1,867 | 3.07 | 3.03 | -0.05 [-0.07, -0.02] | -0.08 | 0.57 [0.55, 0.60] |
| surface-gbm | race | Asian/PI | 310 | 3.21 | 3.17 | -0.05 [-0.11, 0.01] | -0.09 | 0.41 [0.33, 0.48] |
| surface-gbm | race | Black | 194 | 3.18 | 3.13 | -0.05 [-0.14, 0.02] | -0.11 | 0.50 [0.40, 0.58] |
| surface-gbm | race | White | 185 | 3.16 | 3.09 | -0.06 [-0.13, 0.01] | -0.12 | 0.55 [0.44, 0.63] |
| surface-gbm | ses | disadvantaged | 1,798 | 3.10 | 3.05 | -0.04 [-0.07, -0.02] | -0.08 | 0.54 [0.51, 0.56] |
| surface-gbm | ses | not disadvantaged | 772 | 3.12 | 3.06 | -0.06 [-0.10, -0.02] | -0.10 | 0.59 [0.54, 0.62] |
| tfidf-ridge | gender | Male | 1,445 | 3.07 | 3.08 | 0.01 [-0.01, 0.03] | 0.01 | 0.59 [0.56, 0.62] |
| tfidf-ridge | gender | Female | 1,125 | 3.14 | 3.14 | -0.01 [-0.04, 0.02] | -0.01 | 0.61 [0.58, 0.65] |
| tfidf-ridge | race | Hispanic | 1,867 | 3.07 | 3.07 | 0.00 [-0.02, 0.03] | 0.01 | 0.61 [0.59, 0.64] |
| tfidf-ridge | race | Asian/PI | 310 | 3.21 | 3.21 | -0.01 [-0.07, 0.04] | -0.02 | 0.57 [0.47, 0.64] |
| tfidf-ridge | race | Black | 194 | 3.18 | 3.19 | 0.01 [-0.06, 0.09] | 0.02 | 0.51 [0.41, 0.60] |
| tfidf-ridge | race | White | 185 | 3.16 | 3.13 | -0.03 [-0.09, 0.04] | -0.05 | 0.58 [0.49, 0.65] |
| tfidf-ridge | ses | disadvantaged | 1,798 | 3.10 | 3.10 | 0.00 [-0.01, 0.03] | 0.01 | 0.59 [0.57, 0.62] |
| tfidf-ridge | ses | not disadvantaged | 772 | 3.12 | 3.11 | -0.01 [-0.05, 0.02] | -0.02 | 0.62 [0.58, 0.65] |

## Impact ratios

Share of each group scoring at or above a cut, divided by the share of the best-placed group; the scorer's score is rounded to the half point first, as the mean of two raters is. A ratio below 0.8 breaks the four-fifths rule. 'ratio raters' is the same ratio computed on the raters' scores, which says whether the scorer made the gap or found it.

| scorer | attribute | group | cut | rate raters | rate scorer | ratio raters | ratio scorer | below 0.8 |
|---|---|---|---|---|---|---|---|---|
| qwen0.5b ridge | gender | Male | 3.0 | 0.72 | 0.74 | 0.95 | 0.94 | no |
| qwen0.5b ridge | gender | Female | 3.0 | 0.76 | 0.79 | 1.00 | 1.00 | no |
| qwen0.5b ridge | gender | Male | 3.5 | 0.35 | 0.37 | 0.87 | 0.81 | no |
| qwen0.5b ridge | gender | Female | 3.5 | 0.40 | 0.45 | 1.00 | 1.00 | no |
| qwen0.5b ridge | race | Hispanic | 3.0 | 0.72 | 0.75 | 0.88 | 0.91 | no |
| qwen0.5b ridge | race | Asian/PI | 3.0 | 0.78 | 0.83 | 0.96 | 1.00 | no |
| qwen0.5b ridge | race | Black | 3.0 | 0.81 | 0.79 | 1.00 | 0.96 | no |
| qwen0.5b ridge | race | White | 3.0 | 0.75 | 0.77 | 0.92 | 0.93 | no |
| qwen0.5b ridge | race | Hispanic | 3.5 | 0.34 | 0.37 | 0.74 | 0.71 | yes |
| qwen0.5b ridge | race | Asian/PI | 3.5 | 0.46 | 0.46 | 1.00 | 0.89 | no |
| qwen0.5b ridge | race | Black | 3.5 | 0.41 | 0.48 | 0.90 | 0.91 | no |
| qwen0.5b ridge | race | White | 3.5 | 0.43 | 0.52 | 0.94 | 1.00 | no |
| qwen0.5b ridge | ses | disadvantaged | 3.0 | 0.74 | 0.77 | 1.00 | 1.00 | no |
| qwen0.5b ridge | ses | not disadvantaged | 3.0 | 0.73 | 0.75 | 0.99 | 0.99 | no |
| qwen0.5b ridge | ses | disadvantaged | 3.5 | 0.36 | 0.39 | 0.95 | 0.94 | no |
| qwen0.5b ridge | ses | not disadvantaged | 3.5 | 0.38 | 0.42 | 1.00 | 1.00 | no |
| surface-gbm | gender | Male | 3.0 | 0.72 | 0.73 | 0.95 | 0.93 | no |
| surface-gbm | gender | Female | 3.0 | 0.76 | 0.79 | 1.00 | 1.00 | no |
| surface-gbm | gender | Male | 3.5 | 0.35 | 0.32 | 0.87 | 0.81 | no |
| surface-gbm | gender | Female | 3.5 | 0.40 | 0.39 | 1.00 | 1.00 | no |
| surface-gbm | race | Hispanic | 3.0 | 0.72 | 0.74 | 0.88 | 0.90 | no |
| surface-gbm | race | Asian/PI | 3.0 | 0.78 | 0.82 | 0.96 | 1.00 | no |
| surface-gbm | race | Black | 3.0 | 0.81 | 0.79 | 1.00 | 0.96 | no |
| surface-gbm | race | White | 3.0 | 0.75 | 0.79 | 0.92 | 0.96 | no |
| surface-gbm | race | Hispanic | 3.5 | 0.34 | 0.31 | 0.74 | 0.69 | yes |
| surface-gbm | race | Asian/PI | 3.5 | 0.46 | 0.46 | 1.00 | 1.00 | no |
| surface-gbm | race | Black | 3.5 | 0.41 | 0.45 | 0.90 | 0.98 | no |
| surface-gbm | race | White | 3.5 | 0.43 | 0.41 | 0.94 | 0.90 | no |
| surface-gbm | ses | disadvantaged | 3.0 | 0.74 | 0.76 | 1.00 | 1.00 | no |
| surface-gbm | ses | not disadvantaged | 3.0 | 0.73 | 0.75 | 0.99 | 0.99 | no |
| surface-gbm | ses | disadvantaged | 3.5 | 0.36 | 0.34 | 0.95 | 0.93 | no |
| surface-gbm | ses | not disadvantaged | 3.5 | 0.38 | 0.37 | 1.00 | 1.00 | no |
| tfidf-ridge | gender | Male | 3.0 | 0.72 | 0.79 | 0.95 | 0.98 | no |
| tfidf-ridge | gender | Female | 3.0 | 0.76 | 0.81 | 1.00 | 1.00 | no |
| tfidf-ridge | gender | Male | 3.5 | 0.35 | 0.36 | 0.87 | 0.86 | no |
| tfidf-ridge | gender | Female | 3.5 | 0.40 | 0.42 | 1.00 | 1.00 | no |
| tfidf-ridge | race | Hispanic | 3.0 | 0.72 | 0.78 | 0.88 | 0.91 | no |
| tfidf-ridge | race | Asian/PI | 3.0 | 0.78 | 0.86 | 0.96 | 1.00 | no |
| tfidf-ridge | race | Black | 3.0 | 0.81 | 0.86 | 1.00 | 1.00 | no |
| tfidf-ridge | race | White | 3.0 | 0.75 | 0.83 | 0.92 | 0.97 | no |
| tfidf-ridge | race | Hispanic | 3.5 | 0.34 | 0.35 | 0.74 | 0.74 | yes |
| tfidf-ridge | race | Asian/PI | 3.5 | 0.46 | 0.47 | 1.00 | 1.00 | no |
| tfidf-ridge | race | Black | 3.5 | 0.41 | 0.47 | 0.90 | 0.99 | no |
| tfidf-ridge | race | White | 3.5 | 0.43 | 0.44 | 0.94 | 0.93 | no |
| tfidf-ridge | ses | disadvantaged | 3.0 | 0.74 | 0.80 | 1.00 | 1.00 | no |
| tfidf-ridge | ses | not disadvantaged | 3.0 | 0.73 | 0.80 | 0.99 | 1.00 | no |
| tfidf-ridge | ses | disadvantaged | 3.5 | 0.36 | 0.38 | 0.95 | 0.98 | no |
| tfidf-ridge | ses | not disadvantaged | 3.5 | 0.38 | 0.39 | 1.00 | 1.00 | no |

## Error of the scorer regressed on the rater score and writer groups

The scorer's score minus the raters' score, regressed on the raters' score and on the writer groups, with robust standard errors. A group coefficient is the difference in the average error at the same rater score, against the largest group. Only the group terms are shown.

| scorer | term | coefficient | p |
|---|---|---|---|
| qwen0.5b ridge | gender: Female vs Male | 0.032 [0.007, 0.056] | 0.0107 |
| qwen0.5b ridge | race: Asian/PI vs Hispanic | 0.057 [0.021, 0.093] | 0.0021 |
| qwen0.5b ridge | race: Black vs Hispanic | 0.055 [0.007, 0.103] | 0.0257 |
| qwen0.5b ridge | race: Other vs Hispanic | -0.103 [-0.299, 0.093] | 0.3037 |
| qwen0.5b ridge | race: White vs Hispanic | 0.045 [-0.003, 0.094] | 0.0670 |
| qwen0.5b ridge | ses: not disadvantaged vs disadvantaged | -0.005 [-0.031, 0.022] | 0.7365 |
| surface-gbm | gender: Female vs Male | 0.055 [0.028, 0.082] | 0.0001 |
| surface-gbm | race: Asian/PI vs Hispanic | 0.084 [0.038, 0.129] | 0.0003 |
| surface-gbm | race: Black vs Hispanic | 0.056 [-0.001, 0.112] | 0.0549 |
| surface-gbm | race: Other vs Hispanic | 0.034 [-0.133, 0.200] | 0.6926 |
| surface-gbm | race: White vs Hispanic | 0.032 [-0.022, 0.087] | 0.2465 |
| surface-gbm | ses: not disadvantaged vs disadvantaged | -0.008 [-0.037, 0.022] | 0.6066 |
| tfidf-ridge | gender: Female vs Male | 0.027 [0.002, 0.052] | 0.0327 |
| tfidf-ridge | race: Asian/PI vs Hispanic | 0.069 [0.030, 0.107] | 0.0005 |
| tfidf-ridge | race: Black vs Hispanic | 0.070 [0.022, 0.118] | 0.0043 |
| tfidf-ridge | race: Other vs Hispanic | -0.006 [-0.210, 0.199] | 0.9564 |
| tfidf-ridge | race: White vs Hispanic | 0.018 [-0.033, 0.068] | 0.4932 |
| tfidf-ridge | ses: not disadvantaged vs disadvantaged | -0.010 [-0.038, 0.017] | 0.4642 |

## Rater score regressed on the scorer's score and writer groups

The raters' score regressed on the scorer's score and on the writer groups (Cleary, 1968). A group term away from zero would mean that the same scorer score predicts a different rater score depending on the group.

| scorer | term | coefficient | p |
|---|---|---|---|
| qwen0.5b ridge | gender: Female vs Male | -0.001 [-0.032, 0.030] | 0.9349 |
| qwen0.5b ridge | race: Asian/PI vs Hispanic | 0.003 [-0.045, 0.051] | 0.9027 |
| qwen0.5b ridge | race: Black vs Hispanic | -0.008 [-0.068, 0.051] | 0.7857 |
| qwen0.5b ridge | race: Other vs Hispanic | 0.135 [-0.111, 0.380] | 0.2820 |
| qwen0.5b ridge | race: White vs Hispanic | -0.010 [-0.071, 0.051] | 0.7586 |
| qwen0.5b ridge | ses: not disadvantaged vs disadvantaged | 0.007 [-0.027, 0.041] | 0.6909 |
| surface-gbm | gender: Female vs Male | -0.001 [-0.040, 0.038] | 0.9614 |
| surface-gbm | race: Asian/PI vs Hispanic | 0.019 [-0.047, 0.084] | 0.5769 |
| surface-gbm | race: Black vs Hispanic | 0.023 [-0.052, 0.098] | 0.5513 |
| surface-gbm | race: Other vs Hispanic | 0.024 [-0.277, 0.325] | 0.8764 |
| surface-gbm | race: White vs Hispanic | 0.026 [-0.051, 0.103] | 0.5045 |
| surface-gbm | ses: not disadvantaged vs disadvantaged | 0.011 [-0.032, 0.053] | 0.6259 |
| tfidf-ridge | gender: Female vs Male | 0.015 [-0.021, 0.051] | 0.4186 |
| tfidf-ridge | race: Asian/PI vs Hispanic | 0.013 [-0.044, 0.071] | 0.6443 |
| tfidf-ridge | race: Black vs Hispanic | -0.006 [-0.077, 0.066] | 0.8774 |
| tfidf-ridge | race: Other vs Hispanic | 0.053 [-0.262, 0.367] | 0.7425 |
| tfidf-ridge | race: White vs Hispanic | 0.031 [-0.042, 0.104] | 0.4049 |
| tfidf-ridge | ses: not disadvantaged vs disadvantaged | 0.013 [-0.027, 0.054] | 0.5154 |

## Differential functioning of the human trait ratings

Differential item functioning by standardization (Dorans and Kulick): essays are matched on the mean of the other five traits and cut into eight bands, and the gap in the trait between the two groups is averaged over the bands. This audits the human ratings that the scorers learn from, not a scorer.

| group | trait | n focal | DIF in points | DIF in SD |
|---|---|---|---|---|
| Female vs Male | cohesion | 2,846 | 0.045 [0.019, 0.065] | 0.069 |
| Female vs Male | syntax | 2,846 | 0.017 [-0.004, 0.034] | 0.027 |
| Female vs Male | vocabulary | 2,846 | 0.028 [0.010, 0.046] | 0.049 |
| Female vs Male | phraseology | 2,846 | -0.020 [-0.037, -0.001] | -0.031 |
| Female vs Male | grammar | 2,846 | -0.059 [-0.081, -0.035] | -0.085 |
| Female vs Male | conventions | 2,846 | 0.015 [-0.005, 0.036] | 0.022 |
| disadvantaged vs not disadvantaged | cohesion | 4,507 | -0.018 [-0.042, 0.007] | -0.027 |
| disadvantaged vs not disadvantaged | syntax | 4,507 | -0.009 [-0.028, 0.012] | -0.014 |
| disadvantaged vs not disadvantaged | vocabulary | 4,507 | -0.025 [-0.046, -0.004] | -0.044 |
| disadvantaged vs not disadvantaged | phraseology | 4,507 | -0.007 [-0.031, 0.011] | -0.010 |
| disadvantaged vs not disadvantaged | grammar | 4,507 | 0.011 [-0.014, 0.033] | 0.016 |
| disadvantaged vs not disadvantaged | conventions | 4,507 | -0.012 [-0.039, 0.011] | -0.018 |
| Asian/PI vs Hispanic | cohesion | 792 | 0.019 [-0.018, 0.046] | 0.029 |
| Asian/PI vs Hispanic | syntax | 792 | 0.082 [0.052, 0.114] | 0.129 |
| Asian/PI vs Hispanic | vocabulary | 792 | 0.009 [-0.015, 0.037] | 0.016 |
| Asian/PI vs Hispanic | phraseology | 792 | -0.003 [-0.033, 0.029] | -0.004 |
| Asian/PI vs Hispanic | grammar | 792 | -0.047 [-0.082, -0.011] | -0.069 |
| Asian/PI vs Hispanic | conventions | 792 | 0.085 [0.048, 0.121] | 0.128 |
| Black vs Hispanic | cohesion | 515 | -0.023 [-0.064, 0.018] | -0.036 |
| Black vs Hispanic | syntax | 515 | 0.008 [-0.029, 0.042] | 0.012 |
| Black vs Hispanic | vocabulary | 515 | 0.064 [0.028, 0.098] | 0.111 |
| Black vs Hispanic | phraseology | 515 | 0.046 [0.018, 0.081] | 0.071 |
| Black vs Hispanic | grammar | 515 | 0.070 [0.030, 0.111] | 0.103 |
| Black vs Hispanic | conventions | 515 | -0.032 [-0.075, 0.007] | -0.048 |
| White vs Hispanic | cohesion | 471 | 0.015 [-0.042, 0.055] | 0.022 |
| White vs Hispanic | syntax | 471 | 0.002 [-0.039, 0.041] | 0.003 |
| White vs Hispanic | vocabulary | 471 | -0.007 [-0.052, 0.029] | -0.013 |
| White vs Hispanic | phraseology | 471 | 0.024 [-0.013, 0.062] | 0.037 |
| White vs Hispanic | grammar | 471 | 0.048 [0.007, 0.096] | 0.070 |
| White vs Hispanic | conventions | 471 | -0.012 [-0.053, 0.027] | -0.018 |

## Graded response model of the six traits

| trait | discrimination | b1 | b2 | b3 | b4 | b5 |
|---|---|---|---|---|---|---|
| cohesion | 2.51 | -1.62 | -0.70 | 0.23 | 1.11 | 2.24 |
| syntax | 3.11 | -1.38 | -0.53 | 0.43 | 1.25 | 2.29 |
| vocabulary | 2.89 | -2.04 | -1.11 | 0.16 | 1.02 | 2.13 |
| phraseology | 3.34 | -1.47 | -0.65 | 0.23 | 1.03 | 2.17 |
| grammar | 2.64 | -1.29 | -0.44 | 0.38 | 1.21 | 2.14 |
| conventions | 2.49 | -1.46 | -0.60 | 0.31 | 1.19 | 2.28 |

Reliability of the proficiency estimates 0.924, Cronbach alpha 0.926, on 6,482 essays.

## Bias across the range of proficiency

Bands of equal size of the proficiency estimated by the graded response model. A scorer that pulls every essay toward the middle is too generous at the low end and too harsh at the high end, which the overall bias hides.

| scorer | proficiency band | n | raters | scorer mean | bias | MAE |
|---|---|---|---|---|---|---|
| qwen0.5b ridge | 1 | 515 | 2.31 | 2.51 | 0.20 [0.17, 0.23] | 0.34 |
| qwen0.5b ridge | 2 | 513 | 2.81 | 2.87 | 0.07 [0.04, 0.10] | 0.29 |
| qwen0.5b ridge | 3 | 514 | 3.06 | 3.14 | 0.08 [0.05, 0.10] | 0.26 |
| qwen0.5b ridge | 4 | 514 | 3.39 | 3.33 | -0.06 [-0.09, -0.03] | 0.28 |
| qwen0.5b ridge | 5 | 514 | 3.94 | 3.62 | -0.32 [-0.35, -0.29] | 0.39 |
| surface-gbm | 1 | 515 | 2.31 | 2.63 | 0.32 [0.28, 0.36] | 0.44 |
| surface-gbm | 2 | 513 | 2.81 | 2.91 | 0.10 [0.06, 0.13] | 0.33 |
| surface-gbm | 3 | 514 | 3.06 | 3.09 | 0.02 [-0.01, 0.05] | 0.28 |
| surface-gbm | 4 | 514 | 3.39 | 3.24 | -0.15 [-0.18, -0.11] | 0.36 |
| surface-gbm | 5 | 514 | 3.94 | 3.41 | -0.53 [-0.56, -0.49] | 0.57 |
| tfidf-ridge | 1 | 515 | 2.31 | 2.66 | 0.35 [0.32, 0.38] | 0.43 |
| tfidf-ridge | 2 | 513 | 2.81 | 2.93 | 0.12 [0.09, 0.15] | 0.32 |
| tfidf-ridge | 3 | 514 | 3.06 | 3.15 | 0.08 [0.05, 0.11] | 0.27 |
| tfidf-ridge | 4 | 514 | 3.39 | 3.28 | -0.11 [-0.14, -0.08] | 0.31 |
| tfidf-ridge | 5 | 514 | 3.94 | 3.50 | -0.44 [-0.48, -0.41] | 0.50 |

## What narrowing the gaps costs

as fitted: the scorer as it is. reweighed training: training weights that make the group and the score level independent (Kamiran and Calders, 2012). common line: one line from the scorer's score to the rater scale, fitted on other folds. line for each group: one line per group, which needs the writer's group at scoring time. The gap in bias is the largest minus the smallest bias between groups.

| attribute | version | QWK | gap in bias | largest |SMD| | lowest group QWK |
|---|---|---|---|---|---|
| gender | as fitted | 0.74 [0.72, 0.75] | 0.00 | 0.01 | 0.72 |
| gender | reweighed training | 0.73 [0.72, 0.75] | 0.00 | 0.01 | 0.72 |
| gender | common line | 0.73 [0.71, 0.75] | 0.00 | 0.00 | 0.71 |
| gender | line for each group | 0.73 [0.71, 0.75] | 0.00 | 0.00 | 0.71 |
| race | as fitted | 0.74 [0.72, 0.75] | 0.01 | 0.02 | 0.70 |
| race | reweighed training | 0.73 [0.71, 0.75] | 0.02 | 0.04 | 0.70 |
| race | common line | 0.73 [0.71, 0.75] | 0.01 | 0.02 | 0.69 |
| race | line for each group | 0.73 [0.71, 0.75] | 0.00 | 0.00 | 0.67 |
| ses | as fitted | 0.74 [0.72, 0.75] | 0.01 | 0.02 | 0.73 |
| ses | reweighed training | 0.73 [0.72, 0.75] | 0.01 | 0.02 | 0.73 |
| ses | common line | 0.73 [0.71, 0.75] | 0.01 | 0.01 | 0.72 |
| ses | line for each group | 0.73 [0.71, 0.75] | 0.00 | 0.00 | 0.72 |

## Sending the doubtful essays to a person

The essays on which an ensemble of ridge models, each fitted on a bootstrap sample, disagree most are sent to a person, here the first rater, and the second rater is the reference. 'At random' sends the same number of essays chosen by chance. A scorer trained on the mean of two raters agrees more with the second rater than the first rater does, so sending essays to the first rater lowers the QWK here: what the table can show is whether the doubt signal picks out the essays the scorer gets wrong, which is the gap between the two columns.

| scorer | share sent to a rater | essays | QWK, most doubtful first | QWK, at random |
|---|---|---|---|---|
| qwen0.5b ridge | 0.00 | 0 | 0.634 | 0.634 |
| qwen0.5b ridge | 0.10 | 257 | 0.630 | 0.626 |
| qwen0.5b ridge | 0.20 | 514 | 0.622 | 0.621 |
| qwen0.5b ridge | 0.30 | 771 | 0.616 | 0.614 |
| qwen0.5b ridge | 0.50 | 1,286 | 0.610 | 0.602 |
| qwen0.5b ridge | 0.75 | 1,928 | 0.598 | 0.590 |
| qwen0.5b ridge | 1.00 | 2,571 | 0.580 | 0.580 |

