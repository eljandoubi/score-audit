# Model card: a ridge scorer on Qwen2.5-0.5B hidden states

This card describes the scorer that agrees best with the raters in this repository. It is a
research artifact made to show how a scorer is audited. It is not a product and it must not be
used to score real people.

## Model

- Input: an English essay. Output: an overall score from 1.0 to 5.0, rounded to half points
  where a kappa is computed.
- Method: the essay goes through Qwen2.5-0.5B-Instruct (cut at 640 tokens). The hidden states
  of layers 6, 12, 18 and 24 are averaged over the tokens and concatenated, standardized, and
  given to a ridge regression whose penalty is chosen by cross-validation. The set of columns (one
  layer or the four together) was chosen on a held-out fifth of the training essays: the four
  together came first (QWK 0.74, one layer 0.71 to 0.73).
- Training data: the 3,911 training essays of ELLIPSE, with the mean of two raters as target.
- The language model is not fine-tuned. Only the ridge is fitted.
- Licences: code Apache-2.0, corpus CC BY 4.0, language model weights under their own licence.

## Intended use and out of scope

Intended: learning how an automated scorer is checked (agreement with raters, sensitivity to the
input, differences between groups of writers).

Out of scope: any decision about a person (grading, admission, hiring, promotion), other
languages, other ages or genres than school essays by English language learners in the United
States, spoken answers.

## Evaluation data

The 2,571 essays of the ELLIPSE test split. The two raters of each essay come from a second
archive of single ratings (27 raters). The ceiling between raters uses the 6,462 essays that
have two usable ratings. Writer attributes as recorded in the corpus: gender (two values), race
or ethnicity (six values; the groups under 100 test essays are pooled as Other and kept only
in the regressions), economic status (two values). The 44 prompts of the train split are also
those of the test split.

## Results

| | QWK, rater mean | QWK, one rater | loss vs two raters | kappa 0.70 | loss 0.10 | SMD 0.15 |
|---|---|---|---|---|---|---|
| qwen0.5b ridge | 0.74 [0.72, 0.75] | 0.64 [0.62, 0.65] | -0.06 [-0.08, -0.03] | no | yes | yes |
| tfidf ridge | 0.60 [0.58, 0.62] | 0.51 [0.49, 0.53] | 0.07 [0.04, 0.09] | no | yes | yes |
| surface-feature boosting | 0.55 [0.53, 0.57] | 0.47 [0.45, 0.49] | 0.11 [0.08, 0.14] | no | no | yes |

Two raters agree at 0.58 on the same essays. Intervals are bootstrap intervals over essays
(1,000 resamples). The last three columns are the criteria of Williamson, Xi and Breyer (2012)
with the kappa taken against one rater. The scorer is also at exact agreement 0.49 with the
rater mean, mean absolute error 0.31 point, and Bland-Altman limits of -0.78 and +0.77 points.

By group (scorer minus raters, in points, and QWK with the rater mean):

| group | essays | mean error | QWK |
|---|---|---|---|
| male | 1,445 | -0.008 [-0.027, 0.012] | 0.72 |
| female | 1,125 | -0.004 [-0.029, 0.019] | 0.75 |
| Hispanic | 1,867 | -0.007 [-0.027, 0.012] | 0.74 |
| Asian or Pacific Islander | 310 | -0.008 [-0.050, 0.035] | 0.73 |
| Black | 194 | 0.004 [-0.053, 0.059] | 0.70 |
| White | 185 | 0.004 [-0.051, 0.059] | 0.73 |
| economically disadvantaged | 1,798 | -0.004 [-0.020, 0.013] | 0.73 |
| not disadvantaged | 772 | -0.011 [-0.040, 0.016] | 0.75 |

Across the range of proficiency the error is much larger than across groups: +0.20 point for
the weakest fifth of writers (estimated by a graded response model of the six traits) and
-0.32 for the strongest. At the same rater score, the scorer gives 0.06 point more to Asian
and to Black writers than to Hispanic ones and 0.03 more to female than to male writers.

Under input changes (300 essays): 5 percent typos -0.18 point, lower case -0.12, shuffled
sentences -0.12, an order to give the highest score added after the essay +0.01, the essay
twice +0.00, and extra spaces and blank lines +0.06 with the agreement with the raters falling
from 0.76 to 0.63.

## Limits and risks

- It was fitted and tested on one corpus. Nothing is known about other texts, other prompts
  (the test prompts are the training prompts) or other populations.
- Its scores are pulled toward the middle of the scale: weak writers get too much, strong
  writers too little (+0.20 and -0.32 point at the two ends). At a cut of 3.5 the pass rate of
  Hispanic writers is 0.71 of the best-placed group's for the scorer and 0.74 for the raters,
  under the 0.8 of the four-fifths rule in both cases, and 0.81 for male writers (raters 0.87):
  a selection at a high cut would sit close to that line.
- It reacts to extra whitespace, which no reader counts. Normalizing whitespace before scoring
  is the first thing to do and was left out so that the audit shows the problem.
- Group results come from groups of 185 to 1,867 essays, with wide intervals for the smaller
  ones, and from the group labels of the corpus. They say nothing about intersections.
- The reference is two ratings that agree at 0.58, and the human ratings differ between groups
  at the same level on the other traits in 12 of 30 comparisons (up to 0.13 standard
  deviations), so a scorer trained on them can inherit that.
- Adding a calibration line for each group narrows the race gap in error (0.012 to 0.003
  point) at a cost of 0.007 QWK, lowers the QWK of the worst group (0.70 to 0.67) and needs the
  writer's group when scoring.

## Human oversight and the EU AI Act

The doubt signal that was tried (disagreement between ridge models fitted on bootstrap samples)
does not pick out the essays on which the scorer is wrong: at 10 percent routed the QWK is 0.630
against 0.626 for a random choice. A person who checks scores would do better to look at the
strongest and the weakest essays, where the scorer is most wrong.

A system that evaluates the learning outcomes of students (Annex III, point 3(b) of the EU AI
Act) or the candidates of a recruitment (point 4(a)) is a high-risk system. Its provider must
then meet the requirements of Articles 9 to 15: a risk management system, data governance,
technical documentation, record-keeping, transparency toward deployers, human oversight, and
accuracy, robustness and cybersecurity. This repository supplies a part of the evidence for
data description, accuracy, subgroup metrics, robustness and known limits. It is not a
conformity assessment, it has no risk management system, no logging and no oversight design,
and it is not legal advice.
