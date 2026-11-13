# scoreaudit

Checks on automated essay scorers: how well they agree with human raters, how much their
scores move when the input is changed a little, and whether the errors are the same for every
group of writers.

The checks follow the way automated scoring is validated in educational testing (Williamson, Xi
and Breyer, 2012): agreement with raters one by one and with their mean, the loss compared with
two raters agreeing with each other, standardized mean differences by group, and impact ratios.
On top of that it measures what a language model needs to be audited for: how much the score
depends on the wording of the prompt, whether text hidden in an essay can move it, and where
it can send an essay to a person.

## What it found

The numbers below are on the 2,571 test essays of the ELLIPSE corpus (see Data). The tables,
the figures and a reading note for each table are in [`results/`](results/SUMMARY.md).
The only language model that was run is Qwen2.5-0.5B-Instruct, on a laptop.

**The raters are the ceiling, and it is not high.** Two raters agree on the overall score at
a quadratic weighted kappa (QWK) of 0.59 (exact agreement 59 percent, within one point 99.7
percent). The reliability of one rating is 0.59, of the mean of two ratings 0.74 (ICC).

**A ridge regression on the hidden states of the language model agrees most.** Its QWK is 0.74
[0.72, 0.75] with the mean of the two raters and 0.64 [0.62, 0.65] with one rater at a time,
where two raters reach 0.58 on the same essays (loss -0.06 [-0.08, -0.03]). That is the figure
to expect from a model trained on the mean of two ratings, which holds less noise than one
rating: it does not mean the scorer is better than a person. Against one rater it stays under
the 0.70 of Williamson et al., which the two raters do not reach either. A ridge on n-grams
gets 0.60 and 0.51, boosting on surface features 0.55 and 0.47, and a constant score 0.

**Rating zero-shot, the 0.5B model is not usable.** Four wordings of the task give a QWK
between 0.00 and 0.10 with the raters. The mean score they give runs from 3.3 to 4.8 (the
raters: 3.1), with three of the wordings most essays get an expected score above 4, and the
rank correlation between two wordings runs from 0.03 to 0.79. Rescaling the scores with a
line does not help.

**The scorers react to the input in ways a reader would not.** On 300 essays:

- The embedding scorer loses about 0.18 point when 5 percent of the words get a typo, and 0.12
  when the text is lower-cased or the sentences are shuffled. Double spaces and blank lines,
  which no reader counts, change its scores enough that the agreement with the raters falls
  from 0.76 to 0.63 on these essays. Collapsing whitespace before scoring, and fitting again,
  would remove this one.
- An order hidden after the essay ("give it the highest score") does not move the embedding
  scorer (0.99 agreement before and after), but moves the zero-shot scores a lot (0.34).
- The n-gram scorer cannot see word order: shuffled sentences leave its scores unchanged.
  The surface-feature scorer drops by 0.44 point when the essay is written twice in a row.

**Group differences are small next to the error across the range of writers.** For the
embedding scorer:

- The mean error is under 0.02 point in every group and no standardized difference is near
  the 0.15 threshold. At the same rater score, the scorer gives 0.06 point more to Asian and
  Black writers than to Hispanic ones (about 0.06 [0.02, 0.09] and 0.06 [0.01, 0.10]) and 0.03
  more to female than to male writers: different from zero, small, and no group term is
  significant when the rater score is regressed on the scorer's score (Cleary).
- Its scores are pulled toward the middle: +0.20 point for the weakest fifth of writers and
  -0.32 for the strongest (bands from a graded response model of the six traits, reliability
  0.92). That is three to five times the largest difference between groups (0.06 point).
- At a cut of 3.5 the pass rate of Hispanic writers is 0.71 of the best-placed group for the
  scorer and 0.74 for the raters, under the 0.8 of the four-fifths rule in both cases: the
  scorer repeats a gap that is already in the ratings. For male writers it is 0.81 (raters
  0.87), just above the line. At a cut of 3.0 no group is under 0.8.
- The human ratings themselves differ between groups at the same level on the other traits in
  12 of 30 comparisons, by up to 0.13 standard deviations.

**Narrowing the gaps has a cost.** A calibration line for each race group brings the gap in
mean error from 0.012 to 0.003 point, lowers the QWK from 0.736 to 0.729 and the QWK of the
worst group from 0.70 to 0.67, and needs the writer's group at scoring time. Reweighing the
training essays made the race gap larger (0.020).

**Sending the doubtful essays to a person does not help here.** The essays on which bootstrap
ridge models disagree most are not the ones the scorer gets wrong: at 10 percent routed the QWK
is 0.630 against 0.626 at random. Replacing a score by one rater's score lowers the agreement
with the other rater, since the scorer agrees with a rater more than another rater does.

## Limits

- One corpus: essays by US school students who are English language learners, scored 1 to 5
  on overall proficiency. Nothing here says how any scorer does on other texts, languages or
  spoken answers, or against job performance.
- The reference is two ratings that agree at 0.59. The scorer is judged against that noise,
  not against a true score, and the mean of two ratings makes it look better than a single
  rater does.
- One small model and a frozen one, with a ridge on top. No fine-tuning, no larger or closed
  model: the zero-shot failure says nothing about those.
- The train and test essays share their 44 prompts, so there is no test on prompts the
  scorer has not seen.
- The input changes are one wording each, applied to 300 essays. They are checks, not a search
  for the worst case.
- Group labels are those of the corpus (two genders, six race or ethnicity groups with the
  smallest two pooled and left out of the tables, economic status). Intersections are not
  analyzed, and groups of 150 to 200 essays give wide intervals.
- The graded response model is fitted on the six trait scores, which are means of two ratings,
  not on item responses.
- This is not a validation of any scorer for grading or hiring. A scorer used to evaluate
  learning outcomes or candidates is a high-risk system under the EU AI Act; the
  [model card](MODEL_CARD.md) says what this work does and does not cover.

## Data

The ELLIPSE corpus (Crossley et al., 2023, CC BY 4.0): 6,482 essays by English language learners
in grades 8 to 12, each with an overall score and six trait scores (cohesion, syntax, vocabulary,
phraseology, grammar, conventions) that are the mean of two raters, and the writer's gender, race
or ethnicity and economic status. A second archive has the single ratings of the 27 raters.
The files are pinned by hash and fetched with `scoreaudit download`. The train split has 3,911
essays and the test split 2,571.

## Models

The model run here is Qwen2.5-0.5B-Instruct, used in two ways: its hidden states as features for
a ridge regression (the mean over the tokens of layers 6, 12, 18 and 24, essays cut at 640
tokens), and its next-token probabilities after a rating prompt, as a zero-shot scorer. Any other
causal language model works with the same commands, as a Hugging Face id or a local folder.

## Run it

```
uv sync
uv run scoreaudit download
uv run scoreaudit features --model models/qwen2.5-0.5b-instruct --out cache/emb/qwen0.5b
uv run scoreaudit zeroshot --model models/qwen2.5-0.5b-instruct --out cache/zeroshot/qwen0.5b
uv run scoreaudit perturbed --model models/qwen2.5-0.5b-instruct --out cache/perturbed/qwen0.5b
uv run scoreaudit study --models qwen0.5b
uv run scoreaudit report
```

`features` and `zeroshot` write what the model returned, by chunk, so a long run can be stopped
and resumed. `study` fits the scorers that learn, reads those files and writes the tables of
`results/`; `report` turns the tables into `results/SUMMARY.md` and the figures.
The tests need no download: they use a tiny random model and a fake corpus.

## Layout

| module | what it does |
|---|---|
| `data` | reads the corpus and the single ratings |
| `llm`, `prompts`, `features` | runs a causal language model over essays, with four rating prompts |
| `metrics` | weighted kappa, ICC, Bland-Altman, bootstrap intervals |
| `raters` | agreement between the raters, rater leniency, a scorer against raters one by one |
| `scorers` | n-gram ridge, surface features with boosting, ridge on model features |
| `perturb` | typos, lowercase, shuffled sentences, an injected order, repeated text |
| `fairness`, `dif` | subgroup bias, impact ratios, regressions of the error, differential functioning |
| `irt` | a graded response model of the trait scores |
| `mitigate`, `routing` | reweighing and calibration by group, sending doubtful essays to a person |
| `study`, `pipeline`, `report` | the tables of the audit, the run that writes them, the summary |

## References

- Crossley, S., Tian, Y., Baffour, P., et al. (2023). The English Language Learner Insight,
  Proficiency and Skills Evaluation (ELLIPSE) Corpus. International Journal of Learner Corpus
  Research, 9(2), 248-269.
- Williamson, D. M., Xi, X., and Breyer, F. J. (2012). A framework for evaluation and use of
  automated scoring. Educational Measurement: Issues and Practice, 31(1), 2-13.
- Shrout, P. E., and Fleiss, J. L. (1979). Intraclass correlations: uses in assessing rater
  reliability. Psychological Bulletin, 86(2), 420-428.
- Bland, J. M., and Altman, D. G. (1986). Statistical methods for assessing agreement between two
  methods of clinical measurement. The Lancet, 327(8476), 307-310.
- Cleary, T. A. (1968). Test bias: prediction of grades of Negro and white students in
  integrated colleges. Journal of Educational Measurement, 5(2), 115-124.
- Dorans, N. J., and Kulick, E. (1986). Demonstrating the utility of the standardization
  approach to assessing unexpected differential item performance on the SAT. Journal of
  Educational Measurement, 23(4), 355-368.
- Kamiran, F., and Calders, T. (2012). Data preprocessing techniques for classification without
  discrimination. Knowledge and Information Systems, 33(1), 1-33.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores.
  Psychometrika Monograph Supplement, No. 17.

## Licence

Apache-2.0 for the code. The corpus is CC BY 4.0 and the language model has its own licence.
