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

Work in progress.

## Data

The ELLIPSE corpus (Crossley et al., 2023, CC BY 4.0): 6,482 essays by English language learners
in grades 8 to 12, each with an overall score and six trait scores (cohesion, syntax, vocabulary,
phraseology, grammar, conventions) that are the mean of two raters, and the writer's gender, race
or ethnicity and economic status. A second archive has the single ratings of the 27 raters.
The files are pinned by hash and fetched with `scoreaudit download`.

## Models

Every model is an open-weight Qwen2.5 Instruct model, used in two ways: its hidden states as
features for a ridge regression, and its next-token probabilities after a rating prompt, as a
zero-shot scorer. Hugging Face ids or local folders both work.

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
