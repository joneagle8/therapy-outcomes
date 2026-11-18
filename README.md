# Therapy outcomes analysis

Predicting who drops out of talking therapy, using session-level data.

The data is synthetic. Patient-level NHS Talking Therapies data isn't publicly
available, so I wrote a generator that produces data with the same structure
and roughly the same outcome rates and predictors as the published research.
That means the results here show the pipeline and the method working. They
aren't new clinical findings.

## Data

Two tables:

- `data/patients.csv`: one row per patient. Demographics, baseline PHQ-9 and
  GAD-7, risk factors, and final outcome flags.
- `data/sessions.csv`: one row per scheduled session, with whether it was
  attended and the PHQ-9 score if it was.

To regenerate:

```
python generator/generate_data.py --n 5000 --seed 42
```

### Where the generator's effects come from

The paper this builds on, Kanso et al. (2026), *Building a prediction model
for outcomes following treatment in UK NHS Talking Therapies* (medRxiv, v2),
models **outcomes**. It doesn't model dropout. So the generator draws on two
sources, plus some assumptions of my own.

**Outcomes (Kanso et al.).** Each predictor raises the chance of a poor
outcome, weighted roughly in the paper's order of importance: unemployment,
limited English, benefits, psychotropic medication, disability or long-term
condition, being Muslim, baseline severity, and a small effect for ethnic
minority groups. The paper ranks predictors but doesn't give a direction for
each, so I've followed the earlier research it cites, in which minority
groups tend to do worse.

**Dropout (Burke et al., 2025).** [Recovery, completion and further
referral after IAPT in Norfolk and Waveney](https://doi.org/10.1192/bjo.2025.10045)
(*BJPsych Open*, about 117,000 referrals) found younger people, men and
people in more deprived areas were less likely to complete treatment. The
generator gives these a direct effect on dropout, scaled roughly from that
paper's odds ratios. It also adds an IMD deprivation decile, with
unemployment and benefits more common in deprived areas.

**Ethnicity and dropout (my assumption).** Burke found no ethnicity effect on
completion, but compared only White British patients with everyone else, in
a mostly White British area. Kanso, with a much more diverse sample, names
ethnicity as a key predictor of outcomes. I've extended that to a small
direct effect on dropout. It's kept small for two reasons. Burke's "not
completed" also counts people who declined before treatment started. And
Harwood et al. (2023) found the ethnic gaps are mainly at the access stages:
self-referral, assessment, and getting into treatment. The patients modelled
here have already got past those. Religion and language affect dropout only
through outcomes.

**How the predictors relate to each other.** Real predictors overlap, so
the generated ones do too:

- Baseline PHQ-9 and GAD-7 are correlated at about 0.8. Ryan et al. (2013),
  [Factorial invariance of the PHQ and GAD questionnaires](https://doi.org/10.1111/bjc.12028)
  (*British Journal of Clinical Psychology*), found r = 0.74 at first contact
  in about 18,500 London IAPT patients. I've set it slightly higher.
- Disability or a long-term condition makes unemployment more likely (42%
  vs 18%), and both make benefits more likely. In UK data most out-of-work
  benefit claims are for disability or long-term conditions
  ([Health Foundation, Labour Force Survey 2023](https://www.health.org.uk/resources-and-toolkits/data-tools/evidence-hub/health-conditions-among-working-age-people)).
  Those are general population figures, so they guide the direction and
  rough strength rather than exact values.
- Unemployment and benefits are more common in deprived areas.
- Being Muslim and limited English are more common in some ethnic groups.
- Psychotropic medication is independent of the others. I couldn't find a
  source linking it to severity in this population.

**Calibration.** The generator's own summary compares its outcome rates
with the paper's, per measure:

| | Reliable improvement | Recovery | Reliable recovery |
|---|---|---|---|
| Depression (PHQ-9) | 50.9% (paper 47.8%) | 57.7% (55.7%) | 47.6% (46.9%) |
| Anxiety (GAD-7) | 55.0% (57.9%) | 54.5% (52.9%) | 46.2% (49.2%) |

Dropout is 38%, and 20% of referrals end before treatment starts. Kanso's
models reached an AUC of roughly 0.63 to 0.77.

### How improvement and dropout work in the generator

Each patient either responds to therapy, ending up in the low score range
wherever they started, or doesn't respond and improves only a little. Higher
risk makes responding less likely. Scores move steadily towards that end
point over the course, so anyone who leaves early keeps whatever they'd
gained by then. People can leave early in two ways:

- **Disengaging.** More likely for people who engage poorly in the early
  sessions, have more risk factors, are younger, male, or live in a more
  deprived area. They attend up to a point, then stop.
- **Leaving because they feel better.** Some people whose scores drop below
  caseness before the end of the course stop coming within a session or two.

Both count as dropout, and the data doesn't record which kind of leaver
someone is, because real service data usually doesn't either. So dropout
isn't always a bad outcome: about 35% of the people who dropped out had
reliably improved. That matters when interpreting what predicts dropout.

## BigQuery and dbt

The CSVs are loaded into a `raw` dataset in BigQuery, which stands in for a
source system. dbt reads from there and builds everything else into an
`analytics` dataset.

```
scripts/load_raw.sh        # load CSVs into raw.patients and raw.sessions
cd dbt && dbt build        # build models and run tests
```

This needs the Google Cloud CLI, a login (`gcloud auth application-default login`)
and a `dbt/profiles.yml` pointing at your own project. The profile isn't
committed.

### Staging

- `stg_patients`: one row per referred patient. Renames the yes/no columns to
  `is_`/`has_` names so they read clearly in later models.
- `stg_sessions`: one row per scheduled session. The generator writes scores as
  `13.0`, so these are cast back to integers.

Tests check that patient IDs are unique, scores stay inside the PHQ-9 (0-27)
and GAD-7 (0-21) ranges, there's only one row per patient per session, and
every session belongs to a known patient.

### Marts

- `process_measures`: one row per treated patient, built from the sessions
  table. Attendance rate, sessions attended in the first three, whether the
  patient was engaged early (attended 2 of the first 3), PHQ-9 change over
  those early sessions, and the longest run of missed sessions.
- `outcome_features`: score changes and the NHS Talking Therapies outcome
  flags (reliable improvement, reliable deterioration, recovery, reliable
  recovery). These are worked out from the scores using the published
  definitions rather than copied from the source data. Recovery is left null
  for patients who weren't at caseness to begin with, since it doesn't apply
  to them.
- `patient_model_table`: the table the model uses. One row per treated
  patient, with the dropout target and only the predictors that would be
  known by the third session.

That last restriction matters. Total attendance or the last session attended
would predict dropout almost perfectly, but only because they're partly a
record of the dropout itself. They stay in `process_measures` for describing
the data and are kept out of the model table.

The thresholds (caseness, reliable change, the three-session window) are dbt
variables in `dbt_project.yml`, so they're set in one place. There's also a
test checking that attendance counted from the sessions table matches the
patient-level counts.

## Analysis

[`analysis/dropout_model.ipynb`](analysis/dropout_model.ipynb) reads
`patient_model_table` from BigQuery and models dropout two ways: any dropout,
and dropout without reliable improvement. Each gets a logistic regression
for odds ratios, an elastic net to match Kanso et al.'s method, and a
gradient-boosted model, all tested on a held-out 25% of patients. It ends
with a check that the model works equally well across ethnic, religious,
language and deprivation groups.

Main findings:

- Attendance in the first three sessions is by far the strongest predictor.
  Each extra session attended cuts the odds of dropping out by about 63%.
- Age, sex and deprivation come through clearly, in line with Burke et al.
  Under-25s have about 1.5 times the odds of dropping out compared with
  over-35s.
- The paper's social and cultural predictors (unemployment, benefits,
  disability, limited English, being Muslim, ethnicity) matter for leaving
  without improving, not for dropout as a whole. They make recovery less
  likely, so fewer of these patients leave because they feel better, and the
  two kinds of leaver cancel each other out.
- Most of these effects are too small to measure reliably with about 4,000
  patients. A final section regenerates 60,000 patients with the same
  settings and shows the true sizes: each of the paper's predictors raises
  the odds of leaving unimproved by about 1.07 to 1.14. The main data
  misses all of them, and gets some the wrong way round. For example, it
  puts ethnic minority patients at 1.02 against a true 1.12. So a null result in a small or
  coarse comparison isn't evidence of no effect.
- Test AUC is 0.61 to 0.63 for any dropout and 0.76 to 0.78 for dropout
  without improvement, within the range of Kanso et al.'s models on real
  data.

To run it:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebook with the `.venv` kernel. It needs the same Google Cloud
login as dbt.

## Layout

```
therapy-outcomes/
├── generator/   data generator
├── data/        generated CSVs (not committed)
├── scripts/     BigQuery load script
├── dbt/         dbt project (staging and marts)
└── analysis/    dropout model notebook
```
