# Therapy outcomes analysis

Predicting who drops out of talking therapy, using session-level data.

The data is synthetic. Patient-level NHS Talking Therapies data isn't publicly
available, so I wrote a generator that produces data with the same structure
and roughly the same outcome rates and predictors as the published research.
That means the results here show the pipeline and the method working. They
aren't new clinical findings.

## Key findings

- **Not all dropout is bad.** 38% of people who started treatment left
  early, but about a third of them had already reliably improved. Treating
  every early leaver as a failure hides the group that matters: the quarter
  of patients who leave without improving.

- **Missed early sessions are the clearest warning sign.** 80% of people who
  came to only one of their first three sessions left early, against 30% of
  those who came to all three. But 69% of that first group had already
  stopped coming by session 3. Among people who came back after session 3
  the gap is much smaller (37% against 27%). So a missed early session
  mostly means someone has gone or is about to, which is worth chasing
  quickly. A flat PHQ-9 score over those sessions flags leaving without
  improving.

- **Younger people, men and people in more deprived areas leave more often**,
  in line with Burke et al. (2025). Under-25s have about 1.5 times the odds
  of dropping out compared with over-35s.

- **The social and cultural predictors of poorer recovery**, such as
  unemployment, benefits, limited English and ethnicity, **make leaving
  without improving more likely**, but the effects are small. Most of them
  make no difference to dropout overall, because people who are doing worse
  are also less likely to leave because they feel better, so the two cancel
  out. Ethnicity is the exception, because I gave it a small direct effect
  on dropout (about 1.1 times the odds). With 4,000 patients none of these
  effects is detectable. It takes a 60,000-patient run to show their true
  size, a reminder that a null result in a small sample isn't evidence of no
  effect.

- **A model built on what's known by session 3** reaches an AUC of 0.65 for
  any dropout, in line with Shipman and Ushakova's (2026) dropout model on
  real NHS Talking Therapies data, and 0.76 to 0.80 for leaving without
  improving. Much of that comes from people who had already stopped coming.
  For people still attending after session 3 it falls to about 0.55 and
  0.62 to 0.68.

Read more:

- [Plain-language write-up](analysis/WRITEUP.md), for a non-technical reader
- [Analysis notebook](analysis/dropout_model.ipynb), with the full method,
  charts and checks

## How it works

```
generator (Python) -> CSVs -> BigQuery (raw) -> dbt (staging, marts, tests)
                                                   -> notebook (scikit-learn, statsmodels)
                                                   -> dashboard (Looker Studio, in progress)
```

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
outcome. The paper shows the size of each predictor's coefficient in its
models (Figures 1 and 2). Baseline severity has the biggest: being a
depression case (1.18) and PHQ-9 score (0.93) in the depression reliable
improvement model, and similar for anxiety, against 0.1 to 0.4 for
everything else. It sits lower in their averaged ranking only because it
wasn't selected for reliable recovery. So the generator gives severity the
largest weight. Next come unemployment, limited English and benefits, then
disability or long-term condition, psychotropic medication and being
Muslim, with a smaller effect for ethnic minority groups. That roughly
follows the paper's order. Disability is a little higher than the paper
has it, because the paper's top predictor, unemployment due to sickness or
disability, combines the two. The paper gives sizes but not directions, so
I've followed the earlier research it cites, in which these groups tend to
do worse.

**Dropout (Burke et al., 2025).** [Recovery, completion and further
referral after IAPT in Norfolk and Waveney](https://doi.org/10.1192/bjo.2025.10045)
(*BJPsych Open*, about 117,000 referrals) found younger people, men and
people in more deprived areas were less likely to complete treatment. The
generator gives these a direct effect on dropout, scaled roughly from that
paper's odds ratios. It also adds an IMD deprivation decile, with
unemployment and benefits more common in deprived areas.

**Ethnicity and dropout (my assumption).** Kanso, with a much more diverse
sample, names ethnicity as a key predictor of outcomes. Burke found only a
small, non-significant difference in completion, but compared White
British patients with everyone else in a mostly White British area. I've
extended Kanso's finding to a small direct effect on dropout, about 1.1
times the odds. It's kept small because Burke's "not completed" also counts
people who declined before treatment started, and Harwood et al. (2023)
found the ethnic gaps are mainly at the access stages (self-referral,
assessment, getting into treatment), which the patients modelled here have
already got past. Religion and language affect dropout only through
outcomes.

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

Two more marts feed the dashboard:

- `dashboard_patients`: one row per referred patient with readable labels
  (age band, deprivation band, ethnic group, and so on) and a `pathway`
  column: ended at referral, completed, left early and improved, or left
  early without improving. Rates are 0/1 columns so charts can average them.
- `dashboard_phq9_by_session`: average PHQ-9 at each session number for each
  pathway, for a trajectory chart.

The thresholds (caseness, reliable change, the three-session window) are dbt
variables in `dbt_project.yml`, so they're set in one place. There's also a
test checking that attendance counted from the sessions table matches the
patient-level counts.

## Analysis

[`analysis/dropout_model.ipynb`](analysis/dropout_model.ipynb) reads
`patient_model_table` from BigQuery and models dropout two ways: any dropout,
and dropout without reliable improvement. Each gets a logistic regression
for odds ratios, an elastic net to match Kanso et al.'s method, and a
gradient-boosted model, all tested on a held-out 25% of patients. It then
refits the models on just the people who came back after session 3, to
separate real prediction from dropout that had already happened (this is
the only place `process_measures` is used, to find those people). Next it
checks that the model works equally well across ethnic, religious, language
and deprivation groups. A final section regenerates 60,000 patients with the
same settings and refits the model, to show which of the main results are
real and which are small-sample noise. The findings are summarised at the
top of this README and in [`analysis/WRITEUP.md`](analysis/WRITEUP.md).

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
└── analysis/    dropout model notebook and write-up
```
