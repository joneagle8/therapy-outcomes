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

### How the generator is calibrated

I based it on Kanso et al. (2026), *Building a prediction model for outcomes
following treatment in UK NHS Talking Therapies* (medRxiv, v2).

- Predictors of worse outcomes: higher baseline severity, unemployment,
  receiving benefits, disability or long-term condition, psychotropic medication.
- Target rates: about 50% recovery, 60% reliable improvement, and 20% of
  referrals ending before treatment starts.

Their models got an AUC of roughly 0.63 to 0.77. A model trained on this data
will probably do better, because simulated data is cleaner than real records.

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

## Layout

```
therapy-outcomes/
├── generator/   data generator
├── data/        generated CSVs (not committed)
├── scripts/     BigQuery load script
└── dbt/         dbt project (staging models so far)
```
