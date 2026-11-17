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

## Layout

```
therapy-outcomes/
├── generator/   data generator
└── data/        generated CSVs (not committed)
```
