#!/usr/bin/env bash
# Loads the generated CSVs into BigQuery as raw.patients and raw.sessions.
# Usage: scripts/load_raw.sh   (run from the repo root, after gcloud config set project)
set -euo pipefail

LOCATION=EU
DATA_DIR="$(dirname "$0")/../data"

bq --location=$LOCATION mk --dataset --force raw >/dev/null

bq --location=$LOCATION load --replace --skip_leading_rows=1 --source_format=CSV \
  raw.patients "$DATA_DIR/patients.csv" \
  patient_id:INT64,age:INT64,female:BOOL,ethnicity:STRING,religion:STRING,unemployed:BOOL,on_benefits:BOOL,disability_ltc:BOOL,psychotropic_med:BOOL,baseline_phq9:INT64,baseline_gad7:INT64,referral_terminated:BOOL,sessions_offered:INT64,sessions_attended:INT64,dropped_out:BOOL,final_phq9:INT64,final_gad7:INT64,at_caseness:BOOL,reliable_improvement:BOOL,recovery:BOOL,reliable_recovery:BOOL

# phq9_score is FLOAT64 here because the CSV has values like 13.0; staging casts it
bq --location=$LOCATION load --replace --skip_leading_rows=1 --source_format=CSV \
  raw.sessions "$DATA_DIR/sessions.csv" \
  patient_id:INT64,session_number:INT64,attended:BOOL,phq9_score:FLOAT64

bq query --use_legacy_sql=false --format=pretty \
  'select "patients" as t, count(*) as n from raw.patients union all select "sessions", count(*) from raw.sessions'
