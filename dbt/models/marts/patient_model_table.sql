-- One row per patient who started treatment. Predictors are limited to what
-- is known by the end of the early window, so they can't leak the outcome.

with patients as (
    select * from {{ ref('stg_patients') }}
    where not is_referral_terminated
),

process as (
    select * from {{ ref('process_measures') }}
),

outcomes as (
    select * from {{ ref('outcome_features') }}
)

select
    p.patient_id,

    -- known at referral
    p.age,
    p.age < 25 as is_under_25,
    p.is_female,
    p.ethnicity,
    p.religion,
    p.is_unemployed,
    p.is_on_benefits,
    p.has_disability_ltc,
    p.on_psychotropic_med,
    p.baseline_phq9,
    p.baseline_gad7,
    o.is_at_caseness,
    p.sessions_offered,

    -- known by the end of the early window
    pr.early_sessions_attended,
    pr.is_early_engaged,
    pr.early_phq9_change,

    -- targets
    p.is_dropped_out,
    o.is_recovered,
    o.has_reliable_improvement

from patients p
inner join process pr using (patient_id)
inner join outcomes o using (patient_id)
