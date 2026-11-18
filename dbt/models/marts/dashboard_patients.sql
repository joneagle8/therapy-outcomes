-- One row per referred patient, with readable labels for Looker Studio.
-- Rates are 0/1 columns so a chart can average them directly.

with patients as (
    select * from {{ ref('stg_patients') }}
),

process as (
    select * from {{ ref('process_measures') }}
),

outcomes as (
    select * from {{ ref('outcome_features') }}
)

select
    p.patient_id,

    -- who
    p.age,
    case when p.age < 25 then '18-24' when p.age <= 35 then '25-35' else '36+' end as age_band,
    if(p.is_female, 'Female', 'Male') as sex,
    p.ethnicity,
    if(p.ethnicity = 'White', 'White', 'Ethnic minority') as ethnic_group,
    p.religion,
    if(p.has_limited_english, 'Limited English', 'Fluent English') as english,
    p.imd_decile,
    case when p.imd_decile <= 3 then '1-3 (most deprived)'
         when p.imd_decile <= 7 then '4-7'
         else '8-10 (least deprived)' end as deprivation_band,
    if(p.is_unemployed, 'Unemployed', 'Not unemployed') as employment,
    if(p.is_on_benefits, 'On benefits', 'Not on benefits') as benefits,
    if(p.has_disability_ltc, 'Disability / LTC', 'No disability / LTC') as disability_ltc,

    -- where they ended up
    case
        when p.is_referral_terminated then 'Ended at referral'
        when not p.is_dropped_out then 'Completed'
        when o.has_reliable_improvement then 'Left early, improved'
        else 'Left early, not improved'
    end as pathway,

    -- engagement
    pr.sessions_offered,
    pr.sessions_attended,
    pr.attendance_rate,
    pr.early_sessions_attended,

    -- scores
    p.baseline_phq9,
    p.baseline_gad7,
    o.final_phq9,
    o.final_gad7,

    -- 0/1 flags (null where they don't apply)
    if(p.is_referral_terminated, 0, 1) as started_treatment,
    if(p.is_referral_terminated, null, cast(p.is_dropped_out as int64)) as dropped_out,
    if(p.is_referral_terminated, null,
       cast(p.is_dropped_out and not o.has_reliable_improvement as int64)) as left_unimproved,
    cast(o.has_reliable_improvement as int64) as reliably_improved,
    cast(o.is_recovered as int64) as recovered,
    if(p.is_referral_terminated, null, cast(pr.is_early_engaged as int64)) as engaged_early

from patients p
left join process pr using (patient_id)
left join outcomes o using (patient_id)
