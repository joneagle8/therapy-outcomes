with source as (
    select * from {{ source('raw', 'patients') }}
)

select
    patient_id,
    age,
    female as is_female,
    ethnicity,
    religion,
    imd_decile,
    limited_english as has_limited_english,
    unemployed as is_unemployed,
    on_benefits as is_on_benefits,
    disability_ltc as has_disability_ltc,
    psychotropic_med as on_psychotropic_med,
    baseline_phq9,
    baseline_gad7,
    referral_terminated as is_referral_terminated,
    sessions_offered,
    sessions_attended,
    dropped_out as is_dropped_out,
    final_phq9,
    final_gad7,
    at_caseness as is_at_caseness,
    reliable_improvement as has_reliable_improvement,
    recovery as is_recovered,
    reliable_recovery as is_reliably_recovered
from source
