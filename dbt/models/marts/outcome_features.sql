with patients as (
    select * from {{ ref('stg_patients') }}
    where not is_referral_terminated
),

changes as (
    select
        patient_id,
        baseline_phq9,
        baseline_gad7,
        final_phq9,
        final_gad7,
        baseline_phq9 - final_phq9 as phq9_improvement,
        baseline_gad7 - final_gad7 as gad7_improvement,
        baseline_phq9 >= {{ var('phq9_caseness') }}
            or baseline_gad7 >= {{ var('gad7_caseness') }} as is_at_caseness
    from patients
),

flags as (
    select
        *,
        phq9_improvement >= {{ var('phq9_reliable_change') }}
            or gad7_improvement >= {{ var('gad7_reliable_change') }} as has_reliable_drop,
        phq9_improvement <= -{{ var('phq9_reliable_change') }}
            or gad7_improvement <= -{{ var('gad7_reliable_change') }} as has_reliable_rise
    from changes
)

select
    patient_id,
    baseline_phq9,
    baseline_gad7,
    final_phq9,
    final_gad7,
    phq9_improvement,
    gad7_improvement,
    is_at_caseness,
    has_reliable_drop and not has_reliable_rise as has_reliable_improvement,
    has_reliable_rise and not has_reliable_drop as has_reliable_deterioration,
    -- recovery only applies to patients who started at caseness
    if(is_at_caseness,
       final_phq9 < {{ var('phq9_caseness') }} and final_gad7 < {{ var('gad7_caseness') }},
       null) as is_recovered,
    if(is_at_caseness,
       final_phq9 < {{ var('phq9_caseness') }} and final_gad7 < {{ var('gad7_caseness') }}
           and has_reliable_drop and not has_reliable_rise,
       null) as is_reliably_recovered
from flags
