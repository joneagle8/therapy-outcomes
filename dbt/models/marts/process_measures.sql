with sessions as (
    select * from {{ ref('stg_sessions') }}
),

-- how many sessions were missed before each attended one
attended as (
    select
        patient_id,
        session_number,
        phq9_score,
        session_number
            - lag(session_number) over (partition by patient_id order by session_number)
            - 1 as missed_since_last
    from sessions
    where is_attended
),

early_scores as (
    select
        patient_id,
        array_agg(phq9_score order by session_number limit 1)[offset(0)] as first_early_phq9,
        array_agg(phq9_score order by session_number desc limit 1)[offset(0)] as last_early_phq9,
        count(*) as n_early_scores
    from attended
    where session_number <= {{ var('early_window') }}
    group by patient_id
),

per_patient as (
    select
        patient_id,
        count(*) as sessions_offered,
        countif(is_attended) as sessions_attended,
        countif(is_attended and session_number <= {{ var('early_window') }}) as early_sessions_attended,
        min(if(is_attended, session_number, null)) as first_attended_session,
        max(if(is_attended, session_number, null)) as last_attended_session
    from sessions
    group by patient_id
),

gaps as (
    select patient_id, coalesce(max(missed_since_last), 0) as longest_gap
    from attended
    group by patient_id
)

select
    p.patient_id,
    p.sessions_offered,
    p.sessions_attended,
    round(safe_divide(p.sessions_attended, p.sessions_offered), 3) as attendance_rate,
    p.early_sessions_attended,
    p.early_sessions_attended >= 2 as is_early_engaged,
    if(e.n_early_scores >= 2, e.last_early_phq9 - e.first_early_phq9, null) as early_phq9_change,
    p.first_attended_session,
    p.last_attended_session,
    g.longest_gap
from per_patient p
left join early_scores e using (patient_id)
left join gaps g using (patient_id)
