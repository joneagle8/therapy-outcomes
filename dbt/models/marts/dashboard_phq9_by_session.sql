-- Average PHQ-9 at each session number, by pathway, for a trajectory chart.
-- Only attended sessions have a score.

with sessions as (
    select * from {{ ref('stg_sessions') }}
    where is_attended
),

pathways as (
    select patient_id, pathway from {{ ref('dashboard_patients') }}
)

select
    s.session_number,
    p.pathway,
    round(avg(s.phq9_score), 2) as avg_phq9,
    count(*) as patients_attending
from sessions s
inner join pathways p using (patient_id)
group by 1, 2
