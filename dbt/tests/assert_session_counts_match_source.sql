-- Attendance counted from the sessions table should match the per-patient
-- counts in the source. Any rows returned are mismatches.
select p.patient_id, p.sessions_attended, pm.sessions_attended as counted
from {{ ref('stg_patients') }} p
inner join {{ ref('process_measures') }} pm using (patient_id)
where p.sessions_attended != pm.sessions_attended
   or p.sessions_offered != pm.sessions_offered
