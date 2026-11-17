with source as (
    select * from {{ source('raw', 'sessions') }}
)

select
    patient_id,
    session_number,
    attended as is_attended,
    -- generator writes scores as 13.0, so cast back to whole numbers
    cast(phq9_score as int64) as phq9_score
from source
